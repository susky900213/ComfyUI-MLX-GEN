"""Native MLX implementation of the Fibo-2 diffusion transformer.

The module mirrors Diffusers' ``BriaFibo2Transformer2DModel`` without importing
Torch or Diffusers.  The public Fibo checkpoint is a regular Diffusers
checkpoint, so weight conversion is kept in :mod:`mlx_backend`; this file only
contains the numerical model.
"""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import numpy as np
from mlx import nn


ADALN_EMBED_DIM = 256


def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    dtype = x.dtype
    value = x.astype(mx.float32)
    value = value * mx.rsqrt(mx.mean(mx.square(value), axis=-1, keepdims=True) + eps)
    return (value * weight.astype(mx.float32)).astype(dtype)


def _layer_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    value = x.astype(mx.float32)
    mean = value.mean(axis=-1, keepdims=True)
    variance = mx.mean(mx.square(value - mean), axis=-1, keepdims=True)
    return ((value - mean) * mx.rsqrt(variance + eps)).astype(x.dtype)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _select_per_token(
    value_noisy: mx.array, value_clean: mx.array, noise_mask: mx.array
) -> mx.array:
    mask = noise_mask[..., None].astype(mx.bool_)
    noisy = mx.broadcast_to(value_noisy[:, None, :], (value_noisy.shape[0], noise_mask.shape[1], value_noisy.shape[-1]))
    clean = mx.broadcast_to(value_clean[:, None, :], (value_clean.shape[0], noise_mask.shape[1], value_clean.shape[-1]))
    return mx.where(mask, noisy, clean)


def timestep_embedding(timestep: mx.array, dim: int = 256, max_period: int = 10_000) -> mx.array:
    value = timestep.astype(mx.float32).reshape(-1, 1)
    half = dim // 2
    frequencies = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = value * frequencies[None]
    result = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        result = mx.concatenate([result, mx.zeros((result.shape[0], 1), dtype=result.dtype)], axis=-1)
    return result


class FiboTimestepEmbedder(nn.Module):
    def __init__(self, out_size: int, mid_size: int = 1024):
        super().__init__()
        self.mlp = [
            nn.Linear(256, mid_size),
            nn.SiLU(),
            nn.Linear(mid_size, out_size),
        ]

    def __call__(self, timestep: mx.array) -> mx.array:
        x = timestep_embedding(timestep, 256).astype(self.mlp[0].weight.dtype)
        return self.mlp[2](self.mlp[1](self.mlp[0](x)))


class FiboRope:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: Sequence[int] = (32, 48, 48),
        axes_lens: Sequence[int] = (2048, 512, 512),
    ):
        self.theta = float(theta)
        self.axes_dims = tuple(int(value) for value in axes_dims)
        self.axes_lens = tuple(int(value) for value in axes_lens)
        self._inverse: tuple[mx.array, ...] | None = None

    def __call__(self, ids: mx.array) -> tuple[mx.array, mx.array]:
        if ids.ndim != 2 or ids.shape[-1] != len(self.axes_dims):
            raise ValueError(f"Fibo-2 RoPE ids 应为 [tokens,{len(self.axes_dims)}]，收到 {tuple(ids.shape)}")
        if self._inverse is None:
            self._inverse = tuple(
                1.0
                / (self.theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / float(dim)))
                for dim in self.axes_dims
            )
        parts = [ids[:, axis : axis + 1].astype(mx.float32) * inv[None] for axis, inv in enumerate(self._inverse)]
        angles = mx.concatenate(parts, axis=-1)
        return mx.cos(angles), mx.sin(angles)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x is [B, heads, tokens, head_dim]; the Fibo head dimension is 128 and
    # the concatenated RoPE has 64 cosine/sine pairs.
    shape = x.shape
    if cos.shape[-1] * 2 != shape[-1]:
        raise ValueError(
            f"Fibo-2 RoPE 维度与 attention head 不一致：rope={cos.shape[-1] * 2}, head={shape[-1]}"
        )
    pairs = x.astype(mx.float32).reshape(*shape[:-1], -1, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    rotated = mx.stack([real * c - imag * s, imag * c + real * s], axis=-1)
    return rotated.reshape(shape).astype(x.dtype)


class FiboFeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden = (dim * 8 // 3) // 64 * 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(_silu(self.w1(x)) * self.w3(x))


class FiboAttention(nn.Module):
    def __init__(self, query_dim: int, heads: int, dim_head: int, kv_dim: int | None = None, eps: float = 1e-5):
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        kv_dim = int(kv_dim or query_dim)
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_out = [nn.Linear(heads * dim_head, query_dim, bias=False)]
        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
        attention_mask: mx.array | None = None,
        freqs: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch, query_len, _ = hidden_states.shape
        key_len = context.shape[1]
        query = self.to_q(hidden_states).reshape(batch, query_len, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        key = self.to_k(context).reshape(batch, key_len, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        value = self.to_v(context).reshape(batch, key_len, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        query = self.norm_q(query)
        key = self.norm_k(key)
        if freqs is not None:
            query = _apply_rope(query, freqs[0], freqs[1])
            key = _apply_rope(key, freqs[0], freqs[1])
        mask = attention_mask
        if mask is not None and mask.ndim == 2:
            mask = mask[:, None, None, :]
        attended = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=1.0 / math.sqrt(self.dim_head), mask=mask
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, query_len, -1)
        return self.to_out[0](attended)


class FiboTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, eps: float, cross_attention_dim: int | None = None):
        super().__init__()
        self.attention = FiboAttention(dim, heads, dim // heads, eps=eps)
        self.feed_forward = FiboFeedForward(dim)
        self.attention_norm1 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=eps)
        self.adaLN_modulation = [nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim)]
        self.cross_attention = None
        if cross_attention_dim is not None:
            self.cross_attn_norm1 = nn.RMSNorm(dim, eps=eps)
            self.cross_attention = FiboAttention(dim, heads, dim // heads, kv_dim=cross_attention_dim, eps=eps)
            self.cross_attn_norm2 = nn.RMSNorm(dim, eps=eps)
            self.cross_attn_gate = mx.zeros((1,), dtype=mx.float32)

    def __call__(
        self,
        x: mx.array,
        freqs: tuple[mx.array, mx.array],
        *,
        attn_mask: mx.array | None = None,
        adaln_input: mx.array | None = None,
        noise_mask: mx.array | None = None,
        adaln_noisy: mx.array | None = None,
        adaln_clean: mx.array | None = None,
        encoder_hidden_states: mx.array | None = None,
        encoder_attention_mask: mx.array | None = None,
    ) -> mx.array:
        if noise_mask is not None:
            modulation_noisy = self.adaLN_modulation[0](adaln_noisy)
            modulation_clean = self.adaLN_modulation[0](adaln_clean)
            noisy = mx.split(modulation_noisy, 4, axis=-1)
            clean = mx.split(modulation_clean, 4, axis=-1)
            scale_msa = _select_per_token(1.0 + noisy[0], 1.0 + clean[0], noise_mask)
            gate_msa = _select_per_token(mx.tanh(noisy[1]), mx.tanh(clean[1]), noise_mask)
            scale_mlp = _select_per_token(1.0 + noisy[2], 1.0 + clean[2], noise_mask)
            gate_mlp = _select_per_token(mx.tanh(noisy[3]), mx.tanh(clean[3]), noise_mask)
        else:
            modulation = self.adaLN_modulation[0](adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(modulation, 4, axis=-1)
            scale_msa = 1.0 + scale_msa[:, None, :]
            scale_mlp = 1.0 + scale_mlp[:, None, :]
            gate_msa = mx.tanh(gate_msa[:, None, :])
            gate_mlp = mx.tanh(gate_mlp[:, None, :])
        attended = self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs=freqs)
        x = x + gate_msa * self.attention_norm2(attended)
        if self.cross_attention is not None:
            cross = self.cross_attention(
                self.cross_attn_norm1(x),
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            x = x + mx.tanh(self.cross_attn_gate)[None, None, :] * self.cross_attn_norm2(cross)
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        return x


class FiboPerceiverLayer(nn.Module):
    def __init__(self, dim: int, heads: int, cross_dim: int, eps: float):
        super().__init__()
        self.cross_attn_norm = nn.RMSNorm(dim, eps=eps)
        self.cross_attention = FiboAttention(dim, heads, dim // heads, kv_dim=cross_dim, eps=eps)
        self.self_attn_norm = nn.RMSNorm(dim, eps=eps)
        self.self_attention = FiboAttention(dim, heads, dim // heads, eps=eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=eps)
        self.ffn = FiboFeedForward(dim)

    def __call__(
        self,
        latents: mx.array,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        latents = latents + self.cross_attention(
            self.cross_attn_norm(latents), encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask
        )
        latents = latents + self.self_attention(self.self_attn_norm(latents), attention_mask=attention_mask)
        return latents + self.ffn(self.ffn_norm(latents))


class FiboPerceiver(nn.Module):
    def __init__(self, dim: int, heads: int, cross_dim: int, max_queries: int, layers: int, eps: float):
        super().__init__()
        self.latent_queries = mx.random.normal((max_queries, dim), key=mx.random.key(0)) * 0.02
        self.layers = [FiboPerceiverLayer(dim, heads, cross_dim, eps) for _ in range(layers)]
        self.norm_out = nn.RMSNorm(dim, eps=eps)

    def __call__(
        self,
        encoder_hidden_states: mx.array,
        num_queries: int,
        encoder_attention_mask: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        latents = mx.broadcast_to(
            self.latent_queries[:num_queries][None],
            (encoder_hidden_states.shape[0], num_queries, self.latent_queries.shape[-1]),
        )
        for layer in self.layers:
            latents = layer(latents, encoder_hidden_states, encoder_attention_mask, attention_mask)
        return self.norm_out(latents)


class FiboFinalLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int):
        super().__init__()
        self.norm_final = _FiboAffineFreeLayerNorm()
        self.linear = nn.Linear(dim, out_channels)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(min(dim, ADALN_EMBED_DIM), dim)]

    def __call__(self, x: mx.array, conditioning: mx.array) -> mx.array:
        scale = 1.0 + self.adaLN_modulation[1](self.adaLN_modulation[0](conditioning))
        return self.linear(self.norm_final(x) * scale[:, None, :])


class _FiboAffineFreeLayerNorm(nn.Module):
    """Parameter-free LayerNorm kept as a named module for checkpoint parity."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _layer_norm(x, self.eps)


def compute_gist_count(text_lengths: Sequence[int], min_gist: int, max_gist: int, step: int, min_text: int, max_text: int) -> list[int]:
    span = max(1, max_text - min_text)
    result = []
    for length in text_lengths:
        ratio = min(1.0, max(0.0, (float(length) - min_text) / span))
        raw = min_gist + ratio * (max_gist - min_gist)
        result.append(max(min_gist, min(max_gist, int(math.ceil(raw / step)) * step)))
    return result


class Fibo2Transformer(nn.Module):
    """Fibo-2 DiT with the official Diffusers constructor defaults."""

    def __init__(
        self,
        in_channels: int = 32,
        patch_size: int = 2,
        dim: int = 3584,
        n_layers: int = 44,
        n_refiner_layers: int = 2,
        n_heads: int = 28,
        norm_eps: float = 1e-5,
        cap_feat_dim: int = 7680,
        injection_layer_ids: Sequence[int] = (6, 14, 22, 30, 38),
        perceiver_num_layers: int = 2,
        min_num_gist_tokens: int = 256,
        max_num_gist_tokens: int = 512,
        gist_step: int = 64,
        gist_min_text_len: int = 256,
        gist_max_text_len: int = 2048,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: Sequence[int] = (32, 48, 48),
        axes_lens: Sequence[int] = (2048, 512, 512),
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.max_num_gist_tokens = int(max_num_gist_tokens)
        self.min_num_gist_tokens = int(min_num_gist_tokens)
        self.gist_step = int(gist_step)
        self.gist_min_text_len = int(gist_min_text_len)
        self.gist_max_text_len = int(gist_max_text_len)
        self.injection_layer_ids = tuple(int(value) for value in injection_layer_ids)
        self.t_scale = float(t_scale)
        self.x_embedder = nn.Linear(self.patch_size * self.patch_size * self.in_channels, dim)
        self.t_embedder = FiboTimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.rope_embedder = FiboRope(rope_theta, axes_dims, axes_lens)
        self.noise_refiner = [FiboTransformerBlock(dim, n_heads, norm_eps) for _ in range(n_refiner_layers)]
        self.perceiver = FiboPerceiver(dim, n_heads, cap_feat_dim, max_num_gist_tokens, perceiver_num_layers, norm_eps)
        self.layers = [
            FiboTransformerBlock(dim, n_heads, norm_eps, cap_feat_dim if i in self.injection_layer_ids else None)
            for i in range(n_layers)
        ]
        self.final_layer = FiboFinalLayer(dim, self.patch_size * self.patch_size * self.in_channels)

    @staticmethod
    def _coordinate_grid(size: tuple[int, int, int], start: tuple[int, int, int]) -> mx.array:
        axes = [mx.arange(origin, origin + span, dtype=mx.int32) for origin, span in zip(start, size)]
        grids = mx.meshgrid(*axes, indexing="ij")
        return mx.stack(grids, axis=-1)

    def _patchify(self, latents: mx.array) -> mx.array:
        batch, channels, height, width = latents.shape
        p = self.patch_size
        if height % p or width % p:
            raise ValueError("Fibo-2 latent 的高宽必须是 patch_size 的倍数")
        value = latents.reshape(batch, channels, height // p, p, width // p, p)
        return value.transpose(0, 2, 4, 1, 3, 5).reshape(batch, (height // p) * (width // p), -1)

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: mx.array | None = None,
        context_latents: Sequence[mx.array] | None = None,
    ) -> mx.array:
        batch, channels, height, width = hidden_states.shape
        if channels != self.in_channels:
            raise ValueError(f"Fibo-2 transformer 需要 {self.in_channels} 个 latent channel，收到 {channels}")
        p = self.patch_size
        ht, wt = height // p, width // p
        image_tokens = ht * wt
        noisy_cond = self.t_embedder(timestep.astype(mx.float32) * self.t_scale)
        clean_cond = self.t_embedder(mx.zeros_like(timestep.astype(mx.float32)))
        image_plane = self.max_num_gist_tokens + 1
        tokens = [self._patchify(hidden_states)]
        ids = [self._coordinate_grid((1, ht, wt), (image_plane, 0, 0)).reshape(-1, 3)]
        for index, context in enumerate(context_latents or ()):
            tokens.append(self._patchify(context))
            ids.append(
                self._coordinate_grid((1, context.shape[2] // p, context.shape[3] // p), (image_plane + index + 1, 0, 0))
                .reshape(-1, 3)
            )
        projected = self.x_embedder(mx.concatenate(tokens, axis=1))
        freqs = self.rope_embedder(mx.concatenate(ids, axis=0))
        noise_mask = None
        if context_latents:
            noise_mask = mx.concatenate(
                [mx.ones((batch, image_tokens), dtype=mx.int32), mx.zeros((batch, projected.shape[1] - image_tokens), dtype=mx.int32)],
                axis=1,
            )
        for block in self.noise_refiner:
            projected = block(
                projected,
                freqs,
                adaln_input=noisy_cond,
                noise_mask=noise_mask,
                adaln_noisy=noisy_cond,
                adaln_clean=clean_cond,
            )
        if encoder_attention_mask is None:
            text_lengths = [encoder_hidden_states.shape[2]] * batch
        else:
            text_lengths = np.asarray(encoder_attention_mask).sum(axis=1).tolist()
        gist_counts = compute_gist_count(
            text_lengths, self.min_num_gist_tokens, self.max_num_gist_tokens, self.gist_step, self.gist_min_text_len, self.gist_max_text_len
        )
        num_gist = max(gist_counts)
        gist_mask = None
        if min(gist_counts) < num_gist:
            gist_mask = mx.arange(num_gist)[None, :] < mx.array(gist_counts)[:, None]
        gist = self.perceiver(encoder_hidden_states[:, 0], num_gist, encoder_attention_mask, gist_mask)
        gist_freqs = self.rope_embedder(self._coordinate_grid((num_gist, 1, 1), (1, 0, 0)).reshape(-1, 3))
        all_mask = mx.concatenate(
            [mx.ones((batch, projected.shape[1]), dtype=mx.bool_), gist_mask], axis=1
        ) if gist_mask is not None else None
        if gist_mask is None:
            image_start = num_gist
            stream = mx.concatenate([gist, projected], axis=1)
            stream_freqs = (mx.concatenate([gist_freqs[0], freqs[0]], axis=0), mx.concatenate([gist_freqs[1], freqs[1]], axis=0))
        else:
            image_start = 0
            stream = mx.concatenate([projected, gist], axis=1)
            stream_freqs = (mx.concatenate([freqs[0], gist_freqs[0]], axis=0), mx.concatenate([freqs[1], gist_freqs[1]], axis=0))
        projected_noise = mx.concatenate(
            [
                mx.ones((batch, image_tokens), dtype=mx.int32),
                mx.zeros((batch, stream.shape[1] - num_gist - image_tokens), dtype=mx.int32),
            ],
            axis=1,
        )
        stream_noise = (
            mx.concatenate([mx.zeros((batch, num_gist), dtype=mx.int32), projected_noise], axis=1)
            if gist_mask is None
            else mx.concatenate([projected_noise, mx.zeros((batch, num_gist), dtype=mx.int32)], axis=1)
        )
        injection = {layer: index + 1 for index, layer in enumerate(self.injection_layer_ids)}
        for index, block in enumerate(self.layers):
            context = encoder_hidden_states[:, injection[index]] if index in injection else None
            stream = block(
                stream,
                stream_freqs,
                attn_mask=all_mask,
                noise_mask=stream_noise,
                adaln_noisy=noisy_cond,
                adaln_clean=clean_cond,
                encoder_hidden_states=context,
                encoder_attention_mask=encoder_attention_mask,
            )
        image = stream[:, image_start : image_start + image_tokens]
        output = self.final_layer(image, noisy_cond)
        return output.reshape(batch, ht, wt, channels, p, p).transpose(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)


__all__ = ["Fibo2Transformer", "compute_gist_count", "timestep_embedding"]