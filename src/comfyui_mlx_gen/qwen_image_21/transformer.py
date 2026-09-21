# Ported from diffusers' Qwen-Image 2.1 implementation (Apache-2.0).
"""Qwen-Image 2.1 single-stream DiT in MLX.

The text prefix is causal and the target-image block is bidirectional.  With
``causal_condition=True`` the text prefix always uses the t=0 modulation row,
which makes its per-layer K/V reusable after the first denoising step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx import nn


def _layer_norm(x: mx.array, eps: float) -> mx.array:
    """Affine-free LayerNorm, accumulated in fp32 like the reference model."""
    dtype = x.dtype
    value = x.astype(mx.float32)
    mean = value.mean(axis=-1, keepdims=True)
    variance = mx.mean(mx.square(value - mean), axis=-1, keepdims=True)
    return ((value - mean) * mx.rsqrt(variance + eps)).astype(dtype)


def _rms_norm(x: mx.array, weight: mx.array, eps: float, *, zero_centered: bool = False) -> mx.array:
    dtype = x.dtype
    value = x.astype(mx.float32)
    scale = weight.astype(mx.float32) + 1.0 if zero_centered else weight.astype(mx.float32)
    value = value * mx.rsqrt(mx.mean(mx.square(value), axis=-1, keepdims=True) + eps)
    return (value * scale).astype(dtype)


def _gelu_tanh(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def _apply_rope(x: mx.array, frequencies: mx.array) -> mx.array:
    """Complex-pair RoPE used by Qwen Image (adjacent real/imag channels)."""
    pairs = x.reshape(*x.shape[:-1], -1, 2).astype(mx.float32)
    real, imag = pairs[..., 0], pairs[..., 1]
    cos = mx.cos(frequencies)[None, :, None, :]
    sin = mx.sin(frequencies)[None, :, None, :]
    rotated = mx.stack([real * cos - imag * sin, imag * cos + real * sin], axis=-1)
    return rotated.reshape(x.shape).astype(x.dtype)


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm(x, self.weight, self.eps, zero_centered=True)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_in_dim, eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(_gelu_tanh(self.in_layer(self.text_norm(x))))


class QwenImage21TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, embedding_dim, bias=False)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        half = 128
        self._frequencies = mx.exp(
            -math.log(10_000.0) * mx.arange(half, dtype=mx.float32) / half
        )

    def __call__(self, timestep: mx.array, dtype: mx.Dtype) -> mx.array:
        value = timestep.astype(mx.float32).reshape(-1, 1) * 1000.0
        args = value * self._frequencies[None]
        projected = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(dtype)
        return self.linear_2(nn.silu(self.linear_1(projected)))


class QwenImage21TimeTextEmbed(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.timestep_embedder = QwenImage21TimestepEmbedding(embedding_dim)

    def __call__(self, timestep: mx.array, hidden_states: mx.array) -> mx.array:
        return self.timestep_embedder(timestep, hidden_states.dtype)


class QwenImage21FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)
        self.gate_layer = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out(nn.silu(self.gate_layer(x)) * self.proj(x))


@dataclass
class QwenImage21KVLayerCache:
    key: mx.array | None = None
    value: mx.array | None = None


class QwenImage21KVCache:
    def __init__(self, num_layers: int):
        self.layers = [QwenImage21KVLayerCache() for _ in range(num_layers)]

    def layer(self, index: int) -> QwenImage21KVLayerCache:
        return self.layers[index]


class QwenImage21Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = [nn.Linear(inner, dim, bias=False)]
        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)

    def __call__(
        self,
        hidden_states: mx.array,
        rotary_emb: mx.array,
        prefix_len: int,
        layer_cache: QwenImage21KVLayerCache | None,
        cache_mode: str | None,
        segments: tuple[tuple[int, int, bool], ...] | None = None,
    ) -> mx.array:
        batch, seq_len, _ = hidden_states.shape
        query = self.to_q(hidden_states).reshape(batch, seq_len, self.heads, self.dim_head)
        key = self.to_k(hidden_states).reshape(batch, seq_len, self.heads, self.dim_head)
        value = self.to_v(hidden_states).reshape(batch, seq_len, self.heads, self.dim_head)
        query = self.norm_q(query).astype(value.dtype)
        key = self.norm_k(key).astype(value.dtype)
        query = _apply_rope(query, rotary_emb)
        key = _apply_rope(key, rotary_emb)

        if cache_mode == "cached":
            if layer_cache is None or layer_cache.key is None or layer_cache.value is None:
                raise RuntimeError("Qwen-Image 2.1 prefix KV cache 尚未填充")
            key = mx.concatenate([layer_cache.key, key], axis=1)
            value = mx.concatenate([layer_cache.value, value], axis=1)
            output = mx.fast.scaled_dot_product_attention(
                query.transpose(0, 2, 1, 3),
                key.transpose(0, 2, 1, 3),
                value.transpose(0, 2, 1, 3),
                scale=1.0 / math.sqrt(self.dim_head),
            ).transpose(0, 2, 1, 3)
        else:
            if cache_mode == "extract" and layer_cache is not None:
                # Materialise only the prefix slice; retaining a view would pin the full prefill K/V.
                layer_cache.key = mx.array(key[:, :prefix_len])
                layer_cache.value = mx.array(value[:, :prefix_len])
                mx.eval(layer_cache.key, layer_cache.value)

            outputs: list[mx.array] = []
            for start, end, is_text in segments or ():
                mask = None
                if is_text:
                    seg_len = end - start
                    mask = mx.concatenate(
                        [
                            mx.ones((seg_len, start), dtype=mx.bool_),
                            mx.tril(mx.ones((seg_len, seg_len), dtype=mx.bool_)),
                        ],
                        axis=1,
                    )[None, None]
                outputs.append(
                    mx.fast.scaled_dot_product_attention(
                        query[:, start:end].transpose(0, 2, 1, 3),
                        key[:, :end].transpose(0, 2, 1, 3),
                        value[:, :end].transpose(0, 2, 1, 3),
                        scale=1.0 / math.sqrt(self.dim_head),
                        mask=mask,
                    ).transpose(0, 2, 1, 3)
                )
            target_output = mx.fast.scaled_dot_product_attention(
                query[:, prefix_len:].transpose(0, 2, 1, 3),
                key.transpose(0, 2, 1, 3),
                value.transpose(0, 2, 1, 3),
                scale=1.0 / math.sqrt(self.dim_head),
            ).transpose(0, 2, 1, 3)
            outputs.append(target_output)
            output = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)

        output = output.reshape(batch, output.shape[1], -1).astype(query.dtype)
        return self.to_out[0](output)


def _modulation_rows(
    params: mx.array,
    prefix_len: int,
    target_len: int,
    cache_mode: str | None,
) -> mx.array:
    """Select sampled-t rows for target tokens and the final t=0 row for prefix tokens."""
    real = params[:-1, None, :]
    if cache_mode == "cached" or prefix_len == 0:
        return mx.broadcast_to(real, (real.shape[0], target_len, real.shape[-1]))
    zero = mx.broadcast_to(params[-1:, None, :], (real.shape[0], prefix_len, real.shape[-1]))
    target = mx.broadcast_to(real, (real.shape[0], target_len, real.shape[-1]))
    return mx.concatenate([zero, target], axis=1)


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: int = 3, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.attn = QwenImage21Attention(dim, heads, dim_head, eps)
        self.img_mlp = QwenImage21FeedForward(dim, dim * mlp_ratio)

    def __call__(
        self,
        hidden_states: mx.array,
        modulation: mx.array,
        rotary_emb: mx.array,
        prefix_len: int,
        target_len: int,
        layer_cache: QwenImage21KVLayerCache | None,
        cache_mode: str | None,
        segments: tuple[tuple[int, int, bool], ...] | None = None,
    ) -> mx.array:
        mod1, mod2 = mx.split(modulation, 2, axis=-1)
        scale1, gate1 = mx.split(mod1, 2, axis=-1)
        scale1 = _modulation_rows(scale1, prefix_len, target_len, cache_mode)
        gate1 = _modulation_rows(gate1, prefix_len, target_len, cache_mode)
        normalized = _layer_norm(hidden_states, self.eps) * (1.0 + scale1)
        hidden_states = hidden_states + mx.tanh(gate1) * self.attn(
            normalized, rotary_emb, prefix_len, layer_cache, cache_mode, segments
        )

        scale2, gate2 = mx.split(mod2, 2, axis=-1)
        scale2 = _modulation_rows(scale2, prefix_len, target_len, cache_mode)
        gate2 = _modulation_rows(gate2, prefix_len, target_len, cache_mode)
        normalized = _layer_norm(hidden_states, self.eps) * (1.0 + scale2)
        return hidden_states + mx.tanh(gate2) * self.img_mlp(normalized)


class QwenImage21AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.eps = eps

    def __call__(
        self,
        hidden_states: mx.array,
        conditioning: mx.array,
        prefix_len: int,
        target_len: int,
        cache_mode: str | None,
    ) -> mx.array:
        scale = self.linear(nn.silu(conditioning).astype(hidden_states.dtype))
        scale = _modulation_rows(scale, prefix_len, target_len, cache_mode)
        return _layer_norm(hidden_states, self.eps) * (1.0 + scale)


class QwenImage21Rope:
    def __init__(self, theta: float = 10_000.0, axes_dim: tuple[int, int, int] = (16, 56, 56)):
        self.theta = theta
        self.axes_dim = tuple(axes_dim)

    def _angles(self, positions: mx.array, dim: int) -> mx.array:
        frequencies = 1.0 / (self.theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        return positions.astype(mx.float32)[:, None] * frequencies[None]

    def __call__(self, text_len: int, height: int, width: int) -> mx.array:
        # Text advances all axes together. The target image freezes frame at text_len and
        # uses a spatial grid centred on zero.
        text_positions = mx.arange(text_len, dtype=mx.int32)
        frame = mx.concatenate(
            [text_positions, mx.full((height * width,), text_len, dtype=mx.int32)]
        )
        h_positions = mx.arange(-(height - height // 2), height // 2, dtype=mx.int32)
        w_positions = mx.arange(-(width - width // 2), width // 2, dtype=mx.int32)
        image_h = mx.repeat(h_positions, width)
        image_w = mx.tile(w_positions, height)
        height_axis = mx.concatenate([text_positions, image_h])
        width_axis = mx.concatenate([text_positions, image_w])
        return mx.concatenate(
            [
                self._angles(frame, self.axes_dim[0]),
                self._angles(height_axis, self.axes_dim[1]),
                self._angles(width_axis, self.axes_dim[2]),
            ],
            axis=-1,
        )

    def sequence(
        self,
        image_pad_mask: np.ndarray,
        image_shapes: tuple[tuple[int, int], ...],
    ) -> tuple[mx.array, np.ndarray]:
        """Build official interleaved text/image 3D RoPE and per-token image block ids."""
        mask = np.asarray(image_pad_mask, dtype=np.bool_).reshape(-1)
        total_len = int(mask.size)
        frame: list[float] = []
        height_axis: list[float] = []
        width_axis: list[float] = []
        image_ids = np.full((total_len,), -1, dtype=np.int32)
        cursor = 0
        position = 0
        for image_index, (height, width) in enumerate(image_shapes):
            candidates = np.flatnonzero(mask[cursor:])
            if not len(candidates):
                raise ValueError("Qwen-Image 2.1 image_pad_mask 缺少图片槽")
            block_start = cursor + int(candidates[0])
            text_len = block_start - cursor
            text_positions = list(range(position, position + text_len))
            frame.extend(text_positions)
            height_axis.extend(text_positions)
            width_axis.extend(text_positions)
            position += text_len

            block_len = int(height) * int(width)
            block_end = block_start + block_len
            if block_end > total_len or not bool(mask[block_start:block_end].all()):
                raise ValueError(
                    f"Qwen-Image 2.1 第 {image_index + 1} 个图片块需要 {block_len} 个连续槽"
                )
            frame.extend([position] * block_len)
            h_values = np.arange(-(height - height // 2), height // 2, dtype=np.float32)
            w_values = np.arange(-(width - width // 2), width // 2, dtype=np.float32)
            height_axis.extend(np.repeat(h_values, width).tolist())
            width_axis.extend(np.tile(w_values, height).tolist())
            image_ids[block_start:block_end] = image_index
            cursor = block_end
            position += max(int(height), int(width))

        if cursor < total_len:
            trailing = list(range(position, position + total_len - cursor))
            frame.extend(trailing)
            height_axis.extend(trailing)
            width_axis.extend(trailing)
        if len(frame) != total_len:
            raise ValueError("Qwen-Image 2.1 RoPE 序列长度与条件布局不一致")
        return (
            mx.concatenate(
                [
                    self._angles(mx.array(frame, dtype=mx.float32), self.axes_dim[0]),
                    self._angles(mx.array(height_axis, dtype=mx.float32), self.axes_dim[1]),
                    self._angles(mx.array(width_axis, dtype=mx.float32), self.axes_dim[2]),
                ],
                axis=-1,
            ),
            image_ids,
        )


def _prefix_segments(
    image_ids: np.ndarray, prefix_len: int
) -> tuple[tuple[int, int, bool], ...]:
    """Split a text/reference prefix into causal text runs and bidirectional image blocks."""
    if prefix_len <= 0:
        return ()
    values = image_ids[:prefix_len]
    segments: list[tuple[int, int, bool]] = []
    start = 0
    for index in range(1, prefix_len + 1):
        if index == prefix_len or values[index] != values[start]:
            segments.append((start, index, bool(values[start] < 0)))
            start = index
    return tuple(segments)


class QwenImage21Transformer(nn.Module):
    """Unified Qwen-Image 2.1 T2I/edit transformer with block-causal prefix attention."""

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
        causal_condition: bool = True,
    ):
        super().__init__()
        if patch_size != 1:
            raise ValueError(f"Qwen-Image 2.1 MLX 当前只接受 patch_size=1，收到 {patch_size}")
        if not causal_condition:
            raise ValueError("Qwen-Image 2.1 发布权重要求 causal_condition=true")
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_layers = num_layers
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = QwenImage21Rope(10_000.0, axes_dims_rope)
        self.time_text_embed = QwenImage21TimeTextEmbed(self.inner_dim)
        self.txt_in = QwenImage21TextProjection(context_in_dim, self.inner_dim, eps)
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.modulation = [nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False)]
        self.transformer_blocks = [
            QwenImage21TransformerBlock(
                self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, eps
            )
            for _ in range(num_layers)
        ]
        self.norm_out = QwenImage21AdaLayerNormContinuous(self.inner_dim, eps)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        height: int,
        width: int,
        kv_cache: QwenImage21KVCache | None = None,
        kv_cache_mode: str | None = None,
        reference_latents: mx.array | None = None,
        reference_shapes: tuple[tuple[int, int], ...] = (),
        image_pad_mask: mx.array | None = None,
    ) -> mx.array:
        if kv_cache is not None and kv_cache_mode not in ("extract", "cached"):
            raise ValueError("传入 KV cache 时 mode 必须是 extract 或 cached")
        if kv_cache is None and kv_cache_mode is not None:
            raise ValueError("指定 KV cache mode 时必须同时传 cache")
        target_len = int(height) * int(width)
        if hidden_states.shape[1:] != (target_len, self.in_channels):
            raise ValueError(
                f"Qwen-Image 2.1 latent 应为 [B,{target_len},{self.in_channels}]，"
                f"收到 {tuple(hidden_states.shape)}"
            )

        text_states = self.txt_in(encoder_hidden_states)
        refs = reference_latents
        if refs is not None and refs.ndim == 2:
            refs = refs[None]
        ref_len = 0 if refs is None else int(refs.shape[1])
        shapes = tuple((int(h), int(w)) for h, w in reference_shapes)
        if sum(h * w for h, w in shapes) != ref_len:
            raise ValueError(
                "Qwen-Image 2.1 参考 latent 长度与 reference_shapes 不一致："
                f"seq={ref_len}, shapes={shapes}"
            )
        if (refs is None) != (not shapes):
            raise ValueError("Qwen-Image 2.1 reference_latents 与 reference_shapes 必须同时提供")

        vlm_mask = (
            np.zeros((int(text_states.shape[1]),), dtype=np.bool_)
            if image_pad_mask is None
            else np.asarray(image_pad_mask).astype(np.bool_).reshape(-1)
        )
        if vlm_mask.size != int(text_states.shape[1]):
            raise ValueError(
                f"Qwen-Image 2.1 image_pad_mask 长度 {vlm_mask.size} 与文本 {text_states.shape[1]} 不一致"
            )
        if int(vlm_mask.sum()) * 4 != ref_len:
            raise ValueError(
                "Qwen-Image 2.1 每个视觉槽必须对应 2×2 个参考 latent："
                f"slots={int(vlm_mask.sum())}, ref_seq={ref_len}"
            )

        target_slots = target_len // 4
        base_mask = np.concatenate([vlm_mask, np.ones((target_slots,), dtype=np.bool_)])
        repeats = np.where(base_mask, 4, 1)
        base = mx.concatenate(
            [
                text_states,
                mx.zeros((text_states.shape[0], target_slots, text_states.shape[-1]), dtype=text_states.dtype),
            ],
            axis=1,
        )
        expanded_index = mx.array(np.repeat(np.arange(base_mask.size), repeats), dtype=mx.int32)
        joint = base[:, expanded_index]
        expanded_mask = np.repeat(base_mask, repeats)
        image_input = hidden_states if refs is None else mx.concatenate([refs, hidden_states], axis=1)
        image_states = self.img_in(image_input)
        image_positions = mx.array(np.flatnonzero(expanded_mask), dtype=mx.int32)
        if int(image_positions.shape[0]) != int(image_states.shape[1]):
            raise ValueError("Qwen-Image 2.1 展开后的图片槽与 latent 数量不一致")
        joint[:, image_positions, :] = image_states

        rotary, image_ids = self.pos_embed.sequence(
            expanded_mask, shapes + ((int(height), int(width)),)
        )
        prefix_len = int(joint.shape[1]) - target_len
        segments = _prefix_segments(image_ids, prefix_len)
        if kv_cache_mode == "cached":
            joint = joint[:, prefix_len:]
            prefix_len = 0
            rotary = rotary[-target_len:]
            segments = None

        sampled_t = timestep.astype(joint.dtype).reshape(-1)
        time_rows = mx.concatenate([sampled_t, mx.zeros((1,), dtype=joint.dtype)])
        temb = self.time_text_embed(time_rows, joint)
        modulation = self.modulation[1](self.modulation[0](temb))
        for index, block in enumerate(self.transformer_blocks):
            joint = block(
                joint,
                modulation,
                rotary,
                prefix_len,
                target_len,
                kv_cache.layer(index) if kv_cache is not None else None,
                kv_cache_mode,
                segments,
            )
        joint = self.norm_out(joint, temb, prefix_len, target_len, kv_cache_mode)
        output = self.proj_out(joint)
        return output[:, -target_len:]