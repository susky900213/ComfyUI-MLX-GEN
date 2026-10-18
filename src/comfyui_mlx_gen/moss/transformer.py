"""MOSS-Image dense DiT translated from the official PyTorch/ComfyUI model."""

from __future__ import annotations

import math
from typing import Iterable

import mlx.core as mx
from mlx import nn

from .config import MOSSConfig, validate_config


def _situ(hidden: mx.array, gate: mx.array) -> mx.array:
    hidden32 = hidden.astype(mx.float32)
    gate32 = gate.astype(mx.float32)
    value = (4.0 * mx.tanh(gate32 / 4.0) * nn.sigmoid(gate32)) * (
        25.0 * mx.tanh(hidden32 / 25.0)
    )
    return value.astype(hidden.dtype)


def _rope_rotate(x: mx.array, rope: mx.array) -> mx.array:
    """Apply the official MOSS split-half rotary convention.

    ``x`` is ``[B,S,heads,head_dim]`` and ``rope`` is ``[1,S,head_dim]``.
    """
    sin, cos = mx.split(rope, 2, axis=-1)
    sin = mx.concatenate([sin, sin], axis=-1)[:, :, None, :].astype(x.dtype)
    cos = mx.concatenate([cos, cos], axis=-1)[:, :, None, :].astype(x.dtype)
    first, second = mx.split(x, 2, axis=-1)
    rotated = mx.concatenate([-second, first], axis=-1)
    return x * cos + rotated * sin


def _attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rope: mx.array,
    heads: int,
    attention_mask: mx.array | None = None,
) -> mx.array:
    batch, sequence, _ = q.shape
    head_dim = q.shape[-1] // heads
    query = q.reshape(batch, sequence, heads, head_dim)
    key = k.reshape(batch, sequence, heads, head_dim)
    value = v.reshape(batch, sequence, heads, head_dim)
    query = _rope_rotate(query, rope).transpose(0, 2, 1, 3)
    key = _rope_rotate(key, rope).transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    output = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=1.0 / math.sqrt(head_dim),
        mask=attention_mask,
    )
    return output.transpose(0, 2, 1, 3).reshape(batch, sequence, -1)


class MOSSImageTextBlock(nn.Module):
    """The two unmodulated text-refinement blocks in the official MOSS DiT."""

    def __init__(self, config: MOSSConfig):
        super().__init__()
        self.heads = config.num_heads
        head_dim = config.dim // config.num_heads
        self.norm1 = nn.RMSNorm(config.dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(config.dim, eps=1e-6)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.proj = nn.Linear(config.dim, config.dim)
        self.ffn_gate = nn.Linear(config.dim, config.ffn_dim)
        self.ffn_up = nn.Linear(config.dim, config.ffn_dim)
        self.ffn_down = nn.Linear(config.ffn_dim, config.dim)

    def __call__(self, x: mx.array, rope: mx.array, token_mask: mx.array | None = None) -> mx.array:
        q, k, value = mx.split(self.qkv(self.norm1(x)), 3, axis=-1)
        batch, sequence, _ = q.shape
        head_dim = q.shape[-1] // self.heads
        q = self.q_norm(q.reshape(batch, sequence, self.heads, head_dim)).reshape(batch, sequence, -1)
        k = self.k_norm(k.reshape(batch, sequence, self.heads, head_dim)).reshape(batch, sequence, -1)
        attention_mask = None
        if token_mask is not None:
            attention_mask = token_mask[:, None, :, None] & token_mask[:, None, None, :]
        x = x + self.proj(_attention(q, k, value, rope, self.heads, attention_mask))
        normalized = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(normalized), self.ffn_gate(normalized)))


class MOSSImageTextProjection(nn.Module):
    def __init__(self, config: MOSSConfig):
        super().__init__()
        self.proj = nn.Linear(config.text_dim, config.dim)
        self.blocks = [MOSSImageTextBlock(config) for _ in range(2)]

    def __call__(self, x: mx.array, rope: mx.array, token_mask: mx.array | None = None) -> mx.array:
        x = self.proj(x.astype(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope, token_mask)
        return x


class MOSSImageBlock(nn.Module):
    def __init__(self, config: MOSSConfig):
        super().__init__()
        self.heads = config.num_heads
        head_dim = config.dim // config.num_heads
        self.norm1 = nn.RMSNorm(config.dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(config.dim, eps=1e-6)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.proj = {
            "q": nn.Linear(config.dim, config.dim, bias=False),
            "k": nn.Linear(config.dim, config.dim, bias=False),
            "v": nn.Linear(config.dim, config.dim, bias=False),
            "o": nn.Linear(config.dim, config.dim, bias=False),
            "ffn_up_gate": nn.Linear(config.dim, 2 * config.ffn_dim, bias=False),
            "ffn_down": nn.Linear(config.ffn_dim, config.dim, bias=False),
        }

    def __call__(
        self,
        x: mx.array,
        rope: mx.array,
        modulation: mx.array,
        token_mask: mx.array | None = None,
    ) -> mx.array:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mx.split(modulation, 6, axis=1)
        # mx.split(axis=1) already returns [B,1,D], which is the broadcast
        # shape corresponding to PyTorch's ``unbind(1)`` followed by
        # ``[:, None]`` in the reference implementation.
        normalized = self.norm1(x) * (1.0 + scale_a) + shift_a
        dtype = self.proj["q"].weight.dtype
        normalized = normalized.astype(dtype)
        q = self.proj["q"](normalized)
        k = self.proj["k"](normalized)
        value = self.proj["v"](normalized)
        batch, sequence, _ = q.shape
        head_dim = q.shape[-1] // self.heads
        q = self.q_norm(q.reshape(batch, sequence, self.heads, head_dim)).reshape(batch, sequence, -1)
        k = self.k_norm(k.reshape(batch, sequence, self.heads, head_dim)).reshape(batch, sequence, -1)
        attention_mask = None
        if token_mask is not None:
            attention_mask = token_mask[:, None, :, None] & token_mask[:, None, None, :]
        x = x + self.proj["o"](
            _attention(q, k, value, rope, self.heads, attention_mask)
        ) * gate_a
        normalized = self.norm2(x) * (1.0 + scale_f) + shift_f
        hidden, gate = mx.split(self.proj["ffn_up_gate"](normalized.astype(dtype)), 2, axis=-1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f


class MOSSImageRoPE:
    def __init__(self, head_dim: int, base: float):
        self.num_bands = head_dim // 8
        self.base = float(base)

    def __call__(self, height: int, width: int, text_length: int) -> mx.array:
        bands = 1.0 / (self.base ** (mx.arange(self.num_bands, dtype=mx.float32) / self.num_bands))
        rows = mx.arange(height, dtype=mx.float32)
        cols = mx.arange(width, dtype=mx.float32)
        grid_h, grid_w = mx.meshgrid(rows, cols, indexing="ij")
        image = mx.stack(
            [mx.zeros_like(grid_h), mx.full_like(grid_h, float(text_length)), grid_h, grid_w], axis=-1
        ).reshape(-1, 4)
        image = image.at[:, 2].add(-(height - 1) / 2.0)
        image = image.at[:, 3].add(-(width - 1) / 2.0)
        text = mx.zeros((text_length, 4), dtype=mx.float32)
        text[:, 1] = mx.arange(text_length, dtype=mx.float32)
        positions = mx.concatenate([image, text], axis=0)
        phase = positions[:, :, None] * bands[None, None, :]
        # [S,4,bands] -> [S,head_dim], matching torch.cat((sin,cos), -2).flatten(-2)
        return mx.concatenate([mx.sin(phase), mx.cos(phase)], axis=1).reshape(1, -1, self.num_bands * 8)


def _timestep_embedding(timestep: mx.array, freq_dim: int, dtype: mx.Dtype) -> mx.array:
    values = timestep.astype(mx.float32).reshape(-1, 1)
    half = freq_dim // 2
    frequencies = mx.exp(-math.log(10_000.0) * mx.arange(half, dtype=mx.float32) / half)
    phase = values * frequencies[None]
    return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1).astype(dtype)


def _validate_skip_layers(skip_layers: Iterable[int] | None, depth: int) -> frozenset[int]:
    values = () if skip_layers is None else tuple(skip_layers)
    if any(type(index) is not int or not 0 <= index < depth for index in values):
        raise ValueError(f"MOSS skip_layers 必须是 [0,{depth}) 内的整数")
    return frozenset(values)


class MOSSImageTransformer(nn.Module):
    """Official MOSS dense DiT with spatial latents and optional block skipping."""

    def __init__(self, config: MOSSConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = MOSSConfig.from_dict(kwargs) if kwargs else MOSSConfig()
        validate_config(config)
        self.config = config
        self.latent_channels = config.latent_channels
        self.freq_dim = config.freq_dim
        self.video_embedder = nn.Linear(config.latent_channels, config.dim)
        self.time_embedding = [nn.Linear(config.freq_dim, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim)]
        self.shared_block_modulation = nn.Linear(config.dim, 6 * config.dim)
        self.video_head_modulation = nn.Linear(config.dim, 2 * config.dim)
        self.text_embedder = MOSSImageTextProjection(config)
        self.rope = MOSSImageRoPE(config.dim // config.num_heads, config.rope_base)
        self.layers = [MOSSImageBlock(config) for _ in range(config.num_layers)]
        self.final_norm_video = nn.RMSNorm(config.dim, eps=1e-6)
        self.final_linear_video = nn.Linear(config.dim, config.latent_channels, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array | float,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: mx.array | None = None,
        skip_layers: Iterable[int] | None = None,
    ) -> mx.array:
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError("MOSS hidden_states 必须是 [B,128,H,W]")
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != hidden_states.shape[0]:
            raise ValueError("MOSS encoder_hidden_states 必须是 [B,S,4096]")
        mask = encoder_attention_mask
        if mask is None:
            mask = mx.ones(encoder_hidden_states.shape[:2], dtype=mx.bool_)
        if tuple(mask.shape) != tuple(encoder_hidden_states.shape[:2]):
            raise ValueError("MOSS encoder_attention_mask 形状不匹配")
        skip = _validate_skip_layers(skip_layers, len(self.layers))
        batch, _, height, width = hidden_states.shape
        times = mx.array(timestep, dtype=mx.float32).reshape(-1)
        if times.shape[0] == 1:
            times = mx.broadcast_to(times, (batch,))
        if times.shape[0] != batch:
            raise ValueError("MOSS timestep 数量必须为 1 或 batch size")
        dtype = self.video_embedder.weight.dtype
        time = self.time_embedding[2](nn.silu(self.time_embedding[0](_timestep_embedding(times, self.freq_dim, dtype))))
        modulation = self.shared_block_modulation(time.astype(self.shared_block_modulation.weight.dtype)).reshape(batch, 6, -1)
        head = self.video_head_modulation(time.astype(self.video_head_modulation.weight.dtype)).reshape(batch, 2, -1)
        outputs = []
        for index in range(batch):
            valid = mask[index].astype(mx.bool_)
            context = encoder_hidden_states[index : index + 1]
            if int(mx.sum(valid).item()) == 0:
                raise ValueError("每个 MOSS 样本至少需要一个有效文本 token")
            video = hidden_states[index : index + 1].transpose(0, 2, 3, 1).reshape(1, height * width, -1)
            video = self.video_embedder(video.astype(dtype))
            rope = self.rope(height, width, int(context.shape[1])).astype(video.dtype)
            text = self.text_embedder(context, rope[:, height * width :], valid[None])
            value = mx.concatenate([video, text], axis=1)
            token_mask = mx.concatenate(
                [mx.ones((1, height * width), dtype=mx.bool_), valid[None]], axis=1
            )
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip:
                    value = layer(value, rope, modulation[index : index + 1], token_mask)
            value = self.final_norm_video(value[:, : height * width].astype(self.final_norm_video.weight.dtype))
            value = value * (1.0 + head[index : index + 1, 1:2]) + head[index : index + 1, 0:1]
            value = self.final_linear_video(value.astype(self.final_linear_video.weight.dtype))
            outputs.append(value.transpose(0, 2, 1).reshape(1, -1, height, width))
        return mx.concatenate(outputs, axis=0)