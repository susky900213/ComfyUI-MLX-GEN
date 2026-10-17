"""FLUX 3 Image Klein DiT in MLX.

The implementation follows the public BFL reference implementation and the
Apache-2.0 Diffusers implementation.  The checkpoint is mode-major: every
sample owns an image stream, an optional reference-image stream and a text
stream.  The streams self-attend independently for five blocks and are then
joined for the 28 single-stream blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn


MODES = ("image", "image_cond", "text")
IN_CHANNELS = {"image": 128, "image_cond": 128, "text": 20_480}
HIDDEN = 3072
HEADS = 24
HEAD_DIM = 128
DEPTH = 5
DEPTH_SINGLE = 28
AXES_DIM = (32, 32, 32, 32)
MLP_RATIO = 3.0
THETA = 10_000


@dataclass
class Flux3Packed:
    """Mode-major inputs.

    ``mode_indices[mode][sample]`` is the half-open range of that sample in
    the corresponding concatenated array.  Arrays use ``[tokens, channels]``
    (batch size is represented by the ranges, matching the BFL reference).
    """

    data: dict[str, mx.array]
    ids: dict[str, mx.array]
    timesteps: dict[str, mx.array]
    mode_indices: dict[str, list[tuple[int, int]]]

    @property
    def num_samples(self) -> int:
        return len(self.mode_indices["image"])


def timestep_embedding(t: mx.array, dim: int = 256, max_period: int = 10_000) -> mx.array:
    """BFL's cosine/sine flow-timestep embedding."""
    value = t.astype(mx.float32).reshape(-1, 1) * 1000.0
    half = dim // 2
    frequencies = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = value * frequencies[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(t.dtype)


def _rope_frequencies(ids: mx.array) -> tuple[mx.array, mx.array]:
    """Return per-pair cos/sin for ``(t, h, w, l)`` ids."""
    positions = ids.astype(mx.float32)
    cos_parts: list[mx.array] = []
    sin_parts: list[mx.array] = []
    for axis, width in enumerate(AXES_DIM):
        inverse = 1.0 / (THETA ** (mx.arange(0, width, 2, dtype=mx.float32) / width))
        angles = positions[:, axis : axis + 1] * inverse[None]
        cos_parts.append(mx.cos(angles))
        sin_parts.append(mx.sin(angles))
    return mx.concatenate(cos_parts, axis=-1), mx.concatenate(sin_parts, axis=-1)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply 2x2 rotary matrices to ``[1, heads, tokens, head_dim]``."""
    shape = x.shape
    pairs = x.astype(mx.float32).reshape(shape[0], shape[1], shape[2], -1, 2)
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = mx.stack([c * real - s * imag, s * real + c * imag], axis=-1)
    return rotated.reshape(shape).astype(x.dtype)


def _rms_norm(x: mx.array, scale: mx.array, eps: float = 1e-6) -> mx.array:
    value = x.astype(mx.float32)
    value = value * mx.rsqrt(mx.mean(mx.square(value), axis=-1, keepdims=True) + eps)
    return (value * scale.astype(mx.float32)).astype(x.dtype)


def _layer_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    value = x.astype(mx.float32)
    mean = mx.mean(value, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(value - mean), axis=-1, keepdims=True)
    return ((value - mean) * mx.rsqrt(variance + eps)).astype(x.dtype)


def _modulated_norm(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return (1.0 + scale) * _layer_norm(x) + shift


def _attention(q: mx.array, k: mx.array, v: mx.array, bounds: list[tuple[int, int]]) -> mx.array:
    """Attention over each sample range; samples never attend to one another."""
    out = mx.zeros_like(q)
    for start, end in bounds:
        if end <= start:
            continue
        part = mx.fast.scaled_dot_product_attention(
            q[:, :, start:end],
            k[:, :, start:end],
            v[:, :, start:end],
            scale=1.0 / math.sqrt(HEAD_DIM),
        )
        out[:, :, start:end] = part
    return out


class Flux3MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.silu(self.in_layer(x)))


class Flux3Modulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 3, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(nn.silu(x))


class Flux3Attention(nn.Module):
    def __init__(self, dim: int, heads: int = HEADS, dim_head: int = HEAD_DIM):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-6)
        self.to_out = [nn.Linear(inner, dim, bias=False)]

    def __call__(self, x: mx.array, rope: tuple[mx.array, mx.array], bounds: list[tuple[int, int]]) -> mx.array:
        tokens = x.shape[0]
        q = self.to_q(x).reshape(1, tokens, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = self.to_k(x).reshape(1, tokens, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = self.to_v(x).reshape(1, tokens, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        q = _apply_rope(self.norm_q(q), *rope)
        k = _apply_rope(self.norm_k(k), *rope)
        attended = _attention(q, k, v, bounds)
        attended = attended.transpose(0, 2, 1, 3).reshape(tokens, -1)
        return self.to_out[0](attended)


class Flux3FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = MLP_RATIO):
        super().__init__()
        inner = int(dim * mult)
        self.linear_in = nn.Linear(dim, inner * 2, bias=False)
        self.linear_out = nn.Linear(inner, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.linear_in(x), 2, axis=-1)
        return self.linear_out(nn.silu(gate) * up)


class Flux3ModeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN, eps=1e-6, affine=False)
        self.attn = Flux3Attention(HIDDEN)
        self.ff = Flux3FeedForward(HIDDEN)

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array],
        bounds: list[tuple[int, int]],
        modulation: mx.array,
    ) -> mx.array:
        shift, scale, gate = mx.split(modulation, 3, axis=-1)
        modulated = _modulated_norm(x, shift, scale)
        return x + gate * (self.attn(modulated, rope, bounds) + self.ff(modulated))


class Flux3SingleBlock(Flux3ModeBlock):
    """Same arithmetic as a mode block, with per-token mode modulation."""


def _join_single_stream_modulations(
    single: dict[str, mx.array],
    placement: list[tuple[str, int, int, int]],
) -> mx.array:
    """Join complete ``(shift, scale, gate)`` vectors in joint-token order.

    ``Flux3Modulation`` already emits all three vectors as one ``3 * HIDDEN``
    array.  The single-stream block splits that array into the three ``HIDDEN``
    parts, so the arrays must be concatenated once here rather than collecting
    the same complete vector once per split part.
    """
    chunks = [single[mode][left:right, :] for mode, left, right, _offset in placement]
    if not chunks:
        raise ValueError("FLUX 3 joint stream 不能为空")
    return mx.concatenate(chunks, axis=0)


class Flux3AdaLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN * 2, bias=False)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        scale, shift = mx.split(self.linear(nn.silu(temb)), 2, axis=-1)
        return _layer_norm(x) * (1.0 + scale) + shift


class Flux3Transformer(nn.Module):
    """FLUX 3 DiT, with ``flow_map=True`` for the distilled checkpoint."""

    def __init__(
        self,
        in_channels: int = 128,
        cond_in_channels: int = 128,
        num_layers: int = DEPTH,
        num_single_layers: int = DEPTH_SINGLE,
        attention_head_dim: int = HEAD_DIM,
        num_attention_heads: int = HEADS,
        joint_attention_dim: int = 20_480,
        timestep_channels: int = 256,
        mlp_ratio: float = MLP_RATIO,
        axes_dims_rope: tuple[int, ...] = AXES_DIM,
        rope_theta: int = THETA,
        eps: float = 1e-6,
        flow_map: bool = False,
        **_unused: Any,
    ):
        super().__init__()
        if (in_channels, cond_in_channels, attention_head_dim, num_attention_heads) != (128, 128, 128, 24):
            raise ValueError("FLUX 3 Image Klein 配置必须是 128/128 latent、24×128 attention")
        if tuple(axes_dims_rope) != AXES_DIM or int(rope_theta) != THETA:
            raise ValueError("FLUX 3 Image Klein 的 RoPE 配置必须是 axes=(32,32,32,32), theta=10000")
        self.flow_map = bool(flow_map)
        self.num_layers = int(num_layers)
        self.num_single_layers = int(num_single_layers)
        self.time_in = Flux3MLPEmbedder(timestep_channels, HIDDEN)
        self.ref_time_in = Flux3MLPEmbedder(timestep_channels, HIDDEN) if self.flow_map else None
        self.x_embedders = {
            "image": nn.Linear(in_channels, HIDDEN, bias=False),
            "image_cond": nn.Linear(cond_in_channels, HIDDEN, bias=False),
        }
        self.context_embedder = nn.Linear(joint_attention_dim, HIDDEN, bias=False)
        self.early_stream_modulations = {
            name: Flux3Modulation(HIDDEN) for name in ("image", "image_cond", "txt")
        }
        self.single_stream_modulations = {
            name: Flux3Modulation(HIDDEN) for name in ("image", "image_cond", "txt")
        }
        self.mode_blocks = {
            name: [Flux3ModeBlock() for _ in range(self.num_layers)]
            for name in ("image", "image_cond")
        }
        self.txt_mode_blocks = [Flux3ModeBlock() for _ in range(self.num_layers)]
        self.single_transformer_blocks = [Flux3SingleBlock() for _ in range(self.num_single_layers)]
        self.norm_out = Flux3AdaLayerNorm()
        self.proj_out = nn.Linear(HIDDEN, in_channels, bias=False)

    @staticmethod
    def _mode_key(mode: str) -> str:
        return "txt" if mode == "text" else mode

    @staticmethod
    def _ranges_for_joint(packed: Flux3Packed) -> tuple[list[tuple[int, int]], list[tuple[str, int, int, int]]]:
        # BFL's Flux3SingleTransformerBlock receives the target image first,
        # then the optional reference-image tokens, and finally text.  Keep the
        # mode-major storage layout and the official MODES order here.  The
        # reference implementation packs samples as [text, image_cond, image]
        # only while building the mode-major arrays; that is not the order of
        # the joint single-stream sequence.
        joint_modes = MODES
        bounds: list[tuple[int, int]] = []
        placement: list[tuple[str, int, int, int]] = []
        offset = 0
        for sample in range(packed.num_samples):
            start = offset
            for mode in joint_modes:
                left, right = packed.mode_indices[mode][sample]
                if right > left:
                    length = right - left
                    placement.append((mode, left, right, offset))
                    offset += length
            bounds.append((start, offset))
        return bounds, placement

    def __call__(self, packed: Flux3Packed, timesteps_ref: mx.array | None = None) -> mx.array:
        if (timesteps_ref is not None) != self.flow_map:
            raise ValueError("flow-map DiT 必须且只能提供 timesteps_ref")
        data = dict(packed.data)
        ids = dict(packed.ids)
        effective_timesteps = dict(packed.timesteps)
        # Empty streams still need a projection/modulation path; the phantom row
        # is never included in any real sample range.
        for mode in MODES:
            if data[mode].shape[0] == 0:
                data[mode] = mx.zeros((1, IN_CHANNELS[mode]), dtype=data["image"].dtype)
                ids[mode] = mx.zeros((1, 4), dtype=mx.float32)
                effective_timesteps[mode] = mx.zeros((1,), dtype=data["image"].dtype)

        vec: dict[str, mx.array] = {}
        for mode in MODES:
            vec[mode] = self.time_in(
                timestep_embedding(effective_timesteps[mode].astype(data[mode].dtype))
            )
        if timesteps_ref is not None:
            ref = self.ref_time_in(timestep_embedding(timesteps_ref.astype(data["image"].dtype)))
            for mode in MODES:
                additions = []
                for sample, (left, right) in enumerate(packed.mode_indices[mode]):
                    if right > left:
                        additions.append(mx.broadcast_to(ref[sample : sample + 1], (right - left, HIDDEN)))
                if additions:
                    vec[mode] = vec[mode] + mx.concatenate(additions, axis=0)

        projected = {
            mode: (self.context_embedder(data[mode]) if mode == "text" else self.x_embedders[mode](data[mode]))
            for mode in MODES
        }
        early = {
            mode: self.early_stream_modulations[self._mode_key(mode)](vec[mode])
            for mode in MODES
        }
        single = {
            mode: self.single_stream_modulations[self._mode_key(mode)](vec[mode])
            for mode in MODES
        }
        mode_ropes = {mode: _rope_frequencies(ids[mode]) for mode in MODES}
        for index in range(self.num_layers):
            for mode in ("image", "image_cond"):
                projected[mode] = self.mode_blocks[mode][index](
                    projected[mode], mode_ropes[mode], packed.mode_indices[mode], early[mode]
                )
            projected["text"] = self.txt_mode_blocks[index](
                projected["text"], mode_ropes["text"], packed.mode_indices["text"], early["text"]
            )

        bounds, placement = self._ranges_for_joint(packed)
        chunks: list[mx.array] = []
        id_chunks: list[mx.array] = []
        for mode, left, right, _offset in placement:
            chunks.append(projected[mode][left:right])
            id_chunks.append(ids[mode][left:right])
        hidden = mx.concatenate(chunks, axis=0)
        joint_ids = mx.concatenate(id_chunks, axis=0)
        joint_rope = _rope_frequencies(joint_ids)
        # Each modulation array already contains one complete triplet per token;
        # concatenate it once in joint-token order.  Flux3SingleBlock splits it
        # into shift / scale / gate, each with width HIDDEN.
        hidden_modulation = _join_single_stream_modulations(single, placement)
        for block in self.single_transformer_blocks:
            hidden = block(hidden, joint_rope, bounds, hidden_modulation)

        image = mx.zeros_like(projected["image"])
        for mode, left, right, offset in placement:
            if mode == "image":
                image[left:right] = hidden[offset : offset + right - left]
        return self.proj_out(self.norm_out(image, vec["image"]))