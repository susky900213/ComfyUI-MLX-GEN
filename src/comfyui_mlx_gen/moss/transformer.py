"""MOSS-Image dense DiT translated from the official PyTorch/ComfyUI model."""

from __future__ import annotations

import math
from dataclasses import dataclass
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


def _activation_dtype(module: nn.Module, fallback: mx.Dtype) -> mx.Dtype:
    """Return the floating activation dtype for ordinary and quantized Linear layers.

    ``QuantizedLinear.weight`` is packed (uint32), so it must never be used as
    the activation dtype.  The bias remains in the requested model precision
    and is a reliable dtype marker for layers that have one; bias-free layers
    use the transformer's explicit fallback.
    """
    if isinstance(module, nn.QuantizedLinear):
        return fallback
    bias = getattr(module, "bias", None)
    if isinstance(bias, mx.array) and bias.dtype in (mx.float16, mx.bfloat16, mx.float32):
        return bias.dtype
    weight = getattr(module, "weight", None)
    if isinstance(weight, mx.array) and weight.dtype in (mx.float16, mx.bfloat16, mx.float32):
        return weight.dtype
    return fallback


def _rope_rotate(x: mx.array, rope: mx.array) -> mx.array:
    """Apply ComfyUI's official split-half 2×2 rotary matrices.

    The upstream ``rope_split_half`` first reshapes a head as
    ``[..., 2, head_dim // 2]`` and moves the pair axis to the end.  Each pair
    is therefore ``(x[i], x[i + head_dim // 2])``; after rotation the result is
    moved back to the original ``[first_half | second_half]`` storage order.

    ``x`` is ``[B,S,heads,head_dim]`` and ``rope`` is
    ``[1,S,head_dim // 2,2,2]``.
    """
    half = x.shape[-1] // 2
    rotation_dtype = rope.dtype
    pairs = x.astype(rotation_dtype).reshape(*x.shape[:-1], 2, half).transpose(
        *range(len(x.shape) - 1), len(x.shape), len(x.shape) - 1
    )
    matrices = rope[:, :, None, ...]
    first = pairs[..., 0][..., None]
    second = pairs[..., 1][..., None]
    rotated = matrices[..., 0] * first + matrices[..., 1] * second
    restored = rotated.transpose(
        *range(len(x.shape) - 1), len(x.shape), len(x.shape) - 1
    )
    return restored.reshape(x.shape).astype(x.dtype)


def _attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rope: mx.array,
    heads: int,
    attention_mask: mx.array | None = None,
    reference_kv: "MOSSBlockKVCache | None" = None,
) -> mx.array:
    batch, sequence, _ = q.shape
    head_dim = q.shape[-1] // heads
    query = q.reshape(batch, sequence, heads, head_dim)
    key = k.reshape(batch, sequence, heads, head_dim)
    value = v.reshape(batch, sequence, heads, head_dim)
    query = _rope_rotate(query, rope).transpose(0, 2, 1, 3)
    key = _rope_rotate(key, rope)
    if reference_kv is not None:
        ref_key = reference_kv.keys.reshape(batch, -1, heads, head_dim)
        ref_value = reference_kv.values.reshape(batch, -1, heads, head_dim)
        # The official MOSS attention keeps the current image/text sequence
        # first and appends the cached reference prefix to its K/V sequence.
        # ``reference_kv.keys`` is already RoPE-rotated when it is captured;
        # applying the reference RoPE again here would rotate it twice.
        key = mx.concatenate([key, ref_key], axis=1)
        value = mx.concatenate([value, ref_value], axis=1)
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    if attention_mask is not None and reference_kv is not None:
        reference_mask = mx.ones(
            (batch, 1, 1, reference_kv.keys.shape[1]), dtype=mx.bool_
        )
        attention_mask = mx.concatenate([attention_mask, reference_mask], axis=-1)
    output = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=1.0 / math.sqrt(head_dim),
        mask=attention_mask,
    )
    return output.transpose(0, 2, 1, 3).reshape(batch, sequence, -1)


@dataclass(frozen=True)
class MOSSBlockKVCache:
    """One block's reference K/V and its reference RoPE metadata.

    ``keys`` is stored after RoPE, matching ComfyUI's ``preprocess_refs``
    path.  ``rope`` is retained as metadata for cache validation/inspection.
    """

    keys: mx.array
    values: mx.array
    rope: mx.array


@dataclass(frozen=True)
class MOSSReferenceKVCache:
    """All reference-branch attention K/V tensors for one MOSS sample."""

    layers: tuple[MOSSBlockKVCache, ...]
    image_tokens: int
    text_tokens: int

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("MOSS reference KV cache 不能为空")
        if self.image_tokens <= 0 or self.text_tokens < 0:
            raise ValueError("MOSS reference KV cache 必须包含 image token，text token 数不能为负")


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

    def __call__(
        self,
        x: mx.array,
        rope: mx.array,
        token_mask: mx.array | None = None,
        activation_dtype: mx.Dtype = mx.bfloat16,
        reference_kv: MOSSBlockKVCache | None = None,
        capture_kv: bool = False,
    ) -> mx.array:
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
        return x + self.ffn_down(
            _situ(self.ffn_up(normalized.astype(_activation_dtype(self.ffn_up, activation_dtype))),
                  self.ffn_gate(normalized.astype(_activation_dtype(self.ffn_gate, activation_dtype))))
        )


class MOSSImageTextProjection(nn.Module):
    def __init__(self, config: MOSSConfig):
        super().__init__()
        self.proj = nn.Linear(config.text_dim, config.dim)
        self.blocks = [MOSSImageTextBlock(config) for _ in range(2)]

    def __call__(
        self,
        x: mx.array,
        rope: mx.array,
        token_mask: mx.array | None = None,
        activation_dtype: mx.Dtype = mx.bfloat16,
    ) -> mx.array:
        x = self.proj(x.astype(_activation_dtype(self.proj, activation_dtype)))
        for block in self.blocks:
            x = block(x, rope, token_mask, activation_dtype)
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
        activation_dtype: mx.Dtype = mx.bfloat16,
        reference_kv: MOSSBlockKVCache | None = None,
        capture_kv: bool = False,
    ) -> mx.array:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mx.split(modulation, 6, axis=1)
        # mx.split(axis=1) already returns [B,1,D], which is the broadcast
        # shape corresponding to PyTorch's ``unbind(1)`` followed by
        # ``[:, None]`` in the reference implementation.
        normalized = self.norm1(x) * (1.0 + scale_a) + shift_a
        dtype = _activation_dtype(self.proj["q"], activation_dtype)
        normalized = normalized.astype(dtype)
        q = self.proj["q"](normalized)
        k = self.proj["k"](normalized)
        value = self.proj["v"](normalized)
        batch, sequence, _ = q.shape
        head_dim = q.shape[-1] // self.heads
        q = self.q_norm(q.reshape(batch, sequence, self.heads, head_dim)).reshape(batch, sequence, -1)
        k = self.k_norm(k.reshape(batch, sequence, self.heads, head_dim)).reshape(batch, sequence, -1)
        rotated_k = _rope_rotate(
            k.reshape(batch, sequence, self.heads, head_dim), rope
        ).reshape(batch, sequence, -1)
        attention_mask = None
        if token_mask is not None:
            attention_mask = token_mask[:, None, :, None] & token_mask[:, None, None, :]
        x = x + self.proj["o"](
            _attention(q, k, value, rope, self.heads, attention_mask, reference_kv)
        ) * gate_a
        normalized = self.norm2(x) * (1.0 + scale_f) + shift_f
        normalized = normalized.astype(_activation_dtype(self.proj["ffn_up_gate"], activation_dtype))
        hidden, gate = mx.split(self.proj["ffn_up_gate"](normalized), 2, axis=-1)
        output = x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f
        if capture_kv:
            return output, MOSSBlockKVCache(rotated_k, value, rope)
        return output


class MOSSImageRoPE:
    def __init__(self, head_dim: int, base: float):
        self.num_bands = head_dim // 8
        self.axis_dim = head_dim // 4
        self.base = float(base)

    def _matrices(self, positions: mx.array) -> mx.array:
        """Turn ``[S,4]`` MOSS coordinates into split-half RoPE matrices."""
        # Match the official MOSS reference pipeline's fp32 construction:
        # ``arange(num_bands) / num_bands``.  For the released head width this
        # is numerically the same endpoint as Flux's per-axis formulation.
        exponent = mx.arange(self.num_bands, dtype=mx.float32) / self.num_bands
        bands = 1.0 / (self.base**exponent)
        phase = positions[:, :, None] * bands[None, None, :]
        sine = mx.sin(phase)
        cosine = mx.cos(phase)
        row0 = mx.stack([cosine, -sine], axis=-1)
        row1 = mx.stack([sine, cosine], axis=-1)
        # [S,4,bands,2,2] → [1,S,4*bands,2,2].  The axis/band order is
        # exactly the order produced by ComfyUI's per-axis rope concatenation.
        return mx.stack([row0, row1], axis=-2).reshape(
            1, positions.shape[0], self.num_bands * 4, 2, 2
        )

    def __call__(
        self,
        height: int,
        width: int,
        text_length: int,
        f_offset: float = 0.0,
    ) -> mx.array:
        """RoPE for the target image followed by its refined text tokens."""
        rows = mx.arange(height, dtype=mx.float32)
        cols = mx.arange(width, dtype=mx.float32)
        grid_h, grid_w = mx.meshgrid(rows, cols, indexing="ij")
        image = mx.stack(
            [
                mx.zeros_like(grid_h),
                mx.full_like(grid_h, float(text_length) + float(f_offset)),
                grid_h,
                grid_w,
            ],
            axis=-1,
        ).reshape(-1, 4)
        image = image.at[:, 2].add(-(height - 1) / 2.0)
        image = image.at[:, 3].add(-(width - 1) / 2.0)
        text = mx.zeros((text_length, 4), dtype=mx.float32)
        text[:, 1] = mx.arange(text_length, dtype=mx.float32) + float(f_offset)
        return self._matrices(mx.concatenate([image, text], axis=0))

    def reference(self, height: int, width: int, reference_index: int = 1) -> mx.array:
        """RoPE for one clean reference image at ``t=0``.

        The official MOSS edit path caches only reference image tokens.  Their
        temporal/reference coordinate is ``r=1, f=0``; no text tokens are part
        of this prefix-only cache.
        """
        rows = mx.arange(height, dtype=mx.float32)
        cols = mx.arange(width, dtype=mx.float32)
        grid_h, grid_w = mx.meshgrid(rows, cols, indexing="ij")
        image = mx.stack(
            [
                mx.full_like(grid_h, float(reference_index)),
                mx.zeros_like(grid_h),
                grid_h,
                grid_w,
            ],
            axis=-1,
        ).reshape(-1, 4)
        image = image.at[:, 2].add(-(height - 1) / 2.0)
        image = image.at[:, 3].add(-(width - 1) / 2.0)
        return self._matrices(image)


def _timestep_embedding(timestep: mx.array, freq_dim: int) -> mx.array:
    """Build the timestep sinusoid in fp32, matching the official pipeline.

    The PyTorch reference computes the frequency/phase path in float64 and
    converts the final sinusoid to float32 before the time MLP.  Metal does not
    support float64, but retaining fp32 here is still essential: casting this
    embedding directly to bf16 changes the modulation vector at every denoising
    step and produces a visibly corrupted image.
    """
    # The official MOSS pipeline computes this directly from the scheduler's
    # sigma/timestep.  Unlike ComfyUI's generic timestep_embedding helper, it
    # does not apply an additional time_factor=1000.
    values = timestep.astype(mx.float32).reshape(-1, 1)
    half = freq_dim // 2
    frequencies = mx.exp(-math.log(10_000.0) * mx.arange(half, dtype=mx.float32) / half)
    phase = values * frequencies[None]
    return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)


def _validate_skip_layers(skip_layers: Iterable[int] | None, depth: int) -> frozenset[int]:
    values = () if skip_layers is None else tuple(skip_layers)
    if any(type(index) is not int or not 0 <= index < depth for index in values):
        raise ValueError(f"MOSS skip_layers 必须是 [0,{depth}) 内的整数")
    return frozenset(values)


class MOSSImageTransformer(nn.Module):
    """Official MOSS dense DiT with spatial latents and optional block skipping."""

    def __init__(
        self,
        config: MOSSConfig | None = None,
        activation_dtype: mx.Dtype | None = None,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = MOSSConfig.from_dict(kwargs) if kwargs else MOSSConfig()
        validate_config(config)
        self.config = config
        self.activation_dtype = activation_dtype or mx.float32
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
        reference_kv: MOSSReferenceKVCache | None = None,
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
        dtype = self.activation_dtype
        time_input = _timestep_embedding(times, self.freq_dim)
        time = self.time_embedding[2](
            nn.silu(
                self.time_embedding[0](
                    time_input.astype(_activation_dtype(self.time_embedding[0], dtype))
                )
            )
        ).astype(mx.float32)
        modulation = self.shared_block_modulation(
            time.astype(_activation_dtype(self.shared_block_modulation, dtype))
        ).reshape(batch, 6, -1)
        head = self.video_head_modulation(
            time.astype(_activation_dtype(self.video_head_modulation, dtype))
        ).reshape(batch, 2, -1)
        if reference_kv is not None and len(reference_kv.layers) != len(self.layers):
            raise ValueError(
                "MOSS reference KV cache 的 block 数量与 Transformer 不一致："
                f"{len(reference_kv.layers)} != {len(self.layers)}"
            )
        outputs = []
        for index in range(batch):
            valid = mask[index].astype(mx.bool_)
            valid_length = int(mx.sum(valid).item())
            if valid_length == 0:
                raise ValueError("每个 MOSS 样本至少需要一个有效文本 token")
            # pad_conditioning only pads on the right.  Slice the valid prefix
            # before text refinement and RoPE construction, matching the
            # official PyTorch ``context = context[:, mask[index].bool()]``
            # path instead of merely masking attention after padding changed
            # the text sequence length and image/text positions.
            if not bool(mx.all(valid[:valid_length]).item()) or bool(mx.any(valid[valid_length:]).item()):
                raise ValueError("MOSS attention_mask 必须是右侧 padding 的前缀有效 mask")
            context = encoder_hidden_states[index : index + 1, :valid_length]
            video = hidden_states[index : index + 1].transpose(0, 2, 3, 1).reshape(1, height * width, -1)
            video = self.video_embedder(video.astype(dtype))
            rope = self.rope(
                height,
                width,
                int(context.shape[1]),
                f_offset=-int(context.shape[1]) if reference_kv is not None else 0.0,
            )
            text = self.text_embedder(
                context,
                rope[:, height * width :],
                None,
                self.activation_dtype,
            )
            value = mx.concatenate([video, text], axis=1)
            token_mask = None
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip:
                    value = layer(
                        value,
                        rope,
                        modulation[index : index + 1],
                        token_mask,
                        self.activation_dtype,
                        reference_kv.layers[layer_index] if reference_kv is not None else None,
                    )
            value = self.final_norm_video(
                value[:, : height * width].astype(self.final_norm_video.weight.dtype)
            )
            value = value * (1.0 + head[index : index + 1, 1:2]) + head[index : index + 1, 0:1]
            value = self.final_linear_video(
                value.astype(_activation_dtype(self.final_linear_video, self.activation_dtype))
            )
            outputs.append(value.transpose(0, 2, 1).reshape(1, -1, height, width))
        return mx.concatenate(outputs, axis=0)

    def build_reference_kv(
        self,
        reference_states: mx.array,
        timestep: mx.array | float,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: mx.array | None = None,
    ) -> MOSSReferenceKVCache:
        """Run the reference branch once and cache every block's K/V.

        The official ``ref-prefix-only-t0`` edit path evaluates only the clean
        reference image at timestep zero.  Text is refined independently in the
        target branch and is not part of this cache.
        """
        if reference_states.ndim != 4 or reference_states.shape[1] != self.latent_channels:
            raise ValueError("MOSS reference_states 必须是 [B,128,H,W]")
        if reference_states.shape[0] != 1 or encoder_hidden_states.shape[0] != 1:
            raise ValueError("MOSS reference KV cache 当前要求 batch size=1")
        if encoder_attention_mask is not None and tuple(encoder_attention_mask.shape) != tuple(
            encoder_hidden_states.shape[:2]
        ):
            raise ValueError("MOSS reference attention_mask 形状不匹配")

        times = mx.array(timestep, dtype=mx.float32).reshape(-1)
        if times.shape[0] != 1 or float(times.item()) != 0.0:
            raise ValueError("MOSS reference KV cache 必须使用 timestep=0")

        _, _, height, width = reference_states.shape
        dtype = self.activation_dtype
        time_input = _timestep_embedding(times, self.freq_dim)
        time = self.time_embedding[2](
            nn.silu(
                self.time_embedding[0](
                    time_input.astype(_activation_dtype(self.time_embedding[0], dtype))
                )
            )
        ).astype(mx.float32)
        modulation = self.shared_block_modulation(
            time.astype(_activation_dtype(self.shared_block_modulation, dtype))
        ).reshape(1, 6, -1)
        video = reference_states.transpose(0, 2, 3, 1).reshape(1, height * width, -1)
        video = self.video_embedder(video.astype(dtype))
        rope = self.rope.reference(height, width)
        value = video
        caches: list[MOSSBlockKVCache] = []
        for layer in self.layers:
            value, block_cache = layer(
                value,
                rope,
                modulation,
                None,
                self.activation_dtype,
                capture_kv=True,
            )
            caches.append(block_cache)
        return MOSSReferenceKVCache(
            tuple(caches),
            image_tokens=height * width,
            text_tokens=0,
        )