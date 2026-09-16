import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from mflux.models.wan.model.wan_transformer.wan_embedding import WanRotaryPosEmbed, WanTimeTextImageEmbedding
from mflux.models.wan.model.wan_transformer.wan_fp32_layer_norm import FP32LayerNorm
from mflux.models.wan.model.wan_transformer.wan_transformer_block import (
    WanTransformerBlock,
    WanVACETransformerBlock,
)
from mflux.utils.tensor_health import TensorHealth


@dataclass(frozen=True, kw_only=True)
class WanBlockHealthContext:
    step: int | None = None
    total_steps: int | None = None
    timestep: int | float | None = None
    denoiser: str | None = None
    guidance: float | None = None


@dataclass(frozen=True, kw_only=True)
class WanPreparedPackedSegments:
    hidden_states: mx.array
    rotary_emb: tuple[mx.array, mx.array]
    segment_lengths: tuple[int, ...]
    target_segment_index: int
    target_shape: tuple[int, ...]
    batch_size: int


class WanTransformer(nn.Module):
    LOW_PRECISION_DTYPES = (mx.float16, mx.bfloat16)

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        in_channels: int = 48,
        out_channels: int | None = 48,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 14336,
        num_layers: int = 30,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        vace_layers: list[int] | None = None,
        vace_in_channels: int = 96,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.vace_layers = list(vace_layers) if vace_layers else None

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels,
            self.inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=self.inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=self.inner_dim * 6,
            text_embed_dim=text_dim,
        )
        self.blocks = [
            WanTransformerBlock(
                self.inner_dim,
                ffn_dim,
                num_attention_heads,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
                added_kv_proj_dim=added_kv_proj_dim,
            )
            for _ in range(num_layers)
        ]
        if self.vace_layers is not None:
            if max(self.vace_layers) >= num_layers:
                raise ValueError(f"VACE layers {self.vace_layers} exceed the transformer layer count {num_layers}.")
            if 0 not in self.vace_layers:
                raise ValueError("VACE layers must include layer 0.")
            self.vace_patch_embedding = nn.Conv3d(
                vace_in_channels,
                self.inner_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            )
            self.vace_blocks = [
                WanVACETransformerBlock(
                    self.inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                    added_kv_proj_dim=added_kv_proj_dim,
                    apply_input_projection=index == 0,
                )
                for index in range(len(self.vace_layers))
            ]
        self.norm_out = FP32LayerNorm(self.inner_dim, eps=eps, affine=False)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels * math.prod(patch_size), bias=True)
        self.scale_shift_table = mx.random.normal((1, 2, self.inner_dim)) / self.inner_dim**0.5

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        clear_cache_each_block: bool = False,
        block_health_context: WanBlockHealthContext | None = None,
        control_hidden_states: mx.array | None = None,
        control_hidden_states_scale: list[float] | None = None,
        rotary_emb: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """Run the transformer over one latent chunk.

        Args:
            rotary_emb: Precomputed ``(cos, sin)`` tables shaped like the ones
                :class:`WanRotaryPosEmbed` returns, replacing the ones this module
                would build from the token grid. Only a route whose positions do not
                start at zero needs them - SwiftVR restores a clip chunk by chunk
                against one continuous temporal coordinate system - and passing them
                keeps this module from building a second set the caller then ignores.
                ``None`` is the ordinary Wan path.
        """
        if (control_hidden_states is not None) != (self.vace_layers is not None):
            raise ValueError(
                "control_hidden_states must be provided exactly when the transformer is configured with VACE layers. "
                "If you are running a Wan VACE checkpoint, use the WanVace runtime "
                "(CLI: mlxgen-generate-wan --model wan2.1-vace-1.3b)."
            )
        batch_size, _, num_frames, height, width = hidden_states.shape
        if hidden_states.shape[1] != self.in_channels:
            raise ValueError(
                "Wan transformer input channel mismatch: "
                f"got {hidden_states.shape[1]} channels, expected {self.in_channels}."
            )
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        expected_tokens = post_patch_num_frames * post_patch_height * post_patch_width
        if rotary_emb is None:
            rotary_emb = self.rope(hidden_states)
        elif rotary_emb[0].shape[1] != expected_tokens or rotary_emb[1].shape[1] != expected_tokens:
            raise ValueError(
                f"Supplied rotary tables cover {rotary_emb[0].shape[1]} tokens but the post-patch grid "
                f"{(post_patch_num_frames, post_patch_height, post_patch_width)} holds {expected_tokens}."
            )
        hidden_states = self._patch_embed(hidden_states)
        self._check_block_health(
            enabled=self._block_health_enabled(),
            name="patch_embedding",
            tensor=hidden_states,
            context=block_health_context,
        )

        if timestep.ndim == 2:
            timestep_seq_len = timestep.shape[1]
            timestep = timestep.reshape(-1)
        else:
            timestep_seq_len = None

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            timestep_seq_len=timestep_seq_len,
        )
        if timestep_seq_len is not None:
            timestep_proj = timestep_proj.reshape(batch_size, timestep_seq_len, 6, -1)
        else:
            timestep_proj = timestep_proj.reshape(batch_size, 6, -1)
        self._check_block_health(
            enabled=self._block_health_enabled(),
            name="condition_embedder.temb",
            tensor=temb,
            context=block_health_context,
        )
        self._check_block_health(
            enabled=self._block_health_enabled(),
            name="condition_embedder.timestep_proj",
            tensor=timestep_proj,
            context=block_health_context,
        )
        self._check_block_health(
            enabled=self._block_health_enabled(),
            name="condition_embedder.encoder_hidden_states",
            tensor=encoder_hidden_states,
            context=block_health_context,
        )

        control_hints = None
        if control_hidden_states is not None:
            control_hints = self._vace_control_hints(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                control_hidden_states=control_hidden_states,
                control_hidden_states_scale=control_hidden_states_scale,
                timestep_proj=timestep_proj,
                rotary_emb=rotary_emb,
            )

        block_health_enabled = self._block_health_enabled()
        for block_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                block_name=f"blocks.{block_index}",
                block_health_context=block_health_context,
            )
            if control_hints is not None and block_index in control_hints:
                hint, scale = control_hints[block_index]
                hidden_states = hidden_states + hint * scale
            self._check_block_health(
                enabled=block_health_enabled,
                name=f"blocks.{block_index}.hidden_states",
                tensor=hidden_states,
                context=block_health_context,
            )
            if clear_cache_each_block:
                mx.eval(hidden_states)
                mx.clear_cache()

        hidden_states = self._project_out(hidden_states, temb)
        self._check_block_health(
            enabled=block_health_enabled,
            name="proj_out",
            tensor=hidden_states,
            context=block_health_context,
        )
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = mx.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))
        return hidden_states.reshape(batch_size, -1, num_frames, height, width)

    def forward_packed(
        self,
        *,
        latent_segments: Sequence[mx.array],
        source_ids: Sequence[float],
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        target_segment_index: int = -1,
        clear_cache_each_block: bool = False,
        block_health_context: WanBlockHealthContext | None = None,
    ) -> mx.array:
        prepared = self.prepare_packed_segments(
            latent_segments=latent_segments,
            source_ids=source_ids,
            target_segment_index=target_segment_index,
        )
        return self.forward_prepacked(
            prepared=prepared,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            clear_cache_each_block=clear_cache_each_block,
            block_health_context=block_health_context,
        )

    def prepare_packed_segments(
        self,
        *,
        latent_segments: Sequence[mx.array],
        source_ids: Sequence[float],
        target_segment_index: int = -1,
    ) -> WanPreparedPackedSegments:
        if self.vace_layers is not None:
            raise ValueError("Packed Wan execution is not supported by transformers configured with VACE layers.")
        if not latent_segments:
            raise ValueError("latent_segments must contain at least the target segment.")
        if len(latent_segments) != len(source_ids):
            raise ValueError(
                f"latent_segments and source_ids must have the same length, got "
                f"{len(latent_segments)} and {len(source_ids)}."
            )

        segment_count = len(latent_segments)
        if target_segment_index < 0:
            target_segment_index += segment_count
        if target_segment_index < 0 or target_segment_index >= segment_count:
            raise ValueError(
                f"target_segment_index {target_segment_index} is outside the {segment_count} packed segments."
            )

        resolved_source_ids = [float(source_id) for source_id in source_ids]
        for index, source_id in enumerate(resolved_source_ids):
            if not math.isfinite(source_id):
                raise ValueError(f"source_ids[{index}] must be finite, got {source_ids[index]}.")
            if index == target_segment_index:
                if source_id != 0.0:
                    raise ValueError(f"The target segment must use source ID 0, got {source_id}.")
            elif source_id <= 0.0:
                raise ValueError(
                    f"Conditioning segments must use positive source IDs, got {source_id} at index {index}."
                )

        first_segment = latent_segments[0]
        if first_segment.ndim != 5:
            raise ValueError(f"latent_segments[0] must be 5D [B,C,T,H,W], got {first_segment.shape}.")
        batch_size = first_segment.shape[0]
        segment_dtype = first_segment.dtype
        p_t, p_h, p_w = self.patch_size
        for index, segment in enumerate(latent_segments):
            if segment.ndim != 5:
                raise ValueError(f"latent_segments[{index}] must be 5D [B,C,T,H,W], got {segment.shape}.")
            if segment.shape[0] != batch_size:
                raise ValueError(
                    f"All packed segments must use batch size {batch_size}, got {segment.shape[0]} at index {index}."
                )
            if segment.shape[1] != self.in_channels:
                raise ValueError(
                    f"Wan transformer input channel mismatch at segment {index}: "
                    f"got {segment.shape[1]} channels, expected {self.in_channels}."
                )
            if segment.dtype != segment_dtype:
                raise ValueError(
                    f"All packed segments must use dtype {segment_dtype}, got {segment.dtype} at index {index}."
                )
            if segment.shape[2] % p_t or segment.shape[3] % p_h or segment.shape[4] % p_w:
                raise ValueError(
                    f"latent_segments[{index}] shape {segment.shape[2:]} must be divisible by patch size "
                    f"{self.patch_size}."
                )

        packed_segments = []
        rotary_cos = []
        rotary_sin = []
        segment_lengths = []
        for segment, source_id in zip(latent_segments, resolved_source_ids, strict=True):
            cos, sin = self.rope(segment, source_id=source_id)
            patched = self._patch_embed(segment)
            packed_segments.append(patched)
            rotary_cos.append(cos)
            rotary_sin.append(sin)
            segment_lengths.append(patched.shape[1])

        hidden_states = mx.concatenate(packed_segments, axis=1)
        rotary_emb = (
            mx.concatenate(rotary_cos, axis=1),
            mx.concatenate(rotary_sin, axis=1),
        )
        mx.eval(hidden_states, rotary_emb[0], rotary_emb[1])
        return WanPreparedPackedSegments(
            hidden_states=hidden_states,
            rotary_emb=rotary_emb,
            segment_lengths=tuple(segment_lengths),
            target_segment_index=target_segment_index,
            target_shape=tuple(int(value) for value in latent_segments[target_segment_index].shape),
            batch_size=int(batch_size),
        )

    def forward_prepacked(
        self,
        *,
        prepared: WanPreparedPackedSegments,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        clear_cache_each_block: bool = False,
        block_health_context: WanBlockHealthContext | None = None,
    ) -> mx.array:
        if timestep.ndim != 1 or timestep.shape[0] != prepared.batch_size:
            raise ValueError(
                f"Packed Wan execution requires one scalar timestep per batch item, got {timestep.shape} "
                f"for batch size {prepared.batch_size}."
            )
        if encoder_hidden_states.shape[0] != prepared.batch_size:
            raise ValueError(
                f"encoder_hidden_states batch size {encoder_hidden_states.shape[0]} does not match "
                f"the packed latent batch size {prepared.batch_size}."
            )

        hidden_states = prepared.hidden_states
        rotary_emb = prepared.rotary_emb
        self._check_block_health(
            enabled=self._block_health_enabled(),
            name="packed.patch_embedding",
            tensor=hidden_states,
            context=block_health_context,
        )

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        timestep_proj = timestep_proj.reshape(prepared.batch_size, 6, -1)
        block_health_enabled = self._block_health_enabled()
        self._check_block_health(
            enabled=block_health_enabled,
            name="packed.condition_embedder.temb",
            tensor=temb,
            context=block_health_context,
        )
        self._check_block_health(
            enabled=block_health_enabled,
            name="packed.condition_embedder.timestep_proj",
            tensor=timestep_proj,
            context=block_health_context,
        )
        self._check_block_health(
            enabled=block_health_enabled,
            name="packed.condition_embedder.encoder_hidden_states",
            tensor=encoder_hidden_states,
            context=block_health_context,
        )
        for block_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                block_name=f"blocks.{block_index}",
                block_health_context=block_health_context,
            )
            self._check_block_health(
                enabled=block_health_enabled,
                name=f"blocks.{block_index}.hidden_states",
                tensor=hidden_states,
                context=block_health_context,
            )
            if clear_cache_each_block:
                mx.eval(hidden_states)
                mx.clear_cache()

        hidden_states = self._project_out(hidden_states, temb)
        self._check_block_health(
            enabled=block_health_enabled,
            name="packed.proj_out",
            tensor=hidden_states,
            context=block_health_context,
        )
        target_start = sum(prepared.segment_lengths[: prepared.target_segment_index])
        target_end = target_start + prepared.segment_lengths[prepared.target_segment_index]
        target_tokens = hidden_states[:, target_start:target_end]
        return self._unpatch(target_tokens, prepared.target_shape)

    def _patch_embed(self, hidden_states: mx.array) -> mx.array:
        return self._apply_patch_embedding(self.patch_embedding, hidden_states, self.patch_size)

    def _unpatch(self, hidden_states: mx.array, latent_shape: tuple[int, ...]) -> mx.array:
        batch_size, _, num_frames, height, width = latent_shape
        p_t, p_h, p_w = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size,
            num_frames // p_t,
            height // p_h,
            width // p_w,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = mx.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))
        return hidden_states.reshape(batch_size, -1, num_frames, height, width)

    def _vace_control_hints(
        self,
        *,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        control_hidden_states: mx.array,
        control_hidden_states_scale: list[float] | None,
        timestep_proj: mx.array,
        rotary_emb: tuple[mx.array, mx.array],
    ) -> dict[int, tuple[mx.array, float]]:
        scales = control_hidden_states_scale
        if scales is None:
            scales = [1.0] * len(self.vace_layers)
        if len(scales) != len(self.vace_layers):
            raise ValueError(
                f"Length of control_hidden_states_scale {len(scales)} must equal "
                f"the number of VACE layers {len(self.vace_layers)}."
            )
        control = self._apply_patch_embedding(self.vace_patch_embedding, control_hidden_states, self.patch_size)
        # The reference zero-pads the control sequence to the main sequence length so that
        # block 0's proj_in(control) + hidden_states is shape-compatible.
        padding = hidden_states.shape[1] - control.shape[1]
        if padding < 0:
            raise ValueError(
                f"VACE control sequence ({control.shape[1]}) is longer than the main sequence "
                f"({hidden_states.shape[1]})."
            )
        if padding > 0:
            control = mx.concatenate(
                [control, mx.zeros((control.shape[0], padding, control.shape[2]), dtype=control.dtype)],
                axis=1,
            )
        hints: dict[int, tuple[mx.array, float]] = {}
        for index, block in enumerate(self.vace_blocks):
            conditioning_states, control = block(
                hidden_states,
                encoder_hidden_states,
                control,
                timestep_proj,
                rotary_emb,
            )
            hints[self.vace_layers[index]] = (conditioning_states, float(scales[index]))
        return hints

    def _project_out(self, hidden_states: mx.array, temb: mx.array) -> mx.array:
        if temb.ndim == 3:
            shift, scale = mx.split(self.scale_shift_table[None, :, :, :] + temb[:, :, None, :], 2, axis=2)
            shift = mx.squeeze(shift, axis=2)
            scale = mx.squeeze(scale, axis=2)
        else:
            shift, scale = mx.split(self.scale_shift_table + temb[:, None, :], 2, axis=1)
        hidden_states = self.norm_out(hidden_states.astype(mx.float32)) * (1 + scale) + shift
        return self.proj_out(hidden_states.astype(temb.dtype))

    @classmethod
    def _apply_patch_embedding(
        cls,
        patch_embedding: nn.Conv3d,
        hidden_states: mx.array,
        patch_size: tuple[int, int, int],
    ) -> mx.array:
        if hidden_states.dtype != patch_embedding.weight.dtype:
            hidden_states = hidden_states.astype(patch_embedding.weight.dtype)
        batch_size, channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = patch_size
        patches = hidden_states.reshape(
            batch_size,
            channels,
            num_frames // p_t,
            p_t,
            height // p_h,
            p_h,
            width // p_w,
            p_w,
        )
        patches = mx.transpose(patches, (0, 2, 4, 6, 3, 5, 7, 1))
        patches = patches.reshape(
            batch_size,
            (num_frames // p_t) * (height // p_h) * (width // p_w),
            p_t * p_h * p_w * channels,
        )
        output_dtype = hidden_states.dtype
        accumulation_dtype = mx.float32 if output_dtype in cls.LOW_PRECISION_DTYPES else output_dtype
        weight_matrix = patch_embedding.weight.astype(accumulation_dtype).reshape(patch_embedding.weight.shape[0], -1)
        hidden_states = patches.astype(accumulation_dtype) @ mx.transpose(weight_matrix)
        if patch_embedding.bias is not None:
            hidden_states = hidden_states + patch_embedding.bias.astype(accumulation_dtype)
        if hidden_states.dtype != output_dtype:
            hidden_states = hidden_states.astype(output_dtype)
        return hidden_states

    @staticmethod
    def _block_health_enabled() -> bool:
        return os.environ.get("MFLUX_WAN_BLOCK_HEALTH", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
            "blocks",
            "detail",
            "detailed",
            "all",
        }

    @staticmethod
    def _check_block_health(
        *,
        enabled: bool,
        name: str,
        tensor: mx.array,
        context: WanBlockHealthContext | None,
    ) -> None:
        if not enabled:
            return
        TensorHealth.ensure_finite(
            tensor,
            name=f"wan.transformer.{name}",
            phase="wan-transformer-block",
            step=None if context is None else context.step,
            total_steps=None if context is None else context.total_steps,
            timestep=None if context is None else context.timestep,
            denoiser=None if context is None else context.denoiser,
            guidance=None if context is None else context.guidance,
        )
