from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.qwen.model.qwen_transformer.qwen_rope import QwenEmbedLayer3DRopeMLX, QwenEmbedRopeMLX
from mflux.models.qwen.model.qwen_transformer.qwen_time_text_embed import QwenTimeTextEmbed
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_rms_norm import QwenTransformerRMSNorm


class QwenTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        patch_size: int = 2,
        axes_dims_rope: list[int] | None = None,
        zero_cond_t: bool = False,
        use_layer3d_rope: bool = False,
    ) -> None:
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.zero_cond_t = zero_cond_t
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_norm = QwenTransformerRMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        self.time_text_embed = QwenTimeTextEmbed(timestep_proj_dim=256, inner_dim=self.inner_dim)
        rope_class = QwenEmbedLayer3DRopeMLX if use_layer3d_rope else QwenEmbedRopeMLX
        self.pos_embed = rope_class(theta=10000, axes_dim=axes_dims_rope or [16, 56, 56], scale_rope=True)
        self.transformer_blocks = [QwenTransformerBlock(dim=self.inner_dim, num_heads=num_attention_heads, head_dim=attention_head_dim, zero_cond_t=zero_cond_t) for i in range(num_layers)]  # fmt: off
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels)

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        qwen_image_ids: mx.array | None = None,
        cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
        controlnet_block_samples: list[mx.array] | None = None,
    ) -> mx.array:
        hidden_states = self.img_in(hidden_states)
        batch_size = hidden_states.shape[0]
        timestep = QwenTransformer._compute_timestep(t, config)
        timestep = mx.broadcast_to(timestep, (batch_size,)).astype(hidden_states.dtype)
        img_shapes = QwenTransformer._compute_image_shapes(config=config, cond_image_grid=cond_image_grid)
        modulate_index = None
        if self.zero_cond_t:
            timestep = mx.concatenate([timestep, mx.zeros_like(timestep)], axis=0)
            modulate_index = QwenTransformer._compute_modulate_index(
                batch_size=batch_size,
                img_shapes=img_shapes,
                sequence_length=hidden_states.shape[1],
            )
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        text_embeddings = self.time_text_embed(timestep, hidden_states)
        image_rotary_embeddings = QwenTransformer._compute_rotary_embeddings(
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            pos_embed=self.pos_embed,
            img_shapes=img_shapes,
        )
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = QwenTransformer._apply_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                modulate_index=modulate_index,
            )
            hidden_states = QwenTransformer._apply_controlnet_residual(
                hidden_states=hidden_states,
                block_index=idx,
                total_block_count=len(self.transformer_blocks),
                controlnet_block_samples=controlnet_block_samples,
            )
        if self.zero_cond_t:
            text_embeddings = mx.split(text_embeddings, 2, axis=0)[0]
        hidden_states = self.norm_out(hidden_states, text_embeddings)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states

    @staticmethod
    def _apply_transformer_block(
        idx: int,
        block: QwenTransformerBlock,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: tuple[mx.array, mx.array],
        modulate_index: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        return block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            text_embeddings=text_embeddings,
            image_rotary_emb=image_rotary_embeddings,
            block_idx=idx,
            modulate_index=modulate_index,
        )

    @staticmethod
    def _compute_timestep(
        t: int | float,
        config: Config,
    ) -> mx.array:
        if isinstance(t, int):
            if t < len(config.scheduler.sigmas):
                timestep_idx = t
                time_step = config.scheduler.sigmas[timestep_idx]
            else:
                timestep_idx = None
                for idx, ts in enumerate(config.scheduler.timesteps):
                    if abs(int(ts.item()) - t) < 1:
                        timestep_idx = idx
                        break
                if timestep_idx is None:
                    time_step = t / 1000.0
                else:
                    time_step = config.scheduler.sigmas[timestep_idx]
        else:
            timestep_idx = None
            time_step = t

        timestep = mx.array(np.full((1,), time_step, dtype=np.float32))
        return timestep

    @staticmethod
    def _compute_image_shapes(
        config: Config,
        cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    ) -> list[tuple[int, int, int]]:
        latent_height = config.height // 16
        latent_width = config.width // 16

        if cond_image_grid is None:
            return [(1, latent_height, latent_width)]
        if isinstance(cond_image_grid, list):
            return [(1, latent_height, latent_width)] + cond_image_grid
        return [(1, latent_height, latent_width), cond_image_grid]

    @staticmethod
    def _compute_rotary_embeddings(
        encoder_hidden_states_mask: mx.array,
        pos_embed: QwenEmbedRopeMLX,
        img_shapes: list[tuple[int, int, int]],
    ) -> tuple[mx.array, mx.array]:
        txt_seq_lens = [int(mx.sum(encoder_hidden_states_mask[i]).item()) for i in range(encoder_hidden_states_mask.shape[0])]  # fmt: off
        img_rotary_emb, txt_rotary_emb = pos_embed(video_fhw=img_shapes, txt_seq_lens=txt_seq_lens)
        return img_rotary_emb, txt_rotary_emb

    @staticmethod
    def _compute_modulate_index(
        batch_size: int,
        img_shapes: list[tuple[int, int, int]],
        sequence_length: int,
    ) -> mx.array:
        target_length = QwenTransformer._shape_length(img_shapes[0])
        condition_length = sequence_length - target_length
        if condition_length <= 0:
            return mx.zeros((batch_size, sequence_length), dtype=mx.int32)
        row = mx.concatenate(
            [
                mx.zeros((target_length,), dtype=mx.int32),
                mx.ones((condition_length,), dtype=mx.int32),
            ],
            axis=0,
        )
        return mx.broadcast_to(row[None, :], (batch_size, sequence_length))

    @staticmethod
    def _shape_length(shape: tuple[int, int, int]) -> int:
        frames, height, width = shape
        return frames * height * width

    @staticmethod
    def _apply_controlnet_residual(
        *,
        hidden_states: mx.array,
        block_index: int,
        total_block_count: int,
        controlnet_block_samples: list[mx.array] | None,
    ) -> mx.array:
        if not controlnet_block_samples:
            return hidden_states
        interval = max(1, math.ceil(total_block_count / len(controlnet_block_samples)))
        control_index = min(block_index // interval, len(controlnet_block_samples) - 1)
        return hidden_states + controlnet_block_samples[control_index]
