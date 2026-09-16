from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.qwen.model.qwen_transformer.qwen_rope import QwenEmbedRopeMLX
from mflux.models.qwen.model.qwen_transformer.qwen_time_text_embed import QwenTimeTextEmbed
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_rms_norm import QwenTransformerRMSNorm


class QwenTransformerControlNet(nn.Module):
    def __init__(
        self,
        *,
        controlnet_input_dim: int = 64,
        in_channels: int = 64,
        num_layers: int = 5,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        axes_dims_rope: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.controlnet_x_embedder = nn.Linear(controlnet_input_dim, self.inner_dim)
        self.txt_norm = QwenTransformerRMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        self.time_text_embed = QwenTimeTextEmbed(timestep_proj_dim=256, inner_dim=self.inner_dim)
        self.pos_embed = QwenEmbedRopeMLX(theta=10000, axes_dim=axes_dims_rope or [16, 56, 56], scale_rope=True)
        self.transformer_blocks = [
            QwenTransformerBlock(dim=self.inner_dim, num_heads=num_attention_heads, head_dim=attention_head_dim)
            for _ in range(num_layers)
        ]
        self.controlnet_blocks = [nn.Linear(self.inner_dim, self.inner_dim) for _ in range(num_layers)]

    def __call__(
        self,
        *,
        t: int,
        config: Config,
        hidden_states: mx.array,
        controlnet_cond: mx.array,
        conditioning_scale: float,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
    ) -> list[mx.array]:
        hidden_states = self.img_in(hidden_states)
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)
        batch_size = hidden_states.shape[0]
        timestep = QwenTransformer._compute_timestep(t, config)
        timestep = mx.broadcast_to(timestep, (batch_size,)).astype(hidden_states.dtype)
        img_shapes = QwenTransformer._compute_image_shapes(config=config)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        text_embeddings = self.time_text_embed(timestep, hidden_states)
        image_rotary_embeddings = QwenTransformer._compute_rotary_embeddings(
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            pos_embed=self.pos_embed,
            img_shapes=img_shapes,
        )
        controlnet_block_samples: list[mx.array] = []
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = QwenTransformer._apply_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
            )
            controlnet_block_samples.append(self.controlnet_blocks[idx](hidden_states) * conditioning_scale)
        return controlnet_block_samples
