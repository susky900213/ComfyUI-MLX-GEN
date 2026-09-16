from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_transformer.qwen_attention import QwenAttention
from mflux.models.qwen.model.qwen_transformer.qwen_feed_forward import QwenFeedForward


class QwenTransformerBlock(nn.Module):
    def __init__(self, dim: int = 3072, num_heads: int = 24, head_dim: int = 128, zero_cond_t: bool = False):
        super().__init__()

        self.img_mod_silu = nn.SiLU()
        self.img_mod_linear = nn.Linear(dim, 6 * dim, bias=True)
        self.img_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.attn = QwenAttention(dim=dim, num_heads=num_heads, head_dim=head_dim)
        self.img_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.img_ff = QwenFeedForward(dim=dim)

        self.txt_mod_silu = nn.SiLU()
        self.txt_mod_linear = nn.Linear(dim, 6 * dim, bias=True)
        self.txt_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_ff = QwenFeedForward(dim=dim)
        self.zero_cond_t = zero_cond_t

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array | None,
        text_embeddings: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        block_idx: int | None = None,
        modulate_index: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        img_mod_params = self.img_mod_linear(self.img_mod_silu(text_embeddings))
        txt_embeddings = mx.split(text_embeddings, 2, axis=0)[0] if self.zero_cond_t else text_embeddings
        txt_mod_params = self.txt_mod_linear(self.txt_mod_silu(txt_embeddings))

        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = QwenTransformerBlock._modulate(img_normed, img_mod1, modulate_index)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = QwenTransformerBlock._modulate(txt_normed, txt_mod1)

        img_attn_output, txt_attn_output = self.attn(
            img_modulated=img_modulated,
            txt_modulated=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            block_idx=block_idx,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = QwenTransformerBlock._modulate(img_normed2, img_mod2, modulate_index)

        img_mlp_output = self.img_ff(img_modulated2)

        hidden_states = hidden_states + img_gate2 * img_mlp_output

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = QwenTransformerBlock._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_ff(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        return encoder_hidden_states, hidden_states

    @staticmethod
    def _modulate(x: mx.array, mod_params: mx.array, index: mx.array | None = None) -> tuple[mx.array, mx.array]:
        shift, scale, gate = mx.split(mod_params, 3, axis=-1)
        if index is None:
            return x * (1 + scale[:, None, :]) + shift[:, None, :], gate[:, None, :]

        batch_size = shift.shape[0] // 2
        shift_target, shift_condition = shift[:batch_size], shift[batch_size:]
        scale_target, scale_condition = scale[:batch_size], scale[batch_size:]
        gate_target, gate_condition = gate[:batch_size], gate[batch_size:]

        use_target = mx.expand_dims(index == 0, axis=-1)
        selected_shift = mx.where(use_target, shift_target[:, None, :], shift_condition[:, None, :])
        selected_scale = mx.where(use_target, scale_target[:, None, :], scale_condition[:, None, :])
        selected_gate = mx.where(use_target, gate_target[:, None, :], gate_condition[:, None, :])
        return x * (1 + selected_scale) + selected_shift, selected_gate
