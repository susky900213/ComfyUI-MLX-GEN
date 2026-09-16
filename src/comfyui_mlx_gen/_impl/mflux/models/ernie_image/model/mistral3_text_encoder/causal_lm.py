import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.ernie_image.model.mistral3_text_encoder.decoder_layer import Mistral3DecoderLayer
from mflux.models.ernie_image.model.mistral3_text_encoder.rms_norm import Mistral3RMSNorm
from mflux.models.ernie_image.model.mistral3_text_encoder.rope import Mistral3YarnRotaryEmbedding


class Mistral3CausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 3072,
        num_hidden_layers: int = 26,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9216,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        rope_factor: float = 16.0,
        original_max_position_embeddings: int = 16384,
        llama_4_scaling_beta: float = 0.1,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Mistral3DecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                llama_4_scaling_beta=llama_4_scaling_beta,
                original_max_position_embeddings=original_max_position_embeddings,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Mistral3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Mistral3YarnRotaryEmbedding(
            dim=head_dim,
            base=rope_theta,
            factor=rope_factor,
            original_max_position_embeddings=original_max_position_embeddings,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        hidden_states = self._hidden_states(input_ids=input_ids, attention_mask=attention_mask)
        return self.lm_head(hidden_states).astype(mx.float32)

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int,
        eos_token_id: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> mx.array:
        if seed is not None:
            mx.random.seed(seed)

        generated = input_ids.astype(mx.int32)
        for _ in range(max_new_tokens):
            logits = self(generated)[:, -1, :]
            next_token = self._next_token(logits=logits, temperature=temperature, top_p=top_p)
            generated = mx.concatenate([generated, next_token], axis=1)
            mx.eval(generated)
            if int(next_token[0, 0].item()) == eos_token_id:
                break
        return generated

    def _hidden_states(self, input_ids: mx.array, attention_mask: mx.array | None) -> mx.array:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = self._causal_mask(attention_mask, hidden_states, seq_len)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        return self.norm(hidden_states).astype(ModelConfig.precision)

    @staticmethod
    def _next_token(logits: mx.array, temperature: float, top_p: float) -> mx.array:
        if temperature == 1.0 and top_p == 1.0:
            return mx.argmax(logits, axis=-1)[:, None].astype(mx.int32)

        scaled_logits = logits / max(temperature, 1e-6)
        if top_p < 1.0:
            sorted_indices = mx.argsort(scaled_logits, axis=-1)[:, ::-1]
            sorted_logits = mx.take_along_axis(scaled_logits, sorted_indices, axis=-1)
            sorted_probs = nn.softmax(sorted_logits.astype(mx.float32), axis=-1)
            cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
            remove = cumulative_probs > max(top_p, 0.0)
            remove = mx.concatenate([mx.zeros_like(remove[:, :1]), remove[:, :-1]], axis=-1)
            sorted_logits = mx.where(remove, mx.full_like(sorted_logits, -float("inf")), sorted_logits)
            sampled_rank = mx.random.categorical(sorted_logits, num_samples=1)
            return mx.take_along_axis(sorted_indices, sampled_rank, axis=-1).astype(mx.int32)

        return mx.random.categorical(scaled_logits, num_samples=1).astype(mx.int32)

    @staticmethod
    def _causal_mask(attention_mask: mx.array | None, hidden_states: mx.array, seq_len: int) -> mx.array:
        batch_size = hidden_states.shape[0]
        dtype = hidden_states.dtype
        index = mx.arange(seq_len, dtype=mx.int32)
        allowed = index[None, :] <= index[:, None]
        causal_mask = mx.where(
            allowed,
            mx.zeros((seq_len, seq_len), dtype=dtype),
            mx.full((seq_len, seq_len), -float("inf"), dtype=dtype),
        )
        causal_mask = causal_mask[None, None, :, :]
        if attention_mask is None:
            return causal_mask
        padding_mask = mx.where(
            attention_mask[:, None, None, :] == 1,
            mx.zeros((batch_size, 1, 1, seq_len), dtype=dtype),
            mx.full((batch_size, 1, 1, seq_len), -float("inf"), dtype=dtype),
        )
        return causal_mask + padding_mask
