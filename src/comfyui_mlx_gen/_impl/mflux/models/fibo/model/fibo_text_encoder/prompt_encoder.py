import json
from typing import List, Union

import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.fibo.model.fibo_text_encoder.smol_lm3_3b_text_encoder import SmolLM3_3B_TextEncoder
from mflux.utils.runtime_memory import RuntimeMemory


class PromptEncoder:
    FIBO_BOT_TOKEN = "<|begin_of_text|>"
    FIBO_BOT_TOKEN_ID = 128000

    @staticmethod
    def encode_prompt(
        prompt: str,
        negative_prompt: str | None,
        tokenizer: Tokenizer,
        text_encoder: SmolLM3_3B_TextEncoder,
        guidance: float = 4.0,
        dtype: mx.Dtype | None = None,
        total_transformer_layers: int = 46,
    ) -> tuple[str, mx.array, List[mx.array], mx.array]:
        json.loads(prompt)
        json_prompt = prompt

        prompt_embeds, prompt_layers, prompt_attention_mask = PromptEncoder._get_prompt_embeds(
            prompt=json_prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=3000,
            tokenization_prefix="positive",
        )

        if guidance <= 1.0:
            encoder_hidden_states, prompt_layers = PromptEncoder._prepare_positive_only_output(
                prompt_embeds=prompt_embeds,
                prompt_layers=prompt_layers,
                prompt_attention_mask=prompt_attention_mask,
                total_transformer_layers=total_transformer_layers,
            )
            if dtype is not None:
                encoder_hidden_states, prompt_layers, prompt_attention_mask = PromptEncoder._cast_outputs(
                    encoder_hidden_states=encoder_hidden_states,
                    prompt_layers=prompt_layers,
                    prompt_attention_mask=prompt_attention_mask,
                    dtype=dtype,
                )
            encoder_hidden_states, prompt_layers, prompt_attention_mask = RuntimeMemory.materialize_inference_tree(
                (encoder_hidden_states, prompt_layers, prompt_attention_mask)
            )
            return json_prompt, encoder_hidden_states, prompt_layers, prompt_attention_mask

        if negative_prompt is None:
            negative_prompt = ""

        neg_prompt_embeds, neg_prompt_layers, neg_prompt_attention_mask = PromptEncoder._get_prompt_embeds(
            prompt=negative_prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=3000,
            tokenization_prefix="negative",
        )
        encoder_hidden_states, max_tokens = PromptEncoder._get_encoder_hidden_states(
            prompt_embeds=prompt_embeds,
            neg_prompt_embeds=neg_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            neg_prompt_attention_mask=neg_prompt_attention_mask,
        )
        prompt_attention_mask = PromptEncoder._get_prompt_attention_mask(
            max_tokens=max_tokens,
            prompt_attention_mask=prompt_attention_mask,
            neg_prompt_attention_mask=neg_prompt_attention_mask,
        )

        prompt_layers = PromptEncoder._get_prompt_layers(
            max_tokens=max_tokens,
            prompt_layers=prompt_layers,
            neg_prompt_layers=neg_prompt_layers,
            total_transformer_layers=total_transformer_layers,
        )
        if dtype is not None:
            encoder_hidden_states, prompt_layers, prompt_attention_mask = PromptEncoder._cast_outputs(
                encoder_hidden_states=encoder_hidden_states,
                prompt_layers=prompt_layers,
                prompt_attention_mask=prompt_attention_mask,
                dtype=dtype,
            )
        encoder_hidden_states, prompt_layers, prompt_attention_mask = RuntimeMemory.materialize_inference_tree(
            (encoder_hidden_states, prompt_layers, prompt_attention_mask)
        )
        return json_prompt, encoder_hidden_states, prompt_layers, prompt_attention_mask

    @staticmethod
    def _cast_outputs(
        encoder_hidden_states: mx.array,
        prompt_layers: List[mx.array],
        prompt_attention_mask: mx.array,
        dtype: mx.Dtype,
    ) -> tuple[mx.array, List[mx.array], mx.array]:
        return (
            encoder_hidden_states.astype(dtype),
            [layer.astype(dtype) for layer in prompt_layers],
            prompt_attention_mask.astype(dtype),
        )

    @staticmethod
    def _prepare_positive_only_output(
        prompt_embeds: mx.array,
        prompt_layers: List[mx.array],
        prompt_attention_mask: mx.array,
        total_transformer_layers: int = 46,
    ) -> tuple[mx.array, List[mx.array]]:
        """Prepare encoder output when guidance disables classifier-free guidance."""
        max_tokens = prompt_embeds.shape[1]
        encoder_hidden_states, _ = PromptEncoder._pad_embedding(
            prompt_embeds=prompt_embeds,
            max_tokens=max_tokens,
            attention_mask=prompt_attention_mask,
        )
        prompt_layers = [PromptEncoder._pad_embedding(layer, max_tokens)[0] for layer in prompt_layers]
        if len(prompt_layers) >= total_transformer_layers:
            prompt_layers = prompt_layers[len(prompt_layers) - total_transformer_layers :]
        else:
            prompt_layers = prompt_layers + [prompt_layers[-1]] * (total_transformer_layers - len(prompt_layers))
        return encoder_hidden_states, prompt_layers

    @staticmethod
    def _get_encoder_hidden_states(neg_prompt_attention_mask, neg_prompt_embeds, prompt_attention_mask, prompt_embeds):
        max_tokens = max(neg_prompt_embeds.shape[1], prompt_embeds.shape[1])
        prompt_embeds, prompt_attention_mask = PromptEncoder._pad_embedding(
            max_tokens=max_tokens,
            prompt_embeds=prompt_embeds,
            attention_mask=prompt_attention_mask,
        )
        neg_prompt_embeds, neg_prompt_attention_mask = PromptEncoder._pad_embedding(
            max_tokens=max_tokens,
            prompt_embeds=neg_prompt_embeds,
            attention_mask=neg_prompt_attention_mask,
        )
        encoder_hidden_states = mx.concatenate([neg_prompt_embeds, prompt_embeds], axis=0)
        return encoder_hidden_states, max_tokens

    @staticmethod
    def _get_prompt_attention_mask(max_tokens, neg_prompt_attention_mask, prompt_attention_mask):
        prompt_attention_mask = PromptEncoder._pad_attention_mask(prompt_attention_mask, max_tokens)
        neg_prompt_attention_mask = PromptEncoder._pad_attention_mask(neg_prompt_attention_mask, max_tokens)
        return mx.concatenate([neg_prompt_attention_mask, prompt_attention_mask], axis=0)

    @staticmethod
    def _pad_attention_mask(attention_mask: mx.array, max_tokens: int) -> mx.array:
        batch_size, seq_len = attention_mask.shape
        if max_tokens < seq_len:
            raise ValueError("`max_tokens` must be >= current sequence length.")
        if max_tokens == seq_len:
            return attention_mask
        pad_length = max_tokens - seq_len
        padding = mx.zeros((batch_size, pad_length), dtype=attention_mask.dtype)
        return mx.concatenate([attention_mask, padding], axis=1)

    @staticmethod
    def _get_prompt_layers(max_tokens, neg_prompt_layers, prompt_layers, total_transformer_layers: int = 46):
        prompt_layers = [PromptEncoder._pad_embedding(layer, max_tokens)[0] for layer in prompt_layers]
        neg_prompt_layers = [PromptEncoder._pad_embedding(layer, max_tokens)[0] for layer in neg_prompt_layers]
        prompt_layers = [mx.concatenate([neg_prompt_layers[i], prompt_layers[i]], axis=0) for i in range(len(prompt_layers))]  # fmt: off

        if len(prompt_layers) >= total_transformer_layers:
            prompt_layers = prompt_layers[len(prompt_layers) - total_transformer_layers :]
        else:
            prompt_layers = prompt_layers + [prompt_layers[-1]] * (total_transformer_layers - len(prompt_layers))
        return prompt_layers

    @staticmethod
    def _get_prompt_embeds(
        prompt: Union[str, List[str]],
        text_encoder: SmolLM3_3B_TextEncoder,
        tokenizer: Tokenizer,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 3000,
        tokenization_prefix: str | None = None,
    ) -> tuple[mx.array, List[mx.array], mx.array]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts:
            raise ValueError("`prompt` must be a non-empty string or list of strings.")
        prompts = ["" if p is None else p for p in prompts]
        empty_prompt_rows = [p == "" for p in prompts]
        bot_token_id = PromptEncoder._fibo_bot_token_id(tokenizer)

        # Match Diffusers FIBO: an empty prompt is represented by a real BOT token, not by
        # a zero-length all-masked sequence. This matters for the CFG unconditional branch.
        if all(empty_prompt_rows):
            batch_size = len(prompts)
            input_ids_mx = mx.full((batch_size, 1), bot_token_id, dtype=mx.int32)
            attention_mask_mx = mx.ones_like(input_ids_mx)
        else:
            # 1) Tokenize and convert to MX
            tokenizer_output = tokenizer.tokenize(prompt=prompts, max_length=max_sequence_length)
            input_ids_mx = tokenizer_output.input_ids
            attention_mask_mx = tokenizer_output.attention_mask
            if any(empty_prompt_rows):
                row_mask = mx.array(empty_prompt_rows)[:, None]
                input_ids_mx = mx.where(row_mask, mx.ones_like(input_ids_mx) * bot_token_id, input_ids_mx)
                attention_mask_mx = mx.where(row_mask, mx.ones_like(attention_mask_mx), attention_mask_mx)

        # 2) Run MLX text encoder and collect hidden states
        hidden_states_list = text_encoder(
            input_ids=input_ids_mx,
            attention_mask=attention_mask_mx,
            output_hidden_states=True,
        )

        # 3) Build prompt_embeds from last two layers
        last_hidden = hidden_states_list[-1]
        second_last_hidden = hidden_states_list[-2]
        prompt_embeds = mx.concatenate(
            [last_hidden, second_last_hidden],
            axis=-1,
        )

        # 4) Repeat along batch dimension for num_images_per_prompt
        prompt_embeds = PromptEncoder._repeat_interleave_batch(prompt_embeds, num_images_per_prompt)
        attention_mask = PromptEncoder._repeat_interleave_batch(attention_mask_mx, num_images_per_prompt)
        prompt_layers = [PromptEncoder._repeat_interleave_batch(layer, num_images_per_prompt) for layer in hidden_states_list]  # fmt: off
        return prompt_embeds, prompt_layers, attention_mask

    @staticmethod
    def _fibo_bot_token_id(tokenizer: Tokenizer) -> int:
        raw_tokenizer = getattr(tokenizer, "tokenizer", None)
        convert = getattr(raw_tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            token_id = convert(PromptEncoder.FIBO_BOT_TOKEN)
            if isinstance(token_id, int) and token_id >= 0:
                return token_id
        return PromptEncoder.FIBO_BOT_TOKEN_ID

    @staticmethod
    def _pad_embedding(
        prompt_embeds: mx.array,
        max_tokens: int,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        batch_size, seq_len, dim = prompt_embeds.shape

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=prompt_embeds.dtype)
        else:
            attention_mask = attention_mask.astype(prompt_embeds.dtype)

        if max_tokens < seq_len:
            raise ValueError("`max_tokens` must be >= current sequence length.")

        if max_tokens > seq_len:
            pad_length = max_tokens - seq_len
            padding = mx.zeros((batch_size, pad_length, dim), dtype=prompt_embeds.dtype)
            prompt_embeds = mx.concatenate([prompt_embeds, padding], axis=1)

            mask_padding = mx.zeros((batch_size, pad_length), dtype=attention_mask.dtype)
            attention_mask = mx.concatenate([attention_mask, mask_padding], axis=1)

        return prompt_embeds, attention_mask

    @staticmethod
    def _repeat_interleave_batch(
        tensor: mx.array,
        num_images_per_prompt: int,
    ) -> mx.array:
        if num_images_per_prompt == 1:
            return tensor
        batch, *rest = tensor.shape
        tensor = mx.expand_dims(tensor, axis=1)
        tensor = mx.broadcast_to(tensor, (batch, num_images_per_prompt, *rest))
        new_shape = (batch * num_images_per_prompt, *rest)
        return mx.reshape(tensor, new_shape)
