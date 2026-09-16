import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder


class QwenPromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str,
        negative_prompt: str | None,
        prompt_cache: dict[str, tuple[mx.array, ...]],
        qwen_tokenizer: Tokenizer,
        qwen_text_encoder: QwenTextEncoder,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        prompt_embeds, prompt_mask = QwenPromptEncoder.encode_positive_prompt(
            prompt=prompt,
            prompt_cache=prompt_cache,
            qwen_tokenizer=qwen_tokenizer,
            qwen_text_encoder=qwen_text_encoder,
        )
        if negative_prompt is None:
            return prompt_embeds, prompt_mask, None, None

        # 0. Create a cache key that combines both prompts
        cache_key = f"CFG|{prompt}|NEG|{negative_prompt}"

        # 1. Return prompt encodings if already cached
        if cache_key in prompt_cache:
            cached = prompt_cache[cache_key]
            return cached[0], cached[1], cached[2], cached[3]

        # 2. Encode the negative prompt
        neg_output = qwen_tokenizer.tokenize(negative_prompt)
        neg_prompt_embeds, neg_prompt_mask = qwen_text_encoder(
            input_ids=neg_output.input_ids, attention_mask=neg_output.attention_mask
        )

        prompt_embeds, prompt_mask, neg_prompt_embeds, neg_prompt_mask = QwenPromptEncoder._materialize_tensors(
            prompt_embeds,
            prompt_mask,
            neg_prompt_embeds,
            neg_prompt_mask,
        )

        # 4. Cache the result (all 4 values)
        result = (prompt_embeds, prompt_mask, neg_prompt_embeds, neg_prompt_mask)
        prompt_cache[cache_key] = result
        return result

    @staticmethod
    def encode_positive_prompt(
        prompt: str,
        prompt_cache: dict[str, tuple[mx.array, ...]],
        qwen_tokenizer: Tokenizer,
        qwen_text_encoder: QwenTextEncoder,
    ) -> tuple[mx.array, mx.array]:
        cache_key = f"POS|{prompt}"
        if cache_key in prompt_cache:
            cached = prompt_cache[cache_key]
            return cached[0], cached[1]

        pos_output = qwen_tokenizer.tokenize(prompt)
        prompt_embeds, prompt_mask = qwen_text_encoder(
            input_ids=pos_output.input_ids,
            attention_mask=pos_output.attention_mask,
        )
        prompt_embeds, prompt_mask = QwenPromptEncoder._materialize_tensors(prompt_embeds, prompt_mask)
        result = (prompt_embeds, prompt_mask)
        prompt_cache[cache_key] = result
        return result

    @staticmethod
    def _materialize_tensors(*tensors: mx.array) -> tuple[mx.array, ...]:
        materialized = tuple(mx.stop_gradient(tensor) for tensor in tensors)
        mx.eval(*materialized)
        return materialized
