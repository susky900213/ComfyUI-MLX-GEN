import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder


class PromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str,
        tokenizer: Tokenizer,
        text_encoder: TextEncoder,
        prompt_cache: dict[str, mx.array] | None = None,
    ) -> mx.array:
        if prompt_cache is not None and prompt in prompt_cache:
            return prompt_cache[prompt]
        output = tokenizer.tokenize(prompt)
        cap_feats = text_encoder(output.input_ids, output.attention_mask)
        num_valid = int(mx.sum(output.attention_mask[0]).item())
        encodings = mx.stop_gradient(cap_feats[0, :num_valid, :])
        mx.eval(encodings)
        if prompt_cache is not None:
            prompt_cache[prompt] = encodings
        return encodings
