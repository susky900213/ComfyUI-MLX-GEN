from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping


class BonsaiImageWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_vae_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3",
                hf_subdir="tokenizer",
                tokenizer_class="Qwen2TokenizerFast",
                encoder_class=LanguageTokenizer,
                max_length=512,
                use_chat_template=True,
                chat_template_kwargs={"enable_thinking": False},
                download_patterns=["tokenizer/**", "added_tokens.json", "chat_template.jinja"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "vae/*.safetensors",
            "vae/*.json",
            "transformer-packed-mflux/*.safetensors",
            "transformer-packed-mflux/*.json",
            "text_encoder-mlx-4bit/*.safetensors",
            "text_encoder-mlx-4bit/*.json",
            "tokenizer/**",
            "scheduler/*.json",
            "manifest.json",
            "README.md",
            "LICENSE",
            "NOTICE.md",
            "added_tokens.json",
            "chat_template.jinja",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")
