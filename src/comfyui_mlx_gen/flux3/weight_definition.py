"""Component definitions for the official Diffusers FLUX 3 Image Klein repo."""

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition

from .weight_mapping import Flux3WeightMapping


def _flux2_vae_mapping():
    from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping

    return Flux2WeightMapping.get_vae_mapping()


class Flux3WeightDefinition:
    @staticmethod
    def get_components():
        return [
            ComponentDefinition(
                name="vae", hf_subdir="vae", precision=ModelConfig.precision,
                mapping_getter=_flux2_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer", hf_subdir="transformer", precision=ModelConfig.precision,
                mapping_getter=Flux3WeightMapping.get_transformer_mapping,
                num_blocks=28,
            ),
            ComponentDefinition(
                name="text_encoder", hf_subdir="text_encoder", precision=ModelConfig.precision,
                mapping_getter=Flux3WeightMapping.get_text_encoder_mapping,
                num_blocks=24, num_layers=36,
            ),
        ]

    @staticmethod
    def get_tokenizers():
        return [
            TokenizerDefinition(
                name="qwen3vl_4b", hf_subdir="text_encoder", tokenizer_class="Qwen2TokenizerFast",
                encoder_class=LanguageTokenizer, max_length=4096, use_chat_template=True,
                chat_template_kwargs={"enable_thinking": False},
            )
        ]

    @staticmethod
    def get_download_patterns():
        return [
            "transformer/*.safetensors", "transformer/*.json", "text_encoder/**",
            "vae/*.safetensors", "vae/*.json", "tokenizer/**",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")