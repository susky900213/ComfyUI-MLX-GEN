"""Component definitions for an official Diffusers Fibo-2 snapshot."""

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition

from ..moss.weight_definition import _vae_mapping
from .backend import FIBO2_DIFFUSERS_MAX_SEQUENCE_LENGTH
from .weight_mapping import Fibo2WeightMapping


class Fibo2WeightDefinition:
    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        """Quantize the DiT backbone but keep the timestep MLP in BF16.

        Fibo-2's AdaLN modulation is driven by ``t_embedder``.  Quantizing that
        small MLP collapses its output magnitude by roughly an order of
        magnitude and makes every denoising block follow the wrong trajectory.
        The official Diffusers model likewise excludes ``t_embedder`` from
        layerwise low-precision casting.
        """

        if path == "t_embedder" or path.startswith("t_embedder."):
            return False
        return hasattr(module, "to_quantized")

    @staticmethod
    def get_components():
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                mapping_getter=Fibo2WeightMapping.get_transformer_mapping,
                loading_mode="multi_json",
                precision=ModelConfig.precision,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                mapping_getter=Fibo2WeightMapping.get_text_encoder_mapping,
                num_blocks=24,
                num_layers=36,
                loading_mode="single",
                precision=ModelConfig.precision,
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                mapping_getter=_vae_mapping,
                loading_mode="single",
                precision=ModelConfig.precision,
            ),
        ]

    @staticmethod
    def get_tokenizers():
        return [
            TokenizerDefinition(
                name="fibo2",
                hf_subdir="processor",
                tokenizer_class="Qwen2TokenizerFast",
                encoder_class=LanguageTokenizer,
                # Runtime default used by Diffusers' BriaFibo2Pipeline; this
                # is not a published Bria model-architecture hard limit.
                max_length=FIBO2_DIFFUSERS_MAX_SEQUENCE_LENGTH,
                add_special_tokens=False,
            )
        ]

    @staticmethod
    def get_download_patterns():
        return [
            "model_index.json",
            "processor/**",
            "scheduler/**",
            "text_encoder/**",
            "transformer/**",
            "vae/**",
        ]


__all__ = ["Fibo2WeightDefinition"]