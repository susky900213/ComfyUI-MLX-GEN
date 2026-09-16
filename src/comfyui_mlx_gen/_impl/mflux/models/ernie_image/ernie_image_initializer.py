from pathlib import Path

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.download_policy import DownloadRequiredError, is_huggingface_repo_id
from mflux.models.common.lora.lora_compatibility import LoRACompatibility
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.ernie_image.model.ernie_image_transformer import ErnieImageTransformer2DModel
from mflux.models.ernie_image.model.ernie_image_vae import ErnieImageVAE
from mflux.models.ernie_image.model.mistral3_text_encoder import Mistral3CausalLM, Mistral3TextEncoder
from mflux.models.ernie_image.weights.ernie_image_lora_mapping import ErnieImageLoRAMapping
from mflux.models.ernie_image.weights.ernie_image_weight_definition import (
    ErnieImagePromptEnhancerWeightDefinition,
    ErnieImageWeightDefinition,
)


class ErnieImageInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        LoRACompatibility.validate_for_model_config(
            model_config=model_config,
            selected_model=path,
            lora_paths=lora_paths,
        )
        ErnieImageInitializer._init_config(model, model_config)
        model.model_path = path
        weights = ErnieImageInitializer._load_weights(path)
        ErnieImageInitializer._init_tokenizers(model, path)
        ErnieImageInitializer._init_models(model)
        ErnieImageInitializer._apply_weights(model, weights, quantize)
        ErnieImageInitializer._apply_lora(model, lora_paths, lora_scales)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=ErnieImageWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=ErnieImageWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = ErnieImageVAE()
        model.transformer = ErnieImageTransformer2DModel()
        model.text_encoder = Mistral3TextEncoder()
        model.prompt_enhancer = None

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=ErnieImageWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        result = LoRALoader.load_and_apply_lora_detailed(
            lora_mapping=ErnieImageLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
        model.lora_application_result = result
        model.lora_application_reports = result.reports
        model.lora_paths = result.resolved_paths
        model.lora_scales = result.resolved_scales

    @staticmethod
    def init_prompt_enhancer(model) -> None:
        local_path = Path(model.model_path).expanduser()
        if local_path.exists() and not (local_path / "pe").exists():
            raise FileNotFoundError(
                "ERNIE Prompt Enhancer files were not found in the local model path. "
                "Use a full ERNIE Image Turbo source snapshot, or run "
                "`mlxgen download --model baidu/ERNIE-Image-Turbo --all-files`."
            )

        try:
            weights = WeightLoader.load(
                weight_definition=ErnieImagePromptEnhancerWeightDefinition,
                model_path=model.model_path,
            )
            tokenizer = TokenizerLoader.load(
                definition=ErnieImagePromptEnhancerWeightDefinition.get_tokenizers()[0],
                model_path=model.model_path,
            )
        except DownloadRequiredError as exc:
            if is_huggingface_repo_id(model.model_path):
                raise DownloadRequiredError(
                    model.model_path,
                    artifact="prompt enhancer",
                    message=(
                        "MLX-Gen will not download ERNIE Prompt Enhancer files during generation.\n"
                        "Download the full ERNIE Image Turbo repository before using --use-prompt-enhancer:\n"
                        f"  mlxgen download --model {model.model_path} --all-files\n"
                        "Then run generation again with the same command."
                    ),
                    download_command=f"mlxgen download --model {model.model_path} --all-files",
                    prepare_command=None,
                ) from exc
            raise
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "ERNIE Prompt Enhancer files were not found in the local model path. "
                "Use a full ERNIE Image Turbo source snapshot, or run "
                "`mlxgen download --model baidu/ERNIE-Image-Turbo --all-files`."
            ) from exc

        model.prompt_enhancer = Mistral3CausalLM()
        model.tokenizers["ernie_prompt_enhancer"] = tokenizer
        WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=model.bits,
            weight_definition=ErnieImagePromptEnhancerWeightDefinition,
            models={"prompt_enhancer": model.prompt_enhancer},
        )
