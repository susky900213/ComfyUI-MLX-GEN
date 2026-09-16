import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.lora_compatibility import LoRACompatibility
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import QwenVisionLanguageEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import VisionTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_controlnet import QwenTransformerControlNet
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.tokenizer.qwen_vision_language_processor import QwenVisionLanguageProcessor
from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer
from mflux.models.qwen.weights.qwen_controlnet_weight_definition import QwenControlnetWeightDefinition
from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition


class QwenImageInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
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
        QwenImageInitializer._init_config(model, model_config)
        weights = QwenImageInitializer._load_weights(path)
        QwenImageInitializer._init_tokenizers(model, path)
        QwenImageInitializer._init_models(model)
        QwenImageInitializer._apply_weights(model, weights, quantize)
        QwenImageInitializer._apply_lora(model, lora_paths, lora_scales)

    @staticmethod
    def init_edit(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        # Use model_path if provided, otherwise fall back to model_config.model_name
        path = model_path if model_path else model_config.model_name
        LoRACompatibility.validate_for_model_config(
            model_config=model_config,
            selected_model=path,
            lora_paths=lora_paths,
        )
        QwenImageInitializer._init_config(model, model_config)
        weights = QwenImageInitializer._load_weights(path)
        QwenImageInitializer._init_tokenizers(model, path)
        QwenImageInitializer._init_edit_models(model)
        QwenImageInitializer._apply_weights(model, weights, quantize)
        QwenImageInitializer._apply_lora(model, lora_paths, lora_scales)

        # Add vision-language tokenizer
        raw_tokenizer = model.tokenizers["qwen"].tokenizer
        processor = QwenVisionLanguageProcessor(tokenizer=raw_tokenizer)
        model.tokenizers["qwen_vl"] = QwenVisionLanguageTokenizer(
            processor=processor,
            max_length=1024,
            use_picture_prefix=False,
        )
        model.qwen_vl_encoder = QwenVisionLanguageEncoder(encoder=model.text_encoder.encoder)

    @staticmethod
    def init_controlnet(
        model,
        *,
        controlnet_model: str,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        QwenImageInitializer.init(
            model=model,
            model_config=model_config,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
        controlnet_component = QwenControlnetWeightDefinition.get_controlnet_component()
        repo_id, file_pattern = QwenImageInitializer._parse_controlnet_model_spec(controlnet_model)
        controlnet_weights = WeightLoader.load_single(
            component=controlnet_component,
            repo_id=repo_id,
            file_pattern=file_pattern,
        )
        model.controlnet_model = controlnet_model
        model.transformer_controlnet = QwenTransformerControlNet(
            controlnet_input_dim=QwenImageInitializer._controlnet_input_dim(controlnet_weights),
            num_layers=controlnet_weights.num_transformer_blocks(component_name="transformer_controlnet"),
            axes_dims_rope=model.model_config.transformer_overrides.get("axes_dims_rope"),
        )
        WeightApplier.apply_and_quantize_single(
            weights=controlnet_weights,
            model=model.transformer_controlnet,
            component=controlnet_component,
            quantize_arg=None,
            quantization_predicate=QwenWeightDefinition.quantization_predicate,
        )

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.controlnet_condition_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        weights = WeightLoader.load(
            weight_definition=QwenWeightDefinition,
            model_path=model_path,
        )
        QwenImageInitializer._validate_text_encoder_weights(weights, model_path)
        return weights

    @staticmethod
    def _validate_text_encoder_weights(weights: LoadedWeights, model_path: str) -> None:
        projection = QwenImageInitializer._get_nested_value(
            weights.components,
            "text_encoder.encoder.layers.0.self_attn.q_proj.weight",
        )
        if projection is None or not getattr(projection, "shape", None):
            return

        try:
            has_signal = bool(mx.any(projection != 0).item())
        except (AttributeError, TypeError, ValueError):
            return

        if not has_signal:
            raise ValueError(
                "Qwen text encoder weights appear corrupt: "
                "text_encoder.encoder.layers.0.self_attn.q_proj.weight is all zero. "
                f"Delete the local Hugging Face cache for {model_path!r}, then run "
                f"`mlxgen download --model {model_path}` again."
            )

    @staticmethod
    def _get_nested_value(values: dict, path: str):
        current = values
        for part in path.split("."):
            if isinstance(current, list):
                if not part.isdigit():
                    return None
                index = int(part)
                if index >= len(current):
                    return None
                current = current[index]
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=QwenWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = QwenVAE()
        model.transformer = QwenImageInitializer._transformer_from_config(model.model_config)
        model.text_encoder = QwenTextEncoder()

    @staticmethod
    def _init_edit_models(model) -> None:
        model.vae = QwenVAE()
        model.transformer = QwenImageInitializer._transformer_from_config(model.model_config)
        model.text_encoder = QwenTextEncoder()
        model.text_encoder.encoder.visual = VisionTransformer()

    @staticmethod
    def _transformer_from_config(model_config: ModelConfig) -> QwenTransformer:
        overrides = model_config.transformer_overrides
        return QwenTransformer(
            axes_dims_rope=overrides.get("axes_dims_rope"),
            zero_cond_t=bool(overrides.get("zero_cond_t", False)),
            use_layer3d_rope=bool(overrides.get("use_layer3d_rope", False)),
        )

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=QwenWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        result = LoRALoader.load_and_apply_lora_detailed(
            lora_mapping=QwenLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
        model.lora_application_result = result
        model.lora_application_reports = result.reports
        model.lora_paths = result.resolved_paths
        model.lora_scales = result.resolved_scales

    @staticmethod
    def _parse_controlnet_model_spec(controlnet_model: str) -> tuple[str, str]:
        repo_id, separator, file_pattern = controlnet_model.partition(":")
        if not separator or not repo_id or not file_pattern:
            raise ValueError(
                "--controlnet-model must be an exact Hugging Face reference in the form "
                "'repo-id:path/to/file.safetensors'."
            )
        if not file_pattern.endswith(".safetensors"):
            raise ValueError("--controlnet-model must point to an exact .safetensors file.")
        return repo_id, file_pattern

    @staticmethod
    def _controlnet_input_dim(controlnet_weights: LoadedWeights) -> int:
        weight = QwenImageInitializer._get_nested_value(
            controlnet_weights.components,
            "transformer_controlnet.controlnet_x_embedder.weight",
        )
        if weight is None or len(weight.shape) != 2:
            return 64
        return int(weight.shape[1])
