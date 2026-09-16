from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.fibo.model.fibo_text_encoder import SmolLM3_3B_TextEncoder
from mflux.models.fibo.model.fibo_transformer import FiboTransformer
from mflux.models.fibo.model.fibo_vae.wan_2_2_vae import Wan2_2_VAE
from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition


class FIBOInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        if lora_paths or lora_scales:
            raise ValueError("FIBO does not support LoRA weights in MLX-Gen yet.")
        path = model_path if model_path else model_config.model_name
        FIBOInitializer._init_config(model, model_config)
        weights = FIBOInitializer._load_weights(path)
        FIBOInitializer._init_tokenizers(model, path)
        FIBOInitializer._init_models(model)
        FIBOInitializer._validate_weights(weights)
        FIBOInitializer._apply_weights(model, weights, quantize)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=FIBOWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=FIBOWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = Wan2_2_VAE()
        model.text_encoder = SmolLM3_3B_TextEncoder()
        model.transformer = FiboTransformer()

    @staticmethod
    def _validate_weights(weights: LoadedWeights) -> None:
        transformer_weights = weights.components.get("transformer", {})
        if not FIBOInitializer._has_weight(transformer_weights, "norm_out.linear.bias"):
            raise ValueError(
                "FIBO transformer weights are missing `norm_out.linear.bias`. "
                "Re-download the source model or re-prepare the FIBO folder with the current MLX-Gen version."
            )
        sensitive_quantized_paths = [
            path
            for path in FIBOInitializer._iter_quantized_paths(transformer_weights)
            if FIBOWeightDefinition._is_q8_sensitive_path(path)
        ]
        if sensitive_quantized_paths:
            raise ValueError(
                "FIBO transformer weights use an incompatible older quantization layout for "
                f"`{sensitive_quantized_paths[0]}`. Re-prepare the FIBO folder with the current MLX-Gen version."
            )

        vae_weights = weights.components.get("vae", {})
        if FIBOInitializer._contains_quantization_tensors(vae_weights):
            raise ValueError(
                "FIBO VAE weights use an incompatible older quantization layout. "
                "Re-prepare the FIBO folder with the current MLX-Gen version."
            )

    @staticmethod
    def _has_weight(weights: dict, dotted_key: str) -> bool:
        if dotted_key in weights:
            return True
        current = weights
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return True

    @staticmethod
    def _contains_quantization_tensors(weights: dict) -> bool:
        stack = [weights]
        while stack:
            current = stack.pop()
            if not isinstance(current, dict):
                continue
            if "scales" in current or "biases" in current:
                return True
            stack.extend(value for value in current.values() if isinstance(value, dict))
        return False

    @staticmethod
    def _iter_quantized_paths(weights: dict, prefix: str = ""):
        if not isinstance(weights, dict):
            return
        for key in weights:
            if isinstance(key, str) and (key.endswith(".scales") or key.endswith(".biases")):
                yield key.rsplit(".", 1)[0]
        if "scales" in weights or "biases" in weights:
            yield prefix
        for key, value in weights.items():
            if isinstance(value, dict):
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from FIBOInitializer._iter_quantized_paths(value, next_prefix)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=FIBOWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )
