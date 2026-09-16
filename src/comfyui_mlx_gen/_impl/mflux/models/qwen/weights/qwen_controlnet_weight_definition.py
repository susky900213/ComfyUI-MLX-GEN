from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping


class QwenControlnetWeightDefinition:
    @staticmethod
    def get_controlnet_component() -> ComponentDefinition:
        return ComponentDefinition(
            name="transformer_controlnet",
            hf_subdir="",
            loading_mode="single",
            precision=ModelConfig.precision,
            skip_quantization=True,
            mapping_getter=QwenWeightMapping.get_controlnet_transformer_mapping,
        )
