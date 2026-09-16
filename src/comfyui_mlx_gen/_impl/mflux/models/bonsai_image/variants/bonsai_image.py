from __future__ import annotations

from mlx import nn

from mflux.models.bonsai_image.bonsai_image_initializer import BonsaiImageInitializer
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein


class BonsaiImage(Flux2Klein):
    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig | None = None,
    ):
        nn.Module.__init__(self)
        BonsaiImageInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config or ModelConfig.bonsai_image_ternary(),
        )

    def _configure_generation_scheduler(self, config: Config) -> None:
        # Prism's Bonsai checkpoints are tuned for FlowMatch Euler with fixed shift 3.0.
        if hasattr(config.scheduler, "set_mu"):
            config.scheduler.set_mu(3.0)

    def save_model(self, base_path: str) -> None:
        raise NotImplementedError(
            "Bonsai checkpoints are already MLX-packed. Use `mlxgen download --model ...` "
            "to cache them locally instead of `mlxgen prepare`."
        )
