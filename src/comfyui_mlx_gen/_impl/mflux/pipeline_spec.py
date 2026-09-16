"""
Lazy pipeline configuration and deferred assembly helpers.

This module deliberately separates two different things:

1. Early graph construction: nodes can create immutable ``PipelineConfig`` /
   ``ComponentConfig`` objects without instantiating modules or loading weights.
2. Materialization: only ``LazyPipeline.materialize()`` /
   ``build_pipeline_from_spec()`` may build a runnable pipeline.

The safe default is pipeline-level lazy loading: early nodes can only pass
configuration through to the existing ``load_generation_model`` path. True
component-wise loading is possible only when all component sources are present
and the runtime route is known to be supported by this helper.
"""

from __future__ import annotations

import gc
import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable

from mflux.python_runtime import (
    LoadedGenerationModel,
    GenerationRuntimePlan,
    load_generation_model,
    resolve_generation_runtime,
)


ConfigMapping = Mapping[str, Any]

# Stable per-process cache. Values are LoadedGenerationModel instances, while
# keys are stable hashes of immutable pipeline configs. This intentionally
# caches the assembled runtime wrapper, not individual components.
_DEFAULT_PIPELINE_CACHE: dict[str, Any] = {}
_DEFAULT_PIPELINE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ComponentConfig:
    """Configuration for one pipeline component.

    This is not a materialized component and does not load any weight file. It
    is a data-only specification that can be passed through ComfyUI-style
    graph nodes before materialization.
    """

    name: str
    kind: str
    source: str | Path | None = None
    weight_paths: tuple[str, ...] = ()
    config_overrides: ConfigMapping = field(default_factory=dict)
    loading_options: ConfigMapping = field(default_factory=dict)

    def is_complete(self) -> bool:
        return self.source is not None or bool(self.weight_paths)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "source": str(self.source) if self.source is not None else None,
            "weight_paths": list(self.weight_paths),
            "config_overrides": dict(self.config_overrides),
            "loading_options": dict(self.loading_options),
        }

    @property
    def cache_key(self) -> str:
        return _stable_hash(self.as_dict())


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a complete deferred pipeline.

    Construction of this object must never load weights or call model
    constructors. ``LazyPipeline`` stores this config and only materializes the
    pipeline at generation time.
    """

    model: str | None = "flux2-klein-4b"
    model_path: str | Path | None = None
    base_model: str | None = None
    family: str | None = None
    task: str = "auto"
    image_count: int = 0
    video_count: int = 0
    reference_image_count: int = 0
    i2i_mode: str | None = "auto"
    has_image_strength: bool = False
    has_video_strength: bool = False
    has_video_mask: bool = False
    has_mask: bool = False
    has_control_image: bool = False
    has_outpaint: bool = False
    has_reframe: bool = False
    has_lora: bool = False
    quantize: int | None = None
    lora_paths: tuple[str, ...] = ()
    lora_scales: tuple[float, ...] = ()
    lora_target_roles: tuple[str, ...] = ()
    model_kwargs: ConfigMapping = field(default_factory=dict)
    component_configs: tuple[ComponentConfig, ...] = ()
    componentwise: bool = False

    def component_config(self, name: str) -> ComponentConfig | None:
        for component in self.component_configs:
            if component.name == name:
                return component
        return None

    def with_component_config(self, component: ComponentConfig) -> "PipelineConfig":
        """Return a copy with one component config replaced or appended."""

        remaining = tuple(c for c in self.component_configs if c.name != component.name)
        return replace(self, component_configs=(component, *remaining))

    def componentwise_is_complete(self) -> bool:
        for name in ("transformer", "vae", "text_encoder"):
            component = self.component_config(name)
            if component is None or not component.is_complete():
                return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "base_model": self.base_model,
            "family": self.family,
            "task": self.task,
            "image_count": self.image_count,
            "video_count": self.video_count,
            "reference_image_count": self.reference_image_count,
            "i2i_mode": self.i2i_mode,
            "has_image_strength": self.has_image_strength,
            "has_video_strength": self.has_video_strength,
            "has_video_mask": self.has_video_mask,
            "has_mask": self.has_mask,
            "has_control_image": self.has_control_image,
            "has_outpaint": self.has_outpaint,
            "has_reframe": self.has_reframe,
            "has_lora": self.has_lora,
            "quantize": self.quantize,
            "lora_paths": list(self.lora_paths),
            "lora_scales": list(self.lora_scales),
            "lora_target_roles": list(self.lora_target_roles),
            "model_kwargs": dict(self.model_kwargs),
            "component_configs": [c.as_dict() for c in self.component_configs],
            "componentwise": self.componentwise,
        }

    @property
    def cache_key(self) -> str:
        return f"{self.model or 'custom'}::{_stable_hash(self.as_dict())}"


class LazyPipeline:
    """A pipeline configuration that is materialized only on first use.

    ``is_materialized`` remains false until ``materialize()`` is called. When
    several ``LazyPipeline`` objects share the same ``cache_key``, they resolve
    to the same cached loaded pipeline.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        builder: Callable[[PipelineConfig], Any] | None = None,
        cache: dict[str, Any] | None = None,
        cache_lock: threading.Lock | None = None,
    ) -> None:
        self.config = config
        self._builder = builder
        self._cache = _DEFAULT_PIPELINE_CACHE if cache is None else cache
        self._cache_lock = _DEFAULT_PIPELINE_LOCK if cache_lock is None else cache_lock
        self._materialized: Any | None = None
        self._cache_key = config.cache_key

    @property
    def cache_key(self) -> str:
        return self._cache_key

    @property
    def is_materialized(self) -> bool:
        return self._materialized is not None

    @property
    def materialized(self) -> Any | None:
        return self._materialized

    def materialize(self) -> Any:
        return build_pipeline_from_spec(
            self.config,
            builder=self._builder,
            cache=self._cache,
            cache_lock=self._cache_lock,
            cache_key=self._cache_key,
        )

    def generate_output(self, **kwargs: Any) -> Any:
        """Materialize if needed, then run the existing whole-pipeline API."""

        return self.materialize().generate_output(**kwargs)

    def generate_outputs(self, **kwargs: Any) -> Any:
        """Materialize if needed, then run the existing multi-output API."""

        return self.materialize().generate_outputs(**kwargs)


def build_pipeline_from_spec(
    config: PipelineConfig,
    *,
    builder: Callable[[PipelineConfig], Any] | None = None,
    cache: dict[str, Any] | None = None,
    cache_lock: threading.Lock | None = None,
    cache_key: str | None = None,
) -> Any:
    """Return the cached/deferred pipeline for ``config``.

    With no ``builder``, the default materializer uses the existing
    ``load_generation_model`` path. A caller can inject ``builder`` for tests
    or an adapter that assembles components without loading weights.
    """

    key = cache_key or config.cache_key
    active_cache = _DEFAULT_PIPELINE_CACHE if cache is None else cache
    active_lock = _DEFAULT_PIPELINE_LOCK if cache_lock is None else cache_lock

    with active_lock:
        cached = active_cache.get(key)
        if cached is not None:
            return cached

        pipeline = builder(config) if builder is not None else _default_build_pipeline(config)
        active_cache[key] = pipeline
        return pipeline


def lazy_pipeline(
    config: PipelineConfig,
    *,
    builder: Callable[[PipelineConfig], Any] | None = None,
    cache: dict[str, Any] | None = None,
    cache_lock: threading.Lock | None = None,
) -> LazyPipeline:
    """Create a lazy wrapper around ``config`` without materializing anything."""

    return LazyPipeline(config, builder=builder, cache=cache, cache_lock=cache_lock)


def _default_build_pipeline(config: PipelineConfig) -> Any:
    """Materialize the supplied config without performing work earlier."""

    if config.componentwise:
        return _build_componentwise_pipeline(config)

    return load_generation_model(
        model=config.model,
        model_config=None,
        family=config.family,
        base_model=config.base_model,
        image_count=config.image_count,
        video_count=config.video_count,
        reference_image_count=config.reference_image_count,
        task=config.task,
        i2i_mode=config.i2i_mode,
        has_image_strength=config.has_image_strength,
        has_video_strength=config.has_video_strength,
        has_video_mask=config.has_video_mask,
        has_mask=config.has_mask,
        has_control_image=config.has_control_image,
        has_outpaint=config.has_outpaint,
        has_reframe=config.has_reframe,
        has_lora=config.has_lora,
        quantize=config.quantize,
        model_path=str(config.model_path) if config.model_path is not None else None,
        lora_paths=list(config.lora_paths),
        lora_scales=list(config.lora_scales),
        lora_target_roles=list(config.lora_target_roles) if config.lora_target_roles else None,
        model_kwargs=dict(config.model_kwargs),
    )

def _build_componentwise_pipeline(config: PipelineConfig) -> Any:
    """Assemble a runnable pipeline from component specs.

    This is deliberately not a set of independent component nodes: the result is
    still one runnable pipeline object. Components are loaded and applied one at
    a time, then the pipeline is assembled from the runtime plan.
    """

    if not config.componentwise_is_complete():
        raise ValueError(
            "componentwise=True requires complete configs for 'transformer', "
            "'vae' and 'text_encoder' with sources or weight paths"
        )

    runtime = resolve_generation_runtime(
        model=config.model,
        model_config=None,
        family=config.family,
        base_model=config.base_model,
        image_count=config.image_count,
        video_count=config.video_count,
        reference_image_count=config.reference_image_count,
        task=config.task,
        i2i_mode=config.i2i_mode,
        has_image_strength=config.has_image_strength,
        has_video_strength=config.has_video_strength,
        has_video_mask=config.has_video_mask,
        has_mask=config.has_mask,
        has_control_image=config.has_control_image,
        has_outpaint=config.has_outpaint,
        has_reframe=config.has_reframe,
        has_lora=config.has_lora,
    )

    model_class = _model_class_for_runtime(runtime)
    if model_class is None:
        raise ValueError(
            "componentwise assembly is not implemented for this runtime; "
            "keep component nodes as configuration/display nodes and use "
            "pipeline-level lazy loading"
        )

    return _build_flux2_componentwise_pipeline(
        config=config,
        runtime=runtime,
        model_class=model_class,
    )


def _model_class_for_runtime(runtime: GenerationRuntimePlan) -> type[Any] | None:
    runtime_id = runtime.runtime_id

    # Deferred imports keep weight/model construction strictly inside
    # materialization. Only routes whose component architecture and weight
    # mapping are directly verifiable are enabled here.
    if runtime_id == "flux2.klein":
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        return Flux2Klein
    if runtime_id == "flux2.klein-edit":
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

        return Flux2KleinEdit
    if runtime_id == "flux2.klein-inpaint":
        from mflux.models.flux2.variants.edit.flux2_klein_inpaint import Flux2KleinInpaint

        return Flux2KleinInpaint
    if runtime_id == "flux2.klein-outpaint":
        from mflux.models.flux2.variants.edit.flux2_klein_outpaint import Flux2KleinOutpaint

        return Flux2KleinOutpaint
    return None


def _build_flux2_componentwise_pipeline(
    *,
    config: PipelineConfig,
    runtime: GenerationRuntimePlan,
    model_class: type[Any],
) -> LoadedGenerationModel:
    """Build one Flux2 Klein pipeline with per-component load/apply/clear."""

    import mlx.core as mx
    from mlx import nn

    from mflux.callbacks.callback_registry import CallbackRegistry
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.flux2.flux2_initializer import Flux2Initializer
    from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
    from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
    from mflux.utils.compiled_predict_cache import CompiledPredictCache

    model_config = runtime.model_config
    if not _is_flux2_klein_config(model_config):
        raise ValueError("componentwise loading currently supports only Flux2 Klein routes")

    # Instantiate component structures first. No weight tree has been loaded
    # into these modules yet.
    transformer_spec = config.component_config("transformer")
    vae_spec = config.component_config("vae")
    text_encoder_spec = config.component_config("text_encoder")
    assert transformer_spec is not None
    assert vae_spec is not None
    assert text_encoder_spec is not None

    transformer = Flux2Transformer(
        **{**model_config.transformer_overrides, **transformer_spec.config_overrides}
    )
    vae = Flux2VAE()
    text_encoder = Qwen3TextEncoder(
        **{**model_config.text_encoder_overrides, **text_encoder_spec.config_overrides}
    )
    components = {
        "transformer": transformer,
        "vae": vae,
        "text_encoder": text_encoder,
    }

    # Load tokenizers before materializing weights. If no root model path is
    # supplied, component configs must explicitly carry their own sources.
    tokenizer_source = str(config.model_path or config.model or "")
    if not tokenizer_source:
        raise ValueError("componentwise assembly needs model_path or model for tokenizers")

    tokenizers = TokenizerLoader.load_all(
        definitions=Flux2KleinWeightDefinition.get_tokenizers(),
        model_path=tokenizer_source,
    )

    # Materialize and apply each component's weights separately.
    weight_definition = Flux2KleinWeightDefinition
    quantization_level: int | None = None
    for component_def in weight_definition.get_components():
        component_spec = config.component_config(component_def.name)
        if component_spec is None or not component_spec.is_complete():
            raise ValueError(f"missing complete component config for {component_def.name}")

        component_path = _resolve_local_component_path(component_def, component_spec)
        patched_component = replace(component_def, hf_subdir="")
        weights, stored_bits, version = WeightLoader._load_component(component_path, patched_component, {})
        loaded = LoadedWeights(
            components={component_def.name: weights},
            meta_data=MetaData(quantization_level=stored_bits, mflux_version=version),
        )
        applied_bits = WeightApplier.apply_and_quantize_single(
            weights=loaded,
            model=components[component_def.name],
            component=patched_component,
            quantize_arg=config.quantize,
            quantization_predicate=weight_definition.quantization_predicate,
        )
        if applied_bits is not None:
            if quantization_level is None:
                quantization_level = applied_bits
            elif quantization_level != applied_bits:
                raise ValueError(
                    f"conflicting component quantization: {component_def.name} "
                    f"requires q{applied_bits}, pipeline already resolved to q{quantization_level}"
                )

        # Delete the weight tree for this component before the next component.
        del weights
        del loaded
        del patched_component
        gc.collect()
        mx.clear_cache()

    # Create an empty pipeline object without running the constructor (which
    # would immediately load the whole checkpoint), initialize MLX module state,
    # and bind components to it so generation methods work unchanged.
    pipeline = model_class.__new__(model_class)
    nn.Module.__init__(pipeline)

    Flux2Initializer._init_config(pipeline, model_config)
    pipeline.vae = components["vae"]
    pipeline.transformer = components["transformer"]
    pipeline.text_encoder = components["text_encoder"]
    pipeline.tokenizers = tokenizers
    pipeline.bits = quantization_level if quantization_level is not None else config.quantize

    # LoRA is still a load-time config passed into the existing initializer path;
    # it is not a separately runnable component in ComfyUI.
    Flux2Initializer._apply_lora(
        pipeline,
        list(config.lora_paths) or None,
        list(config.lora_scales) or None,
    )

    return LoadedGenerationModel(
        plan=runtime.plan,
        model_config=model_config,
        runtime_id=runtime.runtime_id,
        cache_key_base=runtime.cache_key_base,
        cache_key=config.cache_key,
        model=pipeline,
    )


# CHUNK6
