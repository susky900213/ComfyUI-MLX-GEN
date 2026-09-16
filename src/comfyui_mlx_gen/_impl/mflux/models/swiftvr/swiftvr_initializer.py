"""Construction and weight loading for the SwiftVR restoration model.

Mirrors :class:`SeedVR2Initializer`, including its ``_assert_weight_coverage`` discipline,
which is the only thing standing between a typo and a plausible-looking wrong result:
``WeightMapper.apply_mapping`` silently skips any source key it cannot map, and
``Module.update(..., strict=False)`` silently leaves unmatched parameters at their random
initialisation. Neither raises on its own.

SwiftVR deliberately does NOT route through :class:`WanInitializer`. It shares the Wan
transformer weights, not the Wan runtime: there is no 3D VAE, no text encoder, no
sampler and no guidance, and Wan's ``_transformer_kwargs`` filters overrides against a
closed allow-list that would silently discard every SwiftVR-specific key.
"""

from dataclasses import dataclass
from pathlib import Path

from mlx.utils import tree_flatten

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.swiftvr.model.swiftvr_reae.reae import ReAE
from mflux.models.swiftvr.model.swiftvr_transformer.swiftvr_transformer import SwiftVRTransformer
from mflux.models.swiftvr.model.swiftvr_transformer.window_meta import DEFAULT_WINDOW_HW
from mflux.models.swiftvr.streaming.chunk import LATENT_TEMPORAL_DOWNSCALE
from mflux.models.swiftvr.weights.swiftvr_weight_definition import (
    PROMPT_EMBEDDING_KEY,
    TRANSFORMER_COMPUTED_PARAMETER_KEYS,
    SwiftVRWeightDefinition,
)

# Keys of transformer_overrides that WanTransformer accepts as constructor arguments.
TRANSFORMER_CONSTRUCTOR_KEYS = frozenset(
    {
        "patch_size",
        "num_attention_heads",
        "attention_head_dim",
        "in_channels",
        "out_channels",
        "text_dim",
        "freq_dim",
        "ffn_dim",
        "num_layers",
        "cross_attn_norm",
        "eps",
        "added_kv_proj_dim",
        "rope_max_seq_len",
    }
)

# Keys consumed elsewhere in the SwiftVR runtime, CLI or download path. Listed explicitly
# so an override that belongs to neither set raises instead of vanishing (ADR 0002). Each
# entry names its consumer, because an allow-list nobody reads back is how a dead setting
# gets certified as live:
#   swiftvr_window_size, swiftvr_shift_alternate_layers -> window_settings()
#   reae_config                                         -> reae_kwargs()
#   default_clip_len, default_dit_overlap,
#   swiftvr_inference_timestep                          -> runtime_settings()
#   task, supports_video_to_video, supports_image_to_video -> task inference (mlx_gen)
#   expected_download_bytes, download_headroom_bytes    -> mlxgen download free-space preflight
RUNTIME_OVERRIDE_KEYS = frozenset(
    {
        "swiftvr_window_size",
        "swiftvr_shift_alternate_layers",
        "swiftvr_inference_timestep",
        "reae_config",
        "default_clip_len",
        "default_dit_overlap",
        "task",
        "supports_video_to_video",
        "supports_image_to_video",
        "expected_download_bytes",
        "download_headroom_bytes",
    }
)


@dataclass(frozen=True)
class SwiftVRRuntimeSettings:
    """Run defaults the catalog owns, resolved once per model.

    Every field is read from ``transformer_overrides`` and is required there. A default
    that lives in code as well as in the catalog is a setting with two sources that can
    disagree, and the catalog is the auditable one.

    Attributes:
        clip_len: MIDDLE chunk size in source frames.
        dit_overlap: Latent frames of the previous chunk prepended to each chunk.
        inference_timestep: The constant one-step conditioning timestep.
    """

    clip_len: int
    dit_overlap: int
    inference_timestep: float


class SwiftVRInitializer:
    """Builds the SwiftVR component graph and applies the checkpoint to it."""

    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None = None,
        model_path: str | None = None,
    ) -> None:
        """Populate ``model`` with configuration, submodules and weights.

        Args:
            model: The :class:`SwiftVR` instance being constructed.
            model_config: Catalog entry driving the build.
            quantize: Requested quantization level, or ``None`` for the source dtype.
            model_path: Local checkpoint directory, or ``None`` to resolve the repository.
        """
        path = model_path if model_path else model_config.model_name
        root_path = SwiftVRInitializer._resolve_weight_root(path)
        SwiftVRInitializer._init_config(model, model_config)
        model.swiftvr_resident_weight_bytes = SwiftVRWeightDefinition.estimate_resident_weight_bytes(root_path)

        # Source-side coverage first: WeightMapper.apply_mapping silently drops any
        # checkpoint key it cannot map, so a renamed or extra tensor is invisible to the
        # model-side assertion below, which only ever sees the resulting hole.
        SwiftVRWeightDefinition.assert_source_key_coverage(root_path)

        weights = SwiftVRInitializer._load_weights(root_path)
        SwiftVRInitializer._init_models(model, model_config)
        SwiftVRInitializer._init_prompt_embedding(model, weights, model_config)
        SwiftVRInitializer._assert_weight_coverage(model, weights)
        SwiftVRInitializer._apply_weights(model, weights, quantize)

        # MFSWA is installed last, on the same module instances the weights were applied
        # to. Running the stock global attention would be a different model that raises
        # nothing, so predict_velocity refuses to run until this has happened.
        model.transformer.install_mfswa()

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        """Attach configuration and the callback registry to ``model``."""
        model.model_config = model_config
        model.runtime_settings = SwiftVRInitializer.runtime_settings(model_config)
        model.callbacks = CallbackRegistry()
        model.prompt_embeds = None
        model.swiftvr_resident_weight_bytes = 0

    @staticmethod
    def _resolve_weight_root(model_path: str) -> Path:
        """Resolve the local directory holding the four SwiftVR files.

        Raises:
            ValueError: If no path could be resolved.
        """
        root_path = PathResolution.resolve(
            model_path,
            patterns=SwiftVRWeightDefinition.get_download_patterns(),
        )
        if root_path is None:
            raise ValueError(
                f"SwiftVR requires a resolved model path for '{model_path}'. "
                "Download it with: mlxgen download --model H-oliday/SwiftVR"
            )
        return root_path

    @staticmethod
    def _load_weights(root_path: Path) -> LoadedWeights:
        """Load the transformer, ReAE and prompt-embedding components."""
        return WeightLoader.load(
            weight_definition=SwiftVRWeightDefinition,
            model_path=str(root_path),
        )

    @staticmethod
    def transformer_kwargs(model_config: ModelConfig) -> dict:
        """Project ``transformer_overrides`` onto :class:`WanTransformer` constructor kwargs.

        Every override key must be either a constructor argument or a documented runtime
        key. An unclassified key raises rather than being dropped, so a future catalog
        edit cannot silently lose a setting.

        Raises:
            ValueError: If any override key is unclassified.
        """
        overrides = dict(model_config.transformer_overrides or {})
        unclassified = sorted(set(overrides) - TRANSFORMER_CONSTRUCTOR_KEYS - RUNTIME_OVERRIDE_KEYS)
        if unclassified:
            raise ValueError(
                f"SwiftVR transformer_overrides carries unclassified keys {unclassified}. "
                "Add each key to TRANSFORMER_CONSTRUCTOR_KEYS if WanTransformer accepts it, "
                "or to RUNTIME_OVERRIDE_KEYS if the SwiftVR runtime consumes it."
            )
        kwargs = {key: value for key, value in overrides.items() if key in TRANSFORMER_CONSTRUCTOR_KEYS}
        if "patch_size" in kwargs:
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        return kwargs

    @staticmethod
    def window_settings(model_config: ModelConfig) -> tuple[tuple[int, int], bool]:
        """MFSWA window size and shift policy from the catalog entry."""
        overrides = model_config.transformer_overrides or {}
        window = overrides.get("swiftvr_window_size", list(DEFAULT_WINDOW_HW))
        if len(window) != 2:
            raise ValueError(f"swiftvr_window_size must hold two values, got {window}.")
        shift = bool(overrides.get("swiftvr_shift_alternate_layers", True))
        return (int(window[0]), int(window[1])), shift

    @staticmethod
    def runtime_settings(model_config: ModelConfig) -> SwiftVRRuntimeSettings:
        """Read the run defaults the catalog owns.

        Every key is required. Falling back to a code default would recreate the second
        source of truth this resolver exists to remove, and would let a catalog edit that
        renames a key take effect as silence (ADR 0002).

        Raises:
            ValueError: If a key is absent, or holds a value the route cannot honour.
        """
        overrides = model_config.transformer_overrides or {}
        missing = sorted(
            key
            for key in ("default_clip_len", "default_dit_overlap", "swiftvr_inference_timestep")
            if key not in overrides
        )
        if missing:
            raise ValueError(
                f"SwiftVR catalog entry '{model_config.model_name}' is missing transformer_overrides "
                f"{missing}. These are the route's run defaults and have no code fallback; add them to "
                "the entry rather than letting the runtime invent a value."
            )
        clip_len = int(overrides["default_clip_len"])
        dit_overlap = int(overrides["default_dit_overlap"])
        if clip_len <= 0 or clip_len % LATENT_TEMPORAL_DOWNSCALE:
            raise ValueError(
                f"SwiftVR default_clip_len must be a positive multiple of {LATENT_TEMPORAL_DOWNSCALE}, got {clip_len}."
            )
        if dit_overlap < 0:
            raise ValueError(f"SwiftVR default_dit_overlap must be zero or positive, got {dit_overlap}.")
        return SwiftVRRuntimeSettings(
            clip_len=clip_len,
            dit_overlap=dit_overlap,
            inference_timestep=float(overrides["swiftvr_inference_timestep"]),
        )

    @staticmethod
    def reae_kwargs(model_config: ModelConfig) -> dict:
        """ReAE constructor kwargs from the catalog entry, with tuple-typed flag lists."""
        config = dict((model_config.transformer_overrides or {}).get("reae_config", {}))
        for key in ("decoder_time_upscale", "decoder_space_upscale"):
            if key in config:
                config[key] = tuple(bool(flag) for flag in config[key])
        return config

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        """Build the ReAE and the MFSWA-capable transformer. Weights are applied later."""
        window_hw, shift_alternate_layers = SwiftVRInitializer.window_settings(model_config)
        model.reae = ReAE(**SwiftVRInitializer.reae_kwargs(model_config))
        model.transformer = SwiftVRTransformer(
            window_hw=window_hw,
            shift_alternate_layers=shift_alternate_layers,
            **SwiftVRInitializer.transformer_kwargs(model_config),
        )

    @staticmethod
    def _init_prompt_embedding(model, weights: LoadedWeights, model_config: ModelConfig) -> None:
        """Attach the frozen prompt embedding that replaces SwiftVR's text encoder.

        Raises:
            ValueError: If the tensor is absent, not rank 2 or 3, or does not match the
                transformer's text dimension.
        """
        component = weights.components.get("prompt_embedding")
        embedding = component.get(PROMPT_EMBEDDING_KEY) if isinstance(component, dict) else None
        if embedding is None:
            raise ValueError(
                "SwiftVR checkpoint is missing the frozen prompt embedding "
                f"'{PROMPT_EMBEDDING_KEY}' in prompt_embedding.safetensors; the model has no "
                "text encoder and cannot synthesize one."
            )
        if embedding.ndim not in (2, 3):
            raise ValueError(f"SwiftVR prompt embedding must be rank 2 or 3, got shape {embedding.shape}.")
        text_dim = int((model_config.transformer_overrides or {}).get("text_dim", 4096))
        if embedding.shape[-1] != text_dim:
            raise ValueError(
                f"SwiftVR prompt embedding has {embedding.shape[-1]} features but the "
                f"transformer expects text_dim={text_dim}."
            )
        model.prompt_embeds = embedding if embedding.ndim == 3 else embedding[None]

    @staticmethod
    def _assert_weight_coverage(model, weights: LoadedWeights) -> None:
        """Fail closed on any missing or unexpected weight key.

        Raises:
            ValueError: If a component's parameter set and the provided keys differ.
        """
        stored_quantization = weights.meta_data.quantization_level
        components = (
            # WanRotaryPosEmbed builds freqs_cos/freqs_sin in its constructor, so the
            # transformer holds 827 parameters against the checkpoint's 825. They are
            # excluded by name rather than by a substring test, so a genuinely missing
            # weight still raises.
            ("transformer", model.transformer.transformer, set(TRANSFORMER_COMPUTED_PARAMETER_KEYS)),
            ("reae", model.reae, set()),
        )
        for component_name, component_model, computed_keys in components:
            component_weights = weights.components.get(component_name)
            if component_weights is None:
                raise ValueError(f"SwiftVR checkpoint is missing the '{component_name}' component.")
            expected = {key for key, _ in tree_flatten(component_model.parameters())} - computed_keys
            provided = {
                key
                for key, _ in tree_flatten(component_weights)
                if not (stored_quantization is not None and (key.endswith(".scales") or key.endswith(".biases")))
            }
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            if missing or extra:
                raise ValueError(
                    f"SwiftVR {component_name} weight coverage mismatch: "
                    f"missing={len(missing)} ({', '.join(missing[:5]) if missing else 'none'}); "
                    f"extra={len(extra)} ({', '.join(extra[:5]) if extra else 'none'})"
                )

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        """Apply and optionally quantize the component weights.

        ``prompt_embedding`` is data, not a module, and is deliberately absent from the
        ``models`` mapping.
        """
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=SwiftVRWeightDefinition,
            models={
                "transformer": model.transformer.transformer,
                "reae": model.reae,
            },
        )
