"""Component definitions for the SwiftVR checkpoint (``H-oliday/SwiftVR``).

The repository is single-file per component and holds no tokenizer, no text encoder, no
VAE and no scheduler - SwiftVR ships a frozen prompt embedding instead of a text tower.
Three components are loaded:

* ``transformer`` - 825 tensors, tensor-identical to stock Wan 2.2 TI2V-5B, mapped with
  Wan's own mapping.
* ``reae`` - 128 tensors, the Restoration-aware Autoencoder.
* ``prompt_embedding`` - the frozen ``prompt_emb`` tensor. Data, not a module: it must
  never be handed to :class:`WeightApplier`'s ``models`` dict.

ReAE is excluded from quantization with ``skip_quantization=True``, which
:meth:`WeightApplier._quantize` honours structurally - the component is never offered to
the predicate at all. It is 164 MB in F32, under 1% of the resident bf16 DiT, so a q8
pass would save almost nothing, while the upstream author identifies the autoencoder's
reconstruction capability as the ceiling on output quality. This mirrors the Wan VAE,
which is excluded the same way.
"""

import math
from pathlib import Path
from typing import Iterator, List

from safetensors import safe_open

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.swiftvr.weights.swiftvr_weight_mapping import TRANSFORMER_TENSOR_COUNT, SwiftVRWeightMapping
from mflux.models.wan.weights.wan_weight_definition import WanWeightDefinition

TRANSFORMER_NUM_LAYERS = 30
PROMPT_EMBEDDING_KEY = "prompt_emb"

# The checkpoint also carries the tokenizer trace of the frozen prompt. Neither tensor is
# read at inference - SwiftVR has no text encoder to feed them to - so they are excluded
# by name here rather than dropped somewhere downstream. Any OTHER unexpected key in that
# file is an error, which is what assert_source_key_coverage enforces.
PROMPT_EMBEDDING_UNUSED_KEYS = ("attention_mask", "input_ids")
PROMPT_EMBEDDING_PREFIX_FILTERS = [PROMPT_EMBEDDING_KEY]

# WanTransformer parameters that are computed by WanRotaryPosEmbed.__init__ rather than
# loaded: the checkpoint holds 825 tensors while the module tree holds 827. A weight
# coverage assertion that compares the two sets directly must exclude these, or it will
# report a false missing-key failure on a perfectly complete checkpoint.
TRANSFORMER_COMPUTED_PARAMETER_KEYS = ("rope.freqs_cos", "rope.freqs_sin")


class SwiftVRWeightDefinition:
    """Weight layout for the single-repository SwiftVR source checkpoint."""

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        """The three components a SwiftVR run loads."""
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                num_layers=TRANSFORMER_NUM_LAYERS,
                loading_mode="multi_glob",
                precision=ModelConfig.precision,
                mapping_getter=lambda: SwiftVRWeightMapping.get_transformer_mapping(num_layers=TRANSFORMER_NUM_LAYERS),
            ),
            ComponentDefinition(
                name="reae",
                hf_subdir=".",
                loading_mode="mlx_native",
                precision=ModelConfig.precision,
                mapping_getter=SwiftVRWeightMapping.get_reae_mapping,
                weight_files=["reae.safetensors"],
                skip_quantization=True,
            ),
            ComponentDefinition(
                name="prompt_embedding",
                hf_subdir=".",
                loading_mode="mlx_native",
                precision=ModelConfig.precision,
                mapping_getter=None,
                weight_files=["prompt_embedding.safetensors"],
                weight_prefix_filters=PROMPT_EMBEDDING_PREFIX_FILTERS,
                skip_quantization=True,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        """None. SwiftVR has no text encoder, so there is nothing to tokenize."""
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        """The four files a SwiftVR run needs, about 18.78 GiB in total."""
        return [
            "transformer/config.json",
            "transformer/*.safetensors",
            "reae.safetensors",
            "prompt_embedding.safetensors",
        ]

    @staticmethod
    def quantization_predicate(path: str, module, bits: int | None = None) -> bool:
        """Wan's q8 sensitivity policy, unchanged.

        SwiftVR's transformer weights are the Wan weights, so the same per-path list
        applies verbatim rather than being restated - restating it would let the two
        drift apart silently. ReAE never reaches this predicate: its component sets
        ``skip_quantization=True``.
        """
        return WanWeightDefinition.quantization_predicate(path, module, bits)

    @staticmethod
    def assert_source_key_coverage(root_path: Path) -> None:
        """Fail closed on any checkpoint tensor the mapping does not account for.

        ``WeightMapper.apply_mapping`` skips unmapped source keys in silence and
        ``Module.update(..., strict=False)`` leaves unmatched parameters at their random
        initialisation, so nothing in the loading stack raises on a checkpoint that has
        drifted. This is the source-side half of that guard; the model-side half is the
        initializer's coverage assertion.

        Reads safetensors headers only - no tensor bytes - so it is cheap enough to run
        before every load, including on the 20 GB transformer shard.

        Args:
            root_path: Directory holding ``transformer/``, ``reae.safetensors`` and
                ``prompt_embedding.safetensors``.

        Raises:
            ValueError: If a checkpoint's key set does not match what the mapping and the
                runtime expect. The offending keys are named.
        """
        reae_keys = SwiftVRWeightDefinition._tensor_names(root_path / "reae.safetensors")
        SwiftVRWeightMapping.assert_reae_source_coverage(reae_keys)
        SwiftVRWeightDefinition._assert_prompt_embedding_keys(root_path / "prompt_embedding.safetensors")
        SwiftVRWeightDefinition._assert_transformer_tensor_count(root_path / "transformer")

    @staticmethod
    def estimate_resident_weight_bytes(root_path: Path) -> int:
        """Bytes the loaded weights will occupy in memory, at each component's precision.

        Not the on-disk size: the transformer ships F32 and is converted to bf16 on load,
        so summing file sizes would reserve roughly twice what the run actually holds and
        shrink the working-set budget for no reason. Counts only the tensors that survive
        a component's prefix filter, and reads headers rather than tensors.
        """
        total = 0
        for component in SwiftVRWeightDefinition.get_components():
            if component.precision is None:
                raise ValueError(
                    f"SwiftVR component '{component.name}' declares no precision, so its resident "
                    "size cannot be derived from the checkpoint header. Set precision on the "
                    "ComponentDefinition or teach this estimator the on-disk dtype."
                )
            item_size = component.precision.size
            for file_path in SwiftVRWeightDefinition._component_files(root_path, component):
                for name, element_count in SwiftVRWeightDefinition._tensor_element_counts(file_path):
                    if not SwiftVRWeightDefinition._is_loaded_key(name, component):
                        continue
                    total += element_count * item_size
        return total

    @staticmethod
    def for_saving(model_config) -> "SwiftVRWeightDefinition":
        """Prepared-layout definition for ``mlxgen prepare``. Not supported yet."""
        raise NotImplementedError(
            "SwiftVR has no prepared (quantized) layout. Landing one needs a prepared "
            "ComponentDefinition set writing transformer/ and reae/ subdirectories, a "
            "resolve() arm that recognizes that layout, and a SwiftVR.save_model. It is "
            "deliberately gated until the bf16 source route has model-backed runtime "
            "evidence to validate a quantized package against (ADR 0001)."
        )

    @staticmethod
    def _assert_prompt_embedding_keys(file_path: Path) -> None:
        observed = set(SwiftVRWeightDefinition._tensor_names(file_path))
        if PROMPT_EMBEDDING_KEY not in observed:
            raise ValueError(
                f"SwiftVR prompt embedding file {file_path} has no '{PROMPT_EMBEDDING_KEY}' tensor. "
                "The model has no text encoder and cannot synthesize one."
            )
        unexpected = sorted(observed - {PROMPT_EMBEDDING_KEY} - set(PROMPT_EMBEDDING_UNUSED_KEYS))
        if unexpected:
            raise ValueError(
                f"SwiftVR prompt embedding file {file_path} carries unexpected tensor(s) "
                f"{unexpected}. Only '{PROMPT_EMBEDDING_KEY}' is consumed and only "
                f"{list(PROMPT_EMBEDDING_UNUSED_KEYS)} are knowingly unused; decide what the new "
                "tensors mean rather than dropping them."
            )

    @staticmethod
    def _assert_transformer_tensor_count(transformer_path: Path) -> None:
        """Guard the DiT's tensor count.

        The 825 keys are mapped by Wan's own mapping, so they are not restated here and
        cannot be named individually. A count change is the cheap signal that the
        fine-tune has diverged from stock Wan 2.2 TI2V-5B; any renaming that preserves
        the count surfaces as a missing parameter in the initializer's coverage check.
        """
        observed = sum(
            len(SwiftVRWeightDefinition._tensor_names(shard))
            for shard in sorted(transformer_path.glob("*.safetensors"))
            if not shard.name.startswith("._")
        )
        if observed != TRANSFORMER_TENSOR_COUNT:
            raise ValueError(
                f"SwiftVR transformer holds {observed} tensors but the Wan 2.2 TI2V-5B mapping "
                f"covers exactly {TRANSFORMER_TENSOR_COUNT}. This checkpoint is no longer "
                "tensor-identical to stock Wan; SwiftVR needs its own transformer mapping before "
                "it can be loaded."
            )

    @staticmethod
    def _component_files(root_path: Path, component: ComponentDefinition) -> list[Path]:
        component_root = root_path / component.hf_subdir
        if component.weight_files:
            return [component_root / name for name in component.weight_files if (component_root / name).exists()]
        if not component_root.exists():
            return []
        return sorted(f for f in component_root.rglob("*.safetensors") if f.is_file() and not f.name.startswith("._"))

    @staticmethod
    def _is_loaded_key(name: str, component: ComponentDefinition) -> bool:
        if component.weight_prefix_filters is None:
            return True
        return any(name.startswith(prefix) for prefix in component.weight_prefix_filters)

    @staticmethod
    def _tensor_names(file_path: Path) -> list[str]:
        with safe_open(str(file_path), framework="numpy") as handle:
            return list(handle.keys())

    @staticmethod
    def _tensor_element_counts(file_path: Path) -> Iterator[tuple[str, int]]:
        with safe_open(str(file_path), framework="numpy") as handle:
            for name in handle.keys():
                yield name, math.prod(handle.get_slice(name).get_shape())
