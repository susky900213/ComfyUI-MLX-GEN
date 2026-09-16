"""Weight mapping for the SwiftVR checkpoint (``H-oliday/SwiftVR``).

The transformer is NOT mapped here. All 825 tensors of
``transformer/diffusion_pytorch_model.safetensors`` match stock
``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` in name, shape and dtype - SwiftVR is a fine-tune -
so :meth:`WanWeightMapping.get_transformer_mapping` maps it verbatim and
:meth:`SwiftVRWeightMapping.get_transformer_mapping` is a one-line delegation kept only
so the weight definition reads consistently.

The genuinely new mapping is ReAE. Its checkpoint uses ``nn.Sequential`` positional keys
(``encoder.4.conv.0.weight``) while the MLX modules hold their layers in a ``layers``
list, so every target is the source key with one ``layers.`` segment inserted:

    ``{stack}.{i}.{rest}``  ->  ``{stack}.layers.{i}.{rest}``

plus a layout transpose for convolutions - rank 4 ``(0, 2, 3, 1)`` for Conv2d, rank 5
``(0, 2, 3, 4, 1)`` for the two TGrow Conv3d tensors. Biases pass through untouched. The
targets are generated from explicit index constants rather than hand-written, so the
128 entries cannot drift from the module tree by a typo.

No MemBlock in this checkpoint has ``n_in != n_out``, so no ``skip.*`` tensor exists. A
checkpoint that supplied one would be dropped in silence by
:meth:`WeightMapper.apply_mapping`, which is why :meth:`assert_reae_source_coverage`
exists and must run before the weights are trusted (ADR 0002).
"""

from typing import Iterable

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms
from mflux.models.wan.weights.wan_weight_mapping import WanWeightMapping

# nn.Sequential positions carrying parameters, by kind. Positions absent from every
# tuple are ReLU / Upsample / Clamp and carry no tensors.
ENCODER_CONV2D_INDICES: tuple[int, ...] = (0, 3, 8, 13, 17)
ENCODER_CONV2D_BIASED: tuple[int, ...] = (0, 17)
ENCODER_TPOOL_INDICES: tuple[int, ...] = (2, 7, 12)
ENCODER_MEMBLOCK_INDICES: tuple[int, ...] = (4, 5, 6, 9, 10, 11, 14, 15, 16)

DECODER_CONV2D_INDICES: tuple[int, ...] = (1, 8, 14, 20, 22)
DECODER_CONV2D_BIASED: tuple[int, ...] = (1, 22)
DECODER_MEMBLOCK_INDICES: tuple[int, ...] = (3, 4, 5, 9, 10, 11, 15, 16, 17)
DECODER_TGROW_PROJ_INDICES: tuple[int, ...] = (7,)
DECODER_TGROW_CONV3D_INDICES: tuple[int, ...] = (13, 19)

# Conv positions inside a MemBlock's nn.Sequential.
MEMBLOCK_CONV_INDICES: tuple[int, ...] = (0, 2, 4)

REAE_TENSOR_COUNT = 128
TRANSFORMER_TENSOR_COUNT = 825


class SwiftVRWeightMapping(WeightMapping):
    """Weight targets for SwiftVR's transformer and Restoration-aware Autoencoder."""

    @staticmethod
    def get_mapping() -> list[WeightTarget]:
        """Protocol entry point. SwiftVR always maps per component, never as one blob."""
        raise NotImplementedError(
            "SwiftVR has no single-component mapping; use get_transformer_mapping() or get_reae_mapping()."
        )

    @staticmethod
    def get_transformer_mapping(num_layers: int = 30) -> list[WeightTarget]:
        """Stock Wan transformer mapping - SwiftVR's DiT is tensor-identical to Wan 2.2 TI2V-5B."""
        return WanWeightMapping.get_transformer_mapping(num_layers=num_layers)

    @staticmethod
    def get_reae_mapping() -> list[WeightTarget]:
        """The 128 ReAE targets, generated from the index constants above."""
        mapping: list[WeightTarget] = []
        mapping.extend(SwiftVRWeightMapping._encoder_mapping())
        mapping.extend(SwiftVRWeightMapping._decoder_mapping())
        if len(mapping) != REAE_TENSOR_COUNT:
            raise ValueError(
                f"ReAE mapping generated {len(mapping)} targets but the checkpoint holds "
                f"{REAE_TENSOR_COUNT}; the index constants and the module tree disagree."
            )
        return mapping

    @staticmethod
    def assert_reae_source_coverage(source_keys: Iterable[str]) -> None:
        """Fail closed unless the ReAE checkpoint keys and the mapping sources coincide.

        ``WeightMapper.apply_mapping`` drops any source key it cannot place without a
        word, so a checkpoint carrying an unexpected tensor - a MemBlock ``skip.weight``,
        an EMA copy - would load cleanly and restore video with a silently different
        autoencoder. This is the only check that sees the source side.

        Args:
            source_keys: Every tensor name in ``reae.safetensors``.

        Raises:
            ValueError: If any checkpoint key is unmapped, or any mapping source is
                absent from the checkpoint. Both sides are named.
        """
        observed = set(source_keys)
        expected = {target.from_pattern[0] for target in SwiftVRWeightMapping.get_reae_mapping()}
        unmapped = sorted(observed - expected)
        absent = sorted(expected - observed)
        if unmapped or absent:
            raise ValueError(
                "ReAE checkpoint does not match the SwiftVR mapping: "
                f"{len(unmapped)} unmapped checkpoint key(s) "
                f"({', '.join(unmapped[:5]) if unmapped else 'none'}); "
                f"{len(absent)} mapped key(s) absent from the checkpoint "
                f"({', '.join(absent[:5]) if absent else 'none'}). "
                "Update the index constants in swiftvr_weight_mapping.py to match the checkpoint."
            )

    @staticmethod
    def _encoder_mapping() -> list[WeightTarget]:
        mapping: list[WeightTarget] = []
        for index in ENCODER_CONV2D_INDICES:
            mapping.extend(SwiftVRWeightMapping._conv2d("encoder", f"{index}", bias=index in ENCODER_CONV2D_BIASED))
        for index in ENCODER_TPOOL_INDICES:
            mapping.extend(SwiftVRWeightMapping._conv2d("encoder", f"{index}.conv", bias=False))
        for index in ENCODER_MEMBLOCK_INDICES:
            mapping.extend(SwiftVRWeightMapping._memblock("encoder", index))
        return mapping

    @staticmethod
    def _decoder_mapping() -> list[WeightTarget]:
        mapping: list[WeightTarget] = []
        for index in DECODER_CONV2D_INDICES:
            mapping.extend(SwiftVRWeightMapping._conv2d("decoder", f"{index}", bias=index in DECODER_CONV2D_BIASED))
        for index in DECODER_MEMBLOCK_INDICES:
            mapping.extend(SwiftVRWeightMapping._memblock("decoder", index))
        for index in DECODER_TGROW_PROJ_INDICES:
            mapping.extend(SwiftVRWeightMapping._conv2d("decoder", f"{index}.proj", bias=False))
        mapping.extend(
            WeightTarget(
                to_pattern=f"decoder.layers.{index}.conv3d.weight",
                from_pattern=[f"decoder.{index}.conv3d.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            )
            for index in DECODER_TGROW_CONV3D_INDICES
        )
        return mapping

    @staticmethod
    def _memblock(stack: str, index: int) -> list[WeightTarget]:
        mapping: list[WeightTarget] = []
        for conv_index in MEMBLOCK_CONV_INDICES:
            mapping.extend(SwiftVRWeightMapping._conv2d(stack, f"{index}.conv.{conv_index}", bias=True))
        return mapping

    @staticmethod
    def _conv2d(stack: str, suffix: str, *, bias: bool) -> list[WeightTarget]:
        """Targets for one Conv2d, inserting the ``layers.`` segment and transposing OIHW to OHWI."""
        mapping = [
            WeightTarget(
                to_pattern=f"{stack}.layers.{suffix}.weight",
                from_pattern=[f"{stack}.{suffix}.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            )
        ]
        if bias:
            mapping.append(
                WeightTarget(
                    to_pattern=f"{stack}.layers.{suffix}.bias",
                    from_pattern=[f"{stack}.{suffix}.bias"],
                )
            )
        return mapping
