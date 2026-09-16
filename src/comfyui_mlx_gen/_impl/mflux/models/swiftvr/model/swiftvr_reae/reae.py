"""SwiftVR's Restoration-aware Autoencoder (ReAE).

ReAE is a 40,946,364-parameter causal video autoencoder that replaces the Wan 3D VAE
(704.69M) while reproducing its latent contract exactly - 48 channels, 16x spatial and
4x temporal compression - which is why the unmodified Wan transformer accepts its
latents.

The published checkpoint stores the encoder and decoder as ``nn.Sequential`` with
positional keys (``encoder.3.weight``, ``decoder.13.conv3d.weight``, ...), so the layer
lists below must keep the non-parametric entries (``ReLU``, ``Upsample``, ``Clamp``) at
their original indices: an index shift would silently misalign every subsequent tensor.
The weight mapping inserts a single ``layers.`` segment and transposes convolution
layouts; nothing else is renamed. Every index here was reconciled against the 128 keys
of ``reae.safetensors``, and :func:`summarize_reae_parameters` re-checks the totals.

This module owns the graph only. Causality lives entirely in
:mod:`mflux.models.swiftvr.streaming.streaming_reae`, which walks these lists and
carries the MemBlock and TPool boundary buffers across chunks. There is deliberately no
``__call__`` here: running the stacks without that state would produce seams at every
chunk boundary with no error raised.

Ported from ``swiftvr/models/reae.py``. Checked end to end against the reference with
the published weights over a FIRST/MIDDLE/LAST sequence (12 + 8 + 5 frames in,
9 + 8 + 8 out): encoder 5.6e-06, decoder 1.0e-05 in float32.
"""

from dataclasses import dataclass

from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.swiftvr.model.swiftvr_reae.reae_blocks import Clamp, MemBlock, TGrow, TPool, reae_conv

ENCODER_HIDDEN_CHANNELS = 64
IMAGE_CHANNELS = 3

# The encoder's compression is structural, not configurable: three stride-2 convolutions
# at indices 3, 8 and 13, and TPool(2) at indices 2 and 7 (index 12 is TPool(1), a
# channel mix with no temporal effect). The decoder's upscale flags must reproduce them
# or the round trip silently changes geometry - see ReAE._validate_upscale_flags.
ENCODER_SPATIAL_STRIDES = 3
ENCODER_TEMPORAL_STRIDES = 2

# Published checkpoint totals, used by summarize_reae_parameters as a topology self-check.
REAE_TENSOR_COUNT = 128
REAE_PARAMETER_COUNT = 40_946_364


class ReAEEncoder(nn.Module):
    """Pixel-to-latent stack. Parameter paths are ``encoder.layers.{i}...``.

    Spatial compression is ``patch_size`` (pixel unshuffle, applied by the streaming
    codec) times three stride-2 convolutions at indices 3, 8 and 13, i.e. 16x at the
    default ``patch_size=2``. Temporal compression is ``TPool(2)`` at index 2 and
    ``TPool(2)`` at index 7, i.e. 4x; the ``TPool(1)`` at index 12 is a channel mix with
    no temporal effect, and omitting it would be a silent quality regression.
    """

    def __init__(
        self,
        image_channels: int = IMAGE_CHANNELS,
        patch_size: int = 2,
        hidden_channels: int = ENCODER_HIDDEN_CHANNELS,
        latent_channels: int = 48,
    ) -> None:
        super().__init__()
        width = hidden_channels
        self.layers: list[nn.Module] = [
            reae_conv(image_channels * patch_size**2, width),  # 0
            nn.ReLU(),  # 1
            TPool(width, 2),  # 2
            reae_conv(width, width, stride=2, bias=False),  # 3
            MemBlock(width, width),  # 4
            MemBlock(width, width),  # 5
            MemBlock(width, width),  # 6
            TPool(width, 2),  # 7
            reae_conv(width, width, stride=2, bias=False),  # 8
            MemBlock(width, width),  # 9
            MemBlock(width, width),  # 10
            MemBlock(width, width),  # 11
            TPool(width, 1),  # 12
            reae_conv(width, width, stride=2, bias=False),  # 13
            MemBlock(width, width),  # 14
            MemBlock(width, width),  # 15
            MemBlock(width, width),  # 16
            reae_conv(width, latent_channels),  # 17
        ]


class ReAEDecoder(nn.Module):
    """Latent-to-pixel stack. Parameter paths are ``decoder.layers.{i}...``.

    Mirrors the encoder: three nearest ``Upsample(2)`` stages at indices 6, 12 and 18
    followed by a pixel shuffle applied by the streaming codec, and ``TGrow`` at indices
    7 (stride 1), 13 and 19 (stride 2 each) for 4x temporal growth. The output is
    clamped to ``[0, 1]`` before the pixel shuffle by the codec, not here.
    """

    def __init__(
        self,
        latent_channels: int = 48,
        width_mult: int = 2,
        image_channels: int = IMAGE_CHANNELS,
        patch_size: int = 2,
        decoder_time_upscale: tuple[bool, bool] = (True, True),
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
    ) -> None:
        super().__init__()
        widths = [256 * width_mult, 128 * width_mult, 64 * width_mult, ENCODER_HIDDEN_CHANNELS]
        self.layers: list[nn.Module] = [
            Clamp(),  # 0
            reae_conv(latent_channels, widths[0]),  # 1
            nn.ReLU(),  # 2
            MemBlock(widths[0], widths[0]),  # 3
            MemBlock(widths[0], widths[0]),  # 4
            MemBlock(widths[0], widths[0]),  # 5
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1, mode="nearest"),  # 6
            TGrow(widths[0], 1),  # 7
            reae_conv(widths[0], widths[1], bias=False),  # 8
            MemBlock(widths[1], widths[1]),  # 9
            MemBlock(widths[1], widths[1]),  # 10
            MemBlock(widths[1], widths[1]),  # 11
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1, mode="nearest"),  # 12
            TGrow(widths[1], 2 if decoder_time_upscale[0] else 1),  # 13
            reae_conv(widths[1], widths[2], bias=False),  # 14
            MemBlock(widths[2], widths[2]),  # 15
            MemBlock(widths[2], widths[2]),  # 16
            MemBlock(widths[2], widths[2]),  # 17
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1, mode="nearest"),  # 18
            TGrow(widths[2], 2 if decoder_time_upscale[1] else 1),  # 19
            reae_conv(widths[2], widths[3], bias=False),  # 20
            nn.ReLU(),  # 21
            reae_conv(widths[3], image_channels * patch_size**2),  # 22
        ]


class ReAE(nn.Module):
    """The Restoration-aware Autoencoder graph.

    Attributes:
        patch_size: Pixel (un)shuffle ratio applied outside the layer stacks.
        latent_channels: Latent channel count, 48, matching the Wan TI2V contract.
        spatial_scale: Pixels per latent along H and W.
        temporal_scale: Source frames per latent frame.
        frames_to_trim: Decoder head frames produced by causal padding, dropped once per
            clip. ``2 ** sum(decoder_time_upscale) - 1``.
    """

    def __init__(
        self,
        patch_size: int = 2,
        latent_channels: int = 48,
        width_mult: int = 2,
        decoder_time_upscale: tuple[bool, bool] = (True, True),
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
        image_channels: int = IMAGE_CHANNELS,
    ) -> None:
        super().__init__()
        decoder_time_upscale = tuple(bool(flag) for flag in decoder_time_upscale)
        decoder_space_upscale = tuple(bool(flag) for flag in decoder_space_upscale)
        self._validate_upscale_flags(decoder_time_upscale, decoder_space_upscale)
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.width_mult = width_mult
        self.image_channels = image_channels
        self.decoder_time_upscale = decoder_time_upscale
        self.decoder_space_upscale = decoder_space_upscale
        self.spatial_scale = patch_size * 2 ** sum(decoder_space_upscale)
        self.temporal_scale = 2 ** sum(decoder_time_upscale)
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1
        self.encoder = ReAEEncoder(
            image_channels=image_channels,
            patch_size=patch_size,
            latent_channels=latent_channels,
        )
        self.decoder = ReAEDecoder(
            latent_channels=latent_channels,
            width_mult=width_mult,
            image_channels=image_channels,
            patch_size=patch_size,
            decoder_time_upscale=decoder_time_upscale,
            decoder_space_upscale=decoder_space_upscale,
        )

    @staticmethod
    def _validate_upscale_flags(
        decoder_time_upscale: tuple[bool, ...],
        decoder_space_upscale: tuple[bool, ...],
    ) -> None:
        """Reject decoder geometry that the fixed encoder cannot invert.

        The reference carries these flags without checking them, so a disabled stage
        yields an autoencoder whose decode does not undo its encode - the round trip
        changes the frame count or the resolution and nothing raises. There is one
        checkpoint and it is symmetric, so anything else is a configuration error.

        Raises:
            ValueError: If the enabled stage counts do not match the encoder's fixed
                three spatial and two temporal reductions.
        """
        if sum(decoder_space_upscale) != ENCODER_SPATIAL_STRIDES:
            raise ValueError(
                f"decoder_space_upscale={decoder_space_upscale} enables {sum(decoder_space_upscale)} "
                f"spatial stages but the ReAE encoder always reduces {ENCODER_SPATIAL_STRIDES} times; "
                "the decoder would not invert the encoder."
            )
        if sum(decoder_time_upscale) != ENCODER_TEMPORAL_STRIDES:
            raise ValueError(
                f"decoder_time_upscale={decoder_time_upscale} enables {sum(decoder_time_upscale)} "
                f"temporal stages but the ReAE encoder always pools {ENCODER_TEMPORAL_STRIDES} times; "
                "the decoder would not invert the encoder."
            )


@dataclass(frozen=True)
class ReAEParameterSummary:
    """Parameter footprint of a :class:`ReAE`, as reported by :func:`summarize_reae_parameters`."""

    tensor_count: int
    parameter_count: int
    encoder_parameter_count: int
    decoder_parameter_count: int
    float32_bytes: int
    matches_published_checkpoint: bool


def summarize_reae_parameters(model: ReAE | None = None) -> ReAEParameterSummary:
    """Build (or inspect) a ReAE and report what its topology weighs.

    This is the topology self-check. ``reae.safetensors`` holds exactly
    :data:`REAE_TENSOR_COUNT` tensors totalling :data:`REAE_PARAMETER_COUNT` parameters,
    so a summary with ``matches_published_checkpoint`` False means the layer lists
    disagree with the checkpoint and every weight target derived from them is suspect -
    the counts are the cheapest signal that an index or a channel width drifted.

    It reads shape metadata only, so it neither loads weights nor forces an evaluation.

    Args:
        model: An existing ReAE. A default one is constructed when omitted.

    Returns:
        The tensor and parameter counts, the encoder/decoder split, the float32 weight
        size in bytes (the safetensors header is not included), and whether the totals
        match the published checkpoint.
    """
    model = ReAE() if model is None else model
    tensors = tree_flatten(model.parameters())
    parameter_count = sum(array.size for _, array in tensors)
    encoder_count = sum(array.size for _, array in tree_flatten(model.encoder.parameters()))
    decoder_count = sum(array.size for _, array in tree_flatten(model.decoder.parameters()))
    return ReAEParameterSummary(
        tensor_count=len(tensors),
        parameter_count=parameter_count,
        encoder_parameter_count=encoder_count,
        decoder_parameter_count=decoder_count,
        float32_bytes=parameter_count * 4,
        matches_published_checkpoint=(len(tensors) == REAE_TENSOR_COUNT and parameter_count == REAE_PARAMETER_COUNT),
    )
