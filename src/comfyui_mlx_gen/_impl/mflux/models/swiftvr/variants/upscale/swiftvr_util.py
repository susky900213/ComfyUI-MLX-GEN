"""Geometry and bounds for the SwiftVR restoration route.

Everything here is pure arithmetic over integers: no MLX arrays, no I/O, no model. The
numbers come from the fixed compression contract of the checkpoint - ReAE folds 16 pixels
into one latent in each spatial axis and 4 source frames into one latent frame, and the
transformer's patch embed consumes latents two at a time spatially - so a padded canvas
must be a multiple of 32 and a clip must be ``4a + 1`` frames long.

There is deliberately no predictive memory estimator. SeedVR2's
``estimate_video_restore_working_set_bytes`` solves a different model with a different
attention working set, and calling it with SwiftVR geometry would produce a confident
wrong number. Instead this module carries a pixel-area cap taken from the largest canvas
the port was analysed against, and the runtime checks real peak memory after each chunk.
"""

from mflux.utils.scale_factor import ScaleFactor

# ReAE compresses 16 pixels per latent in each spatial axis; the transformer's patch
# embed then consumes two latents per token, so the padded canvas must clear both.
LATENT_SPATIAL_DOWNSCALE = 16
SPATIAL_PAD_MULTIPLE = 32
LATENT_TEMPORAL_DOWNSCALE = 4

# Largest padded canvas this port has been analysed against, in pixels: 1920 x 1088, the
# 32-aligned form of 1080p. This is a provisional guard rail, not a measured ceiling -
# above it the run is not known to fit, so it fails closed and names the override.
MAX_ANALYSED_CANVAS_PIXELS = 1920 * 1088


class SwiftVRUtil:
    """Canvas, clip and frame-count arithmetic for the SwiftVR route."""

    @staticmethod
    def output_canvas(
        *,
        source_width: int,
        source_height: int,
        resolution: int | ScaleFactor,
    ) -> tuple[int, int]:
        """Output height and width for a restore request.

        SwiftVR restores at the source size, and 1x is the only operating point this port
        has evidence for. That is a real gap, not a technical limit: upstream's documented
        entry point is ``restore_video(..., upscale=4)`` and its reader bilinearly resamples
        every clip onto the output canvas before encoding (``swiftvr/io.py``,
        ``preprocess_clip_uint8``), so the published quality numbers describe 4x, not 1x.
        Writing the resize is a few lines - ``nn.Upsample`` is already used inside ReAE -
        but a scaled route would be a second operating point with no reference comparison
        behind it, which ADR 0001 does not accept. Landing it needs a measured
        bilinear-parity check against ``F.interpolate(..., align_corners=False)`` and a
        restore compared with the reference at 4x, in that order.

        Returns:
            ``(height, width)`` of the restored output.

        Raises:
            ValueError: If ``resolution`` asks for anything other than the source size.
        """
        min_side = min(source_width, source_height)
        if isinstance(resolution, ScaleFactor):
            # Compare the factor itself, not get_scaled_value(): that helper snaps its
            # result down to a multiple of 16, so a 1x request against a 1080-tall source
            # reports 1072 and would look like a rescale.
            requested = f"{resolution.value}x"
            is_source_size = float(resolution.value) == 1.0
        else:
            requested = f"{int(resolution)} on its short side"
            is_source_size = int(resolution) == min_side
        if not is_source_size:
            raise ValueError(
                f"SwiftVR restores at the source resolution; {source_width}x{source_height} was asked for "
                f"{requested}. SwiftVR reaches other sizes by pre-upsampling the degraded input with "
                "bilinear interpolation, which MLX-Gen has not matched or measured, so scaling is not "
                "offered rather than being approximated. Use --resolution 1x for SwiftVR, or "
                "--model seedvr2-3b to upscale."
            )
        return int(source_height), int(source_width)

    @staticmethod
    def padded_canvas(height: int, width: int) -> tuple[int, int]:
        """Canvas rounded up to :data:`SPATIAL_PAD_MULTIPLE`, padded bottom and right."""
        pad = SPATIAL_PAD_MULTIPLE
        return ((height + pad - 1) // pad) * pad, ((width + pad - 1) // pad) * pad

    @staticmethod
    def max_supported_source_frames(rope_max_seq_len: int) -> int:
        """Longest ``4a + 1`` clip whose rotary offsets stay inside the built table.

        The transformer's temporal patch size is 1 and ReAE emits one latent frame per
        four source frames, so ``rope_max_seq_len`` latent positions cover
        ``4 * rope_max_seq_len - 3`` source frames.
        """
        return LATENT_TEMPORAL_DOWNSCALE * int(rope_max_seq_len) - (LATENT_TEMPORAL_DOWNSCALE - 1)

    @staticmethod
    def canvas_bound_error(padded_height: int, padded_width: int) -> str | None:
        """Message for a canvas beyond the analysed envelope, or ``None`` when inside it."""
        pixels = padded_height * padded_width
        if pixels <= MAX_ANALYSED_CANVAS_PIXELS:
            return None
        return (
            f"SwiftVR restore at a padded canvas of {padded_width}x{padded_height} "
            f"({pixels / 1e6:.1f} MP) is beyond the 1920x1088 envelope this route has been "
            "analysed against, and MLX-Gen has no measurement showing it fits. Restore a smaller "
            "source, or pass --force-unsafe-video-memory to run it anyway and accept a possible "
            "out-of-memory abort."
        )
