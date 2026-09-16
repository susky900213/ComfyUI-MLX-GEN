from dataclasses import dataclass, replace
from pathlib import Path

import PIL.Image
import PIL.ImageChops
import PIL.ImageFilter
import PIL.ImageOps
import PIL.ImageStat

from mflux.utils.box_values import AbsoluteBoxValues, BoxValues
from mflux.utils.image_util import ImageUtil

# Edge fill samples a thin strip from the source border and bicubically stretches it
# across the padded extent. The stretch factor is what decides whether the result is a
# plausible continuation or a field of 1-D streaks, so it is the quantity that gets
# bounded - not the strip size.
#
# _EDGE_BASE_STRIP_CAP reproduces the historical strip (source_extent // 8, capped at
# 32 px). _EDGE_MAX_STRETCH is the largest upsample factor the strip is allowed to
# take: every recorded outpaint/reframe validation run sits at or below 10.91x, so 12
# keeps the whole validated envelope bit-identical while capping anything past it.
#
# A bigger strip alone cannot fix deep padding: the artefact is lateral structure
# replicated along the expansion axis, not missing detail, and a 12x smear is still a
# smear. So once a side runs past _edge_reach() px - the depth the base strip covers at
# the bound - the fill also cross-fades toward the neutral border-colour background,
# with an amplitude gated by the overshoot. Inside the reach the gate is zero and the
# canvas is bit-identical to the pre-fix implementation.
_EDGE_BASE_STRIP_CAP = 32
_EDGE_MAX_STRETCH = 12
_EDGE_BASE_BLUR_CAP = 8
_EDGE_BLUR_DIVISOR = 24
_NEUTRAL_BLUR_DIVISOR = 8
_NEUTRAL_BLUR_CAP = 64


@dataclass(frozen=True)
class OutpaintCanvas:
    canvas_path: Path
    source_path: Path
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    paste_left: int
    paste_top: int
    padding: AbsoluteBoxValues
    # A binary mask beside the canvas for routes that lock the source in latent space through a
    # mask input: black over the source (minus a transition band on the sides that gained
    # pixels), white over everything the model has to paint. None on routes that lock some other
    # way, or not at all.
    lock_mask_path: Path | None = None


# Transition band between the locked source and the generated area, in canvas pixels. It is what
# lets the model blend new content into the source across the seam, and it stays absolute rather
# than scaling with the canvas: the latent grid is canvas/16 (FLUX.2) or canvas/8 (Qwen), so 24 px
# is a fixed one-to-three cells at every resolution. Same value as the FLUX.2 route's own band;
# test_outpaint_layer pins the two lock boxes together.
SOURCE_LOCK_TRANSITION_PX = 24


class OutpaintUtil:
    # Public mirror of _EDGE_MAX_STRETCH for callers that select a fill mode and for the
    # capability contract in mflux.task_inference, which cannot import this module (PIL).
    EDGE_FILL_MAX_STRETCH = _EDGE_MAX_STRETCH

    @staticmethod
    def expanded_canvas_size(
        *,
        source_width: int,
        source_height: int,
        padding: AbsoluteBoxValues,
        dimension_multiple: int = 16,
    ) -> tuple[int, int]:
        """The canvas size a padding box produces, after the route's dimension multiple.

        Public because the fill policy has to know the canvas a request would build before any
        canvas exists on disk, and it has to be the same arithmetic `create_expanded_canvas` runs
        rather than a second copy of it.
        """
        if dimension_multiple <= 0:
            raise ValueError("dimension_multiple must be greater than zero.")
        return (
            OutpaintUtil._round_up(source_width + padding.left + padding.right, dimension_multiple),
            OutpaintUtil._round_up(source_height + padding.top + padding.bottom, dimension_multiple),
        )

    @staticmethod
    def create_expanded_canvas(
        *,
        source_path: str | Path,
        padding_value: str,
        output_path: str | Path,
        dimension_multiple: int = 16,
        fill_color: tuple[int, int, int] = (255, 255, 255),
        fill_mode: str = "edge",
        option_name: str = "--outpaint-padding",
    ) -> OutpaintCanvas:
        if dimension_multiple <= 0:
            raise ValueError("dimension_multiple must be greater than zero.")

        source = ImageUtil.load_image(source_path)
        padding = BoxValues.parse(padding_value).normalize_to_dimensions(width=source.width, height=source.height)
        OutpaintUtil._validate_padding(padding, option_name=option_name)

        target_width, target_height = OutpaintUtil.expanded_canvas_size(
            source_width=source.width,
            source_height=source.height,
            padding=padding,
            dimension_multiple=dimension_multiple,
        )
        canvas = OutpaintUtil._create_background(
            source=source,
            target_width=target_width,
            target_height=target_height,
            paste_left=padding.left,
            paste_top=padding.top,
            fill_color=fill_color,
            fill_mode=fill_mode,
        )
        canvas.paste(source, (padding.left, padding.top))

        canvas_path = Path(output_path)
        canvas_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(canvas_path)
        return OutpaintCanvas(
            canvas_path=canvas_path,
            source_path=Path(source_path),
            source_width=source.width,
            source_height=source.height,
            target_width=target_width,
            target_height=target_height,
            paste_left=padding.left,
            paste_top=padding.top,
            padding=padding,
        )

    @staticmethod
    def source_lock_box(
        *, canvas: OutpaintCanvas, transition_px: int = SOURCE_LOCK_TRANSITION_PX
    ) -> tuple[int, int, int, int]:
        """The canvas box a latent lock holds, as (left, top, right, bottom) in canvas pixels.

        The source box, inset by the transition band only on the sides that gained new pixels: a
        side with nothing new on it has no seam and gives up no source. The inset is capped at the
        gap on that side (a side that only gained a round-up sliver never sacrifices more source
        than it added) and at half the source extent, so opposing insets never meet. If the box
        collapses anyway the whole source is held.
        """
        gap_left = max(0, canvas.paste_left)
        gap_top = max(0, canvas.paste_top)
        gap_right = max(0, canvas.target_width - canvas.paste_left - canvas.source_width)
        gap_bottom = max(0, canvas.target_height - canvas.paste_top - canvas.source_height)
        max_inset_x = max(0, canvas.source_width // 2 - 1)
        max_inset_y = max(0, canvas.source_height // 2 - 1)
        left = canvas.paste_left + min(transition_px, gap_left, max_inset_x)
        top = canvas.paste_top + min(transition_px, gap_top, max_inset_y)
        right = canvas.paste_left + canvas.source_width - min(transition_px, gap_right, max_inset_x)
        bottom = canvas.paste_top + canvas.source_height - min(transition_px, gap_bottom, max_inset_y)
        if right <= left or bottom <= top:
            return (
                canvas.paste_left,
                canvas.paste_top,
                canvas.paste_left + canvas.source_width,
                canvas.paste_top + canvas.source_height,
            )
        return left, top, right, bottom

    @staticmethod
    def attach_source_lock_mask(
        *,
        canvas: OutpaintCanvas,
        output_path: str | Path,
        transition_px: int = SOURCE_LOCK_TRANSITION_PX,
    ) -> OutpaintCanvas:
        """Write the source-lock mask for `canvas` and return the canvas pointing at it.

        White (255) is repainted, black (0) is held, matching the mask contract of the masked-edit
        routes that consume it.
        """
        mask = PIL.Image.new("L", (canvas.target_width, canvas.target_height), 255)
        mask.paste(0, OutpaintUtil.source_lock_box(canvas=canvas, transition_px=transition_px))
        mask_path = Path(output_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_path)
        return replace(canvas, lock_mask_path=mask_path)

    @staticmethod
    def composite_source_region(
        *,
        generated_image: PIL.Image.Image,
        canvas: OutpaintCanvas,
        feather_px: int | None = None,
        restore_threshold: float = 12.0,
        preserve_source: bool | None = None,
    ) -> PIL.Image.Image:
        """Blend the untouched source crop back over the generated canvas.

        `preserve_source` states the intent directly:

        * ``True``  - always paste the source back (used with an explicit feather).
        * ``False`` - never paste; kept for callers that want the decoded result untouched.
        * ``None``  - adaptive: paste only when the generated region still matches the
          source within `restore_threshold` mean absolute difference.

        `restore_threshold` stays accepted for the existing call sites. A negative
        value is the legacy spelling of ``preserve_source=False``; it used to work by
        accident, because a mean absolute difference is never below zero and so the
        "diverged too much, skip the paste" guard could never be false.
        """
        if generated_image.size != (canvas.target_width, canvas.target_height):
            raise ValueError(
                "Outpaint output size changed unexpectedly: "
                f"expected {canvas.target_width}x{canvas.target_height}, got "
                f"{generated_image.width}x{generated_image.height}."
            )

        source = ImageUtil.load_image(canvas.source_path)
        if source.size != (canvas.source_width, canvas.source_height):
            raise ValueError("Source image size changed during outpaint generation.")

        mode = OutpaintUtil._resolve_preservation_mode(
            preserve_source=preserve_source,
            feather_px=feather_px,
            restore_threshold=restore_threshold,
        )
        composited = generated_image.convert("RGB").copy()
        restore_difference = OutpaintUtil._source_region_difference(
            generated_image=composited,
            source=source,
            paste_left=canvas.paste_left,
            paste_top=canvas.paste_top,
        )
        applied = mode == "always" or (mode == "adaptive" and restore_difference <= restore_threshold)
        if applied:
            composited.paste(
                source,
                (canvas.paste_left, canvas.paste_top),
                OutpaintUtil._source_mask(source=source, feather_px=feather_px),
            )
        composited.outpaint_preservation_applied = applied
        composited.outpaint_preservation_mode = mode
        composited.outpaint_source_restore_difference = restore_difference
        return composited

    @staticmethod
    def source_region_difference(*, generated_image: PIL.Image.Image, canvas: OutpaintCanvas) -> float:
        """Mean absolute difference (0-255) between the source and its window in `generated_image`.

        The same measurement `composite_source_region` records, exposed for a caller that has to
        take it before a restore changes the window.
        """
        source = ImageUtil.load_image(canvas.source_path)
        return OutpaintUtil._source_region_difference(
            generated_image=generated_image.convert("RGB"),
            source=source,
            paste_left=canvas.paste_left,
            paste_top=canvas.paste_top,
        )

    @staticmethod
    def _resolve_preservation_mode(
        *,
        preserve_source: bool | None,
        feather_px: int | None,
        restore_threshold: float,
    ) -> str:
        if preserve_source is not None:
            return "always" if preserve_source else "never"
        if feather_px is not None:
            return "always"
        if restore_threshold < 0:
            # Legacy sentinel from the strict FLUX.2 outpaint route: "never post-blend".
            return "never"
        return "adaptive"

    @staticmethod
    def attach_metadata(
        *,
        generated_image,
        canvas: OutpaintCanvas,
        padding_value: str,
        preservation: str = "adaptive-content-aware-source-blend",
    ) -> None:
        generated_image.source_image_width = canvas.source_width
        generated_image.source_image_height = canvas.source_height
        extra_metadata = dict(getattr(generated_image, "extra_metadata", None) or {})
        output_image = getattr(generated_image, "image", None)
        extra_metadata.update(
            {
                "outpaint_padding": padding_value,
                "outpaint_source_path": str(canvas.source_path),
                "outpaint_target_width": canvas.target_width,
                "outpaint_target_height": canvas.target_height,
                "outpaint_source_paste_left": canvas.paste_left,
                "outpaint_source_paste_top": canvas.paste_top,
                "outpaint_preservation": preservation,
                "outpaint_source_restore_applied": getattr(
                    output_image,
                    "outpaint_preservation_applied",
                    None,
                ),
                "outpaint_source_restore_difference": getattr(
                    output_image,
                    "outpaint_source_restore_difference",
                    None,
                ),
            }
        )
        generated_image.extra_metadata = extra_metadata

    @staticmethod
    def attach_reframe_metadata(*, generated_image, canvas: OutpaintCanvas, padding_value: str) -> None:
        generated_image.source_image_width = canvas.source_width
        generated_image.source_image_height = canvas.source_height
        extra_metadata = dict(getattr(generated_image, "extra_metadata", None) or {})
        extra_metadata.update(
            {
                "reframe_padding": padding_value,
                "reframe_source_path": str(canvas.source_path),
                "reframe_target_width": canvas.target_width,
                "reframe_target_height": canvas.target_height,
                "reframe_source_paste_left": canvas.paste_left,
                "reframe_source_paste_top": canvas.paste_top,
                "reframe_mode": "expanded-conditioning-canvas",
            }
        )
        generated_image.extra_metadata = extra_metadata

    @staticmethod
    def validate_padding(padding: AbsoluteBoxValues, *, option_name: str = "--outpaint-padding") -> None:
        """Raise ValueError for a negative side or a box that adds no pixels at all.

        Public because the pass planner has to reject a request before it decides how to run
        it, with the same wording the canvas builder uses.
        """
        OutpaintUtil._validate_padding(padding, option_name=option_name)

    @staticmethod
    def _validate_padding(padding: AbsoluteBoxValues, *, option_name: str) -> None:
        parts = (padding.top, padding.right, padding.bottom, padding.left)
        if any(part < 0 for part in parts):
            raise ValueError(f"{option_name} values must be zero or positive.")
        if not any(part > 0 for part in parts):
            raise ValueError(f"{option_name} must add pixels on at least one side.")

    @staticmethod
    def _round_up(value: int, multiple: int) -> int:
        remainder = value % multiple
        if remainder == 0:
            return value
        return value + multiple - remainder

    @staticmethod
    def _create_background(
        *,
        source: PIL.Image.Image,
        target_width: int,
        target_height: int,
        paste_left: int,
        paste_top: int,
        fill_color: tuple[int, int, int],
        fill_mode: str,
    ) -> PIL.Image.Image:
        if fill_mode == "solid":
            return PIL.Image.new("RGB", (target_width, target_height), fill_color)
        if fill_mode == "edge":
            return OutpaintUtil._create_edge_extended_background(
                source=source,
                target_width=target_width,
                target_height=target_height,
                paste_left=paste_left,
                paste_top=paste_top,
                fill_color=fill_color,
            )
        if fill_mode == "neutral":
            return OutpaintUtil._create_neutral_background(
                source=source.convert("RGB"),
                target_width=target_width,
                target_height=target_height,
                paste_left=paste_left,
                paste_top=paste_top,
            )
        if fill_mode != "blur":
            raise ValueError("fill_mode must be 'edge', 'neutral', 'blur', or 'solid'.")

        background = PIL.ImageOps.fit(
            source,
            (target_width, target_height),
            method=PIL.Image.Resampling.BICUBIC,
            centering=(0.5, 0.5),
        )
        blur_radius = max(target_width, target_height) / 24
        return background.filter(PIL.ImageFilter.GaussianBlur(radius=blur_radius)).convert("RGB")

    @staticmethod
    def _create_edge_extended_background(
        *,
        source: PIL.Image.Image,
        target_width: int,
        target_height: int,
        paste_left: int,
        paste_top: int,
        fill_color: tuple[int, int, int],
    ) -> PIL.Image.Image:
        source = source.convert("RGB")
        canvas = PIL.Image.new("RGB", (target_width, target_height), fill_color)
        left = paste_left
        top = paste_top
        right = max(0, target_width - paste_left - source.width)
        bottom = max(0, target_height - paste_top - source.height)
        strip_left, strip_right, strip_top, strip_bottom = OutpaintUtil._edge_strip_extents(
            source_width=source.width,
            source_height=source.height,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )

        if left > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((0, 0, strip_left, source.height)),
                    (left, source.height),
                ),
                (0, top),
            )
        if right > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((source.width - strip_right, 0, source.width, source.height)),
                    (right, source.height),
                ),
                (paste_left + source.width, top),
            )
        if top > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((0, 0, source.width, strip_top)),
                    (source.width, top),
                ),
                (left, 0),
            )
        if bottom > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((0, source.height - strip_bottom, source.width, source.height)),
                    (source.width, bottom),
                ),
                (left, paste_top + source.height),
            )

        OutpaintUtil._paste_corner_extensions(
            canvas=canvas,
            source=source,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            strip_left=strip_left,
            strip_right=strip_right,
            strip_top=strip_top,
            strip_bottom=strip_bottom,
        )
        border = max(left, top, right, bottom)
        if border <= 0:
            return canvas

        canvas = canvas.filter(PIL.ImageFilter.GaussianBlur(radius=OutpaintUtil._edge_blur_radius(border)))
        fade = OutpaintUtil._edge_fade_mask(
            target_width=target_width,
            target_height=target_height,
            source_width=source.width,
            source_height=source.height,
            paste_left=left,
            paste_top=top,
        )
        if fade is None:
            return canvas
        return PIL.Image.composite(
            OutpaintUtil._create_neutral_background(
                source=source,
                target_width=target_width,
                target_height=target_height,
                paste_left=left,
                paste_top=top,
            ),
            canvas,
            fade,
        )

    @staticmethod
    def edge_fill_reach(extent: int) -> int:
        """Padding depth edge fill covers on a source side of `extent` px.

        Public because callers choosing a fill mode need the same bound the fill itself
        uses. Padding at or below the reach is a texture continuation at or under
        `_EDGE_MAX_STRETCH`; past it the fill starts cross-fading to the neutral
        background because a stretched strip stops carrying usable structure.
        """
        return OutpaintUtil._edge_reach(extent)

    @staticmethod
    def _edge_base_strip(extent: int) -> int:
        """The historical strip depth: an eighth of the source side, capped at 32 px."""
        return max(1, min(_EDGE_BASE_STRIP_CAP, extent // 8))

    @staticmethod
    def _edge_reach(extent: int) -> int:
        """Padding depth the base strip covers without exceeding the validated stretch."""
        return OutpaintUtil._edge_base_strip(extent) * _EDGE_MAX_STRETCH

    @staticmethod
    def _edge_strip_extents(
        *,
        source_width: int,
        source_height: int,
        left: int,
        right: int,
        top: int,
        bottom: int,
    ) -> tuple[int, int, int, int]:
        """Per-side strip depths that keep the bicubic stretch at or below the bound.

        The strip only grows once the padded extent passes what the base strip can
        cover; it can never exceed the source side, so very deep padding still runs
        above the bound - that regime is handled by the fade in `_edge_fade_mask`.
        """
        return (
            OutpaintUtil._edge_strip(source_width, left),
            OutpaintUtil._edge_strip(source_width, right),
            OutpaintUtil._edge_strip(source_height, top),
            OutpaintUtil._edge_strip(source_height, bottom),
        )

    @staticmethod
    def _edge_strip(extent: int, pad: int) -> int:
        base = OutpaintUtil._edge_base_strip(extent)
        if pad <= 0:
            return base
        needed = (pad + _EDGE_MAX_STRETCH - 1) // _EDGE_MAX_STRETCH
        return max(1, min(extent, max(base, needed)))

    @staticmethod
    def _edge_blur_radius(border: int) -> int:
        """Seam softener for the pasted strips - deliberately still capped at 8 px.

        This filter runs over the whole background at one radius, so it is driven by
        the deepest side and cannot be aimed at the side that needs it. Lifting the cap
        in step with the padding was measured on the 768x766 / 100%-bottom case: it
        moved the deep band's streak amplitude by 6% (10.47 -> 9.83) while washing out
        the 76 px left band, which is well inside the validated envelope, by 22%
        (41.48 -> 32.43). The scaling that actually removes streaks is spatial, and
        lives in `_edge_fade_mask` plus the neutral background's own radius.
        """
        return min(_EDGE_BASE_BLUR_CAP, max(2, border // _EDGE_BLUR_DIVISOR))

    @staticmethod
    def _edge_fade_mask(
        *,
        target_width: int,
        target_height: int,
        source_width: int,
        source_height: int,
        paste_left: int,
        paste_top: int,
    ) -> PIL.Image.Image | None:
        """Weight of the neutral background over the stretched strip, by distance.

        Per side the fade ramps linearly from 0 at the source seam to
        `min(1, (pad - reach) / reach)` at the outer edge: the amplitude is gated by
        how far the padding overshoots what the strip can honestly cover, and the
        shape keeps the seam itself untouched. A side inside its reach contributes
        nothing, so the whole validated envelope stays bit-identical, and the gate
        grows continuously past it rather than switching on at a threshold.
        """
        right = max(0, target_width - paste_left - source_width)
        bottom = max(0, target_height - paste_top - source_height)
        reach_x = OutpaintUtil._edge_reach(source_width)
        reach_y = OutpaintUtil._edge_reach(source_height)
        if max(paste_left, right) <= reach_x and max(paste_top, bottom) <= reach_y:
            return None

        mask = PIL.Image.new("L", (target_width, target_height), 0)
        nearest = PIL.Image.Resampling.NEAREST
        if paste_left > 0:
            ramp = OutpaintUtil._distance_ramp(paste_left, reach_x, outward=False)
            mask.paste(ramp.resize((paste_left, target_height), resample=nearest), (0, 0))
        if right > 0:
            ramp = OutpaintUtil._distance_ramp(right, reach_x, outward=True)
            mask.paste(ramp.resize((right, target_height), resample=nearest), (paste_left + source_width, 0))
        vertical = PIL.Image.new("L", (target_width, target_height), 0)
        if paste_top > 0:
            ramp = OutpaintUtil._distance_ramp(paste_top, reach_y, outward=False, vertical=True)
            vertical.paste(ramp.resize((target_width, paste_top), resample=nearest), (0, 0))
        if bottom > 0:
            ramp = OutpaintUtil._distance_ramp(bottom, reach_y, outward=True, vertical=True)
            vertical.paste(ramp.resize((target_width, bottom), resample=nearest), (0, paste_top + source_height))
        return PIL.ImageChops.lighter(mask, vertical)

    @staticmethod
    def _distance_ramp(extent: int, reach: int, *, outward: bool, vertical: bool = False) -> PIL.Image.Image:
        gate = min(1.0, max(0.0, (extent - reach) / max(1, reach)))
        values = []
        for index in range(extent):
            distance = index + 1 if outward else extent - index
            weight = gate * distance / extent
            values.append(max(0, min(255, round(255 * weight))))
        ramp = PIL.Image.new("L", (1, extent) if vertical else (extent, 1))
        ramp.putdata(values)
        return ramp

    @staticmethod
    def _border_colors(source: PIL.Image.Image) -> dict[str, tuple[int, int, int]]:
        """Mean RGB of the source strip adjacent to each side."""
        width, height = source.size
        strip_x = OutpaintUtil._edge_base_strip(width)
        strip_y = OutpaintUtil._edge_base_strip(height)

        def mean(box: tuple[int, int, int, int]) -> tuple[int, int, int]:
            channels = PIL.ImageStat.Stat(source.crop(box)).mean
            return tuple(int(round(value)) for value in channels[:3])

        return {
            "left": mean((0, 0, strip_x, height)),
            "right": mean((width - strip_x, 0, width, height)),
            "top": mean((0, 0, width, strip_y)),
            "bottom": mean((0, height - strip_y, width, height)),
        }

    @staticmethod
    def _create_neutral_background(
        *,
        source: PIL.Image.Image,
        target_width: int,
        target_height: int,
        paste_left: int,
        paste_top: int,
    ) -> PIL.Image.Image:
        """Flat per-side border colour, softened so band junctions carry no hard edge."""
        colors = OutpaintUtil._border_colors(source)
        left = paste_left
        top = paste_top
        right = max(0, target_width - paste_left - source.width)
        bottom = max(0, target_height - paste_top - source.height)
        overall = tuple(int(round(value)) for value in PIL.ImageStat.Stat(source).mean[:3])
        canvas = PIL.Image.new("RGB", (target_width, target_height), overall)

        if left > 0:
            canvas.paste(colors["left"], (0, top, left, top + source.height))
        if right > 0:
            canvas.paste(colors["right"], (left + source.width, top, target_width, top + source.height))
        if top > 0:
            canvas.paste(colors["top"], (left, 0, left + source.width, top))
        if bottom > 0:
            canvas.paste(colors["bottom"], (left, top + source.height, left + source.width, target_height))
        for corner_box, sides in (
            ((0, 0, left, top), ("left", "top")),
            ((left + source.width, 0, target_width, top), ("right", "top")),
            ((0, top + source.height, left, target_height), ("left", "bottom")),
            (
                (left + source.width, top + source.height, target_width, target_height),
                ("right", "bottom"),
            ),
        ):
            if corner_box[2] > corner_box[0] and corner_box[3] > corner_box[1]:
                canvas.paste(OutpaintUtil._blend_colors(colors[sides[0]], colors[sides[1]]), corner_box)

        border = max(left, top, right, bottom)
        if border <= 0:
            return canvas
        blur_radius = max(2, min(_NEUTRAL_BLUR_CAP, border // _NEUTRAL_BLUR_DIVISOR))
        return canvas.filter(PIL.ImageFilter.GaussianBlur(radius=blur_radius))

    @staticmethod
    def _blend_colors(first: tuple[int, int, int], second: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(int(round((a + b) / 2)) for a, b in zip(first, second))

    @staticmethod
    def _paste_corner_extensions(
        *,
        canvas: PIL.Image.Image,
        source: PIL.Image.Image,
        left: int,
        top: int,
        right: int,
        bottom: int,
        strip_left: int,
        strip_right: int,
        strip_top: int,
        strip_bottom: int,
    ) -> None:
        if left > 0 and top > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(source.crop((0, 0, strip_left, strip_top)), (left, top)),
                (0, 0),
            )
        if right > 0 and top > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((source.width - strip_right, 0, source.width, strip_top)),
                    (right, top),
                ),
                (left + source.width, 0),
            )
        if left > 0 and bottom > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((0, source.height - strip_bottom, strip_left, source.height)),
                    (left, bottom),
                ),
                (0, top + source.height),
            )
        if right > 0 and bottom > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop(
                        (source.width - strip_right, source.height - strip_bottom, source.width, source.height)
                    ),
                    (right, bottom),
                ),
                (left + source.width, top + source.height),
            )

    @staticmethod
    def _resized_patch(patch: PIL.Image.Image, size: tuple[int, int]) -> PIL.Image.Image:
        return patch.resize(size, resample=PIL.Image.Resampling.BICUBIC)

    @staticmethod
    def _source_mask(source: PIL.Image.Image, feather_px: int | None) -> PIL.Image.Image:
        """Alpha for pasting the source crop back.

        Content-aware by default: the interior is pasted at near-full opacity behind a feathered
        inset, and the source's own edges are pasted at full opacity right up to the border, so
        the seam blends while detail survives. Every outpaint route runs `preserve_source=None`
        (adaptive) and reaches this once the generated window is within the restore threshold.
        """
        if feather_px is None:
            if min(source.width, source.height) < 64:
                return PIL.Image.new("L", source.size, 255)
            return OutpaintUtil._content_aware_source_mask(source=source)
        if feather_px <= 0:
            return PIL.Image.new("L", source.size, 255)

        inset_x = min(feather_px, max(0, source.width // 2 - 1))
        inset_y = min(feather_px, max(0, source.height // 2 - 1))
        mask = PIL.Image.new("L", source.size, 0)
        if inset_x == 0 or inset_y == 0:
            return PIL.Image.new("L", source.size, 255)
        mask.paste(255, (inset_x, inset_y, source.width - inset_x, source.height - inset_y))
        mask = mask.filter(PIL.ImageFilter.GaussianBlur(radius=feather_px / 2))
        exact_inset_x = min(feather_px * 2, max(0, source.width // 2 - 1))
        exact_inset_y = min(feather_px * 2, max(0, source.height // 2 - 1))
        if exact_inset_x > 0 and exact_inset_y > 0:
            mask.paste(
                255,
                (
                    exact_inset_x,
                    exact_inset_y,
                    source.width - exact_inset_x,
                    source.height - exact_inset_y,
                ),
            )
        return PIL.ImageChops.lighter(mask, OutpaintUtil._detail_preservation_mask(source=source))

    @staticmethod
    def _content_aware_source_mask(source: PIL.Image.Image) -> PIL.Image.Image:
        feather_px = min(96, max(32, min(source.width, source.height) // 3))
        inset_x = min(feather_px, max(0, source.width // 2 - 1))
        inset_y = min(feather_px, max(0, source.height // 2 - 1))
        mask = PIL.Image.new("L", source.size, 0)
        if inset_x > 0 and inset_y > 0:
            mask.paste(220, (inset_x, inset_y, source.width - inset_x, source.height - inset_y))
            mask = mask.filter(PIL.ImageFilter.GaussianBlur(radius=feather_px / 2))
        detail_mask = OutpaintUtil._detail_preservation_mask(
            source=source,
            border_fade_px=max(8, feather_px // 3),
        )
        return PIL.ImageChops.lighter(mask, detail_mask)

    @staticmethod
    def _source_region_difference(
        *,
        generated_image: PIL.Image.Image,
        source: PIL.Image.Image,
        paste_left: int,
        paste_top: int,
    ) -> float:
        generated_region = generated_image.crop(
            (
                paste_left,
                paste_top,
                paste_left + source.width,
                paste_top + source.height,
            )
        )
        sample_width = min(96, source.width)
        sample_height = max(1, round(source.height * sample_width / source.width))
        source_sample = source.resize((sample_width, sample_height), resample=PIL.Image.Resampling.BICUBIC)
        generated_sample = generated_region.resize(
            (sample_width, sample_height),
            resample=PIL.Image.Resampling.BICUBIC,
        )
        stat = PIL.ImageStat.Stat(PIL.ImageChops.difference(source_sample, generated_sample))
        return float(sum(stat.mean) / len(stat.mean))

    @staticmethod
    def _detail_preservation_mask(source: PIL.Image.Image, border_fade_px: int = 0) -> PIL.Image.Image:
        edges = PIL.ImageOps.grayscale(source).filter(PIL.ImageFilter.FIND_EDGES)
        edges.paste(0, (0, 0, edges.width, 1))
        edges.paste(0, (0, edges.height - 1, edges.width, edges.height))
        edges.paste(0, (0, 0, 1, edges.height))
        edges.paste(0, (edges.width - 1, 0, edges.width, edges.height))
        edges = edges.filter(PIL.ImageFilter.MaxFilter(size=3)).filter(PIL.ImageFilter.GaussianBlur(radius=1))
        if border_fade_px > 0:
            fade = PIL.Image.new("L", source.size, 0)
            inset_x = min(border_fade_px, max(0, source.width // 2 - 1))
            inset_y = min(border_fade_px, max(0, source.height // 2 - 1))
            if inset_x > 0 and inset_y > 0:
                fade.paste(255, (inset_x, inset_y, source.width - inset_x, source.height - inset_y))
                fade = fade.filter(PIL.ImageFilter.GaussianBlur(radius=border_fade_px / 2))
                edges = PIL.ImageChops.multiply(edges, fade)
        return edges.point(lambda value: 255 if value > 18 else 0)
