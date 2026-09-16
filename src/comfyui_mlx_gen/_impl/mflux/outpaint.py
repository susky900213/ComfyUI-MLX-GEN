"""Model-agnostic outpaint and reframe orchestration.

One implementation of the conditioning-canvas contract, shared by every backend CLI and by
embedding Python applications. Nothing here imports argparse: the CLI adapter that turns a
parsed namespace into an `OutpaintRequest` lives in `mflux.cli.outpaint_cli`.

Route behaviour is declared, never sniffed. Everything a host can observe is read off the
route's published `GenerationCapability` row (fill modes, default mode, whether the fill
option exists at all, the auto stretch bound, the recommended adapter, the preservation
strategy). Only the mechanics the capability schema deliberately does not publish - which
keyword the backend's `generate_image` accepts the canvas through, the base canvas colour
under edge/blur, and the adapter markers that switch `auto` onto a solid canvas - live in
`_OUTPAINT_RUNTIME`, one entry per capability id, next to the code that consumes them. That
table is the only place a sixth model family has to be named.

Scope: this module covers outpaint *by expanded conditioning canvas* - the source is pasted
into a larger canvas, the model denoises the whole canvas, and the route's preservation
strategy decides what happens to the source region afterwards. A route that outpaints by
native masked fill instead (a real mask channel, no expanded canvas) does not fit this
contract and must not be given an `_OUTPAINT_RUNTIME` entry to force it: `outpaint_contract`
raising for an unlisted capability is the intended outcome there, not a gap to paper over.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import PIL.Image
import PIL.ImageStat

from mflux.task_inference import (
    OUTPAINT_FILL_AUTO,
    OUTPAINT_PASS_MODES,
    OUTPAINT_PASSES_AUTO,
    OUTPAINT_TWO_DEEP_AXIS_RATIO,
    GenerationCapability,
    get_model_capabilities,
    resolve_generation_plan,
)
from mflux.utils.box_values import AbsoluteBoxValues, BoxValues
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE
from mflux.utils.image_util import ImageUtil
from mflux.utils.outpaint_util import OutpaintCanvas, OutpaintUtil

# How a backend's generate_image() accepts the expanded canvas. Declared per capability id
# rather than inferred, so adding a family is a table row and not a branch in the policy.
CONDITIONING_CANVAS_OBJECT = "canvas-object"
CONDITIONING_IMAGE_PATHS = "conditioning-image-paths"

# Green-border outpaint adapters are trained to paint into a pure-green canvas, so that exact
# colour is part of their contract and must not be replaced by the generic neutral fill.
FLUX2_GREEN_BORDER_OUTPAINT_LORA_MARKERS = (
    "fal/flux-2-klein-4b-outpaint-lora",
    "ming3d/flux-2-klein-4b-outpaint-lora",
    "flux-outpaint-lora.safetensors",
)
FLUX2_GREEN_BORDER_FILL_COLOR = (0, 255, 0)
FLUX2_GREEN_BORDER_FILL_REASON = (
    "a green-border outpaint LoRA is loaded and that adapter is trained to paint into a pure-green canvas."
)
# Base canvas colour under the edge/blur backgrounds. Only visible where the edge extension does
# not reach, which is nowhere for a rectangular padding box; kept at the historical value so
# `edge` reproduces pre-0.30 conditioning canvases byte for byte.
EDGE_FILL_BASE_COLOR = (255, 255, 255)
# Default colour for an explicit `solid` with no colour, when the source is too small to sample
# a border ring. Mid-gray is the safe generic blank.
OUTPAINT_NEUTRAL_FALLBACK_COLOR = (128, 128, 128)
# The two-deep-axis regime. A request that pads a vertical side *and* a horizontal side past
# OUTPAINT_TWO_DEEP_AXIS_RATIO (declared in mflux.task_inference, published per route as
# `outpaint_auto_split_corner_ratio`) opens a free corner - a block of new canvas that shares
# neither a row nor a column with the source. That corner, not the fill and not the canvas size, is
# what produces subject duplication: every reference token the model can attend to sits diagonally
# away from it, so nothing local anchors the corner and the prompt plus the model's prior fill it
# with a second copy of the subject. Two single-axis passes reach the same canvas with no free
# corner at either pass, which is what `--outpaint-passes auto` does past the ratio. The constant
# is re-exported from here for callers that only know the policy module.
# Restore threshold shared by every outpaint route. The adaptive paste refuses a window the model
# has recomposed, which would ghost; a window held by a latent lock differs from the source only by
# a VAE round trip and the regenerated transition band. Measured across the recorded envelope, the
# worst two-axis geometry and its split passes: locked windows 1.4-15.6 mean-abs (FLUX.2 base 4B
# single pass at the top), recomposed windows 62-68 (the Qwen route before it had a lock). 24 is
# 1.5x the largest locked figure and well under half a recomposition.
OUTPAINT_RESTORE_THRESHOLD = 24.0


class OutpaintError(ValueError):
    """Raised when an outpaint request cannot be satisfied by the selected route.

    Subclasses ValueError so the existing backend CLIs keep converting it to
    `parser.error(...)` with no change to their except clauses.
    """


@dataclass(frozen=True)
class _OutpaintRuntime:
    """Route mechanics the capability schema deliberately does not publish."""

    conditioning: str
    preserve_source: bool | None
    restore_threshold: float
    # Whether the conditioning canvas reaches the padded area at all. On the FLUX.2 route the
    # trajectory starts from pure noise and the reference tokens for cells holding nothing but
    # fill are dropped, so the fill conditions only the seam ring around the source; the Qwen
    # route keeps every canvas token, so its fill is what the model sees in the new area.
    fill_reaches_padding: bool = True
    base_fill_color: tuple[int, int, int] = EDGE_FILL_BASE_COLOR
    neutral_fallback_color: tuple[int, int, int] = OUTPAINT_NEUTRAL_FALLBACK_COLOR
    adapter_fill_color: tuple[int, int, int] | None = None
    adapter_markers: tuple[str, ...] = ()
    adapter_fill_reason: str = ""
    # Whether the route locks the source region in latent space through its mask input. The
    # shared layer then writes a source-lock mask beside every canvas and passes it as
    # `mask_path`. A route that locks the source on its own (the FLUX.2 outpaint variant) leaves
    # this False.
    source_lock_mask: bool = False


_OUTPAINT_RUNTIME: dict[str, _OutpaintRuntime] = {
    "flux2.outpaint": _OutpaintRuntime(
        conditioning=CONDITIONING_CANVAS_OBJECT,
        # The route locks the source in latent space behind a transition band, and the shared
        # layer then pastes the original crop back over the decoded result while the generated
        # window still matches it. The lock cannot keep the source pixel-exact on its own -
        # FLUX.2's VAE decoder couples the whole canvas, so a hard lock still measured 4-16
        # mean-abs of drift, roughly doubled on a split run - and the paste is what returns the
        # original pixels. It used to be withheld here (`preserve_source=False`) on the reading
        # that a post-blend would fight the lock; measured, the two agree: the lock is what makes
        # the paste safe, because the window it holds cannot have been recomposed.
        preserve_source=None,
        restore_threshold=OUTPAINT_RESTORE_THRESHOLD,
        fill_reaches_padding=False,
        adapter_fill_color=FLUX2_GREEN_BORDER_FILL_COLOR,
        adapter_markers=FLUX2_GREEN_BORDER_OUTPAINT_LORA_MARKERS,
        adapter_fill_reason=FLUX2_GREEN_BORDER_FILL_REASON,
    ),
    "qwen.outpaint": _OutpaintRuntime(
        conditioning=CONDITIONING_IMAGE_PATHS,
        # Expanded-canvas generation plus adaptive source restoration: paste the source back
        # only while the generated source window still matches it. The source region is also
        # held in latent space through the edit route's mask input while the padded area is
        # denoised: without it the model is free to recompose the whole canvas, and a prompt
        # asking for "a wider shot" made it shrink and redraw the subject (measured drift 62-68
        # on the 432x240 starship grown 256 right and 148 down, so the restore never applied;
        # 5.9 and restored with the lock, one subject, environment extended). The threshold used
        # to be 12.0, which sat inside the locked-window distribution and skipped a restore by 0.09.
        preserve_source=None,
        restore_threshold=OUTPAINT_RESTORE_THRESHOLD,
        source_lock_mask=True,
    ),
}


@dataclass(frozen=True)
class OutpaintContract:
    """Everything the shared layer needs to run outpaint on one route.

    Published fields are copied off the route's `GenerationCapability` row so a host reading
    `mlxgen capabilities --json` and the shared layer can never disagree; unpublished
    mechanics come from `_OUTPAINT_RUNTIME`.
    """

    capability_id: str
    fill_modes: tuple[str, ...]
    default_fill_mode: str
    supports_fill_option: bool
    auto_edge_fill_max_stretch: float | None
    recommended_lora: str | None
    preservation: str
    # Largest canvas the route's recorded validation runs cover, or None when the row publishes no
    # envelope. Read off the capability row so the guard and `mlxgen capabilities --json` state the
    # same number.
    validated_max_canvas_pixels: int | None
    # Depth past which `--outpaint-passes auto` splits the request into two single-axis passes, or
    # None on a route that never splits. Also what gates the duplication warning: a route that has
    # not been measured to duplicate does not get told that it does.
    auto_split_corner_ratio: float | None
    dimension_multiple: int
    conditioning: str
    preserve_source: bool | None
    restore_threshold: float
    source_lock_mask: bool
    fill_reaches_padding: bool
    base_fill_color: tuple[int, int, int]
    neutral_fallback_color: tuple[int, int, int]
    adapter_fill_color: tuple[int, int, int] | None
    adapter_markers: tuple[str, ...]
    adapter_fill_reason: str

    @property
    def has_auto_policy(self) -> bool:
        """Whether `fill="auto"` selects a mode at run time on this route."""
        return self.supports_fill_option and self.auto_edge_fill_max_stretch is not None


def outpaint_contract(*, capability: GenerationCapability) -> OutpaintContract:
    """Build the run-time outpaint contract for one published capability row.

    Raises OutpaintError when the row does not support outpaint, or when it does but the
    shared layer has no runtime entry for it - that pairing is a packaging bug, not a user
    error, and failing here is what stops a new family from silently getting FLUX.2's
    latent-lock semantics.
    """
    if not capability.supports_outpaint:
        raise OutpaintError(f"Capability {capability.id!r} does not support outpaint.")
    runtime = _OUTPAINT_RUNTIME.get(capability.id)
    if runtime is None:
        raise OutpaintError(
            f"Capability {capability.id!r} advertises outpaint but mflux.outpaint has no runtime "
            "contract for it. Add an _OUTPAINT_RUNTIME entry for the route."
        )
    if capability.outpaint_default_fill_mode is None or capability.outpaint_preservation is None:
        raise OutpaintError(f"Capability {capability.id!r} publishes an incomplete outpaint fill contract.")
    return OutpaintContract(
        capability_id=capability.id,
        fill_modes=capability.outpaint_fill_modes,
        default_fill_mode=capability.outpaint_default_fill_mode,
        supports_fill_option=capability.supports_outpaint_fill,
        auto_edge_fill_max_stretch=capability.outpaint_auto_edge_fill_max_stretch,
        recommended_lora=capability.outpaint_recommended_lora,
        preservation=capability.outpaint_preservation,
        validated_max_canvas_pixels=capability.outpaint_validated_max_canvas_pixels,
        auto_split_corner_ratio=capability.outpaint_auto_split_corner_ratio,
        dimension_multiple=capability.dimension_multiple or 16,
        conditioning=runtime.conditioning,
        preserve_source=runtime.preserve_source,
        restore_threshold=runtime.restore_threshold,
        source_lock_mask=runtime.source_lock_mask,
        fill_reaches_padding=runtime.fill_reaches_padding,
        base_fill_color=runtime.base_fill_color,
        neutral_fallback_color=runtime.neutral_fallback_color,
        adapter_fill_color=runtime.adapter_fill_color,
        adapter_markers=runtime.adapter_markers,
        adapter_fill_reason=runtime.adapter_fill_reason,
    )


def outpaint_contract_for_model(
    *,
    model: str | None = None,
    model_config: Any = None,
    base_model: str | None = None,
) -> OutpaintContract:
    """Resolve the outpaint route for a model name or ModelConfig and return its contract.

    Loads no weights. Prefer `model_config=...` when the caller already resolved one: a local
    `--model-path` folder has no routable model name. Raises TaskInferenceError when the model
    has no outpaint route at all.
    """
    plan = resolve_generation_plan(
        model=model, model_config=model_config, base_model=base_model, image_count=1, has_outpaint=True
    )
    capabilities = get_model_capabilities(model=model, model_config=model_config, base_model=base_model)
    for capability in capabilities.capabilities:
        if capability.id == plan.capability_id:
            return outpaint_contract(capability=capability)
    raise OutpaintError(f"No capability row {plan.capability_id!r} for model {model or model_config!r}.")


@dataclass(frozen=True)
class OutpaintRequest:
    """A plain-value outpaint request. No argparse, no model, no image loading.

    `fill` is None on routes without a fill option, or one of the route's
    `outpaint_fill_modes`. `fill_color_explicit` is separate from `fill_color` on purpose:
    the CLI only warns that a colour is ignored when the user typed it, not when it was
    replayed out of prior metadata.
    """

    padding: str
    fill: str | None = None
    fill_color: tuple[int, int, int] | None = None
    fill_color_explicit: bool = False
    lora_paths: tuple[str, ...] = ()
    requested_lora_paths: tuple[str, ...] = ()
    option_name: str = "--outpaint-padding"
    # "auto", "1" or "2" (OUTPAINT_PASS_MODES). None means the route default, which is auto.
    passes: str | None = None


@dataclass(frozen=True)
class OutpaintAxisDepth:
    """The deepest single padding on one axis, against the source dimension it grows along.

    Depth is per side rather than per axis total, because the free corner a request opens is
    `deepest vertical padding` x `deepest horizontal padding`: padding the top and the bottom
    equally deepens no corner, it only makes a taller band.
    """

    side: str
    padding_px: int
    ratio: float


_NO_AXIS_DEPTH = OutpaintAxisDepth(side="none", padding_px=0, ratio=0.0)


@dataclass(frozen=True)
class OutpaintFillPlan:
    """The resolved conditioning-canvas decision for one outpaint run."""

    # What the caller asked for ("auto" | "edge" | "neutral" | "solid" | "blur").
    requested: str
    # The concrete OutpaintUtil.create_expanded_canvas fill_mode that will run.
    mode: str
    # Only "solid" consumes a colour; edge/neutral/blur derive their canvas from the source.
    fill_color: tuple[int, int, int] | None
    # Human-readable justification, printed for `auto` runs.
    reason: str
    # The side that stretches edge fill hardest, and how far past its reach it runs.
    max_side: str
    max_side_padding_px: int
    max_side_ratio: float
    # Padding depth edge fill covers on that side, and padding / reach. Above 1.0 the strip
    # would stretch past the validated bound.
    max_side_reach_px: int
    max_side_overreach: float
    # Whether an adapter the route declares as canvas-coupled is loaded.
    uses_solid_fill_adapter: bool
    # The canvas this request builds, after the route's dimension multiple.
    canvas_width: int = 0
    canvas_height: int = 0
    # Deepest padding on each axis. Together they measure the free corner the request opens.
    vertical: OutpaintAxisDepth = _NO_AXIS_DEPTH
    horizontal: OutpaintAxisDepth = _NO_AXIS_DEPTH
    # The source this plan was resolved against and the padding box, for the per-side gaps.
    source_width: int = 0
    source_height: int = 0
    padding: AbsoluteBoxValues | None = None

    @property
    def edge_fill_within_reach(self) -> bool:
        return self.max_side_overreach <= 1.0

    @property
    def is_explicit(self) -> bool:
        return self.requested != OUTPAINT_FILL_AUTO

    @property
    def canvas_pixels(self) -> int:
        return self.canvas_width * self.canvas_height

    @property
    def free_corner_ratio(self) -> float:
        """The shallower of the two axis depths - how square the free corner is against the source.

        Zero whenever either axis is untouched, which is what makes single-axis expansion of any
        depth read as no corner at all.
        """
        return min(self.vertical.ratio, self.horizontal.ratio)

    @property
    def expands_two_deep_axes(self) -> bool:
        """Whether both axes are padded past the measured duplication depth.

        A geometry fact against the shared ratio. Whether a given route acts on it (splits, warns)
        is the route's published `auto_split_corner_ratio`; see `opens_deep_corner`.
        """
        return self.free_corner_ratio > OUTPAINT_TWO_DEEP_AXIS_RATIO

    def opens_deep_corner(self, *, contract: OutpaintContract) -> bool:
        """Whether this request opens a corner the route is measured to duplicate into."""
        ratio = contract.auto_split_corner_ratio
        return ratio is not None and self.free_corner_ratio > ratio

    def gap_px(self, side: str) -> int:
        """New canvas pixels on one side, including the dimension round-up sliver.

        The requested padding is what the caller typed; the gap is what the model actually has to
        paint, and on the trailing sides the two differ by up to one dimension multiple.
        """
        if self.padding is None or side == "none":
            return 0
        if side == "top":
            return self.padding.top
        if side == "left":
            return self.padding.left
        if side == "bottom":
            return max(0, self.canvas_height - self.padding.top - self.source_height)
        if side == "right":
            return max(0, self.canvas_width - self.padding.left - self.source_width)
        raise ValueError(f"Unknown side {side!r}.")


def resolve_outpaint_fill_plan(
    *,
    contract: OutpaintContract,
    request: OutpaintRequest,
    source: PIL.Image.Image,
    padding: AbsoluteBoxValues,
    source_size: tuple[int, int] | None = None,
) -> OutpaintFillPlan:
    """Choose the conditioning canvas for one request on one route.

    Routes that publish `supports_outpaint_fill=False` never reach the `auto` policy: they
    resolve to their single published `outpaint_default_fill_mode` and reject any other
    request. That is what keeps a fixed-fill route's recorded validation runs bit-identical
    when the shared policy layer grows a new default.

    `source_size` overrides the geometry the plan is resolved against while `source` still
    supplies the colours. A later pass of a split request runs on the previous pass's output,
    which does not exist yet at planning time; its size is known, and the one colour decision -
    the default `solid` colour - is better taken from the original source anyway, whose border is
    real rather than generated.
    """
    requested = resolve_requested_fill_mode(contract=contract, fill=request.fill)
    source_width, source_height = source_size if source_size is not None else (source.width, source.height)
    max_side, max_side_padding_px, max_side_ratio, max_side_reach_px, max_side_overreach = _largest_relative_padding(
        padding=padding,
        width=source_width,
        height=source_height,
    )
    vertical, horizontal = _axis_depths(padding=padding, width=source_width, height=source_height)
    canvas_width, canvas_height = OutpaintUtil.expanded_canvas_size(
        source_width=source_width,
        source_height=source_height,
        padding=padding,
        dimension_multiple=contract.dimension_multiple,
    )
    uses_adapter = _uses_solid_fill_adapter(contract=contract, request=request)

    def solid_default_color() -> tuple[int, int, int]:
        if uses_adapter and contract.adapter_fill_color is not None:
            return contract.adapter_fill_color
        return mean_border_color(source, fallback=contract.neutral_fallback_color)

    if requested != OUTPAINT_FILL_AUTO:
        mode = requested
        fill_color = (request.fill_color or solid_default_color()) if mode == "solid" else None
        reason = f"--outpaint-fill {requested} was passed explicitly."
    elif uses_adapter and contract.adapter_fill_color is not None:
        mode = "solid"
        fill_color = request.fill_color or contract.adapter_fill_color
        reason = contract.adapter_fill_reason
    elif max_side_overreach <= 1.0:
        mode = "edge"
        fill_color = None
        reason = (
            f"the deepest padding is {max_side} {max_side_padding_px}px "
            f"({max_side_ratio:.0%} of the source {_axis_label(max_side)}), within the "
            f"{max_side_reach_px}px edge-fill reach, so continuing the source border texture is "
            "the better conditioning canvas."
        )
    else:
        # OutpaintUtil supports a "neutral" canvas: a flat per-side border colour taken from the
        # source. That is the better blank canvas than a fixed mid-gray -- equally textureless, so
        # there is nothing for the model to continue, but sitting in the source's own colour
        # neighbourhood so the boundary is not a hard chroma step the model redraws as a seam.
        mode = "neutral"
        fill_color = None
        reason = (
            f"the deepest padding is {max_side} {max_side_padding_px}px "
            f"({max_side_ratio:.0%} of the source {_axis_label(max_side)}), "
            f"{max_side_overreach:.1f}x the {max_side_reach_px}px edge-fill reach; "
            f"{_blank_canvas_reason(contract=contract)}"
        )

    if mode not in contract.fill_modes and mode != OUTPAINT_FILL_AUTO:
        raise OutpaintError(
            f"Fill mode {mode!r} is not supported by {contract.capability_id}; "
            f"supported modes are {', '.join(contract.fill_modes)}."
        )
    return OutpaintFillPlan(
        requested=requested,
        mode=mode,
        fill_color=fill_color,
        reason=reason,
        max_side=max_side,
        max_side_padding_px=max_side_padding_px,
        max_side_ratio=max_side_ratio,
        max_side_reach_px=max_side_reach_px,
        max_side_overreach=max_side_overreach,
        uses_solid_fill_adapter=uses_adapter,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        vertical=vertical,
        horizontal=horizontal,
        source_width=source_width,
        source_height=source_height,
        padding=padding,
    )


def _blank_canvas_reason(*, contract: OutpaintContract) -> str:
    # Why a blank canvas is the better one is route-dependent, so the sentence is chosen by how
    # far the route's fill reaches rather than shared. Where the canvas never enters the
    # trajectory and the padded region's reference tokens are dropped, the fill decides what the
    # seam ring sees and nothing more; claiming it decides whether the model invents new subject
    # matter would be a claim about that route it does not hold.
    if not contract.fill_reaches_padding:
        return (
            "the fill conditions the seam around the source rather than the padded region, and a "
            "blank seam does not hand the model a stretched border strip to continue."
        )
    return "a blank canvas makes the model generate new subject matter instead of smearing the source border."


def resolve_requested_fill_mode(*, contract: OutpaintContract, fill: str | None) -> str:
    """The fill mode one request names on one route, or raise OutpaintError. Loads nothing.

    Pure policy over the route's published fill contract, with no source image and no canvas, so
    a caller that only holds a model name can ask the question before dispatch. `mlxgen generate`
    declares `--outpaint-fill` once for every model, so this is what keeps an unsupported request
    on the route's own terms instead of reaching a backend parser that never declared the option.
    """
    requested = fill or contract.default_fill_mode
    if contract.supports_fill_option:
        if requested not in contract.fill_modes:
            raise OutpaintError(
                f"Fill mode {requested!r} is not supported by {contract.capability_id}; "
                f"supported modes are {', '.join(contract.fill_modes)}."
            )
        return requested
    if requested not in {OUTPAINT_FILL_AUTO, contract.default_fill_mode}:
        raise OutpaintError(
            f"{contract.capability_id} has a fixed {contract.default_fill_mode!r} conditioning canvas "
            f"and does not accept an outpaint fill mode ({requested!r} was requested)."
        )
    return contract.default_fill_mode


def check_outpaint_fill_options(
    *,
    contract: OutpaintContract,
    fill: str | None = None,
    fill_color_requested: bool = False,
    passes: str | None = None,
) -> None:
    """Raise OutpaintError when a route cannot honour the requested canvas options. Loads nothing."""
    resolve_requested_fill_mode(contract=contract, fill=fill)
    if fill_color_requested and not contract.supports_fill_option:
        raise OutpaintError(
            f"{contract.capability_id} has a fixed {contract.default_fill_mode!r} conditioning canvas "
            "and does not accept an outpaint fill color."
        )
    resolve_requested_passes(passes=passes)


def resolve_requested_passes(*, passes: str | int | None) -> str:
    """The `--outpaint-passes` value one request names, normalised, or raise OutpaintError."""
    if passes is None:
        return OUTPAINT_PASSES_AUTO
    requested = str(passes).strip().lower()
    if requested not in OUTPAINT_PASS_MODES:
        raise OutpaintError(
            f"--outpaint-passes must be one of {', '.join(OUTPAINT_PASS_MODES)} ({passes!r} was requested)."
        )
    return requested


@dataclass(frozen=True)
class OutpaintPassPlan:
    """How one request is run: as one canvas, or as two single-axis passes.

    `paddings` are the absolute per-pass boxes in run order, each applied to the previous pass's
    output (the original source for the first). `paste_left` / `paste_top` locate the original
    source in the final canvas, which for a split request can differ from the requested left/top
    by less than one dimension multiple: the intermediate canvas is rounded up on the axis the
    first pass leaves alone, and that sliver is generated in the first pass rather than added
    again in the second, so the final canvas is exactly the one-pass canvas.
    """

    requested: str
    paddings: tuple[AbsoluteBoxValues, ...]
    reason: str
    canvas_width: int
    canvas_height: int
    paste_left: int
    paste_top: int

    @property
    def count(self) -> int:
        return len(self.paddings)

    @property
    def is_split(self) -> bool:
        return self.count > 1

    @property
    def padding_values(self) -> tuple[str, ...]:
        """Each pass's box as the `top,right,bottom,left` string the canvas builder parses."""
        return tuple(format_padding(padding) for padding in self.paddings)


def format_padding(padding: AbsoluteBoxValues) -> str:
    return f"{padding.top},{padding.right},{padding.bottom},{padding.left}"


def resolve_outpaint_pass_plan(
    *,
    contract: OutpaintContract,
    request: OutpaintRequest,
    fill_plan: OutpaintFillPlan,
) -> OutpaintPassPlan:
    """Decide the pass count for one request on one route. Pure geometry, loads nothing.

    `auto` splits exactly when the request opens a corner deeper than the route's published
    `auto_split_corner_ratio` on both axes; a single-axis request of any depth stays one pass, as
    does every request on a route that publishes no ratio. An explicit "1" always runs one pass
    (the guard then warns on a deep corner, because explicit intent wins but never silently). An
    explicit "2" splits any request that pads both axes and raises for one it cannot split.
    """
    requested = resolve_requested_passes(passes=request.passes)
    padding = fill_plan.padding
    if padding is None:
        raise OutpaintError("The fill plan carries no padding box; resolve it with resolve_outpaint_fill_plan.")
    single = (padding,)
    geometry = dict(
        canvas_width=fill_plan.canvas_width,
        canvas_height=fill_plan.canvas_height,
    )
    if requested == "1":
        return OutpaintPassPlan(
            requested=requested,
            paddings=single,
            reason="--outpaint-passes 1 was passed explicitly.",
            paste_left=padding.left,
            paste_top=padding.top,
            **geometry,
        )
    split = _single_axis_passes(fill_plan=fill_plan, dimension_multiple=contract.dimension_multiple)
    if requested == "2":
        if split is None:
            raise OutpaintError(
                "--outpaint-passes 2 splits a request into one horizontal and one vertical pass, which "
                f"needs padding on both axes; {request.option_name} {request.padding!r} on a "
                f"{fill_plan.source_width}x{fill_plan.source_height} source "
                f"{'pads one axis only' if fill_plan.free_corner_ratio == 0.0 else 'adds less than one dimension multiple on one axis'}. "
                "Pass --outpaint-passes 1 or auto."
            )
        paddings, (paste_left, paste_top) = split
        return OutpaintPassPlan(
            requested=requested,
            paddings=paddings,
            reason="--outpaint-passes 2 was passed explicitly.",
            paste_left=paste_left,
            paste_top=paste_top,
            **geometry,
        )
    if not fill_plan.opens_deep_corner(contract=contract):
        if fill_plan.free_corner_ratio == 0.0:
            reason = "the request pads one axis only, so it opens no free corner."
        elif contract.auto_split_corner_ratio is None:
            reason = f"{contract.capability_id} publishes no auto-split depth."
        else:
            reason = f"{_corner_summary(fill_plan)} within the {contract.auto_split_corner_ratio:.0%} auto-split depth on at least one axis."
        return OutpaintPassPlan(
            requested=requested,
            paddings=single,
            reason=reason,
            paste_left=padding.left,
            paste_top=padding.top,
            **geometry,
        )
    if split is None:
        return OutpaintPassPlan(
            requested=requested,
            paddings=single,
            reason=(
                f"{_corner_summary(fill_plan)} past the {contract.auto_split_corner_ratio:.0%} auto-split depth, but the "
                "second pass would add less than one dimension multiple, so the request runs as one pass."
            ),
            paste_left=padding.left,
            paste_top=padding.top,
            **geometry,
        )
    paddings, (paste_left, paste_top) = split
    return OutpaintPassPlan(
        requested=requested,
        paddings=paddings,
        reason=(
            f"{_corner_summary(fill_plan)} past the {contract.auto_split_corner_ratio:.0%} auto-split depth, and a free corner is "
            "where the model paints a second copy of the subject; two single-axis passes reach the "
            "same canvas with no free corner at either pass."
        ),
        paste_left=paste_left,
        paste_top=paste_top,
        **geometry,
    )


def _corner_summary(fill_plan: OutpaintFillPlan) -> str:
    vertical, horizontal = fill_plan.vertical, fill_plan.horizontal
    return (
        f"the request pads {vertical.side} {vertical.padding_px}px ({vertical.ratio:.0%} of the source "
        f"height) and {horizontal.side} {horizontal.padding_px}px ({horizontal.ratio:.0%} of the source "
        f"width), which opens a {fill_plan.gap_px(horizontal.side)}x{fill_plan.gap_px(vertical.side)}px free corner"
    )


def _single_axis_passes(
    *, fill_plan: OutpaintFillPlan, dimension_multiple: int
) -> tuple[tuple[AbsoluteBoxValues, AbsoluteBoxValues], tuple[int, int]] | None:
    """Split a two-axis padding box into a horizontal pass and a vertical pass, or None.

    The deeper axis (by ratio) runs first, so the harder expansion is anchored on the real source
    and the shallower one on a mostly-real intermediate; ties run horizontal first. The second
    pass is sized to land exactly on the one-pass canvas: the intermediate is already rounded up
    on the second axis, and that sliver counts against the second pass's trailing side, then
    against its leading side when the trailing side has nothing left to give.
    """
    padding = fill_plan.padding
    if padding is None or fill_plan.free_corner_ratio == 0.0:
        return None
    width, height = fill_plan.source_width, fill_plan.source_height
    final_width, final_height = fill_plan.canvas_width, fill_plan.canvas_height
    horizontal_first = fill_plan.horizontal.ratio >= fill_plan.vertical.ratio
    if horizontal_first:
        first = AbsoluteBoxValues(top=0, right=padding.right, bottom=0, left=padding.left)
        _, intermediate_height = OutpaintUtil.expanded_canvas_size(
            source_width=width, source_height=height, padding=first, dimension_multiple=dimension_multiple
        )
        top, bottom = _fit_axis(lead=padding.top, intermediate=intermediate_height, final=final_height)
        second = AbsoluteBoxValues(top=top, right=0, bottom=bottom, left=0)
        paste = (padding.left, top)
    else:
        first = AbsoluteBoxValues(top=padding.top, right=0, bottom=padding.bottom, left=0)
        intermediate_width, _ = OutpaintUtil.expanded_canvas_size(
            source_width=width, source_height=height, padding=first, dimension_multiple=dimension_multiple
        )
        left, right = _fit_axis(lead=padding.left, intermediate=intermediate_width, final=final_width)
        second = AbsoluteBoxValues(top=0, right=right, bottom=0, left=left)
        paste = (left, padding.top)
    if max(second.top, second.right, second.bottom, second.left) <= 0:
        return None
    return (first, second), paste


def _fit_axis(*, lead: int, intermediate: int, final: int) -> tuple[int, int]:
    # The intermediate already holds the source plus up to one dimension multiple of round-up
    # filler on its trailing side, generated in the first pass. The second pass adds only what is
    # left between that and the one-pass canvas, so the two agree exactly.
    trail = final - intermediate - lead
    if trail < 0:
        lead = final - intermediate
        trail = 0
    return max(0, lead), max(0, trail)


def guard_outpaint_fill_plan(
    *,
    contract: OutpaintContract,
    fill_plan: OutpaintFillPlan,
    pass_plan: OutpaintPassPlan | None = None,
) -> tuple[str, ...]:
    """Fail closed on a known-bad canvas; return the warnings the caller should surface.

    Two independent checks. The geometry check reports a request that opens a free corner the
    route is measured to duplicate into and that is nevertheless about to run as one pass - either
    because the caller passed `--outpaint-passes 1`, or because the request could not be split.
    With no `pass_plan` the check reads the geometry alone, which is what a caller that has not
    planned passes yet needs to hear. The fill check reports an edge canvas stretched past its
    reach: explicit user intent wins, but never silently, and an `auto` run that reaches that
    state is unreachable today and raises rather than running the measured-bad configuration.

    Warnings are returned, never printed - `emit_canvas_notices` is the one place that decides
    where they go.
    """
    warnings: list[str] = []
    runs_as_one_pass = pass_plan is None or not pass_plan.is_split
    if runs_as_one_pass and fill_plan.opens_deep_corner(contract=contract):
        warnings.append(_two_deep_axis_warning(fill_plan, contract=contract, pass_plan=pass_plan))
    if fill_plan.mode == "edge" and not fill_plan.edge_fill_within_reach:
        if not fill_plan.is_explicit:
            raise OutpaintError(
                f"--outpaint-fill auto resolved to edge with {fill_plan.max_side} padding of "
                f"{fill_plan.max_side_padding_px}px, {fill_plan.max_side_overreach:.1f}x the "
                f"{fill_plan.max_side_reach_px}px edge-fill reach. Edge fill smears at that depth. Pass "
                "--outpaint-fill neutral for a blank conditioning canvas, or --outpaint-fill edge to "
                "force the edge canvas anyway."
            )
        warnings.append(_unsafe_edge_fill_warning(fill_plan, contract=contract))
    return tuple(warnings)


def _two_deep_axis_warning(
    fill_plan: OutpaintFillPlan, *, contract: OutpaintContract, pass_plan: OutpaintPassPlan | None
) -> str:
    # Only remedies with evidence behind them are named, all measured on the two-deep-axis
    # geometry across three seeds. Two single-axis passes are the fix (clean 3/3 on the worst
    # case, and what `auto` does), so the warning's first job is to say why this run is not
    # taking it. Prompt content is the other lever: a subject-naming prompt duplicated 3/3, while
    # an environment-only prompt describing the area to add was clean 3/3 - the source already
    # conditions the model on the subject, so naming it again asks for a second one. Step count is
    # deliberately NOT offered: at 16, 32 and 48 steps duplication persisted on every seed, so
    # raising steps does not move this failure.
    # The conditioning canvas is not offered either on a latent-locked route, where the fill
    # reaches only the seam ring and never the free corner - pointing at --outpaint-fill there
    # would send the caller at a knob that cannot touch the symptom.
    envelope = ""
    validated_pixels = contract.validated_max_canvas_pixels
    if validated_pixels is not None and fill_plan.canvas_pixels > validated_pixels:
        envelope = (
            f" The canvas is also past the {validated_pixels} px "
            f"{contract.capability_id} publishes as its validated maximum."
        )
    canvas_lever = (
        " The conditioning canvas is not the lever on this route: it conditions the seam around "
        "the source, not the free corner."
        if not contract.fill_reaches_padding
        else ""
    )
    if pass_plan is not None and pass_plan.requested == "1":
        one_pass = (
            "Proceeding in one pass because --outpaint-passes 1 was passed. Drop it, or pass "
            "--outpaint-passes auto, to run two single-axis passes instead: they reach the same "
            "canvas with no free corner at either pass, and are measured clean where one pass "
            "duplicates."
        )
    elif pass_plan is not None:
        one_pass = (
            f"Proceeding in one pass: {pass_plan.reason} Two single-axis passes are the measured "
            "remedy where the request can be split."
        )
    else:
        one_pass = (
            "Running the request as two single-axis passes reaches the same canvas with no free "
            "corner at either pass and is measured clean where one pass duplicates; "
            "--outpaint-passes auto takes that route."
        )
    return (
        f"Warning: --outpaint-padding expands both axes deeply - {fill_plan.vertical.side} "
        f"{fill_plan.vertical.padding_px}px ({fill_plan.vertical.ratio:.0%} of the source height) and "
        f"{fill_plan.horizontal.side} {fill_plan.horizontal.padding_px}px "
        f"({fill_plan.horizontal.ratio:.0%} of the source width), opening a "
        f"{fill_plan.gap_px(fill_plan.horizontal.side)}x{fill_plan.gap_px(fill_plan.vertical.side)}px corner of the "
        f"{fill_plan.canvas_width}x{fill_plan.canvas_height} canvas that shares neither a row nor a "
        f"column with the source. {contract.capability_id} is measured to duplicate the subject "
        "there: the model paints a second copy of it into the free corner instead of completing the "
        f"one it was given.{envelope} {one_pass} Describing the area being added rather than the "
        "subject also moves it - a prompt naming the subject asks for one in the new space, and the "
        "source already conditions the model on it. Raising --steps does not move it."
        f"{canvas_lever}"
    )


def _unsafe_edge_fill_warning(fill_plan: OutpaintFillPlan, *, contract: OutpaintContract) -> str:
    # The remediation half is route-dependent: a route that publishes no fill option cannot be
    # told to pick a different canvas, so pointing it at --outpaint-fill would be a lie.
    if contract.supports_fill_option:
        remedy = (
            "Proceeding as requested. Use --outpaint-fill neutral for a blank conditioning canvas, or add "
            f"--lora-paths {contract.recommended_lora} for the validated green-canvas route."
            if contract.recommended_lora
            else "Proceeding as requested. Use --outpaint-fill neutral for a blank conditioning canvas."
        )
    else:
        remedy = (
            f"Proceeding: {contract.capability_id} has a fixed {contract.default_fill_mode} conditioning canvas. "
            "Reduce the padding, or use a route that publishes supports_outpaint_fill."
        )
    return (
        f"Warning: --outpaint-fill edge with {fill_plan.max_side} padding of "
        f"{fill_plan.max_side_padding_px}px ({fill_plan.max_side_ratio:.0%} of the source "
        f"{_axis_label(fill_plan.max_side)}) runs {fill_plan.max_side_overreach:.1f}x the "
        f"{fill_plan.max_side_reach_px}px edge-fill reach. Edge fill stretches a border strip "
        f"across the padded region and produces directional smear past its reach. {remedy}"
    )


def _largest_relative_padding(
    *, padding: AbsoluteBoxValues, width: int, height: int
) -> tuple[str, int, float, int, float]:
    # Each side is measured against the source dimension it grows along: top/bottom against the
    # height, left/right against the width. The side that matters is the one running furthest past
    # what edge fill can cover, so sides rank by overreach (padding / reach), not by raw ratio.
    # Ties resolve in top,right,bottom,left order so the reported side is stable.
    sides = (
        ("top", padding.top, height),
        ("right", padding.right, width),
        ("bottom", padding.bottom, height),
        ("left", padding.left, width),
    )
    best_side, best_pixels, best_ratio, best_reach, best_overreach = "top", 0, 0.0, 0, 0.0
    for name, pixels, base in sides:
        reach = OutpaintUtil.edge_fill_reach(base)
        overreach = (pixels / reach) if reach > 0 else 0.0
        if overreach > best_overreach:
            ratio = (pixels / base) if base > 0 else 0.0
            best_side, best_pixels, best_ratio = name, pixels, ratio
            best_reach, best_overreach = reach, overreach
    if best_reach == 0:
        best_reach = OutpaintUtil.edge_fill_reach(height)
    return best_side, best_pixels, best_ratio, best_reach, best_overreach


def _axis_depths(*, padding: AbsoluteBoxValues, width: int, height: int) -> tuple[OutpaintAxisDepth, OutpaintAxisDepth]:
    """The deepest padding on the vertical and on the horizontal axis.

    These two measure the largest free corner a request opens: its height is the deepest top or
    bottom padding, its width the deepest left or right padding. A request that touches only one
    axis reports zero on the other and therefore opens no corner at all, however deep it runs.
    """

    def deepest(candidates: tuple[tuple[str, int], ...], base: int) -> OutpaintAxisDepth:
        side, pixels = max(candidates, key=lambda candidate: candidate[1])
        if pixels <= 0 or base <= 0:
            return _NO_AXIS_DEPTH
        return OutpaintAxisDepth(side=side, padding_px=pixels, ratio=pixels / base)

    return (
        deepest((("top", padding.top), ("bottom", padding.bottom)), height),
        deepest((("left", padding.left), ("right", padding.right)), width),
    )


def _axis_label(side: str) -> str:
    return "height" if side in {"top", "bottom"} else "width"


def mean_border_color(
    source: PIL.Image.Image,
    *,
    fallback: tuple[int, int, int] = OUTPAINT_NEUTRAL_FALLBACK_COLOR,
) -> tuple[int, int, int]:
    """Mean RGB of the source's outer ring - the flat colour a blank canvas should use.

    A fixed mid-gray is equally flat but plants a hard chroma step along the source boundary,
    and that step is itself a strong edge the model tends to redraw as a visible seam.
    """
    rgb = source.convert("RGB")
    if rgb.width < 2 or rgb.height < 2:
        return fallback
    ring = max(1, min(16, min(rgb.width, rgb.height) // 32))
    strips = (
        rgb.crop((0, 0, rgb.width, ring)),
        rgb.crop((0, rgb.height - ring, rgb.width, rgb.height)),
        rgb.crop((0, 0, ring, rgb.height)),
        rgb.crop((rgb.width - ring, 0, rgb.width, rgb.height)),
    )
    totals = [0.0, 0.0, 0.0]
    sampled_pixels = 0
    for strip in strips:
        pixels = strip.width * strip.height
        if pixels == 0:
            continue
        means = PIL.ImageStat.Stat(strip).mean[:3]
        for channel, mean in enumerate(means):
            totals[channel] += mean * pixels
        sampled_pixels += pixels
    if sampled_pixels == 0:
        return fallback
    return tuple(max(0, min(255, round(total / sampled_pixels))) for total in totals)  # type: ignore[return-value]


def _uses_solid_fill_adapter(*, contract: OutpaintContract, request: OutpaintRequest) -> bool:
    # Match the pre-resolution request as well as the resolved path. LoraResolution rewrites
    # "fal/flux-2-klein-4B-outpaint-lora" into a concrete cache file, so the repo-form marker only
    # ever matched by accident, through whichever basename that repo happened to resolve to;
    # a sibling file in the same repo (the comfy-converted weights) silently did not match.
    if not contract.adapter_markers:
        return False
    specs = [*request.requested_lora_paths, *request.lora_paths]
    return any(_spec_matches_marker(str(spec), contract.adapter_markers) for spec in specs)


def _spec_matches_marker(spec: str, markers: tuple[str, ...]) -> bool:
    normalized = spec.lower()
    # Hugging Face cache directories spell "org/repo" as "models--org--repo"; normalizing the
    # double dash back to a slash lets the repo-form markers match a resolved snapshot path too.
    candidates = (normalized, normalized.replace("--", "/"))
    return any(marker in candidate for candidate in candidates for marker in markers)


@dataclass(kw_only=True)
class _ExpandedCanvasSession:
    """Shared plumbing for the two expanded-canvas workflows.

    Outpaint and reframe genuinely share the canvas builder, the derived generation geometry,
    and the conditioning-image plumbing. They do not share a preservation strategy or a fill
    policy, so those live only on OutpaintSession.
    """

    source_image: str | Path
    padding: str
    canvas: OutpaintCanvas
    _workspace: TemporaryDirectory | None = field(default=None, repr=False)

    @property
    def width(self) -> int:
        """Generation width. Callers must not also pass their own --width."""
        return self.canvas.target_width

    @property
    def height(self) -> int:
        return self.canvas.target_height

    @property
    def canvas_policy(self) -> str:
        return CANVAS_POLICY_EXACT_RESIZE

    @property
    def conditioning_image_paths(self) -> list[Path]:
        """The image paths the model should be conditioned on - the canvas, not the source."""
        return [self.canvas.canvas_path]

    def close(self) -> None:
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


@dataclass(kw_only=True)
class OutpaintSession(_ExpandedCanvasSession):
    """A prepared outpaint run: resolved policy, built canvas, and the generation contract.

    The whole outpaint-specific pipeline is three calls - build (`prepare_outpaint`),
    `generate(...)`, `finalize(...)` - and `generate` already calls `finalize`. A caller that
    owns its own seed loop (the CLI) calls `generate`; a caller that wants the runtime
    wrapper's multi-seed, progress and save behaviour calls `run_outpaint`.

    A request may run as more than one pass (`pass_plan`). `canvas` and `fill_plan` describe the
    first pass, which is built when the session is prepared; every later pass runs on the previous
    pass's output, so its canvas is built inside `generate` when that output exists. `geometry`
    describes the original source inside the final canvas and is what `finalize`, `width` and
    `height` report, whatever the pass count.
    """

    contract: OutpaintContract
    fill_plan: OutpaintFillPlan
    pass_plan: OutpaintPassPlan
    pass_fill_plans: tuple[OutpaintFillPlan, ...]
    request: OutpaintRequest
    geometry: OutpaintCanvas
    warnings: tuple[str, ...] = ()
    fill_color_explicit: bool = False
    _pass_canvases: list[OutpaintCanvas] = field(default_factory=list, repr=False)
    # Per-pass preservation results of the run in progress: what each pass measured against its
    # own source before the route's restore, and whether that restore applied.
    _pass_restore_differences: list[float] = field(default_factory=list, repr=False)
    _pass_restore_applied: list[bool] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self._pass_canvases:
            self._pass_canvases.append(self.canvas)

    @property
    def width(self) -> int:
        """Final generation width, after every pass. Callers must not also pass their own --width."""
        return self.geometry.target_width

    @property
    def height(self) -> int:
        return self.geometry.target_height

    @property
    def passes(self) -> int:
        return self.pass_plan.count

    @property
    def pass_canvases(self) -> tuple[OutpaintCanvas, ...]:
        """The canvases built so far, one per pass that has started; the first is `canvas`."""
        return tuple(self._pass_canvases)

    @property
    def notice(self) -> str:
        """The stderr summary of the resolved pass plan and conditioning canvas(es)."""
        lines: list[str] = []
        if self.pass_plan.is_split or self.pass_plan.requested != OUTPAINT_PASSES_AUTO:
            steps = "; ".join(
                f"pass {index + 1} pads {value} to {plan.canvas_width}x{plan.canvas_height}"
                for index, (value, plan) in enumerate(zip(self.pass_plan.padding_values, self.pass_fill_plans))
            )
            count = f"{self.pass_plan.count} pass" + ("es" if self.pass_plan.count != 1 else "")
            lines.append(f"Outpaint: {count} because {self.pass_plan.reason} {steps[0].upper()}{steps[1:]}.")
        for index, plan in enumerate(self.pass_fill_plans):
            lines.extend(self._fill_notice_lines(index=index, fill_plan=plan))
        return "\n".join(lines)

    def _fill_notice_lines(self, *, index: int, fill_plan: OutpaintFillPlan) -> list[str]:
        prefix = "Outpaint:" if index == 0 else f"Outpaint: pass {index + 1}"
        padding = fill_plan.padding if fill_plan.padding is not None else self.canvas.padding
        color = ""
        if fill_plan.mode == "solid" and fill_plan.fill_color is not None:
            color = f" color={fill_plan.fill_color[0]},{fill_plan.fill_color[1]},{fill_plan.fill_color[2]}"
        lines = [
            f"{prefix} fill={fill_plan.mode}{color}, canvas "
            f"{fill_plan.canvas_width}x{fill_plan.canvas_height} from source "
            f"{fill_plan.source_width}x{fill_plan.source_height}, padding top={padding.top} "
            f"right={padding.right} bottom={padding.bottom} left={padding.left}."
        ]
        if not fill_plan.is_explicit:
            lines.append(f"{prefix} --outpaint-fill auto selected {fill_plan.mode} because {fill_plan.reason}")
        if index == 0 and fill_plan.mode != "solid" and self.fill_color_explicit:
            lines.append(
                f"Outpaint: --outpaint-fill-color only applies to --outpaint-fill solid "
                f"and is ignored by {fill_plan.mode}."
            )
        return lines

    def conditioning_kwargs(self, canvas: OutpaintCanvas | None = None) -> dict[str, Any]:
        """The generate_image(...) keywords that carry one pass's canvas on this route.

        Defaults to the first pass. `generate` is the caller that runs every pass; a caller that
        drives the model itself with these keywords runs the first pass only.
        """
        canvas = canvas if canvas is not None else self.canvas
        if self.contract.conditioning == CONDITIONING_CANVAS_OBJECT:
            return {"canvas": canvas}
        if self.contract.conditioning == CONDITIONING_IMAGE_PATHS:
            kwargs: dict[str, Any] = {
                "image_path": self.source_image,
                "image_paths": [str(canvas.canvas_path)],
                "width": canvas.target_width,
                "height": canvas.target_height,
                "canvas_policy": self.canvas_policy,
            }
            if self.contract.source_lock_mask:
                if canvas.lock_mask_path is None:
                    raise OutpaintError(
                        f"{self.contract.capability_id} locks the source through a mask, but the canvas has none."
                    )
                kwargs["mask_path"] = str(canvas.lock_mask_path)
            return kwargs
        raise OutpaintError(f"Unknown outpaint conditioning mode {self.contract.conditioning!r}.")

    def generate(self, model: Any, /, **generate_kwargs: Any):
        """Run one seed on `model` through every planned pass and finalize the artifact.

        The canvas keywords come from the route contract; everything else (seed, prompt,
        guidance, steps, scheduler, negative_prompt) is the caller's and is passed unchanged to
        every pass. Between passes the route's own preservation is applied to the intermediate
        and it becomes the next pass's source.
        """
        overlap = sorted(set(generate_kwargs) & set(self.conditioning_kwargs()))
        if overlap:
            raise OutpaintError(f"Outpaint owns these generation keywords: {', '.join(overlap)}.")
        # One run's canvases and results: a caller looping seeds through the same session sees
        # the run in progress, not an accumulation across seeds.
        del self._pass_canvases[1:]
        self._pass_restore_differences.clear()
        self._pass_restore_applied.clear()
        generated: Any = None
        for index in range(self.pass_plan.count):
            canvas = self.canvas if index == 0 else self._prepare_pass_canvas(index=index, previous=generated)
            generated = model.generate_image(**self.conditioning_kwargs(canvas), **generate_kwargs)
            if index + 1 < self.pass_plan.count:
                self._settle_pass(generated, canvas=canvas)
        self.finalize(generated)
        return generated

    def _prepare_pass_canvas(self, *, index: int, previous: Any) -> OutpaintCanvas:
        # The previous pass's settled output is the source of this one. It goes through disk on
        # purpose: the canvas builder, the route's reference conditioning and the source lock all
        # read a file, and one lossless PNG round trip is the whole cost of staying model-agnostic.
        workspace = self.canvas.canvas_path.parent
        source_path = workspace / f"outpaint_pass{index + 1}_source.png"
        previous.image.save(source_path)
        fill_plan = self.pass_fill_plans[index]
        canvas = OutpaintUtil.create_expanded_canvas(
            source_path=source_path,
            padding_value=self.pass_plan.padding_values[index],
            output_path=workspace / f"outpaint_pass{index + 1}_canvas.png",
            dimension_multiple=self.contract.dimension_multiple,
            fill_mode=fill_plan.mode,
            fill_color=fill_plan.fill_color or self.contract.base_fill_color,
            option_name=self.request.option_name,
        )
        planned = (fill_plan.canvas_width, fill_plan.canvas_height)
        if (canvas.target_width, canvas.target_height) != planned:
            raise OutpaintError(
                f"Outpaint pass {index + 1} built a {canvas.target_width}x{canvas.target_height} canvas "
                f"where the pass plan expected {planned[0]}x{planned[1]}."
            )
        canvas = _with_source_lock_mask(contract=self.contract, canvas=canvas)
        self._pass_canvases.append(canvas)
        return canvas

    def _settle_pass(self, generated: Any, *, canvas: OutpaintCanvas) -> None:
        # The route's own preservation, applied per pass against that pass's own source. A route
        # that pastes its source back does so here too, so the next pass starts from the best
        # intermediate the route can produce - and the last pass, settled the same way, hands
        # `finalize` a canvas whose source window already carries the earlier passes' restores.
        # Without this a split run would be judged only against the original: each pass's own
        # drift (a VAE round trip plus the lock) adds up, and a two-pass Qwen run measured 12.6
        # against a 12.0 threshold, so the restore that a one-pass run at 5.9 applies was skipped.
        generated.image = OutpaintUtil.composite_source_region(
            generated_image=generated.image,
            canvas=canvas,
            feather_px=None,
            preserve_source=self.contract.preserve_source,
            restore_threshold=self.contract.restore_threshold,
        )
        self._pass_restore_differences.append(float(generated.image.outpaint_source_restore_difference))
        self._pass_restore_applied.append(bool(generated.image.outpaint_preservation_applied))

    def finalize(self, generated: Any) -> None:
        """Composite the source back where the route asks for it, then attach metadata.

        Mutates `generated` in place: it replaces `generated.image`, repoints the recorded
        source paths at the original image rather than the canvas, and records both the
        padding geometry and the resolved fill so `-C metadata.json` replays the resolved
        canvas instead of re-running `auto`. The source region measured and restored is the
        original source at its final position, whatever the pass count.
        """
        underlying_difference: float | None = None
        if self.pass_plan.is_split:
            # What the model drew under the original, before any pass restores it - the number
            # the recorded `outpaint_source_restore_difference` has always meant.
            underlying_difference = OutpaintUtil.source_region_difference(
                generated_image=generated.image, canvas=self.geometry
            )
            last_canvas = self._pass_canvases[-1] if len(self._pass_canvases) == self.pass_plan.count else None
            if last_canvas is not None:
                self._settle_pass(generated, canvas=last_canvas)
        generated.image = OutpaintUtil.composite_source_region(
            generated_image=generated.image,
            canvas=self.geometry,
            feather_px=None,
            preserve_source=self.contract.preserve_source,
            restore_threshold=self.contract.restore_threshold,
        )
        if underlying_difference is not None:
            generated.image.outpaint_source_restore_difference = underlying_difference
        generated.image_path = self.source_image
        generated.image_paths = [self.source_image]
        OutpaintUtil.attach_metadata(
            generated_image=generated,
            canvas=self.geometry,
            padding_value=self.padding,
            preservation=self.contract.preservation,
        )
        extra_metadata = dict(getattr(generated, "extra_metadata", None) or {})
        extra_metadata.update(
            {
                "outpaint_fill": self.fill_plan.mode,
                "outpaint_fill_color": list(self.fill_plan.fill_color)
                if self.fill_plan.fill_color is not None
                else None,
                "outpaint_fill_requested": self.fill_plan.requested,
                "outpaint_fill_reason": self.fill_plan.reason,
                "outpaint_max_side_padding": self.fill_plan.max_side,
                "outpaint_max_side_padding_px": self.fill_plan.max_side_padding_px,
                "outpaint_max_side_padding_ratio": round(self.fill_plan.max_side_ratio, 4),
                "outpaint_edge_fill_reach_px": self.fill_plan.max_side_reach_px,
                "outpaint_edge_fill_overreach": round(self.fill_plan.max_side_overreach, 4),
                "outpaint_passes": self.pass_plan.count,
                "outpaint_passes_requested": self.pass_plan.requested,
                "outpaint_pass_paddings": list(self.pass_plan.padding_values),
                "outpaint_pass_fills": [plan.mode for plan in self.pass_fill_plans],
                "outpaint_pass_reason": self.pass_plan.reason,
                # Per pass, against that pass's own source: what it drew underneath, and whether
                # the route's restore applied. One entry on a single-pass run would repeat the
                # whole-run numbers, so the lists are empty there.
                "outpaint_pass_source_restore_differences": [
                    round(value, 4) for value in self._pass_restore_differences
                ],
                "outpaint_pass_source_restore_applied": list(self._pass_restore_applied),
            }
        )
        generated.extra_metadata = extra_metadata


@dataclass(kw_only=True)
class ReframeSession(_ExpandedCanvasSession):
    """A prepared generative reframe run.

    Reframe shares the canvas builder and the derived geometry with outpaint and nothing
    else: there is no fill policy (the historical edge-extended canvas is the contract) and
    no source preservation (reframe is allowed to recompose the source).
    """

    def conditioning_kwargs(self) -> dict[str, Any]:
        return {
            "image_paths": [str(path) for path in self.conditioning_image_paths],
            "width": self.width,
            "height": self.height,
            "canvas_policy": self.canvas_policy,
        }

    def generate(self, model: Any, /, **generate_kwargs: Any):
        generated = model.generate_image(**self.conditioning_kwargs(), **generate_kwargs)
        self.finalize(generated)
        return generated

    def finalize(self, generated: Any) -> None:
        generated.image_path = self.source_image
        generated.image_paths = [self.source_image]
        OutpaintUtil.attach_reframe_metadata(
            generated_image=generated,
            canvas=self.canvas,
            padding_value=self.padding,
        )


def prepare_outpaint(
    *,
    source_image: str | Path,
    padding: str,
    contract: OutpaintContract | None = None,
    capability: GenerationCapability | None = None,
    model: str | None = None,
    model_config: Any = None,
    base_model: str | None = None,
    fill: str | None = None,
    fill_color: tuple[int, int, int] | None = None,
    fill_color_explicit: bool = False,
    lora_paths: Sequence[str] = (),
    requested_lora_paths: Sequence[str] = (),
    passes: str | int | None = None,
    workspace: str | Path | None = None,
    canvas_name: str = "outpaint_canvas.png",
    option_name: str = "--outpaint-padding",
) -> OutpaintSession:
    """Resolve the outpaint policy and build the first conditioning canvas. Loads no weights.

    Identify the route with exactly one of `contract`, `capability`, or `model`. When
    `workspace` is omitted the session owns a temporary directory and must be closed - use it
    as a context manager, or let `run_outpaint` own it.

    `passes` is "auto" (default), "1" or "2": whether a request that pads both axes deeply runs
    as two single-axis passes. Every pass's fill is resolved here, so `session.notice` and
    `session.warnings` describe the whole run before any weight is loaded; only the later passes'
    canvases wait for the output they are built on.

    Raises OutpaintError for an unsupported fill mode or pass count, a padding value that adds no
    pixels, a request `passes="2"` cannot split, or an `auto` decision that would run a canvas
    measured to smear.
    """
    resolved = _resolve_contract(
        contract=contract, capability=capability, model=model, model_config=model_config, base_model=base_model
    )
    request = OutpaintRequest(
        padding=padding,
        fill=fill,
        fill_color=fill_color,
        fill_color_explicit=fill_color_explicit,
        lora_paths=tuple(str(path) for path in lora_paths),
        requested_lora_paths=tuple(str(path) for path in requested_lora_paths),
        option_name=option_name,
        passes=None if passes is None else str(passes),
    )
    owned_workspace, workspace_path = _resolve_workspace(workspace)
    try:
        source = ImageUtil.load_image(source_image)
        box = BoxValues.parse(padding).normalize_to_dimensions(width=source.width, height=source.height)
        OutpaintUtil.validate_padding(box, option_name=option_name)
        whole = resolve_outpaint_fill_plan(contract=resolved, request=request, source=source, padding=box)
        pass_plan = resolve_outpaint_pass_plan(contract=resolved, request=request, fill_plan=whole)
        pass_fill_plans = _resolve_pass_fill_plans(
            contract=resolved, request=request, source=source, pass_plan=pass_plan, whole=whole
        )
        warnings = guard_outpaint_fill_plan(contract=resolved, fill_plan=whole, pass_plan=pass_plan)
        for plan in pass_fill_plans:
            for warning in guard_outpaint_fill_plan(contract=resolved, fill_plan=plan, pass_plan=pass_plan):
                if warning not in warnings:
                    warnings = (*warnings, warning)
        fill_plan = pass_fill_plans[0]
        canvas = OutpaintUtil.create_expanded_canvas(
            source_path=source_image,
            padding_value=pass_plan.padding_values[0],
            output_path=workspace_path / canvas_name,
            dimension_multiple=resolved.dimension_multiple,
            fill_mode=fill_plan.mode,
            fill_color=fill_plan.fill_color or resolved.base_fill_color,
            option_name=option_name,
        )
        canvas = _with_source_lock_mask(contract=resolved, canvas=canvas)
        geometry = OutpaintCanvas(
            canvas_path=canvas.canvas_path,
            source_path=Path(source_image),
            source_width=source.width,
            source_height=source.height,
            target_width=pass_plan.canvas_width,
            target_height=pass_plan.canvas_height,
            paste_left=pass_plan.paste_left,
            paste_top=pass_plan.paste_top,
            padding=box,
        )
    except BaseException:
        if owned_workspace is not None:
            owned_workspace.cleanup()
        raise
    return OutpaintSession(
        source_image=source_image,
        padding=padding,
        canvas=canvas,
        _workspace=owned_workspace,
        contract=resolved,
        fill_plan=fill_plan,
        pass_plan=pass_plan,
        pass_fill_plans=pass_fill_plans,
        request=request,
        geometry=geometry,
        warnings=warnings,
        fill_color_explicit=fill_color_explicit,
    )


def _with_source_lock_mask(*, contract: OutpaintContract, canvas: OutpaintCanvas) -> OutpaintCanvas:
    # The mask lives beside the canvas it locks, named after it, so a kept workspace reads as
    # pairs: outpaint_canvas.png / outpaint_canvas_mask.png, outpaint_pass2_canvas.png / ...
    if not contract.source_lock_mask:
        return canvas
    mask_path = canvas.canvas_path.with_name(f"{canvas.canvas_path.stem}_mask.png")
    return OutpaintUtil.attach_source_lock_mask(canvas=canvas, output_path=mask_path)


def _resolve_pass_fill_plans(
    *,
    contract: OutpaintContract,
    request: OutpaintRequest,
    source: PIL.Image.Image,
    pass_plan: OutpaintPassPlan,
    whole: OutpaintFillPlan,
) -> tuple[OutpaintFillPlan, ...]:
    # One pass: the whole-request plan is the pass plan, byte for byte the pre-split behaviour.
    # Split: each pass resolves its own fill against the source it will actually run on - the
    # original, then the first pass's canvas size - because the edge-fill reach depends on that
    # source's dimensions, not on the original's.
    if not pass_plan.is_split:
        return (whole,)
    plans: list[OutpaintFillPlan] = []
    size = (source.width, source.height)
    for padding in pass_plan.paddings:
        plan = resolve_outpaint_fill_plan(
            contract=contract, request=request, source=source, padding=padding, source_size=size
        )
        plans.append(plan)
        size = (plan.canvas_width, plan.canvas_height)
    return tuple(plans)


def prepare_reframe(
    *,
    source_image: str | Path,
    padding: str,
    dimension_multiple: int = 16,
    workspace: str | Path | None = None,
    canvas_name: str = "reframe_canvas.png",
    option_name: str = "--reframe-padding",
) -> ReframeSession:
    """Build the reframe conditioning canvas. Loads no weights.

    Reframe keeps the historical edge-extended canvas: it has no fill contract, so it needs
    no capability row.
    """
    owned_workspace, workspace_path = _resolve_workspace(workspace)
    try:
        canvas = OutpaintUtil.create_expanded_canvas(
            source_path=source_image,
            padding_value=padding,
            output_path=workspace_path / canvas_name,
            dimension_multiple=dimension_multiple,
            option_name=option_name,
        )
    except BaseException:
        if owned_workspace is not None:
            owned_workspace.cleanup()
        raise
    return ReframeSession(
        source_image=source_image,
        padding=padding,
        canvas=canvas,
        _workspace=owned_workspace,
    )


def _resolve_contract(
    *,
    contract: OutpaintContract | None,
    capability: GenerationCapability | None,
    model: str | None,
    model_config: Any,
    base_model: str | None,
) -> OutpaintContract:
    named = [
        name
        for name, value in (
            ("contract", contract),
            ("capability", capability),
            ("model", model),
            ("model_config", model_config),
        )
        if value is not None
    ]
    if len(named) != 1:
        raise OutpaintError(
            "Pass exactly one of contract=..., capability=..., model=..., or model_config=... to identify the route."
        )
    if contract is not None:
        return contract
    if capability is not None:
        return outpaint_contract(capability=capability)
    return outpaint_contract_for_model(model=model, model_config=model_config, base_model=base_model)


def _resolve_workspace(workspace: str | Path | None) -> tuple[TemporaryDirectory | None, Path]:
    if workspace is not None:
        return None, Path(workspace)
    owned = TemporaryDirectory(prefix="mlxgen-outpaint-")
    return owned, Path(owned.name)


def run_outpaint(
    *,
    loaded: Any,
    source_image: str | Path,
    padding: str,
    seeds: Sequence[int] = (0,),
    output: str | Path | None = None,
    overwrite: bool = False,
    progress_callback: Any = None,
    save_kwargs: dict[str, Any] | None = None,
    fill: str | None = None,
    fill_color: tuple[int, int, int] | None = None,
    lora_paths: Sequence[str] = (),
    requested_lora_paths: Sequence[str] = (),
    passes: str | int | None = None,
    **generate_kwargs: Any,
) -> list[Any]:
    """Outpaint one source image with a loaded runtime, for one or more seeds.

    `loaded` is a `LoadedGenerationModel` from `load_generation_model(..., has_outpaint=True)`:
    its plan names the route, so the fill policy, preservation strategy and canvas keywords
    are all read from that route's capability row. Multi-seed execution, progress events and
    save semantics are the runtime wrapper's; this only adds the outpaint pipeline, which may
    run each seed as more than one pass (see `prepare_outpaint`).
    """
    with prepare_outpaint(
        source_image=source_image,
        padding=padding,
        capability=_capability_for_loaded(loaded),
        fill=fill,
        fill_color=fill_color,
        lora_paths=lora_paths,
        requested_lora_paths=requested_lora_paths,
        passes=passes,
    ) as session:

        def generate(**kwargs: Any):
            return session.generate(loaded.model, **kwargs)

        return loaded.generate_outputs(
            seeds=seeds,
            output=output,
            overwrite=overwrite,
            progress_callback=progress_callback,
            save_kwargs=save_kwargs,
            generate_method=generate,
            **generate_kwargs,
        )


def _capability_for_loaded(loaded: Any) -> GenerationCapability:
    capabilities = get_model_capabilities(
        model=loaded.model_config.model_name,
        model_config=loaded.model_config,
    )
    for capability in capabilities.capabilities:
        if capability.id == loaded.plan.capability_id:
            return capability
    raise OutpaintError(f"No capability row {loaded.plan.capability_id!r} for the loaded runtime.")
