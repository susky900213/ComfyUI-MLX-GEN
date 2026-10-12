"""Comic storyboard panel detection and splitting.

This node deliberately does not load a vision model.  Panel boundaries in a
storyboard are normally expressed by a gutter or a border that runs across a
large part of the image.  Detecting those lines locally with PIL/NumPy is
cheap, deterministic, and keeps the node independent from the Qwen sampler.
The resulting batch can then be connected to the existing Qwen-Image 2.1
reference-image workflow.
"""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np
from PIL import Image

from .. import image as image_mod


Rect = tuple[int, int, int, int]  # left, top, right, bottom; right/bottom exclusive

MODES = ("auto", "grid", "horizontal", "vertical")
ORDERINGS = ("reading", "column_major")
FIT_MODES = ("pad", "aspect_fit_pad", "stretch")
PADDING_COLORS = ("white", "black", "gray")


def _validate_rect(rect: Rect, width: int, height: int) -> None:
    left, top, right, bottom = rect
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"无效的面板区域 {rect}（源图尺寸 {width}×{height}）")


def _crop_area(width: int, height: int, margin: int) -> Rect:
    margin = int(margin)
    if margin < 0:
        raise ValueError(f"margin 不能为负数：{margin}")
    if 2 * margin >= width or 2 * margin >= height:
        raise ValueError(
            f"margin={margin} 太大，源图尺寸为 {width}×{height}；每边至少要留 1 像素"
        )
    return (margin, margin, width - margin, height - margin)


def _axis_signal(
    rgb: np.ndarray,
    rect: Rect,
    axis: str,
    *,
    separator_width: int,
) -> np.ndarray:
    """Return a separator-likelihood signal for each row or column.

    White gutters are detected by their similarity to white, black borders by
    their similarity to black.  A low transverse gradient/uniform stripe is a
    useful fallback for gray or colored gutters.  Combining these signals is
    intentionally conservative: a single empty row inside a panel should not
    normally be enough to create a new panel unless it is a long, uniform band.
    """

    left, top, right, bottom = rect
    region = rgb[top:bottom, left:right]
    if region.size == 0:
        return np.empty((0,), dtype=np.float32)
    gray = np.asarray(region, dtype=np.float32).mean(axis=2)

    if axis == "x":
        line_length = gray.shape[1]
        transverse_diff = (
            np.abs(np.diff(gray, axis=0)).mean(axis=0)
            if gray.shape[0] > 1
            else np.zeros((line_length,), dtype=np.float32)
        )
    elif axis == "y":
        line_length = gray.shape[0]
        transverse_diff = (
            np.abs(np.diff(gray, axis=1)).mean(axis=1)
            if gray.shape[1] > 1
            else np.zeros((line_length,), dtype=np.float32)
        )
    else:
        raise ValueError(f"未知的投影方向：{axis}")

    # A real separator is usually either close to white/black, or a uniform
    # strip whose color is different from both neighboring panel areas.  The
    # neighbor comparison is important: otherwise a completely flat-color
    # panel would make every line look like a separator.
    if axis == "x":
        line_std = gray.std(axis=0)
    else:
        line_std = gray.std(axis=1)
    light_fraction = (gray >= 235.0).mean(axis=0 if axis == "x" else 1)
    dark_fraction = (gray <= 32.0).mean(axis=0 if axis == "x" else 1)

    if axis == "x":
        line_mean = region.mean(axis=0)
    else:
        line_mean = region.mean(axis=1)
    distance = max(2, int(separator_width) // 2)
    left_sample = np.roll(line_mean, distance, axis=0)
    right_sample = np.roll(line_mean, -distance, axis=0)
    # Ignore wrapped samples at the image edges; those coordinates are already
    # excluded by _separator_spans, but setting them to the current line keeps
    # their scores from leaking into neighboring candidates.
    left_sample[:distance] = line_mean[:distance]
    right_sample[-distance:] = line_mean[-distance:]
    neighbor_contrast = np.minimum(
        np.linalg.norm(line_mean - left_sample, axis=1),
        np.linalg.norm(line_mean - right_sample, axis=1),
    ) / (255.0 * np.sqrt(3.0))

    # Normalize transverse texture against this region rather than using an
    # absolute threshold; scans and screenshots can have very different noise.
    diff_scale = max(3.0, float(np.percentile(transverse_diff, 75)))
    low_texture = 1.0 - np.clip(transverse_diff / (diff_scale * 1.8), 0.0, 1.0)
    uniform = 1.0 - np.clip(line_std / 85.0, 0.0, 1.0)
    neutral = np.maximum(light_fraction, dark_fraction)
    neutral_signal = neutral * (0.55 + 0.45 * uniform)
    uniform_signal = neighbor_contrast * (0.55 + 0.45 * uniform) * (0.65 + 0.35 * low_texture)
    return np.maximum(neutral_signal, uniform_signal * 0.78).astype(np.float32)


def _runs_above(signal: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Convert a one-dimensional signal into inclusive-exclusive runs."""

    if signal.size == 0:
        return []
    good = signal >= float(threshold)
    runs: list[tuple[int, int, float]] = []
    start: int | None = None
    for index, is_good in enumerate(np.r_[good, False]):
        if is_good and start is None:
            start = index
        elif not is_good and start is not None:
            end = index
            runs.append((start, end, float(signal[start:end].mean())))
            start = None
    return runs


def _separator_spans(
    rgb: np.ndarray,
    rect: Rect,
    axis: str,
    *,
    separator_width: int,
    sensitivity: float,
) -> list[tuple[int, int, float]]:
    """Find plausible separator spans in ``rect`` along one axis."""

    signal = _axis_signal(rgb, rect, axis, separator_width=separator_width)
    length = int(signal.size)
    if length < 3:
        return []
    # Higher sensitivity lowers the threshold, but never enough to accept
    # arbitrary texture as a separator.  The UI range is clamped again here so
    # direct Python callers get the same behavior.
    sensitivity = float(np.clip(sensitivity, 0.2, 0.95))
    threshold = 0.89 - 0.43 * sensitivity
    runs = _runs_above(signal, threshold)
    output: list[tuple[int, int, float]] = []
    max_separator = max(2, min(length // 4, max(12, int(separator_width) * 8)))
    for start, end, score in runs:
        # Do not use the blank canvas margin as a separator.  Large blank
        # areas are also not gutters and would create many false panels.
        if start <= 1 or end >= length - 1:
            continue
        if end - start > max_separator:
            continue
        center = (start + end) // 2
        half_width = max(0, (int(separator_width) - (end - start) + 1) // 2)
        start = max(1, center - max(1, (end - start) // 2) - half_width)
        end = min(length - 1, center + (end - start + 1) // 2 + half_width)
        if end > start:
            output.append((start, end, score))
    return output


def _best_split(
    rgb: np.ndarray,
    rect: Rect,
    *,
    separator_width: int,
    min_panel_width: int,
    min_panel_height: int,
    sensitivity: float,
) -> tuple[str, int, int, float] | None:
    """Choose the strongest valid split of a rectangle.

    The returned coordinate is relative to the rectangle's left/top edge and
    spans the separator to remove.  Splitting one axis at a time lets the same
    code handle regular grids and irregular comic layouts.
    """

    left, top, right, bottom = rect
    width, height = right - left, bottom - top
    candidates: list[tuple[str, int, int, float]] = []
    for axis, length, minimum in (
        ("x", width, int(min_panel_width)),
        ("y", height, int(min_panel_height)),
    ):
        for start, end, score in _separator_spans(
            rgb,
            rect,
            axis,
            separator_width=separator_width,
            sensitivity=sensitivity,
        ):
            if start < minimum or length - end < minimum:
                continue
            # Very unbalanced fragments are often an empty speech-balloon gap,
            # not a panel boundary.  Keep a small tolerance for title strips.
            smaller = min(start, length - end)
            if smaller < max(minimum, int(length * 0.08)):
                continue
            candidates.append((axis, start, end, score))
    if not candidates:
        return None
    # Prefer a strong separator, then prefer the split that leaves the more
    # balanced pair.  This avoids choosing a tiny accidental line first.
    return max(
        candidates,
        key=lambda item: (item[3], min(item[1], (width if item[0] == "x" else height) - item[2])),
    )


def detect_auto_panels(
    pil: Image.Image,
    *,
    margin: int = 0,
    separator_width: int = 8,
    min_panel_width: int = 64,
    min_panel_height: int = 64,
    sensitivity: float = 0.65,
    max_panels: int = 64,
) -> list[Rect]:
    """Detect panel rectangles without a fixed row/column or panel count."""

    source = pil.convert("RGB")
    width, height = source.size
    root = _crop_area(width, height, margin)
    rgb = np.asarray(source, dtype=np.uint8)
    minimum_width = max(2, int(min_panel_width))
    minimum_height = max(2, int(min_panel_height))
    separator_width = max(0, int(separator_width))
    max_panels = max(1, int(max_panels))
    leaves: list[Rect] = []
    stack = [root]
    while stack:
        rect = stack.pop()
        split = _best_split(
            rgb,
            rect,
            separator_width=separator_width,
            min_panel_width=minimum_width,
            min_panel_height=minimum_height,
            sensitivity=sensitivity,
        )
        if split is None or len(leaves) + len(stack) + 2 > max_panels:
            leaves.append(rect)
            continue
        axis, start, end, _score = split
        left, top, right, bottom = rect
        if axis == "x":
            children = ((left, top, left + start, bottom), (left + end, top, right, bottom))
        else:
            children = ((left, top, right, top + start), (left, top + end, right, bottom))
        # Push right/bottom first so a later reading-order sort is not affected
        # by stack traversal order.
        stack.extend(children[::-1])
    return leaves


def grid_panels(
    width: int,
    height: int,
    *,
    rows: int,
    columns: int,
    margin: int = 0,
    separator_width: int = 8,
) -> list[Rect]:
    """Create regular grid rectangles; used as an explicit fallback mode."""

    area = _crop_area(width, height, margin)
    rows, columns = int(rows), int(columns)
    if rows < 1 or columns < 1:
        raise ValueError(f"grid_rows/grid_columns 必须至少为 1：{rows}×{columns}")
    left, top, right, bottom = area
    x_edges = [left + round((right - left) * i / columns) for i in range(columns + 1)]
    y_edges = [top + round((bottom - top) * i / rows) for i in range(rows + 1)]
    half = max(0, int(separator_width) // 2)
    rects: list[Rect] = []
    for row in range(rows):
        for column in range(columns):
            x0 = x_edges[column] + (half if column else 0)
            x1 = x_edges[column + 1] - (half if column < columns - 1 else 0)
            y0 = y_edges[row] + (half if row else 0)
            y1 = y_edges[row + 1] - (half if row < rows - 1 else 0)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("grid 参数与 separator_width 冲突，无法生成有效面板")
            rects.append((x0, y0, x1, y1))
    return rects


def _sort_rects(rects: Iterable[Rect], ordering: str) -> list[Rect]:
    if ordering == "reading":
        return sorted(rects, key=lambda rect: (rect[1], rect[0]))
    if ordering == "column_major":
        return sorted(rects, key=lambda rect: (rect[0], rect[1]))
    raise ValueError(f"未知的 ordering：{ordering}")


def _padding_rgb(color: str) -> tuple[int, int, int]:
    values = {"white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128)}
    try:
        return values[color]
    except KeyError as exc:
        raise ValueError(f"未知的 padding_color：{color}") from exc


def _fit_panel(
    panel: Image.Image,
    output_size: tuple[int, int],
    *,
    fit_mode: str,
    padding_color: tuple[int, int, int],
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Return a uniform RGB canvas and its valid image rectangle."""

    output_width, output_height = output_size
    if output_width < 1 or output_height < 1:
        raise ValueError(f"输出尺寸必须为正数：{output_size}")
    if fit_mode == "stretch":
        resized = panel.resize(output_size, Image.Resampling.LANCZOS)
        return resized, (0, 0, output_width, output_height)
    if fit_mode == "aspect_fit_pad":
        ratio = min(output_width / panel.width, output_height / panel.height)
        size = (max(1, round(panel.width * ratio)), max(1, round(panel.height * ratio)))
        resized = panel if size == panel.size else panel.resize(size, Image.Resampling.LANCZOS)
    elif fit_mode == "pad":
        if panel.width > output_width or panel.height > output_height:
            raise ValueError(
                f"面板 {panel.width}×{panel.height} 大于输出画布 {output_width}×{output_height}；"
                "请增大 output_width/output_height，或将 fit_mode 改为 aspect_fit_pad"
            )
        resized = panel
    else:
        raise ValueError(f"未知的 fit_mode：{fit_mode}")
    canvas = Image.new("RGB", output_size, padding_color)
    x = (output_width - resized.width) // 2
    y = (output_height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas, (x, y, x + resized.width, y + resized.height)


def split_storyboard(
    pil: Image.Image,
    *,
    source_index: int = 0,
    mode: str = "auto",
    grid_rows: int = 2,
    grid_columns: int = 2,
    margin: int = 0,
    separator_width: int = 8,
    expansion_pixels: int = 0,
    min_panel_width: int = 64,
    min_panel_height: int = 64,
    sensitivity: float = 0.65,
    ordering: str = "reading",
    max_panels: int = 64,
    output_width: int = 0,
    output_height: int = 0,
    fit_mode: str = "pad",
    padding_color: str = "white",
) -> tuple[list[Image.Image], list[Image.Image], list[dict]]:
    """Split one storyboard image and return panels, masks, and metadata."""

    source = pil.convert("RGB")
    width, height = source.size
    mode = str(mode)
    if mode not in MODES:
        raise ValueError(f"未知的 split_mode：{mode}")
    if mode == "auto":
        rects = detect_auto_panels(
            source,
            margin=margin,
            separator_width=separator_width,
            min_panel_width=min_panel_width,
            min_panel_height=min_panel_height,
            sensitivity=sensitivity,
            max_panels=max_panels,
        )
        detection = "auto"
    elif mode == "grid":
        rects = grid_panels(
            width,
            height,
            rows=grid_rows,
            columns=grid_columns,
            margin=margin,
            separator_width=separator_width,
        )
        detection = "grid"
    elif mode == "horizontal":
        rects = grid_panels(
            width,
            height,
            rows=1,
            columns=grid_columns,
            margin=margin,
            separator_width=separator_width,
        )
        detection = "horizontal"
    else:
        rects = grid_panels(
            width,
            height,
            rows=grid_rows,
            columns=1,
            margin=margin,
            separator_width=separator_width,
        )
        detection = "vertical"
    rects = _sort_rects(rects, ordering)

    expansion_pixels = max(0, int(expansion_pixels))
    expanded: list[tuple[Rect, Rect]] = []
    area = _crop_area(width, height, margin)
    area_left, area_top, area_right, area_bottom = area
    for rect in rects:
        left, top, right, bottom = rect
        crop_rect = (
            max(area_left, left - expansion_pixels),
            max(area_top, top - expansion_pixels),
            min(area_right, right + expansion_pixels),
            min(area_bottom, bottom + expansion_pixels),
        )
        _validate_rect(crop_rect, width, height)
        expanded.append((rect, crop_rect))

    if not expanded:
        raise ValueError("没有检测到有效面板")
    target_width = int(output_width) or max(right - left for _rect, (left, top, right, bottom) in expanded)
    target_height = int(output_height) or max(bottom - top for _rect, (left, top, right, bottom) in expanded)
    pad_color = _padding_rgb(padding_color)

    panels: list[Image.Image] = []
    masks: list[Image.Image] = []
    metadata: list[dict] = []
    for index, (base_rect, crop_rect) in enumerate(expanded):
        left, top, right, bottom = crop_rect
        crop = source.crop(crop_rect)
        panel, valid = _fit_panel(
            crop,
            (target_width, target_height),
            fit_mode=fit_mode,
            padding_color=pad_color,
        )
        mask = Image.new("L", (target_width, target_height), 0)
        mask_draw = Image.new("L", (valid[2] - valid[0], valid[3] - valid[1]), 255)
        mask.paste(mask_draw, (valid[0], valid[1]))
        panels.append(panel)
        masks.append(mask)
        metadata.append(
            {
                "index": index,
                "source_index": int(source_index),
                "box": [int(value) for value in base_rect],
                "crop_box": [int(value) for value in crop_rect],
                "output_valid_box": [int(value) for value in valid],
                "output_size": [target_width, target_height],
                "detection": detection,
            }
        )
    return panels, masks, metadata


class MlxComicStoryboardSplitter:
    """自动识别漫画分镜面板并输出统一尺寸的 IMAGE batch。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "split_mode": (list(MODES), {"default": "auto"}),
                "grid_rows": ("INT", {"default": 2, "min": 1, "max": 32}),
                "grid_columns": ("INT", {"default": 2, "min": 1, "max": 32}),
                "margin": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "separator_width": ("INT", {"default": 8, "min": 0, "max": 256}),
                "expansion_pixels": ("INT", {"default": 0, "min": 0, "max": 256}),
                "min_panel_width": ("INT", {"default": 64, "min": 2, "max": 4096}),
                "min_panel_height": ("INT", {"default": 64, "min": 2, "max": 4096}),
                "auto_sensitivity": ("FLOAT", {"default": 0.65, "min": 0.2, "max": 0.95, "step": 0.05}),
                "ordering": (list(ORDERINGS), {"default": "reading"}),
                "max_panels": ("INT", {"default": 64, "min": 1, "max": 256}),
                "output_width": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "output_height": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "fit_mode": (list(FIT_MODES), {"default": "pad"}),
                "padding_color": (list(PADDING_COLORS), {"default": "white"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("panels", "panel_masks", "panel_report")
    FUNCTION = "split"
    CATEGORY = "MLX/Gen/Image"

    def split(
        self,
        images,
        split_mode="auto",
        grid_rows=2,
        grid_columns=2,
        margin=0,
        separator_width=8,
        expansion_pixels=0,
        min_panel_width=64,
        min_panel_height=64,
        auto_sensitivity=0.65,
        ordering="reading",
        max_panels=64,
        output_width=0,
        output_height=0,
        fit_mode="pad",
        padding_color="white",
    ):
        source_images = image_mod.to_pil_batch(images)
        target_width = int(output_width)
        target_height = int(output_height)
        # With an input batch, each source can contain a different layout.  A
        # per-source automatic canvas would make the final ComfyUI batch
        # ragged, so probe the crop boxes first and choose one global canvas.
        if len(source_images) > 1 and (target_width <= 0 or target_height <= 0):
            probe_reports: list[list[dict]] = []
            for source_index, source in enumerate(source_images):
                _panels, _masks, metadata = split_storyboard(
                    source,
                    source_index=source_index,
                    mode=split_mode,
                    grid_rows=grid_rows,
                    grid_columns=grid_columns,
                    margin=margin,
                    separator_width=separator_width,
                    expansion_pixels=expansion_pixels,
                    min_panel_width=min_panel_width,
                    min_panel_height=min_panel_height,
                    sensitivity=auto_sensitivity,
                    max_panels=max_panels,
                    ordering=ordering,
                    fit_mode=fit_mode,
                    padding_color=padding_color,
                )
                probe_reports.append(metadata)
            if target_width <= 0:
                target_width = max(
                    item["crop_box"][2] - item["crop_box"][0]
                    for metadata in probe_reports
                    for item in metadata
                )
            if target_height <= 0:
                target_height = max(
                    item["crop_box"][3] - item["crop_box"][1]
                    for metadata in probe_reports
                    for item in metadata
                )
        all_panels: list[Image.Image] = []
        all_masks: list[Image.Image] = []
        reports: list[dict] = []
        for source_index, source in enumerate(source_images):
            panels, masks, metadata = split_storyboard(
                source,
                source_index=source_index,
                mode=split_mode,
                grid_rows=grid_rows,
                grid_columns=grid_columns,
                margin=margin,
                separator_width=separator_width,
                expansion_pixels=expansion_pixels,
                min_panel_width=min_panel_width,
                min_panel_height=min_panel_height,
                sensitivity=auto_sensitivity,
                max_panels=max_panels,
                ordering=ordering,
                output_width=target_width,
                output_height=target_height,
                fit_mode=fit_mode,
                padding_color=padding_color,
            )
            all_panels.extend(panels)
            all_masks.extend(masks)
            reports.append(
                {
                    "source_index": source_index,
                    "source_size": [source.width, source.height],
                    "panel_count": len(panels),
                    "panels": metadata,
                }
            )
            print(
                f"[MlxComicStoryboardSplitter] source={source_index} "
                f"{source.width}×{source.height} → {len(panels)} 个分镜"
            )
        if not all_panels:
            raise ValueError("没有生成面板")
        report = json.dumps(
            {
                "node": "MlxComicStoryboardSplitter",
                "split_mode": split_mode,
                "sources": reports,
                "total_panel_count": len(all_panels),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return (image_mod.to_image_batch(all_panels), image_mod.to_mask_batch(all_masks), report)
