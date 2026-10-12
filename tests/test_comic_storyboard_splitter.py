from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from comfyui_mlx_gen.nodes.comic_storyboard_splitter import (
    MlxComicStoryboardSplitter,
    detect_auto_panels,
    split_storyboard,
)


def _storyboard(rows: int = 2, columns: int = 3, gutter: int = 12) -> Image.Image:
    width, height = 600, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = [
        (230, 60, 60),
        (60, 180, 80),
        (60, 90, 220),
        (220, 170, 40),
        (170, 70, 190),
        (40, 180, 180),
    ]
    for row in range(rows):
        for column in range(columns):
            x0 = column * (width // columns) + gutter // 2
            x1 = (column + 1) * (width // columns) - gutter // 2
            y0 = row * (height // rows) + gutter // 2
            y1 = (row + 1) * (height // rows) - gutter // 2
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=colors[row * columns + column])
    return image


def test_auto_detects_panel_count_without_panel_count_input():
    image = _storyboard()

    boxes = detect_auto_panels(
        image,
        separator_width=12,
        min_panel_width=80,
        min_panel_height=80,
    )

    assert len(boxes) == 6
    assert boxes == sorted(boxes, key=lambda box: (box[1], box[0]))


def test_auto_detects_flat_color_horizontal_and_vertical_layouts():
    horizontal = _storyboard(rows=1, columns=3)
    vertical = _storyboard(rows=3, columns=1)

    assert len(
        detect_auto_panels(
            horizontal,
            separator_width=12,
            min_panel_width=80,
            min_panel_height=80,
        )
    ) == 3
    assert len(
        detect_auto_panels(
            vertical,
            separator_width=12,
            min_panel_width=80,
            min_panel_height=80,
        )
    ) == 3


def test_split_outputs_uniform_images_masks_and_report():
    panels, masks, metadata = split_storyboard(
        _storyboard(),
        mode="auto",
        separator_width=12,
        min_panel_width=80,
        min_panel_height=80,
        output_width=256,
        output_height=256,
        fit_mode="aspect_fit_pad",
    )

    assert len(panels) == len(masks) == len(metadata) == 6
    assert {panel.size for panel in panels} == {(256, 256)}
    assert {mask.size for mask in masks} == {(256, 256)}
    assert all(np.asarray(mask).max() == 255 for mask in masks)
    assert all(np.asarray(mask).min() == 0 for mask in masks)
    assert [item["index"] for item in metadata] == list(range(6))
    assert all(item["detection"] == "auto" for item in metadata)


def test_grid_mode_is_deterministic_fallback_and_node_returns_comfy_batches():
    image = _storyboard(rows=2, columns=2)
    node = MlxComicStoryboardSplitter()
    input_batch = np.asarray(image, dtype=np.float32)[None, ...] / 255.0

    panels, masks, report = node.split(
        input_batch,
        split_mode="grid",
        grid_rows=2,
        grid_columns=2,
        separator_width=12,
        output_width=128,
        output_height=96,
        fit_mode="stretch",
    )

    assert tuple(panels.shape) == (4, 96, 128, 3)
    assert tuple(masks.shape) == (4, 96, 128)
    parsed = json.loads(report)
    assert parsed["total_panel_count"] == 4
    assert parsed["sources"][0]["panel_count"] == 4


def test_node_normalizes_automatic_canvas_across_source_batch():
    first = _storyboard(rows=2, columns=2)
    second = _storyboard(rows=1, columns=3)
    node = MlxComicStoryboardSplitter()
    input_batch = np.stack(
        [np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)],
        axis=0,
    ) / 255.0

    panels, masks, report = node.split(
        input_batch,
        split_mode="auto",
        separator_width=12,
        min_panel_width=80,
        min_panel_height=80,
        fit_mode="aspect_fit_pad",
    )

    assert tuple(panels.shape) == (7, 420, 294, 3)
    assert tuple(masks.shape) == (7, 420, 294)
    assert json.loads(report)["total_panel_count"] == 7
