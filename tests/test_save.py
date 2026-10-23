from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from comfyui_mlx_gen.nodes.save import MlxSaveImage, save_images
from comfyui_mlx_gen.types import MlxPilImage


def _images(count: int = 2) -> tuple[Image.Image, ...]:
    return tuple(Image.new("RGBA", (8, 8), (index * 40, 20, 10, 255)) for index in range(count))


def test_save_node_defaults_to_no_filename_prefix():
    inputs = MlxSaveImage.INPUT_TYPES()["required"]
    assert inputs["filename_prefix"][1]["default"] == ""


def test_blank_prefix_uses_source_names_in_selected_directory(tmp_path: Path):
    payload = MlxPilImage(
        images=_images(),
        filenames=("input/first.jpg", "second.webp"),
    )

    result = MlxSaveImage().save(
        payload,
        filename_prefix="",
        capture="<auto>",
        output_directory=str(tmp_path / "selected"),
    )

    saved = [Path(path) for path in result[0].split(", ")] if result[0] != "DROP" else []
    assert (tmp_path / "selected" / "first.png").is_file()
    assert (tmp_path / "selected" / "second.png").is_file()
    assert not list((tmp_path / "selected").glob("*.json"))
    assert saved == []  # OUTPUT_NODE keeps the historical DROP return value.


def test_blank_prefix_falls_back_and_disambiguates_duplicate_names(tmp_path: Path):
    result = save_images(
        uploads_dir=str(tmp_path / "unused"),
        sub="ignored",
        custom="",
        images=_images(),
        metadata={},
        input_info=(),
        size=(8, 8),
        type_="png",
        output_directory=str(tmp_path / "selected"),
        source_filenames=("same.jpg", "same.png"),
    )

    assert result.split(", ") == [
        str(tmp_path / "selected" / "same.png"),
        str(tmp_path / "selected" / "same_1.png"),
    ]


def test_auto_capture_keeps_metadata_json_behavior(tmp_path: Path):
    result = save_images(
        uploads_dir=str(tmp_path),
        sub="auto",
        custom="<auto>",
        images=_images(1),
        metadata={"source": "test"},
        input_info=(),
        size=(8, 8),
        type_="png",
    )

    image_path = Path(result)
    assert image_path.is_file()
    metadata_path = image_path.with_suffix(".json")
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["source"] == "test"