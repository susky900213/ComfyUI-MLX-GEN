from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from comfyui_mlx_gen import NODE_CLASS_MAPPINGS
from comfyui_mlx_gen.nodes.html_scramble_downloader import (
    MlxDownloadScrambledHtmlImages,
    _compose_canvas,
    _resolve_picker_path,
    _reverse_strips,
)


def _strip_spec(
    path: Path,
    width: int,
    height: int,
    rows: int = 4,
    reference_name: str | None = None,
) -> str:
    args = []
    strip_height = height // rows
    for destination_row in range(rows):
        source_row = rows - 1 - destination_row
        args.append(
            [
                0,
                source_row * strip_height,
                width,
                strip_height,
                0,
                destination_row * strip_height,
                width,
                strip_height,
            ]
        )
    reference = path.with_name(reference_name) if reference_name else path
    payload = {"url": reference.resolve().as_uri(), "args": args, "width": width, "height": height}
    return base64.b64encode(json.dumps(payload).encode()).decode()


def _rows(colors: list[tuple[int, int, int]]) -> Image.Image:
    image = Image.new("RGB", (4, len(colors) * 2))
    for row, color in enumerate(colors):
        for y in range(row * 2, row * 2 + 2):
            for x in range(image.width):
                image.putpixel((x, y), color)
    return image


def test_canvas_args_restore_horizontal_strips_in_expected_order(tmp_path):
    original = _rows([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)])
    scrambled = Image.new("RGB", original.size)
    for destination_row in range(4):
        source_row = 3 - destination_row
        scrambled.crop((0, source_row * 2, 4, source_row * 2 + 2)).save(tmp_path / "piece.png")
        scrambled.paste(original.crop((0, source_row * 2, 4, source_row * 2 + 2)), (0, destination_row * 2))

    restored = _compose_canvas(
        scrambled.convert("RGBA"),
        json.loads(
            base64.b64decode(
                _strip_spec(tmp_path / "piece.png", 4, 8),
            )
        ),
    )
    assert restored.convert("RGB").tobytes() == original.tobytes()


def test_node_matches_local_images_orders_restores_and_saves_html_pages(tmp_path):
    first = _rows([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)])
    second = _rows([(10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120)])
    image_directory = tmp_path / "downloaded"
    image_directory.mkdir()
    sources = []
    for index, image in enumerate((first, second), start=1):
        source = image_directory / f"imgi_{index}_0000{index}.webp"
        scrambled = Image.new("RGB", image.size)
        for destination_row in range(4):
            source_row = 3 - destination_row
            scrambled.paste(image.crop((0, source_row * 2, 4, source_row * 2 + 2)), (0, destination_row * 2))
        scrambled.save(source, lossless=True)
        sources.append(source)

    # Deliberately put page 2 before page 1 in the HTML.  data-page must win.
    html = tmp_path / "album.html"
    html.write_text(
        """<html><head><link rel="canonical" href="https://example.test/photo/1"></head><body>
        <div class="other center"><img src="ignored.png"></div>
        <div class="center scramble-page spnotice_chk" id="00002.webp">
          <img data-page="1" data-original="ignored-2.png">
          <canvas data="{spec2}"></canvas>
        </div>
        <div class="center scramble-page spnotice_chk" id="00001.webp">
          <img data-page="0" data-original="ignored-1.png">
          <canvas data="{spec1}"></canvas>
        </div>
        </body></html>""".format(
            spec1=_strip_spec(sources[0], 4, 8, reference_name="00001.webp"),
            spec2=_strip_spec(sources[1], 4, 8, reference_name="00002.webp"),
        ),
        encoding="utf-8",
    )

    result = MlxDownloadScrambledHtmlImages().download(
        str(image_directory),
        str(html),
        10,
        True,
        False,
    )
    payload, saved_files, report, success_count = result

    assert NODE_CLASS_MAPPINGS["MlxDownloadScrambledHtmlImages"] is MlxDownloadScrambledHtmlImages
    assert success_count == 2
    assert [Path(path).name for path in saved_files.splitlines()] == [
        "0000_00001.png",
        "0001_00002.png",
    ]
    assert [image.convert("RGB").tobytes() for image in payload.images] == [
        first.tobytes(),
        second.tobytes(),
    ]
    records = json.loads(report)
    assert [record["page"] for record in records] == [0, 1]
    assert [record["mode"] for record in records] == ["canvas_args", "canvas_args"]
    assert all(Path(path).is_file() for path in saved_files.splitlines())
    assert all(Path(path).parent == image_directory / "image" for path in saved_files.splitlines())


def test_node_reports_unmatched_local_image_and_can_continue(tmp_path):
    image_directory = tmp_path / "downloaded"
    image_directory.mkdir()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(image_directory / "00001.webp")
    html = tmp_path / "album.html"
    html.write_text(
        """<div class="center scramble-page spnotice_chk" id="00001.webp">
          <img src="https://cdn.example/00001.webp">
        </div>
        <div class="center scramble-page spnotice_chk" id="00002.webp">
          <img src="https://cdn.example/00002.webp">
        </div>""",
        encoding="utf-8",
    )

    payload, saved_files, report, success_count = MlxDownloadScrambledHtmlImages().download(
        str(image_directory),
        str(html),
        0,
        True,
        True,
    )

    assert len(payload.images) == 1
    assert success_count == 1
    assert [Path(path).name for path in saved_files.splitlines()] == ["0000_00001.png"]
    records = json.loads(report)
    assert [record["status"] for record in records] == ["success", "error"]
    assert "00002.webp" in records[1]["error"]
    assert (image_directory / "image").is_dir()


def test_reverse_strip_fallback_can_restore_snapshot_without_canvas_data():
    original = _rows([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)])
    scrambled = Image.new("RGB", original.size)
    for destination_row in range(4):
        source_row = 3 - destination_row
        scrambled.paste(original.crop((0, source_row * 2, 4, source_row * 2 + 2)), (0, destination_row * 2))

    restored = _reverse_strips(scrambled.convert("RGBA"), 4)
    assert restored.convert("RGB").tobytes() == original.tobytes()


def test_picker_tokens_resolve_inside_comfyui_directories(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()

    def fake_comfy_directory(name):
        return {"input": input_root, "output": output_root}[name]

    monkeypatch.setattr(
        "comfyui_mlx_gen.nodes.html_scramble_downloader._comfy_directory",
        fake_comfy_directory,
    )

    assert _resolve_picker_path(
        "__comfy_input__/mlx_html_scramble/album",
        "__comfy_input__/",
        "input",
        "图片目录",
    ) == input_root / "mlx_html_scramble/album"
    manual_path = tmp_path / "manual"
    assert _resolve_picker_path(str(manual_path), "__comfy_input__/", "input", "图片目录") == manual_path


def test_picker_tokens_reject_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "comfyui_mlx_gen.nodes.html_scramble_downloader._comfy_directory",
        lambda name: tmp_path / name,
    )

    with pytest.raises(ValueError, match=r"不能使用绝对路径或 \.\."):
        _resolve_picker_path("__comfy_input__/../outside", "__comfy_input__/", "input", "图片目录")