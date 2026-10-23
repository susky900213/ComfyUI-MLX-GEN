"""Directory image batch input/output nodes for ComfyUI-MLX-GEN.

These nodes intentionally use the plugin's internal ``images`` payload instead
of ComfyUI's torch ``IMAGE`` batch.  The internal payload keeps PIL images and
the original input file names together, which lets a folder-to-folder workflow
save each generated image beside its corresponding source name.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ..types import MlxPilImage


IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_NATURAL_SORT_PARTS = re.compile(r"(\d+)")


def _natural_sort_key(path: Path) -> tuple[object, ...]:
    """Sort ``image2`` before ``image10`` while keeping names deterministic."""

    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in _NATURAL_SORT_PARTS.split(path.name)
    )


def _image_paths(directory: str) -> list[Path]:
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ValueError(f"输入目录不存在或不是目录：{root}")
    return sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS),
        key=_natural_sort_key,
    )


def _load_images(directory: str) -> tuple[list[Image.Image], tuple[str, ...]]:
    paths = _image_paths(directory)
    if not paths:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(f"输入目录没有可读取的图片：{Path(directory).expanduser()}（支持：{extensions}）")

    images: list[Image.Image] = []
    names: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as source:
                # Copy while the file handle is open.  Converting to RGBA also
                # makes palette/LA images safe for the downstream Qwen VAE.
                image = source.convert("RGBA").copy()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"无法读取图片：{path}（{exc}）") from exc
        images.append(image)
        names.append(path.name)
    return images, tuple(names)


class MlxLoadImagesFromDirectory:
    """Load every supported image in a directory as one ordered batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_directory": (
                    "STRING",
                    {
                        "default": "<输入图片文件夹的绝对路径>",
                        "multiline": False,
                    },
                )
            }
        }

    RETURN_TYPES = ("images", "STRING", "INT")
    RETURN_NAMES = ("images", "filenames", "count")
    FUNCTION = "load"
    CATEGORY = "MLX/Gen/Image"

    def load(self, input_directory):
        images, filenames = _load_images(input_directory)
        payload = MlxPilImage(images=tuple(images), batch_index=-1, filenames=filenames)
        print(f"[MlxLoadImagesFromDirectory] 已读取 {len(images)} 张图片：{input_directory}")
        return payload, "\n".join(filenames), len(filenames)


def _safe_output_name(filename: str, index: int) -> str:
    """Return a stable PNG name without allowing a path outside the output dir."""

    source_name = Path(filename).name
    stem = Path(source_name).stem or f"image_{index:04d}"
    return f"{index:04d}_{stem}.png"


class MlxSaveImagesToDirectory:
    """Save an internal image batch to a user-selected directory as PNG files."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("images", {}),
                "output_directory": (
                    "STRING",
                    {
                        "default": "<输出图片文件夹的绝对路径>",
                        "multiline": False,
                    },
                ),
                "filenames": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_files",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "MLX/Gen/Image"

    def save(self, images, output_directory, filenames):
        if not images.images:
            raise ValueError("没有图片可保存")
        output = Path(output_directory).expanduser()
        if not str(output).strip() or output.name.startswith("<"):
            raise ValueError("请把输出目录改成实际的绝对路径")
        output.mkdir(parents=True, exist_ok=True)

        source_names = [line.strip() for line in str(filenames).splitlines() if line.strip()]
        if source_names and len(source_names) != len(images.images):
            raise ValueError(
                f"文件名数量（{len(source_names)}）与图片数量（{len(images.images)}）不一致"
            )
        if not source_names:
            source_names = [f"image_{index:04d}.png" for index in range(len(images.images))]

        paths: list[Path] = []
        for index, (image, source_name) in enumerate(zip(images.images, source_names)):
            path = output / _safe_output_name(source_name, index)
            image.save(path, format="PNG")
            paths.append(path)

        saved = "\n".join(str(path) for path in paths)
        print(f"[MlxSaveImagesToDirectory] 已保存 {len(paths)} 张图片：{output}")
        return (saved,)