"""MlxSaveImage：把 MlxPilImage 写到磁盘（PNG + 可选同名 metadata JSON）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Sequence


def auto_filename(prefix: str, images, metadata: dict, size: tuple, seed: int) -> tuple[Path, dict]:
    """按前缀 + 自增序号生成路径与元数据（相同配置命中已有文件时沿用）。"""
    base = Path(prefix)
    if base.is_file():
        return (base, metadata)
    for i in range(100):
        path = base.parent / f"{base.stem}_{i}{base.suffix or '.png'}"
        if not path.exists():
            meta = {**metadata, "seed": seed, "size": list(size)}
            return (path, meta)
    raise RuntimeError(f"自增序号用尽: {prefix}")


def _source_filename(filename: str, index: int) -> str:
    """把源文件名转换为输出目录内安全、稳定的 PNG 文件名。"""
    source = Path(str(filename).strip().replace("\\", "/")).name
    stem = Path(source).stem or f"image_{index:04d}"
    return f"{stem}.png"


def _unique_paths(base: Path, names: Sequence[str]) -> list[Path]:
    """为批量图片生成不互相覆盖的路径。"""
    paths: list[Path] = []
    used: set[Path] = set()
    for index, name in enumerate(names):
        candidate = base / name
        if candidate in used:
            stem = candidate.stem
            suffix = candidate.suffix or ".png"
            candidate = base / f"{stem}_{index}{suffix}"
            counter = index
            while candidate in used:
                counter += 1
                candidate = base / f"{stem}_{counter}{suffix}"
        used.add(candidate)
        paths.append(candidate)
    return paths


def save_images(
    uploads_dir: str,
    sub: str,
    custom: str,
    images,
    metadata: dict,
    input_info: tuple,
    size: tuple,
    type_: str,
    *,
    output_directory: str = "",
    source_filenames: Sequence[str] = (),
) -> str:
    """自己实现的 save_images（行为与 ComfyUI 相近，不复制其代码）。

    - ``output_directory`` 非空时直接写入该目录；为空时写入
      ``uploads/MlxSaveImage/<sub>``。
    - ``custom == "<auto>"`` 时用时间戳；否则用 ``custom`` 作为文件名前缀。
    - ``custom == ""`` 时使用 ``source_filenames`` 的源文件名；没有源文件名时
      回退为 ``image_0000.png`` 等名称。
    - ``custom == "<auto>"`` 时同基名写 metadata JSON。
    """
    if output_directory.strip():
        base = Path(output_directory).expanduser()
    else:
        base = Path(uploads_dir) / "MlxSaveImage" / sub
    base.mkdir(parents=True, exist_ok=True)

    if custom == "<auto>":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = [base / f"{stamp}_{i}.png" for i in range(len(images))]
        for path, img in zip(paths, images):
            img.save(path, format="PNG")
            meta_path = path.with_suffix(".json")
            payload = {**metadata, "input_info": list(input_info), "size": list(size), "type": type_}
            meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif custom:
        paths = [base / f"{custom}_{i}.png" for i in range(len(images))]
        for path, img in zip(paths, images):
            img.save(path, format="PNG")
    else:
        names = [
            _source_filename(source_filenames[index], index)
            if index < len(source_filenames) and str(source_filenames[index]).strip()
            else f"image_{index:04d}.png"
            for index in range(len(images))
        ]
        paths = _unique_paths(base, names)
        for path, img in zip(paths, images):
            img.save(path, format="PNG")
    return ", ".join(str(p) for p in paths)


class MlxSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("images", {}),
                "filename_prefix": ("STRING", {"default": ""}),
                "capture": ("STRING", {"default": "<auto>"}),
            },
            "optional": {
                "output_directory": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可选的保存文件夹绝对路径；留空使用插件 output/MlxSaveImage/",
                    },
                ),
                "source_filenames": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "可选：每行一个源文件名；留空时自动读取 images 载荷中的源文件名",
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "MLX/Gen"

    def save(self, images, filename_prefix, capture, output_directory="", source_filenames=""):
        if not images.images:
            raise ValueError("没有图片可保存")

        explicit_names = tuple(line.strip() for line in str(source_filenames).splitlines() if line.strip())
        payload_names = tuple(getattr(images, "filenames", ()) or ())
        if payload_names and len(payload_names) != len(images.images):
            raise ValueError(f"载荷文件名数量（{len(payload_names)}）与图片数量（{len(images.images)}）不一致")
        names = explicit_names or payload_names
        if explicit_names and len(explicit_names) != len(images.images):
            raise ValueError(f"文件名数量（{len(explicit_names)}）与图片数量（{len(images.images)}）不一致")

        project_root = Path(__file__).resolve().parents[3]  # .../ComfyUI-MLX-GEN
        path = save_images(
            uploads_dir=str(project_root / "output"),
            sub=filename_prefix.strip() or "MlxSaveImage",
            # 空 filename_prefix 表示源文件名模式；此时不受 capture 默认值
            # ``<auto>`` 影响，否则空前缀仍会被时间戳模式抢先处理。
            custom=(capture.strip() or "<auto>") if filename_prefix.strip() else "",
            images=images.images,
            metadata={"source": "ComfyUI-MLX-GEN", "batch_index": images.batch_index},
            input_info=(),
            size=images.images[0].size,
            type_="png",
            output_directory=output_directory,
            source_filenames=names,
        )
        print(f"[MlxSaveImage] 已保存: {path}")
        return ("DROP",)
