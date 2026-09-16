"""MlxSaveImage：把 MlxPilImage 写到磁盘（PNG + 可选同名 metadata JSON）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .. import image


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


def save_images(
    uploads_dir: str,
    sub: str,
    custom: str,
    images,
    metadata: dict,
    input_info: tuple,
    size: tuple,
    type_: str,
) -> str:
    """自己实现的 save_images（行为与 ComfyUI 相近，不复制其代码）。

    - 写入目录：uploads/MlxSaveImage/<sub>
    - 文件名：custom == "<auto>" 时用时间戳；否则用 custom
    - custom == "<auto>" 时同基名写 metadata JSON
    """
    base = Path(uploads_dir) / "MlxSaveImage" / sub
    base.mkdir(parents=True, exist_ok=True)
    if custom == "<auto>":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = [base / f"{stamp}_{i}.png" for i in range(len(images))]
        for path, img in zip(paths, images):
            img.save(path)
            meta_path = path.with_suffix(".json")
            payload = {**metadata, "input_info": list(input_info), "size": list(size), "type": type_}
            meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        paths = [base / f"{custom}_{i}.png" for i in range(len(images))]
        for path, img in zip(paths, images):
            img.save(path)
    return ", ".join(str(p) for p in paths)


class MlxSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("images", {}),
                "filename_prefix": ("STRING", {"default": "MlxSaveImage"}),
                "capture": ("STRING", {"default": "<auto>"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "MLX/Gen"

    def save(self, images, filename_prefix, capture):
        if not images.images:
            raise ValueError("没有图片可保存")
        project_root = Path(__file__).resolve().parents[3]  # .../ComfyUI-MLX-GEN
        path = save_images(
            uploads_dir=str(project_root / "output"),
            sub=filename_prefix or "MlxSaveImage",
            custom=capture or "<auto>",
            images=images.images,
            metadata={"source": "ComfyUI-MLX-GEN", "batch_index": images.batch_index},
            input_info=(),
            size=images.images[0].size,
            type_="png",
        )
        print(f"[MlxSaveImage] 已保存: {path}")
        return ("DROP",)
