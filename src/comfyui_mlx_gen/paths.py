"""路径解析与扫描（独立于 ComfyUI 默认模型目录）。

模型根目录：/Users/apple/ComfyUI-Shared/models/mlx
子目录：transformer/ vae/ text_encoder/ tokenizer/ lora/
每个子目录下可以有多个「模型文件夹」，也可以是单个 .safetensors 文件。
"""

from __future__ import annotations

from pathlib import Path

# 模型根目录（D-3）
MODEL_ROOT = Path("/Users/apple/ComfyUI-Shared/models/mlx")

COMPONENT_DIRS: tuple[str, ...] = ("transformer", "vae", "text_encoder", "tokenizer", "lora")


def component_dir(component: str) -> Path:
    """组件子目录（如 "transformer" → <root>/transformer）。"""
    return MODEL_ROOT / component


def list_component_items(component: str) -> list[str]:
    """列出组件目录下可选的文件夹名与 .safetensors 文件名。"""
    root = component_dir(component)
    if not root.is_dir():
        return []
    items: list[str] = []
    for entry in sorted(root.iterdir()):
        if entry.name.startswith((".", "__")):
            continue
        if entry.is_dir():
            items.append(entry.name)
        elif entry.suffix == ".safetensors":
            items.append(entry.name)
    return items


def scan_loras() -> list[str]:
    """lora/ 下的 .safetensors（含一层子目录）。"""
    root = component_dir("lora")
    if not root.is_dir():
        return []
    out: list[str] = []
    for entry in sorted(root.rglob("*.safetensors")):
        if entry.name.startswith((".", "__")):
            continue
        out.append(str(entry.relative_to(root)))
    return out


def resolve(source: str, selection: str, component: str) -> tuple[str, str]:
    """把 widget 输入解析为 (kind, path)。

    kind: "dir" | "file" | "repo"；path 为绝对路径（repo 时为 repo id）。

    输入形式（相对路径一律先按组件目录解析，再按模型根目录解析）：
    - "z-image-turbo-8bit"          → <component>/z-image-turbo-8bit
    - "transformer/foo.safetensors" → <root>/transformer/foo.safetensors
    - "/abs/.../foo.safetensors"    → 绝对路径
    - source == "hf_repo"           → 不做文件系统检查，直接当 repo id
    """
    if source == "hf_repo":
        return ("repo", selection)
    if source == "model_card":
        raise ValueError(f"model_card 来源未实现: {selection}")

    if str(selection).startswith("/"):
        path = Path(selection).resolve()
    elif "/" in str(selection):
        path = (MODEL_ROOT / selection).resolve()
        if not path.exists():
            path = (component_dir(component) / selection).resolve()
    else:
        path = (component_dir(component) / selection).resolve()
        if not path.exists():
            path = (MODEL_ROOT / selection).resolve()

    if path.is_dir():
        return ("dir", str(path))
    if path.is_file():
        return ("file", str(path))
    return ("missing", str(path))


def relative_to_root(path: str) -> str:
    """绝对路径 → 相对模型根目录的字符串（用于构造 hf_subdir）。"""
    p = Path(path)
    try:
        return str(p.relative_to(MODEL_ROOT))
    except ValueError:
        return str(p)
