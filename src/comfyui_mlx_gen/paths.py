"""路径解析与扫描（独立于 ComfyUI 默认模型目录）。

模型根目录：``<插件根目录>/models/mlx``，不依赖启动 ComfyUI 时的当前工作目录。
子目录：transformer/ unconditional_transformer/ vae/ text_encoder/ tokenizer/ lora/
每个子目录下可以有多个「模型文件夹」，也可以是单个 .safetensors 文件。
"""

from __future__ import annotations

from pathlib import Path

# paths.py 位于 <插件根目录>/src/comfyui_mlx_gen/paths.py。
# 必须先 resolve 再拼 models/mlx；直接写 Path("models/mlx") 会随 ComfyUI 的启动目录变化。
PLUGIN_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PLUGIN_ROOT / "models" / "mlx"

COMPONENT_DIRS: tuple[str, ...] = (
    "transformer",
    "unconditional_transformer",
    "vae",
    "text_encoder",
    "tokenizer",
    "lora",
)


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


def normalize_hf_cache_path(path: str | Path) -> Path:
    """把 Hugging Face cache 仓库外壳解析到 ``refs/main`` 指向的 snapshot。

    ``huggingface_hub`` 的缓存根（``models--org--repo``）本身不含 ``config.json`` 和
    safetensors；真实文件位于 ``snapshots/<revision>/``。ComfyUI 的模型目录经常直接软链
    到这个缓存根，因此在所有组件路径进入加载器前统一解析。已经指向 snapshot 或普通
    模型目录的路径保持不变。

    一个目录只要带 ``refs`` 或 ``snapshots``，或名字符合 HF cache 外壳格式，就按 cache
    外壳严格校验；这样损坏的 ``refs/main`` 不会退化成稍后才出现的“缺 config.json”。
    """
    root = Path(path)
    looks_like_cache = (
        root.name.startswith("models--")
        or (root / "refs").exists()
        or (root / "snapshots").exists()
    )
    if not looks_like_cache:
        return root

    main_ref = root / "refs" / "main"
    if not main_ref.is_file():
        raise FileNotFoundError(
            f"Hugging Face 缓存目录 {root} 缺少 refs/main；"
            "请完成 main revision 的下载，或把软链接直接指向 snapshots/<revision>"
        )
    try:
        revision = main_ref.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise OSError(f"无法读取 Hugging Face revision 文件 {main_ref}: {exc}") from exc
    if not revision or revision in {".", ".."} or Path(revision).name != revision:
        raise ValueError(f"Hugging Face revision 非法：{main_ref} 包含 {revision!r}")

    snapshot = root / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"Hugging Face 缓存 refs/main 指向不存在的 snapshot：{snapshot}"
        )
    if not any(snapshot.iterdir()):
        raise FileNotFoundError(f"Hugging Face snapshot 是空目录：{snapshot}")
    return snapshot.resolve()


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
        return ("dir", str(normalize_hf_cache_path(path)))
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
