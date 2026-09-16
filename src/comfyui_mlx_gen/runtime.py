"""运行环境 + 唯一 import 入口。

- sys.path 指向 _impl（我们自己拷贝的 mflux 源码），不 pip 安装 mflux
- 校验 MLX 版本与 Metal 设备
- 缓存键生成、内存水位检查、缓存清理
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PLUGIN_DIR = Path(__file__).resolve().parent
IMPL_DIR = PLUGIN_DIR / "_impl"
REQUIRED_MLX = "0.31.2"


# --- sys.path / 导入（唯一入口） ---
def ensure_impl_path() -> str:
    p = str(IMPL_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def import_object(spec: str) -> Any:
    """导入 "module:attr" 或 "module:Class.static"。"""
    if not spec:
        raise ValueError("空的 import 路径")
    ensure_impl_path()
    module_path, _, attr = spec.partition(":")
    module = importlib.import_module(module_path)
    if not attr:
        return module
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


# --- MLX 校验（不写死 MLX 代码） ---
def check_mlx() -> str:
    try:
        import mlx.core as mx  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"无法导入 MLX（需要 mlx=={REQUIRED_MLX}），"
            f"请在 ComfyUI 的 .venv 里 pip install -r requirements.txt：{exc}"
        ) from exc
    if not hasattr(mx, "metal") or not mx.metal.is_available():
        raise RuntimeError("MLX 没有可用的 Metal/GPU 设备")
    return getattr(mx, "__version__", "?")


# --- 缓存键 ---
def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f: _to_jsonable(getattr(value, f)) for f in value.__dataclass_fields__}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def cache_key(config: Any) -> str:
    payload = json.dumps(_to_jsonable(config), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- 内存水位 / 缓存清理 ---
def system_total_memory() -> int:
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False)
    if out.returncode == 0 and out.stdout.strip().isdigit():
        return int(out.stdout.strip())
    return 0


def check_memory_watermark(needed_bytes: int, reserve_ratio: float = 0.25) -> None:
    total = system_total_memory()
    if total and needed_bytes > total * (1.0 - reserve_ratio):
        raise MemoryError(
            f"需要约 {needed_bytes / 1e9:.1f} GB，超过可用内存（共 {total / 1e9:.1f} GB，"
            f"预留 {reserve_ratio:.0%}）；请降低分辨率或减少批大小"
        )


def flush_caches() -> None:
    gc.collect()
    try:
        import mlx.core as mx  # type: ignore

        mx.clear_cache()
    except Exception:  # noqa: BLE001
        pass
