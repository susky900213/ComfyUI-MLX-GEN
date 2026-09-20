"""运行环境 + 唯一 import 入口。

- 若插件自带 _impl，则优先使用；否则使用 requirements.txt 安装的 mflux
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
REQUIRED_MLX = ">=0.32.0,<0.33.0"


# --- sys.path / 导入（唯一入口） ---
def ensure_impl_path() -> str:
    if not IMPL_DIR.is_dir():
        # PyPI mflux 0.19.1 自带 Ideogram 4；无需改 sys.path。
        return ""
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
            f"无法导入 MLX（需要 mlx{REQUIRED_MLX}），"
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


# --- MLX free-cache 上限（对应 MlxTransformerLoader 的 compile_cache_limit）----
# 语义与 mflux 对齐：正数按**十进制** GB 计（int(gb * 1000**3)，与 mflux
# 的 --mlx-cache-limit-gb / RuntimeMemory.apply_mlx_cache_limit 一致），
# 设完顺手 mx.clear_cache()，让新上限立刻对后续分配生效。
# 0 = 「不设置」：本进程若先前设过上限，则恢复第一次记录下来的原值
# （≈ MLX 默认，随内存规模缩放，等价于「不设上限」）；没设过就什么都不动。
# mx.set_cache_limit 没有 getter，但它的返回值就是「上一轮的上限」，
# 因此第一次调用时把原值记下来，之后就能原样还原。
_ORIGINAL_CACHE_LIMIT: list[int | None] = [None]


def apply_cache_limit(limit_gb: float, *, label: str = "") -> None:
    """按 widget 档位设 MLX 的 free-cache 上限（GB，0 = 沿用默认 / 还原原值）。"""
    prefix = f"{label} " if label else ""
    try:
        import mlx.core as mx  # type: ignore

        if not hasattr(mx, "set_cache_limit"):
            return
        gb = float(limit_gb or 0)
        if gb > 0:
            target = int(gb * 1000**3)
            note = f"{gb:g} GB"
        elif _ORIGINAL_CACHE_LIMIT[0] is None:
            print(f"[runtime] {prefix}编译缓存上限 0 → 不设置，沿用 MLX 默认")
            return
        else:
            target = int(_ORIGINAL_CACHE_LIMIT[0])
            note = f"{target / 1000**3:.1f} GB（恢复默认）"

        previous = int(mx.set_cache_limit(target))
        if _ORIGINAL_CACHE_LIMIT[0] is None:
            _ORIGINAL_CACHE_LIMIT[0] = previous
        mx.clear_cache()
        print(
            f"[runtime] {prefix}MLX cache 上限 → {note}（上一轮 {previous / 1000**3:.1f} GB）"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[runtime] 设置 MLX cache 上限失败，已忽略：{exc}")


def memory_snapshot() -> str:
    """当前内存水位（活跃 / cache 占用 / 峰值，十进制 GB；MLX 不可用给 n/a）。

    与 mflux 的 MemorySaver 用的是同一组读数接口（``mx.get_active_memory`` /
    ``mx.get_cache_memory`` / ``mx.get_peak_memory``），用来核对 cache 上限
    是否真的压住了 cache 占用。
    """
    try:
        import mlx.core as mx  # type: ignore

        active = int(mx.get_active_memory())
        cached = int(mx.get_cache_memory())
        peak = int(mx.get_peak_memory())
        return (
            f"活跃 {active / 1000**3:.1f} GB / cache {cached / 1000**3:.1f} GB"
            f" / 峰值 {peak / 1000**3:.1f} GB"
        )
    except Exception as exc:  # noqa: BLE001
        return f"n/a（{exc}）"
