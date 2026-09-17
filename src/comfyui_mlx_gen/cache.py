"""进程级缓存：按「类型 + 完整配置」缓存（同一配置第二次命中）。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

# 各类型缓存条目上限（超出即删最早项，并释放 MLX 缓存）
_TYPE_CAPS: dict[str, int] = {
    # text_encoder bundle + vae + transformer 三份都要同时驻留（flux2 edit 全链路）
    "module": 3,
    # 正/负条件各占一条，再多留几档给换提示词的情况
    "prompt_encoding": 6,
    "component_weights": 2,
    "image": 6,
    # 参考图条件（packed latents + grid ids），换图才会有第二条
    "ref_encoding": 2,
}


class Cache:
    def __init__(self) -> None:
        self._data: dict[str, OrderedDict[str, Any]] = {}

    def get_or_create(self, type_: str, key: str, factory: Callable[[], Any]) -> tuple[Any, bool]:
        """返回 (值, 是否命中缓存)。"""
        bucket = self._data.setdefault(type_, OrderedDict())
        if key in bucket:
            bucket.move_to_end(key)
            print(f"[cache] 命中 {type_} {key[:12]}…")
            return bucket[key], True
        value = factory()
        bucket[key] = value
        cap = _TYPE_CAPS.get(type_, 4)
        while len(bucket) > cap:
            bucket.popitem(last=False)
            _flush_mlx()
        return value, False

    def get(self, type_: str, key: str) -> tuple[Any, bool]:
        """只取已有缓存；未命中返回 (None, False)。"""
        bucket = self._data.get(type_, OrderedDict())
        if key in bucket:
            bucket.move_to_end(key)
            print(f"[cache] 命中 {type_} {key[:12]}…")
            return bucket[key], True
        print(f"[cache] 未命中 {type_} {key[:12]}…")
        return None, False

    def keys(self, type_: str) -> list[str]:
        return list(self._data.get(type_, OrderedDict()).keys())


def _flush_mlx() -> None:
    from . import runtime

    runtime.flush_caches()


CACHE = Cache()
