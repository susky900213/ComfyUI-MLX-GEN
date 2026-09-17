"""进程级缓存：按「类型 + 完整配置」缓存（同一配置第二次命中）。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

# 各类型缓存条目上限（超出即删最早项，并释放 MLX 缓存）
_TYPE_CAPS: dict[str, int] = {
    # text_encoder bundle（qwen 编辑时里面还挂着 VL tokenizer × 2 与 VL 编码器）+ vae +
    # transformer 三份都要同时驻留（edit 全链路）
    "module": 3,
    # 正/负条件各占一条，再多留几档给换提示词的情况
    # （flux2/z_image 存编码数组；qwen_edit 存 (embeds, mask)，1.5 MB 量级）
    "prompt_encoding": 6,
    # Ideogram 4 的每条条件为多层 Qwen 特征；只保留当前工作流的一条，换 prompt / 尺寸即淘汰
    "ideogram4_prompt": 1,
    "component_weights": 2,
    "image": 6,
    # 参考图条件（flux2：packed + grid ids + 宽高；qwen_edit：packed + ids + patch 尺寸），
    # 换图 / 换尺寸才会有第二条
    "ref_encoding": 2,
    # 参考图集（MlxRefImageSet 的有序 PIL 元组，uint8 ≈ 3MB/MP）：当前用的一份 + 刚换掉的一份
    "ref_source": 2,
    # --- MiniMax-H3（不与图片链路共用桶：单条配置的组件就多、每条都几十 GB）---
    # transformer / text_encoder / tokenizer / vae / audio_vae 五个 role 各一条
    "h3_module": 5,
    # 正 / 负条件各一条（H3 只用正向，但两个 MlxTextEncoder 会各存一条）
    "h3_prompt": 4,
    # 一次视频生成 = 一条（视频行 + 音频行 + 计划，几 GB 量级，只留最近的一份）
    "h3_latents": 1,
    # --- YuE2（主模型与 VAE 分阶段物化，任一时刻通常只有一条）---
    "yue2_module": 2,
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

    def evict(self, type_: str, key: str) -> bool:
        """主动丢掉一条（H3 这类「用完即释放」的大组件用）；是否真丢到了。"""
        bucket = self._data.get(type_, OrderedDict())
        if key not in bucket:
            return False
        del bucket[key]
        _flush_mlx()
        return True

    def keys(self, type_: str) -> list[str]:
        return list(self._data.get(type_, OrderedDict()).keys())


def _flush_mlx() -> None:
    from . import runtime

    runtime.flush_caches()


CACHE = Cache()
