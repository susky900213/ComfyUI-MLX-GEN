"""Qwen-Image-Edit 专用胶水：视觉塔装配（给 types.py 的 attach_import 用）。

为什么要有这个模块：Qwen 编辑的文本编码器 = `QwenTextEncoder` + **视觉塔**
（`encoder.visual = VisionTransformer()`，权重名 `encoder.visual.*`），而视觉塔必须在
「实例化之后、写权重之前」挂上 —— `Module.update(strict=False)` 会**静默丢弃**模块树里
不存在的键，漏挂就等于视觉塔保持随机初始化（条件全错，而且不会报错）。

本模块只在运行时被 `runtime.import_object` 按需加载，因此不在 import 期依赖 mlx。
"""

from __future__ import annotations


def attach_vision_tower(text_encoder) -> None:
    """给 `QwenTextEncoder.encoder` 挂上视觉塔（`VisionTransformer`）。

    与 mflux `QwenImageInitializer._init_edit_models` 的
    `model.text_encoder.encoder.visual = VisionTransformer()` 完全一致。
    """
    from . import runtime

    visual_cls = runtime.import_object(
        "mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer:VisionTransformer"
    )
    text_encoder.encoder.visual = visual_cls()
