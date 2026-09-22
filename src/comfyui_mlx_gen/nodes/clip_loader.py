"""MlxClipLoader：只登记配置，返回 MlxClipHandle（不实例化组件、不加载权重）。

对齐 ComfyUI 的 CLIP 加载器：输出类型用原生 "CLIP"，供「MLX 文本编码器」
（MlxTextEncoder，输入参数同为 clip）或任何吃 CLIP 的下游节点使用；真正的
加载与编码都发生在 MlxTextEncoder 里，本节点只登记「大类 + 组件 + 权重目录 +
精度」，不 import mflux、不建 tokenizer，也**不选配置变体**（用哪套
ModelConfig 由 weights.config_for_path 按目录名匹配）。

权重目录候选扫磁盘得到，因此新增一份同配置的权重（如 z-image-turbo-4bit）
不必改代码；但选的目录要和该大类的组件对得上，否则 MlxTextEncoder 里会失败。
"""

from __future__ import annotations

from .. import paths, runtime
from ..types import (
    CLIP,
    QWEN_IMAGE_21_MAX_LENGTH,
    MlxClipHandle,
    entry_for,
    model_types,
    validate_model_family,
)

COMPONENTS = ["text_encoder", "tokenizer"]
# `MLX 16bit` 是给用户看的加载类型；内部仍使用 MLX 原生的 `float16`。
# 保留 `float16` 以兼容旧工作流直接序列化的值。
PRECISIONS = ["bfloat16", "MLX 16bit", "float16", "float32"]
# 0 = 不量化（图片链路的现状）；MiniMax-H3 的 Qwen3-VL 编码器不量化要常驻约 50 GB，
# 所以选 minimax_h3 时必须填 8 或 4（MlxTextEncoder 里会挡住 0）
QUANTIZE_OPTIONS = [0, 4, 8]
NO_PATH = "<无可用权重>"


def normalize_precision(value: str) -> str:
    """将 Loader 显示值转换为 MLX/模型加载器使用的 dtype 名称。"""
    text = str(value).strip().casefold()
    if text in {"mlx 16bit", "16bit", "fp16", "float16"}:
        return "float16"
    if text in {"bfloat16", "bf16", "float32", "fp32"}:
        return {"bf16": "bfloat16", "fp32": "float32"}.get(text, text)
    raise ValueError(f"未知精度: {value}；可选：{', '.join(PRECISIONS)}")


def normalize_max_length(model_type: str, value: int) -> int:
    """迁移 Qwen-Image 2.1 旧工作流中的 512 默认值。

    Qwen-Image 2.1 的模板本身会占用一部分上下文，PE 重写后的普通提示词很容易
    超过旧版 Loader 的 512 默认值。这里只迁移历史默认值；其它长度仍按用户设置
    保留，避免把有意设置的较小上限静默改掉。
    """
    length = int(value)
    if model_type == "qwen_image_21" and length == 512:
        return QWEN_IMAGE_21_MAX_LENGTH
    return length


def component_options(component: str) -> list[str]:
    """该组件目录下可选的权重集（扫盘；目录为空时给占位项，避免下拉为空）。"""
    items = paths.list_component_items(component)
    # YuE2 没有独立 text encoder，Breeze 的文本编码器也封装在完整 checkpoint 内；
    # 为了继续复用这个节点，让两者都能选择 transformer/ 里的完整权重目录。
    if component in ("text_encoder", "tokenizer"):
        items += [
            name
            for name in paths.list_component_items("transformer")
            if "yue2" in name.lower() or "breeze" in name.lower()
        ]
    return list(dict.fromkeys(items)) or [NO_PATH]


class MlxClipLoader:
    @classmethod
    def INPUT_TYPES(cls):
        types = model_types()
        default_type = types[0]
        comp = "text_encoder"
        paths_for_comp = component_options(comp)
        return {
            "required": {
                "model_type": (types, {"default": default_type}),
                "component": (COMPONENTS, {"default": comp}),
                "path": (paths_for_comp, {"default": paths_for_comp[0]}),
                "precision": (PRECISIONS, {"default": "bfloat16"}),
                "max_length": (
                    "INT",
                    {
                        "default": QWEN_IMAGE_21_MAX_LENGTH,
                        "min": 1,
                        "max": QWEN_IMAGE_21_MAX_LENGTH,
                    },
                ),
                # 0 = 不量化；MiniMax-H3 请选 8（选 0 时「MLX 文本编码器」会直接报错）
                "quantize": (QUANTIZE_OPTIONS, {"default": 0}),
            }
        }

    RETURN_TYPES = (CLIP,)
    FUNCTION = "load"
    CATEGORY = "MLX/Gen"

    def load(self, model_type, component, path, precision, max_length, quantize):
        # 未知大类 → 直接报错；已知但未验证的大类提示还没实现
        entry = entry_for(model_type)
        validate_model_family(model_type, path)
        if not entry.supported:
            raise NotImplementedError(f"{model_type} 尚未实现：{entry.notes}")
        if component not in COMPONENTS:
            raise ValueError(f"未知组件: {component}")
        precision = normalize_precision(precision)
        max_length = normalize_max_length(model_type, max_length)
        config = {
            "model_type": model_type,
            "component": component,
            "path": path,
            "precision": precision,
            "max_length": int(max_length),
            "quantize": int(quantize),
        }
        handle = MlxClipHandle(
            model_type=model_type,
            component=component,
            source="local",
            path=path,
            precision=precision,
            max_length=int(max_length),
            extra_options={},
            quantize=int(quantize) or None,
            cache_key=runtime.cache_key(config),
        )
        return (handle,)
