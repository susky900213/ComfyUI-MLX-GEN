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
from ..types import CLIP, MlxClipHandle, entry_for, model_types, validate_model_family

COMPONENTS = ["text_encoder", "tokenizer"]
PRECISIONS = ["bfloat16", "float16", "float32"]
# 0 = 不量化（图片链路的现状）；MiniMax-H3 的 Qwen3-VL 编码器不量化要常驻约 50 GB，
# 所以选 minimax_h3 时必须填 8 或 4（MlxTextEncoder 里会挡住 0）
QUANTIZE_OPTIONS = [0, 4, 8]
NO_PATH = "<无可用权重>"


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
                "max_length": ("INT", {"default": 512, "min": 1}),
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
        if precision not in PRECISIONS:
            raise ValueError(f"未知精度: {precision}")
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
