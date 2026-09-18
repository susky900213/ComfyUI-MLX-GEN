"""MlxLoraApply：向 model 里追加 LoRA（模型 / CLIP 各一个入口）。"""

from __future__ import annotations

from dataclasses import replace

from .. import paths
from ..transformer_lora import supported_families
from ..types import CLIP, LoraRef, MlxClipHandle, MlxModelHandle

NO_LORA = "<无 LoRA>"


def _add(model, path: str, strength: float):
    """把 LoraRef 追加到 model（同 path 且强度不同即报错）。"""
    loras = list(model.loras)
    for item in loras:
        if item.path == path:
            if item.strength != strength:
                raise ValueError(
                    f"同一 LoRA 出现两次且强度不同: {path}（{item.strength} 与 {strength}）"
                )
            return model  # 相同值 → 直接复用
    if float(strength) == 0.0:
        return model  # 首次登记 0 强度严格等同基础 handle（缓存键也不变）
    loras.append(LoraRef(path, strength))
    return replace(model, loras=tuple(loras))


class MlxModelLoraApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("model", {}),
                "lora": ([NO_LORA] + paths.scan_loras(), {"default": NO_LORA}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "MLX/Gen"

    def apply(self, model: MlxModelHandle, lora: str, strength: float):
        if model is None:
            raise ValueError("必须先连接 MlxTransformerLoader 的输出")
        if lora == NO_LORA:
            return (model,)
        if model.model_type not in supported_families():
            raise ValueError(
                f"{model.model_type} 当前不支持 Transformer LoRA；"
                f"支持的模型大类：{', '.join(supported_families())}"
            )
        return (_add(model, lora, strength),)


class MlxClipLoraApply:
    """保留旧工作流节点；当前没有经过验证的文本编码器 LoRA mapping。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": (CLIP, {}),
                "lora": ([NO_LORA] + paths.scan_loras(), {"default": NO_LORA}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = (CLIP,)
    FUNCTION = "apply"
    CATEGORY = "MLX/Gen"

    def apply(self, clip: MlxClipHandle, lora: str, strength: float):
        # 参数名必须与 INPUT_TYPES 的键一致（ComfyUI 按关键字传参）
        if clip is None:
            raise ValueError("必须先连接 MlxClipLoader 的输出")
        if lora == NO_LORA:
            return (clip,)
        raise NotImplementedError(
            "MLX CLIP LoRA 尚不支持：mflux 0.19.1 的现有 mapping 只覆盖扩散/视频 "
            "Transformer。请把 LoRA 接到「MLX 模型 LoRA」节点；包含文本编码器权重的 "
            "LoRA 需要单独的 CLIP/Qwen/T5 映射，插件不会再静默登记后忽略。"
        )
