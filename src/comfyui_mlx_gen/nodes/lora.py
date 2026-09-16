"""MlxLoraApply：向 model 里追加 LoRA（模型 / CLIP 各一个入口）。"""

from __future__ import annotations

from dataclasses import replace

from .. import paths
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
        return (_add(model, lora, strength),)


class MlxClipLoraApply:
    """给 CLIP model 挂 LoRA（M1 只登记，不实际应用；与 MlxModelLoraApply 一致）。"""

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
        return (_add(clip, lora, strength),)
