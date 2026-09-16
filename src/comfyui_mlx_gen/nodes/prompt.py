"""MlxConditioning：单条提示词 + 编码它所用的组件配置。

对应 ComfyUI 原生 CLIPTextEncode 节点：接 MlxTextEncoder 的输出（handle），
在本节点上填一条提示词，输出 condition。正、负条件各用一个本节点
（互不干扰），再分别接到 MlxKSamplerMLX 的 positive / negative 入口。
"""

from __future__ import annotations

from ..types import condition, MlxConditioning


class MlxConditioningNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "handle": ("MLX_TEXT_ENCODER", {}),
                "prompt": ("STRING", {"default": "a red cube", "multiline": True}),
            }
        }

    RETURN_TYPES = (condition,)
    FUNCTION = "build_condition"
    CATEGORY = "MLX/Gen"

    def build_condition(self, handle, prompt: str):
        if handle is None:
            raise ValueError("必须先连接 MlxTextEncoder 的输出")
        return (MlxConditioning(handle=handle, prompt=prompt),)
