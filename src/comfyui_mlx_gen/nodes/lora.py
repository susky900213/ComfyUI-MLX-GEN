"""MlxLoraApply：向 model 里追加 LoRA（模型 / CLIP 各一个入口）。"""

from __future__ import annotations

from dataclasses import replace

from .. import paths
from ..transformer_lora import resolve_lora_path, supported_families
from ..types import CLIP, LoraRef, MlxClipHandle, MlxModelHandle
from ..moss import lora as moss_lora

NO_LORA = "<无 LoRA>"


def _add(model, path: str, strength: float, *, canonicalize: bool = False):
    """把 LoraRef 追加到 model（同 path 且强度不同即报错）。"""
    loras = list(model.loras)
    # strength=0 严格等同基础 handle；不要为了规范化一个不会被加载的
    # 路径而要求对应文件存在（旧节点和其它模型 LoRA 都依赖这个语义）。
    if float(strength) == 0.0:
        return model
    resolved_path = resolve_lora_path(path) if canonicalize else path
    for item in loras:
        item_path = resolve_lora_path(item.path) if canonicalize else item.path
        if item_path == resolved_path:
            if float(item.strength) != float(strength):
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
        if model.model_type == "moss":
            # 普通 PEFT MOSS LoRA 走通用节点；只有旧的 dynamic-rank 4-step
            # 文件保留专用节点，因为它没有标准 lora_A/lora_B 结构且强度固定为 1。
            if float(strength) != 0.0:
                moss_lora.validate_generic_node_selection(model, lora, strength)
            return (_add(model, lora, strength, canonicalize=True),)
        if model.model_type not in supported_families():
            raise ValueError(
                f"{model.model_type} 当前不支持 Transformer LoRA；"
                f"支持的模型大类：{', '.join(supported_families())}"
            )
        return (_add(model, lora, strength),)


class MlxMoss4StepLoraApply:
    """Register the official MOSS 4-step adapter at strength 1.0."""

    @classmethod
    def INPUT_TYPES(cls):
        choices = [
            item
            for item in paths.scan_loras()
            if moss_lora.MOSS_LORA_FILENAME_RE.search(item.rsplit("/", 1)[-1])
        ] or [NO_LORA]
        return {
            "required": {
                "model": ("model", {}),
                "lora": (choices, {"default": choices[0]}),
            }
        }

    RETURN_TYPES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "MLX/Gen/MOSS"

    def apply(self, model: MlxModelHandle, lora: str):
        if model is None:
            raise ValueError("必须先连接 MlxTransformerLoader 的输出")
        if lora == NO_LORA:
            return (model,)
        if model.model_type != "moss":
            raise ValueError("「MLX MOSS 4 步 LoRA」只能连接 model_type=moss 的 Transformer")
        if int(model.quantize) != 0:
            raise ValueError("MOSS 4-step LoRA 只支持未量化 dense Transformer；请把 quantize 设为 0")
        if model.precision != "bfloat16":
            raise ValueError("MOSS 4-step LoRA 只支持 bfloat16 Transformer")
        moss_lora.validate_node_selection(model, lora)
        return (_add(model, lora, 1.0),)


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
