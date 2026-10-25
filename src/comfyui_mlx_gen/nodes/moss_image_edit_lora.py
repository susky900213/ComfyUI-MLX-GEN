"""Dedicated node for the MOSS image-edit PEFT adapter directory."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .. import paths
from ..moss import lora as moss_lora
from ..types import LoraRef, MlxModelHandle

NO_LORA = "<无 MOSS Image Edit LoRA>"
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"


def scan_moss_image_edit_loras() -> list[str]:
    """Return PEFT adapter directories relative to the MLX ``lora`` folder."""
    root = paths.component_dir("lora")
    if not root.is_dir():
        return []
    choices: list[str] = []
    for config_file in sorted(root.rglob(ADAPTER_CONFIG)):
        directory = config_file.parent
        weights = directory / ADAPTER_WEIGHTS
        if not weights.is_file():
            continue
        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(config, dict) and config.get("format") == moss_lora.MOSS_IMAGE_EDIT_FORMAT:
            choices.append(str(directory.relative_to(root)))
    return choices


def resolve_moss_image_edit_lora(selection: str) -> str:
    """Resolve a widget directory selection to its adapter weights file."""
    selected = Path(selection).expanduser()
    root = paths.component_dir("lora")
    if selected.is_absolute():
        candidates = (selected,)
    elif len(selected.parts) > 1:
        candidates = (paths.MODEL_ROOT / selected, root / selected)
    else:
        candidates = (root / selected, paths.MODEL_ROOT / selected)

    directory = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if directory is None:
        raise FileNotFoundError(f"未找到 MOSS Image Edit LoRA 目录：{selection}")
    config = directory / ADAPTER_CONFIG
    weights = directory / ADAPTER_WEIGHTS
    if not config.is_file() or not weights.is_file():
        raise ValueError(
            f"{directory} 不是完整的 MOSS Image Edit LoRA 目录；"
            f"需要 {ADAPTER_CONFIG} 和 {ADAPTER_WEIGHTS}"
        )
    return str(weights.absolute())


def _add(model: MlxModelHandle, path: str, strength: float) -> MlxModelHandle:
    """Append the adapter once, while preserving model handles' immutability."""
    if float(strength) == 0.0:
        return model
    for item in model.loras:
        if Path(item.path).absolute() == Path(path).absolute():
            if float(item.strength) != float(strength):
                raise ValueError(
                    "同一 MOSS Image Edit LoRA 出现两次且强度不同："
                    f"（{item.strength} 与 {strength}）"
                )
            return model
    return replace(model, loras=(*model.loras, LoraRef(path, strength)))


class MlxMossImageEditLoraApply:
    """Apply the converted ``moss_image_edit_lora`` PEFT directory to MOSS."""

    @classmethod
    def INPUT_TYPES(cls):
        choices = [NO_LORA, *scan_moss_image_edit_loras()]
        return {
            "required": {
                "model": ("model", {}),
                "lora": (choices, {"default": choices[0]}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "MLX/Gen/MOSS"

    def apply(self, model: MlxModelHandle, lora: str, strength: float):
        if model is None:
            raise ValueError("必须先连接 MlxTransformerLoader 的输出")
        if lora == NO_LORA:
            return (model,)
        if model.model_type != "moss":
            raise ValueError("「MLX MOSS Image Edit LoRA」只能连接 model_type=moss 的 Transformer")
        if int(model.quantize) != 0:
            raise ValueError("MOSS Image Edit LoRA 只支持未量化 dense Transformer；请把 quantize 设为 0")
        if model.precision != "bfloat16":
            raise ValueError("MOSS Image Edit LoRA 只支持 bfloat16 Transformer")

        path = resolve_moss_image_edit_lora(lora)
        moss_lora.validate_image_edit_header(path)
        return (_add(model, path, strength),)
