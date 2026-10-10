"""LoRA mapping for the native Qwen-Image 2.1 MLX transformer.

The MLX port deliberately keeps the module names used by the checkpoint
conversion layer: attention output is a one-element Python list and the image
feed-forward has separate ``proj``, ``out`` and ``gate_layer`` linears.  Keep
the mapping here instead of extending legacy Qwen's mapping, whose ``img_ff``
targets do not exist in this transformer.
"""

from __future__ import annotations

from typing import Any

from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


def _target(module_path: str, source_path: str) -> LoRATarget:
    """Build the standard Diffusers/Comfy LoRA spellings for one local layer."""
    return LoRATarget(
        model_path=module_path,
        possible_up_patterns=[
            f"{source_path}.lora_up.weight",
            f"transformer.{source_path}.lora.up.weight",
            f"{source_path}.lora_B.weight",
            f"diffusion_model.{source_path}.lora_B.weight",
            f"transformer.{source_path}.lora_B.weight",
            f"{source_path}.lora_B.default.weight",
            f"diffusion_model.{source_path}.lora_B.default.weight",
            f"transformer.{source_path}.lora_B.default.weight",
        ],
        possible_down_patterns=[
            f"{source_path}.lora_down.weight",
            f"transformer.{source_path}.lora.down.weight",
            f"{source_path}.lora_A.weight",
            f"diffusion_model.{source_path}.lora_A.weight",
            f"transformer.{source_path}.lora_A.weight",
            f"{source_path}.lora_A.default.weight",
            f"diffusion_model.{source_path}.lora_A.default.weight",
            f"transformer.{source_path}.lora_A.default.weight",
        ],
        possible_alpha_patterns=[
            f"{source_path}.alpha",
            f"diffusion_model.{source_path}.alpha",
            f"transformer.{source_path}.alpha",
        ],
    )


class QwenImage21LoRAMapping(LoRAMapping):
    """Mapping for the seven independently LoRA-addressable qwen21 targets."""

    @staticmethod
    def get_mapping() -> list[Any]:
        targets: list[LoRATarget] = []
        for attention_name in ("to_q", "to_k", "to_v"):
            targets.append(
                _target(
                    f"transformer_blocks.{{block}}.attn.{attention_name}",
                    f"transformer_blocks.{{block}}.attn.{attention_name}",
                )
            )
        targets.append(
            _target(
                # mflux resolves both attributes and list indices by splitting on
                # dots, so the list item is addressed as ``to_out.0``.
                "transformer_blocks.{block}.attn.to_out.0",
                "transformer_blocks.{block}.attn.to_out.0",
            )
        )
        for mlp_name in ("proj", "out", "gate_layer"):
            targets.append(
                _target(
                    f"transformer_blocks.{{block}}.img_mlp.{mlp_name}",
                    f"transformer_blocks.{{block}}.img_mlp.{mlp_name}",
                )
            )
        return targets