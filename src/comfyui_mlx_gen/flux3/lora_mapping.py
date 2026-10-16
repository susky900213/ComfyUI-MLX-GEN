"""LoRA targets for the FLUX 3 Diffusers transformer names."""

from __future__ import annotations

from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


def _target(path: str) -> LoRATarget:
    source = path.replace("{mode}", "{mode}")
    prefixes = [source, f"transformer.{source}", f"diffusion_model.{source}"]
    flat = source.replace(".", "_")
    return LoRATarget(
        model_path=path,
        possible_up_patterns=[
            *(f"{p}.lora_B.weight" for p in prefixes),
            *(f"{p}.lora_B.default.weight" for p in prefixes),
            *(f"{p}.lora_up.weight" for p in prefixes),
        ],
        possible_down_patterns=[
            *(f"{p}.lora_A.weight" for p in prefixes),
            *(f"{p}.lora_A.default.weight" for p in prefixes),
            *(f"{p}.lora_down.weight" for p in prefixes),
        ],
        possible_alpha_patterns=[*(f"{p}.alpha" for p in prefixes)],
        possible_lokr_w1_patterns=[f"lycoris_{flat}.lokr_w1", f"lora_unet_{flat}.lokr_w1"],
        possible_lokr_w2_patterns=[f"lycoris_{flat}.lokr_w2", f"lora_unet_{flat}.lokr_w2"],
    )


class Flux3LoRAMapping(LoRAMapping):
    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets: list[LoRATarget] = []
        for path in (
            "context_embedder", "time_in.in_layer", "time_in.out_layer",
            "ref_time_in.in_layer", "ref_time_in.out_layer", "norm_out.linear", "proj_out",
        ):
            targets.append(_target(path))
        for mode in ("image", "image_cond"):
            targets.append(_target(f"x_embedders.{mode}"))
        for family in ("early_stream_modulations", "single_stream_modulations"):
            for mode in ("image", "image_cond", "txt"):
                targets.append(_target(f"{family}.{mode}.linear"))
        block_names = (
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "ff.linear_in", "ff.linear_out",
        )
        for mode in ("image", "image_cond"):
            for name in block_names:
                targets.append(_target(f"mode_blocks.{mode}.{{block}}.{name}"))
        for name in block_names:
            targets.append(_target(f"txt_mode_blocks.{{block}}.{name}"))
            targets.append(_target(f"single_transformer_blocks.{{block}}.{name}"))
        return targets