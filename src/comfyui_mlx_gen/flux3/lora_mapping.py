"""LoRA targets for FLUX 3 Diffusers and BFL/ComfyUI transformer names."""

from __future__ import annotations

from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


def _target(path: str, *source_paths: str) -> LoRATarget:
    """Create one target with Diffusers and BFL/ComfyUI source aliases.

    The native MLX module tree follows the Diffusers spelling (``attn.to_q``,
    ``ff.linear_in``), while BFL/ComfyUI LoRAs use names such as ``q_proj``
    and ``mlp_in`` below a ``diffusion_model.`` prefix.  These are aliases for
    the same layer, not separate model layers, so all spellings must point at
    one ``LoRATarget``.
    """
    sources = (path, *source_paths)
    prefixes = [
        prefix
        for source in sources
        for prefix in (source, f"transformer.{source}", f"diffusion_model.{source}")
    ]
    flat = path.replace(".", "_")
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
        for path, bfl_path in (
            ("context_embedder", "txt_in"),
            ("time_in.in_layer", "time_in.in_layer"),
            ("time_in.out_layer", "time_in.out_layer"),
            ("ref_time_in.in_layer", "ref_time_in.in_layer"),
            ("ref_time_in.out_layer", "ref_time_in.out_layer"),
            ("norm_out.linear", "final_layer.image.adaLN_modulation.1"),
            ("proj_out", "final_layer.image.linear"),
        ):
            targets.append(_target(path, bfl_path))
        for mode in ("image", "image_cond"):
            targets.append(_target(f"x_embedders.{mode}", f"emb_in.{mode}"))
        for family in ("early_stream_modulations", "single_stream_modulations"):
            for mode in ("image", "image_cond", "txt"):
                targets.append(_target(f"{family}.{mode}.linear", f"{family}.{mode}.lin"))
        block_names = (
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "ff.linear_in", "ff.linear_out",
        )
        for mode in ("image", "image_cond"):
            for name in block_names:
                bfl_name = {
                    "attn.to_q": "q_proj",
                    "attn.to_k": "k_proj",
                    "attn.to_v": "v_proj",
                    "attn.to_out.0": "attn_out",
                    "ff.linear_in": "mlp_in",
                    "ff.linear_out": "mlp_out",
                }[name]
                targets.append(
                    _target(
                        f"mode_blocks.{mode}.{{block}}.{name}",
                        f"content_mode_blocks.{mode}.{{block}}.{bfl_name}",
                    )
                )
        for name in block_names:
            bfl_name = {
                "attn.to_q": "q_proj",
                "attn.to_k": "k_proj",
                "attn.to_v": "v_proj",
                "attn.to_out.0": "attn_out",
                "ff.linear_in": "mlp_in",
                "ff.linear_out": "mlp_out",
            }[name]
            targets.append(
                _target(
                    f"txt_mode_blocks.{{block}}.{name}",
                    f"txt_mode_blocks.{{block}}.{bfl_name}",
                )
            )
            targets.append(
                _target(
                    f"single_transformer_blocks.{{block}}.{name}",
                    f"single_blocks.{{block}}.{bfl_name}",
                )
            )
        return targets