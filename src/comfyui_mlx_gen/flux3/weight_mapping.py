"""Diffusers weight mapping for FLUX 3 Image Klein."""

from __future__ import annotations

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget


class Flux3WeightMapping(WeightMapping):
    @staticmethod
    def get_transformer_mapping() -> list[WeightTarget]:
        targets: list[WeightTarget] = [
            WeightTarget("context_embedder.weight", ["context_embedder.weight", "txt_in.weight", "diffusion_model.txt_in.weight"]),
            WeightTarget("time_in.in_layer.weight", ["timestep_embedder.linear_1.weight"]),
            WeightTarget("time_in.out_layer.weight", ["timestep_embedder.linear_2.weight"]),
            WeightTarget("ref_time_in.in_layer.weight", ["target_timestep_embedder.linear_1.weight"], required=False),
            WeightTarget("ref_time_in.out_layer.weight", ["target_timestep_embedder.linear_2.weight"], required=False),
            WeightTarget("norm_out.linear.weight", ["norm_out.linear.weight", "final_layer.image.adaLN_modulation.1.weight"]),
            WeightTarget("proj_out.weight", ["proj_out.weight", "final_layer.image.linear.weight"]),
        ]
        for mode, source_mode in (("image", "image"), ("image_cond", "image_cond")):
            targets.append(
                WeightTarget(
                    f"x_embedders.{mode}.weight",
                    [f"x_embedders.{source_mode}.weight", f"emb_in.{source_mode}.weight", f"diffusion_model.emb_in.{source_mode}.weight"],
                )
            )
        for modulation in ("early_stream_modulations", "single_stream_modulations"):
            for mode in ("image", "image_cond", "txt"):
                targets.append(
                    WeightTarget(
                        f"{modulation}.{mode}.linear.weight",
                        [f"{modulation}.{mode}.linear.weight", f"{modulation}.{mode}.lin.weight"],
                    )
                )
        for mode in ("image", "image_cond"):
            targets.extend(
                _block_targets(
                    f"mode_blocks.{mode}.{{block}}", max_blocks=5,
                    source_prefixes=(f"mode_blocks.{mode}.{{block}}", f"content_mode_blocks.{mode}.{{block}}"),
                )
            )
        targets.extend(
            _block_targets(
                "txt_mode_blocks.{block}", max_blocks=5,
                source_prefixes=("txt_mode_blocks.{block}",),
            )
        )
        targets.extend(
            _block_targets(
                "single_transformer_blocks.{block}", max_blocks=28,
                source_prefixes=("single_transformer_blocks.{block}", "single_blocks.{block}"),
            )
        )
        return targets

    @staticmethod
    def get_text_encoder_mapping() -> list[WeightTarget]:
        names = (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        )
        targets = [WeightTarget("language_model.embed_tokens.weight", ["model.language_model.embed_tokens.weight"])]
        targets.extend(
            WeightTarget(
                f"language_model.layers.{{layer}}.{name}",
                [f"model.language_model.layers.{{layer}}.{name}"],
                max_blocks=36,
            )
            for name in names
        )
        targets.extend(
            WeightTarget(
                f"visual.{name}",
                [f"model.visual.{name}"],
            )
            for name in ("patch_embed.proj.bias", "pos_embed.weight")
        )
        targets.append(
            WeightTarget(
                "visual.patch_embed.proj.weight",
                ["model.visual.patch_embed.proj.weight"],
                transform=lambda value: value.reshape(value.shape[0], -1),
            )
        )
        vision_names = (
            "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
            "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
            "mlp.linear_fc1.weight", "mlp.linear_fc1.bias", "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
        )
        targets.extend(
            WeightTarget(
                f"visual.blocks.{{block}}.{name}",
                [f"model.visual.blocks.{{block}}.{name}"],
                max_blocks=24,
            )
            for name in vision_names
        )
        merger_names = ("norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias", "linear_fc2.weight", "linear_fc2.bias")
        for merger in ("merger", "deepstack_merger_list.0", "deepstack_merger_list.1", "deepstack_merger_list.2"):
            targets.extend(
                WeightTarget(f"visual.{merger}.{name}", [f"model.visual.{merger}.{name}"])
                for name in merger_names
            )
        return targets

def _block_targets(
    prefix: str,
    *,
    max_blocks: int,
    source_prefixes: tuple[str, ...],
) -> list[WeightTarget]:
    # The Diffusers export calls these modules attn/ff.  The BFL reference
    # checkpoint calls them q_proj/mlp_in/etc.; both are official formats.
    names = (
        ("attn.to_q", ("attn.to_q.weight", "q_proj.weight")),
        ("attn.to_k", ("attn.to_k.weight", "k_proj.weight")),
        ("attn.to_v", ("attn.to_v.weight", "v_proj.weight")),
        ("attn.norm_q", ("attn.norm_q.weight", "norm.query_norm.scale")),
        ("attn.norm_k", ("attn.norm_k.weight", "norm.key_norm.scale")),
        ("attn.to_out.0", ("attn.to_out.0.weight", "attn_out.weight")),
        ("ff.linear_in", ("ff.linear_in.weight", "mlp_in.weight")),
        ("ff.linear_out", ("ff.linear_out.weight", "mlp_out.weight")),
    )
    targets: list[WeightTarget] = []
    # mflux's generic mapper only expands the historical
    # ``transformer_blocks.{block}`` spelling.  FLUX 3 uses mode_blocks and
    # single_blocks, so materialize those indexes explicitly.
    for block in range(max_blocks):
        target_prefix = prefix.replace("{block}", str(block))
        concrete_sources = tuple(
            source_prefix.replace("{block}", str(block))
            for source_prefix in source_prefixes
        )
        for target_name, suffixes in names:
            sources = [f"{source_prefix}.{suffix}" for source_prefix in concrete_sources for suffix in suffixes]
            targets.append(WeightTarget(f"{target_prefix}.{target_name}.weight", sources))
    return targets