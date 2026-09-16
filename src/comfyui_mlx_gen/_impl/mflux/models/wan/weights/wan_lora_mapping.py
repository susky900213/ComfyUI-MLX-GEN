from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class WanLoRAMapping(LoRAMapping):
    _DIFFUSERS_PREFIXES = ("", "transformer.")
    _ORIGINAL_PREFIXES = ("", "diffusion_model.")
    _UP_SUFFIXES = ("lora_B.weight", "lora_up.weight")
    _DOWN_SUFFIXES = ("lora_A.weight", "lora_down.weight")

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets: list[LoRATarget] = []
        for source_name, target_name in (
            ("self_attn.q", "attn1.to_q"),
            ("self_attn.k", "attn1.to_k"),
            ("self_attn.v", "attn1.to_v"),
            ("self_attn.o", "attn1.to_out.0"),
            ("cross_attn.q", "attn2.to_q"),
            ("cross_attn.k", "attn2.to_k"),
            ("cross_attn.v", "attn2.to_v"),
            ("cross_attn.o", "attn2.to_out.0"),
            ("cross_attn.k_img", "attn2.add_k_proj"),
            ("cross_attn.v_img", "attn2.add_v_proj"),
        ):
            targets.append(WanLoRAMapping._attention_target(source_name, target_name))
        targets.append(WanLoRAMapping._ffn_target("ffn.0", "ffn.net.0", musubi_name="ffn_0"))
        targets.append(WanLoRAMapping._ffn_target("ffn.2", "ffn.net.1", musubi_name="ffn_2"))
        return targets

    @staticmethod
    def _attention_target(source_name: str, target_name: str) -> LoRATarget:
        model_path = f"blocks.{{block}}.{target_name}"
        diffusers_base = f"blocks.{{block}}.{target_name}"
        original_base = f"blocks.{{block}}.{source_name}"
        musubi_base = f"blocks_{{block}}_{source_name.replace('.', '_')}"
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=WanLoRAMapping._patterns(
                diffusers_base=diffusers_base,
                original_base=original_base,
                musubi_base=musubi_base,
                suffixes=WanLoRAMapping._UP_SUFFIXES,
            ),
            possible_down_patterns=WanLoRAMapping._patterns(
                diffusers_base=diffusers_base,
                original_base=original_base,
                musubi_base=musubi_base,
                suffixes=WanLoRAMapping._DOWN_SUFFIXES,
            ),
            possible_alpha_patterns=WanLoRAMapping._patterns(
                diffusers_base=diffusers_base,
                original_base=original_base,
                musubi_base=musubi_base,
                suffixes=("alpha",),
            ),
        )

    @staticmethod
    def _ffn_target(source_name: str, target_name: str, *, musubi_name: str) -> LoRATarget:
        model_path = f"blocks.{{block}}.{target_name}"
        diffusers_base = f"blocks.{{block}}.{target_name}"
        original_base = f"blocks.{{block}}.{source_name}"
        musubi_base = f"blocks_{{block}}_{musubi_name}"
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=WanLoRAMapping._patterns(
                diffusers_base=diffusers_base,
                original_base=original_base,
                musubi_base=musubi_base,
                suffixes=WanLoRAMapping._UP_SUFFIXES,
            ),
            possible_down_patterns=WanLoRAMapping._patterns(
                diffusers_base=diffusers_base,
                original_base=original_base,
                musubi_base=musubi_base,
                suffixes=WanLoRAMapping._DOWN_SUFFIXES,
            ),
            possible_alpha_patterns=WanLoRAMapping._patterns(
                diffusers_base=diffusers_base,
                original_base=original_base,
                musubi_base=musubi_base,
                suffixes=("alpha",),
            ),
        )

    @staticmethod
    def _patterns(
        *,
        diffusers_base: str,
        original_base: str,
        musubi_base: str,
        suffixes: tuple[str, ...],
    ) -> list[str]:
        patterns = []
        for prefix in WanLoRAMapping._DIFFUSERS_PREFIXES:
            patterns.extend(f"{prefix}{diffusers_base}.{suffix}" for suffix in suffixes)
        for prefix in WanLoRAMapping._ORIGINAL_PREFIXES:
            patterns.extend(f"{prefix}{original_base}.{suffix}" for suffix in suffixes)
        patterns.extend(f"lora_unet_{musubi_base}.{suffix}" for suffix in suffixes)
        return patterns
