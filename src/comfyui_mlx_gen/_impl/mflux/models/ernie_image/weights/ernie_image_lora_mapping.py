from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class ErnieImageLoRAMapping(LoRAMapping):
    _PREFIXES = ("", "transformer.", "diffusion_model.", "base_model.model.")
    _UP_SUFFIXES = ("lora_B.weight", "lora_up.weight", "lora_B.default.weight", "lora_up.default.weight")
    _DOWN_SUFFIXES = ("lora_A.weight", "lora_down.weight", "lora_A.default.weight", "lora_down.default.weight")

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets = [
            ErnieImageLoRAMapping._target("text_proj"),
            ErnieImageLoRAMapping._target("time_embedding.linear_1"),
            ErnieImageLoRAMapping._target("time_embedding.linear_2"),
            ErnieImageLoRAMapping._target("adaLN_modulation.linear"),
            ErnieImageLoRAMapping._target("final_norm.linear"),
            ErnieImageLoRAMapping._target("final_linear"),
        ]
        for module_path in (
            "self_attention.to_q",
            "self_attention.to_k",
            "self_attention.to_v",
            "self_attention.to_out.0",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.linear_fc2",
        ):
            targets.extend([ErnieImageLoRAMapping._target(f"layers.{{block}}.{module_path}")])
        return targets

    @staticmethod
    def _target(model_path: str) -> LoRATarget:
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=ErnieImageLoRAMapping._patterns(model_path, ErnieImageLoRAMapping._UP_SUFFIXES),
            possible_down_patterns=ErnieImageLoRAMapping._patterns(model_path, ErnieImageLoRAMapping._DOWN_SUFFIXES),
            possible_alpha_patterns=ErnieImageLoRAMapping._patterns(model_path, ("alpha",)),
        )

    @staticmethod
    def _patterns(model_path: str, suffixes: tuple[str, ...]) -> list[str]:
        return [f"{prefix}{model_path}.{suffix}" for prefix in ErnieImageLoRAMapping._PREFIXES for suffix in suffixes]
