"""MiniMax-H3 LoRA 键映射。

基于 mlx-gen 0.37 的 MiniMaxH3LoRAMapping（MIT，Filip Strand / AbstractVision）
移植到插件内，并兼容项目锁定的 mflux 0.19.1 ``LoRATarget``。支持：

* diffusers / LightX2V：``transformer_blocks.N.attn.to_q`` 等拆分模块；
* ComfyUI / ai-toolkit / kohya：``blocks.N.attn.qkv_proj`` 等原始融合模块；
* musubi-tuner：``lora_unet_blocks_N_...`` 扁平键。
"""

from __future__ import annotations

import re

import mlx.core as mx

from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class MiniMaxH3LoRAMapping(LoRAMapping):
    TARGET_MODULES = (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    )
    _BLOCK_PREFIXES = (
        "transformer_blocks.{block}",
        "token_refiner.refiner_blocks.{block}",
    )
    _STATE_DICT_PREFIXES = ("", "transformer.", "diffusion_model.")
    ORIGINAL_STATE_DICT_PREFIXES = ("", "diffusion_model.")
    ORIGINAL_BLOCK_PREFIXES = (
        ("blocks.{block}", "transformer_blocks.{block}"),
        ("token_refiner.blocks.{block}", "token_refiner.refiner_blocks.{block}"),
    )
    ORIGINAL_BLOCK_MODULES = (
        ("attn.out_proj", "attn.to_out.0"),
        ("mlp.fc2", "ff.net.2"),
        ("adaln_proj.linear", "adaln_proj.linear"),
    )
    ORIGINAL_STANDALONE_MODULES = (
        ("video_patch_proj", "proj_in"),
        ("audio_patch_proj", "audio_proj_in"),
        ("condition_proj", "context_embedder"),
        ("time_embedder.proj_in", "time_embedder.linear_1"),
        ("time_embedder.proj_out", "time_embedder.linear_2"),
        ("final_layer.adaln_proj.linear", "norm_out.linear"),
        ("final_layer.video_out", "proj_out"),
        ("final_layer.audio_out", "audio_proj_out"),
    )
    DIFFUSERS_STANDALONE_MODULES = tuple(
        target for _, target in ORIGINAL_STANDALONE_MODULES
    )
    _UP_SUFFIXES = ("lora_B.weight", "lora_up.weight")
    _DOWN_SUFFIXES = ("lora_A.weight", "lora_down.weight")
    _FLATTENED_MODULES = (
        (r"blocks_(\d+)_attn_(qkv|out)_proj", r"blocks.\1.attn.\2_proj"),
        (r"blocks_(\d+)_mlp_fc([12])", r"blocks.\1.mlp.fc\2"),
        (r"blocks_(\d+)_adaln_proj_linear", r"blocks.\1.adaln_proj.linear"),
        (
            r"token_refiner_blocks_(\d+)_attn_(qkv|out)_proj",
            r"token_refiner.blocks.\1.attn.\2_proj",
        ),
        (
            r"token_refiner_blocks_(\d+)_mlp_fc([12])",
            r"token_refiner.blocks.\1.mlp.fc\2",
        ),
        (r"(video|audio)_patch_proj", r"\1_patch_proj"),
        (r"condition_proj", "condition_proj"),
        (r"time_embedder_proj_(in|out)", r"time_embedder.proj_\1"),
        (r"final_layer_adaln_proj_linear", "final_layer.adaln_proj.linear"),
        (r"final_layer_(video|audio)_out", r"final_layer.\1_out"),
    )
    _FLATTENED_PREFIX = "lora_unet_"
    _PEFT_PREFIXES = (
        "base_model.model.model.",
        "base_model.model.",
        "lora_transformer.",
        "lora_unet.",
    )

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets: list[LoRATarget] = []
        targets.extend(
            MiniMaxH3LoRAMapping._diffusers_target(f"{block_prefix}.{module}")
            for block_prefix in MiniMaxH3LoRAMapping._BLOCK_PREFIXES
            for module in MiniMaxH3LoRAMapping.TARGET_MODULES
        )
        targets.append(
            MiniMaxH3LoRAMapping._diffusers_target(
                "transformer_blocks.{block}.adaln_proj.linear"
            )
        )
        targets.extend(
            MiniMaxH3LoRAMapping._diffusers_target(module)
            for module in MiniMaxH3LoRAMapping.DIFFUSERS_STANDALONE_MODULES
        )

        for source_prefix, target_prefix in MiniMaxH3LoRAMapping.ORIGINAL_BLOCK_PREFIXES:
            for index, projection in enumerate(("to_q", "to_k", "to_v")):
                targets.append(
                    MiniMaxH3LoRAMapping._original_target(
                        f"{source_prefix}.attn.qkv_proj",
                        f"{target_prefix}.attn.{projection}",
                        up_transform=MiniMaxH3LoRAMapping.qkv_rows_third(index),
                    )
                )
            targets.append(
                MiniMaxH3LoRAMapping._original_target(
                    f"{source_prefix}.mlp.fc1",
                    f"{target_prefix}.ff.net.0.proj",
                    up_transform=MiniMaxH3LoRAMapping.swap_row_halves,
                )
            )
            for source_module, target_module in MiniMaxH3LoRAMapping.ORIGINAL_BLOCK_MODULES:
                if source_module.startswith("adaln_proj") and source_prefix.startswith(
                    "token_refiner"
                ):
                    continue
                targets.append(
                    MiniMaxH3LoRAMapping._original_target(
                        f"{source_prefix}.{source_module}",
                        f"{target_prefix}.{target_module}",
                    )
                )
        targets.extend(
            MiniMaxH3LoRAMapping._original_target(source, target)
            for source, target in MiniMaxH3LoRAMapping.ORIGINAL_STANDALONE_MODULES
        )
        return targets

    @staticmethod
    def _diffusers_target(path: str) -> LoRATarget:
        prefixes = MiniMaxH3LoRAMapping._STATE_DICT_PREFIXES
        return LoRATarget(
            model_path=path,
            possible_up_patterns=[
                f"{prefix}{path}.{suffix}"
                for prefix in prefixes
                for suffix in MiniMaxH3LoRAMapping._UP_SUFFIXES
            ],
            possible_down_patterns=[
                f"{prefix}{path}.{suffix}"
                for prefix in prefixes
                for suffix in MiniMaxH3LoRAMapping._DOWN_SUFFIXES
            ],
            possible_alpha_patterns=[f"{prefix}{path}.alpha" for prefix in prefixes],
        )

    @staticmethod
    def _original_target(source: str, target: str, up_transform=None) -> LoRATarget:
        prefixes = MiniMaxH3LoRAMapping.ORIGINAL_STATE_DICT_PREFIXES
        return LoRATarget(
            model_path=target,
            possible_up_patterns=[
                f"{prefix}{source}.{suffix}"
                for prefix in prefixes
                for suffix in MiniMaxH3LoRAMapping._UP_SUFFIXES
            ],
            possible_down_patterns=[
                f"{prefix}{source}.{suffix}"
                for prefix in prefixes
                for suffix in MiniMaxH3LoRAMapping._DOWN_SUFFIXES
            ],
            possible_alpha_patterns=[f"{prefix}{source}.alpha" for prefix in prefixes],
            up_transform=up_transform,
        )

    @staticmethod
    def qkv_rows_third(index: int):
        """从融合 ``[q_all; k_all; v_all]`` 的 lora_B 取一个投影。"""

        def transform(tensor: mx.array) -> mx.array:
            rows = tensor.shape[0]
            if rows % 3:
                raise ValueError(
                    "MiniMax-H3 融合 QKV LoRA 的输出行数必须能被 3 整除，"
                    f"收到 {rows}"
                )
            third = rows // 3
            return tensor[index * third : (index + 1) * third]

        return transform

    @staticmethod
    def swap_row_halves(tensor: mx.array) -> mx.array:
        """原始 ``[gate; value]`` lora_B 转成本地 SwiGLU ``[value; gate]``。"""
        rows = tensor.shape[0]
        if rows % 2:
            raise ValueError(
                f"MiniMax-H3 融合 MLP LoRA 的输出行数必须为偶数，收到 {rows}"
            )
        half = rows // 2
        return mx.concatenate([tensor[half:], tensor[:half]], axis=0)

    @staticmethod
    def normalize_state_dict(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """补齐 mflux 0.19.1 尚无的 PEFT 规范化，并恢复 musubi 扁平路径。"""
        peft_normalized: dict[str, mx.array] = {}
        for source_key, value in weights.items():
            key = source_key
            for prefix in MiniMaxH3LoRAMapping._PEFT_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            for matrix in ("lora_A", "lora_B", "lora_up", "lora_down"):
                key = key.replace(f".{matrix}.default.weight", f".{matrix}.weight")
            if key in peft_normalized:
                raise ValueError(f"MiniMax-H3 LoRA 键规范化后重复：{source_key} -> {key}")
            peft_normalized[key] = value

        prefix = MiniMaxH3LoRAMapping._FLATTENED_PREFIX
        if not any(key.startswith(prefix) for key in peft_normalized):
            return peft_normalized
        normalized: dict[str, mx.array] = {}
        for key, value in peft_normalized.items():
            if not key.startswith(prefix):
                normalized[key] = value
                continue
            module, separator, suffix = key[len(prefix) :].partition(".")
            if not separator:
                raise ValueError(f"无效的 musubi-tuner LoRA 键：{key}")
            for pattern, replacement in MiniMaxH3LoRAMapping._FLATTENED_MODULES:
                if re.fullmatch(pattern, module):
                    normalized[f"{re.sub(pattern, replacement, module)}.{suffix}"] = value
                    break
            else:
                raise ValueError(f"{key} 不是已知的 MiniMax-H3 lora_unet_ 模块")
        return normalized

    @staticmethod
    def reject_unsupported_layout(keys) -> None:
        """DiffSynth 的逐 head 交错 QKV 不能按普通三等分解释，必须拒绝。"""
        if any(
            ".attn.qkv_proj.lora_" in key and ".default.weight" in key
            for key in keys
        ):
            raise ValueError(
                "该 MiniMax-H3 LoRA 使用 DiffSynth-Studio 的逐 head 交错融合 QKV "
                "（qkv_proj 的 .default.weight 布局），当前尚未验证其反交错；"
                "请改用 diffusers、ai-toolkit、musubi-tuner 或普通 ComfyUI 布局。"
            )