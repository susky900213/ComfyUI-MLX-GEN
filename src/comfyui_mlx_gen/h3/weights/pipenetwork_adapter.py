"""PipeNetwork 原生 MiniMax-H3 MLX checkpoint → 本地 diffusers 风格模块树。

PipeNetwork/minimax-h3-mlx 的发布包保留 MiniMax 原始模块名，并把量化后的
``weight/scales/biases`` 原样写入 safetensors。本地 H3 实现则拆开 Q/K/V，且 SwiGLU
采用 diffusers 的 ``[value; gate]`` 顺序。本模块只做确定性的键名与布局适配；不反量化，
也不重新量化。
"""

from __future__ import annotations

import re
from typing import Any

import mlx.core as mx

SKIP_KEYS = frozenset({"rope.inv_freq"})
QUANTIZED_SUFFIXES = (".scales", ".biases")

_BLOCK_KEY = re.compile(r"^(token_refiner\.blocks|blocks)\.(\d+)\.(.+)$")


def has_quantized_tensor_keys(keys: Any) -> bool:
    """键集合是否包含 MLX affine quantization 的伴随张量。"""
    return any(str(key).endswith(QUANTIZED_SUFFIXES) for key in keys)


def split_head_interleaved_qkv(tensor: mx.array, num_heads: int) -> tuple[mx.array, mx.array, mx.array]:
    """拆 ``[head, q/k/v, head_rows, ...]``，适用于 weight/scales/biases。"""
    if tensor.ndim < 1 or tensor.shape[0] % (3 * num_heads):
        raise ValueError(
            "PipeNetwork fused QKV 第一维无法按 heads×3 拆分："
            f"shape={tuple(tensor.shape)}, num_heads={num_heads}"
        )
    rows_per_head = tensor.shape[0] // (3 * num_heads)
    shaped = tensor.reshape(num_heads, 3, rows_per_head, *tensor.shape[1:])
    return tuple(shaped[:, index].reshape(-1, *tensor.shape[1:]) for index in range(3))


def reorder_mlp_gate_value(tensor: mx.array) -> mx.array:
    """原生 ``[gate; value]`` 输出行换成本地 SwiGLU 的 ``[value; gate]``。"""
    if tensor.ndim < 1 or tensor.shape[0] % 2:
        raise ValueError(f"PipeNetwork fused MLP 第一维必须是偶数，收到 {tuple(tensor.shape)}")
    gate, value = mx.split(tensor, 2, axis=0)
    return mx.concatenate([value, gate], axis=0)


def adapt_transformer_tensor(key: str, tensor: mx.array, num_heads: int) -> dict[str, mx.array]:
    """把一个原生 checkpoint tensor 转为一个或多个本地 tensor。"""
    if key in SKIP_KEYS:
        return {}

    direct_prefixes = (
        ("video_patch_proj.", "proj_in."),
        ("audio_patch_proj.", "audio_proj_in."),
        ("condition_proj.", "context_embedder."),
        ("time_embedder.proj_in.", "time_embedder.linear_1."),
        ("time_embedder.proj_out.", "time_embedder.linear_2."),
        ("token_refiner.final_norm.", "token_refiner.final_norm."),
        ("final_layer.norm.", "norm_out.norm."),
        ("final_layer.adaln_proj.linear.", "norm_out.linear."),
        ("final_layer.video_out.", "proj_out."),
        ("final_layer.audio_out.", "audio_proj_out."),
    )
    for source, target in direct_prefixes:
        if key.startswith(source):
            return {target + key[len(source) :]: tensor}

    match = _BLOCK_KEY.match(key)
    if match is None:
        # 保留未知键，交给 loader 的 unexpected-key 诊断统一报告。
        return {key: tensor}

    source_root, index, tail = match.groups()
    target_root = (
        f"token_refiner.refiner_blocks.{index}"
        if source_root == "token_refiner.blocks"
        else f"transformer_blocks.{index}"
    )

    qkv_prefix = "attn.qkv_proj."
    if tail.startswith(qkv_prefix):
        leaf = tail[len(qkv_prefix) :]
        q, k, v = split_head_interleaved_qkv(tensor, num_heads)
        return {
            f"{target_root}.attn.to_q.{leaf}": q,
            f"{target_root}.attn.to_k.{leaf}": k,
            f"{target_root}.attn.to_v.{leaf}": v,
        }

    replacements = (
        ("attn.q_norm.", "attn.norm_q."),
        ("attn.k_norm.", "attn.norm_k."),
        ("attn.out_proj.", "attn.to_out.0."),
        ("mlp.fc2.", "ff.net.2."),
        ("adaln_proj.", "adaln_proj."),
        ("norm1.", "norm1."),
        ("norm2.", "norm2."),
    )
    fc1_prefix = "mlp.fc1."
    if tail.startswith(fc1_prefix):
        return {
            f"{target_root}.ff.net.0.proj.{tail[len(fc1_prefix):]}": reorder_mlp_gate_value(tensor)
        }
    for source, target in replacements:
        if tail.startswith(source):
            return {f"{target_root}.{target}{tail[len(source):]}": tensor}
    return {f"{target_root}.{tail}": tensor}


def adapt_transformer_state(
    weights: dict[str, mx.array], num_heads: int
) -> dict[str, mx.array]:
    """批量适配；主要供测试和每个 shard 的 loader 调用。"""
    converted: dict[str, mx.array] = {}
    for key, tensor in weights.items():
        for target, value in adapt_transformer_tensor(key, tensor, num_heads).items():
            if target in converted:
                raise ValueError(f"PipeNetwork 权重映射产生重复目标键：{target}")
            converted[target] = value
    return converted