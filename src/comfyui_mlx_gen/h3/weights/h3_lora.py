"""MiniMax-H3 LoRA 加载及 ComfyUI ``int8-convrot`` 解码。"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx

from comfyui_mlx_gen.types import LoraRef

from .h3_lora_mapping import MiniMaxH3LoRAMapping

_LORA_DOWN_KEY = re.compile(r"(?:lora_A|lora_down)(?:\.default)?\.weight$")
_HADAMARD_CACHE: dict[int, mx.array] = {}


def _regular_hadamard(size: int) -> mx.array:
    """Comfy ConvRot 使用的 regular Hadamard（不是 MLX 的 Sylvester 排列）。"""
    if size in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[size]
    if size < 4 or size & (size - 1) or math.log(size, 4) % 1:
        raise ValueError(f"ConvRot group size 必须是 4 的整数次幂，收到 {size}")
    h4 = mx.array(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=mx.float32,
    )
    h = h4
    current = 4
    while current < size:
        h = mx.kron(h, h4)
        current *= 4
    h = h / math.sqrt(size)
    mx.eval(h)
    _HADAMARD_CACHE[size] = h
    return h


def dequantize_int8_tensorwise(
    qdata: mx.array,
    scale: mx.array,
    *,
    convrot: bool,
    convrot_groupsize: int,
) -> mx.array:
    """逐行反量化，并按 Comfy Kitchen 语义把 ConvRot 权重旋回原基底。"""
    if qdata.ndim != 2:
        raise ValueError(f"int8-convrot LoRA 只支持二维矩阵，收到 shape={qdata.shape}")
    value = qdata.astype(mx.float32) * scale.astype(mx.float32)
    if convrot:
        rows, columns = value.shape
        if columns % convrot_groupsize:
            raise ValueError(
                f"LoRA 输入维 {columns} 不能被 ConvRot group size {convrot_groupsize} 整除"
            )
        h = _regular_hadamard(convrot_groupsize)
        value = mx.matmul(
            value.reshape(rows, columns // convrot_groupsize, convrot_groupsize),
            h.T,
        ).reshape(rows, columns)
    # 配套 ComfyUI-LoraInt8Loader 解到 bf16；保持同样精度并把常驻 LoRA 内存减半。
    return value.astype(mx.bfloat16)


def decode_comfy_quantized_state_dict(
    weights: dict[str, mx.array],
) -> tuple[dict[str, mx.array], int]:
    """将带 ``.comfy_quant`` 的 A/B 矩阵恢复成普通浮点 LoRA state dict。"""
    decoded: dict[str, mx.array] = {}
    consumed: set[str] = set()
    count = 0
    for config_key, config_tensor in weights.items():
        if not config_key.endswith(".comfy_quant"):
            continue
        base = config_key[: -len(".comfy_quant")]
        weight_key = base + ".weight"
        scale_key = base + ".weight_scale"
        if weight_key not in weights or scale_key not in weights:
            raise ValueError(
                f"量化 LoRA {base} 缺少 {weight_key if weight_key not in weights else scale_key}"
            )
        try:
            config = json.loads(bytes(config_tensor.tolist()).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"无法解析 {config_key}：{exc}") from exc
        if config.get("format") != "int8_tensorwise":
            raise ValueError(
                f"MiniMax-H3 LoRA 暂不支持 comfy_quant format={config.get('format')!r}"
            )
        decoded[weight_key] = dequantize_int8_tensorwise(
            weights[weight_key],
            weights[scale_key],
            convrot=bool(config.get("convrot", False)),
            convrot_groupsize=int(config.get("convrot_groupsize", 256)),
        )
        consumed.update((config_key, weight_key, scale_key))
        count += 1

    if not count:
        return weights, 0
    for key, value in weights.items():
        if key not in consumed:
            decoded[key] = value
    return decoded, count


def _metadata_scale(weights: dict[str, mx.array], metadata: dict | None) -> float:
    """PEFT 文件级 alpha/rank；有逐模块 ``.alpha`` 时交给 LoRALoader 处理。"""
    if any(key.endswith(".alpha") for key in weights):
        return 1.0
    alpha = (metadata or {}).get("alpha")
    rank = next(
        (int(value.shape[0]) for key, value in weights.items() if _LORA_DOWN_KEY.search(key)),
        None,
    )
    if alpha is None or rank is None:
        return 1.0
    return float(alpha) / rank


def apply_h3_loras(
    transformer: Any,
    loras: Iterable[LoraRef],
    *,
    role: str = "transformer",
) -> tuple[list[str], list[float]]:
    """加载、解码、映射并包装 H3 Transformer；应用 0 层或半失败都直接报错。"""
    from comfyui_mlx_gen.transformer_lora import resolve_lora_path
    from mflux.models.common.lora.mapping.lora_loader import LoRALoader

    mapping = MiniMaxH3LoRAMapping.get_mapping()
    pattern_mappings = LoRALoader._build_pattern_mappings(mapping)
    resolved_paths: list[str] = []
    user_scales: list[float] = []

    for ref in loras:
        path = resolve_lora_path(ref.path)
        scale = float(ref.strength)
        print(f"[MLX LoRA] minimax_h3/{role}: 加载 {Path(path).name}@{scale:g}")
        try:
            loaded, metadata = mx.load(path, return_metadata=True)
            raw = dict(loaded.items())
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise ValueError(f"无法读取 MiniMax-H3 LoRA {path}：{exc}") from exc

        MiniMaxH3LoRAMapping.reject_unsupported_layout(raw.keys())
        weights, quantized_count = decode_comfy_quantized_state_dict(raw)
        weights = MiniMaxH3LoRAMapping.normalize_state_dict(weights)
        effective_scale = scale * _metadata_scale(weights, metadata)
        applied_count, matched_keys, failed_targets = LoRALoader._apply_lora_with_mapping(
            transformer,
            weights,
            effective_scale,
            pattern_mappings,
            role=role,
        )
        if failed_targets:
            shown = ", ".join(failed_targets[:5])
            raise ValueError(
                f"{Path(path).name} 有 {len(failed_targets)} 个 LoRA 目标无法应用：{shown}"
            )
        if not applied_count:
            endings = sorted({".".join(key.split(".")[-3:]) for key in weights})[:8]
            raise ValueError(
                f"{Path(path).name} 没有任何层匹配 MiniMax-H3；文件键尾示例：{endings}"
            )
        unmatched = set(weights) - matched_keys
        print(
            f"[MLX LoRA] minimax_h3/{role}: {Path(path).name} 已应用到 {applied_count} 层"
            f"（解码 {quantized_count} 个 Comfy INT8 矩阵，未匹配键 {len(unmatched)}）"
        )
        if unmatched:
            print("[MLX LoRA] 未匹配键示例：" + ", ".join(sorted(unmatched)[:5]))
        # 逐文件完成求值，避免多 LoRA 时让后续文件继续持有本文件的解码懒图和 int8 源数组。
        from comfyui_mlx_gen.transformer_lora import _validate_lora_layers

        _validate_lora_layers(transformer)
        mx.eval(transformer.parameters())
        resolved_paths.append(path)
        user_scales.append(scale)
        del loaded, raw, weights

    print(f"[MLX LoRA] minimax_h3/{role}: 全部 {len(resolved_paths)} 个适配器应用成功")
    return resolved_paths, user_scales