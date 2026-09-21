"""Streaming component loader for the official Qwen-Image 2.1 checkpoint.

The 13 GiB transformer and 16 GiB text encoder must not follow mflux's
``load-all -> quantize-copy`` path.  This loader constructs the final module
shape first, then consumes one safetensors shard at a time and immediately
converts/quantizes every tensor before releasing the shard.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from .text_encoder import QwenImage21TextEncoder, load_tokenizer
from .transformer import QwenImage21Transformer
from .vae import QwenImage21VAE

GROUP_SIZE = 64
EVAL_CHUNK = 8
_DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
_FLOAT_DTYPES = (mx.bfloat16, mx.float16, mx.float32)


@dataclass(frozen=True)
class LoadedQwenImage21Component:
    role: str
    module: Any
    bits: int | None
    dtype: str
    path: str


def read_config(path: str | Path) -> dict[str, Any]:
    file = Path(path) / "config.json"
    if not file.is_file():
        raise FileNotFoundError(f"Qwen-Image 2.1 组件目录缺少 config.json: {file}")
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Qwen-Image 2.1 配置不是合法 JSON（{file}）: {exc}") from exc


def build_model(role: str, path: str | Path) -> nn.Module:
    config = read_config(path)
    if role == "transformer":
        return QwenImage21Transformer(
            patch_size=int(config.get("patch_size", 1)),
            in_channels=int(config.get("in_channels", 64)),
            out_channels=int(config.get("out_channels", 64)),
            num_layers=int(config.get("num_layers", 32)),
            attention_head_dim=int(config.get("attention_head_dim", 128)),
            num_attention_heads=int(config.get("num_attention_heads", 32)),
            context_in_dim=int(config.get("context_in_dim", 4096)),
            mlp_ratio=int(config.get("mlp_ratio", 3)),
            axes_dims_rope=tuple(int(v) for v in config.get("axes_dims_rope", (16, 56, 56))),
            eps=float(config.get("eps", 1e-6)),
            causal_condition=bool(config.get("causal_condition", True)),
        )
    if role == "text_encoder":
        text = dict(config.get("text_config") or {})
        vision_config = dict(config.get("vision_config") or {})
        rope = dict(text.get("rope_scaling") or {})
        return QwenImage21TextEncoder(
            vocab_size=int(text.get("vocab_size", 151936)),
            hidden_size=int(text.get("hidden_size", 4096)),
            num_hidden_layers=int(text.get("num_hidden_layers", 36)),
            num_attention_heads=int(text.get("num_attention_heads", 32)),
            num_key_value_heads=int(text.get("num_key_value_heads", 8)),
            head_dim=int(text.get("head_dim", 128)),
            intermediate_size=int(text.get("intermediate_size", 12288)),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(text.get("rope_theta", 5_000_000.0)),
            mrope_section=tuple(int(v) for v in rope.get("mrope_section", (24, 20, 20))),
            vision_config=vision_config,
        )
    if role == "vae":
        return QwenImage21VAE(
            base_dim=int(config.get("base_dim", 96)),
            decoder_base_dim=int(config.get("decoder_base_dim", config.get("base_dim", 96))),
            z_dim=int(config.get("z_dim", 64)),
            dim_mult=tuple(int(v) for v in config.get("dim_mult", (1, 2, 4, 8, 8))),
            num_res_blocks=int(config.get("num_res_blocks", 2)),
            temperal_downsample=tuple(
                bool(v) for v in config.get("temperal_downsample", (False, True, True, True))
            ),
            out_channels=int(config.get("out_channels", 4)),
            in_channels=int(config.get("in_channels", 4)),
            latents_mean=tuple(float(v) for v in config.get("latents_mean", ())),
            latents_std=tuple(float(v) for v in config.get("latents_std", ())),
        )
    raise ValueError(f"未知 Qwen-Image 2.1 组件 role: {role}")


def _source_to_local(role: str, key: str, tensor: mx.array) -> tuple[str, mx.array] | None:
    if role == "text_encoder":
        vision_prefix = "model.visual."
        if key.startswith(vision_prefix):
            local = "visual." + key[len(vision_prefix):]
            if local == "visual.patch_embed.proj.weight":
                tensor = tensor.reshape(tensor.shape[0], -1)
            return local, tensor
        prefix = "model.language_model."
        if not key.startswith(prefix):
            return None
        local = key[len(prefix):]
        if local == "embed_tokens.weight":
            return local, tensor
        if not local.startswith("layers."):
            # The transformer consumes the last decoder layer's pre-norm output;
            # language_model.norm and lm_head are not part of the conditioner.
            return None
        try:
            if int(local.split(".")[1]) >= 36:
                return None
        except (IndexError, ValueError):
            return None
        return local, tensor
    if role == "vae":
        if not key.startswith(("encoder.", "quant_conv.", "decoder.", "post_quant_conv.")):
            return None
        if ".time_conv." in key:
            # Single-frame first-chunk decoding never calls temporal convolutions.
            return None
        if key.endswith(".gamma"):
            tensor = tensor.reshape(tensor.shape[0])
        elif key.endswith(".weight") and tensor.ndim == 4:
            # torch Conv2d (out,in,kH,kW) -> MLX (out,kH,kW,in)
            tensor = tensor.transpose(0, 2, 3, 1)
        return key, tensor
    if role == "transformer":
        return key, tensor
    return None


def _shards(path: str | Path) -> list[Path]:
    root = Path(path)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"Qwen-Image 2.1 目录没有 safetensors: {root}")
    return files


def _quantizable(_path: str, module: nn.Module) -> bool:
    if not isinstance(module, (nn.Linear, nn.Embedding)):
        return False
    weight = getattr(module, "weight", None)
    return weight is not None and int(weight.shape[-1]) % GROUP_SIZE == 0


def _eval(entries: list[tuple[str, mx.array]]) -> None:
    values = [value for _, value in entries]
    for start in range(0, len(values), EVAL_CHUNK):
        mx.eval(*values[start:start + EVAL_CHUNK])
        mx.synchronize()


def _quantize_entry(
    key: str,
    tensor: mx.array,
    quantized_modules: dict[str, nn.Module],
) -> list[tuple[str, mx.array]]:
    parent, _, leaf = key.rpartition(".")
    module = quantized_modules.get(parent)
    if module is None or leaf != "weight":
        return [(key, tensor)]
    weight, scales, biases = mx.quantize(tensor, group_size=GROUP_SIZE, bits=int(module.bits))
    values = [(key, weight), (f"{parent}.scales", scales)]
    if biases is not None:
        values.append((f"{parent}.biases", biases))
    return values


def load(
    role: str,
    path: str,
    quantize: int | None = None,
    precision: str = "bfloat16",
    log: Callable[[str], None] | None = print,
) -> LoadedQwenImage21Component:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Qwen-Image 2.1 {role} 目录不存在: {root}")
    if precision not in _DTYPES:
        raise ValueError(f"未知精度 {precision}（可选：{', '.join(_DTYPES)}）")
    bits = int(quantize) if quantize else None
    if bits not in (None, 4, 8):
        raise ValueError(f"Qwen-Image 2.1 在线量化只支持 4 / 8 位或 0，收到 {quantize}")
    # Convolutional VAE has no Linear/Embedding worth quantizing; retaining q4/q8 in its
    # handle is harmless, but claiming it was quantized would be misleading.
    if role == "vae":
        bits = None

    module = build_model(role, root)
    expected = {key: tuple(value.shape) for key, value in tree_flatten(module.parameters())}
    if bits:
        nn.quantize(module, group_size=GROUP_SIZE, bits=bits, class_predicate=_quantizable)
    quantized_modules = {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    }
    dtype = _DTYPES[precision]
    loaded: dict[str, tuple[int, ...]] = {}

    for shard in _shards(root):
        if log:
            log(f"[Qwen-Image 2.1 加载] {role}: {shard.name}")
        raw = mx.load(str(shard))
        entries: list[tuple[str, mx.array]] = []
        for source_key, source_tensor in raw.items():
            converted = _source_to_local(role, source_key, source_tensor)
            if converted is None:
                continue
            key, tensor = converted
            if key not in expected:
                raise ValueError(
                    f"Qwen-Image 2.1 {role} 权重存在本地模块没有的键 {key}（来自 {source_key}）"
                )
            shape = tuple(tensor.shape)
            if shape != expected[key]:
                raise ValueError(
                    f"Qwen-Image 2.1 {role} 形状不符: {key} checkpoint={shape}, module={expected[key]}"
                )
            if key in loaded:
                raise ValueError(f"Qwen-Image 2.1 {role} 权重跨 shard 重复: {key}")
            if tensor.dtype in _FLOAT_DTYPES:
                tensor = tensor.astype(dtype)
            entries.extend(_quantize_entry(key, tensor, quantized_modules))
            loaded[key] = shape
        if entries:
            module.update(tree_unflatten(entries), strict=False)
            _eval(entries)
        del raw, entries
        gc.collect()
        mx.clear_cache()

    missing = sorted(set(expected) - set(loaded))
    if missing:
        raise ValueError(
            f"Qwen-Image 2.1 {role} 缺 {len(missing)} 个参数，前几个是 {missing[:5]}；"
            "请确认组件软链指向官方 Qwen-Image-2.1 对应子目录"
        )
    if log:
        log(
            f"[Qwen-Image 2.1 加载] {role} 完成：{len(loaded)} 个源张量，"
            f"{precision}{f' / q{bits}' if bits else ''}"
        )
    return LoadedQwenImage21Component(role, module, bits, precision, str(root))


__all__ = ["LoadedQwenImage21Component", "build_model", "load", "load_tokenizer", "read_config"]