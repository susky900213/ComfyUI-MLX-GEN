"""Streaming component loader for the official Qwen-Image 2.1 checkpoint.

The 13 GiB transformer and 16 GiB text encoder must not follow mflux's
``load-all -> quantize-copy`` path.  This loader constructs the final module
shape first, then consumes one safetensors shard at a time and immediately
converts/quantizes every tensor before releasing the shard.
"""

from __future__ import annotations

import gc
import json
import struct
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
_SAFETENSORS_FLOAT_DTYPES = {"F16", "BF16", "F32"}
_SAFETENSORS_ITEMSIZE = {"F16": 2, "BF16": 2, "F32": 4}

# Zabin/Qwen_Image_2.1_Turbo_8Steps is a transformer-only export.  Unlike the
# directory export, it has no config.json, so these values must be kept here
# rather than silently asking an unrelated mflux ModelConfig for defaults.
DEFAULT_TRANSFORMER_CONFIG: dict[str, Any] = {
    "patch_size": 1,
    "in_channels": 64,
    "out_channels": 64,
    "num_layers": 32,
    "attention_head_dim": 128,
    "num_attention_heads": 32,
    "context_in_dim": 4096,
    "mlp_ratio": 3,
    "axes_dims_rope": (16, 56, 56),
    "eps": 1e-6,
    "causal_condition": True,
}


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


def _transformer_config(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if root.is_file():
        return dict(DEFAULT_TRANSFORMER_CONFIG)
    return read_config(root)


def build_model(role: str, path: str | Path) -> nn.Module:
    config = _transformer_config(path) if role == "transformer" else read_config(path)
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


def _read_safetensors_header(path: str | Path) -> tuple[dict[str, Any], int]:
    """读取 safetensors header，不把任何 tensor data 映射进内存。"""
    file = Path(path)
    try:
        size = file.stat().st_size
        with file.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise ValueError("header 长度字段不完整")
            header_size = struct.unpack("<Q", raw_size)[0]
            if header_size <= 0 or header_size > size - 8:
                raise ValueError(f"header 长度非法：{header_size}")
            raw_header = handle.read(header_size)
            if len(raw_header) != header_size:
                raise ValueError("header 内容不完整")
        header = json.loads(raw_header.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error, ValueError) as exc:
        raise ValueError(f"无法读取 safetensors header {file}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header 不是 object：{file}")
    return header, header_size


def _validate_single_file_metadata(header: dict[str, Any], path: Path) -> None:
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError(f"Qwen-Image 2.1 单文件 metadata 必须是 string object：{path}")
    # Zabin 的 Int8 Convrot layout 尚未在 MLX 侧验证。不要把带量化 recipe 的
    # 文件当作普通 F16/BF16 权重静默加载；这种文件必须先有专门的转换器。
    markers = ("quant", "int8", "convrot", "bits", "group_size", "scales", "biases")
    metadata_text = json.dumps(metadata, ensure_ascii=False).casefold()
    if any(marker in metadata_text for marker in markers):
        raise ValueError(
            f"Qwen-Image 2.1 单文件包含未支持的量化/Convrot metadata：{path}"
        )


def _single_transformer_mapping(
    source_key: str,
    source_shape: tuple[int, ...],
    expected: dict[str, tuple[int, ...]],
) -> tuple[tuple[str, ...], str]:
    """Return target keys and mapping kind for one ComfyUI source key."""
    prefix = "model.diffusion_model."
    if not source_key.startswith(prefix):
        raise ValueError(
            "Qwen-Image 2.1 单文件只能包含 model.diffusion_model.* transformer 权重，"
            f"收到 {source_key}"
        )
    local = source_key[len(prefix):]
    if local.endswith(".img_mlp.gate_up.weight"):
        # The Turbo single-file export uses the fused SwiGLU input projection:
        # [gate; up].  The MLX module keeps the two Linear layers separate so
        # that its forward pass can apply SiLU(gate) * up.
        base = local[: -len(".gate_up.weight")]
        gate_key = base + ".gate_layer.weight"
        proj_key = base + ".proj.weight"
        if len(source_shape) != 2:
            raise ValueError(
                f"Qwen-Image 2.1 fused gate_up 必须是二维权重：{source_key}={source_shape}"
            )
        if gate_key not in expected or proj_key not in expected:
            raise ValueError(f"Qwen-Image 2.1 单文件存在本地模块没有的键：{source_key}")
        gate_shape = expected[gate_key]
        proj_shape = expected[proj_key]
        if gate_shape != proj_shape or source_shape != (gate_shape[0] * 2, gate_shape[1]):
            raise ValueError(
                f"Qwen-Image 2.1 fused gate_up 形状不符: {source_key} checkpoint={source_shape}, "
                f"module gate={gate_shape}, up={proj_shape}"
            )
        return (gate_key, proj_key), "gate_up"
    if local.endswith(".img_mlp.net.0.proj.weight"):
        # ComfyUI's Qwen Image FeedForward is GELU, not a fused GEGLU: net.0
        # is the input projection and net.2 is the separate output projection.
        local = local[: -len(".net.0.proj.weight")] + ".proj.weight"
    elif local.endswith(".img_mlp.net.2.weight"):
        local = local[: -len(".net.2.weight")] + ".out.weight"
    key = local
    if key not in expected:
        raise ValueError(f"Qwen-Image 2.1 单文件存在本地模块没有的键：{source_key}")
    if source_shape != expected[key]:
        raise ValueError(
            f"Qwen-Image 2.1 单文件形状不符: {source_key} checkpoint={source_shape}, "
            f"module={expected[key]}"
        )
    return (key,), "direct"


def _validate_single_transformer_header(
    header: dict[str, Any],
    expected: dict[str, tuple[int, ...]],
    path: Path,
    header_size: int,
) -> dict[str, tuple[tuple[str, ...], str]]:
    """Validate source schema and return a tensor-data-free conversion plan."""
    _validate_single_file_metadata(header, path)
    data_size = path.stat().st_size - 8 - header_size
    if data_size < 0:
        raise ValueError(f"Qwen-Image 2.1 单文件 header 超出文件：{path}")
    plan: dict[str, tuple[tuple[str, ...], str]] = {}
    target_sources: dict[str, str] = {}
    block_indexes: set[int] = set()
    data_ranges: list[tuple[int, int, str]] = []
    for source_key, spec in header.items():
        if source_key == "__metadata__":
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"Qwen-Image 2.1 单文件 tensor header 不是 object：{source_key}")
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        offsets = spec.get("data_offsets")
        if dtype not in _SAFETENSORS_FLOAT_DTYPES:
            raise ValueError(
                f"Qwen-Image 2.1 单文件只支持 F16/BF16/F32，{source_key} 是 {dtype!r}"
            )
        if (
            not isinstance(shape, list)
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > data_size
        ):
            raise ValueError(f"Qwen-Image 2.1 单文件 tensor header 非法：{source_key}")
        source_shape = tuple(shape)
        expected_bytes = _SAFETENSORS_ITEMSIZE[dtype]
        for dimension in source_shape:
            expected_bytes *= dimension
        if offsets[1] - offsets[0] != expected_bytes:
            raise ValueError(
                f"Qwen-Image 2.1 单文件 tensor 字节数不符：{source_key}，"
                f"header={offsets[1] - offsets[0]}，expected={expected_bytes}"
            )
        data_ranges.append((offsets[0], offsets[1], source_key))
        targets, kind = _single_transformer_mapping(source_key, source_shape, expected)
        if source_key in plan:
            raise ValueError(f"Qwen-Image 2.1 单文件键重复：{source_key}")
        plan[source_key] = (targets, kind)
        if source_key.startswith("model.diffusion_model.transformer_blocks."):
            parts = source_key.split(".")
            if len(parts) <= 3 or parts[2] != "transformer_blocks" or not parts[3].isdigit():
                raise ValueError(f"Qwen-Image 2.1 单文件 block 键非法：{source_key}")
            block_indexes.add(int(parts[3]))
        for target in targets:
            previous = target_sources.get(target)
            if previous is not None:
                raise ValueError(
                    f"Qwen-Image 2.1 单文件多个 source 键映射到同一参数 {target}："
                    f"{previous} 与 {source_key}"
                )
            target_sources[target] = source_key
    previous_end = 0
    previous_key = ""
    for start, end, source_key in sorted(data_ranges):
        if start < previous_end:
            raise ValueError(
                f"Qwen-Image 2.1 单文件 tensor data_offsets 重叠：{previous_key} 与 {source_key}"
            )
        previous_end = max(previous_end, end)
        previous_key = source_key

    expected_blocks = {
        int(parts[1])
        for key in expected
        if (parts := key.split("."))[:1] == ["transformer_blocks"]
        and len(parts) > 2
        and parts[1].isdigit()
    }
    if block_indexes != expected_blocks:
        missing = sorted(expected_blocks - block_indexes)
        extra = sorted(block_indexes - expected_blocks)
        raise ValueError(
            f"Qwen-Image 2.1 单文件 transformer block 结构不完整：missing={missing[:5]}, extra={extra[:5]}"
        )
    missing = sorted(set(expected) - set(target_sources))
    if missing:
        raise ValueError(
            f"Qwen-Image 2.1 单文件缺 {len(missing)} 个 transformer 参数，前几个是 {missing[:5]}"
        )
    return plan


def _single_transformer_entries(
    source_key: str,
    tensor: mx.array,
    expected: dict[str, tuple[int, ...]],
) -> list[tuple[str, mx.array]]:
    targets, kind = _single_transformer_mapping(source_key, tuple(tensor.shape), expected)
    if kind == "gate_up":
        gate, proj = mx.split(tensor, 2, axis=0)
        return [(targets[0], gate), (targets[1], proj)]
    return [(targets[0], tensor)]


def _load_single_transformer(
    path: Path,
    quantize: int | None,
    precision: str,
    log: Callable[[str], None] | None,
) -> LoadedQwenImage21Component:
    # Hugging Face snapshot files can resolve to extensionless blobs/<sha256>
    # paths.  Validate the safetensors header instead of trusting the filename;
    # _read_safetensors_header also gives explicit errors for empty/truncated or
    # otherwise malformed files.
    header, header_size = _read_safetensors_header(path)
    module = build_model("transformer", path)
    expected = {key: tuple(value.shape) for key, value in tree_flatten(module.parameters())}
    plan = _validate_single_transformer_header(header, expected, path, header_size)

    if precision not in _DTYPES:
        raise ValueError(f"未知精度 {precision}（可选：{', '.join(_DTYPES)}）")

    bits = int(quantize) if quantize else None
    if bits not in (None, 4, 8):
        raise ValueError(f"Qwen-Image 2.1 在线量化只支持 4 / 8 位或 0，收到 {quantize}")
    if bits:
        nn.quantize(module, group_size=GROUP_SIZE, bits=bits, class_predicate=_quantizable)
    quantized_modules = {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    }
    dtype = _DTYPES[precision]
    loaded: set[str] = set()
    if log:
        log(f"[Qwen-Image 2.1 加载] transformer 单文件: {path.name}")
    raw = mx.load(str(path))
    entries: list[tuple[str, mx.array]] = []
    for source_key, source_tensor in raw.items():
        if source_key not in plan:
            raise ValueError(f"Qwen-Image 2.1 单文件 header/data 不一致：{source_key}")
        for key, tensor in _single_transformer_entries(source_key, source_tensor, expected):
            if key in loaded:
                raise ValueError(f"Qwen-Image 2.1 单文件参数重复：{key}")
            if tensor.dtype in _FLOAT_DTYPES:
                tensor = tensor.astype(dtype)
            entries.extend(_quantize_entry(key, tensor, quantized_modules))
            loaded.add(key)
    if set(expected) != loaded:
        missing = sorted(set(expected) - loaded)
        raise ValueError(f"Qwen-Image 2.1 单文件数据缺参数，前几个是 {missing[:5]}")
    module.update(tree_unflatten(entries), strict=False)
    _eval(entries)
    del raw, entries
    gc.collect()
    mx.clear_cache()
    if log:
        log(
            f"[Qwen-Image 2.1 加载] transformer 单文件完成：{len(loaded)} 个参数，"
            f"{precision}{f' / q{bits}' if bits else ''}"
        )
    return LoadedQwenImage21Component("transformer", module, bits, precision, str(path))


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
    if precision not in _DTYPES:
        raise ValueError(f"未知精度 {precision}（可选：{', '.join(_DTYPES)}）")
    bits = int(quantize) if quantize else None
    if bits not in (None, 4, 8):
        raise ValueError(f"Qwen-Image 2.1 在线量化只支持 4 / 8 位或 0，收到 {quantize}")
    if root.is_file():
        if role != "transformer":
            raise ValueError(
                f"Qwen-Image 2.1 单文件只支持 transformer role，收到 {role}: {root}"
            )
        return _load_single_transformer(root, bits, precision, log)
    if not root.is_dir():
        raise FileNotFoundError(f"Qwen-Image 2.1 {role} 目录不存在: {root}")
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