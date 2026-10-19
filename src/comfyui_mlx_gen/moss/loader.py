"""Streaming, schema-checked loaders for the official MOSS component layout."""

from __future__ import annotations

import gc
import inspect
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from .config import MOSSConfig, text_encoder_config
from .transformer import MOSSImageTransformer

GROUP_SIZE = 64
_DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}


class MOSSQwenRMSNorm(nn.Module):
    """Qwen3.5's zero-centered RMSNorm, not MLX's multiplicative RMSNorm.

    Transformers initializes this layer's parameter at zero and applies
    ``normalized * (1 + weight)``.  The released checkpoint therefore contains
    values around zero.  Using ``mlx.nn.RMSNorm`` directly would multiply by
    those small values and collapse every Qwen3.5 hidden state before it reaches
    the MOSS DiT.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        normalized = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + self.eps)
        return (normalized * (1.0 + self.weight.astype(mx.float32))).astype(x.dtype)


@dataclass(frozen=True)
class LoadedMOSSComponent:
    role: str
    module: Any
    path: str
    precision: str
    quantize: int | None = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"MOSS JSON 不是合法配置: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"MOSS JSON 顶层必须是 object: {path}")
    return value


def _component_dir(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if root.is_file():
        root = root.parent
    if not root.is_dir():
        raise FileNotFoundError(f"MOSS 组件目录不存在: {root}")
    return root.resolve()


def _index_files(root: Path, stem: str) -> tuple[list[Path], dict[str, str]]:
    index_path = root / f"{stem}.safetensors.index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"MOSS 权重索引缺少非空 weight_map: {index_path}")
        shards = sorted(set(str(value) for value in weight_map.values()))
        paths = []
        for name in shards:
            relative = Path(name)
            # HF snapshots commonly make component files symlinks into the
            # cache's blobs/ directory. Validate the *logical* path here, then
            # resolve the symlink only for opening; rejecting the resolved
            # parent would incorrectly reject a valid HF cache layout.
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
                raise ValueError(f"MOSS shard 路径越界: {name}")
            logical = root / relative
            if not logical.is_file():
                raise FileNotFoundError(f"MOSS 权重 shard 不存在: {logical}")
            paths.append(logical.resolve())
        return paths, {str(key): str(value) for key, value in weight_map.items()}
    single = root / f"{stem}.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"MOSS 缺少 {stem}.safetensors 或对应 index: {root}")
    return [single.resolve()], {}


def _header_keys(path: Path) -> list[str]:
    """Read only a safetensors header; never map the tensor payload."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"safetensors header 长度字段不完整: {path}")
        header_size = struct.unpack("<Q", raw)[0]
        if header_size <= 0 or header_size > size - 8:
            raise ValueError(f"safetensors header 长度非法: {path}")
        header = json.loads(handle.read(header_size).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header 不是 object: {path}")
    return [key for key in header if key != "__metadata__"]


def validate_index(root: str | Path, stem: str) -> tuple[list[Path], dict[str, str]]:
    """Validate shard names and exact key ownership without reading tensor data."""
    directory = _component_dir(root)
    shards, weight_map = _index_files(directory, stem)
    if not weight_map:
        return shards, weight_map
    actual: dict[str, str] = {}
    resolved_to_logical = {
        (directory / name).resolve(): name for name in set(weight_map.values())
    }
    for shard in shards:
        logical_name = resolved_to_logical[shard]
        for key in _header_keys(shard):
            if key in actual:
                raise ValueError(f"MOSS shard 参数重复: {key}")
            actual[key] = logical_name
    if actual != weight_map:
        missing = sorted(set(weight_map) - set(actual))
        extra = sorted(set(actual) - set(weight_map))
        wrong = sorted(key for key in set(actual) & set(weight_map) if actual[key] != weight_map[key])
        raise ValueError(
            f"MOSS shard 与 index 不一致: missing={missing[:5]}, extra={extra[:5]}, "
            f"wrong_shard={wrong[:5]}"
        )
    return shards, weight_map


def _load_shards(
    root: Path,
    stem: str,
    transform: Callable[[str, mx.array], tuple[str, mx.array] | None],
    expected: dict[str, tuple[int, ...]],
    dtype: mx.Dtype,
) -> list[tuple[str, mx.array]]:
    shards, weight_map = _index_files(root, stem)
    loaded: dict[str, mx.array] = {}
    resolved_to_logical = {
        (root / name).resolve(): name for name in set(weight_map.values())
    }
    for shard in shards:
        raw = mx.load(str(shard), format="safetensors")
        try:
            for source_key, tensor in raw.items():
                if weight_map and weight_map.get(source_key) != resolved_to_logical[shard]:
                    raise ValueError(f"MOSS shard key 不在 index 或归属错误: {source_key}")
                target = transform(source_key, tensor)
                if target is None:
                    continue
                key, value = target
                if key in loaded:
                    raise ValueError(f"MOSS 参数重复: {key}")
                if key not in expected:
                    raise ValueError(f"MOSS checkpoint 存在本地模块没有的 key: {source_key} -> {key}")
                if tuple(value.shape) != expected[key]:
                    raise ValueError(
                        f"MOSS 参数 shape 不符: {source_key} -> {key}, "
                        f"checkpoint={tuple(value.shape)}, module={expected[key]}"
                    )
                loaded[key] = value.astype(dtype) if value.dtype in (mx.float16, mx.bfloat16, mx.float32) else value
        finally:
            del raw
        gc.collect()
    missing = sorted(set(expected) - set(loaded))
    if missing:
        raise ValueError(f"MOSS checkpoint 缺少 {len(missing)} 个本地参数，前几个是 {missing[:8]}")
    return list(loaded.items())


def _module_expected(module: nn.Module) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in tree_flatten(module.parameters())}


def _transformer_key(key: str, _tensor: mx.array) -> tuple[str, mx.array] | None:
    if key == "rope.bands":
        return None
    return key, _tensor


def build_transformer(path: str | Path, activation_dtype: mx.Dtype = mx.bfloat16) -> MOSSImageTransformer:
    return MOSSImageTransformer(MOSSConfig.from_path(path), activation_dtype=activation_dtype)


def load_transformer(
    path: str | Path,
    precision: str = "bfloat16",
    quantize: int | None = None,
    log: Callable[[str], None] | None = print,
) -> LoadedMOSSComponent:
    if precision not in _DTYPES:
        raise ValueError(f"未知 MOSS precision: {precision}")
    root = _component_dir(path)
    validate_index(root, "diffusion_pytorch_model")
    module = build_transformer(root, activation_dtype=_DTYPES[precision])
    expected = _module_expected(module)
    entries = _load_shards(root, "diffusion_pytorch_model", _transformer_key, expected, _DTYPES[precision])
    if quantize:
        if int(quantize) not in (4, 8):
            raise ValueError("MOSS transformer 只支持 q4 / q8 或 0")
        nn.quantize(module, group_size=GROUP_SIZE, bits=int(quantize))
        entries = _quantize_entries(entries, module, int(quantize))
    module.update(tree_unflatten(entries), strict=False)
    mx.eval(*[value for _, value in entries])
    if log:
        log(f"[MOSS] transformer 加载完成：{len(entries)} 个参数，{root}")
    return LoadedMOSSComponent("transformer", module, str(root), precision, quantize)


def _quantize_entries(entries: list[tuple[str, mx.array]], module: nn.Module, bits: int) -> list[tuple[str, mx.array]]:
    modules = dict(module.named_modules())
    result: list[tuple[str, mx.array]] = []
    for key, value in entries:
        parent, _, leaf = key.rpartition(".")
        target = modules.get(parent)
        if target is not None and leaf == "weight" and isinstance(target, nn.QuantizedLinear):
            packed, scales, biases = mx.quantize(value, group_size=GROUP_SIZE, bits=bits)
            result.extend([(key, packed), (f"{parent}.scales", scales)])
            if biases is not None:
                result.append((f"{parent}.biases", biases))
        else:
            result.append((key, value))
    return result


def _qwen35_kwargs(config: dict[str, Any]) -> tuple[Any, Any]:
    from mlx_vlm.models.qwen3_5.config import TextConfig

    text = dict(config["text_config"])
    accepted = inspect.signature(TextConfig).parameters
    text = {key: value for key, value in text.items() if key in accepted}
    return TextConfig(**text), config


def build_text_encoder(path: str | Path) -> Any:
    config = text_encoder_config(path)
    text_config, _ = _qwen35_kwargs(config)
    from mlx_vlm.models.qwen3_5.language import Qwen3_5Model

    module = Qwen3_5Model(text_config)
    # Qwen3.5 uses zero-centered RMSNorm weights in all ordinary decoder norms
    # and attention q/k norms.  Replace MLX's standard multiplicative modules
    # before the checkpoint is loaded so expected keys remain unchanged.
    def zero_centered(value):
        replacement = MOSSQwenRMSNorm(value.weight.shape[0], value.eps)
        replacement.weight = value.weight
        return replacement

    module.norm = zero_centered(module.norm)
    for layer in module.layers:
        layer.input_layernorm = zero_centered(layer.input_layernorm)
        layer.post_attention_layernorm = zero_centered(layer.post_attention_layernorm)
        if not layer.is_linear:
            layer.self_attn.q_norm = zero_centered(layer.self_attn.q_norm)
            layer.self_attn.k_norm = zero_centered(layer.self_attn.k_norm)
    return module


def _text_key(key: str, tensor: mx.array) -> tuple[str, mx.array] | None:
    if key.startswith("language_model."):
        local = key[len("language_model.") :]
        if (
            local.endswith(".linear_attn.conv1d.weight")
            and tensor.ndim == 3
            and tensor.shape[1] == 1
            and tensor.shape[2] != 1
        ):
            # PyTorch Conv1d stores depthwise kernels as [out, in/groups, kernel],
            # while MLX Conv1d expects [out, kernel, in/groups].  Qwen3.5's
            # linear-attention convolution is depthwise, so the official MOSS
            # checkpoint shape [8192, 1, 4] must become [8192, 4, 1].
            tensor = tensor.transpose(0, 2, 1)
        # The DiT consumes the decoder output before the final norm. The MLX
        # Qwen3.5 implementation owns the norm parameter, but it is still
        # loaded so the module remains a complete checkpoint representation.
        return local, tensor
    # MOSS text conditioning is text-only; the Qwen3.5 vision tower is not
    # part of this pipeline and must never be silently dropped during load.
    if key.startswith("visual."):
        # The official Qwen3.5 checkpoint is multimodal, but MOSS-SFT uses its
        # text-only hidden states.  Drop the unused vision tower explicitly;
        # do not pretend it was loaded into the text-only MLX module.
        return None
    return None


def load_text_encoder(
    path: str | Path,
    precision: str = "bfloat16",
    log: Callable[[str], None] | None = print,
) -> LoadedMOSSComponent:
    if precision not in _DTYPES:
        raise ValueError(f"未知 MOSS text encoder precision: {precision}")
    root = _component_dir(path)
    validate_index(root, "model")
    module = build_text_encoder(root)
    expected = _module_expected(module)
    entries = _load_shards(root, "model", _text_key, expected, _DTYPES[precision])
    module.update(tree_unflatten(entries), strict=False)
    mx.eval(*[value for _, value in entries])
    if log:
        log(f"[MOSS] Qwen3.5 text encoder 加载完成：{len(entries)} 个参数，{root}")
    return LoadedMOSSComponent("text_encoder", module, str(root), precision)


def load_tokenizer(path: str | Path, *, max_length: int = 4096) -> Any:
    root = _component_dir(path)
    if not (root / "tokenizer.json").is_file():
        raise FileNotFoundError(
            f"MOSS tokenizer 缺少 tokenizer.json: {root}。"
            "当前共享目录的 processor 软链接可能是断链；请将官方 tokenizer 文件放入该目录。"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    tokenizer.model_max_length = int(max_length)
    return tokenizer