"""Native loaders for the MOSS-Image-SFT Transformer LoRAs.

MOSS has two released adapter layouts.  The original 4-step adapter contains
one ``alpha`` scalar and a dynamically resized ``lora_down`` / ``lora_up`` pair
for each of the six projections in all 42 MOSS image blocks.  The regular SFT
adapter is a fixed-rank PEFT file with ``lora_A`` / ``lora_B`` keys.  Neither
layout matches mflux's generic image mappings, so keep both implementations
local to MOSS.

Adapters are deliberately kept unmerged: every targeted dense Linear becomes
``base(x) + (x @ down.T @ up.T) * alpha / rank``.  This preserves the base
checkpoint and makes the cache key / model handle the sole source of adapter
identity.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn

from ..transformer_lora import active_loras, resolve_lora_path
from ..types import LoraRef

MOSS_LORA_FILENAME_RE = re.compile(
    r"(?:^|[_-])4[_-]?steps?(?:[_-]|\.|$)",
    re.IGNORECASE,
)
MOSS_LORA_KEY_RE = re.compile(
    r"^diffusion_model\.layers\.(\d+)\.proj\."
    r"(q|k|v|o|ffn_up_gate|ffn_down)\."
    r"(alpha|lora_down\.weight|lora_up\.weight)$"
)
MOSS_STANDARD_LORA_KEY_RE = re.compile(
    r"^(?:(?:base_model\.model\.)?(?:transformer\.|diffusion_model\.)|"
    r"base_model\.model\.|transformer\.|diffusion_model\.)?layers\."
    r"(\d+)\.proj\."
    r"(q|k|v|o|ffn_up_gate|ffn_down)\."
    r"lora_(A|B)(?:(?:\.default)?(?:\.weight)?)?$"
)
MOSS_LORA_PROJECTIONS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
MOSS_LORA_LAYERS = 42
MOSS_LORA_MAX_RANK = 128
MOSS_IMAGE_EDIT_FORMAT = "moss-edit-lora"

_MOSS_MODULE_RE = re.compile(
    r"^(?:(?:base_model\.model\.)?(?:transformer\.|diffusion_model\.)|"
    r"base_model\.model\.|transformer\.|diffusion_model\.)?layers\."
    r"(\d+)\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)$"
)


@dataclass(frozen=True)
class MOSSLoRATarget:
    layer: int
    projection: str
    rank: int
    # Header validation can only verify that alpha is scalar.  Its value is
    # read from the payload when the adapter is attached.
    alpha: float | None
    down_shape: tuple[int, int]
    up_shape: tuple[int, int]


def _expected_shapes(projection: str) -> tuple[int, int]:
    if projection == "ffn_down":
        return 8192, 3072
    if projection == "ffn_up_gate":
        return 3072, 16384
    return 3072, 3072


def _header(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only the safetensors header; no payload is mapped or loaded."""
    file = Path(path)
    try:
        size = file.stat().st_size
        with file.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise ValueError("header size field is incomplete")
            header_size = struct.unpack("<Q", raw_size)[0]
            if header_size <= 0 or header_size > size - 8:
                raise ValueError(f"invalid header size {header_size}")
            raw_header = handle.read(header_size)
    except OSError as exc:
        raise ValueError(f"无法读取 MOSS 4-step LoRA 文件头 {file}: {exc}") from exc
    try:
        value = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"MOSS 4-step LoRA header 不是合法 JSON: {file}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"MOSS 4-step LoRA header 必须是 object: {file}")
    metadata = value.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"MOSS 4-step LoRA metadata 必须是 object: {file}")
    return value, metadata


def _target_key(layer: int, projection: str, suffix: str) -> str:
    return f"diffusion_model.layers.{layer}.proj.{projection}.{suffix}"


def _standard_target_key(layer: int, projection: str, matrix: str) -> str:
    return f"transformer.layers.{layer}.proj.{projection}.lora_{matrix}.weight"


def _standard_key_candidates(layer: int, projection: str, matrix: str) -> tuple[str, ...]:
    """Return the PEFT spellings accepted by the native MOSS loader.

    The converted image-edit adapter is intentionally not a byte-for-byte copy
    of the usual PEFT output: it omits both ``transformer.`` and ``.weight``.
    Keep the canonical Diffusers/PEFT spelling first so existing adapters retain
    their established behavior, then accept the unprefixed and ``default``
    variants emitted by different PEFT versions.
    """
    prefixes = (
        "transformer.",
        "",
        "base_model.model.transformer.",
        "base_model.model.",
        "diffusion_model.",
        "base_model.model.diffusion_model.",
    )
    suffixes = (
        f"lora_{matrix}.weight",
        f"lora_{matrix}",
        f"lora_{matrix}.default.weight",
        f"lora_{matrix}.default",
    )
    return tuple(
        f"{prefix}layers.{layer}.proj.{projection}.{suffix}"
        for prefix in prefixes
        for suffix in suffixes
    )


def _standard_key_map(entries: dict[str, Any]) -> dict[tuple[int, str, str], str]:
    """Normalize accepted PEFT tensor names to ``(layer, projection, A/B)``."""
    result: dict[tuple[int, str, str], str] = {}
    for key in entries:
        match = MOSS_STANDARD_LORA_KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"包含非法 MOSS PEFT LoRA key: {key}")
        target = (int(match.group(1)), match.group(2), match.group(3))
        if target in result:
            raise ValueError(
                f"MOSS PEFT LoRA 为 {target} 提供了重复 tensor：{result[target]} 和 {key}"
            )
        result[target] = key
    return result


def _standard_tensor_key(keys: Any, layer: int, projection: str, matrix: str) -> str:
    """Find the actual payload key for one normalized PEFT target."""
    available = set(keys)
    for candidate in _standard_key_candidates(layer, projection, matrix):
        if candidate in available:
            return candidate
    raise KeyError(
        f"missing MOSS PEFT tensor: layers.{layer}.proj.{projection}.lora_{matrix}"
    )


def _adapter_config(file: Path, metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Read a PEFT directory config or the copy embedded in safetensors metadata."""
    config_file = file.parent / "adapter_config.json"
    raw: Any = None
    if config_file.is_file():
        try:
            raw = json.loads(config_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{config_file} 不是合法的 adapter_config.json") from exc
    elif "adapter_config" in metadata:
        raw = metadata["adapter_config"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file.name} 的 adapter_config metadata 不是合法 JSON") from exc
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{file.name} 的 adapter_config 必须是 object")
    return raw


def _module_specs(config: dict[str, Any] | None) -> dict[tuple[int, str], tuple[int, float]]:
    """Normalize per-module rank/alpha declarations from a PEFT config."""
    if config is None:
        return {}
    modules = config.get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("MOSS adapter_config 的 modules 必须是 object")
    specs: dict[tuple[int, str], tuple[int, float]] = {}
    for name, value in modules.items():
        match = _MOSS_MODULE_RE.fullmatch(str(name))
        if match is None or not isinstance(value, dict):
            raise ValueError(f"MOSS adapter_config 包含非法 module: {name}")
        try:
            rank = int(value["rank"])
            alpha = float(value["alpha"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"MOSS adapter_config 的 {name} rank/alpha 无效") from exc
        if not 1 <= rank <= MOSS_LORA_MAX_RANK or not math.isfinite(alpha):
            raise ValueError(f"MOSS adapter_config 的 {name} rank/alpha 无效：rank={rank}, alpha={alpha}")
        specs[(int(match.group(1)), match.group(2))] = (rank, alpha)
    return specs


def is_image_edit_adapter(path: str | Path) -> bool:
    """Return whether a MOSS adapter declares the dedicated image-edit format."""
    file = Path(path)
    config = _adapter_config(file, _header(file)[1])
    return bool(config and config.get("format") == MOSS_IMAGE_EDIT_FORMAT)


def has_image_edit_adapter(refs: tuple[LoraRef, ...]) -> bool:
    """Return whether active refs include the dedicated image-edit adapter."""
    for ref in active_loras(refs):
        path = resolve_lora_path(ref.path)
        if is_image_edit_adapter(path):
            return True
    return False


def validate_header(path: str | Path) -> dict[tuple[int, str], MOSSLoRATarget]:
    """Validate the complete official 42-layer dynamic-rank layout.

    This function intentionally does not load the 327 MB payload.  It is used
    by the node so an incompatible file is rejected before a transformer is
    materialized.
    """
    file = Path(path)
    if file.suffix.lower() != ".safetensors":
        raise ValueError(f"MOSS 4-step LoRA 必须是 .safetensors 文件: {file}")
    if not MOSS_LORA_FILENAME_RE.search(file.name):
        raise ValueError(
            f"{file.name} 没有 4-step 文件标识；MOSS 4-step 节点不会把未知步数的 LoRA 当作 4 步使用"
        )
    entries, metadata = _header(file)
    if metadata.get("ss_network_dim") != "Dynamic" or metadata.get("ss_network_alpha") != "Dynamic":
        raise ValueError(
            f"{file.name} 不是官方 MOSS dynamic-rank LoRA："
            f"ss_network_dim={metadata.get('ss_network_dim')!r}, "
            f"ss_network_alpha={metadata.get('ss_network_alpha')!r}"
        )

    expected_keys = {
        _target_key(layer, projection, suffix)
        for layer in range(MOSS_LORA_LAYERS)
        for projection in MOSS_LORA_PROJECTIONS
        for suffix in ("alpha", "lora_down.weight", "lora_up.weight")
    }
    actual_keys = set(entries)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"{file.name} 的 MOSS LoRA 层覆盖不完整："
            f"missing={missing[:4]}, extra={extra[:4]}"
        )

    targets: dict[tuple[int, str], MOSSLoRATarget] = {}
    for key, value in entries.items():
        match = MOSS_LORA_KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"{file.name} 包含非法 MOSS LoRA key: {key}")
        if value.get("dtype") != "BF16":
            raise ValueError(
                f"{file.name} 的 {key} dtype={value.get('dtype')!r}，只支持 BF16 单文件 LoRA"
            )
        layer = int(match.group(1))
        projection = match.group(2)
        suffix = match.group(3)
        shape = tuple(int(item) for item in value.get("shape", ()))
        if suffix == "alpha":
            if shape:
                raise ValueError(f"{file.name} 的 {key} 必须是标量 alpha，收到 shape={shape}")
            continue
        if len(shape) != 2:
            raise ValueError(f"{file.name} 的 {key} 必须是二维矩阵，收到 shape={shape}")

        old = targets.get((layer, projection))
        down_shape = shape if suffix == "lora_down.weight" else (old.down_shape if old else ())
        up_shape = shape if suffix == "lora_up.weight" else (old.up_shape if old else ())
        if old is None:
            targets[(layer, projection)] = MOSSLoRATarget(
                layer, projection, 0, None, down_shape, up_shape
            )
        else:
            targets[(layer, projection)] = MOSSLoRATarget(
                old.layer, old.projection, old.rank, old.alpha, down_shape, up_shape
            )

    for key, target in list(targets.items()):
        layer, projection = key
        down = entries[_target_key(layer, projection, "lora_down.weight")]
        up = entries[_target_key(layer, projection, "lora_up.weight")]
        down_shape = tuple(int(item) for item in down["shape"])
        up_shape = tuple(int(item) for item in up["shape"])
        input_dims, output_dims = _expected_shapes(projection)
        rank = down_shape[0]
        if not 1 <= rank <= MOSS_LORA_MAX_RANK:
            raise ValueError(f"{file.name} 的 {key} rank={rank} 超出 1..{MOSS_LORA_MAX_RANK}")
        if down_shape != (rank, input_dims) or up_shape != (output_dims, rank):
            raise ValueError(
                f"{file.name} 的 {key} shape 不匹配：down={down_shape}, up={up_shape}，"
                f"期望 down=({rank},{input_dims}), up=({output_dims},{rank})"
            )
        alpha_info = entries[_target_key(layer, projection, "alpha")]
        if alpha_info.get("shape") != []:
            raise ValueError(f"{file.name} 的 {key} alpha 必须是标量")
        targets[key] = MOSSLoRATarget(layer, projection, rank, None, down_shape, up_shape)
    return targets


def validate_standard_header(path: str | Path) -> dict[tuple[int, str], MOSSLoRATarget]:
    """Validate the regular fixed-rank PEFT MOSS LoRA layout.

    The returned targets use ``down_shape`` for PEFT ``lora_A`` and
    ``up_shape`` for PEFT ``lora_B`` so they can share the same native wrapper
    as the 4-step adapter.  Only the safetensors header is read here.
    """
    file = Path(path)
    if file.suffix.lower() != ".safetensors":
        raise ValueError(f"MOSS 普通 LoRA 必须是 .safetensors 文件: {file}")
    entries, metadata = _header(file)
    config = _adapter_config(file, metadata)
    if config is not None and config.get("format") not in (None, MOSS_IMAGE_EDIT_FORMAT):
        raise ValueError(f"{file.name} 不是支持的 MOSS PEFT adapter format: {config.get('format')!r}")

    adapter_metadata: dict[str, Any] = {}
    raw_adapter_metadata = metadata.get("lora_adapter_metadata")
    if isinstance(raw_adapter_metadata, str):
        try:
            parsed = json.loads(raw_adapter_metadata)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file.name} 的 lora_adapter_metadata 不是合法 JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{file.name} 的 lora_adapter_metadata 必须是 object")
        adapter_metadata = parsed
    elif raw_adapter_metadata is not None:
        if not isinstance(raw_adapter_metadata, dict):
            raise ValueError(f"{file.name} 的 lora_adapter_metadata 必须是 object")
        adapter_metadata = raw_adapter_metadata

    module_specs = _module_specs(config)
    try:
        global_rank = int(adapter_metadata.get("r", 0))
        global_alpha = float(adapter_metadata.get("lora_alpha", global_rank))
    except (TypeError, ValueError) as exc:
        if not module_specs:
            raise ValueError(f"{file.name} 的 PEFT rank/alpha 元数据无效") from exc
        global_rank = global_alpha = 0
    if module_specs:
        if len(module_specs) != MOSS_LORA_LAYERS * len(MOSS_LORA_PROJECTIONS):
            raise ValueError(
                f"{file.name} 的 MOSS adapter_config 目标覆盖不完整："
                f"实际 {len(module_specs)}，期望 {MOSS_LORA_LAYERS * len(MOSS_LORA_PROJECTIONS)}"
            )
    elif not 1 <= global_rank <= MOSS_LORA_MAX_RANK or not math.isfinite(global_alpha):
        raise ValueError(
            f"{file.name} 的 PEFT rank/alpha 无效：rank={global_rank}, alpha={global_alpha}"
        )

    key_map = _standard_key_map(entries)
    expected_targets = {
        (layer, projection, matrix)
        for layer in range(MOSS_LORA_LAYERS)
        for projection in MOSS_LORA_PROJECTIONS
        for matrix in ("A", "B")
    }
    actual_targets = set(key_map)
    if actual_targets != expected_targets:
        missing = sorted(expected_targets - actual_targets)
        extra = sorted(actual_targets - expected_targets)
        raise ValueError(
            f"{file.name} 的 MOSS 普通 LoRA 层覆盖不完整："
            f"missing={missing[:4]}, extra={extra[:4]}"
        )

    targets: dict[tuple[int, str], MOSSLoRATarget] = {}
    for (layer, projection, matrix), key in key_map.items():
        value = entries[key]
        if value.get("dtype") != "BF16":
            raise ValueError(
                f"{file.name} 的 {key} dtype={value.get('dtype')!r}，"
                "只支持 BF16 单文件 LoRA"
            )
        shape = tuple(int(item) for item in value.get("shape", ()))
        if len(shape) != 2:
            raise ValueError(f"{file.name} 的 {key} 必须是二维矩阵，收到 shape={shape}")
        old = targets.get((layer, projection))
        down_shape = shape if matrix == "A" else (old.down_shape if old else ())
        up_shape = shape if matrix == "B" else (old.up_shape if old else ())
        rank, alpha = module_specs.get((layer, projection), (global_rank, global_alpha))
        targets[(layer, projection)] = MOSSLoRATarget(
            layer,
            projection,
            rank,
            alpha,
            down_shape,
            up_shape,
        )

    for (layer, projection), target in targets.items():
        rank, alpha = module_specs.get((layer, projection), (global_rank, global_alpha))
        input_dims, output_dims = _expected_shapes(projection)
        if target.down_shape != (rank, input_dims) or target.up_shape != (output_dims, rank):
            raise ValueError(
                f"{file.name} 的 layers.{layer}.proj.{projection} shape 不匹配："
                f"lora_A={target.down_shape}, lora_B={target.up_shape}，"
                f"期望 lora_A=({rank},{input_dims}), lora_B=({output_dims},{rank})"
            )
        targets[(layer, projection)] = MOSSLoRATarget(
            target.layer,
            target.projection,
            rank,
            alpha,
            target.down_shape,
            target.up_shape,
        )
    return targets


def validate_image_edit_header(path: str | Path) -> dict[tuple[int, str], MOSSLoRATarget]:
    """Validate the dedicated MOSS image-edit PEFT adapter contract."""
    file = Path(path)
    entries, metadata = _header(file)
    config = _adapter_config(file, metadata)
    if config is None or config.get("format") != MOSS_IMAGE_EDIT_FORMAT:
        actual = None if config is None else config.get("format")
        raise ValueError(
            f"{file.name} 不是 MOSS image-edit adapter（format={actual!r}）；"
            f"需要 format={MOSS_IMAGE_EDIT_FORMAT!r}"
        )
    if config.get("format_version") != 1:
        raise ValueError(f"{file.name} 的 MOSS image-edit adapter format_version 不支持")
    return validate_standard_header(file)


class MOSSLoRALinear(nn.Module):
    """A dense Linear wrapped by one or more non-baked low-rank branches.

    ``branches`` deliberately stays as a list of small dictionaries instead of
    nesting ``MOSSLoRALinear`` instances.  That gives every target exactly one
    wrapper, so standard PEFT and dynamic-rank 4-step adapters can be attached
    in either order without making one wrapper the special ``base`` of another.
    ``nn.Module`` tracks arrays inside lists/dicts as parameters, which also
    keeps the wrapper compatible with MLX evaluation and cache lifetimes.
    """

    def __init__(
        self,
        base: nn.Linear,
        down: mx.array,
        up: mx.array,
        scale: float,
        *,
        branches: list[dict[str, Any]] | None = None,
        adapter_key: str | None = None,
    ):
        super().__init__()
        self.base = base
        self.branches = list(branches or [])
        object.__setattr__(self, "_adapter_keys", set())
        self.add_branch(down, up, scale, adapter_key=adapter_key)

    def add_branch(
        self,
        down: mx.array,
        up: mx.array,
        scale: float,
        *,
        adapter_key: str | None = None,
    ) -> bool:
        """Append one low-rank branch without changing the dense base.

        ``adapter_key`` is the resolved file path.  It is not part of the
        parameter tree, but lets a repeated materialization pass avoid adding
        the same file twice to an already wrapped target.
        """
        if adapter_key is not None and adapter_key in self._adapter_keys:
            return False
        self.branches.append(
            {
                "down": down,
                "up": up,
                "scale": float(scale),
            }
        )
        if adapter_key is not None:
            self._adapter_keys.add(adapter_key)
        return True

    def __call__(self, x: mx.array) -> mx.array:
        result = self.base(x)
        for branch_spec in self.branches:
            down = branch_spec["down"]
            up = branch_spec["up"]
            branch_input = x.astype(down.dtype)
            branch = mx.matmul(branch_input, down.T)
            branch = mx.matmul(branch, up.T)
            result = result + branch.astype(result.dtype) * mx.array(
                branch_spec["scale"], dtype=result.dtype
            )
        return result


def _validate_base_request_refs(model_handle: Any, refs: tuple[LoraRef, ...]) -> tuple[LoraRef, ...]:
    active = active_loras(refs)
    if not active:
        return ()
    if getattr(model_handle, "model_type", None) != "moss":
        raise ValueError("MOSS LoRA 只能连接 model_type=moss 的 Transformer")
    if int(getattr(model_handle, "quantize", 0)) != 0:
        raise ValueError("MOSS LoRA 只支持未量化 dense Transformer；请把 quantize 设为 0")
    if getattr(model_handle, "precision", None) != "bfloat16":
        raise ValueError("MOSS LoRA 只支持 bfloat16 Transformer")
    return active


def _validate_request_refs(model_handle: Any, refs: tuple[LoraRef, ...]) -> tuple[LoraRef, ...]:
    active = _validate_base_request_refs(model_handle, refs)
    if not active:
        return ()
    for ref in active:
        if float(ref.strength) != 1.0:
            raise ValueError("MOSS 4-step LoRA 的强度固定为 1.0")
    return active


def _validate_standard_request_refs(
    model_handle: Any, refs: tuple[LoraRef, ...]
) -> tuple[LoraRef, ...]:
    return _validate_base_request_refs(model_handle, refs)


def _validate_node_addition(model_handle: Any, selection: str, strength: float) -> None:
    """Validate a MOSS adapter addition before returning the new model handle.

    A node may be evaluated more than once with the same selection, so the
    exact same path/strength remains an idempotent re-application.  Different
    paths are now valid: the native injector combines their branches in one
    ``MOSSLoRALinear`` wrapper.
    """
    existing = active_loras(tuple(getattr(model_handle, "loras", ())))
    if not existing:
        return

    candidate_path = resolve_lora_path(selection)
    for item in existing:
        if resolve_lora_path(item.path) == candidate_path:
            if float(item.strength) != float(strength):
                raise ValueError(
                    f"同一 MOSS LoRA 出现两次且强度不同: {selection}"
                    f"（{item.strength} 与 {strength}）"
                )
            return


def _is_standard_layout(path: str | Path) -> bool:
    entries, _metadata = _header(path)
    return any(MOSS_STANDARD_LORA_KEY_RE.fullmatch(key) for key in entries)


def validate_sampling_request(model_handle: Any, steps: int, guidance: float, scheduler: str) -> bool:
    """Validate sampler settings that are independent of the requested step count."""
    refs = tuple(getattr(model_handle, "loras", ()))
    active = _validate_base_request_refs(model_handle, refs)
    if not active:
        return False
    for ref in active:
        path = resolve_lora_path(ref.path)
        if not _is_standard_layout(path):
            _validate_request_refs(model_handle, (ref,))
    if float(guidance) != 1.0:
        raise ValueError("MOSS LoRA 必须使用 CFG/guidance=1.0")
    if scheduler != "moss_flow_match_euler":
        raise ValueError("MOSS LoRA 必须使用 moss_flow_match_euler（Euler shift=5）")
    return True


def _validate_transformer(transformer: Any) -> Any:
    layers = getattr(transformer, "layers", None)
    if layers is None or len(layers) != MOSS_LORA_LAYERS:
        raise ValueError(f"MOSS Transformer 必须有 {MOSS_LORA_LAYERS} 个 dense layers")
    return layers


def _base_for_target(
    layers: Any, layer_index: int, projection: str, *, label: str
) -> nn.Linear | MOSSLoRALinear:
    base = layers[layer_index].proj.get(projection)
    dense = base.base if isinstance(base, MOSSLoRALinear) else base
    if not isinstance(dense, nn.Linear):
        raise ValueError(
            f"MOSS LoRA 目标 layers.{layer_index}.proj.{projection} "
            f"不是未量化 dense Linear（{label}）"
        )
    input_dims, output_dims = _expected_shapes(projection)
    if tuple(dense.weight.shape) != (output_dims, input_dims):
        raise ValueError(
            f"MOSS 基础层 layers.{layer_index}.proj.{projection} shape={tuple(dense.weight.shape)}，"
            f"期望 {(output_dims, input_dims)}"
        )
    return base


def _attach_branch(
    layers: Any,
    layer_index: int,
    projection: str,
    down: mx.array,
    up: mx.array,
    scale: float,
    *,
    adapter_key: str,
    label: str,
) -> None:
    """Attach a branch to a target, creating the composite wrapper once."""
    current = _base_for_target(layers, layer_index, projection, label=label)
    if isinstance(current, MOSSLoRALinear):
        current.add_branch(down, up, scale, adapter_key=adapter_key)
    else:
        layers[layer_index].proj[projection] = MOSSLoRALinear(
            current,
            down,
            up,
            scale,
            adapter_key=adapter_key,
        )


def apply_4step_lora(transformer: Any, model_handle: Any, *, role: str = "transformer") -> None:
    """Apply every validated dynamic-rank MOSS adapter to the transformer."""
    if role != "transformer":
        raise ValueError(f"MOSS 4-step LoRA 只支持 transformer role，收到 {role!r}")
    refs = _validate_base_request_refs(model_handle, tuple(getattr(model_handle, "loras", ())))
    four_step_refs = []
    for ref in refs:
        path = resolve_lora_path(ref.path)
        if not _is_standard_layout(path):
            _validate_request_refs(model_handle, (ref,))
            four_step_refs.append((ref, path))
    if not four_step_refs:
        return
    layers = _validate_transformer(transformer)
    for ref, path in four_step_refs:
        targets = validate_header(path)
        try:
            weights = dict(mx.load(path, format="safetensors"))
        except (OSError, ValueError, RuntimeError) as exc:
            raise ValueError(f"无法读取 MOSS 4-step LoRA payload {path}: {exc}") from exc

        try:
            for (layer_index, projection), target in targets.items():
                down = weights[_target_key(layer_index, projection, "lora_down.weight")]
                up = weights[_target_key(layer_index, projection, "lora_up.weight")]
                alpha = weights[_target_key(layer_index, projection, "alpha")]
                mx.eval(down, up, alpha)
                alpha_value = float(alpha.item())
                if not mx.isfinite(alpha).item():
                    raise ValueError(
                        f"MOSS 4-step LoRA 的 layers.{layer_index}.proj.{projection} alpha "
                        f"必须是有限 scalar，收到 {alpha_value:g}"
                    )
                _attach_branch(
                    layers,
                    layer_index,
                    projection,
                    down,
                    up,
                    alpha_value / float(target.rank),
                    adapter_key=path,
                    label="4-step",
                )
            mx.eval(transformer.parameters())
        finally:
            # The wrapper owns all matrix arrays after replacement.  Do not retain a
            # second dictionary containing the complete payload.
            del weights
        print(
            f"[MLX LoRA] moss/transformer: 已以未合并方式应用 {Path(path).name} "
            f"（{len(targets)} 个目标层）"
        )


def apply_standard_lora(transformer: Any, model_handle: Any, *, role: str = "transformer") -> None:
    """Apply every regular fixed-rank PEFT MOSS adapter at its strength."""
    if role != "transformer":
        raise ValueError(f"MOSS LoRA 只支持 transformer role，收到 {role!r}")
    refs = _validate_standard_request_refs(model_handle, tuple(getattr(model_handle, "loras", ())))
    standard_refs = []
    for ref in refs:
        path = resolve_lora_path(ref.path)
        if _is_standard_layout(path):
            standard_refs.append((ref, path))
    if not standard_refs:
        return
    layers = _validate_transformer(transformer)
    for ref, path in standard_refs:
        targets = validate_standard_header(path)
        try:
            weights = dict(mx.load(path, format="safetensors"))
        except (OSError, ValueError, RuntimeError) as exc:
            raise ValueError(f"无法读取 MOSS 普通 LoRA payload {path}: {exc}") from exc

        try:
            strength = float(ref.strength)
            for (layer_index, projection), target in targets.items():
                down = weights[
                    _standard_tensor_key(weights, layer_index, projection, "A")
                ]
                up = weights[
                    _standard_tensor_key(weights, layer_index, projection, "B")
                ]
                mx.eval(down, up)
                _attach_branch(
                    layers,
                    layer_index,
                    projection,
                    down,
                    up,
                    strength * float(target.alpha) / float(target.rank),
                    adapter_key=path,
                    label="PEFT",
                )
            mx.eval(transformer.parameters())
        finally:
            del weights
        print(
            f"[MLX LoRA] moss/transformer: 已以未合并方式应用 {Path(path).name} "
            f"（{len(targets)} 个目标层，strength={ref.strength:g}）"
        )


def apply_lora(transformer: Any, model_handle: Any, *, role: str = "transformer") -> None:
    """Dispatch and combine all MOSS adapters by their safetensors layouts."""
    refs = _validate_base_request_refs(model_handle, tuple(getattr(model_handle, "loras", ())))
    if not refs:
        return
    if role != "transformer":
        raise ValueError(f"MOSS LoRA 只支持 transformer role，收到 {role!r}")
    for ref in refs:
        path = resolve_lora_path(ref.path)
        if not _is_standard_layout(path):
            _validate_request_refs(model_handle, (ref,))
            validate_header(path)
    apply_standard_lora(transformer, model_handle, role=role)
    apply_4step_lora(transformer, model_handle, role=role)


def validate_node_selection(model_handle: Any, selection: str) -> str:
    """Resolve and header-check a node selection without loading its payload."""
    _validate_node_addition(model_handle, selection, 1.0)
    refs = _validate_request_refs(
        model_handle,
        (LoraRef(selection, 1.0),),
    )
    if not refs:
        raise ValueError("MOSS 4-step LoRA 选择不能为空")
    path = resolve_lora_path(selection)
    validate_header(path)
    return path


def validate_standard_node_selection(model_handle: Any, selection: str, strength: float) -> str:
    """Resolve and header-check a regular MOSS PEFT selection."""
    refs = _validate_standard_request_refs(
        model_handle,
        (LoraRef(selection, strength),),
    )
    if not refs:
        raise ValueError("MOSS 普通 LoRA 选择不能为空")
    path = resolve_lora_path(selection)
    validate_standard_header(path)
    return path


def validate_generic_node_selection(model_handle: Any, selection: str, strength: float) -> str:
    """Validate a MOSS selection used by the shared model-LoRA node.

    Standard PEFT adapters are accepted here.  The legacy 4-step adapter is
    deliberately rejected with an actionable message instead of being silently
    treated as a regular LoRA with a user-selected strength.
    """
    _validate_node_addition(model_handle, selection, strength)
    refs = _validate_standard_request_refs(model_handle, (LoraRef(selection, strength),))
    if not refs:
        raise ValueError("MOSS 普通 LoRA 选择不能为空")
    path = resolve_lora_path(selection)
    entries, _metadata = _header(path)
    if is_image_edit_adapter(path):
        raise ValueError(
            "MOSS image-edit LoRA 不能接通用「MLX 模型 LoRA」节点；"
            "请使用「MLX MOSS Image Edit LoRA」节点"
        )
    if any(MOSS_STANDARD_LORA_KEY_RE.fullmatch(key) for key in entries):
        validate_standard_header(path)
        return path
    if MOSS_LORA_FILENAME_RE.search(Path(path).name) or any(
        MOSS_LORA_KEY_RE.fullmatch(key) for key in entries
    ):
        raise ValueError(
            "MOSS 4-step LoRA 不能接通用「MLX 模型 LoRA」节点；"
            "请使用「MLX MOSS 4 步 LoRA」节点"
        )
    raise ValueError(
        f"{Path(path).name} 不是支持的 MOSS 普通 PEFT LoRA；"
        "需要 transformer.layers.*.proj.*.lora_A/B.weight"
    )