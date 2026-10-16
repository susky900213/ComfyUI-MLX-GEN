"""把 ``MlxModelLoraApply`` 登记的适配器真正应用到 MLX Transformer。

图片家族直接复用 mflux 0.19.1 的 LoRALoader 与官方 mapping。LoRA 保持为量化
基础 Linear 外层的低秩分支（``bake_lora=False``）：这样 q4/q8 checkpoint 不会为了
融合 LoRA 被整体反量化、再升级为 q8，同时多个 LoRA 仍可由 mflux 正确叠加。

MiniMax-H3 的模型实现位于本插件内，且已发布的 ComfyUI LoRA 还可能使用
``int8_tensorwise + ConvRot`` 存储，因此走 ``h3.weights.h3_lora`` 的专用入口。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx

from . import paths, runtime
from .types import LoraRef


_IMAGE_MAPPING_IMPORTS = {
    "z_image": "mflux.models.z_image.weights.z_image_lora_mapping:ZImageLoRAMapping",
    "flux2": "mflux.models.flux2.weights.flux2_lora_mapping:Flux2LoRAMapping",
    "flux3": "comfyui_mlx_gen.flux3.lora_mapping:Flux3LoRAMapping",
    "qwen_image": "mflux.models.qwen.weights.qwen_lora_mapping:QwenLoRAMapping",
    "qwen_edit": "mflux.models.qwen.weights.qwen_lora_mapping:QwenLoRAMapping",
    "qwen_image_21": "comfyui_mlx_gen.qwen_image_21.lora_mapping:QwenImage21LoRAMapping",
    "ideogram4": "mflux.models.ideogram4.weights.ideogram4_lora_mapping:Ideogram4LoRAMapping",
}


def supported_families() -> tuple[str, ...]:
    """当前已接入 Transformer LoRA 的模型大类。"""
    return (*_IMAGE_MAPPING_IMPORTS, "minimax_h3")


def mapping_for_family(family: str) -> list[Any]:
    """返回图片家族的 mflux LoRA mapping；未知家族明确报错。"""
    spec = _IMAGE_MAPPING_IMPORTS.get(family)
    if spec is None:
        raise ValueError(
            f"模型大类 {family!r} 尚不支持 Transformer LoRA；"
            f"当前支持：{', '.join(supported_families())}"
        )
    mapping = runtime.import_object(spec).get_mapping()
    _add_peft_pattern_aliases(mapping)
    return mapping


def _add_peft_pattern_aliases(mapping: list[Any]) -> None:
    """补齐 mflux 0.19.1 Loader 尚未统一处理的 PEFT 前缀与 ``default`` 中缀。"""

    def aliases(patterns: list[str]) -> None:
        additions: list[str] = []
        for pattern in patterns:
            if pattern.startswith(("base_model.", "lora_unet_", "lycoris_")):
                candidates = [pattern]
            else:
                candidates = [pattern, f"base_model.model.{pattern}"]
            for candidate in candidates:
                additions.append(candidate)
                for matrix in ("lora_A", "lora_B"):
                    suffix = f".{matrix}.weight"
                    if candidate.endswith(suffix):
                        additions.append(candidate[: -len(suffix)] + f".{matrix}.default.weight")
        patterns.extend(item for item in additions if item not in patterns)

    for target in mapping:
        aliases(target.possible_up_patterns)
        aliases(target.possible_down_patterns)
        aliases(target.possible_alpha_patterns)


def resolve_lora_path(selection: str) -> str:
    """把节点保存的 ``lora/`` 相对路径解析为经过校验的绝对文件路径。

    这里不能复用会调用 :meth:`Path.resolve` 的通用 ``paths.resolve``：Hugging Face
    snapshot 中的 ``*.safetensors`` 通常是指向 ``blobs/<sha256>`` 的符号链接，而
    MLX 依靠传入路径的扩展名判断文件格式。若把链接解引用到无扩展名 blob，后续
    ``mx.load`` 会报无法推断格式。保留逻辑链接路径，同时仍校验其最终目标存在且
    是文件。
    """
    selected = Path(selection).expanduser()
    if selected.is_absolute():
        candidates = (selected,)
    elif len(selected.parts) > 1:
        candidates = (paths.MODEL_ROOT / selected, paths.component_dir("lora") / selected)
    else:
        candidates = (paths.component_dir("lora") / selected, paths.MODEL_ROOT / selected)

    path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    absolute = path.absolute()
    if not absolute.exists():
        raise FileNotFoundError(f"未找到 LoRA 权重：{absolute}")
    if not absolute.is_file() or absolute.suffix.lower() != ".safetensors":
        raise ValueError(f"LoRA 必须是 .safetensors 文件，收到：{absolute}")
    return str(absolute)


def active_loras(loras: Iterable[LoraRef]) -> tuple[LoraRef, ...]:
    """strength=0 严格等同基础模型，不加载文件也不创建空 LoRA 包装层。"""
    return tuple(ref for ref in loras if float(ref.strength) != 0.0)


def apply_transformer_loras(
    family: str,
    transformer: Any,
    loras: Iterable[LoraRef],
    *,
    role: str = "transformer",
) -> tuple[list[str], list[float]]:
    """在基础权重（及基础量化）加载完成后应用一组 Transformer LoRA。"""
    refs = active_loras(loras)
    if not refs:
        return [], []

    if family == "minimax_h3":
        from .h3.weights.h3_lora import apply_h3_loras

        return apply_h3_loras(transformer, refs, role=role)

    mapping = mapping_for_family(family)
    resolved = [resolve_lora_path(ref.path) for ref in refs]
    _reject_comfy_quantized_image_loras(resolved, family)
    scales = [float(ref.strength) for ref in refs]
    loader = runtime.import_object(
        "mflux.models.common.lora.mapping.lora_loader:LoRALoader"
    )
    print(
        f"[MLX LoRA] {family}/{role}: 准备应用 {len(resolved)} 个适配器："
        + ", ".join(f"{Path(path).name}@{scale:g}" for path, scale in zip(resolved, scales))
    )
    try:
        if family == "qwen_image_21":
            _validate_qwen_image_21_lora_compatibility(resolved, transformer, mapping)
        applied_paths, applied_scales = loader.load_and_apply_lora(
            lora_mapping=mapping,
            transformer=transformer,
            lora_paths=resolved,
            lora_scales=scales,
            role=role,
            bake_lora=False,
        )
        _validate_lora_layers(transformer)
        mx.eval(transformer.parameters())
    except Exception as exc:  # noqa: BLE001 - 增补家族/role 后保留原异常链
        raise RuntimeError(
            f"LoRA 应用失败（模型={family}, role={role}）：{exc}"
        ) from exc
    print(f"[MLX LoRA] {family}/{role}: 已成功应用 {len(applied_paths)} 个适配器")
    return list(applied_paths), list(applied_scales)


def _match_lora_pattern(pattern: str, key: str) -> int | None:
    """Match one mflux mapping pattern and return its block number.

    Qwen's LyCORIS keys flatten dots to underscores, so using a normal path
    split here would be incorrect.  The mapping has already limited the
    pattern to one target; this helper only resolves ``{block}``.
    """
    if "{block}" not in pattern:
        return 0 if pattern == key else None
    prefix, suffix = pattern.split("{block}", 1)
    if not key.startswith(prefix) or not key.endswith(suffix):
        return None
    block_text = key[len(prefix) : len(key) - len(suffix) if suffix else None]
    if not block_text.isdigit():
        return None
    return int(block_text)


def _target_for_lokr_key(mapping: list[Any], key: str) -> tuple[Any, int] | None:
    """Find the mapping target named by a LoKr factor key."""
    suffix = ".lokr_w1"
    if not key.endswith(suffix):
        return None
    for target in mapping:
        for pattern in target.possible_lokr_w1_patterns:
            block = _match_lora_pattern(pattern, key)
            if block is not None:
                return target, block
    return None


def _get_module_by_path(module: Any, path: str) -> Any:
    """Resolve an MLX module path, including list indexes such as ``to_out.0``."""
    current = module
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def _validate_qwen_image_21_lora_compatibility(
    resolved: list[str], transformer: Any, mapping: list[Any]
) -> None:
    """Reject a known-incompatible Qwen LoKr before mutating the base model.

    ``mflux`` supports LoKr itself, but it can only report a mismatch after it
    has started replacing target modules.  Qwen-Image 2.1 has 32 blocks with
    4096-wide attention, while older Qwen-Image LoKr files commonly have 60
    blocks with 3072-wide attention.  Reading only safetensors headers lets us
    turn that late, ambiguous failure into an actionable error.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        # mflux already depends on safetensors; if an unusual installation does
        # not expose the Python package, let its normal loader report the error.
        return

    for path in resolved:
        try:
            with safe_open(path, framework="numpy") as file:
                keys = list(file.keys())
                lokr_keys = [key for key in keys if key.endswith(".lokr_w1")]
                if not lokr_keys:
                    continue
                metadata = file.metadata() or {}
                block_numbers = {
                    int(match.group(1))
                    for key in lokr_keys
                    if (match := re.search(r"transformer_blocks_(\d+)_", key))
                }
                block_count = len(getattr(transformer, "transformer_blocks", ()))
                too_many_blocks = block_numbers and max(block_numbers) >= block_count
                mismatch: tuple[str, tuple[int, int], tuple[int, int]] | None = None

                for w1_key in lokr_keys:
                    match = _target_for_lokr_key(mapping, w1_key)
                    if match is None:
                        continue
                    target, block = match
                    w2_key = w1_key[: -len(".lokr_w1")] + ".lokr_w2"
                    if w2_key not in keys:
                        continue
                    target_path = target.model_path.format(block=block)
                    try:
                        target_module = _get_module_by_path(transformer, target_path)
                        expected_out, packed_input = target_module.weight.shape
                        bits = getattr(target_module, "bits", None)
                        expected_input = (
                            int(packed_input) * (32 // int(bits))
                            if bits is not None
                            else int(packed_input)
                        )
                        w1_shape = tuple(file.get_slice(w1_key).get_shape())
                        w2_shape = tuple(file.get_slice(w2_key).get_shape())
                        if len(w1_shape) != 2 or len(w2_shape) != 2:
                            continue
                        actual = (
                            int(w1_shape[0]) * int(w2_shape[0]),
                            int(w1_shape[1]) * int(w2_shape[1]),
                        )
                        expected = (int(expected_out), expected_input)
                        if actual != expected and mismatch is None:
                            mismatch = (target_path, actual, expected)
                    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                        # An unsupported target is left to mflux, which will
                        # include the exact target path in its normal error.
                        continue

                if too_many_blocks or mismatch is not None:
                    base_model = metadata.get("ss_base_model_version", "未知")
                    details: list[str] = []
                    if too_many_blocks:
                        details.append(
                            f"LoRA 包含最高第 {max(block_numbers)} 个 transformer block，"
                            f"当前 2.1 模型只有 {block_count} 个 block"
                        )
                    if mismatch is not None:
                        target_path, actual, expected = mismatch
                        details.append(
                            f"目标 {target_path} 的 LoKr 重建尺寸为 {actual[0]}×{actual[1]}，"
                            f"当前模型需要 {expected[0]}×{expected[1]}"
                        )
                    raise ValueError(
                        f"LoRA {Path(path).name} 与 Qwen-Image 2.1 Transformer 架构不兼容："
                        + "；".join(details)
                        + f"。文件 metadata 的 ss_base_model_version={base_model!r}。"
                        "这通常表示它是旧版 qwen_image LoRA，而不是 Qwen-Image 2.1 LoRA；"
                        "请换用包含 32 个 transformer_blocks、且目标维度为 4096 的 2.1 LoRA。"
                    )
                # A file with only unsupported names should retain mflux's
                # detailed unmatched-key error rather than being mislabeled by
                # this preflight.
        except OSError as exc:
            raise ValueError(f"无法读取 LoRA 文件头 {Path(path).name}: {exc}") from exc


def _base_linear_shape(linear: Any) -> tuple[int, int]:
    """返回基础 Linear 的 ``(input_dims, output_dims)``，兼容量化 packed weight。"""
    output_dims, packed_input_dims = linear.weight.shape
    bits = getattr(linear, "bits", None)
    input_dims = packed_input_dims * (32 // int(bits)) if bits is not None else packed_input_dims
    return int(input_dims), int(output_dims)


def _validate_lora_layers(transformer: Any) -> None:
    """mflux 0.19.1 不预检 A/B 形状；在采样前统一验证，避免第一步才失败。"""
    lora_cls = runtime.import_object(
        "mflux.models.common.lora.layer.linear_lora_layer:LoRALinear"
    )
    fused_cls = runtime.import_object(
        "mflux.models.common.lora.layer.fused_linear_lora_layer:FusedLoRALinear"
    )
    for path, module in transformer.named_modules():
        if isinstance(module, lora_cls):
            pairs = ((module.linear, module),)
        elif isinstance(module, fused_cls):
            pairs = tuple(
                (module.base_linear, item)
                for item in module.loras
                if isinstance(item, lora_cls)
            )
        else:
            continue
        for base, lora in pairs:
            input_dims, output_dims = _base_linear_shape(base)
            actual = (tuple(lora.lora_A.shape), tuple(lora.lora_B.shape))
            if (
                lora.lora_A.ndim != 2
                or lora.lora_B.ndim != 2
                or int(lora.lora_A.shape[0]) != input_dims
                or int(lora.lora_B.shape[1]) != output_dims
                or int(lora.lora_A.shape[1]) != int(lora.lora_B.shape[0])
            ):
                raise ValueError(
                    f"LoRA 形状与目标层 {path or '<root>'} 不匹配："
                    f"base=(input={input_dims}, output={output_dims})，A/B={actual}"
                )


def _reject_comfy_quantized_image_loras(resolved: list[str], family: str) -> None:
    """mflux 0.19.1 不认识 Comfy ``comfy_quant``；禁止把 int8 当普通矩阵误用。"""
    try:
        from safetensors import safe_open
    except ImportError:  # mflux 的正式依赖通常会提供；缺失时由 LoRALoader 做常规读取
        return
    for path in resolved:
        with safe_open(path, framework="numpy") as file:
            if any(key.endswith(".comfy_quant") for key in file.keys()):
                raise ValueError(
                    f"{Path(path).name} 使用 ComfyUI 量化 LoRA 格式。当前 {family} 的 mflux "
                    "mapping 只支持普通浮点 LoRA；int8-convrot 解码目前仅为 MiniMax-H3 "
                    "完成了验证，不能把量化整数静默当作浮点权重应用。"
                )