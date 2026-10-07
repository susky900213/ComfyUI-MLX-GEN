#!/usr/bin/env python3
"""Validate an official Qwen-Image 2.1 snapshot without loading tensor data.

This constructs MLX module shapes and reads only safetensors headers.  It is safe
to run before a real smoke test and catches missing/wrong component symlinks,
key mappings and convolution layout mismatches.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx.utils import tree_flatten  # noqa: E402

from comfyui_mlx_gen.qwen_image_21.loader import (  # noqa: E402
    _read_safetensors_header,
    _validate_single_transformer_header,
    build_model,
)


def mapped_shape(role: str, key: str, shape: tuple[int, ...]):
    """Return one or more local keys and their header-only shapes."""
    if role == "text_encoder":
        vision_prefix = "model.visual."
        if key.startswith(vision_prefix):
            local = "visual." + key[len(vision_prefix):]
            if local == "visual.patch_embed.proj.weight":
                shape = (shape[0], math.prod(shape[1:]))
            return [(local, shape)]
        prefix = "model.language_model."
        if not key.startswith(prefix):
            return None
        local = key[len(prefix):]
        if local == "embed_tokens.weight":
            return [(local, shape)]
        if not local.startswith("layers.") or int(local.split(".")[1]) >= 36:
            return None
        return [(local, shape)]
    if role == "vae":
        if not key.startswith(("encoder.", "quant_conv.", "decoder.", "post_quant_conv.")) or ".time_conv." in key:
            return None
        if key.endswith(".gamma"):
            shape = (shape[0],)
        elif key.endswith(".weight") and len(shape) == 4:
            shape = (shape[0], shape[2], shape[3], shape[1])
        return [(key, shape)]
    if role == "transformer" and key.endswith(".img_mlp.gate_up.weight"):
        # Turbo single-file exports fuse the SwiGLU gate and up projections as
        # [gate; up].  The MLX module stores them as separate Linear weights.
        base = key[: -len(".gate_up.weight")]
        if len(shape) != 2 or shape[0] % 2:
            return [(key, shape)]
        half = shape[0] // 2
        local_prefix = "model.diffusion_model."
        if base.startswith(local_prefix):
            base = base[len(local_prefix):]
        return [
            (base + ".gate_layer.weight", (half, shape[1])),
            (base + ".proj.weight", (half, shape[1])),
        ]
    return [(key, shape)]


def validate_component(snapshot: Path, role: str) -> None:
    directory = snapshot / role
    model = build_model(role, directory)
    expected = {key: tuple(value.shape) for key, value in tree_flatten(model.parameters())}
    source: dict[str, tuple[int, ...]] = {}
    ignored = 0
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{role} 没有 safetensors: {directory}")
    for file in files:
        with safe_open(file, framework="numpy") as handle:
            for key in handle.keys():
                converted = mapped_shape(role, key, tuple(handle.get_slice(key).get_shape()))
                if converted is None:
                    ignored += 1
                else:
                    for local_key, local_shape in converted:
                        source[local_key] = local_shape
    missing = sorted(set(expected) - set(source))
    extra = sorted(set(source) - set(expected))
    mismatched = [
        (key, expected[key], source[key])
        for key in sorted(set(expected) & set(source))
        if expected[key] != source[key]
    ]
    if missing or extra or mismatched:
        raise ValueError(
            f"{role} 对账失败：missing={missing[:5]} extra={extra[:5]} "
            f"mismatched={mismatched[:5]}"
        )
    print(
        f"[OK] {role}: {len(expected)} 个所需参数全部匹配；"
        f"按统一生成/编辑契约忽略 {ignored} 个 LM head/temporal-conv 参数"
    )


def validate_transformer_file(path: Path) -> None:
    """Validate a transformer-only single file without loading tensor payloads."""
    header, header_size = _read_safetensors_header(path)
    model = build_model("transformer", path)
    expected = {key: tuple(value.shape) for key, value in tree_flatten(model.parameters())}
    plan = _validate_single_transformer_header(header, expected, path, header_size)
    fused = sum(kind == "gate_up" for _, kind in plan.values())
    print(
        f"[OK] transformer 单文件: {len(plan)} 个源 tensor 映射到 {len(expected)} 个本地参数；"
        f"gate_up={fused}；未读取 tensor data"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    if snapshot.is_file():
        validate_transformer_file(snapshot)
        return
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Qwen-Image 2.1 snapshot 不存在: {snapshot}")
    for role in ("transformer", "text_encoder", "vae"):
        validate_component(snapshot, role)


if __name__ == "__main__":
    main()