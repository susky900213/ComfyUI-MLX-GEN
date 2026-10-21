"""Native MLX Fibo-2 component loading and sampling.

This module is intentionally independent from the legacy Torch/Diffusers
``backend.py``.  A unified MLX Fibo workflow enters here and never imports
Torch or Diffusers.  The input checkpoint remains the official Diffusers
layout; tensors are mapped into MLX modules while loading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import numpy as np

from .. import components, paths, runtime
from ..cache import CACHE
from ..types import ComponentDef, MlxModelEntry, MlxLatentHandle
from .backend import (
    Fibo2ClipHandle,
    Fibo2ModelHandle,
    Fibo2ValidationError,
    Fibo2VaeHandle,
    inspect_model_path,
    is_edit_turbo_model_path,
    validate_request,
)
from .mlx_scheduler import Fibo2Scheduler
from .mlx_text_encoder import Fibo2TextEncoder
from .mlx_transformer import Fibo2Transformer


FIBO2_DEFAULT_NEGATIVE = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)


FIBO2_ENTRY = MlxModelEntry(
    family="fibo2",
    weight_def="comfyui_mlx_gen.fibo2.weight_definition:Fibo2WeightDefinition",
    components={
        "transformer": ComponentDef("transformer", "comfyui_mlx_gen.fibo2.mlx_transformer:Fibo2Transformer", "local"),
        "text_encoder": ComponentDef("text_encoder", "comfyui_mlx_gen.fibo2.mlx_text_encoder:Fibo2TextEncoder", "local"),
        "vae": ComponentDef("vae", "mflux.models.flux2.model.flux2_vae.vae:Flux2VAE", "local"),
        "tokenizer": ComponentDef("fibo2", "", "local"),
    },
    default_config="",
    default_steps=30,
    default_scheduler="fibo_flow_match_euler",
    default_guidance=5.0,
    supports_compile=False,
    supported=True,
    notes="Fibo-2 native MLX backend; requires an official local Diffusers-format snapshot.",
)


@dataclass(frozen=True)
class Fibo2MLXBundle:
    transformer: Any
    text_encoder: Fibo2TextEncoder
    tokenizer: Any
    vae: Any


def _local_root(path: str | Path) -> Path:
    info = inspect_model_path(path)
    if info["kind"] != "local":
        raise Fibo2ValidationError(
            "MLX Fibo-2 不能直接从 Hugging Face repo id 加载；请先下载官方 snapshot，"
            "再把包含 model_index.json 的本地目录放入 models/mlx/transformer/"
        )
    return Path(info["path"])


def _read_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Fibo2ValidationError(f"无法读取 Fibo-2 配置: {path}: {exc}") from exc


def _transformer_kwargs(root: Path) -> dict[str, Any]:
    config = _read_config(root / "transformer" / "config.json")
    accepted = {
        "in_channels", "patch_size", "dim", "n_layers", "n_refiner_layers", "n_heads", "norm_eps",
        "cap_feat_dim", "injection_layer_ids", "perceiver_num_layers", "min_num_gist_tokens",
        "max_num_gist_tokens", "gist_step", "gist_min_text_len", "gist_max_text_len", "rope_theta",
        "t_scale", "axes_dims", "axes_lens",
    }
    result = {key: config[key] for key in accepted if key in config}
    for key in ("injection_layer_ids", "axes_dims", "axes_lens"):
        if key in result:
            result[key] = tuple(result[key])
    return result


def _component_path(root: Path, role: str) -> Path:
    path = root / role
    if not path.is_dir():
        raise FileNotFoundError(f"Fibo-2 checkpoint 缺少 {role}/: {path}")
    return path


def load_bundle(handle: Fibo2ModelHandle, cache=CACHE) -> Fibo2MLXBundle:
    root = _local_root(handle.model_path)
    key = runtime.cache_key({"kind": "fibo2_mlx_bundle", "model": handle, "root": str(root)})

    def build() -> Fibo2MLXBundle:
        transformer, _ = components.create_and_load(
            FIBO2_ENTRY,
            "transformer",
            "dir",
            str(_component_path(root, "transformer")),
            int(handle.quantize) or None,
            class_kwargs=_transformer_kwargs(root),
        )
        text_encoder, _ = components.create_and_load(
            FIBO2_ENTRY,
            "text_encoder",
            "dir",
            str(_component_path(root, "text_encoder")),
            None,
            class_kwargs={"max_length": 4096},
        )
        tokenizer = components.load_tokenizer(
            FIBO2_ENTRY,
            "tokenizer",
            "dir",
            str(_component_path(root, "processor")),
            max_length=4096,
        )
        vae, _ = components.create_and_load(
            FIBO2_ENTRY,
            "vae",
            "dir",
            str(_component_path(root, "vae")),
            None,
        )
        for module in (transformer, text_encoder, vae):
            for value in module.parameters().values() if isinstance(module.parameters(), dict) else ():
                if hasattr(value, "dtype") and value.dtype in (mx.float16, mx.bfloat16, mx.float32):
                    pass
        return Fibo2MLXBundle(transformer, text_encoder, tokenizer, vae)

    return cache.get_or_create("module", key, build)[0]


def _as_mlx_image(image: Any) -> mx.array:
    import PIL.Image

    if not isinstance(image, PIL.Image.Image):
        raise ValueError("Fibo-2 MLX 参考图必须是 PIL.Image")
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return mx.array(array.transpose(2, 0, 1)[None, ...])


def _trained_image_size(aspect_ratio: float, area: int = 1024 * 1024) -> tuple[int, int]:
    multiple = 16
    min_side = 4 * multiple
    sizes = [
        (width, height)
        for width in range(min_side, 2 * area // min_side, multiple)
        for height in range(min_side, 2 * area // width, multiple)
        if 0.9 * area <= width * height <= 1.1 * area
    ]
    if not sizes:
        raise ValueError(f"无法为宽高比 {aspect_ratio:.4f} 找到 Fibo-2 训练尺寸")
    return min(
        sizes,
        key=lambda size: (
            round(1.0 - min((size[0] / size[1]) / aspect_ratio, aspect_ratio / (size[0] / size[1])), 9),
            abs(size[0] * size[1] - area),
            -size[0],
        ),
    )


def _resize_crop(image: Any, width: int, height: int) -> Any:
    from PIL import Image

    image = image.convert("RGB")
    scale = max(height / image.height, width / image.width)
    resized = image.resize((int(np.ceil(image.width * scale)), int(np.ceil(image.height * scale))), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _preprocess_context(images: Sequence[Any], width: int, height: int) -> list[Any]:
    if not images:
        return []
    if len(images) == 1:
        return [_resize_crop(images[0], width, height)]
    result = []
    for image in images:
        target_width, target_height = _trained_image_size(image.width / image.height)
        result.append(_resize_crop(image, target_width, target_height))
    return result


def image_batch_to_pil(value: Any) -> list[Any]:
    """Convert a ComfyUI IMAGE batch without importing Torch."""

    from PIL import Image

    if value is None:
        return []
    data = value.detach().cpu().float().numpy() if hasattr(value, "detach") else np.asarray(value)
    if data.ndim == 3:
        data = data[None]
    if data.ndim != 4 or data.shape[-1] not in (3, 4):
        raise ValueError(f"Fibo-2 IMAGE 输入必须是 [B,H,W,3/4]，收到 {data.shape}")
    return [
        Image.fromarray(np.rint(np.clip(item[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
        for item in data
    ]


def _patchify(latents: mx.array) -> mx.array:
    batch, channels, height, width = latents.shape
    value = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
    return value.transpose(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height // 2, width // 2)


def _unpatchify(latents: mx.array) -> mx.array:
    batch, channels, height, width = latents.shape
    value = latents.reshape(batch, channels // 4, 2, 2, height, width)
    return value.transpose(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, height * 2, width * 2)


def _encode_context(bundle: Fibo2MLXBundle, images: Sequence[Any], width: int, height: int) -> list[mx.array]:
    result = []
    for image in _preprocess_context(images, width, height):
        pixels = _as_mlx_image(image)
        latent = bundle.vae.encode(pixels)
        packed = _patchify(latent)
        mean = mx.array(bundle.vae.bn.running_mean).reshape(1, -1, 1, 1)
        variance = mx.array(bundle.vae.bn.running_var).reshape(1, -1, 1, 1)
        packed = (packed - mean) / mx.sqrt(variance + 1e-4)
        result.append(_unpatchify(packed))
    return result


def _decode(bundle: Fibo2MLXBundle, latents: mx.array) -> mx.array:
    packed = _patchify(latents)
    mean = mx.array(bundle.vae.bn.running_mean).reshape(1, -1, 1, 1)
    variance = mx.array(bundle.vae.bn.running_var).reshape(1, -1, 1, 1)
    plain = _unpatchify(packed * mx.sqrt(variance + 1e-4) + mean)
    return bundle.vae.decode(plain)


def decode_latent(handle: Fibo2ModelHandle, latent: mx.array, cache=CACHE) -> mx.array:
    """Decode a native MLX latent without crossing a Torch/Diffusers boundary."""

    return _decode(load_bundle(handle, cache), latent)


def run(
    model: Fibo2ModelHandle,
    *,
    prompt: str,
    negative_prompt: str | None,
    images: Sequence[Any] | None,
    seed: int,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    batch_size: int,
    cache=CACHE,
) -> MlxLatentHandle:
    context = list(images or ())
    validate_request(model.model_path, context, None, steps, guidance)
    bundle = load_bundle(model, cache)
    if width % 16 or height % 16:
        raise Fibo2ValidationError("Fibo-2 的 width / height 必须是 16 的倍数")
    latent_h, latent_w = height // 8, width // 8
    # Diffusers' Fibo-2 pipeline deliberately samples and retains latents in
    # float32, independent of the transformer/VAE weight dtype.  Keeping this
    # precision through scheduler.step is important for the flow-matching
    # trajectory; casting the initial noise to bfloat16 produces visibly noisy
    # decoded images.
    noise = mx.random.normal(
        (batch_size, 32, latent_h, latent_w), key=mx.random.key(int(seed)), dtype=mx.float32
    )
    tokenizer = bundle.tokenizer
    cond, cond_mask = bundle.text_encoder.encode(tokenizer, prompt, context)
    uncond, uncond_mask = bundle.text_encoder.encode(tokenizer, negative_prompt or FIBO2_DEFAULT_NEGATIVE, context)
    if batch_size != 1:
        cond = mx.broadcast_to(cond, (batch_size, *cond.shape[1:]))
        uncond = mx.broadcast_to(uncond, (batch_size, *uncond.shape[1:]))
        cond_mask = mx.broadcast_to(cond_mask, (batch_size, cond_mask.shape[1]))
        uncond_mask = mx.broadcast_to(uncond_mask, (batch_size, uncond_mask.shape[1]))
    context_latents = _encode_context(bundle, context, width, height) if context else None
    if context_latents and batch_size != 1:
        context_latents = [mx.broadcast_to(value, (batch_size, *value.shape[1:])) for value in context_latents]
    scheduler = Fibo2Scheduler()
    timesteps = scheduler.set_timesteps(steps, (latent_h // 2) * (latent_w // 2))
    current = noise
    for timestep in timesteps:
        t = mx.full((batch_size,), float(timestep) / 1000.0, dtype=mx.float32)
        prediction = bundle.transformer(current, t, cond, cond_mask, context_latents)
        if guidance > 1.0:
            uncond_prediction = bundle.transformer(current, t, uncond, uncond_mask, context_latents)
            prediction = uncond_prediction + float(guidance) * (prediction - uncond_prediction)
        current = scheduler.step(prediction, current)
        mx.eval(current)
    key = runtime.cache_key({"kind": "fibo2_mlx_latent", "model": model, "seed": seed, "prompt": prompt, "negative": negative_prompt, "steps": steps, "guidance": guidance, "width": width, "height": height, "images": [str(image) for image in context]})
    cache.get_or_create("component_weights", key, lambda: current)
    return MlxLatentHandle(
        kind="packed",
        shape=tuple(current.shape),
        dtype=str(current.dtype),
        cache_key=key,
        model="fibo2",
        source="local",
        path=model.model_path,
        precision=model.dtype,
        height=height,
        width=width,
        model_cache_key=runtime.cache_key(model),
    )


__all__ = [
    "FIBO2_ENTRY",
    "Fibo2MLXBundle",
    "decode_latent",
    "image_batch_to_pil",
    "load_bundle",
    "run",
]