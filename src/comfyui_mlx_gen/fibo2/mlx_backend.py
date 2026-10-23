"""Native MLX Fibo-2 component loading and sampling.

This module is intentionally independent from the legacy Torch/Diffusers
``backend.py``.  A unified MLX Fibo workflow enters here and never imports
Torch or Diffusers.  The input checkpoint remains the official Diffusers
layout; tensors are mapped into MLX modules while loading.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from .. import components, paths, runtime
from ..cache import CACHE
from ..progress import throw_if_interrupted
from ..types import ComponentDef, MlxModelEntry, MlxLatentHandle
from .backend import (
    Fibo2ClipHandle,
    Fibo2ModelHandle,
    Fibo2ValidationError,
    Fibo2VaeHandle,
    FIBO2_DIFFUSERS_MAX_SEQUENCE_LENGTH,
    FIBO2_TOKENIZER_MODEL_MAX_LENGTH,
    inspect_model_path,
    is_edit_turbo_model_path,
    validate_request,
)
from .mlx_scheduler import Fibo2Scheduler
from .mlx_text_encoder import Fibo2TextEncoder
from .mlx_transformer import Fibo2Transformer, compute_gist_count


FIBO2_TEXT_ENCODER_IMAGE_SIZE = 768
FIBO2_EVAL_CHUNK = 8
FIBO2_HEARTBEAT_INTERVAL_SECONDS = 30.0


FIBO2_DEFAULT_NEGATIVE = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)


def _log(message: str) -> None:
    """立即写入 ComfyUI 控制台和 ``comfyui.log``，避免长任务日志滞留。"""

    print(message, flush=True)


def _format_duration(seconds: float) -> str:
    """把步骤耗时格式化成适合长时间推理日志的紧凑文本。"""

    value = max(0.0, float(seconds))
    if value >= 3600:
        hours, remainder = divmod(value, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h{int(minutes):02d}m{seconds:04.1f}s"
    if value >= 60:
        minutes, seconds = divmod(value, 60)
        return f"{int(minutes)}m{seconds:04.1f}s"
    return f"{value:.2f}s"


@contextmanager
def _denoise_heartbeat(
    step: int,
    total_steps: int,
    started: float,
    *,
    interval: float = FIBO2_HEARTBEAT_INTERVAL_SECONDS,
):
    """Report wall-clock liveness while one lazy MLX denoising step is pending.

    The worker deliberately does not call MLX/Metal APIs.  A Fibo forward is a
    single lazy graph from Python's point of view, and the expensive work may
    happen either while constructing it or in the final ``mx.eval``.  Therefore
    this heartbeat only reports elapsed wall time; it must not imply per-layer
    completion or a percentage that MLX cannot expose.
    """

    stop = threading.Event()

    def report() -> None:
        while not stop.wait(max(0.001, float(interval))):
            _log(
                f"[Fibo-2 去噪] 步数 {step}/{total_steps} 仍在计算 | "
                f"本步已用 {_format_duration(time.perf_counter() - started)} | "
                "Transformer 构图/Metal 执行/同步尚未返回（无逐层百分比回调）"
            )

    worker = threading.Thread(
        target=report,
        name=f"fibo2-denoise-heartbeat-{step}",
        daemon=True,
    )
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()


def _aligned_dimension_suggestions(value: int) -> tuple[int, int]:
    lower = max(16, (int(value) // 16) * 16)
    upper = lower if lower == int(value) else lower + 16
    return lower, upper


def aligned_dimensions(width: int, height: int) -> tuple[int, int]:
    """Return the nearest Fibo-2 canvas, preferring the smaller side on ties.

    Fibo-2 patchifies an 8× downsampled latent in 2×2 blocks, so each public
    canvas side must be divisible by 16.  User-facing nodes call this helper so
    dimensions supplied by ``GetImageSize`` do not require a manual resize node.
    The strict ``validate_dimensions`` guard remains at the backend boundary.
    """

    def align(value: int) -> int:
        value = int(value)
        lower, upper = _aligned_dimension_suggestions(value)
        return lower if value - lower <= upper - value else upper

    return align(width), align(height)


def validate_dimensions(width: int, height: int) -> None:
    """在加载模型前校验 Fibo-2 画布，并给出相邻合法尺寸。"""

    if int(width) % 16 == 0 and int(height) % 16 == 0:
        return
    width_lower, width_upper = _aligned_dimension_suggestions(width)
    height_lower, height_upper = _aligned_dimension_suggestions(height)
    raise Fibo2ValidationError(
        "Fibo-2 的 width / height 必须是 16 的倍数，"
        f"收到 {int(width)}×{int(height)}；width 可用 {width_lower} 或 {width_upper}，"
        f"height 可用 {height_lower} 或 {height_upper}"
    )


def format_size_routes(routes: Sequence[Sequence[int]]) -> str:
    parts = []
    for index, route in enumerate(routes, start=1):
        if len(route) != 6:
            continue
        source_w, source_h, vae_w, vae_h, qwen_w, qwen_h = (
            int(value) for value in route
        )
        parts.append(
            f"图 {index}：原图 {source_w}×{source_h} → VAE {vae_w}×{vae_h} "
            f"→ Qwen3-VL {qwen_w}×{qwen_h}"
        )
    return "；".join(parts)


def _denoise_sequence_metrics(
    width: int,
    height: int,
    context_latents: Sequence[Any] | None,
) -> tuple[int, int, int, float, float]:
    """Return target/reference/joint tokens and 512px linear/quadratic ratios.

    The baseline keeps the same number of reference images.  This makes an
    edit-to-edit comparison rather than overstating it against text-to-image:
    one 512px target plus one 512px reference contains 2048 image tokens.
    """

    target_tokens = (int(height) // 16) * (int(width) // 16)
    references = tuple(context_latents or ())
    reference_tokens = sum(
        (int(value.shape[2]) // 2) * (int(value.shape[3]) // 2)
        for value in references
    )
    joint_tokens = target_tokens + reference_tokens
    baseline_tokens = (1 + len(references)) * ((512 // 16) ** 2)
    token_ratio = joint_tokens / baseline_tokens
    return target_tokens, reference_tokens, joint_tokens, token_ratio, token_ratio * token_ratio


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
    default_scheduler="flow_match_euler_discrete",
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


def _materialize_module(module: Any, label: str, chunk: int = FIBO2_EVAL_CHUNK) -> None:
    """分块物化加载/量化后的参数，避免把惰性工作推迟到首次推理。

    Fibo-2 Transformer 的 BF16 存档约 16 GB。q8 转换由 MLX 惰性构图；如果加载后
    不显式求值，用户会看到“模型已加载”后第一个去噪 step 长时间不动。整棵参数树
    一次性 ``mx.eval`` 又可能形成超长 Metal command buffer，因此沿用 H3/Qwen 2.1
    加载器验证过的 8 数组分块策略。
    """

    arrays = [value for _name, value in tree_flatten(module.parameters())]
    if not arrays:
        return
    step = max(1, int(chunk))
    total_chunks = (len(arrays) + step - 1) // step
    report_every = max(1, total_chunks // 10)
    for chunk_index, start in enumerate(range(0, len(arrays), step), start=1):
        throw_if_interrupted()
        mx.eval(*arrays[start : start + step])
        mx.synchronize()
        if chunk_index == total_chunks or chunk_index % report_every == 0:
            _log(
                f"[Fibo-2 加载] {label} 参数物化 "
                f"{chunk_index}/{total_chunks} | {runtime.memory_snapshot()}"
            )


def load_bundle(
    handle: Fibo2ModelHandle,
    cache=CACHE,
    *,
    clip: Fibo2ClipHandle | None = None,
    on_component_loaded: Callable[[str], None] | None = None,
) -> Fibo2MLXBundle:
    """Load the DiT and Qwen3-VL with the selected sequence budget.

    ``clip`` is optional for compatibility with direct callers.  Omitting it
    uses ``Fibo2ClipHandle``'s Diffusers-aligned 4096 default.
    """

    clip = clip or Fibo2ClipHandle(model_path=handle.model_path)
    if Path(clip.model_path).expanduser() != Path(handle.model_path).expanduser():
        raise Fibo2ValidationError(
            "Fibo-2 文本编码器与 Transformer 必须选择同一个 checkpoint："
            f"{clip.model_path!r} != {handle.model_path!r}"
        )
    root = _local_root(handle.model_path)
    # max_length is a per-encode budget, not a weight/configuration difference.
    # Keeping it out of this key prevents a 4096 -> 8192 widget change from
    # retaining a second complete DiT + Qwen3-VL bundle in unified memory.
    model_weight_config = {
        "model_path": handle.model_path,
        "dtype": handle.dtype,
        "device": handle.device,
        "loras": handle.loras,
        "quantize": handle.quantize,
    }
    clip_weight_config = {
        "model_path": clip.model_path,
        "dtype": clip.dtype,
        "device": clip.device,
        "loras": clip.loras,
        "quantize": clip.quantize,
    }
    key = runtime.cache_key(
        {
            "kind": "fibo2_mlx_bundle",
            # compile_cache_limit is a runtime allocator policy.  It must not
            # retain a second 17+ GB bundle when only that widget changes.
            "model": model_weight_config,
            "clip": clip_weight_config,
            "root": str(root),
        }
    )

    def build() -> Fibo2MLXBundle:
        started = time.perf_counter()
        _log("[Fibo-2 加载] 1/3 Transformer 权重…")
        transformer, _ = components.create_and_load(
            FIBO2_ENTRY,
            "transformer",
            "dir",
            str(_component_path(root, "transformer")),
            int(handle.quantize) or None,
            class_kwargs=_transformer_kwargs(root),
        )
        _materialize_module(transformer, "Transformer")
        _log(
            f"[Fibo-2 加载] 1/3 Transformer 完成，耗时 "
            f"{time.perf_counter() - started:.2f}s | {runtime.memory_snapshot()}"
        )
        if on_component_loaded is not None:
            on_component_loaded("加载 Transformer")
        started = time.perf_counter()
        _log("[Fibo-2 加载] 2/3 Qwen3-VL 权重…")
        text_encoder, _ = components.create_and_load(
            FIBO2_ENTRY,
            "text_encoder",
            "dir",
            str(_component_path(root, "text_encoder")),
            None,
            class_kwargs={"max_length": FIBO2_DIFFUSERS_MAX_SEQUENCE_LENGTH},
        )
        _materialize_module(text_encoder, "Qwen3-VL")
        _log(
            f"[Fibo-2 加载] 2/3 Qwen3-VL 完成，耗时 "
            f"{time.perf_counter() - started:.2f}s | {runtime.memory_snapshot()}"
        )
        if on_component_loaded is not None:
            on_component_loaded("加载 Qwen3-VL")
        started = time.perf_counter()
        _log("[Fibo-2 加载] 3/3 tokenizer / processor…")
        tokenizer = components.load_tokenizer(
            FIBO2_ENTRY,
            "tokenizer",
            "dir",
            str(_component_path(root, "processor")),
            # The native encoder tokenizes without truncation and enforces the
            # selected request budget itself.  Keep the wrapper permissive so
            # it cannot reintroduce a hidden 4096-token truncation later.
            max_length=FIBO2_TOKENIZER_MODEL_MAX_LENGTH,
        )
        _log(
            f"[Fibo-2 加载] 3/3 tokenizer / processor 完成，耗时 "
            f"{time.perf_counter() - started:.2f}s"
        )
        if on_component_loaded is not None:
            on_component_loaded("加载 tokenizer / processor")
        return Fibo2MLXBundle(transformer, text_encoder, tokenizer)

    bundle, hit = cache.get_or_create("module", key, build)
    if hit and on_component_loaded is not None:
        _log("[Fibo-2 加载] Transformer + Qwen3-VL + tokenizer 命中缓存")
        for label in ("加载 Transformer（缓存）", "加载 Qwen3-VL（缓存）", "加载 tokenizer（缓存）"):
            on_component_loaded(label)
    return bundle


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
    return [
        _resize_crop(image, target_width, target_height)
        for image, (target_width, target_height) in zip(
            images, context_target_sizes(images, width, height), strict=True
        )
    ]


def context_target_sizes(
    images: Sequence[Any], width: int, height: int
) -> list[tuple[int, int]]:
    """返回参考图进入 VAE 时的尺寸，不执行代价较高的像素缩放。"""

    if not images:
        return []
    if len(images) == 1:
        return [(int(width), int(height))]
    return [_trained_image_size(image.width / image.height) for image in images]


def preprocess_text_encoder_context(
    images: Sequence[Any],
    width: int,
    height: int,
    image_size: int = FIBO2_TEXT_ENCODER_IMAGE_SIZE,
) -> list[Any]:
    """Prepare the Qwen3-VL copies exactly as the official Fibo-2 pipeline.

    The VLM must see the same crops used for the context latents, with each crop
    bounded to 768 pixels on its longest side.  Passing the original source
    image here can by itself exceed the current MLX encoder's Diffusers-aligned
    4096-token sequence budget, even when the edit instruction is only a few
    words.  That budget is not presented as a Bria model-architecture limit.
    """

    if int(image_size) < 1:
        raise ValueError("Fibo-2 text encoder image_size 必须大于 0")
    result = []
    for image in _preprocess_context(images, width, height):
        text_encoder_image = image.copy()
        text_encoder_image.thumbnail((int(image_size), int(image_size)))
        result.append(text_encoder_image)
    return result


def _patchify(latents: mx.array) -> mx.array:
    batch, channels, height, width = latents.shape
    value = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
    return value.transpose(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height // 2, width // 2)


def _unpatchify(latents: mx.array) -> mx.array:
    batch, channels, height, width = latents.shape
    value = latents.reshape(batch, channels // 4, 2, 2, height, width)
    return value.transpose(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, height * 2, width * 2)


def load_vae(handle: Fibo2VaeHandle, cache=CACHE) -> Any:
    """Materialize only the Fibo-2 VAE for encoder/decoder nodes."""

    root = _local_root(handle.model_path)

    def build():
        vae, _ = components.create_and_load(
            FIBO2_ENTRY,
            "vae",
            "dir",
            str(_component_path(root, "vae")),
            None,
        )
        return vae

    return cache.get_or_create("module", handle.cache_key, build)[0]


def encode_context(
    vae: Any,
    images: Sequence[Any],
    width: int,
    height: int,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[mx.array]:
    """Encode Fibo-2 context images in the dedicated VAE encoder node."""

    result = []
    prepared = _preprocess_context(images, width, height)
    total = len(prepared)
    for index, image in enumerate(prepared):
        throw_if_interrupted()
        started = time.perf_counter()
        source = images[index]
        _log(
            f"[Fibo-2 参考图] VAE 编码 {index + 1}/{total} 开始："
            f"原图 {source.width}×{source.height} → 实际编码 {image.width}×{image.height}"
        )
        pixels = _as_mlx_image(image)
        latent = vae.encode(pixels)
        packed = _patchify(latent)
        mean = mx.array(vae.bn.running_mean).reshape(1, -1, 1, 1)
        variance = mx.array(vae.bn.running_var).reshape(1, -1, 1, 1)
        packed = (packed - mean) / mx.sqrt(variance + 1e-4)
        encoded = _unpatchify(packed)
        # MLX 是惰性求值；必须在编码节点内物化，否则真正的 VAE 工作会被推迟到
        # 采样器第 1 步，看起来就像去噪循环卡死，同时 VAE 还可能被过早释放。
        mx.eval(encoded)
        result.append(encoded)
        _log(
            f"[Fibo-2 参考图] VAE 编码 {index + 1}/{total} 完成，"
            f"耗时 {time.perf_counter() - started:.2f}s | {runtime.memory_snapshot()}"
        )
        if on_progress is not None:
            on_progress(index + 1, total)
    return result


def decode_latent(
    handle: Fibo2VaeHandle,
    latent: mx.array,
    cache=CACHE,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> mx.array:
    """Decode a native MLX latent in the dedicated VAE decoder node."""

    total = 2
    throw_if_interrupted()
    started = time.perf_counter()
    _log(f"[Fibo-2 解码] 阶段 1/{total}：加载 VAE…")
    vae = load_vae(handle, cache)
    _log(
        f"[Fibo-2 解码] 阶段 1/{total} 完成，耗时 "
        f"{time.perf_counter() - started:.2f}s | {runtime.memory_snapshot()}"
    )
    if on_progress is not None:
        on_progress("加载 VAE", 1, total)
    throw_if_interrupted()
    started = time.perf_counter()
    _log(f"[Fibo-2 解码] 阶段 2/{total}：VAE 解码图像…")
    packed = _patchify(latent)
    mean = mx.array(vae.bn.running_mean).reshape(1, -1, 1, 1)
    variance = mx.array(vae.bn.running_var).reshape(1, -1, 1, 1)
    plain = _unpatchify(packed * mx.sqrt(variance + 1e-4) + mean)
    decoded = vae.decode(plain)
    mx.eval(decoded)
    _log(
        f"[Fibo-2 解码] 阶段 2/{total} 完成，耗时 "
        f"{time.perf_counter() - started:.2f}s | {runtime.memory_snapshot()}"
    )
    if on_progress is not None:
        on_progress("VAE 解码", 2, total)
    return decoded


def run(
    model: Fibo2ModelHandle,
    *,
    clip: Fibo2ClipHandle | None = None,
    prompt: str,
    negative_prompt: str | None,
    images: Sequence[Any] | None,
    context_latents: Sequence[mx.array] | None = None,
    reference_size_routes: Sequence[Sequence[int]] | None = None,
    seed: int,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    batch_size: int,
    cache=CACHE,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> MlxLatentHandle:
    context = list(images or ())
    validate_request(model.model_path, context, None, steps, guidance)
    validate_dimensions(width, height)
    clip = clip or Fibo2ClipHandle(model_path=model.model_path)
    cfg_active = guidance > 1.0
    # 三个 bundle 子组件 + 正向条件 + 可选负向条件 + N 个去噪 step。
    total_stages = int(steps) + 4 + int(cfg_active)
    completed = 0
    overall_started = time.perf_counter()
    forwards_per_step = 2 if cfg_active else 1
    latent_h, latent_w = height // 8, width // 8
    (
        image_tokens,
        reference_tokens,
        joint_image_tokens,
        token_ratio,
        attention_ratio,
    ) = _denoise_sequence_metrics(
        width,
        height,
        context_latents,
    )
    _log(
        f"[Fibo-2] 开始：目标生成 {width}×{height}，batch={batch_size}，steps={steps}，"
        f"参考图={len(context)}，max_length={clip.max_length}，"
        f"CFG={'开启' if cfg_active else '关闭'}（每步 {forwards_per_step} 次 Transformer 前向），"
        f"latent={latent_w}×{latent_h}，目标 tokens={image_tokens}，"
        f"参考 tokens={reference_tokens}，联合图像 tokens={joint_image_tokens}"
    )
    size_route_text = format_size_routes(reference_size_routes or ())
    if size_route_text:
        _log(f"[Fibo-2] 参考图尺寸链路：{size_route_text}；生成目标 {width}×{height}")
    elif context:
        qwen_sizes = "、".join(f"{image.width}×{image.height}" for image in context)
        _log(f"[Fibo-2] Qwen3-VL 实际参考图尺寸：{qwen_sizes}")
    if width * height > 1024 * 1024:
        _log(
            f"[Fibo-2 警告] 当前目标 {width}×{height} 超过 1MP；"
            f"目标与 {len(context_latents or ())} 份参考 latent 会进入同一自注意力流，"
            f"联合图像 tokens={joint_image_tokens}。相对同样参考图数量的 512×512 任务，"
            f"序列约 {token_ratio:.1f} 倍、全注意力 token-pair 约 {attention_ratio:.1f} 倍；"
            "每个 Transformer 步骤可能极慢并占用大量统一内存。"
        )

    def finish_stage(label: str, started: float) -> None:
        nonlocal completed
        completed += 1
        _log(
            f"[Fibo-2] 阶段 {completed}/{total_stages} 完成：{label}，"
            f"耗时 {time.perf_counter() - started:.2f}s | {runtime.memory_snapshot()}"
        )
        if on_progress is not None:
            on_progress(label, completed, total_stages)

    def advance_load_stage(label: str) -> None:
        nonlocal completed
        completed += 1
        _log(f"[Fibo-2] 总进度 {completed}/{total_stages}：{label}完成")
        if on_progress is not None:
            on_progress(label, completed, total_stages)
        throw_if_interrupted()

    throw_if_interrupted()
    _log(
        f"[Fibo-2] 阶段 1/{total_stages}：加载 Transformer"
        "（首次运行需要读取并转换权重）…"
    )
    bundle = load_bundle(
        model,
        cache,
        clip=clip,
        on_component_loaded=advance_load_stage,
    )
    # Diffusers' Fibo-2 pipeline deliberately samples and retains latents in
    # float32, independent of the transformer/VAE weight dtype.  Keeping this
    # precision through scheduler.step is important for the flow-matching
    # trajectory; casting the initial noise to bfloat16 produces visibly noisy
    # decoded images.
    noise = mx.random.normal(
        (batch_size, 32, latent_h, latent_w), key=mx.random.key(int(seed)), dtype=mx.float32
    )
    tokenizer = bundle.tokenizer
    throw_if_interrupted()
    started = time.perf_counter()
    _log(
        f"[Fibo-2] 阶段 {completed + 1}/{total_stages}：Qwen3-VL 正向条件编码"
        f"（参考图 {len(context)} 张）…"
    )
    cond, cond_mask = bundle.text_encoder.encode(
        tokenizer, prompt, context, max_length=clip.max_length
    )
    finish_stage(f"Qwen3-VL 正向条件编码（{int(cond.shape[2])} tokens）", started)
    text_lengths = np.asarray(cond_mask).sum(axis=1).tolist()
    gist_counts = compute_gist_count(
        text_lengths,
        int(getattr(bundle.transformer, "min_num_gist_tokens", 256)),
        int(getattr(bundle.transformer, "max_num_gist_tokens", 512)),
        int(getattr(bundle.transformer, "gist_step", 64)),
        int(getattr(bundle.transformer, "gist_min_text_len", 256)),
        int(getattr(bundle.transformer, "gist_max_text_len", 2048)),
    )
    num_gist = max(gist_counts)
    _log(
        f"[Fibo-2 去噪] 实际序列：目标 {image_tokens} + 参考 {reference_tokens} "
        f"+ gist {num_gist} = 主干 {joint_image_tokens + num_gist} tokens；"
        f"noise refiner 处理 {joint_image_tokens} tokens"
    )
    uncond = uncond_mask = None
    if cfg_active:
        # The official pipeline only encodes the negative branch when CFG is
        # active, and edit images condition the positive branch only.
        throw_if_interrupted()
        started = time.perf_counter()
        _log(
            f"[Fibo-2] 阶段 {completed + 1}/{total_stages}：Qwen3-VL 负向条件编码…"
        )
        uncond, uncond_mask = bundle.text_encoder.encode(
            tokenizer,
            negative_prompt or FIBO2_DEFAULT_NEGATIVE,
            max_length=clip.max_length,
        )
        finish_stage(f"Qwen3-VL 负向条件编码（{int(uncond.shape[2])} tokens）", started)
    if batch_size != 1:
        cond = mx.broadcast_to(cond, (batch_size, *cond.shape[1:]))
        cond_mask = mx.broadcast_to(cond_mask, (batch_size, cond_mask.shape[1]))
        if uncond is not None and uncond_mask is not None:
            uncond = mx.broadcast_to(uncond, (batch_size, *uncond.shape[1:]))
            uncond_mask = mx.broadcast_to(uncond_mask, (batch_size, uncond_mask.shape[1]))
    prepared_context = list(context_latents) if context_latents is not None else []
    if len(prepared_context) != len(context):
        raise Fibo2ValidationError(
            "Fibo-2 原始参考图与 context latent 数量不一致；请重新运行 MlxVAEEncoder"
        )
    context_latents = prepared_context or None
    if context_latents and batch_size != 1:
        context_latents = [mx.broadcast_to(value, (batch_size, *value.shape[1:])) for value in context_latents]
    scheduler = Fibo2Scheduler()
    timesteps = scheduler.set_timesteps(steps, (latent_h // 2) * (latent_w // 2))
    current = noise
    timestep_text = ", ".join(f"{float(value):.3f}" for value in timesteps)
    total_forwards = len(timesteps) * forwards_per_step
    _log(
        f"[Fibo-2 去噪] 计划：共 {len(timesteps)} 步，"
        f"Transformer 前向共 {total_forwards} 次，timesteps=[{timestep_text}]"
    )
    denoise_started = time.perf_counter()
    step_durations: list[float] = []
    for step, timestep in enumerate(timesteps):
        throw_if_interrupted()
        started = time.perf_counter()
        _log(
            f"[Fibo-2 去噪] 步数 {step + 1}/{len(timesteps)} 开始 | "
            f"总阶段 {completed + 1}/{total_stages} | t={float(timestep):.3f} | "
            f"本步 Transformer 前向 {forwards_per_step} 次 | "
            f"去噪已用 {_format_duration(started - denoise_started)}"
        )
        t = mx.full((batch_size,), float(timestep) / 1000.0, dtype=mx.float32)
        with _denoise_heartbeat(step + 1, len(timesteps), started):
            prediction = bundle.transformer(current, t, cond, cond_mask, context_latents)
            if cfg_active:
                uncond_prediction = bundle.transformer(current, t, uncond, uncond_mask, context_latents)
                prediction = uncond_prediction + float(guidance) * (prediction - uncond_prediction)
            current = scheduler.step(prediction, current)
            mx.eval(current)
        finished = time.perf_counter()
        step_duration = finished - started
        step_durations.append(step_duration)
        average = sum(step_durations) / len(step_durations)
        remaining_steps = len(timesteps) - step - 1
        eta = average * remaining_steps
        completed += 1
        label = f"去噪 step {step + 1}/{len(timesteps)}"
        _log(
            f"[Fibo-2 去噪] 步数 {step + 1}/{len(timesteps)} 完成 | "
            f"本步 {_format_duration(step_duration)} | "
            f"去噪累计 {_format_duration(finished - denoise_started)} | "
            f"平均 {_format_duration(average)}/步 | "
            f"预计剩余 {_format_duration(eta)} | "
            f"Transformer 前向累计 {(step + 1) * forwards_per_step}/{total_forwards} | "
            f"总阶段 {completed}/{total_stages} | {runtime.memory_snapshot()}"
        )
        if on_progress is not None:
            on_progress(label, completed, total_stages)
    _log(
        f"[Fibo-2] 采样完成：{len(timesteps)}/{len(timesteps)} 步，总耗时 "
        f"{_format_duration(time.perf_counter() - overall_started)} | "
        f"{runtime.memory_snapshot()}"
    )
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
    "FIBO2_TEXT_ENCODER_IMAGE_SIZE",
    "Fibo2MLXBundle",
    "aligned_dimensions",
    "context_target_sizes",
    "decode_latent",
    "encode_context",
    "format_size_routes",
    "load_bundle",
    "load_vae",
    "preprocess_text_encoder_context",
    "run",
    "validate_dimensions",
]