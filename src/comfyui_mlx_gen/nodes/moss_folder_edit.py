"""Standalone resumable MOSS Image Edit folder node.

Each source image is an independent MOSS reference-edit request.  The node
keeps only handles and short-lived cache entries between requests, so a large
folder does not retain every reference latent, text condition, K/V prefix, or
sampled latent until the final image is returned.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .. import image as image_mod
from .. import pipeline, runtime
from ..cache import Cache
from ..progress import SamplingProgress
from ..types import (
    CLIP,
    MOSS_MAX_LENGTH,
    MlxClipHandle,
    MlxConditioning,
    MlxModelHandle,
    MlxPilImage,
    MlxVaeHandle,
    entry_for,
    model,
    vae,
    validate_model_family,
)
from .directory_images import IMAGE_EXTENSIONS, _image_paths, _natural_sort_key, _safe_output_name


UPLOAD_ROOT = "mlx_moss"
OUTPUT_ROOT = "moss-folder-edit"
SCHEDULER = "moss_flow_match_euler"
RESOLUTION_TIMINGS = ("采样时降分辨率", "先降分辨率", "后降分辨率")
DEFAULT_PROMPT = "保持原图主体、构图和视角，只按照提示词编辑图像。"
DEFAULT_NEGATIVE_PROMPT = "低清晰度，模糊，畸形，水印，黑边，裁切主体"


def _load_one_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            return source.convert("RGB").copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"无法读取图片：{path}（{exc}）") from exc


def _load_existing_output(path: Path) -> Image.Image | None:
    """Return a valid existing PNG; incomplete/corrupt files are retried."""

    if not path.is_file() or path.suffix.casefold() != ".png":
        return None
    try:
        with Image.open(path) as existing:
            existing.verify()
        with Image.open(path) as existing:
            return existing.convert("RGB").copy()
    except (OSError, UnidentifiedImageError, ValueError):
        return None


def _comfy_input_directory() -> Path:
    try:
        import folder_paths
    except ModuleNotFoundError:
        return Path.cwd() / "input"
    return Path(folder_paths.get_input_directory())


def _comfy_output_directory() -> Path:
    try:
        import folder_paths
    except ModuleNotFoundError:
        return Path.cwd() / "output"
    return Path(folder_paths.get_output_directory())


def _resolve_input_directory(folder_token: str) -> Path:
    """Resolve a picker token or a manually entered server-side directory.

    Browser folder selections are uploaded below ComfyUI's ``input/``
    directory and represented by a relative ``mlx_moss/...`` token.  A user
    may also type an absolute path that exists on the machine running
    ComfyUI; the browser cannot expose its own local path to the server.
    """

    token = str(folder_token).strip().replace("\\", "/")
    if not token:
        raise ValueError("请点击「选择输入文件夹」，或输入 ComfyUI 服务器上的文件夹路径")

    manual_path = Path(token).expanduser()
    if manual_path.is_absolute():
        resolved = manual_path.resolve()
        if not resolved.is_dir():
            raise ValueError(f"手动输入的文件夹不存在或不是目录：{resolved}")
        return resolved

    relative = Path(token)
    if ".." in relative.parts:
        raise ValueError("输入文件夹不能包含 ..")
    if not relative.parts or relative.parts[0] != UPLOAD_ROOT:
        raise ValueError(
            f"相对路径必须来自「选择输入文件夹」上传的 {UPLOAD_ROOT}/ 目录；"
            "手动输入时请填写服务器上的绝对路径"
        )

    root = _comfy_input_directory().expanduser().resolve()
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("输入文件夹路径不能经过符号链接")
    resolved = current.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("输入文件夹必须位于 ComfyUI input/ 目录内") from exc
    if not resolved.is_dir():
        raise ValueError(f"未找到已上传的输入文件夹：{resolved}")
    return resolved


def _secure_image_paths(input_directory: Path) -> list[Path]:
    for child in input_directory.iterdir():
        if child.is_symlink():
            raise ValueError(f"输入文件夹不允许符号链接：{child}")
    return _image_paths(str(input_directory))


def _output_directory(input_directory: Path) -> Path:
    input_root = _comfy_input_directory().expanduser().resolve()
    output_root = _comfy_output_directory().expanduser().resolve()
    try:
        relative = input_directory.relative_to(input_root)
    except ValueError:
        # Manual paths can live outside input/. Keep outputs under ComfyUI's
        # output directory and add a short path digest so two folders with the
        # same basename cannot accidentally resume each other's results.
        folder_name = input_directory.name or "root"
        digest = hashlib.sha256(str(input_directory).encode("utf-8")).hexdigest()[:12]
        relative = Path("manual") / f"{folder_name}-{digest}"
    output = (output_root / OUTPUT_ROOT / relative).resolve()
    try:
        output.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("输出目录必须位于 ComfyUI output/ 目录内") from exc
    output.mkdir(parents=True, exist_ok=True)
    return output


def _evict_if_present(cache: Cache, bucket: str, key: str) -> None:
    if key:
        cache.evict(bucket, key)


def _cleanup_image(cache: Cache, reference_key: str, positive_key: str, negative_key: str, kv_keys: tuple[str, ...], latent) -> None:
    """Evict all per-image arrays, continuing if one cleanup hook fails."""

    cleanups = (
        lambda: _evict_if_present(cache, "moss_ref_encoding", reference_key),
        lambda: _evict_if_present(cache, "prompt_encoding", positive_key),
        lambda: _evict_if_present(cache, "prompt_encoding", negative_key),
        *(lambda key=key: _evict_if_present(cache, "moss_ref_kv", key) for key in kv_keys),
        lambda: _evict_if_present(
            cache, "component_weights", latent.cache_key if latent is not None else ""
        ),
        runtime.flush_caches,
    )
    for cleanup in cleanups:
        try:
            cleanup()
        except Exception as exc:  # noqa: BLE001
            print(f"[MlxMossFolderEdit] 图片缓存清理失败，继续处理：{exc}")


def _cleanup_resources(entry, model_handle, clip, vae_handle, cache) -> None:
    cleanups = (
        pipeline.clear_compiled_predict_caches,
        lambda: pipeline.release_sampler_components(entry, model_handle, cache),
        lambda: pipeline.release_encoder(clip, cache),
        lambda: pipeline.release_vae_component(vae_handle, cache),
        runtime.flush_caches,
    )
    for cleanup in cleanups:
        try:
            cleanup()
        except Exception as exc:  # noqa: BLE001
            print(f"[MlxMossFolderEdit] 释放资源失败，继续清理：{exc}")


class MlxMossFolderEdit:
    """Process supported images independently with resumable MOSS Image Edit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_directory": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可点击「选择输入文件夹」上传图片，也可直接输入 ComfyUI 服务器上的文件夹绝对路径",
                    },
                ),
                "vae": (vae, {}),
                "clip": (CLIP, {}),
                "model": (model, {}),
                "prompt": ("STRING", {"default": DEFAULT_PROMPT, "multiline": True}),
                "negative_prompt": (
                    "STRING",
                    {"default": DEFAULT_NEGATIVE_PROMPT, "multiline": True},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**63 - 1, "control_after_generate": True},
                ),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0}),
                "resolution_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "按 MOSS 参考图预处理后的宽高乘以此系数；例如 0.5 表示目标尺寸减半，具体作用取决于缩放时机",
                    },
                ),
                "resolution_timing": (
                    list(RESOLUTION_TIMINGS),
                    {
                        "default": RESOLUTION_TIMINGS[0],
                        "tooltip": "选择只缩小采样画布、先缩小参考图后采样，或在生成完成后缩小最终图片",
                    },
                ),
            }
        }

    RETURN_TYPES = ("images", "STRING", "STRING", "INT")
    RETURN_NAMES = ("images", "saved_files", "report", "success_count")
    FUNCTION = "edit_folder"
    CATEGORY = "MLX/Gen/Image"

    @staticmethod
    def _validate_handles(vae_handle, clip, model_handle) -> None:
        for value, expected, label in (
            (vae_handle, MlxVaeHandle, "vae"),
            (clip, MlxClipHandle, "clip"),
            (model_handle, MlxModelHandle, "model"),
        ):
            if not isinstance(value, expected):
                raise ValueError(f"{label} 必须连接对应的 MLX 加载器输出")
        for handle, label in ((vae_handle, "VAE"), (clip, "CLIP"), (model_handle, "模型")):
            if handle.model_type != "moss":
                raise ValueError(f"{label} 必须选择 moss，收到 {handle.model_type!r}")
        validate_model_family("moss", vae_handle.path)
        validate_model_family("moss", clip.path)
        validate_model_family("moss", model_handle.model_path)

    @staticmethod
    def _condition_text(text: str) -> str:
        return text if text and text.strip() else " "

    @staticmethod
    def _scaled_dimensions(image: Image.Image, resolution_scale: float) -> tuple[int, int]:
        """Return a scaled MOSS canvas aligned to its 16-pixel latent grid."""

        scale = float(resolution_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"分辨率系数必须是大于 0 的有限数字，收到 {resolution_scale!r}")

        def aligned(value: int) -> int:
            # MOSS spatial latents require both target dimensions to be a
            # positive multiple of 16, even when a very small scale is used.
            return max(16, int(value * scale) // 16 * 16)

        return aligned(image.width), aligned(image.height)

    @staticmethod
    def _resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        if image.size == size:
            return image
        return image.resize(size, Image.Resampling.LANCZOS)

    @staticmethod
    def _cache_key(kind: str, **values) -> str:
        return runtime.cache_key({"kind": kind, "version": 1, **values})

    def edit_folder(
        self,
        input_directory,
        vae,
        clip,
        model,
        prompt,
        negative_prompt,
        seed,
        steps,
        guidance=1.0,
        resolution_scale=1.0,
        resolution_timing=RESOLUTION_TIMINGS[0],
    ):
        self._validate_handles(vae, clip, model)
        if resolution_timing not in RESOLUTION_TIMINGS:
            raise ValueError(f"未知分辨率缩放时机：{resolution_timing!r}")
        steps = int(steps)
        if steps < 1:
            raise ValueError(f"steps 必须大于 0，收到 {steps}")
        guidance = float(guidance)
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError(f"guidance 必须是大于等于 0 的有限数字，收到 {guidance!r}")
        resolution_scale = float(resolution_scale)
        if not math.isfinite(resolution_scale) or resolution_scale <= 0:
            raise ValueError(f"分辨率系数必须是大于 0 的有限数字，收到 {resolution_scale!r}")

        resolved_input = _resolve_input_directory(input_directory)
        paths = _secure_image_paths(resolved_input)
        if not paths:
            extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
            raise ValueError(f"输入目录没有可读取的图片：{input_directory}（支持：{extensions}）")
        ordered_paths = sorted(paths, key=_natural_sort_key)
        output = _output_directory(resolved_input)
        total = len(ordered_paths)
        progress = SamplingProgress(total)
        generated: list[Image.Image | None] = [None] * total
        saved_paths: list[str | None] = [None] * total
        records: list[dict[str, object] | None] = [None] * total
        pending: list[tuple[int, Path, Path]] = []

        for index, source_path in enumerate(ordered_paths):
            destination = output / _safe_output_name(source_path.name, index)
            existing = _load_existing_output(destination)
            if existing is None:
                pending.append((index, source_path, destination))
                continue
            generated[index] = existing
            saved_paths[index] = str(destination)
            records[index] = {
                "index": index,
                "source": str(source_path),
                "output": str(destination),
                "status": "skipped",
                "reason": "输出文件已存在且可读取",
            }
            progress.update()

        if not pending:
            report = json.dumps([record for record in records if record is not None], ensure_ascii=False, indent=2)
            return (
                MlxPilImage(images=tuple(image for image in generated if image is not None)),
                "\n".join(path for path in saved_paths if path is not None),
                report,
                total,
            )

        entry = entry_for(model.model_type)
        cache = Cache(
            caps={
                "module": 4,
                "prompt_encoding": 4,
                "component_weights": 2,
                "moss_ref_encoding": 1,
                "moss_ref_kv": 2,
            }
        )
        positive_text = self._condition_text(prompt)
        negative_text = self._condition_text(negative_prompt)
        encoder_key = runtime.cache_key({"kind": "module", "clip": clip})
        sampler_components = None
        encoder_components = None
        vae_module = None

        try:
            runtime.apply_cache_limit(model.compile_cache_limit, label=entry.family)
            sampler_components = pipeline.prepare_sampler_components(entry, model, cache)
            encoder_components = pipeline.prepare_encoder(entry, clip, cache, encoder_key)
            vae_module = pipeline.vae_component(vae, cache)
            for index, source_path, destination in pending:
                reference_key = positive_key = negative_key = ""
                kv_keys: tuple[str, ...] = ()
                latent = None
                try:
                    source_image = _load_one_image(source_path)
                    # MOSS's existing reference preprocessing (including its
                    # area cap and 16-pixel center crop) remains the base
                    # resolution.  This keeps the default path unchanged and
                    # avoids introducing an unsupported reference-resolution
                    # mode into the MOSS encoder.
                    base_image = pipeline.prepare_moss_reference_image(source_image)
                    output_width, output_height = self._scaled_dimensions(
                        base_image, resolution_scale
                    )
                    if resolution_timing == RESOLUTION_TIMINGS[0]:
                        generation_image = base_image
                        width, height = output_width, output_height
                    elif resolution_timing == RESOLUTION_TIMINGS[1]:
                        generation_image = self._resize_image(
                            base_image, (output_width, output_height)
                        )
                        width, height = output_width, output_height
                    else:
                        generation_image = base_image
                        width, height = base_image.size
                    # ``generation_image`` is either the already prepared
                    # image or a 16-pixel-aligned resize of it.  Do not run
                    # MOSS's area-capping preprocessor a second time: that
                    # would undo an explicitly requested scale above 1.0.
                    prepared = generation_image
                    digest = image_mod.digest([source_image])
                    common_key_values = {
                        "source": str(source_path.resolve()),
                        "digest": digest,
                        "source_size": tuple(source_image.size),
                        "prepared_size": tuple(prepared.size),
                        "sampling_size": (int(width), int(height)),
                        "output_size": (int(output_width), int(output_height)),
                        "resolution_scale": resolution_scale,
                        "resolution_timing": resolution_timing,
                        "vae": vae,
                        "clip": clip,
                        "model": model,
                        "prompt": positive_text,
                        "negative_prompt": negative_text,
                        "steps": steps,
                        "guidance": guidance,
                        "seed": int(seed) + index,
                    }
                    reference_key = self._cache_key("moss_folder_reference", **common_key_values)
                    pipeline.encode_moss_reference_image(
                        vae_module, prepared, cache, reference_key
                    )
                    positive_key = self._cache_key(
                        "moss_folder_prompt",
                        branch="positive",
                        **common_key_values,
                    )
                    negative_key = self._cache_key(
                        "moss_folder_prompt",
                        branch="negative",
                        **common_key_values,
                    )
                    pipeline.encode_text(
                        entry,
                        encoder_components,
                        positive_text,
                        cache,
                        positive_key,
                        max_length=MOSS_MAX_LENGTH,
                    )
                    pipeline.encode_text(
                        entry,
                        encoder_components,
                        negative_text,
                        cache,
                        negative_key,
                        max_length=MOSS_MAX_LENGTH,
                    )
                    positive = MlxConditioning(
                        clip=clip,
                        text=positive_text,
                        encoding_key=positive_key,
                    )
                    negative = MlxConditioning(
                        clip=clip,
                        text=negative_text,
                        encoding_key=negative_key,
                    )
                    moss_reference_kv_key = pipeline.moss_reference_kv_cache_key(
                        model, reference_key, positive_key
                    )
                    negative_kv_key = pipeline.moss_reference_kv_cache_key(
                        model, reference_key, negative_key
                    )
                    kv_keys = (moss_reference_kv_key, negative_kv_key)
                    params = {
                        "seed": int(seed) + index,
                        "steps": steps,
                        "height": int(height),
                        "width": int(width),
                        "batch_size": 1,
                        "positive_encoding_key": positive.encoding_key,
                        "negative_encoding_key": negative.encoding_key,
                        "guidance": guidance,
                        "scheduler_name": SCHEDULER,
                        "compile_model": False,
                        "cache_limit_gb": int(model.compile_cache_limit),
                        "ref_cache_key": "",
                        "ref_count": 0,
                        "moss_ref_cache_key": reference_key,
                        "moss_kv_cache_key": moss_reference_kv_key,
                        "moss_sampling_version": 10,
                        "kv_cache": "off",
                    }
                    latent = pipeline.run_sampler(entry, model, sampler_components, params, cache)
                    latent_array, hit = cache.get("component_weights", latent.cache_key)
                    if not hit:
                        raise RuntimeError("latent 数组不在缓存里，请重新运行文件夹编辑")
                    decoded = pipeline.decode_latents(
                        entry,
                        {"vae": vae_module},
                        latent_array[0],
                        latent.height,
                        latent.width,
                    )
                    result_image = image_mod.to_pil(decoded, batch_index=-1)[0].convert("RGB")
                    if resolution_timing == RESOLUTION_TIMINGS[2]:
                        result_image = self._resize_image(
                            result_image, (output_width, output_height)
                        )
                    result_image.save(destination, format="PNG")
                    if _load_existing_output(destination) is None:
                        raise RuntimeError(f"图片保存后未找到有效 PNG：{destination}")
                    generated[index] = result_image
                    saved_paths[index] = str(destination)
                    records[index] = {
                        "index": index,
                        "source": str(source_path),
                        "output": str(destination),
                        "status": "success",
                        "seed": int(seed) + index,
                        "width": int(output_width),
                        "height": int(output_height),
                        "resolution_scale": resolution_scale,
                        "resolution_timing": resolution_timing,
                    }
                except Exception as exc:  # noqa: BLE001
                    records[index] = {
                        "index": index,
                        "source": str(source_path),
                        "output": str(destination),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"[MlxMossFolderEdit] 处理失败，继续下一张：{source_path} | {exc}")
                finally:
                    _cleanup_image(cache, reference_key, positive_key, negative_key, kv_keys, latent)
                    progress.update()
        finally:
            sampler_components = None
            encoder_components = None
            vae_module = None
            _cleanup_resources(entry, model, clip, vae, cache)

        available_images = [image for image in generated if image is not None]
        available_paths = [path for path in saved_paths if path is not None]
        report_records = [record for record in records if record is not None]
        if not available_images:
            raise RuntimeError(
                "文件夹中没有图片处理成功：\n"
                + json.dumps(report_records, ensure_ascii=False, indent=2)
            )
        return (
            MlxPilImage(images=tuple(available_images)),
            "\n".join(available_paths),
            json.dumps(report_records, ensure_ascii=False, indent=2),
            len(available_images),
        )


__all__ = ["MlxMossFolderEdit", "DEFAULT_PROMPT", "DEFAULT_NEGATIVE_PROMPT"]