"""Standalone resumable FLUX 3 Klein folder-edit node.

The browser uploads a selected folder below ``input/mlx_flux3_klein`` and the
node receives only that relative token.  Every source image is processed as a
separate FLUX 3 edit request; no input folder is used as an output directory.
"""

from __future__ import annotations

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
    MlxClipHandle,
    MlxConditioning,
    MlxModelHandle,
    MlxPilImage,
    MlxReferenceImages,
    MlxVaeHandle,
    entry_for,
    model,
    vae,
    validate_model_family,
)
from .directory_images import IMAGE_EXTENSIONS, _image_paths, _natural_sort_key, _safe_output_name


UPLOAD_ROOT = "mlx_flux3_klein"
OUTPUT_ROOT = "flux3-klein-folder-edit"
SCHEDULER = "flux3_flow_map"
RESIZE_MODES = ("auto", "aspect_area_crop", "keep_resolution")
# Keep the same three scaling timings as the Qwen-Image 2.1 folder node.
# FLUX 3 additionally aligns every sampling canvas side to its 16-pixel grid.
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
    """Return a readable PNG, treating partial/corrupt files as unfinished."""

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
    """Resolve only a picker token below ComfyUI's input directory."""

    token = str(folder_token).strip().replace("\\", "/")
    if not token:
        raise ValueError("请先点击「选择输入文件夹」并选择图片文件夹")
    relative = Path(token)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("输入文件夹必须是 input/ 下的相对目录，不能使用绝对路径或 ..")
    if not relative.parts or relative.parts[0] != UPLOAD_ROOT:
        raise ValueError(f"输入文件夹必须来自「选择输入文件夹」上传的 {UPLOAD_ROOT}/ 目录")

    root = _comfy_input_directory().expanduser().resolve()
    input_path = root
    for part in relative.parts:
        input_path /= part
        if input_path.is_symlink():
            raise ValueError("输入文件夹路径不能经过符号链接")
    input_path = input_path.resolve()
    try:
        input_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("输入文件夹必须位于 ComfyUI input/ 目录内") from exc
    if not input_path.is_dir():
        raise ValueError(f"未找到已上传的输入文件夹：{input_path}")
    return input_path


def _secure_image_paths(input_directory: Path) -> list[Path]:
    """Return direct image files while rejecting every directory symlink."""

    # ``Path.is_file`` follows symlinks, so checking only the result of
    # ``_image_paths`` would allow a link in the uploaded folder to read an
    # arbitrary file outside the selected folder.  Folder editing is deliberately
    # non-recursive; reject links to files and directories before filtering the
    # supported image extensions.
    for child in input_directory.iterdir():
        if child.is_symlink():
            raise ValueError(f"输入文件夹不允许符号链接：{child}")
    return _image_paths(str(input_directory))


def _output_directory(input_directory: Path) -> Path:
    """Create an isolated output directory outside the uploaded input folder."""

    input_root = _comfy_input_directory().expanduser().resolve()
    output_root = _comfy_output_directory().expanduser().resolve()
    output = output_root / OUTPUT_ROOT / input_directory.relative_to(input_root)
    output = output.resolve()
    try:
        output.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("输出目录必须位于 ComfyUI output/ 目录内") from exc
    output.mkdir(parents=True, exist_ok=True)
    return output


def _evict_if_present(cache: Cache, bucket: str, key: str) -> None:
    if key:
        cache.evict(bucket, key)


def _cleanup_flux3_resources(entry, model, clip, vae, cache) -> None:
    """Release every batch-owned resource without letting one cleanup hide another."""

    cleanups = (
        pipeline.clear_compiled_predict_caches,
        lambda: pipeline.release_sampler_components(entry, model, cache),
        lambda: pipeline.release_encoder(clip, cache),
        lambda: pipeline.release_vae_component(vae, cache),
        runtime.flush_caches,
    )
    for cleanup in cleanups:
        try:
            cleanup()
        except Exception as exc:  # noqa: BLE001
            print(f"[MlxFlux3KleinFolderEdit] 释放资源失败，继续清理：{exc}")


def _cleanup_flux3_image(cache: Cache, ref_key: str, positive_key: str, negative_key: str, latent) -> None:
    """Release one image's cache entries without aborting the folder loop."""

    cleanups = (
        lambda: _evict_if_present(cache, "ref_encoding", ref_key),
        lambda: _evict_if_present(cache, "prompt_encoding", positive_key),
        lambda: _evict_if_present(cache, "prompt_encoding", negative_key),
        lambda: _evict_if_present(
            cache,
            "component_weights",
            latent.cache_key if latent is not None else "",
        ),
        runtime.flush_caches,
    )
    for cleanup in cleanups:
        try:
            cleanup()
        except Exception as exc:  # noqa: BLE001
            print(f"[MlxFlux3KleinFolderEdit] 图片缓存清理失败，继续处理：{exc}")


class MlxFlux3KleinFolderEdit:
    """Process supported images independently with FLUX 3 Klein edit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_directory": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "点击「选择输入文件夹」选择图片文件夹；图片会上传到 ComfyUI input/",
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
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**63 - 1,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 16, "min": 1, "max": 100}),
                "resize_mode": (list(RESIZE_MODES), {"default": "auto"}),
                "resolution_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "按每张图片原始宽高乘以此系数；例如 0.5 表示目标尺寸减半，具体作用取决于缩放时机",
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
            if handle.model_type != "flux3":
                raise ValueError(f"{label} 必须选择 flux3，收到 {handle.model_type!r}")
        validate_model_family("flux3", vae_handle.path)
        validate_model_family("flux3", clip.path)
        validate_model_family("flux3", model_handle.model_path)

    @staticmethod
    def _condition_text(text: str) -> str:
        return text if text and text.strip() else " "

    @staticmethod
    def _scaled_dimensions(image: Image.Image, resolution_scale: float) -> tuple[int, int]:
        """Return a scaled FLUX 3 canvas aligned to the 16-pixel packing grid."""

        scale = float(resolution_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"分辨率系数必须是大于 0 的有限数字，收到 {resolution_scale!r}")

        def aligned(value: int) -> int:
            # Very small factors still need one valid FLUX 3 grid cell.
            return max(16, int(value * scale) // 16 * 16)

        return aligned(image.width), aligned(image.height)

    @staticmethod
    def _resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        if image.size == size:
            return image
        return image.resize(size, Image.Resampling.LANCZOS)

    @staticmethod
    def _reference_key(
        vae_handle,
        image: Image.Image,
        resize_mode: str,
        resolution_timing: str,
        resolution_scale: float,
        sampling_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> str:
        return pipeline.reference_image_cache_key(
            vae_handle,
            image_mod.digest([image]),
            {
                "count": 1,
                "resize_mode": resize_mode,
                "resolution_timing": resolution_timing,
                "resolution_scale": float(resolution_scale),
                "reference_width": int(image.width),
                "reference_height": int(image.height),
                "sampling_width": int(sampling_size[0]),
                "sampling_height": int(sampling_size[1]),
                "output_width": int(output_size[0]),
                "output_height": int(output_size[1]),
                "family": "flux3",
            },
        )

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
        resize_mode="auto",
        resolution_scale=1.0,
        resolution_timing=RESOLUTION_TIMINGS[0],
    ):
        self._validate_handles(vae, clip, model)
        if resize_mode not in RESIZE_MODES:
            raise ValueError(f"FLUX 3 参考图只支持 {' / '.join(RESIZE_MODES)}，收到 {resize_mode!r}")
        if resolution_timing not in RESOLUTION_TIMINGS:
            raise ValueError(f"未知分辨率缩放时机：{resolution_timing!r}")
        steps = int(steps)
        if steps < 1:
            raise ValueError(f"steps 必须大于 0，收到 {steps}")
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
        cache = Cache(caps={"module": 4, "prompt_encoding": 4, "ref_encoding": 2, "component_weights": 2})
        positive_text = self._condition_text(prompt)
        negative_text = self._condition_text(negative_prompt)
        encoder_key = runtime.cache_key({"kind": "module", "clip": clip})
        sampler_components = None
        encoder_components = None

        try:
            runtime.apply_cache_limit(model.compile_cache_limit, label=entry.family)
            sampler_components = pipeline.prepare_sampler_components(entry, model, cache)
            encoder_components = pipeline.prepare_encoder(entry, clip, cache, encoder_key)
            for index, source_path, destination in pending:
                ref_key = positive_key = negative_key = ""
                latent = None
                vae_module = None
                try:
                    source_image = _load_one_image(source_path)
                    output_width, output_height = self._scaled_dimensions(
                        source_image, resolution_scale
                    )
                    # “采样时降”只调整目标画布；“先降”同时缩小传入的参考图；
                    # “后降”先用原始画布生成，再缩小解码后的最终图片。
                    if resolution_timing == RESOLUTION_TIMINGS[0]:
                        generation_image = source_image
                        width, height = output_width, output_height
                    elif resolution_timing == RESOLUTION_TIMINGS[1]:
                        generation_image = self._resize_image(
                            source_image, (output_width, output_height)
                        )
                        width, height = output_width, output_height
                    else:
                        generation_image = source_image
                        width, height = self._scaled_dimensions(source_image, 1.0)

                    ref_key = self._reference_key(
                        vae,
                        generation_image,
                        resize_mode,
                        resolution_timing,
                        resolution_scale,
                        (width, height),
                        (output_width, output_height),
                    )
                    vae_module = pipeline.vae_component(vae, cache)
                    packed, ids, ref_width, ref_height = pipeline.encode_flux3_reference_images(
                        vae_module, [generation_image], cache, ref_key, 1, resize_mode
                    )
                    cache.get_or_create(
                        "ref_encoding", ref_key, lambda: (packed, ids, ref_width, ref_height)
                    )
                    reference = MlxReferenceImages(
                        model_type=vae.model_type,
                        vae_path=vae.path,
                        count=1,
                        height=ref_height,
                        width=ref_width,
                        cache_key=ref_key,
                        vae_cache_key=vae.cache_key,
                        precision=vae.precision,
                        quantize=vae.quantize,
                        edit_kind="flux3",
                        seq_len=int(packed.shape[1]),
                    )
                    positive_key = pipeline.prompt_encoding_key(clip, positive_text, reference.cache_key)
                    negative_key = pipeline.prompt_encoding_key(clip, negative_text, reference.cache_key)
                    pipeline.encode_text(entry, encoder_components, positive_text, cache, positive_key)
                    pipeline.encode_text(entry, encoder_components, negative_text, cache, negative_key)
                    positive = MlxConditioning(
                        clip=clip,
                        text=positive_text,
                        encoding_key=positive_key,
                        ref_cache_key=reference.cache_key,
                    )
                    negative = MlxConditioning(
                        clip=clip,
                        text=negative_text,
                        encoding_key=negative_key,
                        ref_cache_key=reference.cache_key,
                    )
                    params = {
                        "seed": int(seed) + index,
                        "steps": steps,
                        "height": int(height),
                        "width": int(width),
                        "batch_size": 1,
                        "positive_encoding_key": positive.encoding_key,
                        "negative_encoding_key": negative.encoding_key,
                        "guidance": 1.0,
                        "scheduler_name": SCHEDULER,
                        "compile_model": bool(model.compile),
                        "cache_limit_gb": int(model.compile_cache_limit),
                        "ref_cache_key": reference.cache_key,
                        "ref_count": reference.count,
                        "flux3_base": False,
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
                        result_image = self._resize_image(result_image, (output_width, output_height))
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
                    print(f"[MlxFlux3KleinFolderEdit] 处理失败，继续下一张：{source_path} | {exc}")
                finally:
                    _cleanup_flux3_image(cache, ref_key, positive_key, negative_key, latent)
                    progress.update()
        finally:
            sampler_components = None
            encoder_components = None
            _cleanup_flux3_resources(entry, model, clip, vae, cache)

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


__all__ = ["MlxFlux3KleinFolderEdit", "DEFAULT_PROMPT", "DEFAULT_NEGATIVE_PROMPT"]