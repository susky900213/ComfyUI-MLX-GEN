"""逐张执行 Qwen-Image 2.1 文件夹人物替换并保存结果。

这个节点与 ``MlxQwenImage21FolderEdit`` 保持独立。每张文件夹图片都是一
次单独的生成请求，并且 Qwen-Image 2.1 的两张视觉参考始终按下面的顺序
传递：

1. 当前文件夹图片（场景、构图和原人物位置）；
2. ``person_image``（要应用的人物参考）。

参考图顺序同时用于原生 VAE 条件和 Qwen3-VL 条件，不能把人物图放到第
一张，否则提示词里的视觉编号语义会发生变化。
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
    MlxModelHandle,
    MlxPilImage,
    MlxVaeHandle,
    entry_for,
    model,
    vae,
    validate_model_family,
)
from .directory_images import IMAGE_EXTENSIONS, _natural_sort_key, _safe_output_name


SCHEDULERS = ("flow_match_euler_discrete",)
KV_CACHE_MODES = ("auto", "off")
RESOLUTION_TIMINGS = ("采样时降分辨率", "先降分辨率", "后降分辨率")
UPLOAD_ROOT = "mlx_qwen_image_21"

DEFAULT_PROMPT = (
    "将参考图2中的人物身份、面部特征、发型、服装和身体比例应用到参考图1中的人物，"
    "保持参考图1的构图、镜头视角、动作姿态、人物数量和背景不变，保持自然真实的光影和细节。"
)
DEFAULT_NEGATIVE_PROMPT = (
    "模糊，低清晰度，人物身份错误，人物数量改变，脸部变形，发型错误，服装错误，"
    "肢体畸形，手指错误，背景改变，水印，黑边，裁切主体，乱码文字，不可读文字"
)


def _load_one_image(path: Path) -> Image.Image:
    """在文件句柄仍打开时复制图片，避免下游读取已关闭的文件。"""

    try:
        with Image.open(path) as source:
            return source.convert("RGBA").copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"无法读取图片：{path}（{exc}）") from exc


def _load_existing_output(path: Path) -> Image.Image | None:
    """读取已经生成的输出；不存在或损坏时返回 ``None``，让它重新生成。"""

    if not path.is_file():
        return None
    try:
        return _load_one_image(path)
    except ValueError as exc:
        print(f"[MlxQwenImage21FolderReplace] 已有输出无法读取，将重新生成：{path} | {exc}")
        return None


def _comfy_input_directory() -> Path:
    """返回 ComfyUI 的服务器端 input 目录。"""

    try:
        import folder_paths
    except ModuleNotFoundError:
        return Path.cwd() / "input"
    return Path(folder_paths.get_input_directory())


def _resolve_input_directory(folder_token: str) -> Path:
    """将 picker token 解析为 input/ 下的安全目录。"""

    token = str(folder_token).strip().replace("\\", "/")
    if not token:
        raise ValueError("请先点击「选择输入文件夹」并选择图片文件夹")
    relative = Path(token)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("输入文件夹必须是 ComfyUI input/ 下的相对目录，不能使用绝对路径或 ..")
    if not relative.parts or relative.parts[0] != UPLOAD_ROOT:
        raise ValueError(f"输入文件夹必须来自「选择输入文件夹」上传的 {UPLOAD_ROOT}/ 目录")

    root = _comfy_input_directory().expanduser().resolve()
    input_path = (root / relative).resolve()
    try:
        input_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("输入文件夹必须位于 ComfyUI input/ 目录内") from exc
    if not input_path.is_dir():
        raise ValueError(f"未找到已上传的输入文件夹：{input_path}")
    return input_path


def _image_paths(directory: str | Path) -> list[Path]:
    """枚举所选目录的直接图片，并明确排除 ``images/`` 输出目录。"""

    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ValueError(f"输入目录不存在或不是目录：{root}")
    output_directory = root / "images"
    return sorted(
        (
            path
            for path in root.iterdir()
            if path != output_directory
            and path.is_file()
            and path.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=_natural_sort_key,
    )


def _output_directory(input_directory: str | Path) -> Path:
    """创建并返回所选文件夹下的 ``images/`` 输出目录。"""

    input_path = Path(input_directory).expanduser().resolve()
    output = input_path / "images"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _evict_if_present(cache: Cache, bucket: str, key: str) -> None:
    if key:
        cache.evict(bucket, key)


class MlxQwenImage21FolderReplace:
    """将人物参考图逐张应用到文件夹中的 Qwen-Image 2.1 图片。"""

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
                "person_image": (
                    "IMAGE",
                    {"tooltip": "人物参考图；每张文件夹图片会作为参考图1，人物图作为参考图2"},
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
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0}),
                "scheduler": (list(SCHEDULERS), {"default": SCHEDULERS[0]}),
                "kv_cache": (list(KV_CACHE_MODES), {"default": "auto"}),
                "resolution_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "按每张源图片原始宽高乘以此系数",
                    },
                ),
                "resolution_timing": (
                    list(RESOLUTION_TIMINGS),
                    {
                        "default": RESOLUTION_TIMINGS[0],
                        "tooltip": "选择只缩小采样画布、先缩小源图，或在生成完成后缩小结果",
                    },
                ),
            }
        }

    RETURN_TYPES = ("images", "STRING", "STRING", "INT")
    RETURN_NAMES = ("images", "saved_files", "report", "success_count")
    FUNCTION = "replace_folder"
    CATEGORY = "MLX/Gen/Image"

    @staticmethod
    def _validate_handles(vae_handle, clip, model_handle) -> None:
        if not isinstance(vae_handle, MlxVaeHandle):
            raise ValueError("vae 必须连接「MLX VAE 加载」的输出")
        if not isinstance(clip, MlxClipHandle):
            raise ValueError("clip 必须连接「MLX CLIP 加载」的输出")
        if not isinstance(model_handle, MlxModelHandle):
            raise ValueError("model 必须连接「MLX 模型加载」的输出")
        for handle, label in ((vae_handle, "VAE"), (clip, "CLIP"), (model_handle, "模型")):
            if handle.model_type != "qwen_image_21":
                raise ValueError(f"{label} 必须选择 qwen_image_21，收到 {handle.model_type!r}")
        validate_model_family("qwen_image_21", vae_handle.path)
        validate_model_family("qwen_image_21", clip.path)
        validate_model_family("qwen_image_21", model_handle.model_path)

    @staticmethod
    def _condition_text(text: str) -> str:
        return text if text and text.strip() else " "

    @staticmethod
    def _scaled_dimensions(image: Image.Image, resolution_scale: float) -> tuple[int, int]:
        scale = float(resolution_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"分辨率系数必须是大于 0 的有限数字，收到 {resolution_scale!r}")
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        return width, height

    @staticmethod
    def _resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        if image.size == size:
            return image
        return image.resize(size, Image.Resampling.LANCZOS)

    @staticmethod
    def _reference_key(vae_handle, source_image: Image.Image, person_image: Image.Image) -> str:
        # The list order is part of the digest and therefore prevents accidentally
        # reusing a condition with the person image in the first visual slot.
        return pipeline.reference_image_cache_key(
            vae_handle,
            image_mod.digest([source_image, person_image]),
            {
                "count": 2,
                "resize_mode": "keep_resolution",
                "family": "qwen_image_21",
                "width": int(source_image.width),
                "height": int(source_image.height),
            },
        )

    def replace_folder(
        self,
        input_directory,
        person_image,
        vae,
        clip,
        model,
        prompt,
        negative_prompt,
        seed,
        steps,
        guidance,
        scheduler,
        kv_cache="auto",
        resolution_scale=1.0,
        resolution_timing=RESOLUTION_TIMINGS[0],
    ):
        self._validate_handles(vae, clip, model)
        if scheduler not in SCHEDULERS:
            raise ValueError(
                "Qwen-Image 2.1 文件夹人物替换只能使用 flow_match_euler_discrete，"
                f"收到 {scheduler!r}"
            )
        if kv_cache not in KV_CACHE_MODES:
            raise ValueError(f"未知 kv_cache 模式：{kv_cache!r}")
        if resolution_timing not in RESOLUTION_TIMINGS:
            raise ValueError(f"未知分辨率缩放时机：{resolution_timing!r}")

        steps = int(steps)
        guidance = float(guidance)
        resolution_scale = float(resolution_scale)
        if not math.isfinite(resolution_scale) or resolution_scale <= 0:
            raise ValueError(f"分辨率系数必须是大于 0 的有限数字，收到 {resolution_scale!r}")

        resolved_input_directory = _resolve_input_directory(str(input_directory))
        ordered_paths = _image_paths(resolved_input_directory)
        if not ordered_paths:
            extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
            raise ValueError(
                f"输入目录没有可读取的图片：{input_directory}（支持：{extensions}）"
            )

        total_images = len(ordered_paths)
        output = _output_directory(resolved_input_directory)
        progress = SamplingProgress(total_images)
        print(f"[MlxQwenImage21FolderReplace] 输出目录：{output}")
        print(f"[MlxQwenImage21FolderReplace] 开始处理：共 {total_images} 张图片")

        # 输出名由源图片的自然排序位置和文件名共同决定。只把能够被 PIL
        # 正常读取的预期输出视为已完成，避免空文件或损坏文件被永久跳过。
        generated: list[Image.Image | None] = [None] * total_images
        saved_paths: list[str | None] = [None] * total_images
        records: list[dict[str, object] | None] = [None] * total_images
        pending: list[tuple[int, Path, Path]] = []
        processed = 0
        for index, source_path in enumerate(ordered_paths):
            destination = output / _safe_output_name(source_path.name, index)
            existing = _load_existing_output(destination)
            if existing is None:
                pending.append((index, source_path, destination))
                continue
            existing_image = existing
            generated[index] = existing_image
            saved_paths[index] = str(destination)
            records[index] = {
                "index": index,
                "source": str(source_path),
                "output": str(destination),
                "status": "skipped",
                "reason": "已有生成图片",
                "width": existing_image.width,
                "height": existing_image.height,
            }
            processed += 1
            print(
                f"[MlxQwenImage21FolderReplace] [{processed}/{total_images}] 已存在，跳过生成："
                f"{destination}"
            )
            progress.update()

        if not pending:
            print(
                f"[MlxQwenImage21FolderReplace] 所有 {total_images} 张图片均已有有效输出，跳过生成"
            )
            return (
                MlxPilImage(images=tuple(image for image in generated if image is not None)),
                "\n".join(path for path in saved_paths if path is not None),
                json.dumps([record for record in records if record is not None], ensure_ascii=False, indent=2),
                total_images,
            )

        person_pils = image_mod.to_pil_batch(person_image, preserve_alpha=True)
        if not person_pils:
            raise ValueError("人物参考图为空")
        if len(person_pils) > 1:
            print("[MlxQwenImage21FolderReplace] 人物参考图包含多个 batch，只使用第一张")
        person_reference = person_pils[0]

        entry = entry_for(model.model_type)
        cache = Cache(
            caps={
                "module": 4,
                "prompt_encoding": 4,
                "ref_encoding": 2,
                "component_weights": 2,
            }
        )
        positive_text = self._condition_text(prompt)
        negative_text = self._condition_text(negative_prompt)
        sampler_components = None
        encoder_components = None
        encoder_key = runtime.cache_key({"kind": "module", "clip": clip})

        runtime.apply_cache_limit(model.compile_cache_limit, label=entry.family)
        try:
            sampler_components = pipeline.prepare_sampler_components(entry, model, cache)
            encoder_components = pipeline.prepare_encoder(entry, clip, cache, encoder_key)
            for index, source_path, destination in pending:
                ref_key = ""
                positive_key = ""
                negative_key = ""
                latent = None
                try:
                    source_image = _load_one_image(source_path)
                    output_width, output_height = self._scaled_dimensions(
                        source_image, resolution_scale
                    )
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
                        width, height = source_image.size
                    pipeline.qwen21_sampling.validate_dimensions(height, width)

                    # Both encoders receive the same ordered list. Do not replace
                    # this with a set or a batch whose order is implementation-defined.
                    ref_key = self._reference_key(vae, generation_image, person_reference)
                    vae_module = pipeline.vae_component(vae, cache)
                    try:
                        pipeline.encode_qwen21_reference_images(
                            vae_module,
                            [generation_image, person_reference],
                            cache,
                            ref_key,
                            2,
                            width,
                            height,
                            "keep_resolution",
                        )
                    finally:
                        vae_module = None
                        pipeline.release_vae_component(vae, cache)

                    reference = pipeline.cached_qwen21_reference(cache, ref_key)
                    positive_key = pipeline.prompt_encoding_key(clip, positive_text, ref_key)
                    negative_key = pipeline.prompt_encoding_key(clip, negative_text, ref_key)
                    pipeline.encode_text(
                        entry,
                        encoder_components,
                        positive_text,
                        cache,
                        positive_key,
                        reference_images=reference["images"],
                        max_length=clip.max_length,
                    )
                    pipeline.encode_text(
                        entry,
                        encoder_components,
                        negative_text,
                        cache,
                        negative_key,
                        reference_images=reference["images"],
                        max_length=clip.max_length,
                    )

                    params = {
                        "seed": int(seed) + index,
                        "steps": steps,
                        "height": height,
                        "width": width,
                        "batch_size": 1,
                        "positive_encoding_key": positive_key,
                        "negative_encoding_key": negative_key,
                        "guidance": guidance,
                        "scheduler_name": scheduler,
                        "compile_model": bool(model.compile) and entry.supports_compile,
                        "cache_limit_gb": int(model.compile_cache_limit),
                        "ref_cache_key": ref_key,
                        "ref_count": 2,
                        "kv_cache": kv_cache,
                    }
                    latent = pipeline.run_sampler(entry, model, sampler_components, params, cache)

                    cached_latent, hit = cache.get("component_weights", latent.cache_key)
                    if not hit:
                        raise RuntimeError("当前图片的 latent 已失效，请重新运行文件夹人物替换节点")
                    vae_module = pipeline.vae_component(vae, cache)
                    try:
                        decoded = pipeline.decode_latents(
                            entry,
                            {"vae": vae_module},
                            cached_latent,
                            latent.height,
                            latent.width,
                        )
                        decoded_images = image_mod.to_pil(decoded, batch_index=-1)
                    finally:
                        vae_module = None
                        pipeline.release_vae_component(vae, cache)
                    if len(decoded_images) != 1:
                        raise RuntimeError(f"单张处理应得到 1 张输出，实际得到 {len(decoded_images)} 张")

                    result_image = decoded_images[0]
                    if resolution_timing == RESOLUTION_TIMINGS[2]:
                        result_image = self._resize_image(
                            result_image, (output_width, output_height)
                        )

                    result_image.save(destination, format="PNG")
                    if not destination.is_file():
                        raise RuntimeError(f"图片保存后未找到输出文件：{destination}")
                    generated[index] = result_image
                    saved_paths[index] = str(destination)
                    records[index] = {
                        "index": index,
                        "source": str(source_path),
                        "output": str(destination),
                        "status": "success",
                        "seed": int(seed) + index,
                        "width": output_width,
                        "height": output_height,
                        "resolution_scale": resolution_scale,
                        "resolution_timing": resolution_timing,
                        "reference_order": ["source", "person"],
                    }
                    print(
                        f"[MlxQwenImage21FolderReplace] [{index + 1}/{total_images}] 已保存：{destination}"
                    )
                except Exception as exc:  # noqa: BLE001
                    records[index] = {
                        "index": index,
                        "source": str(source_path),
                        "output": str(destination),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(
                        f"[MlxQwenImage21FolderReplace] [{index + 1}/{total_images}] 处理失败，继续下一张："
                        f"{source_path} | {exc}"
                    )
                finally:
                    try:
                        _evict_if_present(cache, "ref_encoding", ref_key)
                        _evict_if_present(cache, "prompt_encoding", positive_key)
                        _evict_if_present(cache, "prompt_encoding", negative_key)
                        if latent is not None:
                            _evict_if_present(cache, "component_weights", latent.cache_key)
                        runtime.flush_caches()
                    finally:
                        progress.update()
                        processed += 1
                        available = sum(image is not None for image in generated)
                        failed = processed - available
                        print(
                            f"[MlxQwenImage21FolderReplace] 处理进度：{processed}/{total_images} "
                            f"（成功 {available}，失败 {failed}）"
                        )
        finally:
            pipeline.clear_compiled_predict_caches()
            pipeline.release_sampler_components(entry, model, cache)
            pipeline.release_encoder(clip, cache)

        success_count = sum(image is not None for image in generated)
        if not success_count:
            report = json.dumps([record for record in records if record is not None], ensure_ascii=False, indent=2)
            raise RuntimeError(f"文件夹中没有图片处理成功：\n{report}")
        return (
            MlxPilImage(images=tuple(image for image in generated if image is not None)),
            "\n".join(path for path in saved_paths if path is not None),
            json.dumps([record for record in records if record is not None], ensure_ascii=False, indent=2),
            success_count,
        )


__all__ = [
    "MlxQwenImage21FolderReplace",
    "DEFAULT_PROMPT",
    "DEFAULT_NEGATIVE_PROMPT",
]