"""逐张执行 Qwen-Image 2.1 文件夹图生图并保存结果。

这个节点和 ``MlxQwenImage21StoryboardBatchEdit`` 是两条独立链路：

* 不把整个目录加载成一个 ComfyUI IMAGE batch；
* 每次只把一张图片作为 Qwen-Image 2.1 的单个视觉参考；
* 每张图片使用 ``batch_size=1``、``ref_count=1`` 独立采样；
* 当前图片解码并写入输出目录后，才开始下一张图片。
* 输出目录中已经存在且可读取的对应 PNG 会直接跳过，支持中断后继续处理。

节点内部复用已有 pipeline API，但不修改旧节点或公共采样器的行为。
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
from .directory_images import IMAGE_EXTENSIONS, _image_paths, _natural_sort_key, _safe_output_name


SCHEDULERS = ("flow_match_euler_discrete",)
KV_CACHE_MODES = ("auto", "off")
# 保留三种缩放时机：默认模式只缩小采样画布；“先降”还会缩小传入的参考图；
# “后降”则在解码后缩小最终输出。
RESOLUTION_TIMINGS = ("采样时降分辨率", "先降分辨率", "后降分辨率")
UPLOAD_ROOT = "mlx_qwen_image_21"

DEFAULT_PROMPT = (
    "真实世界风格。人物的面部、发型、服装和身体比例保持原样，保持原图的构图和镜头视角，"
    "不要改变人物身份，不要改变人物数量，不要改变主体的动作和姿态，不要改变原图背景。"
    "如果图片中有任何文字、招牌、字幕、标签、包装文字或对话框，请识别其原意，"
    "将文字全部替换为准确、自然、清晰、可读的简体中文，并保持原文字的位置、字号层级、"
    "字体风格、颜色、方向和排版布局；没有文字时不要额外添加文字。"
)
DEFAULT_NEGATIVE_PROMPT = (
    "模糊，低清晰度，人物身份改变，人物数量改变，脸部变形，肢体畸形，"
    "英文文字，日文文字，韩文文字，乱码文字，不可读文字，错误翻译，水印，黑边，裁切主体"
)


def _load_one_image(path: Path) -> Image.Image:
    """在文件句柄仍打开时复制一张图片，避免延迟读取已关闭的文件。"""

    try:
        with Image.open(path) as source:
            return source.convert("RGBA").copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"无法读取图片：{path}（{exc}）") from exc


def _load_existing_output(path: Path) -> Image.Image | None:
    """Return a readable existing output image, or ``None`` when it is absent/broken.

    A partially written or corrupted PNG must not count as a completed item;
    returning ``None`` makes the normal generation path replace it.
    """

    if not path.is_file():
        return None
    try:
        with Image.open(path) as existing:
            existing.verify()
        return _load_one_image(path)
    except (OSError, UnidentifiedImageError, ValueError):
        return None


def _comfy_input_directory() -> Path:
    """Return ComfyUI's server-side input directory.

    The browser cannot expose a local absolute path to a node.  The folder
    picker uploads the selected images below this directory and passes only a
    relative folder token to the node.
    """

    try:
        import folder_paths
    except ModuleNotFoundError:
        return Path.cwd() / "input"
    return Path(folder_paths.get_input_directory())


def _resolve_input_directory(folder_token: str) -> Path:
    """Resolve a folder-picker token below ComfyUI's ``input/`` directory."""

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


def _output_directory(input_directory: str | Path) -> Path:
    """Create and return ``image/`` below the selected input folder.

    The folder picker uploads the user's selected folder to ComfyUI's input
    directory.  Generated images intentionally stay beside those uploaded
    source images instead of going to ComfyUI's global ``output/`` directory.
    """

    input_path = Path(input_directory).expanduser().resolve()
    output = input_path / "image"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _evict_if_present(cache: Cache, bucket: str, key: str) -> None:
    if key:
        cache.evict(bucket, key)


class MlxQwenImage21FolderEdit:
    """按自然排序逐张处理文件夹图片的 Qwen-Image 2.1 图生图节点。"""

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
        """Return the integer canvas obtained by scaling an input image.

        Qwen-Image 2.1 requires each public dimension to be at least 32 pixels.
        Keep the requested aspect ratio and round each scaled side independently;
        the sampler itself will handle its internal 32-pixel latent alignment.
        """

        scale = float(resolution_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"分辨率系数必须是大于 0 的有限数字，收到 {resolution_scale!r}")
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        return width, height

    @staticmethod
    def _resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        """Resize a PIL image with the same high-quality filter in both modes."""

        if image.size == size:
            return image
        return image.resize(size, Image.Resampling.LANCZOS)

    @staticmethod
    def _reference_key(vae_handle, image: Image.Image) -> str:
        return pipeline.reference_image_cache_key(
            vae_handle,
            image_mod.digest([image]),
            {
                "count": 1,
                "resize_mode": "keep_resolution",
                "family": "qwen_image_21",
                "width": int(image.width),
                "height": int(image.height),
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
        guidance,
        scheduler,
        kv_cache="auto",
        resolution_scale=1.0,
        resolution_timing=RESOLUTION_TIMINGS[0],
    ):
        self._validate_handles(vae, clip, model)
        if scheduler not in SCHEDULERS:
            raise ValueError(
                "Qwen-Image 2.1 文件夹逐图处理只能使用 flow_match_euler_discrete，"
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
        paths = _image_paths(str(resolved_input_directory))
        if not paths:
            # _image_paths 目前不会返回空列表，但保留这个守卫，避免未来筛选逻辑变化
            # 后产生一个没有任何处理记录的“成功”节点。
            extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
            raise ValueError(f"输入目录没有可读取的图片：{input_directory}（支持：{extensions}）")
        ordered_paths = sorted(paths, key=_natural_sort_key)
        total_images = len(ordered_paths)
        output = _output_directory(resolved_input_directory)
        progress = SamplingProgress(total_images)
        print(f"[MlxQwenImage21FolderEdit] 输出目录：{output}")
        print(f"[MlxQwenImage21FolderEdit] 开始处理：共 {total_images} 张图片")

        # 用自然排序后的原始 index 生成稳定的输出名；即使中途失败或跳过，
        # 后续重跑也必须继续使用同一个名字，不能因为待处理列表变短而改名。
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

            generated[index] = existing
            saved_paths[index] = str(destination)
            records[index] = {
                "index": index,
                "source": str(source_path),
                "output": str(destination),
                "status": "skipped",
                "reason": "输出文件已存在且可读取",
            }
            processed += 1
            progress.update()
            print(
                f"[MlxQwenImage21FolderEdit] [{processed}/{total_images}] 已存在，跳过：{destination}"
            )

        if not pending:
            # 全部结果都已存在时不加载模型；这也是“断点续跑”最省资源的路径。
            report_records = [record for record in records if record is not None]
            print(
                f"[MlxQwenImage21FolderEdit] 已有 {total_images} 个可用输出，跳过全部图片"
            )
            return (
                MlxPilImage(images=tuple(image for image in generated if image is not None)),
                "\n".join(path for path in saved_paths if path is not None),
                json.dumps(report_records, ensure_ascii=False, indent=2),
                total_images,
            )

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

        # Transformer 和 text_encoder/tokenizer 在整个目录循环中复用；参考图、文本
        # 编码结果和 latent 仍按图片使用独立缓存键，并在该图片保存后主动淘汰，
        # 绝不会组成一个 batch。
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
                    output_width, output_height = self._scaled_dimensions(source_image, resolution_scale)
                    # 默认“采样时降”只缩小采样器画布，参考图仍保留原始尺寸；“先降”
                    # 则在参考图编码前先缩小图片，并用缩小后的图片和画布采样；“后降”
                    # 保持原尺寸生成，最后再缩小解码结果。
                    if resolution_timing == RESOLUTION_TIMINGS[0]:
                        generation_image = source_image
                        width, height = output_width, output_height
                    elif resolution_timing == RESOLUTION_TIMINGS[1]:
                        generation_image = self._resize_image(
                            source_image,
                            (output_width, output_height),
                        )
                        width, height = output_width, output_height
                    else:
                        generation_image = source_image
                        width, height = source_image.size
                    reference_width, reference_height = source_image.size
                    if resolution_timing == RESOLUTION_TIMINGS[1]:
                        reference_width, reference_height = generation_image.size
                    pipeline.qwen21_sampling.validate_dimensions(height, width)

                    # 1) 当前图片单独进行 Qwen-Image 2.1 原生参考图编码。
                    vae_module = pipeline.vae_component(vae, cache)
                    try:
                        ref_key = self._reference_key(vae, generation_image)
                        pipeline.encode_qwen21_reference_images(
                            vae_module,
                            [generation_image],
                            cache,
                            ref_key,
                            1,
                            reference_width,
                            reference_height,
                            "keep_resolution",
                        )
                    finally:
                        vae_module = None
                        pipeline.release_vae_component(vae, cache)

                    reference = pipeline.cached_qwen21_reference(cache, ref_key)

                    # 2) 当前图片自己的正/负条件编码；reference_images 永远只有一张。
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

                    # 3) 只采当前图片：batch_size=1、ref_count=1。
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
                        "ref_count": 1,
                        "kv_cache": kv_cache,
                    }
                    latent = pipeline.run_sampler(entry, model, sampler_components, params, cache)

                    # 4) 当前 latent 单独解码，然后立即保存；不等整个目录采样完成。
                    cached_latent, hit = cache.get("component_weights", latent.cache_key)
                    if not hit:
                        raise RuntimeError("当前图片的 latent 已失效，请重新运行文件夹逐图节点")
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
                        result_image = self._resize_image(result_image, (output_width, output_height))

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
                    }
                    print(f"[MlxQwenImage21FolderEdit] 已保存：{destination}")
                except Exception as exc:  # noqa: BLE001
                    records[index] = {
                        "index": index,
                        "source": str(source_path),
                        "output": str(destination),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(
                        f"[MlxQwenImage21FolderEdit] 处理失败，继续下一张："
                        f"{source_path} | {exc}"
                    )
                finally:
                    try:
                        # 当前图片处理完后释放它的参考条件、文本条件和 latent；输出 PIL
                        # 与已经写入磁盘的 PNG 不依赖这些缓存。
                        _evict_if_present(cache, "ref_encoding", ref_key)
                        _evict_if_present(cache, "prompt_encoding", positive_key)
                        _evict_if_present(cache, "prompt_encoding", negative_key)
                        if latent is not None:
                            _evict_if_present(cache, "component_weights", latent.cache_key)
                        runtime.flush_caches()
                    finally:
                        # 无论当前图片成功还是失败，都算作已处理，避免进度条停在
                        # 失败图片上；ComfyUI 中此调用会同步更新节点进度条。
                        progress.update()
                        processed += 1
                        available = sum(image is not None for image in generated)
                        failed = processed - available
                        print(
                            f"[MlxQwenImage21FolderEdit] 处理进度：{processed}/{total_images} "
                            f"（成功/跳过 {available}，失败 {failed}）"
                        )
        finally:
            pipeline.clear_compiled_predict_caches()
            pipeline.release_sampler_components(entry, model, cache)
            pipeline.release_encoder(clip, cache)

        available_images = [image for image in generated if image is not None]
        available_paths = [path for path in saved_paths if path is not None]
        report_records = [record for record in records if record is not None]
        success_count = len(available_images)
        if not success_count:
            report = json.dumps(report_records, ensure_ascii=False, indent=2)
            raise RuntimeError(f"文件夹中没有图片处理成功：\n{report}")
        return (
            MlxPilImage(images=tuple(available_images)),
            "\n".join(available_paths),
            json.dumps(report_records, ensure_ascii=False, indent=2),
            success_count,
        )


__all__ = ["MlxQwenImage21FolderEdit", "DEFAULT_PROMPT", "DEFAULT_NEGATIVE_PROMPT"]