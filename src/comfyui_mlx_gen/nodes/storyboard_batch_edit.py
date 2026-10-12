"""Generate one Qwen-Image 2.1 result for every detected storyboard panel.

The regular Qwen-Image 2.1 edit graph intentionally treats an IMAGE batch as
multiple references for *one* generation.  That is useful for multi-image
editing, but it is not a loop.  This node is the explicit loop for storyboard
workflows: every panel gets its own one-image reference condition, seed, Qwen
sampling call, and VAE decode.  The decoded PIL images are returned as one
``images`` payload so ComfyUI can preview/save them as a batch.
"""

from __future__ import annotations

from .. import image as image_mod
from .. import pipeline, runtime
from ..cache import Cache
from ..types import (
    CLIP,
    MlxClipHandle,
    MlxModelHandle,
    MlxVaeHandle,
    entry_for,
    model,
    vae,
    validate_model_family,
)


SCHEDULERS = ("flow_match_euler_discrete",)
KV_CACHE_MODES = ("auto", "off")
MAX_PANELS = 10


class MlxQwenImage21StoryboardBatchEdit:
    """对拆出的每个分镜单独执行一次 Qwen-Image 2.1 编辑。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panels": ("IMAGE", {}),
                "vae": (vae, {}),
                "clip": (CLIP, {}),
                "model": (model, {}),
                "prompt": (
                    "STRING",
                    {
                        "default": "保持这个分镜的角色、画风和构图，生成高质量的漫画分镜画面",
                        "multiline": True,
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "模糊，低清晰度，人物身份改变，肢体畸形，乱码文字，水印，黑边",
                        "multiline": True,
                    },
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
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "width": ("INT", {"default": 768, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 4096, "step": 32}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0}),
                "scheduler": (list(SCHEDULERS), {"default": SCHEDULERS[0]}),
                "kv_cache": (list(KV_CACHE_MODES), {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("images",)
    RETURN_NAMES = ("images",)
    FUNCTION = "edit"
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
        validate_model_family("qwen_image_21", model_handle.model_path)
        validate_model_family("qwen_image_21", vae_handle.path)
        validate_model_family("qwen_image_21", clip.path)

    @staticmethod
    def _condition_text(text: str) -> str:
        # Match MlxTextEncoder's empty-prompt behaviour and keep cache keys
        # identical to the normal Qwen-Image 2.1 text node.
        return text if text and text.strip() else " "

    def edit(
        self,
        panels,
        vae,
        clip,
        model,
        prompt,
        negative_prompt,
        seed,
        steps,
        width,
        height,
        guidance,
        scheduler,
        kv_cache="auto",
    ):
        self._validate_handles(vae, clip, model)
        if scheduler not in SCHEDULERS:
            raise ValueError(
                "Qwen-Image 2.1 分镜批处理只能使用 flow_match_euler_discrete，"
                f"收到 {scheduler!r}"
            )
        if kv_cache not in KV_CACHE_MODES:
            raise ValueError(f"未知 kv_cache 模式：{kv_cache!r}")

        width, height = int(width), int(height)
        steps = int(steps)
        guidance = float(guidance)
        qwen21_sampling = pipeline.qwen21_sampling
        qwen21_sampling.validate_dimensions(height, width)

        source_panels = image_mod.to_pil_batch(panels, preserve_alpha=True)
        if not source_panels:
            raise ValueError("没有可生成的分镜面板")
        if len(source_panels) > MAX_PANELS:
            raise ValueError(
                f"分镜批次包含 {len(source_panels)} 张图；最多支持 {MAX_PANELS} 张，"
                "请把 MlxComicStoryboardSplitter.max_panels 设为 10"
            )

        entry = entry_for(model.model_type)
        # The larger temporary limits keep all per-panel reference/condition
        # keys alive while the transformer samples them one by one.  They are
        # local to this node execution and disappear with this Cache object.
        batch_cache = Cache(
            caps={
                "module": 4,
                "prompt_encoding": MAX_PANELS * 2 + 2,
                "ref_encoding": MAX_PANELS + 2,
                "component_weights": MAX_PANELS + 2,
            }
        )
        reference_keys: list[str] = []
        positive_keys: list[str] = []
        negative_keys: list[str] = []
        latents = []
        positive_text = self._condition_text(prompt)
        negative_text = self._condition_text(negative_prompt)

        # Stage 1: encode every panel as a one-image Qwen native reference.
        # No panel is ever passed together with another panel here.
        try:
            vae_module = pipeline.vae_component(vae, batch_cache)
            for panel in source_panels:
                digest = image_mod.digest([panel])
                ref_key = pipeline.reference_image_cache_key(
                    vae,
                    digest,
                    {
                        "count": 1,
                        "resize_mode": "auto",
                        "family": "qwen_image_21",
                        "width": width,
                        "height": height,
                    },
                )
                pipeline.encode_qwen21_reference_images(
                    vae_module,
                    [panel],
                    batch_cache,
                    ref_key,
                    1,
                    width,
                    height,
                    "auto",
                )
                reference_keys.append(ref_key)
        finally:
            pipeline.release_vae_component(vae, batch_cache)

        # Stage 2: encode positive/negative text against each panel's own
        # reference images.  The visual tower therefore sees one picture per
        # condition, never the complete storyboard batch.
        encoder_key = runtime.cache_key({"kind": "module", "clip": clip})
        try:
            encoder_components = pipeline.prepare_encoder(entry, clip, batch_cache, encoder_key)
            for ref_key in reference_keys:
                reference = pipeline.cached_qwen21_reference(batch_cache, ref_key)
                positive_key = pipeline.prompt_encoding_key(clip, positive_text, ref_key)
                negative_key = pipeline.prompt_encoding_key(clip, negative_text, ref_key)
                pipeline.encode_text(
                    entry,
                    encoder_components,
                    positive_text,
                    batch_cache,
                    positive_key,
                    reference_images=reference["images"],
                    max_length=clip.max_length,
                )
                pipeline.encode_text(
                    entry,
                    encoder_components,
                    negative_text,
                    batch_cache,
                    negative_key,
                    reference_images=reference["images"],
                    max_length=clip.max_length,
                )
                positive_keys.append(positive_key)
                negative_keys.append(negative_key)
        finally:
            pipeline.release_encoder(clip, batch_cache)

        # Stage 3: one independent sampler request per panel.  The only
        # difference between adjacent requests is the panel reference/cache
        # key and the deterministic per-panel seed.
        runtime.apply_cache_limit(model.compile_cache_limit, label=entry.family)
        sampler_components = None
        try:
            sampler_components = pipeline.prepare_sampler_components(entry, model, batch_cache)
            for index, ref_key in enumerate(reference_keys):
                params = {
                    "seed": int(seed) + index,
                    "steps": steps,
                    "height": height,
                    "width": width,
                    "batch_size": 1,
                    "positive_encoding_key": positive_keys[index],
                    "negative_encoding_key": negative_keys[index],
                    "guidance": guidance,
                    "scheduler_name": scheduler,
                    "compile_model": bool(model.compile) and entry.supports_compile,
                    "cache_limit_gb": int(model.compile_cache_limit),
                    "ref_cache_key": ref_key,
                    "ref_count": 1,
                    "kv_cache": kv_cache,
                }
                latents.append(
                    pipeline.run_sampler(entry, model, sampler_components, params, batch_cache)
                )
        finally:
            pipeline.clear_compiled_predict_caches()
            pipeline.release_sampler_components(entry, model, batch_cache)

        # Stage 4: decode the independent latent results into one output batch.
        decoded_images = []
        try:
            vae_module = pipeline.vae_component(vae, batch_cache)
            for latent in latents:
                cached, hit = batch_cache.get("component_weights", latent.cache_key)
                if not hit:
                    raise RuntimeError("分镜 latent 已失效，请重新运行批量编辑节点")
                decoded = pipeline.decode_latents(
                    entry,
                    {"vae": vae_module},
                    cached,
                    latent.height,
                    latent.width,
                )
                decoded_images.extend(image_mod.to_pil(decoded, batch_index=-1))
        finally:
            pipeline.release_vae_component(vae, batch_cache)

        if len(decoded_images) != len(source_panels):
            raise RuntimeError(
                f"分镜批处理结果数量不一致：输入 {len(source_panels)}，输出 {len(decoded_images)}"
            )
        print(
            f"[MlxQwenImage21StoryboardBatchEdit] 已独立生成 {len(decoded_images)} 张分镜图片"
        )
        from ..types import MlxPilImage

        return (MlxPilImage(images=tuple(decoded_images)),)
