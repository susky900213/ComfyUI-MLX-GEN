"""MlxTextEncoder：MLX 文本编码器（对应 ComfyUI 原生 CLIPTextEncode 节点）。

输入与 ComfyUI 的 CLIPTextEncode 一样：clip（MlxClipLoader 产出的纯数据
handle）+ text（一条提示词）；输出 condition。正、负条件各用一个本
节点（互不干扰），再分别接到 MlxKSamplerMLX 的 positive / negative 入口。

与 ComfyUI 的区别：handle 里只有「大类 + 组件 + 权重目录 + 精度」这类配置，
真正的 text_encoder / tokenizer 由本节点按需创建（存进 cache.py），
编码结果也只把缓存键写进 MlxConditioning —— 数组留在 cache.py，避免 ComfyUI
的节点输出缓存长期持有大对象。
"""

from __future__ import annotations

from .. import pipeline, runtime
from ..cache import CACHE
from ..types import (
    CLIP,
    condition,
    entry_for,
    MlxClipHandle,
    MlxConditioning,
    MlxH3VisualCondition,
    MlxReferenceImages,
    h3_keyframes as h3_keyframes_type,
    ref_images as ref_images_type,
    validate_model_family,
)


class MlxTextEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {"default": "a red cube", "multiline": True, "dynamicPrompts": True},
                ),
                "clip": (CLIP, {}),
            },
            "optional": {
                "h3_keyframes": (h3_keyframes_type, {}),
                "ref_images": (ref_images_type, {}),
                # 放在既有可选 socket 后面，保持历史 H3 / Qwen-Image 2.1
                # 工作流的输入 slot 顺序不变。
                "prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "可连接 Qwen-Image 2.1 Prompt Enhancer；连接后覆盖上方文本",
                    },
                ),
            },
        }

    RETURN_TYPES = (condition,)
    FUNCTION = "encode"
    CATEGORY = "MLX/Gen"

    def encode(self, text, clip, h3_keyframes=None, ref_images=None, prompt=None):
        if clip is None:
            raise ValueError("必须先连接 MlxClipLoader 的输出")
        if not isinstance(clip, MlxClipHandle):
            raise ValueError(
                "clip 必须是 MlxClipLoader 产出的 handle；"
                "ComfyUI 原生 CLIP（torch 权重）不能被 MLX 模型使用"
            )
        # prompt 是可选外部 STRING socket，优先级高于旧的 widget 文本。
        # 这样 PE 节点只会改变正向提示词，不会触碰 h3/ref_images 条件。
        if prompt is not None:
            text = prompt
        # 配置由 handle 里的大类决定；未验证的大类直接报错（不静默回退）
        entry = entry_for(clip.model_type)
        validate_model_family(clip.model_type, clip.path)
        if not entry.supported:
            raise NotImplementedError(f"{clip.model_type} 尚未实现：{entry.notes}")
        if h3_keyframes is not None and not isinstance(h3_keyframes, MlxH3VisualCondition):
            raise ValueError("h3_keyframes 必须连接「MLX H3 关键帧 / 多图参考 / 视频条件」的输出")
        if h3_keyframes is not None and entry.family != "minimax_h3":
            raise ValueError(f"h3_keyframes 只支持 minimax_h3，当前条件大类是 {entry.family}")
        if ref_images is not None and not isinstance(ref_images, MlxReferenceImages):
            raise ValueError("ref_images 必须连接「MLX VAE 编码」的输出")
        if ref_images is not None and entry.family != "qwen_image_21":
            raise ValueError(
                f"MlxTextEncoder 的 ref_images 只供 qwen_image_21 原生编辑使用，当前是 {entry.family}"
            )
        if ref_images is not None and (
            ref_images.model_type != clip.model_type or ref_images.edit_kind != "qwen_image_21"
        ):
            raise ValueError("参考图不是由当前 qwen_image_21 VAE 原生编码的")
        if entry.family == "qwen_edit":
            raise ValueError(
                "Qwen-Image-Edit 的条件必须带参考图，请用「MLX Qwen 编辑条件」"
                "（MlxQwenEditEncoder）节点，而不是本节点 —— 否则会得到一条"
                "没有任何视觉信息的条件"
            )
        # 音频生成家族的文本编码器都封装在主 checkpoint 内：YuE2 的正/负文本分别
        # 是 style / lyrics；Breeze 的正向文本是目标台词，负向文本只为保持标准连线。
        # 这里都只透传原文，真正编码发生在采样器物化完整模型之后。
        if entry.media == "audio":
            prompt = text or ""
            cond = MlxConditioning(
                clip=clip,
                text=prompt,
                encoding_key=runtime.cache_key(
                    {"kind": f"{entry.family}_text", "clip": clip, "text": prompt}
                ),
            )
            return (cond,)

        # 空提示词按空格编码（与 mflux 对 negative_prompt 的处理一致）
        prompt = text if text and text.strip() else " "

        # Ideogram 4 的 position ids / image token 数取决于目标宽高，而文本节点此时
        # 还不知道采样尺寸。因此这里只传递原始 caption；真正的规范化、校验和编码由
        # 采样器在拿到 width/height 后完成。负向节点保持标准连线，但其文本不会被模型使用。
        if entry.family == "ideogram4":
            return (MlxConditioning(clip=clip, text=text or "", encoding_key=""),)

        # 视频 / 音频家族（MiniMax-H3）：先组装成 H3 的三段式 presentation，再用
        # Qwen3-VL 编码；条件编码器不量化要常驻约 50 GB，所以必须 q8 / q4
        if entry.media == "video":
            if int(clip.quantize or 0) not in (4, 8):
                raise ValueError(
                    f"{entry.family} 的条件编码器必须量化（在「MLX 条件加载器」里把 "
                    f"quantize 改成 8 或 4；当前是 {clip.quantize}）"
                )
            composed = pipeline.compose_h3_prompt(prompt)
            visual_vae = h3_keyframes.vae if h3_keyframes is not None else None
            if visual_vae is not None and (
                visual_vae.model_type != clip.model_type or visual_vae.role != "vae"
            ):
                raise ValueError(
                    "H3 视觉条件必须使用当前文本条件大类的视频 VAE（role=vae）："
                    f"条件是 model_type={visual_vae.model_type!r}, "
                    f"role={visual_vae.role!r}，文本条件是 {clip.model_type!r}"
                )
            encoding_key = pipeline.h3_prompt_encoding_key(clip, composed, h3_keyframes)
            try:
                comps = pipeline.prepare_h3_encoder(
                    entry, clip, CACHE, runtime.cache_key({"kind": "module", "clip": clip})
                )
                embeds, tags = pipeline.encode_h3_prompt(
                    entry, comps, composed, CACHE, encoding_key, h3_keyframes
                )
            finally:
                # 条件编码器约 30 GB；成功或异常都必须立即释放。
                comps = None
                pipeline.release_h3_encoder(entry, clip, CACHE)
            print(
                f"[MlxTextEncoder] {entry.family}: {int(tags.shape[0])} 个 token，"
                f"条件张量 {tuple(embeds.shape)} {embeds.dtype}"
            )
            cond = MlxConditioning(
                clip=clip,
                text=composed,
                encoding_key=encoding_key,
                h3_keyframes=h3_keyframes,
            )
            return (cond,)

        ref_key = ref_images.cache_key if ref_images is not None else ""
        reference_pils = (
            pipeline.cached_qwen21_reference(CACHE, ref_key)["images"] if ref_key else ()
        )

        # 1) 按需创建 text_encoder + tokenizer（同一 handle 第二次执行直接命中缓存）
        # 参考缓存先读取：若已经淘汰，应在加载 8B 条件编码器之前快速失败。
        comps_key = runtime.cache_key({"kind": "module", "clip": clip})
        try:
            comps = pipeline.prepare_encoder(entry, clip, CACHE, comps_key)
            # 2) 编码本条提示词；数组只进缓存，handle 里只留键
            encoding_key = pipeline.prompt_encoding_key(clip, prompt, ref_key)
            pipeline.encode_text(
                entry,
                comps,
                prompt,
                CACHE,
                encoding_key,
                reference_images=reference_pils,
                max_length=clip.max_length,
            )
        finally:
            # 编码结果已经独立放进 prompt_encoding；条件节点不应把 text encoder
            # 权重留到采样器，更不能在编码异常时留下半加载 bundle。
            comps = None
            pipeline.release_encoder(clip, CACHE)

        # 3) 交给 MlxKSamplerMLX 的 positive / negative 入口
        cond = MlxConditioning(
            clip=clip,
            text=prompt,
            encoding_key=encoding_key,
            ref_cache_key=ref_key,
        )
        return (cond,)

