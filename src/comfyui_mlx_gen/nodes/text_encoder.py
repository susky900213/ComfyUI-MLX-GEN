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
            }
        }

    RETURN_TYPES = (condition,)
    FUNCTION = "encode"
    CATEGORY = "MLX/Gen"

    def encode(self, text, clip):
        if clip is None:
            raise ValueError("必须先连接 MlxClipLoader 的输出")
        if not isinstance(clip, MlxClipHandle):
            raise ValueError(
                "clip 必须是 MlxClipLoader 产出的 handle；"
                "ComfyUI 原生 CLIP（torch 权重）不能被 MLX 模型使用"
            )
        # 配置由 handle 里的大类决定；未验证的大类直接报错（不静默回退）
        entry = entry_for(clip.model_type)
        if not entry.supported:
            raise NotImplementedError(f"{clip.model_type} 尚未实现：{entry.notes}")
        if entry.family == "qwen_edit":
            raise ValueError(
                "Qwen-Image-Edit 的条件必须带参考图，请用「MLX Qwen 编辑条件」"
                "（MlxQwenEditEncoder）节点，而不是本节点 —— 否则会得到一条"
                "没有任何视觉信息的条件"
            )
        # 空提示词按空格编码（与 mflux 对 negative_prompt 的处理一致）
        prompt = text if text and text.strip() else " "

        # Ideogram 4 的 position ids / image token 数取决于目标宽高，而文本节点此时
        # 还不知道采样尺寸。因此这里只传递原始 caption；真正的规范化、校验和编码由
        # 采样器在拿到 width/height 后完成。负向节点保持标准连线，但其文本不会被模型使用。
        if entry.family == "ideogram4":
            return (MlxConditioning(clip=clip, text=text or "", encoding_key=""),)

        # 视频 / 音频家族（MiniMax-H3）：先组装成 H3 的三段式 presentation，再用
        # Qwen3-VL 编码；条件编码器不量化要常驻约 50 GB，所以必须 q8 / q4
        if entry.media != "image":
            if int(clip.quantize or 0) not in (4, 8):
                raise ValueError(
                    f"{entry.family} 的条件编码器必须量化（在「MLX 条件加载器」里把 "
                    f"quantize 改成 8 或 4；当前是 {clip.quantize}）"
                )
            composed = pipeline.compose_h3_prompt(prompt)
            comps = pipeline.prepare_h3_encoder(
                entry, clip, CACHE, runtime.cache_key({"kind": "module", "clip": clip})
            )
            encoding_key = pipeline.h3_prompt_encoding_key(clip, composed)
            embeds, tags = pipeline.encode_h3_prompt(entry, comps, composed, CACHE, encoding_key)
            print(
                f"[MlxTextEncoder] {entry.family}: {int(tags.shape[0])} 个 token，"
                f"条件张量 {tuple(embeds.shape)} {embeds.dtype}"
            )
            # 编码结果已进 h3_prompt 桶（采样器只按键取张量）：条件编码器约 30 GB，
            # 用完立刻丢掉，别占着到采样阶段（换提示词重跑时会重新懒加载）
            pipeline.release_h3_encoder(entry, clip, CACHE)
            cond = MlxConditioning(clip=clip, text=composed, encoding_key=encoding_key)
            return (cond,)

        # 1) 按需创建 text_encoder + tokenizer（同一 handle 第二次执行直接命中缓存）
        comps_key = runtime.cache_key({"kind": "module", "clip": clip})
        comps = pipeline.prepare_encoder(entry, clip, CACHE, comps_key)

        # 2) 编码本条提示词；数组只进缓存，handle 里只留键
        encoding_key = pipeline.prompt_encoding_key(clip, prompt)
        pipeline.encode_text(entry, comps, prompt, CACHE, encoding_key)

        # 3) 交给 MlxKSamplerMLX 的 positive / negative 入口
        cond = MlxConditioning(clip=clip, text=prompt, encoding_key=encoding_key)
        return (cond,)

