"""MlxQwenEditEncoder：Qwen-Image-Edit 的文本条件（必须带参考图）。

与 `MlxTextEncoder` 的区别：Qwen 编辑的条件不是「纯文本」——它要把参考图送进
Qwen2.5-VL 视觉塔，产出 `(embeds, mask)` 给编辑 transformer。正、负各用一个本节点
（各自一条文本），**两个节点都要接同一批图片**：mflux 的负向提示词同样带图，
否则 CFG 会把「图与图的差异」也算成指令差异。

结构：clip（MlxClipLoader 输出）+ text + images → condition（与 MlxTextEncoder 同一个类型名，
直接接 MlxKSamplerMLX 的 positive / negative）。
数组只进 cache.py 的 "prompt_encoding" 桶，handle 里只留缓存键。
"""

from __future__ import annotations

from .. import image as image_mod
from .. import pipeline, runtime
from ..cache import CACHE
from ..types import CLIP, MlxClipHandle, MlxConditioning, condition, entry_for


class MlxQwenEditEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": True},
                ),
                "clip": (CLIP, {}),
                "images": ("IMAGE", {}),
                "max_images": ("INT", {"default": 4, "min": 1, "max": 8}),
            }
        }

    RETURN_TYPES = (condition,)
    RETURN_NAMES = ("condition",)
    FUNCTION = "encode"
    CATEGORY = "MLX/Gen"

    def encode(self, text, clip, images, max_images):
        if clip is None or not isinstance(clip, MlxClipHandle):
            raise ValueError(
                "必须连接 MlxClipLoader 的输出；ComfyUI 原生 CLIP（torch 权重）不能被 MLX 模型使用"
            )
        if images is None:
            raise ValueError("必须连接图片（用 ComfyUI 自带的「Load Image」+「Batch Images」）")
        entry = entry_for(clip.model_type)
        if not entry.supported:
            raise NotImplementedError(f"{clip.model_type} 尚未实现：{entry.notes}")
        if entry.family != "qwen_edit":
            raise ValueError(
                f"{clip.model_type} 不用本节点；普通文生图请用「MLX 文本编码器」（MlxTextEncoder）"
            )
        pils = image_mod.to_pil_batch(images)
        count = min(len(pils), int(max_images))
        if count <= 0:
            raise ValueError("参考图为空")
        pils = pils[:count]
        # 空提示词按空格编码（与 MlxTextEncoder / mflux 对 negative_prompt 的处理一致）
        prompt = text if text and text.strip() else " "

        # 1) 按需创建 text_encoder（含视觉塔）+ tokenizer + VL 两层（同一 handle 命中缓存）
        comps_key = runtime.cache_key({"kind": "module", "clip": clip})
        comps = pipeline.prepare_encoder(entry, clip, CACHE, comps_key)

        # 2) 编码本条条件（带图）；数组只进缓存，handle 里只留键
        encoding_key = pipeline.edit_prompt_encoding_key(
            clip, prompt, image_mod.digest(images), count
        )
        pipeline.encode_edit_conditioning(entry, comps, prompt, pils, CACHE, encoding_key)

        # 3) 交给 MlxKSamplerMLX 的 positive / negative 入口
        cond = MlxConditioning(clip=clip, text=prompt, encoding_key=encoding_key)
        print(f"[MlxQwenEditEncoder] {count} 张参考图 | 文本 {prompt[:32]!r} → condition")
        return (cond,)
