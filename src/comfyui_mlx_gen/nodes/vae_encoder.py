"""MlxVAEEncoder：把 ComfyUI 的 IMAGE + 一个 VAE handle 编码成 Flux.2 参考图条件（edit 用）。

图片加载交给 ComfyUI 自带的「Load Image」（多张时用自带的「Batch Images」），
VAE 由「MLX VAE 加载器」提供（同一个 handle 通常也接到「MLX VAE 解码」上），
本节点只负责：

1. 预处理每张参考图（默认：等比缩放到面积 ≤1MP + 中心裁剪到 16 的倍数，与 mflux 一致）；
2. 按 handle 物化 VAE（命中缓存则复用编码器/解码器共用的那一份）并 encode；
3. patchify → bn 归一化（用 vae.bn 的 running_mean/var）→ pack → 生成 grid ids
   （第 i 张参考图的 t 坐标 = 10 + 10 * i，目标图是 0）；
4. 数组存进 cache.py 的 "ref_encoding" 桶，输出 MlxReferenceImages（只带键）。

输出类型 `mlx_ref_images` 只能接到 MlxKSamplerMLX 的可选入口 `ref_images`；
width / height 是首张参考图预处理后的尺寸（16 的倍数），可接到采样器的
width / height（在 UI 里把这两个 widget「Convert widget to input」）。
"""

from __future__ import annotations

from .. import image as image_mod
from .. import pipeline
from ..cache import CACHE
from ..types import MlxReferenceImages, MlxVaeHandle, entry_for, ref_images, vae

RESIZE_MODES = ["aspect_area_crop", "keep_resolution"]


class MlxVAEEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "vae": (vae, {}),
                "max_reference_images": ("INT", {"default": 2, "min": 1, "max": 8}),
                "resize_mode": (RESIZE_MODES, {"default": "aspect_area_crop"}),
            }
        }

    RETURN_TYPES = (ref_images, "INT", "INT")
    RETURN_NAMES = ("ref_images", "width", "height")
    FUNCTION = "encode"
    CATEGORY = "MLX/Gen"

    def encode(self, images, vae, max_reference_images, resize_mode):
        if images is None:
            raise ValueError("必须连接图片（用 ComfyUI 自带的「Load Image」节点）")
        if vae is None or not isinstance(vae, MlxVaeHandle):
            raise ValueError(
                "必须连接「MLX VAE 加载器」的输出；"
                "ComfyUI 原生 VAE（torch 权重）不能被 MLX 节点使用"
            )
        if resize_mode not in RESIZE_MODES:
            raise ValueError(f"未知的 resize_mode: {resize_mode}")
        entry = entry_for(vae.model_type)
        if entry.family != "flux2":
            raise NotImplementedError(
                f"{vae.model_type} 暂不支持参考图编辑（目前只有 flux2 的 edit 路径）"
            )
        pils = image_mod.to_pil_batch(images)
        count = min(len(pils), int(max_reference_images))
        params = {"count": count, "resize_mode": resize_mode}
        cache_key = pipeline.reference_image_cache_key(vae, image_mod.digest(images), params)
        # 按 handle 物化 VAE：与 MlxVAEDecoder 接同一个 handle 时就是同一份实例
        vae_module = pipeline.vae_component(vae, CACHE)
        packed, ids, width, height = pipeline.encode_reference_images(
            entry, vae_module, pils, CACHE, cache_key, count, resize_mode
        )
        handle = MlxReferenceImages(
            model_type=vae.model_type,
            vae_path=vae.path,
            count=count,
            height=height,
            width=width,
            cache_key=cache_key,
            vae_cache_key=vae.cache_key,
            precision=vae.precision,
            quantize=vae.quantize,
        )
        print(
            f"[MlxVAEEncoder] {count} 张参考图 → seq={packed.shape[1]}，"
            f"建议尺寸 {width}×{height}"
        )
        return (handle, width, height)
