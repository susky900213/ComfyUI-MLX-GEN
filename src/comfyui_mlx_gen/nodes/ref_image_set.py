"""MlxRefImageSet：把多张（尺寸可以各不相同）参考图打包成**有序**的参考图源。

为什么需要这个节点（而不是直接用自带的「Batch Images」）：
ComfyUI 自带的 Batch Images（`comfy_extras/nodes_post_processing.py:BatchImagesNode` →
`batch_images`）会把**第 2 张起**的图改写成第 1 张的尺寸
（`comfy.utils.common_upscale(..., crop="center")`：先居中裁到目标宽高比、再插值到目标像素）；
而 FLUX.2 的参考图是**各自按自己的尺寸**编码的（对齐 mflux
`_Flux2KleinEditHelpers.prepare_reference_image`：面积 ≤1MP 的等比缩放 + 居中裁到 16 的倍数），
顺序又决定参考图 grid ids 的 t 坐标（第 i 张 = 10 + 10*i）——合并成同尺寸批次会先丢掉这两件事。

本节点只做「保序 + 保尺寸」的搬运：图片本体存进 cache.py 的 "ref_source" 桶，
handle（MlxRefImageSource）只带「张数 + 每张尺寸 + 有序摘要 + 缓存键」。
真正的 VAE 编码仍由 MlxVAEEncoder（ref_source 入口）做，因此编码语义、VAE 实例、
ref_encoding 缓存键口径都只有一个来源。

槽位：image1 必填，image2..image10 可选；每个槽位本身也可以是批次（按批次顺序展开）。
Qwen-Image 2.1 可直接使用全部 10 个槽；legacy 图片家族仍由下游 VAE 编码节点执行
各自的参考图数量上限。

用法（与 qwen 的「Picture N」不同，FLUX.2 的文本里没有图像 token，**顺序只能靠接线表达**）：
提示词里说「第一张 / 第二张」时，请按 image1 → image2 → … 的顺序对齐；
本节点的 report 输出列出每张的尺寸与 t 坐标，可接到自带「Preview as Text」核对。
"""

from __future__ import annotations

from .. import image as image_mod
from .. import pipeline
from ..cache import CACHE
from ..types import MlxRefImageSource, ref_source

SLOTS = tuple(f"image{index}" for index in range(1, 11))
T_COORD_BASE = 10  # 与 pipeline.REFERENCE_T_COORD_BASE 一致：第 i 张的 t = 10 + 10 * i
T_COORD_STEP = 10


class MlxRefImageSet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image1": ("IMAGE", {})},
            "optional": {name: ("IMAGE", {}) for name in SLOTS[1:]},
        }

    RETURN_TYPES = (ref_source, "INT", "STRING")
    RETURN_NAMES = ("ref_source", "count", "report")
    FUNCTION = "pack"
    CATEGORY = "MLX/Gen"

    def pack(
        self,
        image1,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
    ):
        values = dict(
            zip(
                SLOTS,
                (
                    image1,
                    image2,
                    image3,
                    image4,
                    image5,
                    image6,
                    image7,
                    image8,
                    image9,
                    image10,
                ),
            )
        )
        pils = []
        for name in SLOTS:
            value = values[name]
            if value is None:
                continue
            # 槽位也可以接批次（同尺寸的一组）；按批次顺序展开，顺序即 t 坐标顺序
            batch = image_mod.to_pil_batch(value)
            if not batch:
                raise ValueError(f"{name} 是空的批次")
            pils.extend(batch)
        if not pils:
            raise ValueError("至少连接一张参考图（image1 必填）")

        sizes = tuple((pil.width, pil.height) for pil in pils)
        digest = image_mod.digest(pils)
        cache_key = pipeline.ref_source_key(digest, sizes)
        pipeline.store_ref_source(pils, CACHE, cache_key)

        handle = MlxRefImageSource(
            count=len(pils), sizes=sizes, digest=digest, cache_key=cache_key
        )
        report = " | ".join(
            f"参考图 {i + 1}: {width}x{height} (t={T_COORD_BASE + T_COORD_STEP * i})"
            for i, (width, height) in enumerate(sizes)
        )
        print(f"[MlxRefImageSet] {len(pils)} 张 → ref_source | {report}")
        return (handle, len(pils), report)