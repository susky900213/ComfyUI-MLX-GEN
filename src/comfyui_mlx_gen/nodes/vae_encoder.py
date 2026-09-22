"""MlxVAEEncoder：把参考图编码成参考图条件（edit 用）。

两个参考图入口（**二选一**，都接会直接报错）：

- `images`（`IMAGE` 批次）：一批**同尺寸**的参考图（自带「Batch Images」的输出）；
- `ref_source`（`mlx_ref_image_src`）：「MLX 参考图集」输出的**有序图集**，每张图保留
  自己的尺寸与位置（顺序 = 编码顺序 = 参考图 grid ids 的 t 坐标顺序）。

为什么要有第二个入口：自带「Batch Images」会把第 2 张起的图改写成第 1 张的尺寸
（`comfy/utils.py:common_upscale(..., crop="center")`：先居中裁到目标宽高比、再插值），
而 FLUX.2 的参考图是**各自按自己的尺寸**编码的；同尺寸时两个入口等价（`images` 更省事）。

VAE 由「MLX VAE 加载器」提供，本节点按 handle 物化 VAE 并编码。两个大类语义不同：

- flux2：预处理（≤1MP + 裁到 16 倍数）→ VAE encode → patchify → bn 归一化 → pack
  → grid ids（第 i 张参考图的 t 坐标 = 10 + 10 * i，目标图是 0）；输出首张参考图的
  宽高作为「建议生成尺寸」；
- qwen_edit：每张参考图**缩放到目标编辑尺寸**（16 的倍数）→ VAE encode → pack
  → 沿 seq concat；grid 是 (1, height//16, width//16)；输出的宽高就是采样器**必须**
  用的宽高（参考图 latent 与目标尺寸强绑定，不一致会被采样器直接拒绝）。
  qwen_edit 暂不支持 `ref_source` 入口（它的参考图要按目标尺寸拉伸，文本条件也要带
  同一批图 —— 见 docs/FLUX2_MULTI_IMAGE_EDIT_IMPLEMENTATION.md §3.2 第 5 条）。
- qwen_image_21：每张图保持比例缩放到目标面积并对齐 32 倍数，同一份 RGBA 像素
  经原生 VAE encoder 变成 64 通道参考 latent，白底 RGB 副本交给 Qwen3-VL；最多
  10 张，支持 `ref_source` 保留异尺寸图的顺序。

数组存进 cache.py 的 "ref_encoding" 桶，输出 MlxReferenceImages（只带键）。
输出类型 `mlx_ref_images` 只能接到 MlxKSamplerMLX 的可选入口 `ref_images`；
width / height 可接到采样器的 width / height（在 UI 里把这两个 widget
「Convert widget to input」），也可以手工填成一致的值。

签名里前 6 个参数的位置顺序与改造前完全一致（`images, vae, max_reference_images,
resize_mode, width, height`）——既有文档 / 冒烟脚本里的位置调用不受影响。
"""

from __future__ import annotations

from .. import image as image_mod
from .. import pipeline
from ..cache import CACHE
from ..types import (
    MlxRefImageSource,
    MlxReferenceImages,
    MlxVaeHandle,
    entry_for,
    ref_images,
    ref_source,
    validate_model_family,
    vae,
)

RESIZE_MODES = ("auto", "aspect_area_crop", "keep_resolution", "stretch", "aspect_fit_pad")


def resolve_resize_mode(family: str, resize_mode: str) -> str:
    """`auto` → flux2 用 aspect_area_crop（现状）、qwen_edit 用 stretch（与 mflux 一致）。

    显式值不做跨 family 兜底：给 flux2 传 stretch / 给 qwen_edit 传 keep_resolution
    一律报错，避免「参数被静默忽略」这类难以排查的行为。
    """
    if resize_mode != "auto":
        return resize_mode
    if family == "flux2":
        return "aspect_area_crop"
    if family == "qwen_image_21":
        return "auto"
    return "stretch"


def resolve_reference_images(
    entry,
    images,
    ref_source_handle,
    max_reference_images: int,
) -> tuple[list, str, int]:
    """参考图来源解析（批次 / 图集二选一）→ `(pils, digest, count)`。

    - 批次入口（images）：按 max_reference_images 截断（**沿用既有行为**，老工作流的
      `max_reference_images=2` 语义不能悄悄变）；
    - 图集入口（ref_source）：**不截断**，张数超过上限直接报错 —— 图集的张数是显式接线
      出来的，「少看了后面几张」会让提示词里的「第二张 / 第三张」指向错的东西。
    """
    if images is not None and ref_source_handle is not None:
        raise ValueError(
            "images（同尺寸批次）与 ref_source（参考图集）只能接一个："
            "两个都接会重复计数；尺寸各不相同时请只接 ref_source"
        )
    if ref_source_handle is not None:
        if not isinstance(ref_source_handle, MlxRefImageSource):
            raise ValueError("ref_source 必须接「MLX 参考图集」（MlxRefImageSet）的输出")
        if entry.family not in ("flux2", "qwen_image_21"):
            raise NotImplementedError(
                f"{entry.family} 还不支持 ref_source 入口（目前只有 flux2 / qwen_image_21）："
                "qwen_edit 的参考图要按目标尺寸拉伸、文本条件也要带同一批图，"
                "请继续用「Batch Images」→ images 批次"
            )
        family_limit = 10 if entry.family == "qwen_image_21" else 8
        limit = min(family_limit, max(1, int(max_reference_images)))
        if int(ref_source_handle.count) > limit:
            raise ValueError(
                f"参考图集里有 {ref_source_handle.count} 张，但本节点的 "
                f"max_reference_images={limit}：请把上限改到 ≥ 张数"
                "（图集的张数不会像批次那样被静默截断）"
            )
        pils = list(pipeline.load_ref_source(CACHE, ref_source_handle.cache_key))
        return pils, ref_source_handle.digest, int(ref_source_handle.count)
    if images is None:
        raise ValueError(
            "必须连接参考图：同尺寸的一批图用「Load Image」+「Batch Images」接 images；"
            "尺寸各不相同的多张图用「MLX 参考图集」（MlxRefImageSet）接 ref_source"
        )
    pils = list(image_mod.to_pil_batch(images, preserve_alpha=entry.family == "qwen_image_21"))
    family_limit = 10 if entry.family == "qwen_image_21" else 8
    count = min(len(pils), max(1, int(max_reference_images)), family_limit)
    if count <= 0:
        raise ValueError("参考图为空")
    return pils[:count], image_mod.digest(images), count


class MlxVAEEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": (vae, {}),
                "max_reference_images": ("INT", {"default": 2, "min": 1, "max": 10}),
                "resize_mode": (list(RESIZE_MODES), {"default": "auto"}),
                # qwen_edit 用：0 = 用首张参考图尺寸（等价 mflux 的 DimensionResolver）。
                # 采样器的 width/height 必须与这里输出的宽高一致（不接就把采样器填成同值）。
                "width": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "height": ("INT", {"default": 0, "min": 0, "max": 2048}),
            },
            "optional": {
                # 参考图的两个来源（**二选一**）：
                # - images：一批同尺寸的图（自带「Batch Images」的输出）—— flux2 / qwen_edit 都可用；
                # - ref_source：有序图集（「MLX 参考图集」的输出，每张各自尺寸）—— 目前只有 flux2。
                #   images 之所以改成 optional：只接图集时它必须是"可以不接"，否则排队阶段
                #   就会报「Required input is missing」。
                "images": ("IMAGE", {}),
                "ref_source": (ref_source, {}),
            },
        }

    RETURN_TYPES = (ref_images, "INT", "INT")
    RETURN_NAMES = ("ref_images", "width", "height")
    FUNCTION = "encode"
    CATEGORY = "MLX/Gen"

    def encode(
        self,
        images=None,
        vae=None,
        max_reference_images=2,
        resize_mode="auto",
        width=0,
        height=0,
        ref_source=None,
    ):
        try:
            return self._encode(
                images, vae, max_reference_images, resize_mode, width, height, ref_source
            )
        finally:
            # ref_encoding 只保存编码后的数组 / PIL；参考图编码节点不应把 VAE
            # 继续留给后续采样器。
            vae_module = None
            if isinstance(vae, MlxVaeHandle):
                pipeline.release_vae_component(vae, CACHE)

    def _encode(
        self,
        images=None,
        vae=None,
        max_reference_images=2,
        resize_mode="auto",
        width=0,
        height=0,
        ref_source=None,
    ):
        if vae is None or not isinstance(vae, MlxVaeHandle):
            raise ValueError(
                "必须连接「MLX VAE 加载器」的输出；"
                "ComfyUI 原生 VAE（torch 权重）不能被 MLX 节点使用"
            )
        entry = entry_for(vae.model_type)
        validate_model_family(vae.model_type, vae.path)
        if not entry.supported:
            raise NotImplementedError(f"{vae.model_type} 尚未实现：{entry.notes}")
        # 1) 参考图来源（批次 / 图集）→ 有序 PIL 列表 + 摘要 + 张数
        pils, digest, count = resolve_reference_images(
            entry, images, ref_source, max_reference_images
        )
        resolved_mode = resolve_resize_mode(entry.family, resize_mode)
        # 按 handle 物化 VAE：与 MlxVAEDecoder 接同一个 handle 时就是同一份实例
        vae_module = pipeline.vae_component(vae, CACHE)

        if entry.family == "qwen_image_21":
            params = {
                "count": count,
                "resize_mode": resolved_mode,
                "family": "qwen_image_21",
                "width": int(width),
                "height": int(height),
            }
            cache_key = pipeline.reference_image_cache_key(vae, digest, params)
            encoded = pipeline.encode_qwen21_reference_images(
                vae_module,
                pils,
                CACHE,
                cache_key,
                count,
                int(width),
                int(height),
                resolved_mode,
            )
            handle = MlxReferenceImages(
                model_type=vae.model_type,
                vae_path=vae.path,
                count=count,
                height=int(encoded["height"]),
                width=int(encoded["width"]),
                cache_key=cache_key,
                vae_cache_key=vae.cache_key,
                precision=vae.precision,
                quantize=vae.quantize,
                edit_kind="qwen_image_21",
                seq_len=int(encoded["latents"].shape[1]),
            )
            print(
                f"[MlxVAEEncoder] qwen_image_21 | {count} 张参考图 → "
                f"seq={encoded['latents'].shape[1]}，建议画布 {handle.width}×{handle.height}"
            )
            return (handle, handle.width, handle.height)

        if entry.family == "flux2":
            if resolved_mode not in ("aspect_area_crop", "keep_resolution"):
                raise ValueError(
                    f"flux2 只支持 aspect_area_crop / keep_resolution，收到 {resolved_mode}"
                )
            params = {"count": count, "resize_mode": resolved_mode, "family": "flux2"}
            cache_key = pipeline.reference_image_cache_key(vae, digest, params)
            packed, _ids, use_w, use_h = pipeline.encode_reference_images(
                entry, vae_module, pils, CACHE, cache_key, count, resolved_mode
            )
            handle = MlxReferenceImages(
                model_type=vae.model_type,
                vae_path=vae.path,
                count=count,
                height=use_h,
                width=use_w,
                cache_key=cache_key,
                vae_cache_key=vae.cache_key,
                precision=vae.precision,
                quantize=vae.quantize,
                edit_kind="flux2",
                seq_len=int(packed.shape[1]),
            )
            print(
                f"[MlxVAEEncoder] flux2 | {count} 张参考图 → seq={packed.shape[1]}，"
                f"建议尺寸 {use_w}×{use_h}"
            )
            return (handle, use_w, use_h)

        if entry.family == "qwen_edit":
            if resolved_mode not in pipeline.QWEN_EDIT_RESIZE_MODES:
                raise ValueError(
                    f"qwen_edit 只支持 {' / '.join(pipeline.QWEN_EDIT_RESIZE_MODES)}"
                    f"（默认 stretch），收到 {resolved_mode}"
                )
            params = {
                "count": count,
                "resize_mode": resolved_mode,
                "family": "qwen_edit",
                # 宽高只是「widget 原值」，派生后的真实尺寸由首图决定；两者一起进键，换哪个都换键
                "width": int(width),
                "height": int(height),
            }
            cache_key = pipeline.reference_image_cache_key(vae, digest, params)
            packed, _ids, cond_grid, use_w, use_h = pipeline.encode_edit_reference_latents(
                entry,
                vae_module,
                pils,
                CACHE,
                cache_key,
                count,
                int(width),
                int(height),
                resolved_mode,
            )
            handle = MlxReferenceImages(
                model_type=vae.model_type,
                vae_path=vae.path,
                count=count,
                height=use_h,
                width=use_w,
                cache_key=cache_key,
                vae_cache_key=vae.cache_key,
                precision=vae.precision,
                quantize=vae.quantize,
                edit_kind="qwen_edit",
                cond_grid=cond_grid,
                seq_len=int(packed.shape[1]),
            )
            print(
                f"[MlxVAEEncoder] qwen_edit | {count} 张参考图 → {use_w}×{use_h}，"
                f"seq={packed.shape[1]}（目标尺寸请与采样器一致）"
            )
            return (handle, use_w, use_h)

        raise NotImplementedError(
            f"{entry.family} 暂不支持参考图编辑（本节点只接 flux2 / qwen_edit 两类；"
            "qwen_image 是文生图大类，要用参考图请改选 qwen_edit 大类 + "
            "qwen-image-edit-2511-8bit 权重）"
        )
