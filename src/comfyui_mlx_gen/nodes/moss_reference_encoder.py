"""MOSS Image Edit reference-image encoder.

This is intentionally separate from ``MlxVAEEncoder``.  MOSS consumes a
Flux.2-normalized VAE latent in spatial ``[B,128,H/16,W/16]`` form and later
turns it into transformer K/V cache; the generic ``mlx_ref_images`` contract
is packed-token based and has different cache ownership semantics.
"""

from __future__ import annotations

from .. import image as image_mod
from .. import pipeline, runtime
from ..cache import CACHE
from ..types import MlxMossReferenceImage, MlxVaeHandle, entry_for, moss_ref_image, validate_model_family, vae


class MlxMossReferenceImageEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": (vae, {}),
                "images": ("IMAGE", {}),
                "max_reference_images": ("INT", {"default": 1, "min": 1, "max": 1}),
            }
        }

    RETURN_TYPES = (moss_ref_image, "INT", "INT")
    RETURN_NAMES = ("reference", "width", "height")
    FUNCTION = "encode"
    CATEGORY = "MLX/Gen/MOSS"

    def encode(self, vae, images, max_reference_images=1):
        if not isinstance(vae, MlxVaeHandle):
            raise ValueError("MOSS 参考图编码必须连接 MlxVAELoader 的输出")
        if vae.model_type != "moss":
            raise ValueError("MOSS 参考图编码只能使用 model_type=moss 的 VAE")
        if int(max_reference_images) != 1:
            raise ValueError("MOSS Image Edit LoRA 当前只支持一张参考图")
        entry = entry_for(vae.model_type)
        validate_model_family(vae.model_type, vae.path)
        if not entry.supported:
            raise NotImplementedError(f"{vae.model_type} 尚未实现：{entry.notes}")
        pils = list(image_mod.to_pil_batch(images))
        if not pils:
            raise ValueError("MOSS 参考图不能为空")
        prepared = pipeline.prepare_moss_reference_image(pils[0])
        digest = image_mod.digest(images)
        cache_key = runtime.cache_key(
            {
                "kind": "moss_reference_image",
                "version": 1,
                "vae": vae,
                "digest": digest,
                "width": int(prepared.width),
                "height": int(prepared.height),
            }
        )
        kv_cache_key = pipeline.moss_reference_kv_identity_key(cache_key)
        vae_module = None
        try:
            vae_module = pipeline.vae_component(vae, CACHE)
            packed = pipeline.encode_moss_reference_image(
                vae_module, prepared, CACHE, cache_key
            )
        finally:
            vae_module = None
            pipeline.release_vae_component(vae, CACHE)
        handle = MlxMossReferenceImage(
            model_type="moss",
            vae_path=vae.path,
            count=1,
            height=int(prepared.height),
            width=int(prepared.width),
            cache_key=cache_key,
            vae_cache_key=vae.cache_key,
            precision=vae.precision,
            quantize=vae.quantize,
            latent_height=int(packed.shape[2]),
            latent_width=int(packed.shape[3]),
            kv_cache_key=kv_cache_key,
        )
        print(
            f"[MOSS 参考图] 1 张 → spatial latent {tuple(packed.shape)}，"
            f"尺寸 {prepared.width}×{prepared.height}"
        )
        return (handle, int(prepared.width), int(prepared.height))