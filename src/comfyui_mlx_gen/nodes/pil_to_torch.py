"""MlxPilToTorch：把 MlxPilImage 转成 ComfyUI 的 IMAGE / MASK。"""

from __future__ import annotations

from .. import image


class MlxPilToTorch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("images", {})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "to_torch"
    CATEGORY = "MLX/Gen"

    def to_torch(self, images):
        imgs = images.images
        if images.batch_index >= 0:
            imgs = [imgs[images.batch_index % len(imgs)]]
        return (image.to_image_batch(imgs), image.to_mask_batch(imgs))
