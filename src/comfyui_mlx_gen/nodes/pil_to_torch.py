"""MlxPilToTorch：把 MlxPilImage 转成 ComfyUI 的 IMAGE / MASK（H3 还带一路 AUDIO）。"""

from __future__ import annotations

import torch

from .. import image
from ..h3.video import to_comfy_audio


class MlxPilToTorch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("images", {})}}

    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO")
    RETURN_NAMES = ("images", "masks", "audio")
    FUNCTION = "to_torch"
    CATEGORY = "MLX/Gen"

    def to_torch(self, images):
        imgs = images.images
        if images.batch_index >= 0:
            imgs = [imgs[images.batch_index % len(imgs)]]
        # 只解码了音频（H3 的 audio_vae）时没有图片：给 1×1 黑图占位，
        # 免得单独接音频也被「没有图片」挡住
        if imgs:
            batch = image.to_image_batch(imgs)
            mask = image.to_mask_batch(imgs)
        else:
            batch = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            mask = torch.zeros((1, 1, 1), dtype=torch.float32)
        return (batch, mask, to_comfy_audio(images.audio))
