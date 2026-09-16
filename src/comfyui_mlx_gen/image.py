"""MLX / PIL → ComfyUI 图像与掩码类型。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image


def denormalize(arr: Any) -> Any:
    """与 mflux 的 _denormalized 相同：x/2 + 0.5，并裁剪到 [0,1]。"""
    import mlx.core as mx

    return mx.clip(arr / 2 + 0.5, 0, 1)


def to_pil(arr: Any, batch_index: int = -1) -> tuple[Image.Image, ...]:
    """VAE 解码结果（[N,3,H,W] / [N,H,W,3] / [N,3,1,H,W]，值域 [-1,1]）→ PIL 元组。"""
    import mlx.core as mx

    data = np.array(mx.array(denormalize(arr)).astype(mx.float32))
    if data.ndim == 5:
        data = data[:, :, 0, :, :]  # [N,3,1,H,W] → [N,3,H,W]
    if data.ndim == 3:
        data = data[None, ...]
    if data.shape[1] == 3 and data.shape[-1] != 3:
        data = np.transpose(data, (0, 2, 3, 1))  # → [N,H,W,3]
    if batch_index >= 0:
        data = data[[batch_index % data.shape[0]]]
    out: list[Image.Image] = []
    for i in range(data.shape[0]):
        arr8 = np.rint(np.clip(data[i], 0.0, 1.0) * 255).astype(np.uint8)
        out.append(Image.fromarray(arr8, mode="RGB"))
    return tuple(out)


def to_image_batch(images: Sequence[Image.Image]) -> torch.Tensor:
    """PIL 元组 → ComfyUI IMAGE（[N,H,W,3] float32, 0..1）。"""
    if not images:
        raise ValueError("没有图片")
    ref = images[0].size
    stack: list[np.ndarray] = []
    for img in images:
        if img.size != ref:
            raise ValueError(f"图片尺寸不一致: {img.size} 与 {ref}")
        stack.append(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(stack, axis=0))


def to_mask_batch(images: Sequence[Image.Image]) -> torch.Tensor:
    """PIL 元组 → ComfyUI MASK（[N,H,W] float32, 0..1）。"""
    if not images:
        raise ValueError("没有图片")
    ref = images[0].size
    stack: list[np.ndarray] = []
    for img in images:
        if img.size != ref:
            raise ValueError(f"图片尺寸不一致: {img.size} 与 {ref}")
        stack.append(np.asarray(img.convert("L"), dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(stack, axis=0))
