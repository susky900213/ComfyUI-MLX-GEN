"""MLX / PIL → ComfyUI 图像与掩码类型。"""

from __future__ import annotations

import hashlib
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


def to_pil_batch(images: Any) -> tuple[Image.Image, ...]:
    """ComfyUI IMAGE（torch [B,H,W,C] float32 0..1）→ PIL 元组（参考图编码用）。"""
    data = images
    if hasattr(data, "detach"):
        data = data.detach().to("cpu").float().numpy()
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 3:
        data = data[None, ...]
    if data.shape[-1] == 4:
        data = data[..., :3]
    if data.shape[-1] != 3:
        raise ValueError(f"IMAGE 通道数不是 3/4: {data.shape}")
    out: list[Image.Image] = []
    for i in range(data.shape[0]):
        arr8 = np.rint(np.clip(data[i], 0.0, 1.0) * 255.0).astype(np.uint8)
        out.append(Image.fromarray(arr8, mode="RGB"))
    return tuple(out)


def digest(items: Any) -> str:
    """IMAGE 张量 / PIL 批次的稳定摘要（做缓存键，保证换图必换键）。

    用未预处理的原始输入算：1MP float32 ≈ 12MB，毫秒级，相比 VAE 前向可忽略。
    """
    h = hashlib.sha256()
    if hasattr(items, "detach"):
        arr = items.detach().to("cpu").float().numpy()
        h.update(str(tuple(arr.shape)).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()
    for img in items:
        rgb = img.convert("RGB")
        h.update(str(rgb.size).encode())
        h.update(np.asarray(rgb, dtype=np.uint8).tobytes())
    return h.hexdigest()
