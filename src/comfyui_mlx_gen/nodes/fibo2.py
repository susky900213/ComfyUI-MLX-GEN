"""ComfyUI nodes for the isolated Fibo 2.0 Torch backend.

The module itself does not import Torch, NumPy, or Pillow.  Those optional
libraries are imported only when a node actually converts an image or runs a
pipeline, so existing MLX-only installations can still register all nodes.
"""

from __future__ import annotations

import json
from typing import Any

from ..fibo2 import backend


def _image_runtime() -> tuple[Any, Any, Any]:
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - host dependent
        raise backend.Fibo2BackendUnavailable(
            "Fibo 2 图像节点需要可选依赖 Torch、NumPy 和 Pillow；"
            "不会影响现有 MLX 节点"
        ) from exc
    return np, torch, Image


def _as_numpy(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().to("cpu").float().numpy()
    return np.asarray(value)


def _image_batch_to_pil(value: Any) -> list[Any]:
    """Convert ComfyUI IMAGE ``[B,H,W,C]`` to RGB PIL images."""

    if value is None:
        return []
    np, _, Image = _image_runtime()
    data = _as_numpy(value, np)
    if data.ndim == 3:
        data = data[None, ...]
    if data.ndim != 4 or data.shape[-1] not in (3, 4):
        raise ValueError(f"Fibo 2 的 IMAGE 输入必须是 [B,H,W,3/4]，收到 {data.shape}")
    if data.shape[0] == 0:
        raise ValueError("Fibo 2 的 IMAGE 输入不能为空")
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    return [
        Image.fromarray(
            np.rint(np.clip(item[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8),
            mode="RGB",
        )
        for item in data
    ]


def _mask_to_pil(value: Any, source: Any) -> Any:
    """Convert one ComfyUI MASK to a single L image, preserving white=regenerate."""

    if value is None:
        return None
    np, _, Image = _image_runtime()
    data = _as_numpy(value, np)
    if data.ndim == 4:
        if data.shape[0] != 1:
            raise ValueError("Fibo 2 mask 只能有一个 batch 项")
        if data.shape[1] == 1:
            data = data[0, 0]
        elif data.shape[-1] == 1:
            data = data[0, :, :, 0]
        else:
            raise ValueError(f"Fibo 2 mask 必须是 [H,W] 或 [1,H,W]，收到 {data.shape}")
    elif data.ndim == 3:
        if data.shape[0] == 1:
            data = data[0]
        elif data.shape[-1] == 1:
            data = data[..., 0]
        else:
            raise ValueError(f"Fibo 2 mask 只能包含一个 batch 项，收到 {data.shape}")
    if data.ndim != 2:
        raise ValueError(f"Fibo 2 mask 必须是 [H,W]，收到 {data.shape}")
    mask = Image.fromarray(
        np.rint(np.clip(np.nan_to_num(data, nan=0.0), 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    )
    if mask.size != source.size:
        raise ValueError(f"Fibo 2 mask 尺寸 {mask.size} 必须与 source image 尺寸 {source.size} 一致")
    return mask


def _to_comfy_image(images: Any) -> Any:
    """Convert Diffusers PIL output to ComfyUI IMAGE ``torch.float32``."""

    np, torch, Image = _image_runtime()
    if isinstance(images, Image.Image):
        images = [images]
    elif not isinstance(images, (list, tuple)):
        images = [images]
    if not images:
        raise RuntimeError("Fibo 2 pipeline 没有返回图片")
    arrays = []
    size = None
    for image in images:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        if size is None:
            size = image.size
        elif image.size != size:
            raise ValueError(f"Fibo 2 输出图片尺寸不一致: {image.size} 与 {size}")
        arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(arrays, axis=0))


def _cache_key(model_path: str, dtype: str, device: str) -> str:
    return json.dumps(
        {"family": "fibo2", "path": model_path, "dtype": dtype, "device": device},
        sort_keys=True,
    )


class Fibo2ModelLoader:
    """Validate a Fibo-2 checkpoint and return a lazy model handle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {"default": "briaai/Fibo-2.0-edit-turbo", "multiline": False},
                ),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["auto", "mps", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("FIBO2_MODEL",)
    RETURN_NAMES = ("fibo2_model",)
    FUNCTION = "load"
    CATEGORY = "Fibo 2.0/Torch"

    def load(self, model_path: str, dtype: str, device: str):
        path = str(model_path).strip()
        backend.inspect_model_path(path)
        normalized_dtype = backend.normalize_dtype(dtype)
        normalized_device = backend.normalize_device(device)
        handle = backend.Fibo2ModelHandle(
            model_path=path,
            dtype=normalized_dtype,
            device=normalized_device,
            cache_key=_cache_key(path, normalized_dtype, normalized_device),
        )
        return (handle,)


class Fibo2Sampler:
    """Run Fibo 2.0 dev or edit-turbo with up to five context images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FIBO2_MODEL", {}),
                "prompt": (
                    "STRING",
                    {
                        "default": '{"short_description":"a cinematic portrait"}',
                        "multiline": True,
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "source image；edit-turbo 必须连接"}),
                "mask": ("MASK", {"tooltip": "白色区域重绘；只能与一张 source image 一起使用"}),
                "reference_1": ("IMAGE", {}),
                "reference_2": ("IMAGE", {}),
                "reference_3": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "sample"
    CATEGORY = "Fibo 2.0/Torch"

    def sample(
        self,
        model: backend.Fibo2ModelHandle,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        image: Any = None,
        mask: Any = None,
        reference_1: Any = None,
        reference_2: Any = None,
        reference_3: Any = None,
    ):
        if not isinstance(model, backend.Fibo2ModelHandle):
            raise ValueError("model 必须连接 Fibo2ModelLoader，而不是 MLX 或 legacy FIBO loader")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Fibo 2 prompt 不能为空（推荐使用结构化 JSON caption）")

        source_images = _image_batch_to_pil(image)
        if backend.is_edit_turbo_model_path(model.model_path) and not source_images:
            raise ValueError("Fibo-2.0-edit-turbo 必须连接 image source；reference 不能替代 source")
        images = list(source_images)
        for reference in (reference_1, reference_2, reference_3):
            images.extend(_image_batch_to_pil(reference))
        if mask is not None and len(source_images) != 1:
            raise ValueError("Fibo 2 mask 必须与恰好一张 source image 一起连接")
        mask_image = _mask_to_pil(mask, source_images[0]) if mask is not None else None

        result = backend.run_pipeline(
            model,
            prompt=prompt,
            images=images or None,
            mask=mask_image,
            seed=int(seed),
            steps=int(steps),
            guidance=float(guidance),
        )
        return (_to_comfy_image(result),)


__all__ = ["Fibo2ModelLoader", "Fibo2Sampler"]