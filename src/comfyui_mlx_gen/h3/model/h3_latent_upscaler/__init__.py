"""MiniMax-H3 latent 放大网络（MLX 移植，3D 变体）。"""

from comfyui_mlx_gen.h3.model.h3_latent_upscaler.latent_resizer_3d import (
    LatentResizer3D,
    adapt_state,
    build,
    detect_arch,
    load,
    upscale_latents,
)

__all__ = [
    "LatentResizer3D",
    "adapt_state",
    "build",
    "detect_arch",
    "load",
    "upscale_latents",
]
