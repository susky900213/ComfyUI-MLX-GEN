"""Native MOSS spatial latent layout."""

from __future__ import annotations

import mlx.core as mx

LATENT_CHANNELS = 128
VAE_SCALE_FACTOR = 16


def validate_dimensions(height: int, width: int) -> tuple[int, int]:
    height, width = int(height), int(width)
    if height <= 0 or width <= 0 or height % VAE_SCALE_FACTOR or width % VAE_SCALE_FACTOR:
        raise ValueError(
            f"MOSS 的 width / height 必须是正数且为 16 的倍数，收到 {width}×{height}"
        )
    return height, width


def create_noise(seed: int, height: int, width: int, dtype=mx.float32) -> mx.array:
    height, width = validate_dimensions(height, width)
    return mx.random.normal(
        (1, LATENT_CHANNELS, height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR),
        key=mx.random.key(int(seed)),
    ).astype(dtype)


def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
    height, width = validate_dimensions(height, width)
    if latents.ndim == 3:
        latents = latents[None]
    expected = (LATENT_CHANNELS, height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR)
    if tuple(latents.shape[1:]) != expected:
        raise ValueError(f"MOSS latent 应为 [B,{expected[0]},{expected[1]},{expected[2]}，收到 {tuple(latents.shape)}")
    return latents


def prepare_vae_latents(latents: mx.array, *, mean: mx.array, variance: mx.array) -> mx.array:
    """Undo the Flux.2 VAE BN statistics before decoding, as in the official pipeline."""
    if latents.ndim != 4 or latents.shape[1] != LATENT_CHANNELS:
        raise ValueError("MOSS VAE latent 必须是 [B,128,H,W]")
    eps = mx.array(1e-4, dtype=latents.dtype)
    return latents * mx.sqrt(variance.astype(latents.dtype) + eps).reshape(1, -1, 1, 1) + mean.astype(latents.dtype).reshape(1, -1, 1, 1)