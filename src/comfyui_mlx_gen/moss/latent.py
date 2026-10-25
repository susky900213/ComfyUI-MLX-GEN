"""Native MOSS spatial latent layout."""

from __future__ import annotations

import mlx.core as mx

LATENT_CHANNELS = 128
VAE_SCALE_FACTOR = 16
VAE_LATENT_CHANNELS = 32
PATCH_SIZE = 2
BN_EPS = 1e-4


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


def create_zero(height: int, width: int, dtype=mx.float32) -> mx.array:
    """Create the zero target latent required by MOSS Image Edit."""
    height, width = validate_dimensions(height, width)
    return mx.zeros(
        (1, LATENT_CHANNELS, height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR),
        dtype=dtype,
    )


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


def _ensure_4d(latents: mx.array) -> mx.array:
    if latents.ndim == 5 and latents.shape[2] == 1:
        return latents[:, :, 0, :, :]
    if latents.ndim != 4:
        raise ValueError(f"MOSS VAE latent 必须是 4D/5D，收到 {tuple(latents.shape)}")
    return latents


def patchify_flux2_latents(latents: mx.array) -> mx.array:
    """Convert ``[B,32,H/8,W/8]`` to Flux.2/MOSS ``[B,128,H/16,W/16]``."""
    latents = _ensure_4d(latents)
    if latents.shape[1] != VAE_LATENT_CHANNELS:
        raise ValueError(
            f"MOSS 参考 VAE latent 必须有 {VAE_LATENT_CHANNELS} 个通道，"
            f"收到 {latents.shape[1]}"
        )
    height, width = int(latents.shape[2]), int(latents.shape[3])
    if height % PATCH_SIZE or width % PATCH_SIZE:
        raise ValueError(
            "MOSS 参考 VAE latent 的空间尺寸必须可被 2×2 patchify："
            f"收到 {height}×{width}"
        )
    value = latents.reshape(
        latents.shape[0], VAE_LATENT_CHANNELS, height // 2, 2, width // 2, 2
    )
    value = value.transpose(0, 1, 3, 5, 2, 4)
    return value.reshape(latents.shape[0], LATENT_CHANNELS, height // 2, width // 2)


def normalize_flux2_latents(
    latents: mx.array, *, mean: mx.array, variance: mx.array
) -> mx.array:
    """Apply Flux.2 VAE BN normalization used by the MOSS edit adapter."""
    latents = _ensure_4d(latents)
    if latents.shape[1] != LATENT_CHANNELS:
        raise ValueError(f"MOSS patchified latent 必须是 [B,128,H,W]，收到 {tuple(latents.shape)}")
    mean = mx.array(mean, dtype=latents.dtype).reshape(1, LATENT_CHANNELS, 1, 1)
    variance = mx.array(variance, dtype=latents.dtype).reshape(1, LATENT_CHANNELS, 1, 1)
    return (latents - mean) / mx.sqrt(variance + mx.array(BN_EPS, dtype=latents.dtype))


def prepare_reference_latent(
    encoded: mx.array, *, mean: mx.array, variance: mx.array
) -> mx.array:
    """Flux.2 VAE output → normalized MOSS spatial reference latent."""
    return normalize_flux2_latents(patchify_flux2_latents(encoded), mean=mean, variance=variance)


def validate_reference_latent(latents: mx.array) -> tuple[int, int, int]:
    """Validate a cached MOSS reference latent and return ``(B,H,W)``."""
    if latents.ndim != 4 or latents.shape[1] != LATENT_CHANNELS:
        raise ValueError(f"MOSS reference latent 必须是 [B,128,H,W]，收到 {tuple(latents.shape)}")
    if latents.shape[2] <= 0 or latents.shape[3] <= 0:
        raise ValueError("MOSS reference latent 的空间尺寸必须为正数")
    return int(latents.shape[0]), int(latents.shape[2]), int(latents.shape[3])