"""Qwen-Image 2.1 latent layout and dynamic FlowMatch Euler schedule."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


def validate_dimensions(height: int, width: int) -> tuple[int, int]:
    height, width = int(height), int(width)
    if height < 32 or width < 32 or height % 32 or width % 32:
        raise ValueError(
            f"Qwen-Image 2.1 的宽高必须是 32 的倍数且至少 32，收到 {width}×{height}"
        )
    return height, width


def create_noise(seed: int, height: int, width: int, dtype=mx.bfloat16) -> mx.array:
    height, width = validate_dimensions(height, width)
    mx.random.seed(int(seed))
    latent_h, latent_w = height // 16, width // 16
    return mx.random.normal((1, latent_h * latent_w, 64)).astype(dtype)


def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
    height, width = validate_dimensions(height, width)
    if latents.ndim == 2:
        latents = latents[None]
    batch, sequence, channels = latents.shape
    latent_h, latent_w = height // 16, width // 16
    if (sequence, channels) != (latent_h * latent_w, 64):
        raise ValueError(
            f"Qwen-Image 2.1 latent 形状应为 [B,{latent_h * latent_w},64]，"
            f"收到 {tuple(latents.shape)}"
        )
    # Public VAE layout is [B,C,T,H,W].
    return latents.transpose(0, 2, 1).reshape(batch, channels, 1, latent_h, latent_w)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 8192,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


def _time_shift(mu: float, sigma: float, timesteps: np.ndarray) -> np.ndarray:
    # diffusers FlowMatchEulerDiscreteScheduler.time_shift, exponential mode.
    exponent = math.exp(float(mu))
    return exponent / (exponent + np.power(1.0 / timesteps - 1.0, float(sigma)))


def sigma_schedule(num_steps: int, image_seq_len: int) -> np.ndarray:
    if int(num_steps) < 1:
        raise ValueError(f"Qwen-Image 2.1 steps 必须 >= 1，收到 {num_steps}")
    sigmas = np.linspace(1.0, 1.0 / int(num_steps), int(num_steps), dtype=np.float32)
    mu = calculate_shift(image_seq_len)
    sigmas = _time_shift(mu, 1.0, sigmas).astype(np.float32)
    # shift_terminal=0.02: stretch the schedule while preserving sigma[0] == 1.
    one_minus = 1.0 - sigmas
    if one_minus[-1] > 0:
        sigmas = 1.0 - one_minus / (one_minus[-1] / (1.0 - 0.02))
    return np.concatenate([sigmas.astype(np.float32), np.zeros((1,), dtype=np.float32)])


def euler_step(noise: mx.array, latents: mx.array, sigma: float, sigma_next: float) -> mx.array:
    return (
        latents.astype(mx.float32)
        + noise.astype(mx.float32) * np.float32(sigma_next - sigma)
    ).astype(latents.dtype)