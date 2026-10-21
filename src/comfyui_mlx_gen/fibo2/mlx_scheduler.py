"""Flow-Match Euler schedule used by Fibo-2, implemented without Diffusers."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


def calculate_shift(image_seq_len: int, base_seq_len: int = 256, max_seq_len: int = 4096, base_shift: float = 0.5, max_shift: float = 1.15) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


def _time_shift(mu: float, sigma: float, values: np.ndarray) -> np.ndarray:
    exponent = math.exp(float(mu))
    return exponent / (exponent + np.power(1.0 / values - 1.0, float(sigma))
    )


class Fibo2Scheduler:
    def __init__(self, *, base_shift: float = 0.5, max_shift: float = 1.15, shift_terminal: float | None = None):
        self.base_shift = float(base_shift)
        self.max_shift = float(max_shift)
        self.shift_terminal = shift_terminal
        self.sigmas = np.empty((0,), dtype=np.float32)
        self.timesteps = np.empty((0,), dtype=np.float32)
        self._step_index = 0

    def set_timesteps(self, steps: int, image_seq_len: int) -> np.ndarray:
        steps = int(steps)
        if steps < 1:
            raise ValueError("Fibo-2 steps 必须 >= 1")
        sigmas = np.linspace(1.0, 1.0 / steps, steps, dtype=np.float32)
        mu = calculate_shift(image_seq_len, base_shift=self.base_shift, max_shift=self.max_shift)
        sigmas = _time_shift(mu, 1.0, sigmas).astype(np.float32)
        if self.shift_terminal is not None:
            one_minus = 1.0 - sigmas
            sigmas = 1.0 - one_minus / (one_minus[-1] / (1.0 - float(self.shift_terminal)))
        self.sigmas = np.concatenate([sigmas, np.zeros((1,), dtype=np.float32)])
        self.timesteps = sigmas * 1000.0
        self._step_index = 0
        return self.timesteps

    def step(self, noise: mx.array, sample: mx.array) -> mx.array:
        if self._step_index >= len(self.sigmas) - 1:
            raise RuntimeError("Fibo-2 scheduler.step 次数超过当前 schedule")
        sigma = float(self.sigmas[self._step_index])
        sigma_next = float(self.sigmas[self._step_index + 1])
        self._step_index += 1
        # The official pipeline keeps the diffusion trajectory in float32 even
        # when the transformer/VAE weights are bfloat16.  Returning sample.dtype
        # here silently turns every following step back into bfloat16 when the
        # initial noise was created with the model precision.
        return sample.astype(mx.float32) + noise.astype(mx.float32) * np.float32(sigma_next - sigma)


__all__ = ["Fibo2Scheduler", "calculate_shift"]