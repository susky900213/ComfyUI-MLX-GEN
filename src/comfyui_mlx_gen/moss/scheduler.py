"""MOSS-Image nearest-grid Flow-Matching Euler scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class MOSSImageScheduler:
    num_train_timesteps: int = 1000
    shift: float = 5.0
    training_shift: float = 1.0

    def __post_init__(self) -> None:
        if self.num_train_timesteps < 2 or self.shift <= 0 or self.training_shift <= 0:
            raise ValueError("MOSS scheduler 的 train timesteps 和 shift 必须为正数")
        self.sigmas = np.empty((0,), dtype=np.float32)
        self.timesteps = np.empty((0,), dtype=np.float32)
        self._step_index = 0

    def _grid(self, count: int, shift: float) -> np.ndarray:
        total = float(self.num_train_timesteps)
        lower = shift / (total + shift - 1.0)
        values = np.linspace(1.0, lower, int(count), dtype=np.float64)
        return (shift * values / (1.0 + (shift - 1.0) * values)).astype(np.float32)

    def set_timesteps(self, num_inference_steps: int) -> np.ndarray:
        steps = int(num_inference_steps)
        if steps < 1:
            raise ValueError("MOSS inference steps 必须 >= 1")
        sigmas = self._grid(steps, float(self.shift))
        training = self._grid(self.num_train_timesteps, float(self.training_shift))[::-1]
        upper = np.searchsorted(training, sigmas, side="left").clip(0, self.num_train_timesteps - 1)
        lower = np.maximum(upper - 1, 0)
        nearest = np.where(
            np.abs(sigmas - training[lower]) <= np.abs(training[upper] - sigmas),
            lower,
            upper,
        )
        snapped = training[nearest].astype(np.float32)
        # torch.unique_consecutive preserves denoising order; np.unique would
        # sort the schedule and silently reverse MOSS's Euler trajectory.
        if len(snapped) > 1:
            keep = np.concatenate(([True], snapped[1:] != snapped[:-1]))
            snapped = snapped[keep]
        # Diffusers' FlowMatchEulerDiscreteScheduler converts custom sigmas
        # to timesteps with ``sigma * num_train_timesteps`` and appends a
        # terminal zero sigma.  Keep that exact contract rather than using
        # the latent grid's ``num_train_timesteps - 1`` scale.
        self.sigmas = np.concatenate([snapped, np.zeros((1,), dtype=np.float32)])
        self.timesteps = (snapped * float(self.num_train_timesteps)).astype(np.float32)
        self._step_index = 0
        return self.timesteps

    def step(self, model_output: mx.array, timestep: float | mx.array, sample: mx.array) -> mx.array:
        if self.sigmas.size == 0:
            raise RuntimeError("必须先调用 MOSSImageScheduler.set_timesteps")
        index = self._step_index
        if index >= len(self.sigmas) - 1:
            raise RuntimeError("MOSS scheduler.step 次数超过当前 schedule")
        sigma = float(self.sigmas[index])
        sigma_next = float(self.sigmas[index + 1])
        self._step_index += 1
        return sample.astype(mx.float32) + model_output.astype(mx.float32) * np.float32(sigma_next - sigma)