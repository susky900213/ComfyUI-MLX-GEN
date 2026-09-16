import math
from functools import partial
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config

from mflux.models.common.schedulers.base_scheduler import BaseScheduler


@partial(mx.compile, shapeless=True)
def _step(noise, latents, s1, s2):
    output_dtype = noise.dtype
    dt = (s1 - s2).astype(mx.float32)
    sample = latents.astype(mx.float32)
    model_output = noise.astype(mx.float32)
    return (sample + dt * model_output).astype(output_dtype)


class FlowMatchEulerDiscreteScheduler(BaseScheduler):
    def __init__(self, config: "Config"):
        self.config = config
        self.model_config = config.model_config
        self.num_train_timesteps = 1000
        self.shift_terminal = 0.02
        FlowMatchEulerDiscreteScheduler._validate_num_inference_steps(config.num_inference_steps)
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas()

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def set_image_seq_len(self, image_seq_len: int) -> None:
        self._timesteps, self._sigmas = FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas(
            image_seq_len=image_seq_len,
            num_inference_steps=self.config.num_inference_steps,
            base_seq_len=self.model_config.sigma_base_seq_len,
            max_seq_len=self.model_config.sigma_max_seq_len,
            base_shift=self.model_config.sigma_base_shift,
            max_shift=self.model_config.sigma_max_shift,
            shift_terminal=self.model_config.sigma_shift_terminal,
        )

    def set_mu(self, mu: float) -> None:
        num_steps = self.config.num_inference_steps
        sigmas = mx.linspace(1.0, 1.0 / num_steps, num_steps, dtype=mx.float32)
        sigmas = FlowMatchEulerDiscreteScheduler._time_shift_exponential_array(mu, 1.0, sigmas)
        self._timesteps = sigmas * self.num_train_timesteps
        sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)
        self._sigmas = sigmas

    @staticmethod
    def _validate_num_inference_steps(num_inference_steps: int) -> None:
        if num_inference_steps < 2:
            raise ValueError("FlowMatchEulerDiscreteScheduler requires at least 2 inference steps.")

    @staticmethod
    def _compute_linear_mu(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return float(m * image_seq_len + b)

    @staticmethod
    def get_timesteps_and_sigmas(
        image_seq_len: int,
        num_inference_steps: int,
        num_train_timesteps: int = 1000,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift_terminal: float | None = None,
    ) -> tuple[mx.array, mx.array]:
        FlowMatchEulerDiscreteScheduler._validate_num_inference_steps(num_inference_steps)
        sigmas = mx.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=mx.float32)
        mu = FlowMatchEulerDiscreteScheduler._compute_linear_mu(
            image_seq_len=image_seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            base_shift=base_shift,
            max_shift=max_shift,
        )
        sigmas = FlowMatchEulerDiscreteScheduler._time_shift_exponential_array(mu, 1.0, sigmas)
        if shift_terminal is not None:
            sigmas = FlowMatchEulerDiscreteScheduler._stretch_to_terminal_array(sigmas, shift_terminal)
        timesteps = sigmas * num_train_timesteps
        sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)
        return timesteps, sigmas

    @staticmethod
    def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666
        if image_seq_len > 4300:
            return float(a2 * image_seq_len + b2)
        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        return float(a * num_steps + b)

    @staticmethod
    def _time_shift_exponential(mu: float, sigma_power: float, t: float) -> float:
        return math.exp(mu) / (math.exp(mu) + ((1.0 / t - 1.0) ** sigma_power))

    @staticmethod
    def _time_shift_exponential_array(mu: float, sigma_power: float, t: mx.array) -> mx.array:
        return mx.exp(mu) / (mx.exp(mu) + ((1.0 / t - 1.0) ** sigma_power))

    @staticmethod
    def _stretch_to_terminal_array(sigmas: mx.array, shift_terminal: float) -> mx.array:
        one_minus = 1.0 - sigmas
        scale = one_minus[-1] / (1.0 - shift_terminal)
        return 1.0 - (one_minus / scale)

    def _stretch_to_terminal(self, sigmas: list[float]) -> list[float]:
        one_minus_sigmas = [1.0 - s for s in sigmas]
        scale_factor = one_minus_sigmas[-1] / (1.0 - self.shift_terminal)
        stretched = [1.0 - (oms / scale_factor) for oms in one_minus_sigmas]
        return stretched

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        num_steps = self.config.num_inference_steps
        sigma_min = 1.0 / self.num_train_timesteps
        sigma_max = 1.0
        timesteps_linear = [
            sigma_max * self.num_train_timesteps
            - i * (sigma_max - sigma_min) * self.num_train_timesteps / (num_steps - 1)
            for i in range(num_steps)
        ]
        sigmas_linear = [t / self.num_train_timesteps for t in timesteps_linear]
        sigmas_shifted = [FlowMatchEulerDiscreteScheduler._time_shift_exponential(1.0, 1.0, s) for s in sigmas_linear]
        sigmas_final = self._stretch_to_terminal(sigmas_shifted)
        timesteps = [s * self.num_train_timesteps for s in sigmas_final]
        sigmas_with_zero = sigmas_final + [0.0]
        sigmas_arr = mx.array(sigmas_with_zero, dtype=mx.float32)
        timesteps_arr = mx.array(timesteps, dtype=mx.float32)
        return sigmas_arr, timesteps_arr

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        sigmas = kwargs.get("sigmas", self._sigmas)
        return _step(noise, latents, sigmas[timestep + 1], sigmas[timestep])

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        return latents
