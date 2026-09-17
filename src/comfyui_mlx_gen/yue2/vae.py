"""YuE2 Oobleck VAE decoder（MLX channels-last 实现）。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

FRAME = 1920  # 每个声学 latent 帧对应 48 kHz 音频的 1920 个采样点
OUTPUT_TRIM = 64  # Oobleck 卷积边界会从整段末尾裁掉的采样点数


def audio_samples(frames: int) -> int:
    """返回 ``frames`` 个声学 latent 经 Oobleck 解码后的精确采样点数。"""
    return max(0, int(frames) * FRAME - OUTPUT_TRIM)


class Snake(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x):
        alpha, beta = mx.exp(self.alpha), mx.exp(self.beta)
        return x + (1.0 / (beta + 1e-9)) * mx.sin(x * alpha) ** 2


class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = [
            Snake(channels),
            nn.Conv1d(channels, channels, 7, dilation=dilation, padding=3 * dilation),
            Snake(channels),
            nn.Conv1d(channels, channels, 1),
        ]

    def __call__(self, x):
        residual = x
        for layer in self.layers:
            residual = layer(residual)
        return x + residual


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layers = [
            Snake(in_channels),
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(out_channels, 1),
            ResidualUnit(out_channels, 3),
            ResidualUnit(out_channels, 9),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OobleckDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config["channels"]
        multiples = [1] + list(config["c_mults"])
        strides = config["strides"]
        layers = [nn.Conv1d(config["latent_dim"], multiples[-1] * channels, 7, padding=3)]
        for index in range(len(multiples) - 1, 0, -1):
            layers.append(
                DecoderBlock(
                    multiples[index] * channels,
                    multiples[index - 1] * channels,
                    strides[index - 1],
                )
            )
        layers += [
            Snake(multiples[0] * channels),
            nn.Conv1d(multiples[0] * channels, config["out_channels"], 7, padding=3, bias=False),
        ]
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def decode_tiled(self, latents, core=1024, halo=16):
        """``[T,64]`` latent → ``[samples,2]`` float32 波形。"""
        frames = latents.shape[0]
        total = audio_samples(frames)
        output = []
        for start in range(0, frames, core):
            end = min(frames, start + core)
            left, right = max(0, start - halo), min(frames, end + halo)
            tile = self(latents[None, left:right])[0]
            crop = (start - left) * FRAME
            output.append(tile[crop : crop + min(end * FRAME, total) - start * FRAME])
            mx.eval(output[-1])
        return mx.concatenate(output)


def load_vae(path: Path) -> OobleckDecoder:
    """从一个 YuE2 精度变体目录加载 VAE（各精度变体共用同款 VAE）。"""
    path = Path(path)
    config = json.loads((path / "vae_config.json").read_text())
    decoder = OobleckDecoder(config["decoder_config"])
    decoder.load_weights(str(path / "vae.safetensors"))
    mx.eval(decoder.parameters())
    return decoder