# Ported from diffusers' AutoencoderKLQwenImage21 (Apache-2.0).
"""Qwen-Image 2.1 single-frame RGBA VAE.

The released image VAE is expressed as a video VAE, but its causal convolution
specialization folds the singleton temporal axis away and executes Conv2d.  Both
encoder and decoder are kept because the unified model uses normalized VAE
latents as reference-image tokens during editing.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn


def _rms_norm_channels(x: mx.array, gamma: mx.array, eps: float = 1e-12) -> mx.array:
    dtype = x.dtype
    value = x.astype(mx.float32)
    norm = mx.sqrt(mx.sum(mx.square(value), axis=-1, keepdims=True))
    value = value / mx.maximum(norm, mx.array(eps, dtype=mx.float32))
    value = value * math.sqrt(int(gamma.shape[0]))
    return (value * gamma.reshape(1, 1, 1, -1).astype(mx.float32)).astype(dtype)


class QwenImage21RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm_channels(x, self.gamma)


class QwenImage21ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = QwenImage21RMSNorm(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImage21RMSNorm(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else None

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        x = self.conv1(nn.silu(self.norm1(x)))
        x = self.conv2(nn.silu(self.norm2(x)))
        return x + shortcut


class QwenImage21AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = QwenImage21RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        batch, height, width, channels = x.shape
        qkv = self.to_qkv(self.norm(x)).reshape(batch, height * width, channels * 3)
        query, key, value = mx.split(qkv, 3, axis=-1)
        attended = mx.fast.scaled_dot_product_attention(
            query[:, None], key[:, None], value[:, None], scale=1.0 / math.sqrt(channels)
        )
        attended = attended[:, 0].reshape(batch, height, width, channels)
        return identity + self.proj(attended)


class QwenImage21MidBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.resnets = [QwenImage21ResidualBlock(dim, dim), QwenImage21ResidualBlock(dim, dim)]
        self.attentions = [QwenImage21AttentionBlock(dim)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        return self.resnets[1](self.attentions[0](x))


class QwenImage21SpatialDownsampler(nn.Module):
    """Official asymmetric right/bottom padding followed by stride-2 convolution."""

    def __init__(self, dim: int):
        super().__init__()
        # Keep the numeric path used by ``downsampler.resample.1.*`` in the checkpoint.
        self.resample = [nn.Identity(), nn.Conv2d(dim, dim, 3, stride=2)]

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
        return self.resample[1](x)


class QwenImage21AvgDown:
    """Parameter-free residual shortcut from the released residual encoder."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int):
        factor = int(factor_t) * int(factor_s) * int(factor_s)
        if in_channels * factor % out_channels:
            raise ValueError("Qwen-Image 2.1 AvgDown channel ratio is not integral")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.factor_t = int(factor_t)
        self.factor_s = int(factor_s)
        self.group_size = in_channels * factor // out_channels

    def __call__(self, x: mx.array) -> mx.array:
        batch, height, width, channels = x.shape
        factor = self.factor_s
        if height % factor or width % factor:
            raise ValueError(f"Qwen-Image 2.1 encoder 输入 latent 网格不可整除: {width}×{height}")
        patches = x.reshape(
            batch, height // factor, factor, width // factor, factor, channels
        ).transpose(0, 1, 3, 5, 2, 4)
        # The first temporal chunk is left-padded with zeros before averaging.  Images have a
        # singleton temporal axis, so put their spatial patches in the final temporal slot.
        temporal = mx.zeros(
            (*patches.shape[:4], self.factor_t, factor, factor), dtype=x.dtype
        )
        temporal[..., -1, :, :] = patches
        grouped = temporal.reshape(
            batch, height // factor, width // factor, self.out_channels, self.group_size
        )
        return grouped.mean(axis=-1)


class QwenImage21ResidualDownBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temporal_downsample: bool,
        down_flag: bool,
    ):
        super().__init__()
        current = in_dim
        self.resnets = []
        for _ in range(num_res_blocks):
            self.resnets.append(QwenImage21ResidualBlock(current, out_dim))
            current = out_dim
        self.downsampler = QwenImage21SpatialDownsampler(out_dim) if down_flag else None
        self.avg_shortcut = QwenImage21AvgDown(
            in_dim,
            out_dim,
            2 if temporal_downsample and down_flag else 1,
            2 if down_flag else 1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + self.avg_shortcut(shortcut)


class QwenImage21Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 96,
        z_dim: int = 128,
        dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        temporal_downsample: tuple[bool, ...] = (False, True, True, True),
    ):
        super().__init__()
        dims = [dim * value for value in (1, *dim_mult)]
        self.conv_in = nn.Conv2d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            down_flag = index != len(dim_mult) - 1
            self.down_blocks.append(
                QwenImage21ResidualDownBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks,
                    bool(temporal_downsample[index]) if down_flag else False,
                    down_flag,
                )
            )
        self.mid_block = QwenImage21MidBlock(dims[-1])
        self.norm_out = QwenImage21RMSNorm(dims[-1])
        self.conv_out = nn.Conv2d(dims[-1], z_dim, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class QwenImage21SpatialUpsampler(nn.Module):
    """Nearest 2x spatial upsample followed by the checkpoint's 3x3 convolution."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        # Keep the numeric ``.1`` path so diffusers' ``resample.1.*`` keys map directly.
        self.resample = [nn.Identity(), nn.Conv2d(dim, out_dim, 3, padding=1)]

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        return self.resample[1](x)


class QwenImage21DupUp:
    """Parameter-free residual shortcut used by the released residual decoder."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 2):
        factor = factor_t * factor_s * factor_s
        if out_channels * factor % in_channels:
            raise ValueError("Qwen-Image 2.1 DupUp channel ratio is not integral")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.repeats = out_channels * factor // in_channels

    def __call__(self, x: mx.array, *, first_chunk: bool = True) -> mx.array:
        batch, height, width, _ = x.shape
        expanded = mx.repeat(x, self.repeats, axis=-1).reshape(
            batch,
            height,
            width,
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
        )
        # A generated image has one temporal chunk. DupUp creates factor_t frames and
        # ``first_chunk`` keeps only the last one, exactly as diffusers does.
        temporal = expanded[..., -1 if first_chunk else 0, :, :]
        return temporal.transpose(0, 1, 4, 2, 5, 3).reshape(
            batch, height * self.factor_s, width * self.factor_s, self.out_channels
        )


class QwenImage21ResidualUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temporal_upsample: bool,
        up_flag: bool,
    ):
        super().__init__()
        current = in_dim
        self.resnets = []
        for _ in range(num_res_blocks + 1):
            self.resnets.append(QwenImage21ResidualBlock(current, out_dim))
            current = out_dim
        self.upsampler = QwenImage21SpatialUpsampler(out_dim, out_dim) if up_flag else None
        self.avg_shortcut = (
            QwenImage21DupUp(in_dim, out_dim, 2 if temporal_upsample else 1) if up_flag else None
        )

    def __call__(self, x: mx.array, *, first_chunk: bool = True) -> mx.array:
        shortcut = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsampler is not None:
            x = self.upsampler(x)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(shortcut, first_chunk=first_chunk)
        return x


class QwenImage21Decoder(nn.Module):
    def __init__(
        self,
        dim: int = 144,
        z_dim: int = 64,
        dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        temporal_upsample: tuple[bool, ...] = (True, True, True, False),
        out_channels: int = 4,
    ):
        super().__init__()
        dims = [dim * value for value in (dim_mult[-1], *reversed(dim_mult))]
        self.conv_in = nn.Conv2d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImage21MidBlock(dims[0])
        self.up_blocks = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            up_flag = index != len(dim_mult) - 1
            self.up_blocks.append(
                QwenImage21ResidualUpBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks,
                    bool(temporal_upsample[index]) if up_flag else False,
                    up_flag,
                )
            )
        self.norm_out = QwenImage21RMSNorm(dims[-1])
        self.conv_out = nn.Conv2d(dims[-1], out_channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x, first_chunk=True)
        return self.conv_out(nn.silu(self.norm_out(x)))


class QwenImage21VAE(nn.Module):
    """RGBA VAE with deterministic reference encoding and normalized-latent decode."""

    def __init__(
        self,
        base_dim: int = 96,
        decoder_base_dim: int = 144,
        z_dim: int = 64,
        dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        temperal_downsample: tuple[bool, ...] = (False, True, True, True),
        out_channels: int = 4,
        in_channels: int = 4,
        latents_mean: tuple[float, ...] | list[float] = (),
        latents_std: tuple[float, ...] | list[float] = (),
        **_unused,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = QwenImage21Encoder(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=tuple(dim_mult),
            num_res_blocks=num_res_blocks,
            temporal_downsample=tuple(temperal_downsample),
        )
        self.quant_conv = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = QwenImage21Decoder(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=tuple(dim_mult),
            num_res_blocks=num_res_blocks,
            temporal_upsample=tuple(reversed(tuple(temperal_downsample))),
            out_channels=out_channels,
        )
        # Keep normalization constants as Python data; MLX treats every array attached to a
        # Module as a trainable parameter and would otherwise require non-existent checkpoint keys.
        self.latents_mean = tuple(float(value) for value in (latents_mean or [0.0] * z_dim))
        self.latents_std = tuple(float(value) for value in (latents_std or [1.0] * z_dim))

    def encode(self, images: mx.array) -> mx.array:
        """Encode ``[B,H,W,4]`` pixels in ``[-1,1]`` to normalized ``[B,64,H/16,W/16]``."""
        if images.ndim != 4 or images.shape[-1] != 4:
            raise ValueError(
                f"Qwen-Image 2.1 VAE encode 需要 [B,H,W,4] RGBA，收到 {tuple(images.shape)}"
            )
        if images.shape[1] % 16 or images.shape[2] % 16:
            raise ValueError("Qwen-Image 2.1 VAE encode 的宽高必须是 16 的倍数")
        moments = self.quant_conv(self.encoder(images.astype(self.encoder.conv_in.weight.dtype)))
        # Official edit pipeline uses posterior.mode()/argmax: the first half is the mean.
        latents = moments[..., : self.z_dim].transpose(0, 3, 1, 2)
        mean = mx.array(self.latents_mean, dtype=mx.float32).reshape(1, -1, 1, 1)
        std = mx.array(self.latents_std, dtype=mx.float32).reshape(1, -1, 1, 1)
        normalized = ((latents.astype(mx.float32) - mean) / std).astype(latents.dtype)
        mx.eval(normalized)
        return normalized

    def decode(self, latents: mx.array) -> mx.array:
        if latents.ndim != 5 or latents.shape[1] != self.z_dim or latents.shape[2] != 1:
            raise ValueError(
                f"Qwen-Image 2.1 VAE 需要 [B,{self.z_dim},1,H,W]，收到 {tuple(latents.shape)}"
            )
        # Official pipeline denormalizes before calling the VAE. Keep that contract inside
        # this component so callers cannot accidentally omit the 64-channel statistics.
        value = latents.astype(mx.float32)
        mean = mx.array(self.latents_mean, dtype=mx.float32).reshape(1, -1, 1, 1, 1)
        std = mx.array(self.latents_std, dtype=mx.float32).reshape(1, -1, 1, 1, 1)
        value = (value * std + mean).astype(self.post_quant_conv.weight.dtype)
        value = value[:, :, 0].transpose(0, 2, 3, 1)
        decoded = self.decoder(self.post_quant_conv(value))
        decoded = mx.clip(decoded, -1.0, 1.0)
        return decoded.transpose(0, 3, 1, 2)[:, :, None]