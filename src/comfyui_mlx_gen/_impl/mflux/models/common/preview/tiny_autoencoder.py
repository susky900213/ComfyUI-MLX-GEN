"""MLX port of the TAESD-family tiny autoencoder (diffusers `AutoencoderTiny`).

Structure, layer order, and attribute names follow
`diffusers/models/autoencoders/autoencoder_tiny.py` and its `EncoderTiny`/`DecoderTiny`
in `diffusers/models/autoencoders/vae.py`, so published checkpoints map key-for-key.

The MLX deviation is layout only: convolutions run channels-last, so the tensor is
transposed once on entry and once on exit instead of around every convolution.
"""

import mlx.core as mx
from mlx import nn

CLAMP_MAGNITUDE = 3.0


class AutoencoderTinyBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_midblock_gn: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.fuse = nn.ReLU()
        # The FLUX.2 variant adds a global channel-mixing branch before the residual.
        # GroupNorm is pytorch-compatible so grouping matches the checkpoint's NCHW layout.
        self.pool = (
            nn.Sequential(
                nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=4, dims=4 * in_channels, pytorch_compatible=True),
                nn.ReLU(),
                nn.Conv2d(4 * in_channels, in_channels, kernel_size=1, bias=False),
            )
            if use_midblock_gn
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))


class EncoderTiny(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: tuple[int, ...],
        block_out_channels: tuple[int, ...],
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for i, num_block in enumerate(num_blocks):
            num_channels = block_out_channels[i]
            if i == 0:
                layers.append(nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=2, bias=False))
            layers.extend(AutoencoderTinyBlock(num_channels, num_channels) for _ in range(num_block))
        layers.append(nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        # Upstream maps the [-1, 1] image convention to [0, 1] before the stack.
        return self.layers(x * 0.5 + 0.5)


class DecoderTiny(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: tuple[int, ...],
        block_out_channels: tuple[int, ...],
        upsampling_scaling_factor: int,
        use_midblock_gn: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        for i, num_block in enumerate(num_blocks):
            is_final_block = i == (len(num_blocks) - 1)
            num_channels = block_out_channels[i]
            # Upstream applies the mid-block branch to the deepest group only.
            midblock_gn = use_midblock_gn and i == 0
            layers.extend(
                AutoencoderTinyBlock(num_channels, num_channels, use_midblock_gn=midblock_gn) for _ in range(num_block)
            )
            if not is_final_block:
                layers.append(nn.Upsample(scale_factor=upsampling_scaling_factor, mode="nearest"))
            conv_out_channel = num_channels if not is_final_block else out_channels
            layers.append(
                nn.Conv2d(
                    num_channels,
                    conv_out_channel,
                    kernel_size=3,
                    padding=1,
                    bias=is_final_block,
                )
            )
        self.layers = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.tanh(x / CLAMP_MAGNITUDE) * CLAMP_MAGNITUDE
        x = self.layers(x)
        # Upstream returns the [-1, 1] image convention.
        return x * 2.0 - 1.0


class TinyAutoencoder(nn.Module):
    """Config-driven tiny autoencoder; defaults reproduce the reference TAESD graph."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 4,
        encoder_block_out_channels: tuple[int, ...] = (64, 64, 64, 64),
        decoder_block_out_channels: tuple[int, ...] = (64, 64, 64, 64),
        num_encoder_blocks: tuple[int, ...] = (1, 3, 3, 3),
        num_decoder_blocks: tuple[int, ...] = (3, 3, 3, 1),
        upsampling_scaling_factor: int = 2,
        scaling_factor: float = 1.0,
        shift_factor: float = 0.0,
        use_midblock_gn: bool = False,
        with_encoder: bool = False,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.spatial_scale = upsampling_scaling_factor ** (len(decoder_block_out_channels) - 1)
        self.decoder = DecoderTiny(
            in_channels=latent_channels,
            out_channels=out_channels,
            num_blocks=num_decoder_blocks,
            block_out_channels=decoder_block_out_channels,
            upsampling_scaling_factor=upsampling_scaling_factor,
            use_midblock_gn=use_midblock_gn,
        )
        self.encoder = (
            EncoderTiny(
                in_channels=in_channels,
                out_channels=latent_channels,
                num_blocks=num_encoder_blocks,
                block_out_channels=encoder_block_out_channels,
            )
            if with_encoder
            else None
        )

    def decode(self, latents: mx.array) -> mx.array:
        """Decode pipeline-space latents (B, C, H, W) or (B, C, 1, H, W) to (B, 3, 1, H*s, W*s) in [-1, 1]."""
        had_temporal_axis = latents.ndim == 5
        if had_temporal_axis:
            # Refuse multi-frame input rather than silently previewing only the first frame:
            # the image tiny autoencoders have no temporal modelling (video needs a TAEHV variant).
            if latents.shape[2] != 1:
                raise ValueError(f"This tiny autoencoder decodes single frames; got {latents.shape[2]} temporal steps.")
            latents = latents[:, :, 0, :, :]
        latents = latents / self.scaling_factor + self.shift_factor
        decoded = self.decoder(mx.transpose(latents, (0, 2, 3, 1)))
        decoded = mx.transpose(decoded, (0, 3, 1, 2))
        return decoded[:, :, None, :, :] if had_temporal_axis else decoded

    def encode(self, image: mx.array) -> mx.array:
        if self.encoder is None:
            raise ValueError("This tiny autoencoder was loaded without its encoder.")
        had_temporal_axis = image.ndim == 5
        if had_temporal_axis:
            if image.shape[2] != 1:
                raise ValueError(f"This tiny autoencoder encodes single frames; got {image.shape[2]} temporal steps.")
            image = image[:, :, 0, :, :]
        encoded = self.encoder(mx.transpose(image, (0, 2, 3, 1)))
        encoded = mx.transpose(encoded, (0, 3, 1, 2))
        latents = (encoded - self.shift_factor) * self.scaling_factor
        return latents[:, :, None, :, :] if had_temporal_axis else latents
