import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_vae.common.conv3d import CausalConv3d, MemoryState
from mflux.models.seedvr2.model.seedvr2_vae.decoder.decoder_3d import Decoder3D
from mflux.models.seedvr2.model.seedvr2_vae.encoder.encoder_3d import Encoder3D


class SeedVR2VAE(nn.Module):
    scaling_factor: float = 0.9152
    spatial_scale = 8

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        block_out_channels: tuple = (128, 256, 512, 512),
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.use_slicing = False
        self.slicing_sample_min_size = 4
        self.slicing_latent_min_size = 1

        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=2,
            temporal_down_blocks=2,
        )

        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=3,
            temporal_up_blocks=2,
        )

    def encode(self, x: mx.array) -> mx.array:
        x = x[:, :, None, :, :] if x.ndim == 4 else x
        h = self._encode_with_slicing(x)
        mean, _ = mx.split(h, 2, axis=1)
        latent = mean
        latent_scaled = latent * self.scaling_factor
        return latent_scaled

    def decode(self, z: mx.array) -> mx.array:
        z = z[:, :, None, :, :] if z.ndim == 4 else z
        z = z / self.scaling_factor
        decoded = self._decode_with_slicing(z)
        return decoded

    def set_causal_slicing(self, *, split_size: int | None) -> None:
        if split_size is None:
            self.use_slicing = False
            return
        if split_size <= 0:
            raise ValueError("split_size must be greater than zero.")
        self.use_slicing = True
        self.slicing_sample_min_size = split_size
        self.slicing_latent_min_size = max(1, split_size // 4)

    def _reset_causal_memories(self) -> None:
        for module in self.modules():
            if isinstance(module, CausalConv3d):
                module.memory = None

    def _encode_with_slicing(self, x: mx.array) -> mx.array:
        if not self.use_slicing or x.shape[2] <= self.slicing_sample_min_size + 1:
            self._reset_causal_memories()
            out = self.encoder(x, memory_state=MemoryState.DISABLED)
            self._reset_causal_memories()
            return out

        slices = []
        x_slices = []
        start = 1
        while start < x.shape[2]:
            end = min(x.shape[2], start + self.slicing_sample_min_size)
            x_slices.append(x[:, :, start:end, :, :])
            start = end

        self._reset_causal_memories()
        first_slice = mx.concatenate([x[:, :, :1, :, :], x_slices[0]], axis=2)
        slices.append(self.encoder(first_slice, memory_state=MemoryState.INITIALIZING))
        slices.extend(self.encoder(chunk, memory_state=MemoryState.ACTIVE) for chunk in x_slices[1:])
        self._reset_causal_memories()
        return mx.concatenate(slices, axis=2)

    def _decode_with_slicing(self, z: mx.array) -> mx.array:
        if not self.use_slicing or z.shape[2] <= self.slicing_latent_min_size + 1:
            self._reset_causal_memories()
            out = self.decoder(z, memory_state=MemoryState.DISABLED)
            self._reset_causal_memories()
            return out

        slices = []
        z_slices = []
        start = 1
        while start < z.shape[2]:
            end = min(z.shape[2], start + self.slicing_latent_min_size)
            z_slices.append(z[:, :, start:end, :, :])
            start = end

        self._reset_causal_memories()
        first_slice = mx.concatenate([z[:, :, :1, :, :], z_slices[0]], axis=2)
        slices.append(self.decoder(first_slice, memory_state=MemoryState.INITIALIZING))
        slices.extend(self.decoder(chunk, memory_state=MemoryState.ACTIVE) for chunk in z_slices[1:])
        self._reset_causal_memories()
        return mx.concatenate(slices, axis=2)
