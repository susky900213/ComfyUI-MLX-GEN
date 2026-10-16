"""FLUX 3 latent packing (Flux VAE 32 channels → 128-channel tokens)."""

from __future__ import annotations

import mlx.core as mx


class Flux3LatentCreator:
    VAE_SCALE_FACTOR = 8
    PACK_FACTOR = 2
    TOKEN_CHANNELS = 128

    @classmethod
    def patchify_latents(cls, latents: mx.array) -> mx.array:
        if latents.ndim == 5 and latents.shape[2] == 1:
            latents = latents[:, :, 0]
        if latents.ndim != 4 or latents.shape[1] != 32:
            raise ValueError(f"FLUX 3 VAE latent 应为 [B,32,H,W]，收到 {tuple(latents.shape)}")
        batch, channels, height, width = latents.shape
        if height % 2 or width % 2:
            raise ValueError("FLUX 3 VAE latent 的空间尺寸必须是偶数")
        value = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
        value = value.transpose(0, 1, 3, 5, 2, 4)
        return value.reshape(batch, channels * 4, height // 2, width // 2)

    @staticmethod
    def pack_latents(latents: mx.array) -> mx.array:
        batch, channels, height, width = latents.shape
        return latents.reshape(batch, channels, height * width).transpose(0, 2, 1)

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        if latents.ndim == 2:
            latents = latents[None]
        if latents.ndim != 3:
            raise ValueError(f"FLUX 3 packed latent 应为 [B,seq,128]，收到 {tuple(latents.shape)}")
        latent_height, latent_width = int(height) // 16, int(width) // 16
        expected = latent_height * latent_width
        if int(latents.shape[1]) != expected:
            raise ValueError(
                f"FLUX 3 packed latent seq={latents.shape[1]}，但 {width}×{height} 需要 {expected}"
            )
        return latents.reshape(latents.shape[0], latent_height, latent_width, 128).transpose(0, 3, 1, 2)

    @staticmethod
    def prepare_grid_ids(latents: mx.array, *, t_coord: int = 0, layer_coord: int = 0) -> mx.array:
        if latents.ndim == 4:
            batch, _, height, width = latents.shape
        elif latents.ndim == 3:
            batch, tokens, _ = latents.shape
            height = int(tokens**0.5)
            width = tokens // max(1, height)
        else:
            raise ValueError("FLUX 3 ids 的 latent 维度不正确")
        rows = mx.broadcast_to(mx.arange(height)[:, None], (height, width)).reshape(-1)
        cols = mx.broadcast_to(mx.arange(width)[None, :], (height, width)).reshape(-1)
        ids = mx.stack(
            [mx.full(rows.shape, t_coord), rows.astype(mx.float32) * 0.5,
             cols.astype(mx.float32) * 0.5, mx.full(rows.shape, layer_coord)], axis=-1
        )
        return mx.broadcast_to(ids[None], (batch, ids.shape[0], 4))

    @staticmethod
    def grid_ids(batch: int, height: int, width: int, *, t_coord: int = 0, layer_coord: int = 0) -> mx.array:
        rows = mx.broadcast_to(mx.arange(height)[:, None], (height, width)).reshape(-1)
        cols = mx.broadcast_to(mx.arange(width)[None, :], (height, width)).reshape(-1)
        ids = mx.stack(
            [mx.full(rows.shape, t_coord), rows.astype(mx.float32) * 0.5,
             cols.astype(mx.float32) * 0.5, mx.full(rows.shape, layer_coord)], axis=-1
        )
        return mx.broadcast_to(ids[None], (batch, ids.shape[0], 4))

    @classmethod
    def prepare_packed_latents(cls, seed: int, height: int, width: int, batch_size: int = 1):
        if height % 16 or width % 16:
            raise ValueError("FLUX 3 的宽高必须是 16 的倍数")
        latent_h, latent_w = height // 16, width // 16
        noise = mx.random.normal((batch_size, latent_h * latent_w, 128), key=mx.random.key(seed)).astype(mx.bfloat16)
        ids = cls.grid_ids(batch_size, latent_h, latent_w, t_coord=0)
        return noise, ids, latent_h, latent_w
