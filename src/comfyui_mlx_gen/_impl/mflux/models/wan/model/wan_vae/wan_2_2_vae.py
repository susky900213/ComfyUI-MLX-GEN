import gc
from collections.abc import Iterator

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.fibo.model.fibo_vae.common.wan_2_2_causal_conv_3d import Wan2_2_CausalConv3d
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_mid_block import Wan2_2_MidBlock
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_resample import Wan2_2_Resample
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_residual_block import Wan2_2_ResidualBlock
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_rms_norm import Wan2_2_RMSNorm
from mflux.models.fibo.model.fibo_vae.decoder.wan_2_2_decoder_3d import Wan2_2_Decoder3d
from mflux.models.fibo.model.fibo_vae.encoder.wan_2_2_encoder_3d import Wan2_2_Encoder3d


class Wan2_1_Encoder3d(nn.Module):
    supports_cache = True

    def __init__(
        self,
        in_channels: int,
        dim: int,
        z_dim: int,
        dim_mult: list[int],
        num_res_blocks: int,
        attn_scales: list[float],
        temporal_downsample: list[bool],
        non_linearity: str = "silu",
    ):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        self.conv_in = Wan2_2_CausalConv3d(in_channels, dims[0], 3, padding=1)
        self.down_blocks: list[nn.Module] = []
        scale = 1.0
        for block_index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            current_dim = in_dim
            for _ in range(num_res_blocks):
                self.down_blocks.append(Wan2_2_ResidualBlock(current_dim, out_dim, non_linearity))
                current_dim = out_dim
            if scale in attn_scales:
                raise NotImplementedError("Wan2.1 VAE attention down blocks are not implemented in MLX-Gen.")
            if block_index != len(dim_mult) - 1:
                mode = "downsample3d" if temporal_downsample[block_index] else "downsample2d"
                self.down_blocks.append(Wan2_2_Resample(out_dim, mode=mode))
                scale /= 2.0
        self.mid_block = Wan2_2_MidBlock(out_dim, non_linearity, num_layers=1)
        self.norm_out = Wan2_2_RMSNorm(out_dim, images=False)
        self.conv_out = Wan2_2_CausalConv3d(out_dim, z_dim, 3, padding=1)

    def __call__(
        self,
        x: mx.array,
        feat_cache: list[mx.array | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> mx.array:
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = Wan2_2_VAE.cache_slice(x, feat_cache[idx])
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = self.norm_out(x)
        x = nn.silu(x)
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = Wan2_2_VAE.cache_slice(x, feat_cache[idx])
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
            return x
        return self.conv_out(x)


class Wan2_1_UpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.resnets: list[Wan2_2_ResidualBlock] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(Wan2_2_ResidualBlock(current_dim, out_dim, non_linearity))
            current_dim = out_dim
        self.upsamplers: list[Wan2_2_Resample] = []
        if upsample_mode is not None:
            self.upsamplers.append(Wan2_2_Resample(out_dim, mode=upsample_mode))

    def __call__(
        self,
        x: mx.array,
        block_idx: int | None = None,
        first_chunk: bool = False,
        feat_cache: list[mx.array | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> mx.array:
        del block_idx, first_chunk
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for upsampler in self.upsamplers:
            x = upsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x


class Wan2_1_Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult: list[int],
        num_res_blocks: int,
        temporal_upsample: list[bool],
        non_linearity: str = "silu",
        out_channels: int = 3,
    ):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = Wan2_2_CausalConv3d(z_dim, dims[0], 3, padding=1, name="decoder_conv_in")
        self.mid_block = Wan2_2_MidBlock(dims[0], non_linearity, num_layers=1)
        self.up_blocks: list[Wan2_1_UpBlock] = []
        for block_index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            effective_in_dim = in_dim if block_index == 0 else in_dim // 2
            up_flag = block_index != len(dim_mult) - 1
            upsample_mode = None
            if up_flag:
                upsample_mode = "upsample3d" if temporal_upsample[block_index] else "upsample2d"
            self.up_blocks.append(
                Wan2_1_UpBlock(
                    in_dim=effective_in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            )
        self.norm_out = Wan2_2_RMSNorm(out_dim, images=False)
        self.conv_out = Wan2_2_CausalConv3d(out_dim, out_channels, 3, padding=1, name="decoder_conv_out")

    def __call__(
        self,
        x: mx.array,
        feat_cache: list[mx.array | str | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> mx.array:
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = Wan2_2_VAE.cache_slice(x, feat_cache[idx])
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, block_idx=i, first_chunk=first_chunk, feat_cache=feat_cache, feat_idx=feat_idx)
        x = self.norm_out(x)
        x = nn.silu(x)
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = Wan2_2_VAE.cache_slice(x, feat_cache[idx])
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class Wan2_2_VAE(nn.Module):
    COMPACT_FEATURE_CACHE_POLICY_ID = "wan-compact-feature-cache-v1"
    SPATIAL_TILING_POLICY_ID = "wan21-diffusers-0.35.2-256x256-stride192-v1"
    TILED_DECODE_MODE = "bounded_tile_major_spatial_vae"
    Z_DIM = 48
    ENCODER_BASE_DIM = 160
    DECODER_BASE_DIM = 256
    DIM_MULT = [1, 2, 4, 4]
    NUM_RES_BLOCKS = 2
    OUT_CHANNELS = 12
    PATCH_SIZE = 2
    SPATIAL_SCALE = 16
    TEMPORAL_SCALE = 4
    TILE_SAMPLE_MIN_HEIGHT = 256
    TILE_SAMPLE_MIN_WIDTH = 256
    TILE_SAMPLE_STRIDE_HEIGHT = 192
    TILE_SAMPLE_STRIDE_WIDTH = 192
    LATENTS_MEAN = np.array([-0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813, 0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667], dtype=np.float32)  # fmt: off
    LATENTS_STD = np.array([0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659, 0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744], dtype=np.float32)  # fmt: off
    WAN21_LATENTS_MEAN = np.array([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921], dtype=np.float32)  # fmt: off
    WAN21_LATENTS_STD = np.array([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160], dtype=np.float32)  # fmt: off

    def __init__(
        self,
        base_dim: int = ENCODER_BASE_DIM,
        decoder_base_dim: int | None = DECODER_BASE_DIM,
        z_dim: int = Z_DIM,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = NUM_RES_BLOCKS,
        attn_scales: list[float] | None = None,
        temporal_downsample: list[bool] | None = None,
        in_channels: int = OUT_CHANNELS,
        out_channels: int = OUT_CHANNELS,
        patch_size: int = PATCH_SIZE,
        scale_factor_spatial: int = SPATIAL_SCALE,
        scale_factor_temporal: int = TEMPORAL_SCALE,
        is_residual: bool = True,
        latents_mean: list[float] | np.ndarray | None = None,
        latents_std: list[float] | np.ndarray | None = None,
    ):
        super().__init__()
        dim_mult = dim_mult or self.DIM_MULT
        attn_scales = attn_scales or []
        temporal_downsample = temporal_downsample or [False, True, True]
        decoder_base_dim = base_dim if decoder_base_dim is None else decoder_base_dim
        self.z_dim = z_dim
        self.patch_size = patch_size
        self.spatial_scale = scale_factor_spatial
        self.temporal_scale = scale_factor_temporal
        self.tile_sample_min_height = self.TILE_SAMPLE_MIN_HEIGHT
        self.tile_sample_min_width = self.TILE_SAMPLE_MIN_WIDTH
        self.tile_sample_stride_height = self.TILE_SAMPLE_STRIDE_HEIGHT
        self.tile_sample_stride_width = self.TILE_SAMPLE_STRIDE_WIDTH
        self.latents_mean = np.array(
            latents_mean if latents_mean is not None else self._default_mean(z_dim), dtype=np.float32
        )
        self.latents_std = np.array(
            latents_std if latents_std is not None else self._default_std(z_dim), dtype=np.float32
        )
        if is_residual:
            self.encoder = Wan2_2_Encoder3d(
                in_channels=in_channels,
                dim=base_dim,
                z_dim=z_dim * 2,
                dim_mult=dim_mult,
                num_res_blocks=num_res_blocks,
                attn_scales=attn_scales,
                temporal_downsample=temporal_downsample,
            )
            self.decoder = Wan2_2_Decoder3d(
                dim=decoder_base_dim,
                z_dim=z_dim,
                dim_mult=dim_mult,
                num_res_blocks=num_res_blocks,
                temporal_upsample=temporal_downsample[::-1],
                out_channels=out_channels,
            )
        else:
            self.encoder = Wan2_1_Encoder3d(
                in_channels=in_channels,
                dim=base_dim,
                z_dim=z_dim * 2,
                dim_mult=dim_mult,
                num_res_blocks=num_res_blocks,
                attn_scales=attn_scales,
                temporal_downsample=temporal_downsample,
            )
            self.decoder = Wan2_1_Decoder3d(
                dim=decoder_base_dim,
                z_dim=z_dim,
                dim_mult=dim_mult,
                num_res_blocks=num_res_blocks,
                temporal_upsample=temporal_downsample[::-1],
                out_channels=out_channels,
            )
        self.quant_conv = Wan2_2_CausalConv3d(z_dim * 2, z_dim * 2, 1, padding=0, name="quant_conv")
        self.post_quant_conv = Wan2_2_CausalConv3d(z_dim, z_dim, 1, padding=0, name="post_quant_conv")

    def encode(
        self,
        images: mx.array,
        *,
        clear_cache_each_slice: bool = False,
        tile_spatial: bool = False,
    ) -> mx.array:
        if images.ndim == 4:
            images = images.reshape(images.shape[0], images.shape[1], 1, images.shape[2], images.shape[3])
        if images.ndim != 5:
            raise ValueError(f"Expected Wan VAE encode input with shape [B,C,F,H,W], got {images.shape}")

        sample_height, sample_width = images.shape[-2:]
        if tile_spatial:
            self._require_spatial_tiling_supported()
        images = self.patchify(images, patch_size=self.patch_size)
        if getattr(self.encoder, "supports_cache", False):
            if tile_spatial and self._should_tile_sample(height=sample_height, width=sample_width):
                encoded = self._encode_cached_tiled(images, clear_cache_each_slice=clear_cache_each_slice)
            else:
                encoded = self._encode_cached(images, clear_cache_each_slice=clear_cache_each_slice)
                encoded = self.quant_conv(encoded)
        else:
            encoded = self.encoder(images)
            encoded = self.quant_conv(encoded)
        return encoded[:, : self.z_dim]

    def encode_normalized(
        self,
        images: mx.array,
        *,
        clear_cache_each_slice: bool = False,
        tile_spatial: bool = False,
    ) -> mx.array:
        latents_mean = mx.array(self.latents_mean).reshape(1, self.z_dim, 1, 1, 1)
        latents_std = mx.array(self.latents_std).reshape(1, self.z_dim, 1, 1, 1)
        return (
            self.encode(
                images,
                clear_cache_each_slice=clear_cache_each_slice,
                tile_spatial=tile_spatial,
            )
            - latents_mean
        ) / latents_std

    def decode(
        self,
        latents: mx.array,
        *,
        clear_cache_each_slice: bool = False,
        tile_spatial: bool = False,
    ) -> mx.array:
        if latents.ndim == 4:
            latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])
        if tile_spatial:
            self._require_spatial_tiling_supported()
        if tile_spatial and self._should_tile_latent(height=latents.shape[-2], width=latents.shape[-1]):
            return mx.concatenate(
                list(
                    self.iter_decode_slices(
                        latents,
                        clear_cache_each_slice=clear_cache_each_slice,
                        tile_spatial=True,
                    )
                ),
                axis=2,
            )
        latents = self.post_quant_conv(latents)
        feat_cache = self._new_feature_cache()
        decoded_slices = []
        for frame_idx in range(latents.shape[2]):
            feat_idx = [0]
            decoded = self.decoder(
                latents[:, :, frame_idx : frame_idx + 1, :, :],
                feat_cache=feat_cache,
                feat_idx=feat_idx,
                first_chunk=frame_idx == 0,
            )
            if clear_cache_each_slice:
                self._materialize_feature_cache(decoded, feat_cache)
            else:
                mx.eval(decoded)
            decoded_slices.append(decoded)
            if clear_cache_each_slice:
                gc.collect()
                mx.synchronize()
                mx.clear_cache()
        decoded = mx.concatenate(decoded_slices, axis=2)
        decoded = self.unpatchify(decoded, patch_size=self.patch_size)
        return mx.clip(decoded, -1.0, 1.0)

    def decode_normalized_latents(
        self,
        latents: mx.array,
        *,
        clear_cache_each_slice: bool = False,
        tile_spatial: bool = False,
    ) -> mx.array:
        latents_mean = mx.array(self.latents_mean).reshape(1, self.z_dim, 1, 1, 1)
        latents_std = mx.array(self.latents_std).reshape(1, self.z_dim, 1, 1, 1)
        return self.decode(
            latents * latents_std + latents_mean,
            clear_cache_each_slice=clear_cache_each_slice,
            tile_spatial=tile_spatial,
        )

    def iter_decode_normalized_latent_slices(
        self,
        latents: mx.array,
        *,
        clear_cache_each_slice: bool = False,
        tile_spatial: bool = False,
    ) -> Iterator[mx.array]:
        latents_mean = mx.array(self.latents_mean).reshape(1, self.z_dim, 1, 1, 1)
        latents_std = mx.array(self.latents_std).reshape(1, self.z_dim, 1, 1, 1)
        yield from self.iter_decode_slices(
            latents * latents_std + latents_mean,
            clear_cache_each_slice=clear_cache_each_slice,
            tile_spatial=tile_spatial,
        )

    def iter_decode_slices(
        self,
        latents: mx.array,
        *,
        clear_cache_each_slice: bool = False,
        tile_spatial: bool = False,
    ) -> Iterator[mx.array]:
        if latents.ndim == 4:
            latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])
        if tile_spatial:
            self._require_spatial_tiling_supported()
        if tile_spatial and self._should_tile_latent(height=latents.shape[-2], width=latents.shape[-1]):
            yield from self._iter_decode_tiled_slices(
                latents,
                clear_cache_each_slice=clear_cache_each_slice,
            )
            return
        latents = self.post_quant_conv(latents)
        mx.eval(latents)
        feat_cache = self._new_feature_cache()
        for frame_idx in range(latents.shape[2]):
            feat_idx = [0]
            decoded = self.decoder(
                latents[:, :, frame_idx : frame_idx + 1, :, :],
                feat_cache=feat_cache,
                feat_idx=feat_idx,
                first_chunk=frame_idx == 0,
            )
            if frame_idx == 0 and decoded.shape[2] > 1:
                decoded = decoded[:, :, 1:, :, :]
            if clear_cache_each_slice:
                self._materialize_feature_cache(decoded, feat_cache)
            decoded = self.unpatchify(decoded, patch_size=self.patch_size)
            decoded = mx.clip(decoded, -1.0, 1.0)
            mx.eval(decoded)
            yield decoded
            del decoded
            if clear_cache_each_slice:
                gc.collect()
                mx.synchronize()
                mx.clear_cache()

    def _encode_cached(
        self,
        images: mx.array,
        *,
        clear_cache_each_slice: bool = False,
        quantize_slices: bool = False,
    ) -> mx.array:
        feat_cache = self._new_feature_cache()
        encoded_slices = []
        iterations = 1 + (images.shape[2] - 1) // self.temporal_scale
        for index in range(iterations):
            feat_idx = [0]
            if index == 0:
                chunk = images[:, :, :1]
            else:
                start = 1 + self.temporal_scale * (index - 1)
                chunk = images[:, :, start : start + self.temporal_scale]
            encoded = self.encoder(chunk, feat_cache=feat_cache, feat_idx=feat_idx)
            if quantize_slices:
                encoded = self.quant_conv(encoded)
            if clear_cache_each_slice:
                self._materialize_feature_cache(encoded, feat_cache)
                gc.collect()
                mx.synchronize()
                mx.clear_cache()
            else:
                mx.eval(encoded)
            encoded_slices.append(encoded)
        return mx.concatenate(encoded_slices, axis=2)

    def _encode_cached_tiled(self, images: mx.array, *, clear_cache_each_slice: bool) -> mx.array:
        input_height, input_width = images.shape[-2:]
        tile_height = self.tile_sample_min_height
        tile_width = self.tile_sample_min_width
        stride_height = self.tile_sample_stride_height
        stride_width = self.tile_sample_stride_width
        latent_height = input_height // self.spatial_scale
        latent_width = input_width // self.spatial_scale
        tile_latent_height = self.tile_sample_min_height // self.spatial_scale
        tile_latent_width = self.tile_sample_min_width // self.spatial_scale
        stride_latent_height = self.tile_sample_stride_height // self.spatial_scale
        stride_latent_width = self.tile_sample_stride_width // self.spatial_scale
        blend_height = tile_latent_height - stride_latent_height
        blend_width = tile_latent_width - stride_latent_width
        rows = []
        for top in range(0, input_height, stride_height):
            row = []
            for left in range(0, input_width, stride_width):
                tile = images[:, :, :, top : top + tile_height, left : left + tile_width]
                encoded = self._encode_cached(
                    tile,
                    clear_cache_each_slice=clear_cache_each_slice,
                    quantize_slices=True,
                )
                encoded = mx.contiguous(encoded)
                mx.eval(encoded)
                row.append(encoded)
                if clear_cache_each_slice:
                    gc.collect()
                    mx.synchronize()
                    mx.clear_cache()
            rows.append(row)

        result_rows = []
        for row_index, row in enumerate(rows):
            result_row = []
            for column_index, tile in enumerate(row):
                if row_index > 0:
                    tile = self._blend_vertical(rows[row_index - 1][column_index], tile, blend_height)
                if column_index > 0:
                    tile = self._blend_horizontal(row[column_index - 1], tile, blend_width)
                row[column_index] = tile
                result_row.append(tile[:, :, :, :stride_latent_height, :stride_latent_width])
            result_rows.append(mx.concatenate(result_row, axis=-1))
        encoded = mx.contiguous(mx.concatenate(result_rows, axis=-2)[:, :, :, :latent_height, :latent_width])
        mx.eval(encoded)
        del rows, result_rows, row, result_row, tile
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        return encoded

    def _iter_decode_tiled_slices(
        self,
        latents: mx.array,
        *,
        clear_cache_each_slice: bool,
    ) -> Iterator[mx.array]:
        latent_height, latent_width = latents.shape[-2:]
        tile_latent_height = self.tile_sample_min_height // self.spatial_scale
        tile_latent_width = self.tile_sample_min_width // self.spatial_scale
        stride_latent_height = self.tile_sample_stride_height // self.spatial_scale
        stride_latent_width = self.tile_sample_stride_width // self.spatial_scale
        stride_sample_height = self.tile_sample_stride_height
        stride_sample_width = self.tile_sample_stride_width
        sample_height = latent_height * self.spatial_scale
        sample_width = latent_width * self.spatial_scale
        blend_height = self.tile_sample_min_height - stride_sample_height
        blend_width = self.tile_sample_min_width - stride_sample_width
        row_positions = list(range(0, latent_height, stride_latent_height))
        column_positions = list(range(0, latent_width, stride_latent_width))
        rows = []
        for top in row_positions:
            row = []
            for left in column_positions:
                feat_cache = self._new_feature_cache()
                decoded_slices = []
                for frame_index in range(latents.shape[2]):
                    tile = latents[
                        :,
                        :,
                        frame_index : frame_index + 1,
                        top : top + tile_latent_height,
                        left : left + tile_latent_width,
                    ]
                    tile = self.post_quant_conv(tile)
                    decoded = self.decoder(
                        tile,
                        feat_cache=feat_cache,
                        feat_idx=[0],
                        first_chunk=frame_index == 0,
                    )
                    if clear_cache_each_slice:
                        self._materialize_feature_cache(decoded, feat_cache)
                    else:
                        mx.eval(decoded)
                    decoded_slices.append(decoded)
                    if clear_cache_each_slice:
                        gc.collect()
                        mx.synchronize()
                        mx.clear_cache()
                decoded_tile = mx.contiguous(mx.concatenate(decoded_slices, axis=2))
                mx.eval(decoded_tile)
                row.append(decoded_tile)
                del decoded_slices, feat_cache, decoded, tile
                gc.collect()
                mx.synchronize()
                mx.clear_cache()
            rows.append(row)

        result_rows = []
        for row_index, row in enumerate(rows):
            result_row = []
            for column_index, tile in enumerate(row):
                if row_index > 0:
                    tile = self._blend_vertical(rows[row_index - 1][column_index], tile, blend_height)
                if column_index > 0:
                    tile = self._blend_horizontal(row[column_index - 1], tile, blend_width)
                row[column_index] = tile
                result_row.append(tile[:, :, :, :stride_sample_height, :stride_sample_width])
            result_rows.append(mx.concatenate(result_row, axis=-1))
        decoded = mx.concatenate(result_rows, axis=-2)[:, :, :, :sample_height, :sample_width]
        decoded = self.unpatchify(decoded, patch_size=self.patch_size)
        decoded = mx.contiguous(mx.clip(decoded, -1.0, 1.0))
        mx.eval(decoded)
        del rows, result_rows, row, result_row, tile
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

        start = 0
        for frame_index in range(latents.shape[2]):
            frame_count = 1 if frame_index == 0 else self.temporal_scale
            decoded_slice = mx.contiguous(decoded[:, :, start : start + frame_count])
            mx.eval(decoded_slice)
            yield decoded_slice
            start += frame_count
            del decoded_slice
            if clear_cache_each_slice:
                gc.collect()
                mx.synchronize()
                mx.clear_cache()

    @staticmethod
    def _blend_vertical(above: mx.array, current: mx.array, extent: int) -> mx.array:
        extent = min(above.shape[-2], current.shape[-2], extent)
        if extent <= 0:
            return current
        weight = mx.arange(extent, dtype=mx.float32).reshape(1, 1, 1, extent, 1) / extent
        blended = above[:, :, :, -extent:, :] * (1 - weight) + current[:, :, :, :extent, :] * weight
        return mx.concatenate([blended.astype(current.dtype), current[:, :, :, extent:, :]], axis=-2)

    @staticmethod
    def _blend_horizontal(left: mx.array, current: mx.array, extent: int) -> mx.array:
        extent = min(left.shape[-1], current.shape[-1], extent)
        if extent <= 0:
            return current
        weight = mx.arange(extent, dtype=mx.float32).reshape(1, 1, 1, 1, extent) / extent
        blended = left[:, :, :, :, -extent:] * (1 - weight) + current[:, :, :, :, :extent] * weight
        return mx.concatenate([blended.astype(current.dtype), current[:, :, :, :, extent:]], axis=-1)

    def _should_tile_sample(self, *, height: int, width: int) -> bool:
        return height > self.tile_sample_min_height or width > self.tile_sample_min_width

    def _should_tile_latent(self, *, height: int, width: int) -> bool:
        return (
            height > self.tile_sample_min_height // self.spatial_scale
            or width > self.tile_sample_min_width // self.spatial_scale
        )

    def _require_spatial_tiling_supported(self) -> None:
        if self.patch_size != 1:
            raise ValueError(
                "Wan VAE spatial tiling is currently proven only for the Wan2.1 patch-size-1 VAE; "
                f"got patch_size={self.patch_size}."
            )
        if not getattr(self.encoder, "supports_cache", False):
            raise ValueError("Wan VAE spatial tiling requires the cached causal encoder/decoder implementation.")

    @staticmethod
    def _materialize_feature_cache(output: mx.array, feat_cache: list[mx.array | str | None]) -> None:
        for cache_index, entry in enumerate(feat_cache):
            if isinstance(entry, mx.array):
                feat_cache[cache_index] = mx.contiguous(entry)
        cache_tensors = [entry for entry in feat_cache if isinstance(entry, mx.array)]
        mx.eval(output, *cache_tensors)

    @staticmethod
    def cache_slice(x: mx.array, previous: mx.array | str | None) -> mx.array:
        cache_x = x[:, :, -2:, :, :]
        if cache_x.shape[2] < 2 and previous is not None and previous != "Rep":
            cache_x = mx.concatenate([previous[:, :, -1:, :, :], cache_x], axis=2)
        return cache_x

    @staticmethod
    def patchify(x: mx.array, patch_size: int) -> mx.array:
        if patch_size == 1:
            return x
        batch_size, channels, frames, height, width = x.shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(f"Height ({height}) and width ({width}) must be divisible by patch_size ({patch_size})")
        x = mx.reshape(
            x,
            (
                batch_size,
                channels,
                frames,
                height // patch_size,
                patch_size,
                width // patch_size,
                patch_size,
            ),
        )
        x = mx.transpose(x, (0, 1, 6, 4, 2, 3, 5))
        return mx.reshape(
            x,
            (batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size),
        )

    @staticmethod
    def unpatchify(x: mx.array, patch_size: int) -> mx.array:
        if patch_size == 1:
            return x
        batch_size, c_patches, frames, height, width = x.shape
        channels = c_patches // (patch_size * patch_size)
        x = mx.reshape(x, (batch_size, channels, patch_size, patch_size, frames, height, width))
        x = mx.transpose(x, (0, 1, 4, 5, 3, 6, 2))
        return mx.reshape(x, (batch_size, channels, frames, height * patch_size, width * patch_size))

    @staticmethod
    def _new_feature_cache() -> list[mx.array | str | None]:
        return [None] * 64

    @staticmethod
    def _default_mean(z_dim: int) -> np.ndarray:
        if z_dim == 16:
            return Wan2_2_VAE.WAN21_LATENTS_MEAN
        return Wan2_2_VAE.LATENTS_MEAN

    @staticmethod
    def _default_std(z_dim: int) -> np.ndarray:
        if z_dim == 16:
            return Wan2_2_VAE.WAN21_LATENTS_STD
        return Wan2_2_VAE.LATENTS_STD
