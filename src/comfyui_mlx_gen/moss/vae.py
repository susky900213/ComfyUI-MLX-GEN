"""MOSS-Image Flux.2-format VAE.

MOSS uses the Flux.2 VAE layout, but its released VAE has a different decoder
width from the default Flux.2 VAE used by mflux.  Reusing ``Flux2VAE``
unchanged silently leaves most decoder weights unmatched because mflux applies
weights with ``strict=False``; the resulting randomly initialized decoder
produces snow-like images.  Keep the encoder and packing contract shared while
matching the official MOSS decoder configuration exactly.
"""

from __future__ import annotations

from mflux.models.flux2.model.flux2_vae.decoder.decoder import Flux2Decoder
from mflux.models.flux2.model.flux2_vae.encoder.encoder import Flux2Encoder
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE


class MOSSVAE(Flux2VAE):
    """The official MOSS ``AutoencoderKLFlux2`` architecture in MLX."""

    DECODER_BLOCK_OUT_CHANNELS = (96, 192, 384, 384)

    def __init__(self):
        # The MOSS encoder uses the standard Flux.2 widths.  Only the decoder
        # differs from mflux's generic Flux2VAE defaults.
        super(Flux2VAE, self).__init__()
        self.encoder = Flux2Encoder()
        self.decoder = Flux2Decoder(block_out_channels=self.DECODER_BLOCK_OUT_CHANNELS)
        self.quant_conv = self._make_quant_conv()
        self.post_quant_conv = self._make_post_quant_conv()
        self.bn = self._make_batch_norm()

    @staticmethod
    def _make_quant_conv():
        from mlx import nn

        return nn.Conv2d(64, 64, kernel_size=1, padding=0)

    @staticmethod
    def _make_post_quant_conv():
        from mlx import nn

        return nn.Conv2d(32, 32, kernel_size=1, padding=0)

    @staticmethod
    def _make_batch_norm():
        from mflux.models.flux2.model.flux2_vae.common.batch_norm_stats import Flux2BatchNormStats

        return Flux2BatchNormStats(num_features=128, eps=1e-4, momentum=0.1)