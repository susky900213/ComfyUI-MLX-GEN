"""Resolve the tiny autoencoder that matches a model family's latent space.

Mapping is explicit per family, never inferred from latent-channel count: several
families share a channel count while using entirely different latent semantics, and
decoding one with another's tiny decoder produces confident nonsense.
"""

import logging
from dataclasses import dataclass

from mflux.models.common.preview.tiny_autoencoder import TinyAutoencoder
from mflux.models.common.preview.tiny_autoencoder_loader import TinyAutoencoderLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TinyDecoderSpec:
    repo_id: str
    latent_channels: int
    note: str
    config_overrides: dict | None = None
    weight_file: str | None = None


# Keyed by the latent space a family generates in. Entries are added only once the
# checkpoint has been verified against that family's own VAE on real latents.
TINY_DECODERS: dict[str, TinyDecoderSpec] = {
    "flux.1": TinyDecoderSpec(
        repo_id="madebyollin/taef1",
        latent_channels=16,
        note="FLUX.1 latent space, shared by Z-Image (bit-identical VAE).",
    ),
    "flux.2": TinyDecoderSpec(
        repo_id="madebyollin/taef2",
        latent_channels=32,
        note="FLUX.2 latent space, shared by ERNIE-Image and Bonsai (bit-identical VAE).",
        # This repository publishes weights only; the architecture is the FLUX.2 variant
        # with a mid-block GroupNorm branch, which diffusers' AutoencoderTiny cannot express.
        config_overrides={"latent_channels": 32, "use_midblock_gn": True},
        weight_file="taef2.safetensors",
    ),
}


class PreviewDecoderUnavailable(RuntimeError):
    pass


class PreviewDecoder:
    """Lazily-loaded tiny decoder bound to one latent space."""

    def __init__(self, latent_space: str, model: TinyAutoencoder, spec: TinyDecoderSpec):
        self.latent_space = latent_space
        self.model = model
        self.spec = spec

    @staticmethod
    def latent_space_of(model) -> str | None:
        return getattr(getattr(model, "vae", None), "latent_space", None)

    @staticmethod
    def available_for(latent_space: str | None) -> bool:
        return latent_space in TINY_DECODERS

    @staticmethod
    def resolve(model, mode: str = "auto") -> "PreviewDecoder | None":
        """Return a tiny decoder per the requested mode, or None to use the model's own VAE.

        `auto` never fails a generation: an unmapped family or an undownloaded decoder
        falls back to the full VAE. `tiny` is a explicit request and raises instead.
        """
        if mode == "full":
            return None
        latent_space = PreviewDecoder.latent_space_of(model)
        if mode != "auto":
            return PreviewDecoder.load_for(latent_space)
        if not PreviewDecoder.available_for(latent_space):
            return None
        try:
            return PreviewDecoder.load_for(latent_space)
        except (PreviewDecoderUnavailable, FileNotFoundError) as error:
            logger.warning(
                "Tiny preview decoder for the %s latent space is not downloaded; "
                "step-wise previews will use the full VAE. Download it once with: %s\n%s",
                latent_space,
                f"mlxgen download --model {TINY_DECODERS[latent_space].repo_id}",
                error,
            )
            return None

    @staticmethod
    def load_for(latent_space: str | None) -> "PreviewDecoder":
        spec = TINY_DECODERS.get(latent_space) if latent_space else None
        if spec is None:
            raise PreviewDecoderUnavailable(
                f"No tiny preview decoder is published for the {latent_space!r} latent space. "
                "Run with --preview-decoder full to preview through the full VAE instead."
            )
        model = TinyAutoencoderLoader.load(
            spec.repo_id,
            config_overrides=spec.config_overrides,
            weight_file=spec.weight_file,
        )
        if model.latent_channels != spec.latent_channels:
            raise PreviewDecoderUnavailable(
                f"{spec.repo_id} exposes {model.latent_channels} latent channels; "
                f"the {latent_space} latent space needs {spec.latent_channels}."
            )
        logger.info("Step-wise previews use the %s tiny decoder (approximate; final output uses the full VAE).", spec.repo_id)  # fmt: off
        return PreviewDecoder(latent_space=latent_space, model=model, spec=spec)

    def decode(self, latents, vae=None):
        """Decode in-flight latents. A VAE may expose `to_preview_latents` when its
        in-flight layout differs from the tiny decoder's input (packing, patchify)."""
        prepare = getattr(vae, "to_preview_latents", None)
        return self.model.decode(prepare(latents) if prepare is not None else latents)
