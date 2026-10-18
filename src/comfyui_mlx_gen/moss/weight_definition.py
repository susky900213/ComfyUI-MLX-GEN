"""MOSS component metadata.

The MOSS transformer and text encoder are loaded by ``moss.loader``.  The VAE
uses the official Flux.2 autoencoder implementation only as a component
format, so its mapping is shared without making MOSS a Flux.2 family.
"""

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition


def _vae_mapping():
    from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping

    return Flux2WeightMapping.get_vae_mapping()


class MOSSWeightDefinition:
    @staticmethod
    def get_components():
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=_vae_mapping,
            ),
            # These two entries document the native component names.  The
            # independent MOSS streaming loader handles their tensors.
            ComponentDefinition(name="transformer", hf_subdir="transformer", precision=ModelConfig.precision),
            ComponentDefinition(name="text_encoder", hf_subdir="text_encoder", precision=ModelConfig.precision),
        ]

    @staticmethod
    def get_tokenizers():
        return []
