# SwiftVR one-step video restoration
# Wan 2.2 TI2V-5B transformer with mask-free shifted-window attention, driven over a
# fixed 4a+1 chunk protocol by the Restoration-aware Autoencoder.

from mflux.models.swiftvr.variants.upscale.swiftvr import SwiftVR

__all__ = ["SwiftVR"]
