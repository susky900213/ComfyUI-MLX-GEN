"""FLUX 3 Image Klein MLX implementation.

This package deliberately does not reuse the Flux.2 transformer.  FLUX 3 uses
three mode streams (image, reference-image and Qwen3-VL text) followed by joint
single-stream blocks, and its distilled checkpoint is a flow-map model.
"""

from .dit import Flux3Packed, Flux3Transformer

__all__ = ["Flux3Packed", "Flux3Transformer"]