import mlx.core as mx
from mlx import nn


class WanActivation:
    LOW_PRECISION_DTYPES = (mx.float16, mx.bfloat16)

    @staticmethod
    def silu(x: mx.array) -> mx.array:
        if x.dtype in WanActivation.LOW_PRECISION_DTYPES:
            return nn.silu(x.astype(mx.float32)).astype(x.dtype)
        return nn.silu(x)

    @staticmethod
    def gelu_tanh(x: mx.array) -> mx.array:
        if x.dtype in WanActivation.LOW_PRECISION_DTYPES:
            return nn.gelu_approx(x.astype(mx.float32)).astype(x.dtype)
        return nn.gelu_approx(x)
