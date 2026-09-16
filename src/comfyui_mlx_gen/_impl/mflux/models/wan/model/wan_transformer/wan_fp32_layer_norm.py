import mlx.core as mx
from mlx import nn


class FP32LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,), dtype=mx.float32) if affine else None
        self.bias = mx.zeros((dim,), dtype=mx.float32) if affine else None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        origin_dtype = hidden_states.dtype
        hidden_states_f32 = hidden_states.astype(mx.float32)
        mean = mx.mean(hidden_states_f32, axis=-1, keepdims=True)
        variance = mx.mean(mx.square(hidden_states_f32 - mean), axis=-1, keepdims=True)
        normalized = (hidden_states_f32 - mean) * mx.rsqrt(variance + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight.astype(mx.float32)
        if self.bias is not None:
            normalized = normalized + self.bias.astype(mx.float32)
        return normalized.astype(origin_dtype)
