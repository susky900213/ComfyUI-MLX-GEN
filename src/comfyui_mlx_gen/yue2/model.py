"""YuE2 AR/NAR Mixture-of-Transformers backbone in MLX.

移植自 ``npario/YuE2-3B-MLX`` 的 ``yue2_model.py``。参数名与转换后的
checkpoint 一致；量化方式直接读取 checkpoint 的 ``config.json``，不会再次量化。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class KVCache:
    """逐层追加的 KV cache（按 256 个位置扩容）。"""

    step = 256

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update(self, k, v):
        prev = self.offset
        batch, heads, length, dim = k.shape
        if self.keys is None or prev + length > self.keys.shape[2]:
            size = ((length + self.step - 1) // self.step) * self.step
            new_k = mx.zeros((batch, heads, size, dim), k.dtype)
            new_v = mx.zeros((batch, heads, size, dim), v.dtype)
            if self.keys is None:
                self.keys, self.values = new_k, new_v
            else:
                if prev % self.step:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
        self.offset += length
        self.keys[..., prev : self.offset, :] = k
        self.values[..., prev : self.offset, :] = v
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config["num_attention_heads"]
        self.n_kv = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        hidden = config["hidden_size"]
        dim = self.head_dim
        self.q_proj = nn.Linear(hidden, self.n_heads * dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.n_kv * dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.n_kv * dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(dim, eps=config["rms_norm_eps"])
        self.k_norm = nn.RMSNorm(dim, eps=config["rms_norm_eps"])
        self.rope_base = float(config["rope_theta"])
        self.scale = dim**-0.5

    def project_qkv(self, x, offset):
        batch, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(batch, length, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        k = self.k_norm(self.k_proj(x).reshape(batch, length, self.n_kv, -1)).transpose(
            0, 2, 1, 3
        )
        v = self.v_proj(x).reshape(batch, length, self.n_kv, -1).transpose(0, 2, 1, 3)

        def rope(array):
            return mx.fast.rope(
                array,
                self.head_dim,
                traditional=False,
                base=self.rope_base,
                scale=1.0,
                offset=offset,
            )

        return rope(q), rope(k), v

    def attend(self, q, k, v, mask):
        batch, _, length, _ = q.shape
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, eps = config["hidden_size"], config["rms_norm_eps"]
        self.input_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.mlp = MLP(config)
        self.nar_input_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.nar_self_attn = Attention(config)
        self.nar_pre_mlp_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.nar_mlp = MLP(config)

    def ar(self, x, cache: KVCache, mask):
        q, k, v = self.self_attn.project_qkv(self.input_layernorm(x), cache.offset)
        k, v = cache.update(k, v)
        x = x + self.self_attn.attend(q, k, v, mask)
        return x + self.mlp(self.post_attention_layernorm(x))

    def ar_prefill_kv(self, x):
        q, k, v = self.self_attn.project_qkv(self.input_layernorm(x), 0)
        x = x + self.self_attn.attend(q, k, v, "causal")
        return x + self.mlp(self.post_attention_layernorm(x)), (k, v)

    def nar(self, x, ar_k, ar_v, offset):
        q, k, v = self.nar_self_attn.project_qkv(self.nar_input_layernorm(x), offset)
        k = mx.concatenate([ar_k, k], axis=2)
        v = mx.concatenate([ar_v, v], axis=2)
        x = x + self.nar_self_attn.attend(q, k, v, None)
        return x + self.nar_mlp(self.nar_pre_mlp_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [DecoderLayer(config) for _ in range(config["num_hidden_layers"])]
        self.norm = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_size=256):
        super().__init__()
        self.freq_size = freq_size
        self.fc1 = nn.Linear(freq_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def __call__(self, timestep, dtype):
        half = self.freq_size // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = timestep.astype(mx.float32).reshape(-1, 1) * freqs[None]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(dtype)
        return self.fc2(nn.silu(self.fc1(embedding)))


class LatentPosEmbed(nn.Module):
    def __init__(self, max_frames, hidden):
        super().__init__()
        self.pe = mx.zeros((max_frames, hidden))


class Yue2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden, latent_dim = config["hidden_size"], config["latent_dim"]
        self.model = Backbone(config)
        self.lm_head = nn.Linear(hidden, config["vocab_size"], bias=False)
        self.llm2vae = nn.Linear(hidden, latent_dim)
        self.vae2llm = nn.Linear(latent_dim, hidden)
        self.time_embedder = TimestepEmbedder(hidden)
        self.latent_pos_embed = LatentPosEmbed(config["max_latent_frames"], hidden)

    def ar_step(self, tokens, caches, all_positions=False):
        x = self.model.embed_tokens(tokens)
        mask = "causal" if tokens.shape[1] > 1 else None
        for layer, cache in zip(self.model.layers, caches):
            x = layer.ar(x, cache, mask)
        x = self.model.norm(x if all_positions else x[:, -1:])
        logits = self.lm_head(x)
        return logits if all_positions else logits[:, -1]

    def nar_prefill(self, ar_tokens):
        x = self.model.embed_tokens(mx.array([ar_tokens]))
        cache = []
        for layer in self.model.layers:
            x, kv = layer.ar_prefill_kv(x)
            cache.append(kv)
        mx.eval(cache)
        return cache

    def nar_velocity(self, state, raw_t, ar_cache, ar_length):
        shift = self.config["timestep_shift"]
        sigmoid = mx.sigmoid(mx.array(raw_t, dtype=state.dtype))
        timestep = shift * sigmoid / (1 + (shift - 1) * sigmoid)
        count = state.shape[0] + 2
        x = self.vae2llm(mx.pad(state, [(1, 1), (0, 0)])[None])
        x = x + self.time_embedder(timestep, state.dtype)[None]
        positions = mx.minimum(mx.arange(count), self.config["max_latent_frames"] - 1)
        x = x + self.latent_pos_embed.pe[positions][None]
        for layer, (key, value) in zip(self.model.layers, ar_cache):
            x = layer.nar(x, key, value, ar_length)
        return self.llm2vae(self.model.norm(x))[0, 1:-1]


def load_model(path: Path) -> Yue2Model:
    """从一个 YuE2 精度变体目录加载主模型。"""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    model = Yue2Model(config)
    weights = mx.load(str(path / "model.safetensors"))
    quantization = config.get("quantization")
    if quantization:
        nar_bits = quantization.get("nar_bits", quantization["bits"])

        def predicate(module_path, module):
            if not isinstance(module, nn.Linear) or f"{module_path}.scales" not in weights:
                return False
            return {
                "group_size": quantization["group_size"],
                "bits": nar_bits if ".nar_" in module_path else quantization["bits"],
            }

        nn.quantize(
            model,
            quantization["group_size"],
            quantization["bits"],
            class_predicate=predicate,
        )
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model