"""Diffusers-to-MLX mappings for the three Fibo-2 components."""

from __future__ import annotations

from mflux.models.common.weights.mapping.weight_mapping import WeightTarget

from ..flux3.weight_mapping import Flux3WeightMapping


def _direct(target: str, source: str | None = None, *, required: bool = True) -> WeightTarget:
    return WeightTarget(target, [source or target], required=required)


class Fibo2WeightMapping:
    @staticmethod
    def get_transformer_mapping() -> list[WeightTarget]:
        targets: list[WeightTarget] = [
            _direct("x_embedder.weight"),
            _direct("x_embedder.bias"),
            _direct("t_embedder.mlp.0.weight"),
            _direct("t_embedder.mlp.0.bias"),
            _direct("t_embedder.mlp.2.weight"),
            _direct("t_embedder.mlp.2.bias"),
            _direct("perceiver.latent_queries"),
            _direct("perceiver.norm_out.weight"),
            _direct("final_layer.linear.weight"),
            _direct("final_layer.linear.bias"),
            _direct("final_layer.adaLN_modulation.1.weight"),
            _direct("final_layer.adaLN_modulation.1.bias"),
        ]
        for index in range(2):
            targets.extend(_block_targets(f"noise_refiner.{index}"))
        for index in range(44):
            targets.extend(_block_targets(f"layers.{index}", cross=index in (6, 14, 22, 30, 38)))
        for index in range(2):
            prefix = f"perceiver.layers.{index}"
            targets.extend(
                [
                    _direct(f"{prefix}.cross_attn_norm.weight"),
                    _direct(f"{prefix}.self_attn_norm.weight"),
                    _direct(f"{prefix}.ffn_norm.weight"),
                ]
            )
            targets.extend(_attention_targets(f"{prefix}.cross_attention", cross=True))
            targets.extend(_attention_targets(f"{prefix}.self_attention"))
            targets.extend(_ffn_targets(f"{prefix}.ffn"))
        return targets

    @staticmethod
    def get_text_encoder_mapping() -> list[WeightTarget]:
        # Fibo uses the same Qwen3-VL-4B layout as FLUX 3 Image Klein: 36 text
        # layers, 24 vision blocks, and three visual deep-stack taps.
        return Flux3WeightMapping.get_text_encoder_mapping()


def _attention_targets(prefix: str, *, cross: bool = False) -> list[WeightTarget]:
    source = prefix
    return [
        _direct(f"{prefix}.to_q.weight", f"{source}.to_q.weight"),
        _direct(f"{prefix}.to_k.weight", f"{source}.to_k.weight"),
        _direct(f"{prefix}.to_v.weight", f"{source}.to_v.weight"),
        _direct(f"{prefix}.to_out.0.weight", f"{source}.to_out.0.weight"),
        _direct(f"{prefix}.norm_q.weight", f"{source}.norm_q.weight"),
        _direct(f"{prefix}.norm_k.weight", f"{source}.norm_k.weight"),
    ]


def _ffn_targets(prefix: str) -> list[WeightTarget]:
    return [
        _direct(f"{prefix}.w1.weight"),
        _direct(f"{prefix}.w2.weight"),
        _direct(f"{prefix}.w3.weight"),
    ]


def _block_targets(prefix: str, *, cross: bool = False) -> list[WeightTarget]:
    targets = [
        _direct(f"{prefix}.attention_norm1.weight"),
        _direct(f"{prefix}.ffn_norm1.weight"),
        _direct(f"{prefix}.attention_norm2.weight"),
        _direct(f"{prefix}.ffn_norm2.weight"),
        _direct(f"{prefix}.adaLN_modulation.0.weight"),
        _direct(f"{prefix}.adaLN_modulation.0.bias"),
    ]
    targets.extend(_attention_targets(f"{prefix}.attention"))
    targets.extend(_ffn_targets(f"{prefix}.feed_forward"))
    if cross:
        targets.extend(
            [
                _direct(f"{prefix}.cross_attn_norm1.weight"),
                _direct(f"{prefix}.cross_attn_norm2.weight"),
                _direct(f"{prefix}.cross_attn_gate"),
            ]
        )
        targets.extend(_attention_targets(f"{prefix}.cross_attention", cross=True))
    return targets


__all__ = ["Fibo2WeightMapping"]