"""Qwen3-VL-4B-Instruct conditioning used by FLUX 3 Image Klein."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..h3.model.h3_text_encoder.qwen3_vl_model import Qwen3VLModel
from ..h3.model.h3_text_encoder.qwen3_vl_text_model import Qwen3VLTextModel
from ..h3.model.h3_text_encoder.qwen3_vl_vision_model import Qwen3VLVisionModel


OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
MAX_LENGTH = 4096
PAD_MULTIPLE = 80


class Flux3TextEncoder(Qwen3VLModel):
    """Qwen3-VL-4B with the eight hidden-state taps required by FLUX 3.

    The existing Qwen3-VL implementation is shared with H3 and Qwen-Image
    2.1.  FLUX 3 differs only in the published 4B configuration and in using
    eight intermediate language-model states concatenated to 20480 channels.
    """

    def __init__(self, text_config: dict[str, Any] | None = None, vision_config: dict[str, Any] | None = None):
        text = dict(text_config or {})
        vision = dict(vision_config or {})
        rope_scaling = dict(text.get("rope_scaling") or {})
        mrope = rope_scaling.get("mrope_section", (24, 20, 20))
        language = Qwen3VLTextModel(
            vocab_size=int(text.get("vocab_size", 151936)),
            hidden_size=int(text.get("hidden_size", 2560)),
            num_hidden_layers=int(text.get("num_hidden_layers", 36)),
            num_attention_heads=int(text.get("num_attention_heads", 32)),
            num_key_value_heads=int(text.get("num_key_value_heads", 8)),
            head_dim=int(text.get("head_dim", 128)),
            intermediate_size=int(text.get("intermediate_size", 9728)),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(text.get("rope_theta", 5_000_000)),
            mrope_section=tuple(int(value) for value in mrope),
        )
        visual = Qwen3VLVisionModel(
            hidden_size=int(vision.get("hidden_size", 1024)),
            depth=int(vision.get("depth", 24)),
            num_heads=int(vision.get("num_heads", 16)),
            intermediate_size=int(vision.get("intermediate_size", 4096)),
            in_channels=int(vision.get("in_channels", 3)),
            patch_size=int(vision.get("patch_size", 16)),
            temporal_patch_size=int(vision.get("temporal_patch_size", 2)),
            spatial_merge_size=int(vision.get("spatial_merge_size", 2)),
            num_position_embeddings=int(vision.get("num_position_embeddings", 2304)),
            out_hidden_size=int(vision.get("out_hidden_size", 2560)),
            deepstack_visual_indexes=tuple(
                int(value) for value in vision.get("deepstack_visual_indexes", (5, 11, 17))
            ),
        )
        super().__init__(language_model=language, visual=visual)

    def encode_tokens(self, token_ids: mx.array) -> mx.array:
        """Return padded ``[tokens, 8 * 2560]`` Qwen hidden states."""
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError(f"FLUX 3 文本输入必须是 [1, tokens]，收到 {tuple(token_ids.shape)}")
        positions = Qwen3VLTextModel.text_position_ids(1, int(token_ids.shape[1]))
        language = self.language_model
        hidden = language.embed_tokens(token_ids)
        cos, sin = language.rotary_emb(positions)
        taps: list[mx.array] = [hidden] if 0 in OUTPUT_LAYERS else []
        for index, layer in enumerate(language.layers, start=1):
            hidden = layer(hidden, cos, sin)
            if index in OUTPUT_LAYERS:
                taps.append(hidden)
        if len(taps) != len(OUTPUT_LAYERS):
            raise ValueError(
                f"Qwen3-VL-4B 只有 {len(language.layers)} 层，无法提取 FLUX 3 taps {OUTPUT_LAYERS}"
            )
        return mx.concatenate(taps, axis=-1)


def load_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import json

    root = Path(path)
    config_path = root if root.name == "config.json" else root / "config.json"
    if not config_path.is_file():
        # ComfyUI single-file exports do not carry the HF config.  The official
        # Qwen3-VL-4B values are stable and are used only for construction.
        return {}, {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return dict(data.get("text_config") or {}), dict(data.get("vision_config") or {})


def encode_prompt(
    tokenizer: Any, encoder: Flux3TextEncoder, prompt: str, max_length: int | None = None
) -> tuple[mx.array, mx.array]:
    """Apply the Qwen chat template and return BFL's 80-token padded features."""
    prompt = prompt if prompt else " "
    max_length = min(int(max_length or MAX_LENGTH), MAX_LENGTH)
    try:
        text = tokenizer.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    raw = tokenizer.tokenizer(text, return_tensors="np", truncation=True, max_length=max_length)["input_ids"]
    real = int(raw.shape[1])
    length = min(int(math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE), max_length)
    tokens = tokenizer.tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=length,
    )["input_ids"]
    hidden = encoder.encode_tokens(mx.array(tokens, dtype=mx.int32))
    hidden = hidden.astype(mx.bfloat16)
    mx.eval(hidden)
    positions = mx.arange(length, dtype=mx.float32)
    ids = mx.stack(
        [mx.zeros_like(positions), mx.zeros_like(positions), mx.zeros_like(positions), positions], axis=-1
    )[None]
    return hidden, ids