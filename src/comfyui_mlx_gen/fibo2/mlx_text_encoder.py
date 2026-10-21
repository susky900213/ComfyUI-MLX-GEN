"""Qwen3-VL-4B conditioning used by the native Fibo-2 MLX backend."""

from __future__ import annotations

import json
from typing import Any, Sequence

import mlx.core as mx
import numpy as np
from mlx import nn

from ..h3.model.h3_text_encoder.qwen3_vl_model import rope_index
from ..h3.model.h3_text_encoder.qwen3_vl_text_model import Qwen3VLTextModel
from ..h3.model.h3_text_encoder.qwen3_vl_vision_model import Qwen3VLVisionModel, preprocess_image


VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
IMAGE_TOKEN_ID = 151655
TEXT_ENCODER_OUT_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))


def _rms_normalize(value: mx.array) -> mx.array:
    value32 = value.astype(mx.float32)
    return (value32 * mx.rsqrt(mx.mean(mx.square(value32), axis=-1, keepdims=True) + 1e-6)).astype(value.dtype)


def edit_prompt(prompt: str, count: int) -> str:
    """Build the compact JSON edit prompt used by the official pipeline."""

    try:
        parsed = json.loads(prompt)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    instruction = prompt
    caption: Any = None
    if isinstance(parsed, dict):
        instruction = parsed.get("edit_instruction", prompt)
        caption = parsed.get("structured_caption")
        if caption is None:
            caption = {key: value for key, value in parsed.items() if key != "edit_instruction"}
        if not caption:
            caption = None
    result: dict[str, Any] = (
        {"image": VISION_MARKER} if count == 1 else {f"image_{i}": VISION_MARKER for i in range(1, count + 1)}
    )
    if caption:
        result["structured_caption"] = caption
    result["edit_instruction"] = instruction
    return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


def generation_prompt(prompt: str) -> str:
    """Apply Fibo's training-caption cleanup to a text-to-image prompt."""

    try:
        value = json.loads(prompt)
    except (TypeError, json.JSONDecodeError):
        return prompt or " "
    if not isinstance(value, dict):
        return prompt or " "

    def empty(item: Any) -> bool:
        if isinstance(item, float):
            return bool(np.isnan(item))
        return item is None or (isinstance(item, (str, dict, list)) and not item)

    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: cleaned for key, value in item.items() if key not in {"aesthetic_score", "preference_score"} and not empty(cleaned := clean(value))}
        if isinstance(item, list):
            return [cleaned for value in item if not empty(cleaned := clean(value))]
        return item

    cleaned = clean(value)
    return json.dumps(cleaned or {}, separators=(",", ":"), ensure_ascii=False)


class Fibo2TextEncoder(nn.Module):
    """Qwen3-VL-4B, retaining the six hidden-state bundles Fibo-2 consumes."""

    def __init__(self, *, max_length: int = 4096):
        super().__init__()
        self.max_length = int(max_length)
        self.language_model = Qwen3VLTextModel(
            vocab_size=151936,
            hidden_size=2560,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=9728,
            rms_norm_eps=1e-6,
            rope_theta=5_000_000.0,
            mrope_section=(24, 20, 20),
        )
        self.visual = Qwen3VLVisionModel(
            hidden_size=1024,
            depth=24,
            num_heads=16,
            intermediate_size=4096,
            in_channels=3,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=2304,
            out_hidden_size=2560,
            deepstack_visual_indexes=(5, 11, 17),
        )

    def _hidden_states(
        self,
        token_ids: np.ndarray,
        image_patches: Sequence[np.ndarray] = (),
        image_grids: Sequence[tuple[int, int, int]] = (),
    ) -> list[mx.array]:
        ids = mx.array(np.asarray(token_ids, dtype=np.int32))[None]
        embeds = self.language_model.embed_tokens(ids)
        deepstack = None
        visual_positions = None
        if image_patches:
            merged, features = [], []
            for patches, grid in zip(image_patches, image_grids):
                tokens, deep = self.visual(mx.array(patches), grid)
                merged.append(tokens)
                features.append(deep)
            image_embeds = mx.concatenate(merged, axis=0)
            visual_positions = mx.array(np.flatnonzero(np.asarray(token_ids) == IMAGE_TOKEN_ID).astype(np.int32))
            if int(visual_positions.shape[0]) != int(image_embeds.shape[0]):
                raise ValueError("Fibo-2 图片 token 数与 Qwen3-VL 视觉特征数不一致")
            embeds[:, visual_positions, :] = image_embeds.astype(embeds.dtype)[None]
            deepstack = [mx.concatenate([feature[index] for feature in features], axis=0) for index in range(len(features[0]))]
            positions = rope_index(np.asarray(token_ids), IMAGE_TOKEN_ID, list(image_grids), self.visual.spatial_merge_size)
            position_ids = mx.array(positions.astype(np.int32))[:, None, :]
        else:
            position_ids = self.language_model.text_position_ids(1, embeds.shape[1])
        cos, sin = self.language_model.rotary_emb(position_ids)
        hidden = embeds
        output = [hidden]
        for index, layer in enumerate(self.language_model.layers):
            hidden = layer(hidden, cos, sin)
            if deepstack is not None and index < len(deepstack):
                hidden[:, visual_positions, :] = hidden[:, visual_positions, :] + deepstack[index].astype(hidden.dtype)
            output.append(hidden)
        return output

    def encode(self, tokenizer: Any, prompt: str, images: Sequence[Any] = ()) -> tuple[mx.array, mx.array]:
        selected = list(images)
        body = edit_prompt(prompt, len(selected)) if selected else generation_prompt(prompt)
        raw = PROMPT_TEMPLATE.format(body)
        raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        encoded = raw_tokenizer(raw, add_special_tokens=False)
        ids = [int(value) for value in encoded["input_ids"]]
        image_patches: list[np.ndarray] = []
        image_grids: list[tuple[int, int, int]] = []
        if selected:
            expanded: list[int] = []
            image_index = 0
            merge = self.visual.spatial_merge_size
            for token in ids:
                if token != IMAGE_TOKEN_ID:
                    expanded.append(token)
                    continue
                if image_index >= len(selected):
                    raise ValueError("Fibo-2 prompt 中的图片槽多于参考图")
                patches, grid = preprocess_image(selected[image_index], patch_size=self.visual.patch_size, merge_size=merge, temporal_patch_size=self.visual.temporal_patch_size)
                image_patches.append(patches)
                image_grids.append(grid)
                expanded.extend([IMAGE_TOKEN_ID] * (grid[0] * grid[1] * grid[2] // (merge * merge)))
                image_index += 1
            ids = expanded
        if len(ids) > self.max_length:
            raise ValueError(f"Fibo-2 prompt 编码后超过 max_length={self.max_length}")
        states = self._hidden_states(np.asarray(ids, dtype=np.int32), image_patches, image_grids)
        bundles = []
        for layers in TEXT_ENCODER_OUT_LAYERS:
            bundles.append(mx.concatenate([_rms_normalize(states[layer]) for layer in layers], axis=-1))
        embeds = mx.stack(bundles, axis=1)
        mask = mx.ones((1, embeds.shape[2]), dtype=mx.bool_)
        mx.eval(embeds, mask)
        return embeds, mask


__all__ = ["Fibo2TextEncoder", "edit_prompt", "generation_prompt", "PROMPT_TEMPLATE", "VISION_MARKER"]