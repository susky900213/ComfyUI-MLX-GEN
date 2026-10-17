"""Qwen-Image 2.1 prompt templates and Qwen3-VL-8B multimodal conditioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_model import rope_index
from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_text_model import Qwen3VLTextModel
from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_vision_model import (
    Qwen3VLVisionModel,
    preprocess_image,
)

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
T2I_TEMPLATE = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMAGE_TOKEN_ID = 151655
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653


class QwenImage21TextEncoder(Qwen3VLTextModel):
    """The 36-layer language tower plus the Qwen3-VL vision tower used by edit prompts."""

    def __init__(self, *args, vision_config: dict[str, Any] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        config = dict(vision_config or {})
        self.visual = Qwen3VLVisionModel(
            hidden_size=int(config.get("hidden_size", 1152)),
            depth=int(config.get("depth", 27)),
            num_heads=int(config.get("num_heads", 16)),
            intermediate_size=int(config.get("intermediate_size", 4304)),
            in_channels=int(config.get("in_channels", 3)),
            patch_size=int(config.get("patch_size", 16)),
            temporal_patch_size=int(config.get("temporal_patch_size", 2)),
            spatial_merge_size=int(config.get("spatial_merge_size", 2)),
            num_position_embeddings=int(config.get("num_position_embeddings", 2304)),
            out_hidden_size=int(config.get("out_hidden_size", kwargs.get("hidden_size", 4096))),
            deepstack_visual_indexes=tuple(
                int(value) for value in config.get("deepstack_visual_indexes", (8, 16, 24))
            ),
        )


def load_tokenizer(path: str | Path) -> Any:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"未找到 Qwen-Image 2.1 processor/tokenizer 目录: {root}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True)
    tokenizer.padding_side = "left"
    return tokenizer


def system_token_count(tokenizer: Any) -> int:
    raw = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    return len(tokenizer(raw, add_special_tokens=False)["input_ids"])


def encode_prompt(
    text_encoder: QwenImage21TextEncoder,
    tokenizer: Any,
    prompt: str,
    max_length: int | None = None,
    images: list[Any] | tuple[Any, ...] = (),
) -> tuple[mx.array, mx.array, mx.array]:
    """Return the official pre-norm Qwen3-VL conditioning for T2I or image editing.

    Every visual placeholder is expanded to the number of merged Qwen3-VL tokens for its
    image.  The returned ``image_pad_mask`` deliberately remains at VLM-token granularity;
    the DiT expands each ``True`` slot to the corresponding 2×2 VAE-latent block.
    """
    prompt = prompt if prompt else " "
    selected = list(images or [])
    if selected:
        refs = " ".join(
            f"<image{index + 1}><|vision_start|><|image_pad|><|vision_end|>"
            for index in range(len(selected))
        )
        raw = T2I_TEMPLATE.format(f"{refs}{prompt}")
    else:
        raw = T2I_TEMPLATE.format(prompt)
    encoded = tokenizer(raw, add_special_tokens=False)
    ids = [int(value) for value in encoded["input_ids"]]

    image_patches: list[np.ndarray] = []
    image_grids: list[tuple[int, int, int]] = []
    if selected:
        expanded: list[int] = []
        image_index = 0
        merge = int(text_encoder.visual.spatial_merge_size)
        for token in ids:
            if token != IMAGE_TOKEN_ID:
                expanded.append(token)
                continue
            if image_index >= len(selected):
                raise ValueError("Qwen-Image 2.1 prompt 中的图片槽多于参考图")
            patches, grid = preprocess_image(
                selected[image_index],
                patch_size=int(text_encoder.visual.patch_size),
                merge_size=merge,
                temporal_patch_size=int(text_encoder.visual.temporal_patch_size),
            )
            image_patches.append(patches)
            image_grids.append(grid)
            expanded.extend([IMAGE_TOKEN_ID] * (grid[0] * grid[1] * grid[2] // (merge * merge)))
            image_index += 1
        if image_index != len(selected):
            raise ValueError(
                f"Qwen-Image 2.1 prompt 只生成了 {image_index} 个图片槽，但收到 {len(selected)} 张参考图"
            )
        ids = expanded

    if max_length and len(ids) > int(max_length):
        raise ValueError(
            f"Qwen-Image 2.1 提示词模板编码后有 {len(ids)} token，超过 max_length={max_length}；"
            "请缩短提示词或增大「MLX 条件加载器」的 max_length"
        )
    drop = system_token_count(tokenizer)
    if len(ids) <= drop:
        raise ValueError("Qwen-Image 2.1 提示词在移除 system turn 后没有 token")
    token_ids = mx.array(ids, dtype=mx.int32)[None]
    if image_patches:
        embeds = text_encoder.embed_tokens(token_ids)
        merged: list[mx.array] = []
        features: list[list[mx.array]] = []
        for patches, grid in zip(image_patches, image_grids):
            visual, deep = text_encoder.visual(mx.array(patches), grid)
            merged.append(visual)
            features.append(deep)
        visual_embeds = mx.concatenate(merged, axis=0)
        visual_positions = mx.array(
            np.flatnonzero(np.asarray(ids, dtype=np.int32) == IMAGE_TOKEN_ID), dtype=mx.int32
        )
        if int(visual_positions.shape[0]) != int(visual_embeds.shape[0]):
            raise ValueError(
                f"Qwen-Image 2.1 有 {int(visual_positions.shape[0])} 个图片 token，"
                f"但视觉塔输出 {int(visual_embeds.shape[0])} 个特征"
            )
        embeds[:, visual_positions, :] = visual_embeds.astype(embeds.dtype)[None]
        deepstack = [
            mx.concatenate([feature[stage] for feature in features], axis=0)
            for stage in range(len(features[0]))
        ]
        positions = rope_index(
            np.asarray(ids, dtype=np.int32), IMAGE_TOKEN_ID, image_grids, merge
        )
        hidden = text_encoder(
            token_ids,
            position_ids=mx.array(positions.astype(np.int32))[:, None, :],
            inputs_embeds=embeds,
            deepstack_visual_embeds=deepstack,
            visual_positions=visual_positions,
        )
    else:
        hidden = text_encoder(token_ids)
    hidden = hidden[:, drop:]
    attention_mask = mx.ones(hidden.shape[:2], dtype=mx.bool_)
    image_pad_mask = mx.array(
        np.asarray(ids[drop:], dtype=np.int32) == IMAGE_TOKEN_ID, dtype=mx.bool_
    )[None]
    mx.eval(hidden, attention_mask, image_pad_mask)
    return hidden, attention_mask, image_pad_mask