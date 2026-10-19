"""MOSS prompt formatting and padded conditioning contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import mlx.core as mx

MOSS_MAX_SEQUENCE_LENGTH = 4096
MOSS_CHAT_PREFIX = "<|im_start|>user\n"
MOSS_CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n"


def format_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("MOSS prompt 必须是字符串")
    return f"{MOSS_CHAT_PREFIX}{prompt}{MOSS_CHAT_SUFFIX}"


def validate_token_ids(token_ids: Sequence[int], max_length: int = MOSS_MAX_SEQUENCE_LENGTH) -> list[int]:
    values = [int(value) for value in token_ids]
    if not values:
        raise ValueError("MOSS prompt 至少需要一个 token")
    if int(max_length) < 1 or int(max_length) > MOSS_MAX_SEQUENCE_LENGTH:
        raise ValueError(f"MOSS max_length 必须在 1..{MOSS_MAX_SEQUENCE_LENGTH}")
    return values[: int(max_length)]


@dataclass(frozen=True)
class MOSSConditioning:
    embeddings: mx.array
    attention_mask: mx.array

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 3 or self.attention_mask.ndim != 2:
            raise ValueError("MOSS conditioning 必须是 embeddings[B,S,4096] + mask[B,S]")
        if self.embeddings.shape[:2] != self.attention_mask.shape:
            raise ValueError("MOSS conditioning 的 embedding 与 mask 形状不一致")
        if self.embeddings.shape[-1] != 4096:
            raise ValueError(f"MOSS conditioning 最后一维必须为 4096，收到 {self.embeddings.shape[-1]}")


def pad_conditioning(rows: Sequence[mx.array]) -> MOSSConditioning:
    if not rows:
        raise ValueError("MOSS conditioning 不能为空")
    normalized = []
    for row in rows:
        if row.ndim == 2:
            row = row[None]
        if row.ndim != 3 or row.shape[0] != 1 or row.shape[-1] != 4096:
            raise ValueError(f"每条 MOSS text hidden 必须是 [1,S,4096]，收到 {tuple(row.shape)}")
        normalized.append(row[0])
    length = max(int(row.shape[0]) for row in normalized)
    dtype = normalized[0].dtype
    embeds = mx.zeros((len(normalized), length, 4096), dtype=dtype)
    mask = mx.zeros((len(normalized), length), dtype=mx.bool_)
    for index, row in enumerate(normalized):
        size = int(row.shape[0])
        embeds[index, :size] = row
        mask[index, :size] = True
    return MOSSConditioning(embeds, mask)


def _text_position_ids(input_ids: mx.array) -> mx.array:
    """Return pure-text Qwen3.5 M-RoPE positions.

    Qwen3.5's MLX implementation expects three position axes even for text-only
    input.  Keeping all axes identical matches the Qwen3-VL pure-text layout.
    Passing the positions explicitly also avoids a bug in some mlx-vlm versions
    where the no-KV-cache path tries to read ``cache.offset`` from ``None``.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"MOSS text input_ids 必须是二维 [batch, sequence]，收到 {tuple(input_ids.shape)}")
    batch, sequence = input_ids.shape
    positions = mx.arange(sequence, dtype=mx.int32)[None, None, :]
    return mx.broadcast_to(positions, (3, batch, sequence))


def encode_prompts(
    prompts: Sequence[str] | str,
    tokenizer: Any,
    text_encoder: Any,
    max_length: int = MOSS_MAX_SEQUENCE_LENGTH,
) -> MOSSConditioning:
    values = [prompts] if isinstance(prompts, str) else list(prompts)
    if not values or not all(isinstance(value, str) for value in values):
        raise ValueError("MOSS prompts 必须是非空字符串或字符串列表")
    rows = []
    for prompt in values:
        formatted = format_prompt(prompt)
        encoded = tokenizer(formatted, return_tensors="np", truncation=True, max_length=int(max_length))
        input_ids = mx.array(encoded["input_ids"], dtype=mx.int32)
        output = text_encoder(
            input_ids,
            mask=mx.ones(input_ids.shape, dtype=mx.bool_),
            position_ids=_text_position_ids(input_ids),
        )
        hidden = getattr(output, "hidden_states", None)
        if hidden is None:
            hidden = getattr(output, "last_hidden_state", output)
        if isinstance(hidden, (list, tuple)):
            hidden = hidden[-1]
        rows.append(hidden[0] if hidden.ndim == 3 else hidden)
    return pad_conditioning(rows)