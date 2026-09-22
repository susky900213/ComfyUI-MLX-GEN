"""H3 的提示词组装与条件编码（Qwen3-VL presentation）。

两件事：

1. **组装提示词**：H3 训练时用的是三段带标签的文本（空行分隔）——
   `integrated_multimodal_description:` / `overall_soundscape:` / `non_diegetic_music:`。
   用户只写一段描述时它就是第一段；已经带标签的提示词（MiniMax Context-IR 的产物）
   原样透传。音效 / 配乐在节点上直接写进提示词即可，**不需要给节点加新 widget**；
2. **编码**：tokenizer **原样** tokenize（不加特殊 token、不加 chat 模板、不 padding），
   取 Qwen3-VL 的 `hidden_states[50]`（第 49 层之后的 pre-norm 输出），并给出逐 token 的
   模态标签（文本 → `TEXT_TAG`；视觉块 → `VIDEO_TAG`）。纯文生视频时全是 `TEXT_TAG`，
   视觉塔的输入路径在 `h3_layout` / `qwen3_vl_model` 里都已就位，留给二期的图生视频。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from comfyui_mlx_gen.h3.latent_creator.h3_layout import TEXT_TAG, VIDEO_TAG

PROMPT_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")


def prompt_sections_present(prompt: str) -> tuple[str, ...]:
    """提示词里已经有哪几段标签（只认「行首 + 冒号」的写法，句子里的同名词不算）。"""
    lines = [line.lstrip() for line in prompt.strip().splitlines()]
    return tuple(field for field in PROMPT_FIELDS if any(line.startswith(f"{field}:") for line in lines))


def compose_prompt(prompt: str, soundscape: str | None = None, music: str | None = None) -> str:
    """组装 H3 训练格式的提示词（`prompt` 必须非空；已带标签时原样透传）。"""
    text = prompt.strip()
    if not text:
        raise ValueError("MiniMax-H3 需要非空提示词")
    present = prompt_sections_present(text)
    for field, value in ((PROMPT_FIELDS[1], soundscape), (PROMPT_FIELDS[2], music)):
        if value and value.strip() and field in present:
            # 追加会送给模型两段同类描述，矛盾要到生成完才看得出来 —— 现在就说清楚
            raise ValueError(
                f"提示词里已经有 `{field}:` 这一段了，再给「音效 / 配乐」会多出一段同类描述。"
                "请去掉那一段，或改用节点上的可选输入"
            )
    sections = [text] if present else [f"{PROMPT_FIELDS[0]}: {text}"]
    if soundscape and soundscape.strip():
        sections.append(f"{PROMPT_FIELDS[1]}: {soundscape.strip()}")
    if music and music.strip():
        sections.append(f"{PROMPT_FIELDS[2]}: {music.strip()}")
    return "\n\n".join(sections)


def load_tokenizer(path: str | Path) -> Any:
    """加载 H3 的 tokenizer（Qwen2Tokenizer；原样 tokenize 用，不加 chat 模板）。"""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(
            f"未找到 H3 的 tokenizer 目录: {root}；请在 tokenizer/ 下补一个与条件编码器"
            "同名的软链（例如 tokenizer/MiniMax-H3-back → MiniMax-H3/tokenizer）"
        )
    from transformers import AutoTokenizer  # 延迟导入：与 mflux 用的同一份 transformers

    try:
        return AutoTokenizer.from_pretrained(str(root), trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"加载 H3 的 tokenizer 失败（{root}）: {exc}") from exc


def token_ids(tokenizer: Any, text: str) -> list[int]:
    """一条文本的 token id（不加特殊 token、不 padding）。"""
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(value) for value in ids]


def prompt_digest(prompt: str, keyframes: Sequence[Any] = ()) -> str:
    """提示词（含关键帧内容）的摘要，用来做条件编码的缓存键。"""
    digest = hashlib.sha1(prompt.encode("utf-8"))
    for frame in keyframes:
        digest.update(str(frame.size).encode("utf-8"))
        digest.update(frame.tobytes())
    return digest.hexdigest()


def encode_presentation(
    text_encoder: Any,
    tokenizer: Any,
    prompt: str,
    keyframes: Sequence[Any] = (),
    motion_frames: Sequence[Any] = (),
    motion_timestamps: Sequence[float] = (),
) -> tuple[Any, np.ndarray]:
    """编码 presentation，返回 `(embeds (1, L, 5120), tags (L,) int32)`。

    `keyframes` 是按顺序编号的 `<Picture N>` 图片参考；`motion_frames` 是按 2 fps
    抽样的动作参考帧，每两个相邻帧组成一个真正的 temporal patch，并以
    `<Video 1>` / `<T seconds>` 标签写进 presentation。两者可以同时存在，图片块
    排在视频块之前，分别占用自己的视觉槽。
    """
    ids: list[int] = []
    tags: list[int] = []
    patches: list[np.ndarray] = []
    grids: list[tuple[int, int, int]] = []
    if keyframes:
        from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_model import (
            IMAGE_TOKEN_ID,
            VISION_END_TOKEN_ID,
            VISION_START_TOKEN_ID,
        )
        from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_vision_model import preprocess_image

        merge = int(text_encoder.visual.spatial_merge_size)
        for index, frame in enumerate(keyframes):
            frame_patches, grid = preprocess_image(frame)
            patches.append(frame_patches)
            grids.append(grid)
            label = token_ids(tokenizer, f"<Picture {index + 1}>: ")
            num_image_tokens = (grid[0] * grid[1] * grid[2]) // (merge * merge)
            vision = [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * num_image_tokens + [VISION_END_TOKEN_ID]
            ids += label + vision
            tags += [TEXT_TAG] * len(label) + [VIDEO_TAG] * len(vision)

    if motion_frames:
        from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_model import (
            IMAGE_TOKEN_ID,
            VISION_END_TOKEN_ID,
            VISION_START_TOKEN_ID,
        )
        from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_vision_model import preprocess_video_pair

        if len(motion_frames) < 2:
            raise ValueError("H3 动作参考 presentation 至少需要 2 帧")
        timestamps = list(motion_timestamps) if motion_timestamps else [i / 2.0 for i in range(len(motion_frames))]
        if len(timestamps) != len(motion_frames):
            raise ValueError("H3 动作参考的时间戳数量必须与抽样帧数量一致")
        if len(motion_frames) % 2:
            motion_frames = tuple(motion_frames) + (motion_frames[-1],)
            timestamps.append(timestamps[-1])
        merge = int(text_encoder.visual.spatial_merge_size)
        ids += token_ids(tokenizer, "<Video 1>: ")
        tags += [TEXT_TAG] * len(token_ids(tokenizer, "<Video 1>: "))
        for index in range(0, len(motion_frames), 2):
            block_time = (float(timestamps[index]) + float(timestamps[index + 1])) / 2.0
            label = token_ids(tokenizer, f"<{block_time:.1f} seconds>")
            frame_patches, grid = preprocess_video_pair((motion_frames[index], motion_frames[index + 1]))
            patches.append(frame_patches)
            grids.append(grid)
            num_image_tokens = (grid[0] * grid[1] * grid[2]) // (merge * merge)
            vision = [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * num_image_tokens + [VISION_END_TOKEN_ID]
            ids += label + vision
            tags += [TEXT_TAG] * len(label) + [VIDEO_TAG] * len(vision)

    prompt_ids = token_ids(tokenizer, prompt)
    if not prompt_ids:
        raise ValueError("MiniMax-H3 需要非空提示词（tokenize 之后没有任何 token）")
    ids += prompt_ids
    tags += [TEXT_TAG] * len(prompt_ids)

    import mlx.core as mx

    embeds = text_encoder.encode(
        np.array(ids, dtype=np.int32),
        patches or None,
        grids or None,
    )
    embeds = embeds.astype(mx.bfloat16)
    mx.eval(embeds)
    return embeds, np.array(tags, dtype=np.int32)
