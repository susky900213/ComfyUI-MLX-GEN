"""YuE2 文本 → 语义 codec → 声学 latent 的本地 MLX 生成流程。

本模块只生成声学 latent；VAE 解码由现有 ``MlxVAEDecoder`` 节点触发，因而不会
在采样阶段同时常驻主模型和 VAE，也不调用外部服务或 HTTP 接口。
"""

from __future__ import annotations

import base64
import json
import math
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from .model import KVCache, Yue2Model
from .vae import audio_samples

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
VOCAB, CONTEXT = 184704, 24576
SAMPLE_RATE = 48000
INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": (
        "Generate a melody-only ABC transcription without chord symbols, then generate music "
        "with codec tokens from the given conditions."
    ),
    "full": (
        "Generate a chord-annotated ABC transcription, then generate music with codec tokens "
        "from the given conditions."
    ),
}


class Tokenizer:
    """YuE2 固定的 Qwen 文本 / ABC BPE（不是音频 codec）。"""

    def __init__(self, merge_file: Path):
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "YuE2 需要 tiktoken；请在 ComfyUI 的 Python 环境执行 "
                "pip install -r ComfyUI-MLX-GEN/requirements.txt"
            ) from exc
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (
                line.split() for line in Path(merge_file).read_bytes().splitlines() if line
            )
        }
        specials = [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<R>",
            "<S>",
            "<X>",
            "<mask>",
            "<sep>",
        ]
        specials += [f"<extra_{index}>" for index in range(200)]
        specials[204:206] = ["<abc>", "</abc>"]
        pattern = (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
            r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        )
        self._encoding = tiktoken.Encoding(
            "YuE2",
            pat_str=pattern,
            mergeable_ranks=ranks,
            special_tokens={token: index + len(ranks) for index, token in enumerate(specials)},
        )

    def encode(self, text):
        return self._encoding.encode_ordinary(unicodedata.normalize("NFC", text))

    def decode(self, ids):
        valid = [int(token) for token in ids if 0 <= token < self._encoding.n_vocab]
        return self._encoding.decode(valid, errors="replace")


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 100
    repetition_penalty: float = 1.2
    penalty_window: int = 50
    min_tokens: int = 200
    max_tokens: int = 9000


def load_generation_config(path: Path) -> dict[str, Any]:
    return json.loads((Path(path) / "yue2_generation_config.json").read_text())


def request_text(style, lyrics, cot):
    return f"{INSTRUCTIONS[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"


def token_prefix(tokenizer, style, lyrics, cot, abc_ids=None):
    base = [EOD] + tokenizer.encode(request_text(style, lyrics, cot))
    if cot == "off":
        return base + [ABC_START, ABC_END, MUSIC_START]
    if abc_ids is None:
        return base + [ABC_START]
    return base + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


def negative_prefix(tokenizer, cot, abc_ids):
    base = [EOD] + tokenizer.encode(INSTRUCTIONS[cot])
    if cot == "off":
        return base + [MUSIC_START]
    return base + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


def allowed_mask(phase, dtype):
    indices = mx.arange(VOCAB)
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = (
        (indices < EOD)
        if phase == "abc"
        else ((indices >= CODEC_OFFSET) & (indices < CODEC_OFFSET + CODEC_SIZE))
    )
    return mx.where(allowed | (indices == end), 0.0, -mx.inf).astype(dtype)


def distribution(logits, sampling, history, step, phase, allowed, legacy_off):
    scores = logits if legacy_off else logits.astype(mx.float32)
    end = ABC_END if phase == "abc" else MUSIC_END
    scores = scores + allowed
    if step < sampling.min_tokens:
        scores[end] = -mx.inf
    recent = history[-sampling.penalty_window :]
    if sampling.repetition_penalty != 1.0 and recent:
        frequency = mx.zeros((VOCAB,), scores.dtype).at[mx.array(recent)].add(1)
        alpha = mx.power(mx.array(sampling.repetition_penalty, scores.dtype), frequency)
        scores = mx.where(scores < 0, scores * alpha, scores / alpha)
    if sampling.temperature == 0:
        return scores
    if sampling.temperature != 1:
        scores = scores / sampling.temperature
    kth = VOCAB - min(sampling.top_k, VOCAB)
    threshold = mx.partition(scores, kth)[kth]
    scores = mx.where(scores < threshold, -mx.inf, scores)
    if sampling.top_p < 1:
        order = mx.argsort(-scores)
        values = scores[order]
        probabilities = mx.softmax(values.astype(mx.float32), axis=-1)
        removed = (mx.cumsum(probabilities) - probabilities) > sampling.top_p
        removed = removed & (mx.arange(VOCAB) >= (3 if legacy_off else 1))
        scores = mx.put_along_axis(
            scores, order, mx.where(removed, -mx.inf, values), axis=0
        )
    return scores


def generate_tokens(
    model: Yue2Model,
    prefix,
    sampling: Sampling,
    seed,
    phase,
    negative=None,
    cfg_scale=1.0,
    legacy_off=False,
    on_token=None,
):
    """生成 ABC 或语义 codec token，返回 ``(tokens, 是否达到长度上限)``。"""
    if len(prefix) + sampling.max_tokens > CONTEXT or (
        negative and len(negative) + sampling.max_tokens > CONTEXT
    ):
        raise ValueError("YuE2 提示词 + 生成长度超过 24576 token 上下文")
    if cfg_scale != 1 and negative is None:
        raise ValueError("YuE2 CFG 需要无条件前缀")
    layers = len(model.model.layers)
    caches = [
        [KVCache() for _ in range(layers)] for _ in range(2 if cfg_scale != 1 else 1)
    ]
    conditional = model.ar_step(mx.array([prefix]), caches[0])
    unconditional = (
        model.ar_step(mx.array([negative]), caches[1]) if cfg_scale != 1 else None
    )
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = allowed_mask(phase, conditional.dtype if legacy_off else mx.float32)
    key = mx.random.key(seed)
    history, eos = [], False
    for step in range(sampling.max_tokens):
        logits = (
            conditional
            if cfg_scale == 1
            else unconditional + cfg_scale * (conditional - unconditional)
        )
        scores = distribution(
            logits[0], sampling, history, step, phase, allowed, legacy_off
        )
        if sampling.temperature == 0:
            token = int(mx.argmax(scores).item())
        else:
            key, subkey = mx.random.split(key)
            token = int(mx.random.categorical(scores.astype(mx.float32), key=subkey).item())
        if on_token is not None:
            on_token(phase, token)
        if token == end:
            eos = True
            break
        history.append(token)
        if step + 1 < sampling.max_tokens:
            next_token = mx.array([[token]])
            conditional = model.ar_step(next_token, caches[0])
            if unconditional is not None:
                unconditional = model.ar_step(next_token, caches[1])
    return history, not eos


def chunk_ranges(frames, prefix_tokens, context=CONTEXT):
    size = min((context - prefix_tokens - 3) // 2, CONTEXT)
    if frames < 1 or size < 1:
        raise ValueError("YuE2 没有生成 codec token，或提示词未给声学上下文留下空间")
    return [(start, min(start + size, frames)) for start in range(0, frames, size)]


def _logit(value):
    if 0 < value < 1:
        return max(-20.0, min(20.0, math.log(value / (1 - value))))
    return 20.0 if value >= 1 else -20.0


def synthesize(model, prefix, codec, seed, steps=32, noise=None, on_progress=None):
    """midpoint ODE：从语义 codec 生成 ``[frames,64]`` float32 声学 latent。"""
    if steps < 1:
        raise ValueError("YuE2 NAR steps 必须至少为 1")
    if noise is None:
        noise = mx.random.normal((len(codec), 64), key=mx.random.key(seed))
    delta = 1.0 / steps
    output = []
    ranges = chunk_ranges(len(codec), len(prefix))
    total_steps = len(ranges) * steps
    completed_steps = 0
    for start, end in ranges:
        ar_tokens = prefix + [token + CODEC_OFFSET for token in codec[start:end]] + [MUSIC_END]
        ar_cache = model.nar_prefill(ar_tokens)
        state = noise[start:end].astype(mx.bfloat16)
        for step in range(steps):
            timestep = 1.0 - step * delta
            velocity = model.nar_velocity(state, _logit(timestep), ar_cache, len(ar_tokens))
            midpoint = state - velocity * (delta / 2)
            state = state - model.nar_velocity(
                midpoint, _logit(timestep - delta / 2), ar_cache, len(ar_tokens)
            ) * delta
            mx.eval(state)
            completed_steps += 1
            if on_progress is not None:
                on_progress(completed_steps, total_steps)
        output.append(state.astype(mx.float32))
    return mx.concatenate(output)


def _progress_logger(label: str, log: Callable[[str], None]):
    count, start = [0], time.perf_counter()

    def on_token(_phase, _token):
        count[0] += 1
        if count[0] % 200 == 0:
            elapsed = max(time.perf_counter() - start, 1e-6)
            log(f"[YuE2 {label}] {count[0]} tokens，{count[0] / elapsed:.1f} tok/s")

    return on_token


def generate_music_latents(
    model: Yue2Model,
    tokenizer: Tokenizer,
    generation_config: dict[str, Any],
    style: str,
    lyrics: str,
    *,
    cot: str = "full",
    seed: int = 0,
    steps: int | None = None,
    max_tokens: int | None = None,
    cfg_scale: float | None = None,
    log: Callable[[str], None] = print,
    on_progress: Callable[[int, int], None] | None = None,
):
    """执行 YuE2 的 ABC 规划、语义 AR 与 NAR，返回 latent 和诊断信息。"""
    if cot not in INSTRUCTIONS:
        raise ValueError("YuE2 cot 必须是 off、melody 或 full")
    if not style.strip():
        raise ValueError("YuE2 的 style 不能为空（请写在正向 MlxTextEncoder 中）")
    abc_sampling = Sampling(**generation_config["abc"])
    semantic_sampling = Sampling(**generation_config["semantic"])
    if max_tokens is not None:
        limit = int(max_tokens)
        if not 200 <= limit <= 9000:
            raise ValueError("YuE2 max_tokens 必须在 200..9000 之间")
        semantic_sampling = Sampling(
            **{
                **semantic_sampling.__dict__,
                "max_tokens": limit,
                "min_tokens": min(semantic_sampling.min_tokens, limit),
            }
        )

    abc_ids, abc_text = [], None
    if cot != "off":
        log(f"[YuE2 plan] 正在生成 {cot} ABC 乐谱")
        abc_ids, truncated = generate_tokens(
            model,
            token_prefix(tokenizer, style, lyrics, cot),
            abc_sampling,
            seed,
            "abc",
            on_token=_progress_logger("abc", log),
        )
        abc_text = tokenizer.decode(abc_ids)
        if truncated:
            log("[YuE2 plan] ABC 达到 token 上限")

    prefix = token_prefix(tokenizer, style, lyrics, cot, abc_ids)
    guidance = (1.01 if cot == "off" else 1.0) if cfg_scale is None else float(cfg_scale)
    if not 1.0 <= guidance <= 5.0:
        raise ValueError("YuE2 guidance（CFG scale）必须在 1.0..5.0 之间")
    negative = negative_prefix(tokenizer, cot, abc_ids) if guidance != 1 else None
    log(f"[YuE2 semantic] prefix={len(prefix)} tokens，CFG={guidance:g}")
    semantic_ids, truncated = generate_tokens(
        model,
        prefix,
        semantic_sampling,
        seed,
        "semantic",
        negative,
        guidance,
        legacy_off=cot == "off",
        on_token=_progress_logger("semantic", log),
    )
    if truncated:
        log("[YuE2 semantic] 达到 max_tokens")
    codec = [token - CODEC_OFFSET for token in semantic_ids]
    if not codec:
        raise RuntimeError("YuE2 语义阶段没有生成 codec token")

    nar_steps = int(steps or generation_config.get("ode_steps", 32))
    duration = audio_samples(len(codec)) / SAMPLE_RATE
    log(f"[YuE2 NAR] {len(codec)} 帧（约 {duration:.1f}s），{nar_steps} 步 midpoint")

    def report_nar_progress(index: int, total: int) -> None:
        if on_progress is not None:
            on_progress(index, total)
        if index == total or index % 8 == 0:
            log(f"[YuE2 NAR] step {index}/{total}")

    latents = synthesize(
        model,
        prefix,
        codec,
        seed,
        nar_steps,
        on_progress=report_nar_progress,
    )
    return latents, {
        "abc": abc_text,
        "codec_frames": len(codec),
        "duration": duration,
        "cot": cot,
    }