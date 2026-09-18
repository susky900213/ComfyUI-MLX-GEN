"""MiniMax-H3 的规划、内存预检、联合采样与解码。

这里放的是「流程语义」（参考实现的 `variants/minimax_h3.py` 里那部分），节点只做分派：

- `make_plan()`：帧数吸附到 `17n+5`（并提醒）、画布必须是 32 的倍数且比例在 `[1/4, 4]`、
  算出 latent 尺寸与音频 latent 数、`grid_points = steps + 1`（sigma 网格的末端 0 也占一格）；
- `estimate_peak_bytes()` / `preflight()`：峰值内存是**打包行数**的函数（参考机实测
  271 356 字节/行），超限直接拒绝、接近上限给警告；
- `sample()`：打包成一条序列 + 双整流流（视频 shift 12 / 音频 shift 3）锁步去噪，
  每步只有一次 transformer 前向（guidance 蒸馏模型，**没有 CFG**）；
- `decode_video()` / `decode_audio()`：反打包 → latent 反归一化 → VAE 解码 →
  （视频）ImageNet 反归一化到 uint8 帧 / （音频）截断到 `帧数 / 24` 秒。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from comfyui_mlx_gen.h3.latent_creator.h3_layout import (
    AUDIO_CHANNELS,
    FPS,
    KEYFRAME_ENCODE_SEED,
    KEYFRAME_NOISE_AUG,
    MAX_ASPECT_RATIO,
    MAX_DURATION_SECONDS,
    MAX_NUM_FRAMES,
    MIN_ASPECT_RATIO,
    MIN_DURATION_SECONDS,
    MIN_NUM_FRAMES,
    PATCH_SIZE,
    PIXEL_MEAN,
    PIXEL_STD,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    packed_sequence_length,
    patchify_video_latents,
    unpack_audio_rows,
    unpatchify_video_rows,
    valid_frame_counts,
    video_latent_num_frames,
)
from comfyui_mlx_gen.h3.scheduler.minimax_h3_scheduler import MiniMaxH3Scheduler
from comfyui_mlx_gen.h3.video import AudioTrack

CANVAS_MULTIPLE = 32
DEFAULT_VIDEO_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0
RECOMMENDED_FRAMES = 124  # 17 * 7 + 5 帧 = 24 fps 下 5.17 秒
DEFAULT_TEXT_TOKENS = 200  # 估算打包行数用的文本行数（参考实现的标定值）
VAE_SPATIAL_COMPRESSION = 16  # 视频 VAE 的空间压缩比（config.json: 2×2×2×2×1×1）
# transformer 的 latent patch（时间 1、空间 2×2，= transformer.patch_size）：
# 打包行按它切 latent，unpatchify 也必须用它（`PATCH_SIZE` 只是空间那一段，
# 直接传给 `unpatchify_video_rows` 会连三元组都解不出来）
VIDEO_PATCH: tuple[int, int, int] = (1, PATCH_SIZE, PATCH_SIZE)
# 参考机实测的「每行打包序列的峰值字节」（960×544/124 帧与 1344×768/124 帧同一量级）
PEAK_BYTES_PER_PACKED_ROW = 271_356
# 超过物理内存的这个比例就很危险（有实测在 0.73 被系统 kill）
REQUEST_BUDGET_SHARE = 0.72


@dataclass(frozen=True)
class H3Plan:
    """一次 H3 生成的几何与调度参数（全部由 `make_plan` 算出）。"""

    width: int
    height: int
    num_frames: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    grid_points: int  # sigma 网格点数 = transformer 前向次数 + 1
    video_shift: float
    audio_shift: float
    requested_frames: int = 0  # 用户原本填的帧数（吸附前）

    @property
    def steps(self) -> int:
        return self.grid_points - 1

    @property
    def fps(self) -> int:
        return FPS

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / FPS

    def summary(self) -> str:
        return (
            f"{self.width}×{self.height} / {self.num_frames} 帧（{self.duration_seconds:.2f} s @ {FPS}fps）"
            f" / {self.steps} 步 / video_shift {self.video_shift:g} / audio_shift {self.audio_shift:g}"
        )


def make_plan(
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    video_shift: float = DEFAULT_VIDEO_SHIFT,
    audio_shift: float = DEFAULT_AUDIO_SHIFT,
    log: Callable[[str], None] | None = print,
) -> H3Plan:
    """校验画布 / 时长并算出 latent 尺寸与调度器参数（不合规直接报中文错误）。"""
    width, height = int(width), int(height)
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError(
            f"MiniMax-H3 的宽高必须是 {CANVAS_MULTIPLE} 的倍数，收到 {width}×{height}"
            f"（例如 640×352、960×544、1344×768）"
        )
    if height <= 0 or not MIN_ASPECT_RATIO <= width / height <= MAX_ASPECT_RATIO:
        raise ValueError(f"MiniMax-H3 的宽高比必须在 [1/4, 4] 之内，收到 {width}×{height}")

    requested = int(num_frames)
    aligned = align_num_frames(requested)
    duration = aligned / FPS
    # 只接受「17n+5 且落在 124~345」这一串帧数：吸附后不在里面的（比如 350 → 357）
    # 就是超出 H3 的时长窗口，直接拒绝，不要偷偷多跑一帧
    if aligned not in valid_frame_counts():
        raise ValueError(
            f"MiniMax-H3 只生成 {MIN_DURATION_SECONDS:g}~{MAX_DURATION_SECONDS:g} 秒"
            f"（{FPS} fps），帧数必须是 17n+5 且落在 [{MIN_NUM_FRAMES}, {MAX_NUM_FRAMES}]，"
            f"收到 {requested}（吸附后 {aligned}，{duration:.2f} s）。可用帧数："
            f"{'、'.join(str(v) for v in valid_frame_counts())}"
        )
    if aligned != requested and log:
        log(
            f"⚠️ MiniMax-H3 的帧数是 17n+5：{requested} 帧吸附到 {aligned} 帧"
            f"（{duration:.2f} s）"
        )

    steps = int(steps)
    if steps < 1:
        raise ValueError("steps 至少为 1")
    return H3Plan(
        width=width,
        height=height,
        num_frames=aligned,
        requested_frames=requested,
        num_latent_frames=video_latent_num_frames(aligned),
        latent_height=height // VAE_SPATIAL_COMPRESSION,
        latent_width=width // VAE_SPATIAL_COMPRESSION,
        num_audio_latents=audio_latent_num_frames(aligned),
        grid_points=steps + 1,
        video_shift=float(video_shift),
        audio_shift=float(audio_shift),
    )


def packed_rows(plan: H3Plan, text_tokens: int = DEFAULT_TEXT_TOKENS, keyframe_count: int = 0) -> int:
    """打包序列的行数（峰值内存跟着它走，而不是单独跟画布或帧数走）。"""
    condition_rows = int(keyframe_count) * (plan.latent_height // PATCH_SIZE) * (plan.latent_width // PATCH_SIZE)
    return packed_sequence_length(plan.width, plan.height, plan.num_frames, int(text_tokens)) + condition_rows


def estimate_peak_bytes(
    plan: H3Plan,
    resident_bytes: int,
    text_tokens: int = DEFAULT_TEXT_TOKENS,
    cache_bytes: int = 0,
    keyframe_count: int = 0,
) -> int:
    """预期峰值足迹 = 已物化组件 + MLX 缓存 + 每行标定值 × 行数。"""
    return int(
        resident_bytes
        + cache_bytes
        + PEAK_BYTES_PER_PACKED_ROW * packed_rows(plan, text_tokens, keyframe_count)
    )


def preflight(
    plan: H3Plan,
    resident_bytes: int,
    physical_bytes: int,
    text_tokens: int = DEFAULT_TEXT_TOKENS,
    cache_bytes: int = 0,
    keyframe_count: int = 0,
    log: Callable[[str], None] | None = print,
) -> None:
    """峰值放不下就拒绝，接近上限就警告（不改用户的请求，只把四个缩档说清楚）。"""
    if physical_bytes <= 0:
        return
    expected = estimate_peak_bytes(plan, resident_bytes, text_tokens, cache_bytes, keyframe_count)
    gib = 1024**3
    lever = (
        "请缩时长（帧数最小 124）、换小画布（如 640×352）、降量化档位（q4），"
        "或先跑一次小尺寸冒烟"
    )
    shape = (
        f"{plan.width}×{plan.height} / {plan.num_frames} 帧预计峰值约 {expected / gib:.0f} GiB，"
        f"本机物理内存 {physical_bytes / gib:.0f} GiB"
    )
    if expected >= physical_bytes:
        raise MemoryError(f"MiniMax-H3：{shape}，放不下。{lever}")
    if log and expected > physical_bytes * REQUEST_BUDGET_SHARE:
        log(f"⚠️ MiniMax-H3：{shape}，已经很接近上限（可能被系统在采样中途 kill）。{lever}")


def build_layout(
    tags: np.ndarray,
    plan: H3Plan,
    patch_size: tuple[int, int, int],
    audio_channels: int = AUDIO_CHANNELS,
    keyframe_anchors: tuple[str, ...] = (),
):
    """把提示词标签、关键帧条件行及目标音视频行拼成一条序列。"""
    return build_packed_sequence(
        tags,
        plan.num_latent_frames,
        plan.latent_height,
        plan.latent_width,
        plan.num_audio_latents,
        patch_size=patch_size,
        audio_channels=audio_channels,
        keyframe_anchors=keyframe_anchors,
    )


def encode_keyframe_latents(keyframes: tuple[Any, ...], vae: Any) -> tuple[mx.array, ...]:
    """将画布适配后的 PIL 关键帧编码成归一化的单帧 Video VAE latent。

    每张图都复现官方单关键帧路径：ImageNet 归一化、posterior 用固定 seed 42
    采样、经 float16 舍入后再按 VAE 的 mean/std 归一化。结果不进入长期缓存。
    """
    encoded: list[mx.array] = []
    for frame in keyframes:
        pixels = np.asarray(frame, dtype=np.float32) / 255.0
        pixels = (pixels - np.array(PIXEL_MEAN, dtype=np.float32)) / np.array(PIXEL_STD, dtype=np.float32)
        tensor = mx.array(pixels.transpose(2, 0, 1))[None, :, None]
        mean, logvar = vae.encode(tensor.astype(component_dtype(vae)))
        mean, logvar = mean.astype(mx.float32), logvar.astype(mx.float32)
        noise = mx.random.normal(
            mean.shape,
            key=mx.random.key(KEYFRAME_ENCODE_SEED),
            dtype=mx.float32,
        )
        latents = (mean + mx.exp(0.5 * logvar) * noise).astype(mx.float16).astype(mx.float32)
        latents = (latents - vae.latents_mean.reshape(1, -1, 1, 1, 1)) / vae.latents_std.reshape(
            1, -1, 1, 1, 1
        )
        mx.eval(latents)
        encoded.append(latents)
    return tuple(encoded)


def sample(
    transformer: Any,
    prompt_embeds: Any,
    tags: np.ndarray,
    plan: H3Plan,
    seed: int,
    keyframe_latents: tuple[mx.array, ...] = (),
    keyframe_anchors: tuple[str, ...] = (),
    log: Callable[[str], None] | None = print,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[mx.array, mx.array, Any]:
    """联合去噪：返回 `(video_rows, audio_rows, layout)`（都是未反归一化的 latent 行）。"""
    if len(keyframe_latents) != len(keyframe_anchors):
        raise ValueError(
            "H3 关键帧 latent 与锚点数量不一致："
            f"{len(keyframe_latents)} != {len(keyframe_anchors)}"
        )
    layout = build_layout(
        tags,
        plan,
        tuple(transformer.patch_size),
        keyframe_anchors=keyframe_anchors,
    )
    video_scheduler = MiniMaxH3Scheduler(shift=plan.video_shift)
    audio_scheduler = MiniMaxH3Scheduler(shift=plan.audio_shift)
    video_scheduler.set_timesteps(plan.grid_points)
    audio_scheduler.set_timesteps(plan.grid_points)

    # 抽噪顺序与参考实现一致：整组条件增强噪声在前，随后是目标视频、目标音频。
    keys = mx.random.split(mx.random.key(int(seed)), len(keyframe_latents) + 2)
    key_video, key_audio = keys[-2], keys[-1]
    video_rows = patchify_video_latents(
        mx.random.normal(
            (1, transformer.in_channels, plan.num_latent_frames, plan.latent_height, plan.latent_width),
            key=key_video,
            dtype=mx.float32,
        ),
        tuple(transformer.patch_size),
    )
    if keyframe_latents:
        condition_rows = []
        for condition, key in zip(keyframe_latents, keys[:-2], strict=True):
            expected = (1, int(transformer.in_channels), 1, plan.latent_height, plan.latent_width)
            if tuple(condition.shape) != expected:
                raise ValueError(
                    f"H3 关键帧 latent 形状应为 {expected}，收到 {tuple(condition.shape)}"
                )
            condition_noise = mx.random.normal(condition.shape, key=key, dtype=mx.float32)
            noised = video_scheduler.scale_noise(condition, KEYFRAME_NOISE_AUG, condition_noise)
            condition_rows.append(patchify_video_latents(noised, tuple(transformer.patch_size)))
        video_rows = mx.concatenate([*condition_rows, video_rows], axis=0)
    audio_rows = mx.random.normal(
        (plan.num_audio_latents * AUDIO_CHANNELS, transformer.audio_in_channels),
        key=key_audio,
        dtype=mx.float32,
    )

    condition_video_rows = int(layout.num_condition_video_rows)
    condition_audio_rows = int(layout.num_condition_audio_rows)
    total = len(video_scheduler.timesteps)
    for step, (video_t, audio_t) in enumerate(zip(video_scheduler.timesteps, audio_scheduler.timesteps)):
        unique_timesteps, timestep_indices = build_row_timesteps(
            layout,
            float(video_t),
            float(audio_t),
            max(float(video_t), KEYFRAME_NOISE_AUG),
            1.0,
        )
        video_pred, audio_pred = transformer(
            hidden_states=video_rows[None],
            audio_hidden_states=audio_rows[None],
            encoder_hidden_states=prompt_embeds,
            timestep=unique_timesteps,
            timestep_indices=timestep_indices,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
        )
        # 关键帧条件行保持在 t = 0.999（纯文生视频时这两段切片就是整段）
        video_rows[condition_video_rows:] = video_scheduler.step(
            video_pred[0, condition_video_rows:].astype(mx.float32), step, video_rows[condition_video_rows:]
        )
        audio_rows[condition_audio_rows:] = audio_scheduler.step(
            audio_pred[0, condition_audio_rows:].astype(mx.float32), step, audio_rows[condition_audio_rows:]
        )
        mx.eval(video_rows, audio_rows)
        if on_progress is not None:
            on_progress(step + 1, total)
        if log:
            log(f"[H3 采样] step {step + 1}/{total} t={float(video_t):.3f}")
    # 条件行只服务于 transformer 上下文，不能进入最终 latent：decode_video() 会严格按
    # plan.num_latent_frames 反打包目标行。纯 T2V 时两个 offset 都是 0，行为保持不变。
    return video_rows[condition_video_rows:], audio_rows[condition_audio_rows:], layout


def component_dtype(module: Any) -> mx.Dtype:
    """模块参数的浮点精度（量化层看 scales）；没有浮点参数时退回 bfloat16。"""
    for _, value in tree_flatten(module.parameters()):
        if value.dtype in (mx.float32, mx.bfloat16, mx.float16):
            return value.dtype
    return mx.bfloat16


def decode_video(video_rows: Any, plan: H3Plan, vae: Any) -> np.ndarray:
    """视频行 → `(F, H, W, 3)` uint8（latent 反归一化 → VAE 解码 → ImageNet 反归一化）。

    几何全部取自 `plan`（画布 / 16、`17n+5` → `5n+2`），VAE 只负责解码；
    这样解码节点不需要拿到 transformer（与参考实现一致）。
    """
    latents = unpatchify_video_rows(
        video_rows,
        int(vae.latent_channels),
        plan.num_latent_frames,
        plan.latent_height,
        plan.latent_width,
        VIDEO_PATCH,
    )
    latents = latents * vae.latents_std.reshape(1, -1, 1, 1, 1) + vae.latents_mean.reshape(1, -1, 1, 1, 1)
    pixels = vae.decode(latents.astype(component_dtype(vae)))
    pixels = pixels.astype(mx.float32) * mx.array(PIXEL_STD).reshape(1, 3, 1, 1, 1) + mx.array(PIXEL_MEAN).reshape(
        1, 3, 1, 1, 1
    )
    pixels = mx.clip(pixels * 255.0 + 0.5, 0, 255).astype(mx.uint8)[0].transpose(1, 2, 3, 0)
    mx.eval(pixels)
    return np.array(pixels)


def decode_audio(audio_rows: mx.array, plan: H3Plan, audio_vae: Any) -> AudioTrack:
    """音频行 → `(2, samples)` 波形（截断到 `帧数 / 24` 秒）。"""
    latents = unpack_audio_rows(audio_rows, AUDIO_CHANNELS, plan.num_audio_latents)
    latents = latents * audio_vae.latents_std.reshape(1, -1, 1) + audio_vae.latents_mean.reshape(1, -1, 1)
    waveform = audio_vae.decode(latents.astype(mx.float32))
    mx.eval(waveform)
    track = AudioTrack(
        waveform=np.array(waveform[:, 0, :], dtype=np.float32),
        sample_rate=int(audio_vae.sampling_rate),
    )
    return track.trimmed(plan.duration_seconds)


