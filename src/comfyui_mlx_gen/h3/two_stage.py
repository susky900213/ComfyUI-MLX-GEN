"""MiniMax-H3 二阶段采样（低分前半段 → latent 放大 → 高分精修）。

本模块**只新增**，不改动 `h3/pipeline.py`、`h3/scheduler/`、`h3/latent_creator/` 里的
任何既有函数；所需的编排全部由既有公开入口组装：

- 几何：`h3.pipeline.make_plan` / `build_layout` / `preflight`；
- 调度：`MiniMaxH3Scheduler` —— 新模块只**覆写**实例的 `sigmas` / `timesteps` 公开属性，
  复用同一套 `scale_noise` / `step` 语义跑子网格；
- 采样循环：与 `h3.pipeline.sample` 同构的一份实现，额外支持「注入初始 latent」
  与「视频 / 音频各自起跑 σ」；当不注入初始 latent 且 σ 网格取母网格整段时，
  逐位等价于既有实现（`tests/test_h3_two_stage.py` 用这一点做回归）。

约定（与参考实现一致）

- σ 网格来自 `p ∈ [0, 1]` 的线性刻度经 exp shift 映射：`σ(p) = shift·p / (1 + (shift-1)·p)`；
- 视频 shift = 12.0、音频 shift = 3.0，两模态共用同一 p 刻度、按索引锁步；
- 一阶段停在**母网格**的第 k 个点（8 步母网格停在第 4 点 → 视频 σ=0.9231 / 音频 σ=0.7500）；
- 二阶段用「同一 p 刻度 + 各自 shift」重建子网格，因此 `start_at_sigma=0.9231`
  与 ComfyUI 的 `SplitSigmas(step=4)` 逐点等价（index 4 就是 p=0.5）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from comfyui_mlx_gen.h3.latent_creator.h3_layout import (
    AUDIO_CHANNELS,
    KEYFRAME_NOISE_AUG,
    build_row_timesteps,
    patchify_video_latents,
)
from comfyui_mlx_gen.h3.pipeline import build_layout
from comfyui_mlx_gen.h3.scheduler.minimax_h3_scheduler import MiniMaxH3Scheduler

# 二阶段元数据桶：键与 h3_latents 桶里的 latent 键**相同**，值是一份普通 dict。
# 故意不复用 MlxLatentHandle（types.py 的既有 frozen dataclass），也不改它。
META_BUCKET = "h3_two_stage_meta"
# 视觉条件图的桶名与 pipeline.H3_VISUAL_BUCKET 一致（这里只读，不写既有桶的旧键）
AUDIO_MODES = ("follow_video", "continue_stage1", "regenerate")


# --- σ 刻度 ----------------------------------------------------------------------
def sigma_to_p(sigma: float, shift: float) -> float:
    """σ → p（exp shift 映射的逆）。σ=1 对应 p=1（纯噪声），σ=0 对应 p=0。"""
    sigma, shift = float(sigma), float(shift)
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"σ 必须落在 [0, 1]，收到 {sigma}")
    return sigma / (shift - (shift - 1.0) * sigma)


def p_to_sigma(p: float, shift: float) -> float:
    """p ∈ [0, 1] → σ = shift·p / (1 + (shift-1)·p)（与 MiniMaxH3Scheduler 同式）。"""
    p, shift = float(p), float(shift)
    return shift * p / (1.0 + (shift - 1.0) * p)


def _check_shift(shift: float, name: str = "shift") -> float:
    """两条整流流的 shift 必须 > 0（H3 的视频 12.0 / 音频 3.0）。

    滑杆允许 0，而 `MiniMaxH3Scheduler` 会直接抛英文的 ``shift must be positive`` ——
    这里先给一句能照着改的中文提示（0 / 负数都不是「不位移」的意思）。
    """
    value = float(shift)
    if value <= 0.0:
        raise ValueError(
            f"{name} 必须大于 0，收到 {value:g} —— H3 是双整流流模型，"
            "视频 shift 默认 12.0、音频 shift 默认 3.0（0 不是「不位移」，是非法值）"
        )
    return value


def axis_from_p(shift: float, p_start: float, steps: int) -> np.ndarray:
    """在 `[p_start, 0]` 上取 `steps + 1` 个 p 并映射成 σ（末项恒为 0）。"""
    _check_shift(shift)
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps 至少为 1，收到 {steps}")
    ps = np.linspace(float(p_start), 0.0, steps + 1, dtype=np.float64)
    sigmas = np.array([p_to_sigma(p, shift) for p in ps], dtype=np.float32)
    sigmas[-1] = np.float32(0.0)
    return sigmas


def refine_axis(shift: float, steps: int, start_at_sigma: float) -> np.ndarray:
    """精修子网格：起点 σ = start_at_sigma、终点 0，保持同一 shift 的形状。"""
    return axis_from_p(_check_shift(shift), sigma_to_p(start_at_sigma, shift), steps)


def stage_order_note(start_at_sigma: float, stage1_video_sigma: float | None) -> str:
    """二阶段起点高于一阶段停点时的提醒（返回空串表示没问题）。

    一阶段停下时给下游的是**该 σ 处的 x0 估计**；二阶段会用 `start_at_sigma` 重新加噪：
    `x = (1-σ)·x0 + σ·noise`。若 `start_at_sigma` 比一阶段停下的 σ 还高，等于把刚算出来的
    x0 又推回更高的噪声水平（x0 的权重更小），放大 / 精修的效果会明显变弱。
    """
    if stage1_video_sigma is None:
        return ""
    if float(start_at_sigma) > float(stage1_video_sigma) + 1e-4:
        return (
            f"二阶段起点 σ={float(start_at_sigma):.4f} 高于一阶段停下的 σ={float(stage1_video_sigma):.4f}："
            f"会把刚算出的 x0 重新加噪到更高噪声水平（x0 权重只有 "
            f"{(1.0 - float(start_at_sigma)) * 100:.1f}%），放大/精修效果会明显变弱。"
            "建议 start_at_sigma ≤ 一阶段停下的 σ，或把一阶段的 stop_at_step 调大"
        )
    return ""


def tail_points_note(
    video_shift: float,
    total_steps: int,
    start_at_sigma: float,
    refine_steps: int,
    *,
    distilled: bool = False,
) -> str:
    """二阶段步数不够走完「母网格在 σ 之后的原生台阶」时的提醒（空串 = 没问题）。

    在 σ=0.9231 处只给 1 步，等于要求模型在 92% 噪声下**一步吐出成品**：蒸馏适配器在这个 σ
    的预测本来就是「还有几步要走」的中间估计，直接解码就是**雪花 / 彩色噪点**（实测踩过）。
    `distilled=True`（一阶段的 model 挂了 LoRA）时按原生台阶数严格提醒；
    没有 LoRA 的基座模型（40~50 步）只在「只剩 1 步」这种极端情况下提醒。
    """
    refine_steps = int(refine_steps)
    if refine_steps <= 0:
        return ""
    grid = np.asarray(first_pass_axis(video_shift, int(total_steps), 0), dtype=np.float32)
    # 容差要大于 float32 的表示误差（0.9231 存成 0.9230769…），否则会把起点自己算进去
    native = int(np.sum(grid < float(start_at_sigma) - 1e-4))
    if native <= 0 or refine_steps >= native:
        return ""
    if not (distilled or refine_steps <= 1):
        return ""
    return (
        f"二阶段只有 {refine_steps} 步，而母网格在 σ={float(start_at_sigma):.4f} 之后还有 {native} 段原生台阶："
        f"模型得在一次跳跃里走完（蒸馏适配器在这个 σ 的预测只是中间估计），成片会像雪花 / 噪点。"
        f"建议 steps ≥ {native}（或把一阶段的 stop_at_step 调大、让切点落在更低的 σ）"
    )


def first_pass_axis(shift: float, total_steps: int, stop_at_step: int) -> np.ndarray:
    """一阶段：**母网格**（`total_steps` 步）的前 `stop_at_step` 步；0 = 跑满。

    注意必须保留母网格：`stop=4 / total=8` 停在 σ=0.9231，而「4 步调度」的 σ 是
    `[1.0, .973, .9231, .8, 0]` —— 两者完全不同，只有母网格才能让二阶段接着走。
    """
    total_steps = int(total_steps)
    scheduler = MiniMaxH3Scheduler(shift=_check_shift(shift))
    scheduler.set_timesteps(total_steps + 1)
    grid = np.asarray(scheduler.sigmas, dtype=np.float32)
    stop = total_steps if int(stop_at_step) <= 0 else int(stop_at_step)
    if not 1 <= stop <= len(grid) - 1:
        raise ValueError(
            f"stop_at_step 必须在 1~{len(grid) - 1} 之间（0 = 跑满 {total_steps} 步），收到 {stop_at_step}"
        )
    return grid[: stop + 1]


def scheduler_with(shift: float, sigmas: np.ndarray) -> MiniMaxH3Scheduler:
    """建一个换过 σ / timestep 的调度器（只写公开属性，不改调度器实现）。"""
    scheduler = MiniMaxH3Scheduler(shift=float(shift))
    scheduler.sigmas = np.asarray(sigmas, dtype=np.float32)
    scheduler.timesteps = (np.float32(1.0) - scheduler.sigmas[:-1]).astype(np.float32)
    scheduler.num_inference_steps = int(scheduler.timesteps.shape[0])
    return scheduler


# --- 二阶段的两条子网格 + 音频模式 ------------------------------------------------
@dataclass(frozen=True)
class RefineAxes:
    """二阶段的调度参数（视频 / 音频各自一条子网格 + 音频怎么处理）。"""

    video: np.ndarray
    audio: np.ndarray
    audio_reenoise: bool  # True = 把一阶段音频行当 x0 重新加噪到 audio[0]
    audio_mode: str  # 实际生效的模式（可能是回退后的值）
    audio_start_sigma: float

    def summary(self) -> str:
        tail = "重加噪" if self.audio_reenoise else "不重加噪（接着走）"
        return (
            f"video σ {float(self.video[0]):.4f}→0（{len(self.video) - 1} 步）"
            f" | audio[{self.audio_mode}] σ {float(self.audio[0]):.4f}→0（{tail}）"
        )


def refine_axes(
    video_shift: float,
    audio_shift: float,
    steps: int,
    start_at_sigma: float,
    audio_mode: str,
    *,
    stage1_audio_sigma: float | None = None,
    log: Callable[[str], None] | None = print,
) -> RefineAxes:
    """二阶段的两条 σ 子网格。

    - `follow_video`：音频走「同一 p 刻度」→ σ_a = shift3(p_v)。σ_v=0.7 时 σ_a=0.3684；
      若 σ_v 取母网格切点 0.9231，则 σ_a=0.75，与 `SplitSigmas(step=4)` 完全一致；
    - `continue_stage1`：音频从**一阶段停下时**的 σ_a（母网格切点）继续，不重加噪；
      拿不到一阶段元数据时回退到 `follow_video` 并打印提示；
    - `regenerate`：音频 σ_a 从 1.0 起（= 纯噪声重画，步数只有二阶段这么多）。
    """
    mode = str(audio_mode)
    if mode not in AUDIO_MODES:
        raise ValueError(f"audio_mode 只能是 {' / '.join(AUDIO_MODES)}，收到 {mode!r}")
    if not 0.0 < float(start_at_sigma) <= 1.0:
        raise ValueError(f"start_at_sigma 必须落在 (0, 1]，收到 {start_at_sigma}")
    steps = int(steps)
    video_axis = refine_axis(video_shift, steps, float(start_at_sigma))

    if mode == "regenerate":
        audio_axis = axis_from_p(audio_shift, 1.0, steps)
        return RefineAxes(video_axis, audio_axis, True, mode, float(audio_axis[0]))

    if mode == "continue_stage1" and stage1_audio_sigma is None:
        if log:
            log(
                "[H3 二阶段] audio_mode=continue_stage1 但拿不到一阶段的音频 σ"
                "（latents 不是本插件二阶段链路产出的），已回退到 follow_video"
            )
        mode = "follow_video"

    if mode == "continue_stage1":
        audio_axis = refine_axis(audio_shift, steps, float(stage1_audio_sigma))
        return RefineAxes(video_axis, audio_axis, False, mode, float(audio_axis[0]))

    p_start = sigma_to_p(float(start_at_sigma), float(video_shift))
    audio_axis = axis_from_p(audio_shift, p_start, steps)
    return RefineAxes(video_axis, audio_axis, True, "follow_video", float(audio_axis[0]))


# --- 采样循环（与 h3.pipeline.sample 同构 + init 注入）----------------------------
def denoise(
    transformer: Any,
    prompt_embeds: Any,
    tags: np.ndarray,
    plan: Any,
    seed: int,
    *,
    video_axis: np.ndarray,
    audio_axis: np.ndarray,
    init_video_rows: Any | None = None,
    init_audio_rows: Any | None = None,
    audio_reenoise: bool = True,
    keyframe_latents: tuple[Any, ...] = (),
    keyframe_anchors: tuple[str, ...] = (),
    output: str = "latent",
    label: str = "H3 采样",
    log: Callable[[str], None] | None = print,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[Any, Any, Any]:
    """去噪：返回 `(video_rows, audio_rows, layout)`（未反归一化的 latent 行）。

    `init_*` 为 None 时从纯噪声起跑（要求对应子网格首项 σ=1）；否则按
    `x = (1-σ0)·x0 + σ0·噪声` 注入（flow matching 的前向加噪，与既有
    `scale_noise` 同一公式）。音频行在 `audio_reenoise=False` 时直接沿用 init 内容。

    `output`：
    - `"latent"` = 最后一步 Euler 走完的 x_t（σ=0 时就是成品）；
    - `"denoised"` = **最后一个工作 σ 处的一步 x0 估计** `x0 = x_t + σ·v`，也就是
      ComfyUI `SamplerCustomAdvanced` 的 `denoised_output`。喂给 latent 放大网络
      必须用这一路：放大网络是在**干净** latent 对上训练的，拿 92% 是噪声的 x_t
      去放大只会把噪声放大 → 二阶段重采样后画面就是混沌的（实测踩过）。
      该估计复用最后一步的模型输出，**不额外增加一次前向**。
    """
    if output not in ("latent", "denoised"):
        raise ValueError(f"output 只能是 latent / denoised，收到 {output!r}")
    video_axis = np.asarray(video_axis, dtype=np.float32)
    audio_axis = np.asarray(audio_axis, dtype=np.float32)
    if video_axis.shape != audio_axis.shape:
        raise ValueError(
            f"视频与音频的 σ 子网格必须同长（锁步）：{video_axis.shape} vs {audio_axis.shape}"
        )
    if len(keyframe_latents) != len(keyframe_anchors):
        raise ValueError(
            f"H3 关键帧 latent 与锚点数量不一致：{len(keyframe_latents)} != {len(keyframe_anchors)}"
        )
    if init_video_rows is None and float(video_axis[0]) != 1.0:
        raise ValueError(f"从纯噪声起跑要求视频 σ 首项为 1.0，收到 {float(video_axis[0]):.4f}")
    if init_audio_rows is None and float(audio_axis[0]) != 1.0:
        raise ValueError(f"从纯噪声起跑要求音频 σ 首项为 1.0，收到 {float(audio_axis[0]):.4f}")

    layout = build_layout(
        tags,
        plan,
        tuple(transformer.patch_size),
        keyframe_anchors=keyframe_anchors,
    )
    video_scheduler = scheduler_with(plan.video_shift, video_axis)
    audio_scheduler = scheduler_with(plan.audio_shift, audio_axis)

    # 抽噪顺序与参考实现一致：整组条件增强噪声在前，随后是目标视频、目标音频
    keys = mx.random.split(mx.random.key(int(seed)), len(keyframe_latents) + 2)
    key_video, key_audio = keys[-2], keys[-1]

    video_shape = (
        1,
        int(transformer.in_channels),
        plan.num_latent_frames,
        plan.latent_height,
        plan.latent_width,
    )
    if init_video_rows is None:
        video_rows = patchify_video_latents(
            mx.random.normal(video_shape, key=key_video, dtype=mx.float32),
            tuple(transformer.patch_size),
        )
    else:
        expected = (
            plan.num_latent_frames * (plan.latent_height // 2) * (plan.latent_width // 2),
            int(transformer.in_channels) * 4,
        )
        if tuple(init_video_rows.shape) != expected:
            raise ValueError(
                f"初始视频行形状应为 {expected}，收到 {tuple(init_video_rows.shape)}"
                "（放大后的画布 / 帧数与一阶段不一致？）"
            )
        noise = mx.random.normal(init_video_rows.shape, key=key_video, dtype=mx.float32)
        video_rows = video_scheduler.scale_noise(
            init_video_rows.astype(mx.float32), float(1.0 - video_axis[0]), noise
        )

    if keyframe_latents:
        condition_rows = []
        for condition, key in zip(keyframe_latents, keys[:-2], strict=True):
            expected = (1, int(transformer.in_channels), 1, plan.latent_height, plan.latent_width)
            if tuple(condition.shape) != expected:
                raise ValueError(f"H3 关键帧 latent 形状应为 {expected}，收到 {tuple(condition.shape)}")
            condition_noise = mx.random.normal(condition.shape, key=key, dtype=mx.float32)
            noised = video_scheduler.scale_noise(condition, KEYFRAME_NOISE_AUG, condition_noise)
            condition_rows.append(patchify_video_latents(noised, tuple(transformer.patch_size)))
        video_rows = mx.concatenate([*condition_rows, video_rows], axis=0)

    audio_shape = (plan.num_audio_latents * AUDIO_CHANNELS, int(transformer.audio_in_channels))
    if init_audio_rows is None:
        audio_rows = mx.random.normal(audio_shape, key=key_audio, dtype=mx.float32)
    elif audio_reenoise:
        if tuple(init_audio_rows.shape) != audio_shape:
            raise ValueError(f"初始音频行形状应为 {audio_shape}，收到 {tuple(init_audio_rows.shape)}")
        noise = mx.random.normal(audio_shape, key=key_audio, dtype=mx.float32)
        audio_rows = audio_scheduler.scale_noise(
            init_audio_rows.astype(mx.float32), float(1.0 - audio_axis[0]), noise
        )
    else:
        if tuple(init_audio_rows.shape) != audio_shape:
            raise ValueError(f"初始音频行形状应为 {audio_shape}，收到 {tuple(init_audio_rows.shape)}")
        audio_rows = init_audio_rows.astype(mx.float32)

    condition_video_rows = int(layout.num_condition_video_rows)
    condition_audio_rows = int(layout.num_condition_audio_rows)
    total = len(video_scheduler.timesteps)
    want_x0 = output == "denoised"
    x0_video = x0_audio = None
    for step in range(total):
        video_t = float(video_scheduler.timesteps[step])
        audio_t = float(audio_scheduler.timesteps[step])
        unique_timesteps, timestep_indices = build_row_timesteps(
            layout,
            video_t,
            audio_t,
            max(video_t, KEYFRAME_NOISE_AUG),
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
        # 关键帧条件行保持在 t = 0.999（切片之外的行不参与更新）
        if want_x0 and step == total - 1:
            # 最后一步的 x0 估计（= ComfyUI 的 denoised_output）：x0 = x_t + σ·v
            # 复用本步已经算好的 v，不额外前向；视频 / 音频各用自己的 σ（shift 12 / 3）
            x0_video = video_rows[condition_video_rows:] + mx.array(
                float(video_scheduler.sigmas[step]), dtype=mx.float32
            ) * video_pred[0, condition_video_rows:].astype(mx.float32)
            x0_audio = audio_rows[condition_audio_rows:] + mx.array(
                float(audio_scheduler.sigmas[step]), dtype=mx.float32
            ) * audio_pred[0, condition_audio_rows:].astype(mx.float32)
            mx.eval(x0_video, x0_audio)
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
            log(f"[{label}] step {step + 1}/{total} t={video_t:.3f}（audio t={audio_t:.3f}）")
    if want_x0:
        return x0_video, x0_audio, layout
    return video_rows[condition_video_rows:], audio_rows[condition_audio_rows:], layout


# --- 二阶段元数据（独立桶，不动 types.MlxLatentHandle）-----------------------------
def put_meta(cache: Any, key: str, meta: dict[str, Any]) -> None:
    """把二阶段元数据登记到 `h3_two_stage_meta` 桶（键与 latent 键相同）。"""
    cache.get_or_create(META_BUCKET, key, lambda: dict(meta))


def get_meta(cache: Any, key: str) -> dict[str, Any]:
    """取二阶段元数据；没有就返回空 dict（调用方走保守默认）。"""
    value, hit = cache.get(META_BUCKET, key)
    return dict(value) if hit and isinstance(value, dict) else {}


# --- 新画布下的视觉条件（只读既有桶 + 自建新键，不改条件节点）----------------------
def derive_visual_condition(visual: Any, width: int, height: int, cache: Any) -> Any:
    """把视觉条件的锚点图按**新画布**重新登记一份，返回新的 MlxH3VisualCondition。

    条件节点存的是「已按低分画布 LANCZOS 拉伸过」的图，因此这里是从低分图再拉大
    （略有软化，属已知取舍）；非锚点参考图保持各自尺寸不变，`picture_count` 不变。
    既有条件节点与既有缓存条目都不受影响。
    """
    if visual is None:
        return None
    if int(visual.width) == int(width) and int(visual.height) == int(height):
        return visual

    from PIL import Image

    from comfyui_mlx_gen import pipeline, runtime  # 局部导入：避免与编排层互相牵扯

    images, hit = cache.get(pipeline.H3_VISUAL_BUCKET, visual.images_key)
    if not hit or images is None:
        raise RuntimeError("H3 的参考图缓存已失效，请重新运行「MLX H3 关键帧 / 视觉条件」节点")
    refitted = list(images)
    for index in visual.anchor_images:
        refitted[int(index)] = images[int(index)].resize((int(width), int(height)), Image.LANCZOS)
    digest = runtime.cache_key(
        {
            "kind": "h3_visual_condition",
            "origin": visual.digest,
            "resize_to": [int(width), int(height)],
        }
    )
    images_key = runtime.cache_key({"kind": "h3_keyframe_source", "digest": digest})
    cache.get_or_create(pipeline.H3_VISUAL_BUCKET, images_key, lambda: tuple(refitted))
    print(
        f"[H3 二阶段] 锚点图已按新画布 {int(width)}×{int(height)} 重新登记"
        f"（源是 {visual.width}×{visual.height} 的低分图，略有软化）"
    )
    return type(visual)(
        images_key=images_key,
        picture_count=int(visual.picture_count),
        anchors=tuple(visual.anchors),
        anchor_images=tuple(int(i) for i in visual.anchor_images),
        width=int(width),
        height=int(height),
        digest=digest,
        source=visual.source,
        source_label=visual.source_label,
        vae=visual.vae,
        motion_key=visual.motion_key,
        motion_frame_count=int(visual.motion_frame_count),
        motion_fps=float(visual.motion_fps),
    )




