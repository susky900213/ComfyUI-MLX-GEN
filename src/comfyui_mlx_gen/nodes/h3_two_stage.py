"""MiniMax-H3 二阶段工作流的三个**新**节点（既有节点零改动）。

    MlxH3FirstPassSampler  —— 低分辨率前半段（可停在母网格第 k 步）
    MlxH3LatentUpscaler    —— latent 空间 3D 放大（只给倍率，目标画布由上一段推导）
    MlxH3SecondPassSampler —— 放大后的剩余 σ 精修（含音频三模式）

为什么另起一个文件而不是给 `MlxKSamplerMLX` 加参数：需求是**不修改既有节点**。
因此校验 / 组件准备 / 缓存编排都在这里自己完成，只调用既有模块的公开入口：

- `pipeline.make_plan / preflight / h3_latent_cache_key / h3_state / h3_prompt_encoding_key`
  `prepare_h3_sampler_components / release_h3_sampler_components / encode_h3_keyframes`；
- `h3.pipeline.check_visual_condition_checkpoint`（Base / REF 档位校验）；
- `h3.two_stage`（σ 子网格、采样循环、二阶段元数据、新画布下的锚点重登记）；
- `h3.model.h3_latent_upscaler`（3D 放大网络）。

三个节点的输出与既有采样器**同型**：`MlxLatentHandle(kind="h3_video")`，
`(video_rows, audio_rows, plan)` 写进同一个 `h3_latents` 桶，
因此既有 `MlxVAEDecoder` / `MlxPilToTorch` / `CreateVideo` / `SaveVideo` 直接能接，
老工作流与老 JSON 完全不受影响。
"""

from __future__ import annotations

import math
import time
from typing import Any

import mlx.core as mx

from .. import paths, pipeline, runtime
from ..cache import CACHE
from ..h3 import pipeline as h3_pipeline
from ..h3 import two_stage
from ..h3.latent_creator.h3_layout import (
    MAX_ASPECT_RATIO,
    MIN_ASPECT_RATIO,
    patchify_video_latents,
    unpatchify_video_rows,
)
from ..h3.model import h3_latent_upscaler
from ..progress import SamplingProgress
from ..types import MlxLatentHandle, condition, entry_for, h3_keyframes, model
from .sampler import FRAME_OPTIONS, HEIGHT_OPTIONS, WIDTH_OPTIONS

ALIGN = 32
VIDEO_PATCH = (1, 2, 2)
UPSCALER_DIR = "upscaler"
NO_UPSCALER = "<把放大模型放进 models/mlx/upscaler/>"
# 内置插值只保留为手动诊断选项；不能在缺少/不兼容神经权重时静默选择它。
# 示例工作流和生成器必须显式持久化一个 3D 神经 checkpoint。
BUILTIN_UPSCALE = "<内置：latent 三线性插值（不用模型）>"
# 一阶段交给下游的是哪一路 latent（ComfyUI 的 SamplerCustomAdvanced 也有这两路）：
# - denoised (x0)：最后一个工作 σ 处的一步 x0 估计 —— 喂给 latent 放大网络必须用它
#   （放大网络在干净 latent 对上训练；给 92% 是噪声的 x_t 会把噪声放大 → 画面混沌）；
# - latent (x_t)：最后一步走完的原始 latent —— 跑满（stop_at_step=0）直出时用它，
#   此时与既有 MlxKSamplerMLX 的输出逐位一致。
STAGE1_OUTPUT_X0 = "denoised (x0)"
STAGE1_OUTPUT_XT = "latent (x_t)"
STAGE1_OUTPUTS = (STAGE1_OUTPUT_X0, STAGE1_OUTPUT_XT)


def snap32(value: float) -> int:
    """最近的 32 倍数；平局取上（保证只放大不缩小）。H3 画布必须是 32 的倍数。"""
    quotient = float(value) / ALIGN
    floor = math.floor(quotient)
    nearest = math.ceil(quotient) if quotient - floor >= 0.5 else round(quotient)
    return max(ALIGN, int(nearest) * ALIGN)


def upscaler_items() -> list[str]:
    """神经 3D 模型优先；内置插值只能手动选择用于诊断。"""
    models = paths.list_component_items(UPSCALER_DIR)
    if models:
        return [*models, BUILTIN_UPSCALE]
    return [NO_UPSCALER, BUILTIN_UPSCALE]


def validate_neural_output(source: mx.array, output: mx.array, model_name: str) -> tuple[float, float, float]:
    """拒绝会把雪花送进 VAE 的失控输出。

    H3 二阶段的输入/输出都在 normalized latent 空间。当前已知异常 checkpoint 会把
    `std` 放大约 6 倍；这不是合法的 detail 增益，继续 decode 会得到雪花。这里不做
    任意比例压缩，而是 fail fast，提示换兼容 checkpoint 或手动使用插值诊断。
    """
    if tuple(source.shape[:3]) != tuple(output.shape[:3]):
        raise RuntimeError(
            f"H3 3D 放大器 {model_name!r} 改变了 batch/channel/time 布局："
            f"输入 {tuple(source.shape)}，输出 {tuple(output.shape)}"
        )
    if not bool(mx.all(mx.isfinite(output)).item()):
        raise RuntimeError(
            f"H3 3D 放大器 {model_name!r} 输出了 NaN/Inf；请检查 checkpoint、精度和 MLX 前向"
        )
    input_std = float(mx.std(source))
    output_std = float(mx.std(output))
    ratio = output_std / max(input_std, 1e-6)
    if ratio > 4.0 or output_std > 8.0:
        raise RuntimeError(
            f"H3 3D 放大器 {model_name!r} 输出幅度失控：normalized latent "
            f"输入 std={input_std:.3f}，输出 std={output_std:.3f}，倍率={ratio:.2f}。"
            "继续 VAE 解码会产生雪花/噪点；不要再次添加 VAE mean/std 归一化，"
            "请换与 MiniMax-H3 normalized latent 兼容的 3D checkpoint，"
            f"或手动选择 {BUILTIN_UPSCALE!r} 做诊断。"
        )
    return input_std, output_std, ratio


def interpolate_latents(latents, size_hw: tuple[int, int]):
    """通道优先 `(1,C,T,H,W)` → 三线性插值到 `size_hw`（时间轴不动）。

    与放大网络内部那一步用的是同一个 `nn.Upsample(mode="linear", align_corners=False)`，
    因此只要整个流程用同一种插值，前后就自洽。
    """
    import mlx.nn as nn

    x = latents.astype(mx.float32).transpose(0, 2, 3, 4, 1)  # channels-last
    y = nn.Upsample(
        scale_factor=(1.0, size_hw[0] / x.shape[2], size_hw[1] / x.shape[3]),
        mode="linear",
        align_corners=False,
    )(x)
    mx.eval(y)
    return y.transpose(0, 4, 1, 2, 3)


def resolve_upscaler_path(name: str) -> str:
    """把下拉里的名字解析成具体的 .safetensors 路径（支持目录与 HF cache 外壳）。"""
    if not name or name == NO_UPSCALER:
        raise FileNotFoundError(
            f"{UPSCALER_DIR}/ 里没有放大模型：请把 minimax_h3_latent_upscaler_3d_*.safetensors "
            f"放到 {paths.component_dir(UPSCALER_DIR)}"
        )
    candidate = paths.normalize_hf_cache_path(paths.component_dir(UPSCALER_DIR) / name)
    if candidate.is_file():
        return str(candidate)
    if candidate.is_dir():
        files = sorted(candidate.rglob("*.safetensors"))
        if len(files) == 1:
            return str(files[0])
        if not files:
            raise FileNotFoundError(f"{candidate} 里没有 .safetensors")
        raise FileNotFoundError(
            f"{candidate} 里有多个 safetensors，请在 model_name 里直接选文件："
            f"{'、'.join(f.name for f in files)}"
        )
    raise FileNotFoundError(f"{candidate} 不存在（{UPSCALER_DIR}/ 下的文件或目录）")


def validate_handles(model_handle: Any, positive: Any, negative: Any) -> Any:
    """与既有采样器同口径的三点校验（不动既有代码，只做同样的检查）。"""
    if model_handle is None:
        raise ValueError("必须连接 MlxTransformerLoader 的输出")
    if positive is None or negative is None:
        raise ValueError(
            "必须分别连接 MlxTextEncoder 的输出作为正/负条件，采样器不直接接受文本输入"
        )
    if positive.clip != negative.clip:
        raise ValueError("正/负条件必须接在同一个 MlxClipLoader 上（组件配置不能混用）")
    if positive.clip.model_type != model_handle.model_type:
        raise ValueError(
            f"文本编码器选的是 {positive.clip.model_type}，与采样器的 {model_handle.model_type} 不匹配"
        )
    entry = entry_for(model_handle.model_type)
    if not entry.supported:
        raise NotImplementedError(f"{model_handle.model_type} 尚未实现：{entry.notes}")
    if entry.media != "video":
        raise ValueError(
            f"{entry.family} 不是 H3 视频家族：二阶段节点只服务 MiniMax-H3，请改用「MLX 采样器」"
        )
    return entry


def validate_keyframes(
    model_handle: Any, keyframes: Any, width: int, height: int, positive: Any, label: str
) -> None:
    """H3 视觉条件的档位 / VAE / 画布 / 条件编码一致性（一阶段用）。"""
    if keyframes is None:
        return
    note = h3_pipeline.check_visual_condition_checkpoint(model_handle, keyframes)
    if note:
        print(f"[{label}] {note}")
    if keyframes.vae is None:
        print(
            f"[{label}] {keyframes.source_label or '视觉条件'}：没有首 / 尾帧锚点，"
            "图片只当视觉提示（presentation）"
        )
    elif keyframes.vae.model_type != model_handle.model_type or keyframes.vae.role != "vae":
        raise ValueError(
            "H3 视觉条件必须由当前模型大类的视频 VAE（role=vae）编码："
            f"条件是 model_type={keyframes.vae.model_type!r}, role={keyframes.vae.role!r}，"
            f"采样器是 {model_handle.model_type!r}"
        )
    if keyframes.width != int(width) or keyframes.height != int(height):
        raise ValueError(
            "H3 视觉条件的目标画布必须与采样器一致："
            f"条件是 {keyframes.width}×{keyframes.height}，采样器是 {int(width)}×{int(height)}"
        )
    expected_key = pipeline.h3_prompt_encoding_key(positive.clip, positive.text, keyframes)
    if positive.encoding_key != expected_key:
        raise RuntimeError(
            "H3 关键帧与 Qwen3-VL presentation 编码不匹配；"
            "请重新运行当前连线下的「MLX 文本编码器」节点"
        )


def lora_step_hint(model_handle: Any) -> int | None:
    """从已挂 LoRA 的文件名里读标称步数（`..._4step_...` / `..._8step_...`）。

    LoRA 只改 Transformer，**不会自动改采样器步数**（USAGE_ZH §7）；标称 4 步的适配器
    配 5 步母网格、或标称 8 步却只走 5 步，都会掉出适配器的训练档位。
    """
    import re

    for item in getattr(model_handle, "loras", ()) or ():
        found = re.search(r"(\d+)\s*step", str(getattr(item, "path", "")), re.IGNORECASE)
        if found:
            return int(found.group(1))
    return None


def make_handle(
    model_handle: Any, plan: Any, video_rows: Any, audio_rows: Any, key: str, digest: str = ""
) -> MlxLatentHandle:
    """与 `pipeline.run_h3_sampler` 完全同型的句柄（解码节点据此还原 latent）。"""
    return MlxLatentHandle(
        kind="h3_video",
        shape=tuple(video_rows.shape),
        dtype=str(video_rows.dtype),
        cache_key=key,
        model=model_handle.model_type,
        source="local",
        path=model_handle.model_path,
        precision=model_handle.precision,
        quantize=model_handle.quantize,
        model_cache_key=runtime.cache_key(model_handle),
        height=plan.height,
        width=plan.width,
        num_frames=plan.num_frames,
        duration=plan.duration_seconds,
        video_shift=plan.video_shift,
        audio_shift=plan.audio_shift,
        num_latent_frames=plan.num_latent_frames,
        latent_height=plan.latent_height,
        latent_width=plan.latent_width,
        audio_num_rows=int(audio_rows.shape[0]),
        prompt_digest=digest,
    )


# --- 一阶段（低分辨率前半段）------------------------------------------------------
def run_first_pass(entry, model_handle, params, cache, keyframes, stop_at_step):
    """母网格前 k 步；返回 `(handle, plan, video_axis, audio_axis)`。"""
    output_mode = str(params.get("output_mode", STAGE1_OUTPUT_X0))
    if output_mode not in STAGE1_OUTPUTS:
        raise ValueError(f"output_mode 只能是 {' / '.join(STAGE1_OUTPUTS)}，收到 {output_mode!r}")
    output = "denoised" if output_mode == STAGE1_OUTPUT_X0 else "latent"
    plan = h3_pipeline.make_plan(
        int(params["width"]),
        int(params["height"]),
        int(params["num_frames"]),
        int(params["steps"]),
        float(params["video_shift"]),
        float(params["audio_shift"]),
        log=print,
    )
    video_axis = two_stage.first_pass_axis(plan.video_shift, plan.steps, int(stop_at_step))
    audio_axis = two_stage.first_pass_axis(plan.audio_shift, plan.steps, int(stop_at_step))
    performed = len(video_axis) - 1
    print(f"[H3 一阶段] {plan.summary()}")
    print(
        f"[H3 一阶段] 停在母网格第 {performed}/{plan.steps} 步 → "
        f"video σ {float(video_axis[-1]):.4f} / audio σ {float(audio_axis[-1]):.4f}"
        + ("（已跑满，等价于既有「MLX 采样器」）" if performed == plan.steps else "（留给二阶段接着走）")
    )
    print(
        f"[H3 一阶段] 输出 {output_mode}"
        + (
            "：这是 x0 估计，二阶段/放大网络要的就是它"
            if output == "denoised"
            else "：这是含噪的 x_t，**不要**直接喂给 latent 放大网络（会放大噪声）"
        )
    )
    hint = lora_step_hint(model_handle)
    if hint is None and int(params["steps"]) < 20:
        print(
            f"⚠️ [H3 一阶段] 没挂加速 LoRA 却只跑 {int(params['steps'])} 步：H3 基座模型的官方档是 "
            "40~50 步，步数太少时交出的 x0 本身就是粗糙估计，放大只会把它放大（成片像雪花/噪点）。"
            "要少步数请配 4 步或 8 步的 LightX2V Turbo 适配器，并把 steps 改成 4 / 8"
        )
    elif hint is not None and int(params["steps"]) != hint:
        print(
            f"⚠️ [H3 一阶段] 加速 LoRA 文件名标称 {hint} 步，而 steps={int(params['steps'])}："
            "LoRA 不会自动改采样器步数，请把 steps 设成适配器标称值（并把两段之和也对齐）"
        )
    progress = SamplingProgress(performed)
    key = pipeline.h3_latent_cache_key(params, model_handle)

    state, hit = cache.get(pipeline.H3_LATENT_BUCKET, key)
    if hit and state is not None:
        progress.complete()
        video_rows, audio_rows, stored_plan = state
        return (
            make_handle(
                model_handle, stored_plan, video_rows, audio_rows, key, params["prompt_digest"]
            ),
            stored_plan,
            video_axis,
            audio_axis,
        )

    def compute():
        embeds, tags = pipeline.cached_h3_prompt(cache, params["positive_encoding_key"])
        keyframe_latents = ()
        if keyframes is not None:
            keyframe_latents = pipeline.encode_h3_keyframes(keyframes, cache)[1]
        comps = pipeline.prepare_h3_sampler_components(entry, model_handle, cache)
        try:
            loaded = comps["transformer"]
            h3_pipeline.preflight(
                plan,
                int(loaded.parameter_bytes()),
                int(runtime.system_total_memory()),
                text_tokens=int(tags.shape[0]),
                cache_bytes=pipeline._mlx_memory_bytes(),
                keyframe_count=len(params.get("keyframe_anchors", ())),
                log=print,
            )
            with pipeline.h3_exact_fp32():
                video_rows, audio_rows, _layout = two_stage.denoise(
                    loaded.module,
                    embeds,
                    tags,
                    plan,
                    int(params["seed"]),
                    video_axis=video_axis,
                    audio_axis=audio_axis,
                    keyframe_latents=keyframe_latents,
                    keyframe_anchors=tuple(params.get("keyframe_anchors", ())),
                    output=output,
                    label="H3 一阶段",
                    on_progress=progress.update_absolute,
                )
        finally:
            pipeline.release_h3_sampler_components(entry, model_handle, cache)
        return video_rows, audio_rows, plan

    video_rows, audio_rows, stored_plan = cache.get_or_create(
        pipeline.H3_LATENT_BUCKET, key, compute
    )[0]
    two_stage.put_meta(
        cache,
        key,
        {
            "stage": "first_pass",
            "total_steps": int(plan.steps),
            "stop_step": performed,
            "output_mode": output_mode,
            "has_lora": hint is not None,
            "video_sigma": float(video_axis[-1]),
            "audio_sigma": float(audio_axis[-1]),
            "width": int(stored_plan.width),
            "height": int(stored_plan.height),
            "num_frames": int(stored_plan.num_frames),
        },
    )
    return (
        make_handle(model_handle, stored_plan, video_rows, audio_rows, key, params["prompt_digest"]),
        stored_plan,
        video_axis,
        audio_axis,
    )


class MlxH3FirstPassSampler:
    """MLX H3 一阶段采样：低分辨率联合采样，可停在母网格第 k 步。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (model, {}),
                "positive": (condition, {}),
                "negative": (condition, {}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**63 - 1, "control_after_generate": True},
                ),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "width": (WIDTH_OPTIONS, {"default": 640}),
                "height": (HEIGHT_OPTIONS, {"default": 352}),
                "num_frames": (FRAME_OPTIONS, {"default": 124, "advanced": True}),
                "video_shift": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.1, "max": 30.0, "step": 0.5, "advanced": True,
                     "tooltip": "H3 视频整流流的 shift，必须 > 0（默认 12.0）。0 不是「不位移」，"
                     "传 0 会被直接拒绝"},
                ),
                "audio_shift": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.1, "max": 30.0, "step": 0.5, "advanced": True,
                     "tooltip": "H3 音频整流流的 shift，必须 > 0（默认 3.0）"},
                ),
                "stop_at_step": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 100,
                        "tooltip": "0 = 跑满 steps（与既有「MLX 采样器」行为一致）；"
                        "k = 停在母网格第 k 步，把剩余 σ 留给「MLX H3 二阶段采样」",
                    },
                ),
                "output_mode": (
                    list(STAGE1_OUTPUTS),
                    {
                        "default": STAGE1_OUTPUT_X0,
                        "tooltip": "交给下游哪一路 latent：denoised (x0) = 最后一个工作 σ 处的一步"
                        " x0 估计（干净，放大网络与二阶段要的就是它，推荐）；"
                        "latent (x_t) = 含噪的原始 latent（只在「跑满直出」时需要，"
                        "此时与既有「MLX 采样器」逐位一致）",
                    },
                ),
            }
        }

    RETURN_TYPES = ("latents",)
    FUNCTION = "sample"
    CATEGORY = "MLX/Gen"

    def sample(
        self,
        model,
        positive,
        negative,
        seed,
        steps,
        width,
        height,
        num_frames,
        video_shift,
        audio_shift,
        stop_at_step,
        output_mode=STAGE1_OUTPUT_X0,
    ):
        # 先把「滑杆上合法但模型不接受」的值挡住：两个人最容易踩的是 shift=0
        for name, value in (("video_shift", video_shift), ("audio_shift", audio_shift)):
            if float(value) <= 0.0:
                raise ValueError(
                    f"{name} 必须大于 0，收到 {float(value):g} —— H3 是双整流流模型，"
                    "视频 shift 默认 12.0、音频 shift 默认 3.0（0 不是「不位移」，是非法值）"
                )
        entry = validate_handles(model, positive, negative)
        keyframes = positive.h3_keyframes
        validate_keyframes(model, keyframes, width, height, positive, "MlxH3FirstPassSampler")
        params = {
            "stage": "first_pass",
            "seed": int(seed),
            "steps": int(steps),
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "video_shift": float(video_shift),
            "audio_shift": float(audio_shift),
            "stop_at_step": int(stop_at_step),
            "output_mode": str(output_mode),
            "positive_encoding_key": positive.encoding_key,
            "prompt_digest": positive.text,
            "keyframe_digest": keyframes.digest if keyframes is not None else "",
            "keyframe_anchors": keyframes.anchors if keyframes is not None else (),
        }
        handle, _plan, _video_axis, _audio_axis = run_first_pass(
            entry, model, params, CACHE, keyframes, int(stop_at_step)
        )
        return (handle,)


# --- latent 放大（3D 网络，只给倍率）------------------------------------------------
def run_upscale(latents, model_name, scale, precision, cache):
    """画布 = 上一段画布 × scale（吸附到 32 倍数）；音频行原样透传。"""
    import dataclasses

    if not isinstance(latents, MlxLatentHandle):
        raise ValueError("latents 必须连接「MLX H3 一阶段采样」或本节点的输出")
    if latents.kind != "h3_video":
        raise ValueError(f"本节点只放大 H3 视频 latent（kind=h3_video），收到 {latents.kind!r}")
    scale = float(scale)
    if scale < 1.0:
        raise ValueError(f"放大网络只支持放大（scale >= 1.0），收到 {scale:g}")

    # 目标画布先用句柄里的画布算（不读缓存）：这样 scale=1.0 的「原样返回」永远可用，
    # 而且画布 / 比例 / latent 偶数这些便宜的校验都在加载权重之前做完。
    width_in, height_in = int(latents.width), int(latents.height)
    if width_in <= 0 or height_in <= 0:
        raise ValueError(
            "latents 句柄里没有画布信息（不是本插件 H3 采样器产出的？）："
            "请接「MLX H3 一阶段采样」或本节点的输出"
        )
    width_out, height_out = snap32(width_in * scale), snap32(height_in * scale)
    if (width_out, height_out) == (width_in, height_in):
        print(
            f"[MlxH3LatentUpscaler] scale={scale:g} 吸附到 32 倍数后与输入相同"
            f"（{width_in}×{height_in}），原样返回"
        )
        return latents
    ratio = width_out / height_out
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(
            f"放大后 {width_out}×{height_out} 的宽高比 {ratio:.3f} 超出 H3 的 1/4~4，请调小 scale"
        )
    latent_h_out, latent_w_out = height_out // 16, width_out // 16
    if latent_h_out % 2 or latent_w_out % 2:
        raise ValueError(
            f"放大后 latent {latent_h_out}×{latent_w_out} 必须是偶数（画布 {width_out}×{height_out} 不是 32 的倍数？）"
        )

    parent_meta = two_stage.get_meta(cache, latents.cache_key)
    video_rows, audio_rows, plan = pipeline.h3_state(cache, latents.cache_key)
    if (int(plan.width), int(plan.height)) != (width_in, height_in):
        raise ValueError(
            f"latents 句柄里的画布（{width_in}×{height_in}）与缓存里的 latent 状态"
            f"（{plan.width}×{plan.height}）不一致，请重新跑一次上游节点"
        )

    source = unpatchify_video_rows(
        video_rows,
        int(video_rows.shape[-1]) // 4,
        plan.num_latent_frames,
        plan.latent_height,
        plan.latent_width,
        VIDEO_PATCH,
    )
    mean, std = float(mx.mean(source)), float(mx.std(source))
    if abs(mean) > 2.0 or not 0.3 <= std <= 3.0:
        print(
            f"⚠️ [MlxH3LatentUpscaler] 输入 latent 的统计量不像 H3 归一化空间"
            f"（mean={mean:+.3f}, std={std:.3f}）：放大网络是在 (z-mean)/std 空间上训练的，"
            "结果可能不对（latents 是否来自本插件的 H3 采样器？）"
        )
    target = (plan.num_latent_frames, latent_h_out, latent_w_out)
    print(
        f"[MlxH3LatentUpscaler] {width_in}×{height_in} ×{scale:g} → {width_out}×{height_out}"
        f" | latent {plan.latent_height}×{plan.latent_width} → {latent_h_out}×{latent_w_out}"
        f" | 有效倍率 {(width_out / width_in + height_out / height_in) / 2:.4f}"
        f" | 视频行 {tuple(video_rows.shape)} → {target[0] * target[1] * target[2]} 行"
    )
    t0 = time.perf_counter()
    if str(model_name) == BUILTIN_UPSCALE:
        print("[MlxH3LatentUpscaler] 手动诊断模式：使用内置 latent 三线性插值（不加载神经模型）")
        upscaled = interpolate_latents(source, (latent_h_out, latent_w_out))
    else:
        module = h3_latent_upscaler.load(resolve_upscaler_path(str(model_name)), str(precision), cache)
        upscaled = h3_latent_upscaler.upscale_latents(
            module, source, (latent_h_out, latent_w_out), str(precision), enable_chunking=True
        )
        input_std, output_std, output_ratio = validate_neural_output(source, upscaled, str(model_name))
        print(
            f"[MlxH3LatentUpscaler] normalized latent 统计：输入 std={input_std:.3f} "
            f"→ 输出 std={output_std:.3f}（倍率 {output_ratio:.2f}）"
        )
    new_rows = patchify_video_latents(upscaled, VIDEO_PATCH)
    mx.eval(new_rows)
    elapsed = time.perf_counter() - t0

    new_plan = h3_pipeline.make_plan(
        width_out,
        height_out,
        plan.num_frames,
        plan.steps,
        plan.video_shift,
        plan.audio_shift,
        log=None,
    )
    if int(new_plan.num_audio_latents) != int(plan.num_audio_latents):
        raise RuntimeError(
            f"音频 latent 数在放大前后不一致：{plan.num_audio_latents} → {new_plan.num_audio_latents}"
        )
    print(
        f"[MlxH3LatentUpscaler] 完成：{elapsed:.1f}s | 音频行 {tuple(audio_rows.shape)} 原样透传"
    )
    key = runtime.cache_key(
        {
            "kind": "h3_latents",
            "stage": "upscale",
            "parent": latents.cache_key,
            "model_name": str(model_name),
            "precision": str(precision),
            "width": width_out,
            "height": height_out,
        }
    )
    cache.get_or_create(pipeline.H3_LATENT_BUCKET, key, lambda: (new_rows, audio_rows, new_plan))
    two_stage.put_meta(
        cache,
        key,
        {
            "stage": "upscale",
            "parent": latents.cache_key,
            "scale": scale,
            "effective_scale": (width_out / width_in + height_out / height_in) / 2.0,
            "width": width_out,
            "height": height_out,
            "num_frames": int(plan.num_frames),
            "model_name": str(model_name),
            # 继承「这条 latent 的音频 / 视频当前停在哪」：h3_latents 桶容量只有 1，
            # 上游 meta 会被本条挤掉，所以二阶段要的信息必须随链条带下去
            "video_sigma": parent_meta.get("video_sigma"),
            "audio_sigma": parent_meta.get("audio_sigma"),
            "stop_step": parent_meta.get("stop_step"),
        },
    )
    return dataclasses.replace(
        latents,
        shape=tuple(new_rows.shape),
        dtype=str(new_rows.dtype),
        cache_key=key,
        height=int(new_plan.height),
        width=int(new_plan.width),
        num_latent_frames=int(new_plan.num_latent_frames),
        latent_height=int(new_plan.latent_height),
        latent_width=int(new_plan.latent_width),
        audio_num_rows=int(audio_rows.shape[0]),
    )


class MlxH3LatentUpscaler:
    """MLX H3 Latent 放大（3D）：只给倍率，目标画布由上一段 latent 推导。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("latents", {}),
                "model_name": (upscaler_items(), {}),
                "scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "目标画布 = round32(上一段宽 × scale) × round32(上一段高 × scale)；"
                        "吸附到 32 的倍数（平局取上，只放大不缩小）",
                    },
                ),
            },
            "optional": {"precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"})},
        }

    RETURN_TYPES = ("latents",)
    FUNCTION = "upscale"
    CATEGORY = "MLX/Gen"

    def upscale(self, latents, model_name, scale, precision="bfloat16"):
        return (run_upscale(latents, model_name, scale, precision, CACHE),)


# --- 二阶段（放大后的剩余 σ 精修）--------------------------------------------------
def run_refine(entry, model_handle, positive, params, cache, keyframes, init_latents):
    """从已有 latent 接着去噪；画布 / 帧数 / 两条 shift 全部取自入参 latent。"""
    video_rows, audio_rows, prev_plan = pipeline.h3_state(cache, init_latents.cache_key)
    meta = two_stage.get_meta(cache, init_latents.cache_key)
    width, height = int(prev_plan.width), int(prev_plan.height)
    num_frames = int(prev_plan.num_frames)

    derived = (
        two_stage.derive_visual_condition(keyframes, width, height, cache)
        if keyframes is not None
        else None
    )
    if derived is not None:
        note = h3_pipeline.check_visual_condition_checkpoint(model_handle, derived)
        if note:
            print(f"[H3 二阶段] {note}")
        if derived.vae is not None and (
            derived.vae.model_type != model_handle.model_type or derived.vae.role != "vae"
        ):
            raise ValueError(
                "H3 视觉条件必须由当前模型大类的视频 VAE（role=vae）编码："
                f"{derived.vae.model_type!r} / role={derived.vae.role!r}"
            )
        if derived.anchors and derived.vae is None:
            raise ValueError(
                "H3 的 latent 锚点必须用视频 VAE（role=vae）编码："
                "请把「MLX VAE 加载」的 vae 输出接到条件节点"
            )

    axes = two_stage.refine_axes(
        prev_plan.video_shift,
        prev_plan.audio_shift,
        int(params["steps"]),
        float(params["start_at_sigma"]),
        str(params["audio_mode"]),
        stage1_audio_sigma=meta.get("audio_sigma"),
        log=print,
    )
    plan = h3_pipeline.make_plan(
        width,
        height,
        num_frames,
        int(params["steps"]),
        prev_plan.video_shift,
        prev_plan.audio_shift,
        log=None,
    )
    print(f"[H3 二阶段] {plan.summary()} | {axes.summary()}")
    order_note = two_stage.stage_order_note(float(params["start_at_sigma"]), meta.get("video_sigma"))
    if order_note:
        print(f"⚠️ [H3 二阶段] {order_note}")
    # 二阶段步数必须够走完母网格在起点 σ 之后的原生台阶（1 步从 0.92 到 0 = 雪花/噪点）
    tail_note = two_stage.tail_points_note(
        float(prev_plan.video_shift),
        int(meta.get("total_steps", prev_plan.steps)),
        float(params["start_at_sigma"]),
        int(params["steps"]),
        distilled=bool(meta.get("has_lora")),
    )
    if tail_note:
        print(f"⚠️ [H3 二阶段] {tail_note}")
    hint = lora_step_hint(model_handle)
    if hint is not None and int(meta.get("stop_step", 0)) + int(params["steps"]) != hint:
        print(
            f"⚠️ [H3 二阶段] 加速 LoRA 标称 {hint} 步，但一阶段停了 "
            f"{int(meta.get('stop_step', 0))} 步 + 二阶段 {int(params['steps'])} 步 = "
            f"{int(meta.get('stop_step', 0)) + int(params['steps'])} 步："
            f"请把两段之和对齐到 {hint}（例：{hint // 2} + {hint - hint // 2}）"
        )
    if meta:
        past_video = meta.get("video_sigma")
        past_audio = meta.get("audio_sigma")
        past = (
            f"video σ {past_video:.4f} / audio σ {past_audio:.4f}"
            if isinstance(past_video, (int, float)) and isinstance(past_audio, (int, float))
            else "σ 未知"
        )
        print(
            f"[H3 二阶段] 入参 latent：{meta.get('stage', '?')} 阶段"
            f"（停在母网格第 {meta.get('stop_step', '?')} 步，{past}）"
        )

    params.update(
        {
            "stage": "second_pass",
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "video_shift": float(prev_plan.video_shift),
            "audio_shift": float(prev_plan.audio_shift),
            "audio_mode": axes.audio_mode,
            "audio_start_sigma": float(axes.audio_start_sigma),
            "init_key": init_latents.cache_key,
            "positive_encoding_key": positive.encoding_key,
            "prompt_digest": positive.text,
            "keyframe_digest": derived.digest if derived is not None else "",
            "keyframe_anchors": derived.anchors if derived is not None else (),
            "keyframe_latents_source": derived.images_key if derived is not None else "",
        }
    )
    key = pipeline.h3_latent_cache_key(params, model_handle)
    progress = SamplingProgress(int(axes.video.shape[0]) - 1)

    state, hit = cache.get(pipeline.H3_LATENT_BUCKET, key)
    if hit and state is not None:
        progress.complete()
        new_video, new_audio, stored_plan = state
        return (
            make_handle(model_handle, stored_plan, new_video, new_audio, key, positive.text),
            stored_plan,
        )

    def compute():
        embeds, tags = pipeline.cached_h3_prompt(cache, params["positive_encoding_key"])
        keyframe_latents = ()
        if derived is not None:
            keyframe_latents = pipeline.encode_h3_keyframes(derived, cache)[1]
        comps = pipeline.prepare_h3_sampler_components(entry, model_handle, cache)
        try:
            loaded = comps["transformer"]
            h3_pipeline.preflight(
                plan,
                int(loaded.parameter_bytes()),
                int(runtime.system_total_memory()),
                text_tokens=int(tags.shape[0]),
                cache_bytes=pipeline._mlx_memory_bytes(),
                keyframe_count=len(params.get("keyframe_anchors", ())),
                log=print,
            )
            with pipeline.h3_exact_fp32():
                new_video, new_audio, _layout = two_stage.denoise(
                    loaded.module,
                    embeds,
                    tags,
                    plan,
                    int(params["seed"]),
                    video_axis=axes.video,
                    audio_axis=axes.audio,
                    init_video_rows=video_rows,
                    init_audio_rows=audio_rows,
                    audio_reenoise=axes.audio_reenoise,
                    keyframe_latents=keyframe_latents,
                    keyframe_anchors=tuple(params.get("keyframe_anchors", ())),
                    label="H3 二阶段",
                    on_progress=progress.update_absolute,
                )
        finally:
            pipeline.release_h3_sampler_components(entry, model_handle, cache)
        return new_video, new_audio, plan

    new_video, new_audio, stored_plan = cache.get_or_create(
        pipeline.H3_LATENT_BUCKET, key, compute
    )[0]
    two_stage.put_meta(
        cache,
        key,
        {
            "stage": "second_pass",
            "parent": init_latents.cache_key,
            "steps": int(params["steps"]),
            "start_at_sigma": float(params["start_at_sigma"]),
            "video_sigma": float(axes.video[0]),
            "audio_sigma": float(axes.audio[0]),
            "audio_mode": axes.audio_mode,
            "width": int(stored_plan.width),
            "height": int(stored_plan.height),
            "num_frames": int(stored_plan.num_frames),
        },
    )
    return (
        make_handle(model_handle, stored_plan, new_video, new_audio, key, positive.text),
        stored_plan,
    )


class MlxH3SecondPassSampler:
    """MLX H3 二阶段采样：在放大后的 latent 上跑剩余 σ，音频三模式可选。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (model, {}),
                "positive": (condition, {}),
                "negative": (condition, {}),
                "latents": ("latents", {}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**63 - 1, "control_after_generate": True},
                ),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "start_at_sigma": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "从哪个 σ 接着走：0.7 = 保留较多已放大结构（推荐）；"
                        "0.9231 = 与 ComfyUI 的 SplitSigmas(step=4) 等价（重绘更多）；"
                        "越小越接近「只锐化」",
                    },
                ),
                "audio_mode": (
                    list(two_stage.AUDIO_MODES),
                    {
                        "default": "follow_video",
                        "tooltip": "follow_video = 音频跟视频同一 p 刻度重新加噪再收尾（与 CUDA 参考一致）；"
                        "continue_stage1 = 音频不重加噪、接着一阶段状态走（最保音频）；"
                        "regenerate = 音频从纯噪声重画（音质风险大）",
                    },
                ),
            },
            "optional": {"keyframes": (h3_keyframes, {})},
        }

    RETURN_TYPES = ("latents",)
    FUNCTION = "sample"
    CATEGORY = "MLX/Gen"

    def sample(
        self,
        model,
        positive,
        negative,
        latents,
        seed,
        steps,
        start_at_sigma,
        audio_mode,
        keyframes=None,
    ):
        entry = validate_handles(model, positive, negative)
        if not isinstance(latents, MlxLatentHandle) or latents.kind != "h3_video":
            raise ValueError(
                "latents 必须连接「MLX H3 一阶段采样」或「MLX H3 Latent 放大」的输出"
            )
        if latents.model and latents.model != model.model_type:
            raise ValueError(
                f"latents 是 {latents.model} 产出的，与采样器的 {model.model_type} 不匹配"
            )
        if keyframes is None:
            keyframes = positive.h3_keyframes
        if keyframes is not None:
            expected_key = pipeline.h3_prompt_encoding_key(positive.clip, positive.text, keyframes)
            if positive.encoding_key != expected_key:
                raise RuntimeError(
                    "H3 关键帧与 Qwen3-VL presentation 编码不匹配；"
                    "请重新运行当前连线下的「MLX 文本编码器」节点"
                )
        params = {
            "seed": int(seed),
            "steps": int(steps),
            "start_at_sigma": float(start_at_sigma),
            "audio_mode": str(audio_mode),
        }
        handle, _plan = run_refine(entry, model, positive, params, CACHE, keyframes, latents)
        return (handle,)






