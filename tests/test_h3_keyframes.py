"""MiniMax-H3 首帧 / 尾帧条件的无权重合成回归测试。

运行：
    python tests/test_h3_keyframes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen import image, pipeline  # noqa: E402
from comfyui_mlx_gen.cache import Cache  # noqa: E402
from comfyui_mlx_gen.h3 import pipeline as h3_pipeline  # noqa: E402
from comfyui_mlx_gen.h3.latent_creator.h3_layout import (  # noqa: E402
    AUDIO_CHANNELS,
    KEYFRAME_NOISE_AUG,
    PIXEL_MEAN,
    PIXEL_STD,
    patchify_video_latents,
)
from comfyui_mlx_gen.h3.scheduler.minimax_h3_scheduler import MiniMaxH3Scheduler  # noqa: E402
from comfyui_mlx_gen.nodes import h3_keyframes as keyframe_node  # noqa: E402
from comfyui_mlx_gen.nodes import sampler as sampler_node  # noqa: E402
from comfyui_mlx_gen.nodes import text_encoder as text_encoder_node  # noqa: E402
from comfyui_mlx_gen.types import (  # noqa: E402
    MlxClipHandle,
    MlxConditioning,
    MlxH3Keyframes,
    MlxModelHandle,
    MlxVaeHandle,
    entry_for,
)

FAILED: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK ' if ok else 'FAIL'}] {label}" + (f" | {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def check_raises(label: str, exc_type, message_part: str, call) -> None:
    try:
        call()
    except exc_type as exc:
        check(label, message_part in str(exc), str(exc))
    except Exception as exc:  # noqa: BLE001
        check(label, False, f"异常类型不对：{type(exc).__name__}: {exc}")
    else:
        check(label, False, f"没有抛出 {exc_type.__name__}")


def tensor_image(height: int, width: int, offset: float = 0.0) -> torch.Tensor:
    values = torch.linspace(0.0, 1.0, height * width * 3, dtype=torch.float32)
    return ((values.reshape(1, height, width, 3) + offset) % 1.0).contiguous()


video_vae = MlxVaeHandle(
    model_type="minimax_h3",
    path="MiniMax-H3",
    precision="bfloat16",
    quantize=8,
    role="vae",
    cache_key="test-video-vae",
)


# --------------------------------------------------------- 1. 条件节点：拉伸、摘要、顺序、空输入
old_cache = keyframe_node.CACHE
node_cache = Cache()
keyframe_node.CACHE = node_cache
try:
    first_src = tensor_image(11, 17)
    last_src = tensor_image(9, 13, 0.25)
    node = keyframe_node.MlxH3KeyframeCondition()
    first, _ = node.pack(video_vae, 64, 32, first_frame=first_src)
    last, _ = node.pack(video_vae, 64, 32, last_frame=last_src)
    both, report = node.pack(video_vae, 64, 32, first_frame=first_src, last_frame=last_src)

    cached_both, hit = node_cache.get("h3_keyframe_source", both.cache_key)
    manual_first = image.to_pil_batch(first_src)[0].resize((64, 32), Image.Resampling.LANCZOS)
    manual_last = image.to_pil_batch(last_src)[0].resize((64, 32), Image.Resampling.LANCZOS)
    check("首帧模式锚点为 first", first.anchors == ("first",), str(first.anchors))
    check("尾帧模式锚点为 last", last.anchors == ("last",), str(last.anchors))
    check("首尾帧固定按 first、last 排列", both.anchors == ("first", "last"), str(both.anchors))
    check(
        "关键帧用目标画布 LANCZOS 拉伸后进入独立缓存",
        hit
        and len(cached_both) == 2
        and cached_both[0].size == (64, 32)
        and cached_both[0].tobytes() == manual_first.tobytes()
        and cached_both[1].tobytes() == manual_last.tobytes(),
    )
    check("三种模式产生互异稳定摘要", len({first.digest, last.digest, both.digest}) == 3)
    check("节点报告标明首尾帧模式", "首尾帧" in report and "first,last" in report, report)
    check_raises(
        "首尾帧都未连接时明确拒绝",
        ValueError,
        "至少连接",
        lambda: node.pack(video_vae, 64, 32),
    )
    check_raises(
        "关键帧 IMAGE 不接受多图 batch",
        ValueError,
        "恰好包含一张",
        lambda: node.pack(video_vae, 64, 32, first_frame=torch.cat([first_src, first_src], dim=0)),
    )
    audio_vae = MlxVaeHandle(
        model_type="minimax_h3",
        path="MiniMax-H3",
        precision="bfloat16",
        quantize=8,
        role="audio_vae",
    )
    check_raises(
        "条件节点拒绝 H3 Audio VAE",
        ValueError,
        "role=vae",
        lambda: node.pack(audio_vae, 64, 32, first_frame=first_src),
    )
    wrong_model_vae = MlxVaeHandle(
        model_type="flux2",
        path="flux.2-klein-9b-8bit",
        precision="bfloat16",
        quantize=8,
        role="vae",
    )
    check_raises(
        "条件节点拒绝非 H3 模型的 VAE",
        ValueError,
        "minimax_h3",
        lambda: node.pack(wrong_model_vae, 64, 32, first_frame=first_src),
    )
    check_raises(
        "条件节点拒绝非 32 倍数画布",
        ValueError,
        "32",
        lambda: node.pack(video_vae, 63, 32, first_frame=first_src),
    )
finally:
    keyframe_node.CACHE = old_cache


# --------------------------------------------------------- 2. presentation 键与源图缓存一致性
clip = MlxClipHandle(
    model_type="minimax_h3",
    component="text_encoder",
    source="local",
    path="MiniMax-H3",
    precision="bfloat16",
    quantize=8,
)
plain_key = pipeline.h3_prompt_encoding_key(clip, "prompt")
first_key = pipeline.h3_prompt_encoding_key(clip, "prompt", first)
last_key = pipeline.h3_prompt_encoding_key(clip, "prompt", last)
both_key = pipeline.h3_prompt_encoding_key(clip, "prompt", both)
check("纯文本、首帧、尾帧、首尾帧 presentation 缓存键互相隔离", len({plain_key, first_key, last_key, both_key}) == 4)

prompt_cache = Cache()
prompt_cache.get_or_create("h3_keyframe_source", both.cache_key, lambda: cached_both)
captured: dict[str, object] = {}
old_import = pipeline.runtime.import_object
pipeline.runtime.import_object = lambda _spec: (
    lambda _encoder, _tokenizer, prompt, frames: captured.update(prompt=prompt, frames=frames)
    or (mx.zeros((1, 3, 4)), np.ones(3, dtype=np.int32))
)
try:
    encoded = pipeline.encode_h3_prompt(
        SimpleNamespace(prompt_encoder="fake:encode"),
        {"text_encoder": SimpleNamespace(module=object()), "tokenizer": object()},
        "prompt",
        prompt_cache,
        both_key,
        both,
    )
finally:
    pipeline.runtime.import_object = old_import
check(
    "Qwen3-VL presentation 使用与 latent 锚点相同的有序图像",
    len(captured.get("frames", ())) == 2
    and captured["frames"][0].tobytes() == cached_both[0].tobytes()
    and captured["frames"][1].tobytes() == cached_both[1].tobytes()
    and tuple(encoded[0].shape) == (1, 3, 4),
)

missing = MlxH3Keyframes(
    anchors=("first",),
    width=64,
    height=32,
    digest="missing",
    cache_key="missing",
    vae=video_vae,
)
check_raises(
    "presentation 源图缓存失效时不允许静默错配",
    RuntimeError,
    "缓存已失效",
    lambda: pipeline.encode_h3_prompt(
        SimpleNamespace(prompt_encoder="unused:encode"),
        {"text_encoder": SimpleNamespace(module=object()), "tokenizer": object()},
        "prompt",
        Cache(),
        "missing-prompt",
        missing,
    ),
)


# --------------------------------------------------------- 3. Video VAE posterior 固定 seed 与归一化
class FakeVAE:
    latent_channels = 24

    def __init__(self):
        self.latents_mean = mx.arange(24, dtype=mx.float32) / 100.0
        self.latents_std = mx.ones((24,), dtype=mx.float32) * 2.0
        self.inputs: list[mx.array] = []
        self._parameter = mx.zeros((1,), dtype=mx.float32)

    def parameters(self):
        return {"weight": self._parameter}

    def encode(self, pixels):
        self.inputs.append(pixels)
        shape = (pixels.shape[0], 24, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)
        return mx.zeros(shape, dtype=mx.float32), mx.zeros(shape, dtype=mx.float32)


fake_vae = FakeVAE()
posterior = h3_pipeline.encode_keyframe_latents(tuple(cached_both), fake_vae)
expected_pixels = np.asarray(cached_both[0], dtype=np.float32) / 255.0
expected_pixels = (expected_pixels - np.array(PIXEL_MEAN, dtype=np.float32)) / np.array(
    PIXEL_STD, dtype=np.float32
)
check(
    "Video VAE 输入是 ImageNet 归一化的 BCHFW",
    tuple(fake_vae.inputs[0].shape) == (1, 3, 1, 32, 64)
    and np.allclose(np.asarray(fake_vae.inputs[0][0, :, 0]), expected_pixels.transpose(2, 0, 1), atol=1e-5),
)
check(
    "每个锚点 posterior 都使用固定 seed 42 并经 float16 舍入",
    len(posterior) == 2 and np.array_equal(np.asarray(posterior[0]), np.asarray(posterior[1])),
)


# --------------------------------------------------------- 4. layout + 条件噪声 → 视频噪声 → 音频噪声
class FakeTransformer:
    in_channels = 24
    audio_in_channels = 4
    patch_size = (1, 2, 2)

    def __init__(self):
        self.video_inputs: list[mx.array] = []

    def __call__(self, hidden_states, audio_hidden_states, **_kwargs):
        self.video_inputs.append(hidden_states)
        return mx.zeros_like(hidden_states), mx.zeros_like(audio_hidden_states)


plan = h3_pipeline.make_plan(64, 32, 5, 1, log=None)
transformer = FakeTransformer()
condition = mx.zeros((1, 24, 1, plan.latent_height, plan.latent_width), dtype=mx.float32)
video_rows, audio_rows, layout = h3_pipeline.sample(
    transformer,
    mx.zeros((1, 2, 4), dtype=mx.float32),
    np.ones(2, dtype=np.int32),
    plan,
    123,
    keyframe_latents=(condition, condition),
    keyframe_anchors=("first", "last"),
    log=None,
)
rows_per_anchor = (plan.latent_height // 2) * (plan.latent_width // 2)
keys = mx.random.split(mx.random.key(123), 4)
scheduler = MiniMaxH3Scheduler(plan.video_shift)
expected_conditions = []
for key in keys[:2]:
    noise = mx.random.normal(condition.shape, key=key, dtype=mx.float32)
    expected_conditions.append(
        patchify_video_latents(scheduler.scale_noise(condition, KEYFRAME_NOISE_AUG, noise), (1, 2, 2))
    )
expected_video = patchify_video_latents(
    mx.random.normal(
        (1, 24, plan.num_latent_frames, plan.latent_height, plan.latent_width),
        key=keys[-2],
        dtype=mx.float32,
    ),
    (1, 2, 2),
)
expected_audio = mx.random.normal(
    (plan.num_audio_latents * AUDIO_CHANNELS, 4), key=keys[-1], dtype=mx.float32
)
check(
    "双锚点 layout 产生两帧条件行并按 first、last 赋时间坐标",
    layout.num_condition_video_rows == 2 * rows_per_anchor
    and float(layout.position_ids[2, 0]) == 2.0
    and float(layout.position_ids[2 + rows_per_anchor, 0]) > 2.0,
)
check(
    "随机数顺序是整组条件噪声、视频噪声、音频噪声",
    np.allclose(
        np.asarray(transformer.video_inputs[0][0, : 2 * rows_per_anchor]),
        np.asarray(mx.concatenate(expected_conditions, axis=0)),
    )
    and np.allclose(
        np.asarray(transformer.video_inputs[0][0, 2 * rows_per_anchor :]),
        np.asarray(expected_video),
    )
    and np.allclose(np.asarray(audio_rows), np.asarray(expected_audio)),
)
check(
    "去噪步骤不更新固定关键帧条件行",
    np.array_equal(
        np.asarray(mx.concatenate(expected_conditions, axis=0)),
        np.asarray(transformer.video_inputs[0][0, : 2 * rows_per_anchor]),
    ),
)
check(
    "最终 latent 剥离条件行并只保留可按计划解码的目标视频行",
    tuple(video_rows.shape) == tuple(expected_video.shape)
    and np.allclose(np.asarray(video_rows), np.asarray(expected_video)),
)
plain_transformer = FakeTransformer()
plain_video_rows, plain_audio_rows, plain_layout = h3_pipeline.sample(
    plain_transformer,
    mx.zeros((1, 2, 4), dtype=mx.float32),
    np.ones(2, dtype=np.int32),
    plan,
    321,
    log=None,
)
plain_keys = mx.random.split(mx.random.key(321), 2)
plain_expected_video = patchify_video_latents(
    mx.random.normal(
        (1, 24, plan.num_latent_frames, plan.latent_height, plan.latent_width),
        key=plain_keys[0],
        dtype=mx.float32,
    ),
    (1, 2, 2),
)
plain_expected_audio = mx.random.normal(
    (plan.num_audio_latents * AUDIO_CHANNELS, 4), key=plain_keys[1], dtype=mx.float32
)
check(
    "纯 T2V 保持两路 PRNG、零条件行与可解码输出形状",
    plain_layout.num_condition_video_rows == 0
    and tuple(plain_video_rows.shape) == tuple(plain_expected_video.shape)
    and np.allclose(np.asarray(plain_video_rows), np.asarray(plain_expected_video))
    and np.allclose(np.asarray(plain_audio_rows), np.asarray(plain_expected_audio)),
)
check_raises(
    "关键帧 latent 与锚点数量不一致时拒绝采样",
    ValueError,
    "数量不一致",
    lambda: h3_pipeline.sample(
        FakeTransformer(),
        mx.zeros((1, 2, 4)),
        np.ones(2, dtype=np.int32),
        plan,
        123,
        keyframe_latents=(condition,),
        keyframe_anchors=("first", "last"),
        log=None,
    ),
)

base_rows = h3_pipeline.packed_rows(plan, 2)
check(
    "峰值估算计入每个关键帧的 packed rows",
    h3_pipeline.packed_rows(plan, 2, 2) == base_rows + 2 * rows_per_anchor,
)


# --------------------------------------------------------- 5. sampler 契约、纯 T2V 与 latent 缓存短路
model = MlxModelHandle(
    model_type="minimax_h3",
    model_path="MiniMax-H3",
    quantize=8,
    precision="bfloat16",
    compile=False,
    compile_cache_limit=2,
)


def sampler_call(positive, negative=None, width=64, height=32):
    return sampler_node.MlxKSamplerMLX().sample(
        model,
        positive,
        negative or positive,
        seed=123,
        steps=1,
        width=width,
        height=height,
        batch_size=1,
        guidance=1.0,
        scheduler="minimax_h3",
        num_frames=124,
        video_shift=12.0,
        audio_shift=3.0,
    )


forged_audio_keyframes = MlxH3Keyframes(
    anchors=("first",),
    width=64,
    height=32,
    digest="forged-audio",
    cache_key="forged-audio",
    vae=MlxVaeHandle(
        model_type="minimax_h3",
        path="MiniMax-H3",
        precision="bfloat16",
        quantize=8,
        role="audio_vae",
    ),
)
forged_audio_condition = MlxConditioning(
    clip=clip,
    text="prompt",
    encoding_key=pipeline.h3_prompt_encoding_key(clip, "prompt", forged_audio_keyframes),
    h3_keyframes=forged_audio_keyframes,
)
check_raises(
    "sampler 再次拒绝伪造的 Audio VAE 关键帧 handle",
    ValueError,
    "role=vae",
    lambda: sampler_call(forged_audio_condition),
)
check_raises(
    "文本编码器在加载组件前拒绝伪造的 Audio VAE 关键帧 handle",
    ValueError,
    "role=vae",
    lambda: text_encoder_node.MlxTextEncoder().encode("prompt", clip, forged_audio_keyframes),
)

first_condition = MlxConditioning(
    clip=clip,
    text="prompt",
    encoding_key=pipeline.h3_prompt_encoding_key(clip, "prompt", first),
    h3_keyframes=first,
)
check_raises(
    "sampler 拒绝关键帧画布与采样画布错配",
    ValueError,
    "目标画布必须与采样器一致",
    lambda: sampler_call(first_condition, width=96),
)
stale_condition = MlxConditioning(
    clip=clip,
    text="prompt",
    encoding_key=pipeline.h3_prompt_encoding_key(clip, "prompt"),
    h3_keyframes=first,
)
check_raises(
    "sampler 拒绝 presentation key 与关键帧 handle 错配",
    RuntimeError,
    "presentation 编码不匹配",
    lambda: sampler_call(stale_condition),
)
wrong_sampler_model = MlxModelHandle(
    model_type="flux2",
    model_path="flux.2-klein-9b-8bit",
    quantize=8,
    precision="bfloat16",
    compile=False,
    compile_cache_limit=2,
)
check_raises(
    "sampler 拒绝条件与采样模型大类错配",
    ValueError,
    "不匹配",
    lambda: sampler_node.MlxKSamplerMLX().sample(
        wrong_sampler_model,
        first_condition,
        first_condition,
        seed=123,
        steps=1,
        width=64,
        height=32,
        batch_size=1,
        guidance=1.0,
        scheduler="flow_match_euler_discrete",
    ),
)
other_clip = MlxClipHandle(
    model_type="minimax_h3",
    component="text_encoder",
    source="local",
    path="other-h3-encoder",
    precision="bfloat16",
    quantize=8,
)
other_condition = MlxConditioning(
    clip=other_clip,
    text="prompt",
    encoding_key=pipeline.h3_prompt_encoding_key(other_clip, "prompt"),
)
check_raises(
    "sampler 拒绝正负条件 handle 错配",
    ValueError,
    "同一个 MlxClipLoader",
    lambda: sampler_call(first_condition, negative=other_condition),
)

plain_prompt = "integrated_multimodal_description: synthetic test"
plain_condition = MlxConditioning(
    clip=clip,
    text=plain_prompt,
    encoding_key=pipeline.h3_prompt_encoding_key(clip, plain_prompt),
)
old_sampler_cache = sampler_node.CACHE
old_has_latents = sampler_node.pipeline.has_h3_latents
old_run_h3 = sampler_node.pipeline.run_h3_sampler
old_encode_keyframes = sampler_node.pipeline.encode_h3_keyframes
old_prepare_sampler = sampler_node.pipeline.prepare_h3_sampler_components
short_circuit_seen: list[object] = []
sentinel_handle = object()
sampler_node.CACHE = Cache()
sampler_node.pipeline.has_h3_latents = lambda *_args: True
sampler_node.pipeline.run_h3_sampler = (
    lambda _entry, _model, comps, _params, _cache, **_kwargs: short_circuit_seen.append(comps)
    or sentinel_handle
)
sampler_node.pipeline.encode_h3_keyframes = lambda *_args: (_ for _ in ()).throw(
    AssertionError("纯 T2V 缓存命中不应编码关键帧")
)
sampler_node.pipeline.prepare_h3_sampler_components = lambda *_args: (_ for _ in ()).throw(
    AssertionError("latent 缓存命中不应加载 transformer")
)
try:
    cached_result = sampler_call(plain_condition)
finally:
    sampler_node.CACHE = old_sampler_cache
    sampler_node.pipeline.has_h3_latents = old_has_latents
    sampler_node.pipeline.run_h3_sampler = old_run_h3
    sampler_node.pipeline.encode_h3_keyframes = old_encode_keyframes
    sampler_node.pipeline.prepare_h3_sampler_components = old_prepare_sampler
check(
    "纯 T2V latent 缓存命中在加载 VAE/transformer 前短路",
    cached_result == (sentinel_handle,) and short_circuit_seen == [None],
)

direct_cache = Cache()
direct_params = {
    "seed": 123,
    "steps": 1,
    "height": 32,
    "width": 64,
    "num_frames": 5,
    "video_shift": 12.0,
    "audio_shift": 3.0,
    "positive_encoding_key": "unused-on-hit",
    "prompt_digest": plain_prompt,
    "keyframe_digest": "",
    "keyframe_anchors": (),
}
direct_plan = h3_pipeline.make_plan(64, 32, 5, 1, log=None)
direct_key = pipeline.h3_latent_cache_key(direct_params, model)
direct_cache.get_or_create(
    "h3_latents",
    direct_key,
    lambda: (
        mx.zeros((3, 4), dtype=mx.float32),
        mx.zeros((2, 4), dtype=mx.float32),
        direct_plan,
    ),
)
direct_handle = pipeline.run_h3_sampler(
    entry_for("minimax_h3"), model, None, direct_params, direct_cache
)
check(
    "run_h3_sampler 命中缓存时 comps=None 也不会访问组件",
    direct_handle.cache_key == direct_key and direct_handle.shape == (3, 4),
)


# --------------------------------------------------------- 6. 编码器、VAE、transformer 成功及异常释放
old_prepare_encoder = text_encoder_node.pipeline.prepare_h3_encoder
old_encode_prompt = text_encoder_node.pipeline.encode_h3_prompt
old_release_encoder = text_encoder_node.pipeline.release_h3_encoder
encoder_releases: list[str] = []
successful_encoder_prepare = lambda *_args: {"fake": object()}
text_encoder_node.pipeline.prepare_h3_encoder = successful_encoder_prepare
text_encoder_node.pipeline.release_h3_encoder = (
    lambda _entry, _clip, _cache: encoder_releases.append("released")
)
try:
    text_encoder_node.pipeline.encode_h3_prompt = (
        lambda *_args: (mx.zeros((1, 2, 4)), np.ones(2, dtype=np.int32))
    )
    text_encoder_node.MlxTextEncoder().encode("success", clip)
    text_encoder_node.pipeline.encode_h3_prompt = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("synthetic encoder failure")
    )
    try:
        text_encoder_node.MlxTextEncoder().encode("failure", clip)
    except RuntimeError as exc:
        encoder_failed = "synthetic encoder failure" in str(exc)
    else:
        encoder_failed = False
    text_encoder_node.pipeline.prepare_h3_encoder = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("synthetic encoder prepare failure")
    )
    try:
        text_encoder_node.MlxTextEncoder().encode("prepare failure", clip)
    except RuntimeError as exc:
        encoder_prepare_failed = "synthetic encoder prepare failure" in str(exc)
    else:
        encoder_prepare_failed = False
finally:
    text_encoder_node.pipeline.prepare_h3_encoder = old_prepare_encoder
    text_encoder_node.pipeline.encode_h3_prompt = old_encode_prompt
    text_encoder_node.pipeline.release_h3_encoder = old_release_encoder
check(
    "H3 条件编码器在成功、编码异常和准备异常路径都释放",
    encoder_failed
    and encoder_prepare_failed
    and encoder_releases == ["released", "released", "released"],
)

lifecycle_cache = Cache()
lifecycle_cache.get_or_create("h3_keyframe_source", first.cache_key, lambda: (cached_both[0],))
old_prepare_vae = pipeline.prepare_h3_vae
old_encode_latents = pipeline.h3_pipeline.encode_keyframe_latents
old_release_vae = pipeline.release_h3_vae
vae_releases: list[str] = []
successful_vae_prepare = lambda *_args: object()
pipeline.prepare_h3_vae = successful_vae_prepare
pipeline.release_h3_vae = lambda *_args: vae_releases.append("released")
try:
    pipeline.h3_pipeline.encode_keyframe_latents = lambda *_args: ("latent",)
    vae_success = pipeline.encode_h3_keyframes(first, lifecycle_cache)
    pipeline.h3_pipeline.encode_keyframe_latents = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("synthetic VAE failure")
    )
    try:
        pipeline.encode_h3_keyframes(first, lifecycle_cache)
    except RuntimeError as exc:
        vae_failed = "synthetic VAE failure" in str(exc)
    else:
        vae_failed = False
    pipeline.prepare_h3_vae = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("synthetic VAE prepare failure")
    )
    try:
        pipeline.encode_h3_keyframes(first, lifecycle_cache)
    except RuntimeError as exc:
        vae_prepare_failed = "synthetic VAE prepare failure" in str(exc)
    else:
        vae_prepare_failed = False
finally:
    pipeline.prepare_h3_vae = old_prepare_vae
    pipeline.h3_pipeline.encode_keyframe_latents = old_encode_latents
    pipeline.release_h3_vae = old_release_vae
check(
    "Video VAE 在成功、编码异常和准备异常路径都释放",
    vae_success == ("latent",)
    and vae_failed
    and vae_prepare_failed
    and vae_releases == ["released", "released", "released"],
)

old_sampler_cache = sampler_node.CACHE
old_has_latents = sampler_node.pipeline.has_h3_latents
old_prepare_sampler = sampler_node.pipeline.prepare_h3_sampler_components
old_run_h3 = sampler_node.pipeline.run_h3_sampler
old_release_sampler = sampler_node.pipeline.release_h3_sampler_components
transformer_releases: list[str] = []
sampler_node.CACHE = Cache()
sampler_node.pipeline.has_h3_latents = lambda *_args: False
sampler_node.pipeline.prepare_h3_sampler_components = lambda *_args: {"transformer": object()}
sampler_node.pipeline.release_h3_sampler_components = (
    lambda *_args: transformer_releases.append("released")
)
try:
    sampler_node.pipeline.run_h3_sampler = lambda *_args, **_kwargs: sentinel_handle
    transformer_success = sampler_call(plain_condition)
    sampler_node.pipeline.run_h3_sampler = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("synthetic transformer failure")
    )
    try:
        sampler_call(plain_condition)
    except RuntimeError as exc:
        transformer_failed = "synthetic transformer failure" in str(exc)
    else:
        transformer_failed = False
    sampler_node.pipeline.prepare_h3_sampler_components = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("synthetic transformer prepare failure")
    )
    try:
        sampler_call(plain_condition)
    except RuntimeError as exc:
        transformer_prepare_failed = "synthetic transformer prepare failure" in str(exc)
    else:
        transformer_prepare_failed = False
finally:
    sampler_node.CACHE = old_sampler_cache
    sampler_node.pipeline.has_h3_latents = old_has_latents
    sampler_node.pipeline.prepare_h3_sampler_components = old_prepare_sampler
    sampler_node.pipeline.run_h3_sampler = old_run_h3
    sampler_node.pipeline.release_h3_sampler_components = old_release_sampler
check(
    "H3 transformer 在成功、采样异常和准备异常路径都释放",
    transformer_success == (sentinel_handle,)
    and transformer_failed
    and transformer_prepare_failed
    and transformer_releases == ["released", "released", "released"],
)


# --------------------------------------------------------- 7. 三份可导入工作流的节点 / link / slot 契约
workflow_modes = {
    "minimax-h3-i2va-first-frame.json": (True, False),
    "minimax-h3-i2va-last-frame.json": (False, True),
    "minimax-h3-i2va-first-last-frame.json": (True, True),
}
for workflow_name, expected_mode in workflow_modes.items():
    workflow = json.loads((ROOT / "workflows" / workflow_name).read_text(encoding="utf-8"))
    workflow_nodes = {item["id"]: item for item in workflow["nodes"]}
    workflow_links = {item[0]: item for item in workflow["links"]}
    valid = (
        len(workflow_nodes) == len(workflow["nodes"])
        and len(workflow_links) == len(workflow["links"])
        and workflow["last_node_id"] == max(workflow_nodes)
        and workflow["last_link_id"] == max(workflow_links)
    )
    for link_id, source, source_slot, target, target_slot, type_name in workflow["links"]:
        output = workflow_nodes[source]["outputs"][source_slot]
        input_ = workflow_nodes[target]["inputs"][target_slot]
        valid = valid and (
            link_id in (output.get("links") or [])
            and input_.get("link") == link_id
            and output["type"] == input_["type"] == type_name
        )
    for node_id, workflow_node in workflow_nodes.items():
        for input_slot, input_ in enumerate(workflow_node.get("inputs", [])):
            link_id = input_.get("link")
            # ComfyUI 原生 widget 输入不属于可连线 socket，序列化时合法地没有 slot_index。
            if "widget" not in input_ or link_id is not None:
                valid = valid and input_.get("slot_index") == input_slot
            if link_id is not None:
                valid = valid and (
                    link_id in workflow_links
                    and workflow_links[link_id][3:6]
                    == [node_id, input_slot, input_["type"]]
                )
        for output_slot, output in enumerate(workflow_node.get("outputs", [])):
            valid = valid and output.get("slot_index") == output_slot
            for link_id in output.get("links") or []:
                valid = valid and (
                    link_id in workflow_links
                    and workflow_links[link_id][1:3] == [node_id, output_slot]
                    and workflow_links[link_id][5] == output["type"]
                )
    keyframe_workflow_node = next(
        item for item in workflow["nodes"] if item["type"] == "MlxH3KeyframeCondition"
    )
    text_workflow_node = next(
        item for item in workflow["nodes"] if item["type"] == "MlxTextEncoder"
    )
    sampler_workflow_node = next(
        item for item in workflow["nodes"] if item["type"] == "MlxKSamplerMLX"
    )
    vae_workflow_nodes = [
        item for item in workflow["nodes"] if item["type"] == "MlxVAELoader"
    ]
    load_image_nodes = [item for item in workflow["nodes"] if item["type"] == "LoadImage"]
    actual_mode = tuple(
        keyframe_workflow_node["inputs"][slot]["link"] is not None for slot in (1, 2)
    )
    keyframe_text_link = workflow_links[text_workflow_node["inputs"][1]["link"]]
    valid = valid and (
        actual_mode == expected_mode
        and len(load_image_nodes) == sum(expected_mode)
        and sorted(item["widgets_values"][4] for item in vae_workflow_nodes)
        == ["audio_vae", "vae"]
        and keyframe_workflow_node["widgets_values"] == [640, 352]
        and sampler_workflow_node["widgets_values"][2:4] == [640, 352]
        and text_workflow_node["inputs"][1]["name"] == "h3_keyframes"
        and keyframe_text_link[1:5]
        == [keyframe_workflow_node["id"], 0, text_workflow_node["id"], 1]
    )
    check(f"工作流 {workflow_name} 的节点、link、slot 与关键帧模式有效", valid)


if FAILED:
    raise SystemExit(f"\n{len(FAILED)} 项失败：{', '.join(FAILED)}")
print("\n全部 MiniMax-H3 关键帧条件合成检查通过。")