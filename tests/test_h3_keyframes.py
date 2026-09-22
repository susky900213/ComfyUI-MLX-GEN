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

from comfyui_mlx_gen import image, paths, pipeline  # noqa: E402
from comfyui_mlx_gen.cache import CACHE, Cache  # noqa: E402
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
from comfyui_mlx_gen.nodes import ref_image_set as ref_set_node  # noqa: E402
from comfyui_mlx_gen.nodes import sampler as sampler_node  # noqa: E402
from comfyui_mlx_gen.nodes import text_encoder as text_encoder_node  # noqa: E402
from comfyui_mlx_gen.types import (  # noqa: E402
    MlxClipHandle,
    MlxConditioning,
    MlxH3VisualCondition,
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

    cached_both, hit = node_cache.get("h3_keyframe_source", both.images_key)
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


# ------------------------------------------------- 1b. 多图参考条件（N 张只进 presentation）
# 参考图集与 H3 视觉条件节点共用 cache.py 的同一个实例：两边都得指到同一份缓存，
# 否则会出现「读不到参考图」的假故障。h3_keyframe_source 桶只留 2 条，
# 因此这里换成独立 Cache，并在每次 pack 之后马上核对。
old_keyframe_cache = keyframe_node.CACHE
old_ref_cache = ref_set_node.CACHE
iso_cache = Cache()
keyframe_node.CACHE = iso_cache
ref_set_node.CACHE = iso_cache
try:
    ref_set = ref_set_node.MlxRefImageSet()
    multi_source, multi_count, _ = ref_set.pack(
        tensor_image(8, 12), tensor_image(10, 14, 0.5), tensor_image(6, 9, 0.75)
    )
    check("参考图集按顺序打包 3 张", multi_count == 3, str(multi_count))

    multi_node = keyframe_node.MlxH3MultiReferenceCondition()
    pure_ref, pure_report = multi_node.pack(multi_source, "none", True, 64, 32)
    pure_images, pure_hit = iso_cache.get("h3_keyframe_source", pure_ref.images_key)
    check(
        "纯参考：三张图都进 presentation，但没有 latent 锚点",
        pure_ref.anchors == ()
        and pure_ref.anchor_images == ()
        and pure_ref.picture_count == 3
        and pure_ref.vae is None
        and pure_hit
        and [pil.size for pil in pure_images] == [(12, 8), (14, 10), (9, 6)],
        f"{pure_ref.anchors} / {pure_ref.anchor_images} / {pure_ref.picture_count}",
    )
    check(
        "纯参考时不物化 Video VAE，也不占 latent 行",
        pipeline.encode_h3_keyframes(pure_ref, iso_cache) == ((), ()),
    )

    ref_first, _ = multi_node.pack(multi_source, "first", True, 64, 32, vae=video_vae)
    first_images, first_hit = iso_cache.get("h3_keyframe_source", ref_first.images_key)
    check(
        "按源宽高比解析画布（4:3 给出 32 倍数且比例接近）",
        ref_first.width % 32 == 0
        and ref_first.height % 32 == 0
        and abs(ref_first.width / ref_first.height - 12 / 8) < 0.2,
        f"{ref_first.width}×{ref_first.height}",
    )
    check(
        "只钉第 1 张：它按画布拉伸，后两张仍是原尺寸",
        ref_first.anchors == ("first",)
        and ref_first.anchor_images == (0,)
        and first_hit
        and first_images[0].size == (ref_first.width, ref_first.height)
        and first_images[1].size == (14, 10)
        and first_images[2].size == (9, 6),
        str([pil.size for pil in first_images]),
    )

    ref_last, _ = multi_node.pack(multi_source, "last", True, 64, 32, vae=video_vae)
    check(
        "只钉最后 1 张：锚点指向第 3 张",
        ref_last.anchors == ("last",) and ref_last.anchor_images == (2,),
        f"{ref_last.anchors} / {ref_last.anchor_images}",
    )

    ref_both, both_report = multi_node.pack(multi_source, "first_last", True, 64, 32, vae=video_vae)
    both_images, both_hit = iso_cache.get("h3_keyframe_source", ref_both.images_key)
    check(
        "首尾都钉时锚点映射到非相邻下标（0 与 2）",
        ref_both.anchors == ("first", "last")
        and ref_both.anchor_images == (0, 2)
        and ref_both.picture_count == 3,
        f"{ref_both.anchors} / {ref_both.anchor_images}",
    )
    check(
        "被点名当锚点的两张图按画布拉伸，中间那张仍是原尺寸",
        both_hit
        and both_images[0].size == (ref_both.width, ref_both.height)
        and both_images[1].size == (14, 10)
        and both_images[2].size == (ref_both.width, ref_both.height),
        str([pil.size for pil in both_images]),
    )
    check(
        "换锚点会换 handle（摘要与图片缓存键都跟着变）",
        len({pure_ref.digest, ref_first.digest, ref_last.digest, ref_both.digest}) == 4
        and len(
            {
                pure_ref.images_key,
                ref_first.images_key,
                ref_last.images_key,
                ref_both.images_key,
            }
        )
        == 4,
    )
    check(
        "报告里写明 <Picture 1..N> 与锚点映射",
        "<Picture 1..3>" in pure_report
        and "无锚点（纯参考）" in pure_report
        and "first←第1张" in both_report
        and "last←第3张" in both_report,
        f"{pure_report} / {both_report}",
    )
    single_source, _, _ = ref_set.pack(tensor_image(8, 12))
    check_raises(
        "只有一张参考图时拒绝选 first_last",
        ValueError,
        "至少要 2 张",
        lambda: multi_node.pack(single_source, "first_last", True, 64, 32, vae=video_vae),
    )
    check_raises(
        "锚点不是 none 却没接 VAE 时明确拒绝",
        ValueError,
        "需要视频 VAE",
        lambda: multi_node.pack(multi_source, "first", True, 64, 32),
    )
    check_raises(
        "没接参考图集（handle 类型不对）时拒绝",
        ValueError,
        "必须连接「MLX 参考图集」",
        lambda: multi_node.pack("not-a-ref-source", "none", True, 64, 32),
    )


    big_source, big_count, _ = ref_set.pack(
        torch.cat([tensor_image(8, 12), tensor_image(8, 12, 0.1), tensor_image(8, 12, 0.2)], dim=0),
        torch.cat([tensor_image(8, 12), tensor_image(8, 12, 0.3), tensor_image(8, 12, 0.4)], dim=0),
        torch.cat([tensor_image(8, 12), tensor_image(8, 12, 0.5), tensor_image(8, 12, 0.6)], dim=0),
        torch.cat([tensor_image(8, 12), tensor_image(8, 12, 0.7), tensor_image(8, 12, 0.8)], dim=0),
    )
    check("批次槽位展开后共 12 张参考图", big_count == 12, str(big_count))
    check_raises(
        "参考图超过 9 张时直接拒绝（H3-Base-Ref2VA 的官方上限是 9 张）",
        ValueError,
        "最多送 9 张",
        lambda: multi_node.pack(big_source, "none", True, 64, 32),
    )

    # --------------------------------------------- 1c. 视频条件（从源视频取一帧当锚点）
    frames = torch.cat(
        [tensor_image(36, 64), tensor_image(36, 64, 0.25), tensor_image(36, 64, 0.5)], dim=0
    )
    fake_video = SimpleNamespace(
        file_path="fake-source.mp4",
        get_components=lambda: SimpleNamespace(images=frames, audio=None, frame_rate=25),
    )
    video_node = keyframe_node.MlxH3VideoCondition()
    cont, cont_report = video_node.pack(
        fake_video, "continue_from_end", True, 64, 32, vae=video_vae
    )
    manual_last_frame = image.to_pil_batch(frames)[-1].resize(
        (cont.width, cont.height), Image.Resampling.LANCZOS
    )
    cont_images, cont_hit = iso_cache.get("h3_keyframe_source", cont.images_key)
    check(
        "续写模式取源视频最后一帧钉成 first 锚点",
        cont.anchors == ("first",)
        and cont.anchor_images == (0,)
        and cont_hit
        and cont_images[0].tobytes() == manual_last_frame.tobytes(),
    )
    back, _ = video_node.pack(fake_video, "continue_from_start", True, 64, 32, vae=video_vae)
    manual_first_frame = image.to_pil_batch(frames)[0].resize(
        (back.width, back.height), Image.Resampling.LANCZOS
    )
    back_images, back_hit = iso_cache.get("h3_keyframe_source", back.images_key)
    check(
        "倒补模式取源视频第一帧钉成 last 锚点",
        back.anchors == ("last",)
        and back_hit
        and back_images[0].tobytes() == manual_first_frame.tobytes(),
    )
    check(
        "按源视频宽高比解析画布（16:9 → 32 倍数且比例接近）",
        cont.width % 32 == 0
        and cont.height % 32 == 0
        and cont.width > 64
        and abs(cont.width / cont.height - 64 / 36) < 0.05
        and cont.source == "video",
        f"{cont.width}×{cont.height}",
    )
    check(
        "报告与 source_label 标明源视频与取用的帧",
        "源视频第 3 / 3 帧" in cont.source_label
        and "源视频第 3 帧当 first 锚点" in cont_report,
        f"{cont.source_label} / {cont_report}",
    )
    check_raises(
        "没接 VAE 时拒绝（锚点必须能编码成 latent）",
        ValueError,
        "必须把取出来的那一帧编码成 latent 锚点",
        lambda: video_node.pack(fake_video, "continue_from_end", True, 64, 32),
    )
    check_raises(
        "源视频只有一帧时拒绝做续写条件",
        ValueError,
        "不足以做续写条件",
        lambda: video_node.pack(
            SimpleNamespace(
                file_path="one-frame.mp4",
                get_components=lambda: SimpleNamespace(
                    images=tensor_image(36, 64), audio=None, frame_rate=25
                ),
            ),
            "continue_from_end",
            True,
            64,
            32,
            vae=video_vae,
        ),
    )
    check_raises(
        "接的不是 VIDEO（没有 get_components）时拒绝",
        ValueError,
        "必须连接 ComfyUI 的「Load Video」",
        lambda: video_node.pack(object(), "continue_from_end", True, 64, 32, vae=video_vae),
    )

    # --------------------------------------------- 1e. 图片外观 + 视频动作 / 运镜
    motion_frames = torch.cat(
        [tensor_image(8, 16, index / 50.0) for index in range(22)], dim=0
    )
    motion_video = SimpleNamespace(
        file_path="fake-motion.mp4",
        get_components=lambda: SimpleNamespace(images=motion_frames, audio=None, frame_rate=25),
    )
    motion_image = tensor_image(7, 11, 0.4)
    motion_node = keyframe_node.MlxH3MotionReferenceWithImageCondition()
    combined, combined_report = motion_node.pack(
        motion_image,
        motion_video,
        64,
        32,
        124,
        False,
        2.0,
        vae=video_vae,
    )
    combined_images, combined_image_hit = iso_cache.get(
        "h3_keyframe_source", combined.images_key
    )
    combined_motion, combined_motion_hit = iso_cache.get(
        "h3_motion_source", combined.motion_key
    )
    check(
        "图片动作迁移保留一张 <Picture 1> 且视频裁成合法 17n+5 帧",
        combined.picture_count == 1
        and combined.anchors == ()
        and combined.source == "motion_reference_with_picture"
        and combined.motion_frame_count == 22
        and combined_images[0].size == (64, 32)
        and combined_image_hit,
    )
    check(
        "图片动作迁移默认按 2 fps 抽样并保留完整视频缓存",
        combined_motion_hit
        and len(combined_motion["frames"]) == 22
        and len(combined_motion["presentation_frames"]) == 2
        and "<Picture 1>" in combined_report
        and "<Video 1>" in combined_report,
        combined_report,
    )
    combined_again, _ = motion_node.pack(
        motion_image,
        motion_video,
        64,
        32,
        124,
        False,
        2.0,
        vae=video_vae,
    )
    check(
        "图片或动作视频不变时组合条件命中同一组缓存键",
        combined_again.digest == combined.digest
        and combined_again.motion_key == combined.motion_key,
    )
    check_raises(
        "图片动作迁移没有视频 VAE 时拒绝",
        ValueError,
        "必须连接「MLX VAE 加载器」",
        lambda: motion_node.pack(motion_image, motion_video, 64, 32, 124, False, 2.0),
    )
    check(
        "图片动作迁移只能使用 REF transformer",
        h3_pipeline.checkpoint_tier_for_condition(combined) == "ref"
        and "源视频同时" in h3_pipeline.check_visual_condition_checkpoint(
            SimpleNamespace(model_path="MiniMax-H3-ref"), combined
        ),
    )
finally:
    keyframe_node.CACHE = old_keyframe_cache
    ref_set_node.CACHE = old_ref_cache


# ---------------------------------------- 1d. 任务档位 ↔ transformer 权重（参考要用 REF）
# 「参考生视频」走的是官方 H3-Base-Ref2VA 那条权重（transformer_ref / MiniMax-H3-ref）：
# 只有它学过按参考生成，用 Base 时参考图只会被当成文本里的一段描述。
base_model = MlxModelHandle(
    model_type="minimax_h3",
    model_path="MiniMax-H3",
    quantize=8,
    precision="bfloat16",
    compile=False,
    compile_cache_limit=2,
)
ref_model = MlxModelHandle(
    model_type="minimax_h3",
    model_path="MiniMax-H3-ref",
    quantize=8,
    precision="bfloat16",
    compile=False,
    compile_cache_limit=2,
)
serve_ref_model = MlxModelHandle(
    model_type="minimax_h3",
    model_path="MiniMax-H3-Ref2VA-MLX-Serve-8bit.safetensors",
    quantize=8,
    precision="bfloat16",
    compile=False,
    compile_cache_limit=2,
)
check("MiniMax-H3 仍然算 Base 权重", not h3_pipeline.uses_ref_checkpoint(base_model.model_path))
check(
    "目录名 MiniMax-H3-ref 识别成 REF 权重",
    h3_pipeline.uses_ref_checkpoint(ref_model.model_path),
    ref_model.model_path,
)
check(
    "单文件 MiniMax-H3-Ref2VA-… 也识别成 REF 权重",
    h3_pipeline.uses_ref_checkpoint(serve_ref_model.model_path),
    serve_ref_model.model_path,
)
check(
    "3 张纯参考 → 该用 REF；首尾锚点 → 该用 Base；纯文生视频 → 两档都行",
    h3_pipeline.checkpoint_tier_for_condition(pure_ref) == "ref"
    and h3_pipeline.checkpoint_tier_for_condition(both) == "base"
    and h3_pipeline.checkpoint_tier_for_condition(None) == "any",
)
check_raises(
    "拿 Base 权重跑参考生视频（3 张参考图）直接拒绝",
    ValueError,
    "必须用 Minimax-H3-REF",
    lambda: h3_pipeline.check_visual_condition_checkpoint(base_model, pure_ref),
)
check_raises(
    "2 张但都不钉锚点：Base 权重同样拒绝（纯参考只有 REF 学过）",
    ValueError,
    "必须用 Minimax-H3-REF",
    lambda: h3_pipeline.check_visual_condition_checkpoint(
        base_model,
        MlxH3VisualCondition(
            images_key="two-refs",
            picture_count=2,
            anchors=(),
            width=64,
            height=32,
            digest="two-refs",
            source="reference",
            source_label="2 张参考图",
        ),
    ),
)
check(
    "REF 权重 + 3 张参考图放行，并说明图只进 presentation",
    "只进 Qwen3-VL 的 presentation"
    in h3_pipeline.check_visual_condition_checkpoint(ref_model, pure_ref),
)
check(
    "用 REF 权重跑首尾锚点只给提示，不报错",
    "I2VA / FL2VA" in h3_pipeline.check_visual_condition_checkpoint(ref_model, both),
)
check(
    "Base 权重 + 首尾锚点完全对味（不需要提示）",
    h3_pipeline.check_visual_condition_checkpoint(base_model, both) == "",
)
check_raises(
    "REF 权重下 10 张图仍超上限（官方 9 张）",
    ValueError,
    "最多喂 9 张",
    lambda: h3_pipeline.check_visual_condition_checkpoint(
        ref_model,
        MlxH3VisualCondition(
            images_key="ten-refs",
            picture_count=10,
            anchors=("first", "last"),
            anchor_images=(0, 9),
            width=64,
            height=32,
            digest="ten-refs",
            source="reference",
            source_label="10 张参考图",
            vae=video_vae,
        ),
    ),
)
check_raises(
    "Base 权重下 3 张以上按上限直接拒绝",
    ValueError,
    "最多喂 2 张",
    lambda: h3_pipeline.check_visual_condition_checkpoint(
        base_model,
        MlxH3VisualCondition(
            images_key="three-refs",
            picture_count=3,
            anchors=("first", "last"),
            anchor_images=(0, 2),
            width=64,
            height=32,
            digest="three-refs",
            source="reference",
            source_label="3 张参考图（钉首尾）",
            vae=video_vae,
        ),
    ),
)


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
prompt_cache.get_or_create("h3_keyframe_source", both.images_key, lambda: cached_both)
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

missing = MlxH3VisualCondition(
    images_key="missing",
    picture_count=1,
    anchors=("first",),
    anchor_images=(0,),
    width=64,
    height=32,
    digest="missing",
    source="keyframe",
    source_label="缺失的关键帧",
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


def make_audio_vae_condition() -> MlxH3VisualCondition:
    """尝试伪造「用 Audio VAE 编码锚点」的 handle（构造阶段就该挡住）。"""
    return MlxH3VisualCondition(
        images_key="forged-audio",
        picture_count=1,
        anchors=("first",),
        anchor_images=(0,),
        width=64,
        height=32,
        digest="forged-audio",
        source="keyframe",
        source_label="伪造的音频 VAE 关键帧",
        vae=MlxVaeHandle(
            model_type="minimax_h3",
            path="MiniMax-H3",
            precision="bfloat16",
            quantize=8,
            role="audio_vae",
        ),
    )


check_raises(
    "handle 在数据类构造阶段就拒绝 H3 Audio VAE",
    ValueError,
    "role=vae",
    make_audio_vae_condition,
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
lifecycle_cache.get_or_create("h3_keyframe_source", first.images_key, lambda: (cached_both[0],))
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
    vae_success == (("first",), ("latent",))
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
        and sampler_workflow_node["widgets_values"][3:5] == [640, 352]
        and text_workflow_node["inputs"][1]["name"] == "h3_keyframes"
        and keyframe_text_link[1:5]
        == [keyframe_workflow_node["id"], 0, text_workflow_node["id"], 1]
    )
    check(f"工作流 {workflow_name} 的节点、link、slot 与关键帧模式有效", valid)


# ------------------------------------ 8. 七份新工作流（多图 / 视频 / 纯参考 / 加速 LoRA）
def workflow_is_valid(workflow):
    """核对工作流的节点、link、slot 是否自洽（links 数组与节点槽位双向对得上）。"""
    nodes = {item["id"]: item for item in workflow["nodes"]}
    links = {item[0]: item for item in workflow["links"]}
    valid = (
        len(nodes) == len(workflow["nodes"])
        and len(links) == len(workflow["links"])
        and workflow["last_node_id"] == max(nodes)
        and workflow["last_link_id"] == max(links)
    )
    for link_id, source, source_slot, target, target_slot, type_name in workflow["links"]:
        output = nodes[source]["outputs"][source_slot]
        input_ = nodes[target]["inputs"][target_slot]
        valid = valid and (
            link_id in (output.get("links") or [])
            and input_.get("link") == link_id
            and output["type"] == input_["type"] == type_name
        )
    for _, workflow_node in nodes.items():
        for input_slot, input_ in enumerate(workflow_node.get("inputs", [])):
            link_id = input_.get("link")
            valid = valid and input_.get("slot_index") == input_slot
            if link_id is not None:
                valid = valid and (
                    link_id in links
                    and links[link_id][3:6] == [workflow_node["id"], input_slot, input_["type"]]
                )
        for output_slot, output in enumerate(workflow_node.get("outputs", [])):
            valid = valid and output.get("slot_index") == output_slot
            for link_id in output.get("links") or []:
                valid = valid and (
                    link_id in links
                    and links[link_id][1:3] == [workflow_node["id"], output_slot]
                    and links[link_id][5] == output["type"]
                )
    return valid, nodes, links


VISUAL_TYPES = {
    "MlxH3KeyframeCondition",
    "MlxH3MultiReferenceCondition",
    "MlxH3VideoCondition",
    "MlxH3MotionReferenceCondition",
    "MlxH3MotionReferenceWithImageCondition",
}
# 值 = (必须出现的节点, 视觉条件是否带 latent 锚点, 期望的 transformer 权重集)。
# 参考生视频（多图 / 纯参考）必须落在 Minimax-H3-ref 上，其余都留在 Base。
WORKFLOW_CONTRACTS = {
    "minimax-h3-single-image-to-video.json": (
        {"MlxH3KeyframeCondition", "MlxTextEncoder", "MlxKSamplerMLX", "MlxVAELoader"},
        True,
        "MiniMax-H3",
    ),
    "minimax-h3-multi-image-to-video.json": (
        {"MlxRefImageSet", "MlxH3MultiReferenceCondition", "MlxKSamplerMLX"},
        True,
        "MiniMax-H3-ref",
    ),
    "minimax-h3-reference-only-to-video.json": (
        {"MlxRefImageSet", "MlxH3MultiReferenceCondition", "MlxKSamplerMLX"},
        False,
        "MiniMax-H3-ref",
    ),
    # 全能参考（4 张图 + Ref2VA 的 4 步加速 LoRA）：跟「纯参考」同一条契约，
    # 只是多了一颗必须挂在 transformer → 采样器之间的 LoRA
    "minimax-h3-all-reference-to-video.json": (
        {"MlxRefImageSet", "MlxH3MultiReferenceCondition", "MlxModelLoraApply",
         "MlxKSamplerMLX"},
        False,
        "MiniMax-H3-ref",
    ),
    "minimax-h3-video-continuation-keep-audio.json": (
        {"LoadVideo", "MlxH3VideoCondition", "GetVideoComponents", "CreateVideo",
         "ConcatenateVideo", "AudioConcat", "SaveVideo"},
        True,
        "MiniMax-H3",
    ),
    "minimax-h3-video-continuation-drop-audio.json": (
        {"LoadVideo", "MlxH3VideoCondition", "CreateVideo", "SaveVideo"},
        True,
        "MiniMax-H3",
    ),
    "minimax-h3-video-continuation-replace-audio.json": (
        {"LoadVideo", "MlxH3VideoCondition", "LoadAudio", "CreateVideo", "SaveVideo"},
        True,
        "MiniMax-H3",
    ),
    "minimax-h3-motion-transfer-with-image.json": (
        {"LoadImage", "LoadVideo", "MlxH3MotionReferenceWithImageCondition", "MlxKSamplerMLX"},
        False,
        "MiniMax-H3-ref",
    ),
}
for workflow_name, (required_types, has_anchor, expected_checkpoint) in WORKFLOW_CONTRACTS.items():
    workflow = json.loads((ROOT / "workflows" / workflow_name).read_text(encoding="utf-8"))
    valid, workflow_nodes, workflow_links = workflow_is_valid(workflow)
    types = {item["type"] for item in workflow["nodes"]}
    visual = next(item for item in workflow["nodes"] if item["type"] in VISUAL_TYPES)
    text = next(item for item in workflow["nodes"] if item["type"] == "MlxTextEncoder")
    sampler = next(item for item in workflow["nodes"] if item["type"] == "MlxKSamplerMLX")
    transformer = next(item for item in workflow["nodes"] if item["type"] == "MlxTransformerLoader")
    clip_loader = next(item for item in workflow["nodes"] if item["type"] == "MlxClipLoader")
    vae_loader = next(item for item in workflow["nodes"] if item["type"] == "MlxVAELoader")
    # 视觉源必须真的接到条件节点：只连 VAE、或者漏连 ref_images，都等于图白配
    source_inputs = [item for item in visual["inputs"] if item["name"] != "vae"]
    wired_sources = sum(1 for item in source_inputs if item["link"] is not None)
    if visual["type"] == "MlxH3KeyframeCondition":
        # 首 / 尾帧允许只接一张，但一张都不接就是配错
        valid = valid and wired_sources >= 1
    elif visual["type"] == "MlxH3MotionReferenceWithImageCondition":
        source_names = {"image", "video"}
        valid = valid and wired_sources == 2 and all(
            item["link"] is not None
            for item in source_inputs
            if item["name"] in source_names
        )
    else:
        # 图集 / 源视频是这类条件的唯一入口，必须接上
        valid = valid and wired_sources == len(source_inputs) == 1
    keyframe_links = [
        item for item in workflow["links"] if item[3] == text["id"] and item[4] == 1
    ]
    condition_links = [
        item for item in workflow["links"] if item[3] == sampler["id"] and item[4] in (1, 2)
    ]
    valid = (
        valid
        and required_types <= types
        and len(keyframe_links) == 1
        and keyframe_links[0][1] == visual["id"]
        and keyframe_links[0][5] == "mlx_h3_keyframes"
        and len(condition_links) == 2
        and all(item[1] == text["id"] for item in condition_links)
        # 视觉条件的画布必须与采样器一致，否则采样器会直接报错。
        # 组合节点的前两个 widget 是 width / height；旧视觉节点的最后两个
        # widget 才是 width / height。
        and (
            visual["widgets_values"][:2] if visual["type"] == "MlxH3MotionReferenceWithImageCondition"
            else visual["widgets_values"][-2:]
        ) == sampler["widgets_values"][3:5]
        # seed 后面必须写前端插的 control_after_generate，否则整排 widget 会错位一格
        and sampler["widgets_values"][1] in {"fixed", "increment", "decrement", "randomize"}
        # 没有锚点时不该接视频 VAE；有锚点时必须接
        and (
            (visual["inputs"][1]["link"] is not None) == has_anchor
            if visual["type"] != "MlxH3MotionReferenceWithImageCondition"
            else visual["inputs"][7]["link"] is not None
        )
        # 参考生视频（不钉锚点 / 2 张以上）必须落在 REF 权重上，
        # 否则采样器会在 check_visual_condition_checkpoint 里直接报错
        and transformer["widgets_values"][1] == expected_checkpoint
        and h3_pipeline.uses_ref_checkpoint(transformer["widgets_values"][1])
        == (expected_checkpoint == "MiniMax-H3-ref")
        and (has_anchor or h3_pipeline.uses_ref_checkpoint(transformer["widgets_values"][1]))
        # 条件编码器 / tokenizer / VAE 两边共用同一套（REF 只换 transformer）
        and clip_loader["widgets_values"][2] == "MiniMax-H3"
        and vae_loader["widgets_values"][1] == "MiniMax-H3"
        and workflow["id"].startswith("mlx-")
        and bool(workflow["extra_workflow"]["title"])
    )
    check(f"工作流 {workflow_name} 的节点、link、slot 与视觉条件契约有效", valid)


# ------------------------------------ 9. 加速 LoRA 的链路与步数（唯一走 4 步的工作流）
accel_workflow = json.loads(
    (ROOT / "workflows" / "minimax-h3-all-reference-to-video.json").read_text(encoding="utf-8")
)
accel_valid, accel_nodes, _ = workflow_is_valid(accel_workflow)
accel_sampler_id = next(i for i, node in accel_nodes.items() if node["type"] == "MlxKSamplerMLX")
accel_lora_id = next(i for i, node in accel_nodes.items() if node["type"] == "MlxModelLoraApply")
accel_transformer_id = next(
    i for i, node in accel_nodes.items() if node["type"] == "MlxTransformerLoader"
)
# 模型链必须是「transformer → 加速 LoRA → 采样器」，中间不能有人绕过 LoRA
accel_model_chain = {
    tuple(item[1:5])
    for item in accel_workflow["links"]
    if item[5] == "model" and item[3] in (accel_lora_id, accel_sampler_id)
}
check(
    "工作流 minimax-h3-all-reference-to-video 走 Ref2VA 的 4 步加速 LoRA",
    accel_valid
    and accel_model_chain
    == {
        (accel_transformer_id, 0, accel_lora_id, 0),
        (accel_lora_id, 0, accel_sampler_id, 0),
    }
    # 步数按适配器标称填 4（填 50 步等于白装）；H3 不看 scheduler，
    # 但写 minimax_h3 才不会让采样节点打印「已忽略 scheduler」的提示
    # （widgets：0 seed / 1 control_after_generate / 2 steps / … / 7 scheduler）
    and accel_nodes[accel_sampler_id]["widgets_values"][2] == 4
    and accel_nodes[accel_sampler_id]["widgets_values"][7] == "minimax_h3"
    and accel_nodes[accel_lora_id]["widgets_values"]
    == ["minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors",
        1.0]
    # 文件名必须是 lora/ 里真存在的那颗（任务标记是 ref2v；fl2v 与未标任务的别换）
    and accel_nodes[accel_lora_id]["widgets_values"][0] in paths.scan_loras(),
)


if FAILED:
    raise SystemExit(f"\n{len(FAILED)} 项失败：{', '.join(FAILED)}")
print("\n全部 MiniMax-H3 关键帧条件合成检查通过。")