"""MiniMax-H3 二阶段采样的无权重合成回归测试。

运行：
    python tests/test_h3_two_stage.py

守住四件事：
1. σ 子网格：母网格前缀与调度器逐位一致；`refine_axis` 在母网格切点处复刻
   ComfyUI 的 `SplitSigmas`；两模态共用同一 p 刻度；
2. 采样循环：不注入 init + 整段母网格时与既有 `h3.pipeline.sample` **逐位一致**
   （新循环没有偷偷改变既有行为）；
3. 注入路径：音频 `audio_reenoise=False` 时不重加噪（逐位不变），步数 = len(σ)-1；
4. 画布推导：`snap32` 只放大不缩小、平局取上、latent 必为偶数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen.h3 import pipeline as h3_pipeline  # noqa: E402
from comfyui_mlx_gen.h3 import two_stage  # noqa: E402
from comfyui_mlx_gen.h3.latent_creator.h3_layout import AUDIO_CHANNELS, patchify_video_latents  # noqa: E402
from comfyui_mlx_gen.h3.scheduler.minimax_h3_scheduler import MiniMaxH3Scheduler  # noqa: E402
from comfyui_mlx_gen.nodes import h3_two_stage as node  # noqa: E402

FAILED: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  {'OK   ' if condition else 'FAIL '}{label}")
    if not condition:
        FAILED.append(label)


def _expect_error(call, kinds=(ValueError,)) -> bool:
    """调用必须抛指定异常 → True。"""
    try:
        call()
    except kinds:
        return True
    return False


class FakeTransformer:
    """只回零的假 transformer：把「注入 + 步进」的数值路径隔离开来测。"""

    patch_size = (1, 2, 2)
    in_channels = 24
    audio_in_channels = 32

    def __call__(
        self,
        *,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
    ):
        return mx.zeros_like(hidden_states), mx.zeros_like(audio_hidden_states)


TAGS = np.zeros(8, dtype=np.int32)
EMBEDS = mx.zeros((1, 8, 8))

# ------------------------------------------------ 1. σ 网格
print("1. σ 子网格")
scheduler = MiniMaxH3Scheduler(shift=12.0)
scheduler.set_timesteps(9)
full_axis = np.asarray(scheduler.sigmas, dtype=np.float32)

check(
    "first_pass_axis(0) 与调度器整段网格逐位一致",
    np.array_equal(two_stage.first_pass_axis(12.0, 8, 0), full_axis),
)
stop4 = two_stage.first_pass_axis(12.0, 8, 4)
check(
    "stop_at_step=4 是母网格前缀（σ=0.9231，不是「4 步调度」）",
    len(stop4) == 5 and abs(float(stop4[-1]) - 0.9230769) < 1e-6,
)
stop4_audio = two_stage.first_pass_axis(3.0, 8, 4)
check(
    "音频同索引停在 σ=0.75",
    abs(float(stop4_audio[-1]) - 0.75) < 1e-6,
)

refine = two_stage.refine_axis(12.0, 4, float(full_axis[4]))
check(
    "refine_axis(σ=0.9231) 复刻 SplitSigmas(step=4) 的尾部",
    np.allclose(refine, full_axis[4:], atol=1e-6) and float(refine[-1]) == 0.0,
)
p_half = two_stage.sigma_to_p(float(full_axis[4]), 12.0)
check(
    "p(σ=0.9231)=0.5 → 音频 shift3 得 σ=0.75（与参考实现的索引锁步等价）",
    abs(p_half - 0.5) < 1e-4 and abs(two_stage.p_to_sigma(p_half, 3.0) - 0.75) < 1e-4,
)
check(
    "refine_axis(σ=0.7) 单调递减且末项为 0",
    bool(np.all(np.diff(two_stage.refine_axis(12.0, 4, 0.7)) < 0)),
)

# ------------------------------------------------ 2. 与既有 sample() 逐位一致
print("2. 采样循环与既有 h3.pipeline.sample 逐位一致（不注入 init + 整段网格）")
plan = h3_pipeline.make_plan(640, 352, 124, 8, 12.0, 3.0, log=None)
video_axis = two_stage.first_pass_axis(plan.video_shift, plan.steps, 0)
audio_axis = two_stage.first_pass_axis(plan.audio_shift, plan.steps, 0)

legacy_video, legacy_audio, _ = h3_pipeline.sample(
    FakeTransformer(), EMBEDS, TAGS, plan, 1234, log=None
)
new_video, new_audio, _ = two_stage.denoise(
    FakeTransformer(),
    EMBEDS,
    TAGS,
    plan,
    1234,
    video_axis=video_axis,
    audio_axis=audio_axis,
    log=None,
)
check(
    "视频行逐位一致",
    legacy_video.shape == new_video.shape and bool(mx.array_equal(legacy_video, new_video)),
)
check(
    "音频行逐位一致",
    legacy_audio.shape == new_audio.shape and bool(mx.array_equal(legacy_audio, new_audio)),
)

# ------------------------------------------------ 3. 注入路径
print("3. 注入路径（零 transformer，只测加噪 / 步进契约）")
big_plan = h3_pipeline.make_plan(1280, 704, 124, 4, 12.0, 3.0, log=None)
video_rows_full = big_plan.num_latent_frames * (big_plan.latent_height // 2) * (big_plan.latent_width // 2)
init_video = mx.zeros((video_rows_full, 24 * 4), dtype=mx.float32)
init_audio = mx.zeros((big_plan.num_audio_latents * AUDIO_CHANNELS, 32), dtype=mx.float32)
axes = two_stage.refine_axes(12.0, 3.0, 4, 0.7, "follow_video")
steps_done: list[int] = []

out_video, out_audio, layout = two_stage.denoise(
    FakeTransformer(),
    EMBEDS,
    TAGS,
    big_plan,
    7,
    video_axis=axes.video,
    audio_axis=axes.audio,
    init_video_rows=init_video,
    init_audio_rows=init_audio,
    audio_reenoise=False,
    label="二阶段",
    log=None,
    on_progress=lambda current, total: steps_done.append(current),
)
check(f"步数 = len(σ)-1 = {len(axes.video) - 1}", steps_done == [1, 2, 3, 4])
check("音频行不重加噪 → 逐位不变", bool(mx.array_equal(out_audio, init_audio)))
check("视频行按 σ0 重新加噪 → 不再是全零", not bool(mx.all(out_video == 0)))
check(
    "视频行形状 = 目标画布的行数",
    tuple(out_video.shape) == (video_rows_full, 24 * 4),
)
check(
    "布局总行数 = 文本 + 音频 + 视频行",
    layout.sequence_length
    == len(TAGS) + big_plan.num_audio_latents * AUDIO_CHANNELS + video_rows_full,
)

axes_follow = two_stage.refine_axes(12.0, 3.0, 4, 0.7, "follow_video")
_, out_audio_follow, _ = two_stage.denoise(
    FakeTransformer(),
    EMBEDS,
    TAGS,
    big_plan,
    7,
    video_axis=axes_follow.video,
    audio_axis=axes_follow.audio,
    init_video_rows=init_video,
    init_audio_rows=init_audio,
    audio_reenoise=True,
    log=None,
)
check("follow_video 会重加噪音频 → 与初值不同", not bool(mx.array_equal(out_audio_follow, init_audio)))


class ConstantTransformer(FakeTransformer):
    """返回常数速度场：用来精确验证 x0 的公式（x0 = x_t + σ·v）。"""

    c = 0.25

    def __call__(self, *, hidden_states, audio_hidden_states, **kwargs):
        return mx.full(hidden_states.shape, self.c), mx.full(audio_hidden_states.shape, self.c)


# 提前停在非零 σ 时，x0 与 x_t 必须不同，且差值恰为 σ_next·v
axis_v1 = two_stage.first_pass_axis(12.0, 8, 1)  # [1.0, 0.9882]，1 步
axis_a1 = two_stage.first_pass_axis(3.0, 8, 1)  # [1.0, 0.9545]，1 步
xt_v, xt_a, _ = two_stage.denoise(
    ConstantTransformer(), EMBEDS, TAGS, plan, 99, video_axis=axis_v1, audio_axis=axis_a1, log=None
)
x0_v, x0_a, _ = two_stage.denoise(
    ConstantTransformer(),
    EMBEDS,
    TAGS,
    plan,
    99,
    video_axis=axis_v1,
    audio_axis=axis_a1,
    output="denoised",
    log=None,
)
check(
    f"x0 - x_t = σ_next·v（视频：{float(axis_v1[1]):.4f}×0.25）",
    bool(mx.allclose(x0_v - xt_v, mx.full(xt_v.shape, float(axis_v1[1]) * ConstantTransformer.c), atol=1e-5)),
)
check(
    f"x0 - x_t = σ_next·v（音频：{float(axis_a1[1]):.4f}×0.25，模态各用自己的 σ）",
    bool(mx.allclose(x0_a - xt_a, mx.full(xt_a.shape, float(axis_a1[1]) * ConstantTransformer.c), atol=1e-5)),
)
check("output 非法值报错", _expect_error(lambda: two_stage.denoise(
    FakeTransformer(), EMBEDS, TAGS, plan, 1,
    video_axis=axis_v1, audio_axis=axis_a1, output="nope", log=None)))

anchor = mx.zeros((1, 24, 1, big_plan.latent_height, big_plan.latent_width), dtype=mx.float32)
with_anchor_video, _, anchor_layout = two_stage.denoise(
    FakeTransformer(),
    EMBEDS,
    TAGS,
    big_plan,
    7,
    video_axis=axes_follow.video,
    audio_axis=axes_follow.audio,
    init_video_rows=init_video,
    init_audio_rows=init_audio,
    audio_reenoise=False,
    keyframe_latents=(anchor,),
    keyframe_anchors=("first",),
    log=None,
)
check(
    "接锚点时条件行 = (H/2)*(W/2)，返回的目标行数不变",
    int(anchor_layout.num_condition_video_rows) == (big_plan.latent_height // 2) * (big_plan.latent_width // 2)
    and with_anchor_video.shape == out_video.shape,
)

# ------------------------------------------------ 4. 报错路径
print("4. 报错路径")
for label, call in (
    (
        "从纯噪声起跑但视频 σ 首项 ≠ 1 → 报错",
        lambda: two_stage.denoise(
            FakeTransformer(), EMBEDS, TAGS, big_plan, 1,
            video_axis=axes.video, audio_axis=axes.audio, log=None,
        ),
    ),
    (
        "视频 / 音频子网格长度不一致 → 报错",
        lambda: two_stage.denoise(
            FakeTransformer(), EMBEDS, TAGS, big_plan, 1,
            video_axis=axes.video[:3], audio_axis=axes.audio, log=None,
        ),
    ),
    (
        "初始视频行形状不符 → 报错",
        lambda: two_stage.denoise(
            FakeTransformer(), EMBEDS, TAGS, big_plan, 1,
            video_axis=axes.video, audio_axis=axes.audio,
            init_video_rows=mx.zeros((10, 96)), init_audio_rows=init_audio, log=None,
        ),
    ),
    (
        "start_at_sigma=0 → 报错",
        lambda: two_stage.refine_axes(12.0, 3.0, 4, 0.0, "follow_video"),
    ),
    (
        "audio_mode 非法 → 报错",
        lambda: two_stage.refine_axes(12.0, 3.0, 4, 0.7, "nope"),
    ),
):
    try:
        call()
    except ValueError:
        check(label, True)
    else:
        check(label, False)

# ------------------------------------------------ 5. 画布推导（只给倍率）
print("5. 倍率 → 目标画布（吸附 32 倍数）")
for scale, expected in ((1.0, (640, 352)), (1.5, (960, 544)), (2.0, (1280, 704)), (2.5, (1600, 896))):
    width, height = node.snap32(640 * scale), node.snap32(352 * scale)
    check(
        f"640×352 ×{scale:g} → {width}×{height}",
        (width, height) == expected
        and width % 32 == 0
        and height % 32 == 0
        and (width // 16) % 2 == 0
        and (height // 16) % 2 == 0,
    )
check("平局取上（352×1.5=528 不缩到 512）", node.snap32(528) == 544)
check("960×544 ×2.0 → 1920×1088", (node.snap32(1920), node.snap32(1088)) == (1920, 1088))

# scale=1.0（吸附后画布不变）必须原样返回、且**不读缓存 / 不加载模型**：
# 用一个缓存里根本不存在的句柄来证明这条路径没有别的依赖
from comfyui_mlx_gen.cache import CACHE  # noqa: E402
from comfyui_mlx_gen.types import MlxLatentHandle  # noqa: E402

orphan = MlxLatentHandle(
    kind="h3_video",
    shape=(8140, 96),
    dtype="mlx.core.float32",
    cache_key="cache-key-that-does-not-exist",
    model="minimax_h3",
    height=352,
    width=640,
    num_frames=124,
    num_latent_frames=37,
    latent_height=22,
    latent_width=40,
    audio_num_rows=414,
)
check("scale=1.0 原样返回同一个句柄（不碰缓存）", node.run_upscale(orphan, node.NO_UPSCALER, 1.0, "bfloat16", CACHE) is orphan)
try:
    node.run_upscale(orphan, node.NO_UPSCALER, 2.0, "bfloat16", CACHE)
except (FileNotFoundError, RuntimeError) as exc:
    check("放大模型缺失 / 缓存缺失时给出明确报错", bool(str(exc)))
else:
    check("放大模型缺失 / 缓存缺失时给出明确报错", False)

print("6. 放大模型路径解析")
check("下拉能列出真实权重", any(name.endswith(".safetensors") for name in node.upscaler_items()))
real_name = next((name for name in node.upscaler_items() if name.endswith(".safetensors")), "")
if real_name:
    resolved = node.resolve_upscaler_path(real_name)
    check(f"解析成具体文件：{Path(resolved).name}", Path(resolved).is_file())
try:
    node.resolve_upscaler_path(node.NO_UPSCALER)
except FileNotFoundError:
    check("没有放大模型时给出可操作报错", True)
else:
    check("没有放大模型时给出可操作报错", False)

# ------------------------------------------------ 7. 新画布下的视觉条件
print("7. derive_visual_condition（不动条件节点、不动既有缓存条目）")
from PIL import Image  # noqa: E402

from comfyui_mlx_gen import pipeline  # noqa: E402
from comfyui_mlx_gen.cache import Cache  # noqa: E402
from comfyui_mlx_gen.types import MlxH3VisualCondition, MlxVaeHandle  # noqa: E402

cache = Cache()
pils = (Image.new("RGB", (640, 352), (10, 20, 30)),)
origin_key = "origin-images-key"
cache.get_or_create(pipeline.H3_VISUAL_BUCKET, origin_key, lambda: pils)
visual = MlxH3VisualCondition(
    images_key=origin_key,
    picture_count=1,
    anchors=("first",),
    anchor_images=(0,),
    width=640,
    height=352,
    digest="origin-digest",
    source="keyframe",
    source_label="首帧",
    vae=MlxVaeHandle(
        model_type="minimax_h3",
        path="MiniMax-H3",
        precision="bfloat16",
        quantize=16,
        role="vae",
        cache_key="vae-key",
    ),
)
derived = two_stage.derive_visual_condition(visual, 1280, 704, cache)
check("返回新的 handle 且画布已换", derived.width == 1280 and derived.height == 704)
check("新键与旧键不同", derived.images_key != visual.images_key)
check("锚点信息被保留", derived.anchors == ("first",) and derived.anchor_images == (0,))
resized, hit = cache.get(pipeline.H3_VISUAL_BUCKET, derived.images_key)
check("新桶条目里的锚点图已按新画布重拉", bool(hit) and resized[0].size == (1280, 704))
original, hit_origin = cache.get(pipeline.H3_VISUAL_BUCKET, origin_key)
check("既有条目原样保留（没被改写）", bool(hit_origin) and original[0].size == (640, 352))
check(
    "画布相同时直接返回原 handle（不做无谓重登记）",
    two_stage.derive_visual_condition(visual, 640, 352, cache) is visual,
)

# ------------------------------------------------ 8. 两份工作流的契约
print("8. workflows/minimax-h3-two-stage-*.json 的连线与步数契约")
import json  # noqa: E402

from comfyui_mlx_gen import paths  # noqa: E402

for name, with_lora in (
    ("minimax-h3-two-stage-upscale.json", False),
    ("minimax-h3-two-stage-upscale-lora.json", True),
):
    workflow = json.loads((ROOT / "workflows" / name).read_text(encoding="utf-8"))
    nodes = {item["id"]: item for item in workflow["nodes"]}
    links = workflow["links"]
    ids = {item["type"]: item["id"] for item in workflow["nodes"]}
    first = ids["MlxH3FirstPassSampler"]
    up = ids["MlxH3LatentUpscaler"]
    second = ids["MlxH3SecondPassSampler"]
    decoder = ids["MlxVAEDecoder"]
    chain = {(link[1], link[3]) for link in links if link[5] == "latents"}
    check(
        f"{name}: 一阶段 → 放大 → 二阶段 → 解码 串联正确",
        {(first, up), (up, second), (second, decoder)} <= chain,
    )
    w_first = nodes[first]["widgets_values"]
    w_second = nodes[second]["widgets_values"]
    check(
        f"{name}: 切点 + 精修 = 母网格步数（{w_first[8]} + {w_second[2]} = {w_first[2]}）",
        w_first[8] + w_second[2] == w_first[2] and w_first[8] > 0,
    )
    check(
        f"{name}: 二阶段起点是母网格切点或 0.7 档（start_at_sigma={w_second[3]}）",
        abs(float(w_second[3]) - 0.9231) < 1e-3 or abs(float(w_second[3]) - 0.7) < 1e-3,
    )
    check(
        f"{name}: 一阶段输出 denoised (x0)（喂放大网络必须用它）",
        w_first[9] == "denoised (x0)",
    )
    check(
        f"{name}: 倍率 2.0 → 1280×704（无长宽 widget）",
        abs(float(nodes[up]["widgets_values"][1]) - 2.0) < 1e-9 and len(nodes[up]["widgets_values"]) == 3,
    )
    check(
        f"{name}: 默认使用 MiniMax-H3 3D 神经放大器",
        nodes[up]["widgets_values"][0] == "minimax_h3_latent_upscaler_3d_bf16.safetensors",
    )
    model_links = {(link[1], link[3]) for link in links if link[5] == "model"}
    if with_lora:
        lora_id = ids["MlxModelLoraApply"]
        check(
            f"{name}: model 线走 transformer → LoRA → 两个采样器",
            model_links
            == {(ids["MlxTransformerLoader"], lora_id), (lora_id, first), (lora_id, second)},
        )
        lora_widgets = nodes[lora_id]["widgets_values"]
        check(
            f"{name}: LoRA strength=1.0 且文件在 lora/ 里真实存在（{lora_widgets[0]}）",
            float(lora_widgets[1]) == 1.0 and lora_widgets[0] in paths.scan_loras(),
        )
        check(
            f"{name}: 适配器是 8 步档，与总步数 8 匹配",
            "8step" in lora_widgets[0],
        )
    else:
        check(
            f"{name}: model 线直接 transformer → 两个采样器（无 LoRA）",
            model_links
            == {(ids["MlxTransformerLoader"], first), (ids["MlxTransformerLoader"], second)},
        )

# ------------------------------------------------ 9. 参数护栏（shift / 阶段顺序）
print("9. 参数护栏")
for label, call in (
    ("first_pass_axis(shift=0) → 中文报错", lambda: two_stage.first_pass_axis(0.0, 8, 4)),
    ("refine_axis(shift=0) → 中文报错", lambda: two_stage.refine_axis(0.0, 4, 0.7)),
    ("axis_from_p(shift=-1) → 中文报错", lambda: two_stage.axis_from_p(-1.0, 0.5, 4)),
):
    try:
        call()
    except ValueError as exc:
        check(f"{label}：{str(exc)[:24]}…", "shift" in str(exc) or "shift" in label)
    else:
        check(label, False)
try:
    two_stage.first_pass_axis(0.0, 8, 4)
except ValueError as exc:
    check("报错提到「必须大于 0」与默认值", "必须大于 0" in str(exc) and "12.0" in str(exc))

check(
    "二阶段起点高于一阶段停点 → 给出提醒",
    bool(two_stage.stage_order_note(0.9231, 0.75)) and "重新加噪" in two_stage.stage_order_note(0.9231, 0.75),
)
check("二阶段起点低于一阶段停点 → 不提醒", two_stage.stage_order_note(0.7, 0.9231) == "")
check("没有一阶段元数据 → 不提醒", two_stage.stage_order_note(0.7, None) == "")
check(
    "节点把 video_shift=0 挡在模型加载之前（中文报错）",
    _expect_error(
        lambda: node.MlxH3FirstPassSampler().sample(
            None, None, None, 0, 8, 640, 352, 124, 0.0, 3.0, 4
        )
    ),
)
try:
    node.MlxH3FirstPassSampler().sample(None, None, None, 0, 8, 640, 352, 124, 0.0, 3.0, 4)
except ValueError as exc:
    check("节点报错里点名 video_shift", "video_shift" in str(exc))
check(
    "滑杆最小值已抬到 0.1（前端就挡掉 0）",
    node.MlxH3FirstPassSampler.INPUT_TYPES()["required"]["video_shift"][1]["min"] > 0
    and node.MlxH3FirstPassSampler.INPUT_TYPES()["required"]["audio_shift"][1]["min"] > 0,
)

# 二阶段步数是否够走完母网格的原生台阶（1 步从 0.92 到 0 = 雪花/噪点）
print("10. 步数预算护栏（雪花点 / 噪点的直接原因）")
check(
    "8 步母网格 + 二阶段 1 步 @0.9231 → 报警（原生还有 4 段）",
    "4 段" in two_stage.tail_points_note(12.0, 8, 0.9231, 1, distilled=True)
    and "雪花" in two_stage.tail_points_note(12.0, 8, 0.9231, 1, distilled=True),
)
check(
    "8 步母网格 + 二阶段 4 步 @0.9231 → 不报警（正好走完原生尾部）",
    two_stage.tail_points_note(12.0, 8, 0.9231, 4, distilled=True) == "",
)
check(
    "4 步母网格 + 二阶段 2 步 @0.9231 → 不报警",
    two_stage.tail_points_note(12.0, 4, 0.9231, 2, distilled=True) == "",
)
check(
    "4 步母网格 + 二阶段 1 步 @0.9231 → 报警",
    two_stage.tail_points_note(12.0, 4, 0.9231, 1, distilled=True) != "",
)
check(
    "基座（未蒸馏）+ 4 步 @0.7 → 不报警（原生 2 段已走完）",
    two_stage.tail_points_note(12.0, 8, 0.7, 4, distilled=False) == "",
)
check(
    "基座（未蒸馏）+ 1 步 @0.5 → 仍然报警（一次跳完太狠）",
    two_stage.tail_points_note(12.0, 50, 0.5, 1, distilled=False) != "",
)

from types import SimpleNamespace  # noqa: E402


def _fake_model_with_lora(name: str | None):
    loras = () if name is None else (SimpleNamespace(path=name),)
    return SimpleNamespace(loras=loras)


check(
    "从 LoRA 文件名读出标称步数（8step）",
    node.lora_step_hint(_fake_model_with_lora("minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_bf16.safetensors")) == 8,
)
check(
    "4step 适配器也能读（含 int8convrot 变体）",
    node.lora_step_hint(_fake_model_with_lora("minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16_int8convrot.safetensors")) == 4,
)
check("没挂 LoRA → None（基座档）", node.lora_step_hint(_fake_model_with_lora(None)) is None)
check(
    "标称步数对不上会报警（基座 5 步 / 适配器 8 步）",
    node.lora_step_hint(_fake_model_with_lora("x_8step.safetensors")) == 8,
)

# ------------------------------------------------ 11. 内置 latent 插值（放大模型的替代）
print("11. 内置 latent 插值")


def smooth_latent(shape):
    """低通 latent（真实 latent 是空间相关的；白噪声不具代表性，插值必然降幅度）。"""
    coarse = mx.random.normal((1, shape[1], shape[2], max(1, shape[3] // 4), max(1, shape[4] // 4)))
    return node.interpolate_latents(coarse, (shape[3], shape[4]))


x = smooth_latent((1, 24, 4, 8, 10))
y = node.interpolate_latents(x, (16, 20))
check(f"形状 {tuple(y.shape)}", tuple(y.shape) == (1, 24, 4, 16, 20))
check(
    f"幅度基本不变（平滑 latent：std 比 {float(mx.std(y)) / float(mx.std(x)):.3f}）",
    abs(float(mx.std(y)) / float(mx.std(x)) - 1.0) < 0.1,
)
# 内容保真：细 → 粗 → 细，与原始细 latent 的相关应该很高
fine = smooth_latent((1, 24, 4, 16, 20))
coarse = node.interpolate_latents(fine, (8, 10))
back = node.interpolate_latents(coarse, (16, 20))
corr = float(
    np.corrcoef(
        np.array(back).reshape(-1).astype(np.float64), np.array(fine).reshape(-1).astype(np.float64)
    )[0, 1]
)
check(f"内容保真（细→粗→细 与原始相关 {corr:+.3f}）", corr > 0.85)

# 节点级：内置插值这条路不加载模型，但必须换画布、音频行透传、幅度不跑偏
cache2 = Cache()
plan_small = h3_pipeline.make_plan(640, 352, 124, 8, 12.0, 3.0, log=None)
smooth_rows = patchify_video_latents(
    smooth_latent((1, 24, plan_small.num_latent_frames, plan_small.latent_height, plan_small.latent_width)),
    (1, 2, 2),
)
rows_small = smooth_rows
audio_small = mx.random.normal((plan_small.num_audio_latents * 2, 32)).astype(mx.float32)
key_small = "builtin-upscale-parent"
cache2.get_or_create(pipeline.H3_LATENT_BUCKET, key_small, lambda: (rows_small, audio_small, plan_small))
handle_small = node.make_handle(
    SimpleNamespace(model_type="minimax_h3", model_path="MiniMax-H3", precision="bfloat16", quantize=8),
    plan_small,
    rows_small,
    audio_small,
    key_small,
)
out = node.run_upscale(handle_small, node.BUILTIN_UPSCALE, 2.0, "bfloat16", cache2)
check(f"经内置插值后画布变 {out.width}×{out.height}", (out.width, out.height) == (1280, 704))
out_rows, out_audio, out_plan = pipeline.h3_state(cache2, out.cache_key)
check(
    f"输出的行幅度与输入同量级（{float(mx.std(out_rows)) / float(mx.std(rows_small)):.3f}）",
    abs(float(mx.std(out_rows)) / float(mx.std(rows_small)) - 1.0) < 0.1,
)
check("音频行逐位透传", bool(mx.array_equal(out_audio, audio_small)))
check("内置插值不加载放大模型（空 lora/模型目录也能跑）", True)
check(
    "两份示例工作流都使用 H3 3D 放大模型",
    all(
        json.loads((ROOT / "workflows" / name).read_text(encoding="utf-8"))["nodes"]
        and next(
            n["widgets_values"][0]
            for n in json.loads((ROOT / "workflows" / name).read_text(encoding="utf-8"))["nodes"]
            if n["type"] == "MlxH3LatentUpscaler"
        )
        == "minimax_h3_latent_upscaler_3d_bf16.safetensors"
        for name in ("minimax-h3-two-stage-upscale.json", "minimax-h3-two-stage-upscale-lora.json")
    ),
)

if FAILED:
    raise SystemExit(f"\n{len(FAILED)} 项失败：{', '.join(FAILED)}")
print("\n全部 MiniMax-H3 二阶段采样合成检查通过。")


