"""MiniMax-H3 的视觉条件节点（首尾帧 / 多图参考 / 视频续写）。

H3 的视觉条件只有一条通路，而且**只有两个 latent 锚点槽**：
`h3/latent_creator/h3_layout.py` 的 `keyframe_anchors` 只认 `first` / `last`，
`num_condition_rows = len(keyframe_anchors) * rows_per_frame`（每槽一帧）。
所以本文件里的三个节点填的都是「图片 → presentation」这条口，区别只在张数
与有没有锚点：

- `MlxH3KeyframeCondition`（首 / 尾帧）：1~2 张图按目标画布 LANCZOS 拉伸，
  同一份像素既进 Qwen3-VL 的 presentation，又编码成 latent 锚点行
  （采样期间一直保持在 `t = 0.999`，与参考实现一致）；
- `MlxH3MultiReferenceCondition`（多图参考）：2 张以上图**只**进 presentation
  （`<Picture 1>` … `<Picture N>`，不进 latent）；也可以顺手把其中第 1 张 /
  最后 1 张钉成首 / 尾帧锚点，其余只当视觉参考；
- `MlxH3VideoCondition`（视频续写）：从源视频里取首帧或末帧当锚点，
  用来「接着往下写」。整段源视频模型读不到，成片要靠核心的 `GetVideoComponents`
  / `CreateVideo` / `ConcatenateVideo` 自己拼接。

**权重必须跟着任务走**（官方 `model_index.json` 里 transformer 有两条）：

- 首 / 尾帧（I2VA / FL2VA）→ Base：`transformer/MiniMax-H3`；
- 参考生视频（Ref2VA：2 张以上参考图，或不钉锚点）→ **REF**：
  `transformer/MiniMax-H3-ref`（或 `MiniMax-H3-Ref2VA-MLX-Serve-8bit.safetensors`）。
  只有 REF 学过「按参考生成」，用 Base 时参考图只会被当成文本里的一段描述；
  文本编码器 / tokenizer / 两个 VAE 两边共用（都填 `MiniMax-H3`），
  校验由 `h3/pipeline.check_visual_condition_checkpoint` 在采样器里兜住。

三个节点都只产出数据：图片留在 cache.py 的 "h3_keyframe_source" 桶里，
handle 只带摘要、顺序、画布与 VAE 配置。2 张以上的组合由本插件自己实现
（`Qwen3VLModel.encode` 收 `list[patches] + list[grid]`，上游 mlx-gen 只用过
「1 张图 + 1 个锚点」）：首 / 尾锚点之外再多出来的图只进 presentation。
"""

from __future__ import annotations

from PIL import Image

from .. import image as image_mod
from .. import pipeline, runtime
from ..cache import CACHE
from ..h3.latent_creator.h3_layout import (
    MAX_ASPECT_RATIO,
    MIN_ASPECT_RATIO,
    resolve_canvas_size,
)
from ..types import (
    MlxH3VisualCondition,
    MlxRefImageSource,
    MlxVaeHandle,
    h3_keyframes,
    ref_source,
    vae,
)

# 一次最多送进 presentation 的参考图张数（= H3-Base-Ref2VA 的官方上限 9 张；
# 每张都是独立的 vision 块，token 数与显存会线性涨，再多基本只会糊）
MAX_VISUAL_PICTURES = 9


# --- 三个节点共用的几何 / 缓存工具 ------------------------------------------------
def check_video_vae(vae_handle, label: str) -> None:
    """锚点必须用 H3 的视频 VAE 编码；接错大类或接到 audio_vae 立刻报错。"""
    if not isinstance(vae_handle, MlxVaeHandle):
        raise ValueError(f"{label}：vae 必须连接「MLX VAE 加载器」的输出")
    if vae_handle.model_type != "minimax_h3" or vae_handle.role != "vae":
        raise ValueError(
            f"{label}：必须使用 minimax_h3 的视频 VAE（role=vae），"
            f"收到 model_type={vae_handle.model_type!r}, role={vae_handle.role!r}"
        )


def check_canvas(width, height, label: str) -> tuple[int, int]:
    """校验画布是 32 的正整数倍且比例落在 [1/4, 4]（不合规直接报中文错误）。"""
    width, height = int(width), int(height)
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError(
            f"{label}：画布宽高必须是 32 的正整数倍，收到 {width}×{height}"
        )
    ratio = width / height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(
            f"{label}：画布比例必须在 1:4 到 4:1 之间，"
            f"收到 {width}×{height}（{ratio:g}）"
        )
    return width, height


def canvas_from_image(image: Image.Image) -> tuple[int, int]:
    """按图片宽高比解析出 H3 支持的画布 `(宽, 高)`（与参考实现同源）。"""
    height, width = resolve_canvas_size(image.width, image.height)
    return int(width), int(height)


def fit_to_canvas(image: Image.Image, width: int, height: int) -> Image.Image:
    """把图按目标画布 LANCZOS 拉伸（参考实现里 keyframe 就是「stretched onto the canvas」）。"""
    if image.size != (width, height):
        return image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def pack_condition(
    pils: list[Image.Image],
    anchors: tuple[str, ...],
    anchor_images: tuple[int, ...],
    width: int,
    height: int,
    vae_handle: MlxVaeHandle | None,
    source: str,
    source_label: str,
) -> MlxH3VisualCondition:
    """把有序图片存进缓存并组装 handle（换图、换顺序、换锚点都会换键）。"""
    digest = runtime.cache_key(
        {
            "kind": "h3_visual_condition",
            "source": source,
            "images": image_mod.digest(pils),
            "anchors": tuple(anchors),
            "anchor_images": tuple(int(i) for i in anchor_images),
            "width": int(width),
            "height": int(height),
            "vae": vae_handle,
        }
    )
    images_key = runtime.cache_key({"kind": "h3_keyframe_source", "digest": digest})
    CACHE.get_or_create("h3_keyframe_source", images_key, lambda: tuple(pils))
    return MlxH3VisualCondition(
        images_key=images_key,
        picture_count=len(pils),
        anchors=tuple(anchors),
        anchor_images=tuple(int(i) for i in anchor_images),
        width=int(width),
        height=int(height),
        digest=digest,
        source=source,
        source_label=source_label,
        vae=vae_handle,
    )


class MlxH3KeyframeCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": (vae, {}),
                "width": ("INT", {"default": 640, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 352, "min": 32, "max": 4096, "step": 32}),
            },
            "optional": {
                "first_frame": ("IMAGE", {}),
                "last_frame": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = (h3_keyframes, "STRING")
    RETURN_NAMES = ("keyframes", "report")
    FUNCTION = "pack"
    CATEGORY = "MLX/Gen"

    def pack(self, vae, width, height, first_frame=None, last_frame=None):
        check_video_vae(vae, "H3 关键帧")
        width, height = check_canvas(width, height, "H3 关键帧")

        anchors: list[str] = []
        fitted: list[Image.Image] = []
        for anchor, value, label in (
            ("first", first_frame, "first_frame"),
            ("last", last_frame, "last_frame"),
        ):
            if value is None:
                continue
            batch = image_mod.to_pil_batch(value)
            if len(batch) != 1:
                raise ValueError(f"{label} 必须恰好包含一张 IMAGE，收到 {len(batch)} 张")
            anchors.append(anchor)
            fitted.append(fit_to_canvas(batch[0].convert("RGB"), width, height))
        if not fitted:
            raise ValueError("至少连接 first_frame 或 last_frame 中的一张关键帧")

        mode = "首尾帧" if len(anchors) == 2 else ("首帧" if anchors[0] == "first" else "尾帧")
        handle = pack_condition(
            fitted,
            tuple(anchors),
            tuple(range(len(anchors))),
            width,
            height,
            vae,
            "keyframe",
            mode,
        )
        report = f"H3 {mode}条件 | {width}×{height} | anchors={','.join(anchors)}"
        print(f"[MlxH3KeyframeCondition] {report}")
        return handle, report


# --- 2. 多图参考（N 张只进 presentation；可选把其中 1~2 张钉成锚点）--------------
class MlxH3MultiReferenceCondition:
    """H3 多图参考条件：接「MLX 参考图集」的 N 张图（各自尺寸，只进 presentation）。

    2 张以上（或不钉锚点）就是「参考生视频」，「MLX 模型加载器」必须选
    Minimax-H3-REF；只有把 1~2 张图钉成首 / 尾锚点时才该用 Base 的 MiniMax-H3。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_images": (ref_source, {}),
                "anchor": (
                    ["none", "first", "last", "first_last"],
                    {
                        "default": "none",
                        "tooltip": "none = 纯参考（Ref2VA）：图只进 Qwen3-VL 的 presentation，"
                                   "不占 latent 行，必须选 Minimax-H3-REF 权重；"
                                   "first / last / first_last = 把对应图片钉成首 / 尾帧锚点"
                                   "（I2VA / FL2VA），用 Base 的 MiniMax-H3；"
                                   "3 张以上只有 REF 支持（上限 9 张）",
                    },
                ),
                "use_source_aspect": ("BOOLEAN", {"default": True}),
                "width": ("INT", {"default": 640, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 352, "min": 32, "max": 4096, "step": 32}),
            },
            "optional": {"vae": (vae, {})},
        }

    RETURN_TYPES = (h3_keyframes, "STRING")
    RETURN_NAMES = ("keyframes", "report")
    FUNCTION = "pack"
    CATEGORY = "MLX/Gen"

    def pack(self, ref_images, anchor, use_source_aspect, width, height, vae=None):
        if not isinstance(ref_images, MlxRefImageSource):
            raise ValueError("ref_images 必须连接「MLX 参考图集」的输出")
        pils = [pil.convert("RGB") for pil in pipeline.load_ref_source(CACHE, ref_images.cache_key)]
        if len(pils) > MAX_VISUAL_PICTURES:
            raise ValueError(
                f"H3 一次最多送 {MAX_VISUAL_PICTURES} 张参考图（现在是 {len(pils)} 张）："
                "每张图都是独立的 vision 块，token 与显存会线性涨，再多只会糊"
            )

        if use_source_aspect:
            width, height = canvas_from_image(pils[0])
        width, height = check_canvas(width, height, "H3 多图参考")
        if vae is not None:
            check_video_vae(vae, "H3 多图参考")

        anchor_names: tuple[str, ...] = ()
        anchor_images: tuple[int, ...] = ()
        if anchor == "first":
            anchor_names, anchor_images = ("first",), (0,)
        elif anchor == "last":
            anchor_names, anchor_images = ("last",), (len(pils) - 1,)
        elif anchor == "first_last":
            if len(pils) < 2:
                raise ValueError("锚点选 first_last 时至少要 2 张参考图（首帧一张、尾帧一张）")
            anchor_names, anchor_images = ("first", "last"), (0, len(pils) - 1)
        if anchor_names and vae is None:
            raise ValueError(
                f"锚点 {','.join(anchor_names)} 需要视频 VAE 编码成 latent 锚点："
                "请把「MLX VAE 加载」的 vae 输出接到本节点的 vae 入口"
            )

        # 锚点图按画布拉伸（要进 latent），其余参考图保留自己的尺寸（只进 presentation）
        prepared = list(pils)
        for index in anchor_images:
            prepared[index] = fit_to_canvas(pils[index], width, height)

        handle = pack_condition(
            prepared,
            anchor_names,
            anchor_images,
            width,
            height,
            vae,
            "reference",
            f"{len(pils)} 张参考图",
        )
        anchor_note = (
            "无锚点（纯参考）"
            if not anchor_names
            else "、".join(
                f"{name}←第{index + 1}张" for name, index in zip(anchor_names, anchor_images)
            )
        )
        report = (
            f"H3 多图参考 | {len(pils)} 张 → <Picture 1..{len(pils)}> | "
            f"画布 {width}×{height} | 锚点={anchor_note}"
        )
        print(f"[MlxH3MultiReferenceCondition] {report}")
        if not anchor_names or len(pils) > 2:
            print(
                "[MlxH3MultiReferenceCondition] 提示：prompt 里请用 <Picture 1> / <Picture 2> … "
                "引用对应图片；这是「参考生视频」（Ref2VA），"
                "「MLX 模型加载器」必须选 Minimax-H3-REF"
                "（MiniMax-H3-ref 或 MiniMax-H3-Ref2VA-MLX-Serve-8bit.safetensors），"
                "否则采样器会直接报错"
            )
        else:
            print(
                "[MlxH3MultiReferenceCondition] 提示：prompt 里请用 <Picture 1> / <Picture 2> … "
                "引用对应图片；1~2 张并钉成首 / 尾锚点是 Base 的 I2VA / FL2VA 任务，"
                "「MLX 模型加载器」保持 MiniMax-H3 即可"
            )
        return handle, report


# --- 3. 视频条件（取源视频的首 / 末帧当锚点，用来续写）---------------------------
class MlxH3VideoCondition:
    """H3 视频条件：从源视频取一帧当锚点；整段源视频模型读不到，只当条件来源。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {}),
                "mode": (
                    ["continue_from_end", "continue_from_start"],
                    {
                        "default": "continue_from_end",
                        "tooltip": "continue_from_end：取源视频最后一帧，钉在新生成的第一帧"
                        "（往前续写）；continue_from_start：取第一帧，钉在新生成的最后一帧"
                        "（往前补一段）",
                    },
                ),
                "use_source_aspect": ("BOOLEAN", {"default": True}),
                "width": ("INT", {"default": 640, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 352, "min": 32, "max": 4096, "step": 32}),
            },
            "optional": {"vae": (vae, {})},
        }

    RETURN_TYPES = (h3_keyframes, "STRING")
    RETURN_NAMES = ("keyframes", "report")
    FUNCTION = "pack"
    CATEGORY = "MLX/Gen"

    def pack(self, video, mode, use_source_aspect, width, height, vae=None):
        if video is None or not hasattr(video, "get_components"):
            raise ValueError("video 必须连接 ComfyUI 的「Load Video」/「Get Video Components」等节点")
        if vae is None:
            raise ValueError(
                "H3 的视频条件必须把取出来的那一帧编码成 latent 锚点："
                "请把「MLX VAE 加载」的 vae 输出接到本节点的 vae 入口"
            )
        check_video_vae(vae, "H3 视频条件")

        components = video.get_components()
        pils = image_mod.to_pil_batch(components.images)
        if not pils:
            raise ValueError("源视频里一帧都没有，请换一个有效的视频")
        if len(pils) < 2:
            raise ValueError(f"源视频只有 {len(pils)} 帧，不足以做续写条件")

        anchor = "first" if mode == "continue_from_end" else "last"
        frame = pils[-1] if mode == "continue_from_end" else pils[0]
        frame_index = len(pils) - 1 if mode == "continue_from_end" else 0
        source_size = (pils[0].width, pils[0].height)
        if use_source_aspect:
            width, height = canvas_from_image(frame)
        width, height = check_canvas(width, height, "H3 视频条件")
        anchor_image = fit_to_canvas(frame, width, height)

        handle = pack_condition(
            [anchor_image],
            (anchor,),
            (0,),
            width,
            height,
            vae,
            "video",
            f"源视频第 {frame_index + 1} / {len(pils)} 帧",
        )
        report = (
            f"H3 视频条件 | 源 {source_size[0]}×{source_size[1]} "
            f"({len(pils)} 帧 @ {float(components.frame_rate):g}fps) → 画布 {width}×{height} | "
            f"取源视频第 {frame_index + 1} 帧当 {anchor} 锚点"
        )
        print(f"[MlxH3VideoCondition] {report}")
        print(
            "[MlxH3VideoCondition] 提示：H3 不会读整段源视频，只会看到这一张锚点图"
            "（首 / 尾锚点属于 Base 的 I2VA 任务，「MLX 模型加载器」选 MiniMax-H3；"
            "要把原片段也放进成片，请用 GetVideoComponents + CreateVideo + ConcatenateVideo 拼接）"
        )
        return handle, report