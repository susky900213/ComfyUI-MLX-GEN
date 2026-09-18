"""MiniMax-H3 首帧 / 尾帧条件节点。

同一组画布适配后的像素同时供 Qwen3-VL 视觉 presentation 与 H3 Video VAE
latent 锚点使用。图片放在进程缓存中，节点输出只保留稳定摘要和轻量 handle。
"""

from __future__ import annotations

from PIL import Image

from .. import image as image_mod
from .. import runtime
from ..cache import CACHE
from ..types import MlxH3Keyframes, MlxVaeHandle, h3_keyframes, vae


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
        if not isinstance(vae, MlxVaeHandle):
            raise ValueError("vae 必须连接「MLX VAE 加载器」的输出")
        if vae.model_type != "minimax_h3" or vae.role != "vae":
            raise ValueError(
                "H3 关键帧必须使用 minimax_h3 的视频 VAE（role=vae），"
                f"收到 model_type={vae.model_type!r}, role={vae.role!r}"
            )
        width, height = int(width), int(height)
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError(f"MiniMax-H3 关键帧画布宽高必须是 32 的正整数倍，收到 {width}×{height}")

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
            frame = batch[0].convert("RGB")
            if frame.size != (width, height):
                frame = frame.resize((width, height), Image.Resampling.LANCZOS)
            anchors.append(anchor)
            fitted.append(frame)
        if not fitted:
            raise ValueError("至少连接 first_frame 或 last_frame 中的一张关键帧")

        pixel_digest = image_mod.digest(fitted)
        digest = runtime.cache_key(
            {
                "kind": "h3_keyframes",
                "anchors": tuple(anchors),
                "width": width,
                "height": height,
                "pixels": pixel_digest,
                "vae": vae,
            }
        )
        cache_key = runtime.cache_key({"kind": "h3_keyframe_source", "digest": digest})
        CACHE.get_or_create("h3_keyframe_source", cache_key, lambda: tuple(fitted))
        handle = MlxH3Keyframes(
            anchors=tuple(anchors),
            width=width,
            height=height,
            digest=digest,
            cache_key=cache_key,
            vae=vae,
        )
        mode = "首尾帧" if len(anchors) == 2 else ("首帧" if anchors[0] == "first" else "尾帧")
        report = f"H3 {mode}条件 | {width}×{height} | anchors={','.join(anchors)}"
        print(f"[MlxH3KeyframeCondition] {report}")
        return handle, report