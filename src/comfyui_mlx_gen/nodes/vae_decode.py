"""VAE 解码节点：

- `MlxVAEDecodeRawPIL`（原节点，保留以兼容既有工作流）：自带
  `model_type/model_path/precision/quantize` widget，一步到位解出 PIL；
- `MlxVAEDecoder`（新）：从 `vae` handle（MlxVAELoader 输出）+ `latents` 解出 PIL，
  与 `MlxVAEEncoder` 接同一个 handle 时必然共用同一份 VAE 实例。

两个节点的 VAE 缓存键**完全相同**（都走 `pipeline.vae_component`），因此同一条
工作流里只会驻留一份 VAE（键里含 model_type，升级后首次运行会重新加载一次，
之后新旧节点共享）。解码本身不需要 ModelConfig，因此没有配置变体可选。
"""

from __future__ import annotations

from .. import image, paths, pipeline, runtime
from ..cache import CACHE
from ..types import (
    MlxPilImage,
    MlxVaeHandle,
    entry_for,
    model_types,
    vae,
    validate_model_family,
)

NO_PATH = "<无可用权重>"


def vae_handle_from_widgets(
    model_type, model_path, precision, quantize, role: str = "vae"
) -> MlxVaeHandle:
    """把 widget 里的取值包成 MlxVaeHandle（键与 MlxVAELoader 完全一致）。"""
    return MlxVaeHandle(
        model_type=model_type,
        path=model_path,
        precision=precision,
        quantize=int(quantize),
        role=role,
        cache_key=runtime.cache_key(
            {
                "kind": "vae",
                "model_type": model_type,
                "path": model_path,
                "precision": precision,
                "quantize": int(quantize),
                "role": role,
            }
        ),
    )


def _cached_latents(latents):
    """按 latent 句柄的键取数组（不在缓存里就提示重跑采样器）。"""
    arr, hit = CACHE.get("component_weights", latents.cache_key)
    if not hit:
        raise RuntimeError("latent 数组不在缓存里，请重新运行 MlxKSamplerMLX")
    return arr


def _decode_breeze(latents, batch_index: int) -> MlxPilImage:
    """Breeze：缓存中的最终一维 waveform → 24 kHz AUDIO 载荷（不再加载模型）。"""
    cached, hit = CACHE.get("breeze_waveform", latents.cache_key)
    if not hit:
        raise RuntimeError("Breeze waveform 不在缓存里，请重新运行 MlxKSamplerMLX")
    return pipeline.decode_breeze_waveform(cached, batch_index)


def _decode_with_vae(entry, vae_module, latents, arr, batch_index) -> MlxPilImage:
    """按 batch 逐张解码（latent 形状由 `pipeline.decode_latents` 按大类还原）。"""
    if batch_index >= 0:
        arr = arr[[batch_index % arr.shape[0]]]
    pils = []
    for i in range(arr.shape[0]):
        decoded = pipeline.decode_latents(
            entry, {"vae": vae_module}, arr[i], latents.height, latents.width
        )
        pils.extend(image.to_pil(decoded, batch_index=-1))
    return MlxPilImage(images=tuple(pils), batch_index=batch_index)


def _decode_h3(latents, vae_handle, audio_handle, batch_index: int) -> MlxPilImage:
    """H3：按 VAE 的 role 解 latent（视频行与音频行存在同一条 latent 状态里）。

    主 handle 是 `role="vae"` → 出帧序列；只有把第二个 MlxVAELoader
    （`role="audio_vae"`）接到 `audio_vae` 入口，才会同时带上音轨。
    """
    try:
        main = pipeline.decode_h3_latents(latents, vae_handle, CACHE, batch_index)
        if vae_handle.role == "audio_vae":
            # 只解音频（帧给不了，MlxPilToTorch 会占位一张 1×1 黑图）。
            return main
        if audio_handle is None:
            print(
                "[MlxVAEDecodeRawPIL] 没接 audio_vae（再放一个「MLX VAE 加载器」，"
                "model_path 选 MiniMax-H3、role 选 audio_vae，接到本节点的 audio_vae 入口）："
                "只出画面，没有声音"
            )
            return main
        if audio_handle.role != "audio_vae":
            raise ValueError(
                f"audio_vae 入口接到的 handle role 是 {audio_handle.role!r}，"
                "那样只会把视频 VAE 再解一遍、拿不到音轨："
                "请在第二个「MLX VAE 加载器」里把 role 选成 audio_vae（model_path 选 MiniMax-H3）"
            )
        audio = pipeline.decode_h3_latents(latents, audio_handle, CACHE, -1)
        return MlxPilImage(
            images=main.images, batch_index=batch_index, fps=main.fps, audio=audio.audio
        )
    finally:
        # PIL / AudioTrack 已经是独立输出；无论第二路解码是否抛错，都不能留下
        # 已经用过的 H3 VAE。
        pipeline.release_h3_vae(vae_handle, CACHE)
        if audio_handle is not None and getattr(audio_handle, "role", None) == "audio_vae":
            pipeline.release_h3_vae(audio_handle, CACHE)


def _decode_yue2(latents, vae_handle, batch_index: int) -> MlxPilImage:
    """YuE2：``[frames,64]`` 声学 latent → 48 kHz 立体声载荷。"""
    arr = _cached_latents(latents)
    try:
        return pipeline.decode_yue2_latents(arr, vae_handle, CACHE, batch_index)
    finally:
        pipeline.release_yue2_vae(vae_handle, CACHE)


class MlxVAEDecodeRawPIL:
    @classmethod
    def INPUT_TYPES(cls):
        types = model_types()
        default_type = types[0]
        # 候选取 vae/ + audio_vae/ 的并集（H3 的音频 VAE 两个目录都可能放）
        vae_paths = list(
            dict.fromkeys(
                cls._paths("vae") + cls._paths("audio_vae")
                + [
                    name
                    for name in paths.list_component_items("transformer")
                    if "breeze" in name.lower()
                ]
            )
        ) or [NO_PATH]
        return {
            "required": {
                "latents": ("latents", {}),
                "model_type": (types, {"default": default_type}),
                "model_path": (vae_paths, {"default": vae_paths[0]}),
                "precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "quantize": ([0, 4, 8, 16], {"default": 8}),
                "batch_index": ("INT", {"default": -1, "min": -1, "max": 3}),
            },
            # 只有 MiniMax-H3 用得上：接第二个 MlxVAELoader（role=audio_vae）才出声音
            "optional": {"audio_vae": (vae, {})},
        }

    @staticmethod
    def _paths(role: str) -> list[str]:
        """某 role 目录下可选的权重集（扫盘，新增目录自动出现在下拉里）。"""
        return paths.list_component_items(role) or [NO_PATH]

    RETURN_TYPES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "MLX/Gen"

    def decode(self, latents, model_type, model_path, precision, quantize, batch_index,
               audio_vae=None):
        entry = entry_for(model_type)
        validate_model_family(model_type, model_path)
        if not entry.supported:
            raise NotImplementedError(f"{model_type} 尚未实现：{entry.notes}")
        if latents.model and latents.model != model_type:
            raise ValueError(
                f"latents 是 {latents.model} 产出的，而 VAE 选的是 {model_type}，"
                "请让解码器与采样器的 model_type 一致"
            )
        if latents.kind == "h3_video":
            # 视频行与音频行在同一条 latent 状态里，按 handle 的 role 决定解哪一边
            handle = vae_handle_from_widgets(model_type, model_path, precision, quantize)
            return (_decode_h3(latents, handle, audio_vae, batch_index),)
        if latents.kind == "yue2_audio":
            handle = vae_handle_from_widgets(model_type, model_path, precision, quantize)
            return (_decode_yue2(latents, handle, batch_index),)
        if latents.kind == "breeze_audio":
            return (_decode_breeze(latents, batch_index),)
        arr = _cached_latents(latents)
        # 按同款 handle 走共享键 → 与 MlxVAELoader / MlxVAEDecoder 命中同一份 VAE
        handle = vae_handle_from_widgets(model_type, model_path, precision, quantize)
        try:
            vae_module = pipeline.vae_component(handle, CACHE)
            return (_decode_with_vae(entry, vae_module, latents, arr, batch_index),)
        finally:
            vae_module = None
            pipeline.release_vae_component(handle, CACHE)


class MlxVAEDecoder:
    """MLX VAE 解码：`vae`（MlxVAELoader 输出）+ `latents` → `images`。

    解码逻辑与 MlxVAEDecodeRawPIL 完全相同，唯一区别是 VAE 由 handle 传入，
    因此与 MlxVAEEncoder 接同一个 handle 时必然共用同一份 VAE 实例。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": (vae, {}),
                "latents": ("latents", {}),
                "batch_index": ("INT", {"default": -1, "min": -1, "max": 3}),
            },
            # 只有 MiniMax-H3 用得上：接第二个 MlxVAELoader（role=audio_vae）才出声音
            "optional": {"audio_vae": (vae, {})},
        }

    RETURN_TYPES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "MLX/Gen"

    def decode(self, vae, latents, batch_index, audio_vae=None):
        if vae is None or not isinstance(vae, MlxVaeHandle):
            raise ValueError(
                "必须连接「MLX VAE 加载器」的输出；"
                "ComfyUI 原生 VAE（torch 权重）不能被 MLX 节点使用"
            )
        entry = entry_for(vae.model_type)
        validate_model_family(vae.model_type, vae.path)
        if not entry.supported:
            raise NotImplementedError(f"{vae.model_type} 尚未实现：{entry.notes}")
        # latent 由哪个大类产的就该用哪个大类的 VAE 解（不静默降级）
        if latents.model and latents.model != vae.model_type:
            raise ValueError(
                f"latents 是 {latents.model} 产出的，而 VAE 选的是 {vae.model_type}，"
                "请让解码器与采样器的 model_type 一致"
            )
        if latents.kind == "h3_video":
            # 一份 latent 状态里有视频行 + 音频行，按 handle 的 role 取对应那一边
            return (_decode_h3(latents, vae, audio_vae, batch_index),)
        if latents.kind == "yue2_audio":
            return (_decode_yue2(latents, vae, batch_index),)
        if latents.kind == "breeze_audio":
            return (_decode_breeze(latents, batch_index),)
        arr = _cached_latents(latents)
        # 与 MlxVAEEncoder / MlxVAEDecodeRawPIL 共用同一个键 → 全流程只驻留一份 VAE
        try:
            vae_module = pipeline.vae_component(vae, CACHE)
            return (_decode_with_vae(entry, vae_module, latents, arr, batch_index),)
        finally:
            vae_module = None
            pipeline.release_vae_component(vae, CACHE)
