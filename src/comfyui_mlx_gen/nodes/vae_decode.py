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
from ..types import MlxPilImage, MlxVaeHandle, entry_for, model_types, vae

NO_PATH = "<无可用权重>"


def vae_handle_from_widgets(model_type, model_path, precision, quantize) -> MlxVaeHandle:
    """把 widget 里的四个取值包成 MlxVaeHandle（键与 MlxVAELoader 完全一致）。"""
    return MlxVaeHandle(
        model_type=model_type,
        path=model_path,
        precision=precision,
        quantize=int(quantize),
        cache_key=runtime.cache_key(
            {
                "kind": "vae",
                "model_type": model_type,
                "path": model_path,
                "precision": precision,
                "quantize": int(quantize),
            }
        ),
    )


def _cached_latents(latents):
    """按 latent 句柄的键取数组（不在缓存里就提示重跑采样器）。"""
    arr, hit = CACHE.get("component_weights", latents.cache_key)
    if not hit:
        raise RuntimeError("latent 数组不在缓存里，请重新运行 MlxKSamplerMLX")
    return arr


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


class MlxVAEDecodeRawPIL:
    @classmethod
    def INPUT_TYPES(cls):
        types = model_types()
        default_type = types[0]
        vae_paths = cls._paths("vae")
        return {
            "required": {
                "latents": ("latents", {}),
                "model_type": (types, {"default": default_type}),
                "model_path": (vae_paths, {"default": vae_paths[0]}),
                "precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "quantize": ([4, 8, 16], {"default": 8}),
                "batch_index": ("INT", {"default": -1, "min": -1, "max": 3}),
            }
        }

    @staticmethod
    def _paths(role: str) -> list[str]:
        """某 role 目录下可选的权重集（扫盘，新增目录自动出现在下拉里）。"""
        return paths.list_component_items(role) or [NO_PATH]

    RETURN_TYPES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "MLX/Gen"

    def decode(self, latents, model_type, model_path, precision, quantize, batch_index):
        entry = entry_for(model_type)
        if not entry.supported:
            raise NotImplementedError(f"{model_type} 尚未实现：{entry.notes}")
        arr = _cached_latents(latents)
        # 按同款 handle 走共享键 → 与 MlxVAELoader / MlxVAEDecoder 命中同一份 VAE
        handle = vae_handle_from_widgets(model_type, model_path, precision, quantize)
        vae_module = pipeline.vae_component(handle, CACHE)
        return (_decode_with_vae(entry, vae_module, latents, arr, batch_index),)


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
            }
        }

    RETURN_TYPES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "MLX/Gen"

    def decode(self, vae, latents, batch_index):
        if vae is None or not isinstance(vae, MlxVaeHandle):
            raise ValueError(
                "必须连接「MLX VAE 加载器」的输出；"
                "ComfyUI 原生 VAE（torch 权重）不能被 MLX 节点使用"
            )
        entry = entry_for(vae.model_type)
        if not entry.supported:
            raise NotImplementedError(f"{vae.model_type} 尚未实现：{entry.notes}")
        # latent 由哪个大类产的就该用哪个大类的 VAE 解（不静默降级）
        if latents.model and latents.model != vae.model_type:
            raise ValueError(
                f"latents 是 {latents.model} 产出的，而 VAE 选的是 {vae.model_type}，"
                "请让解码器与采样器的 model_type 一致"
            )
        arr = _cached_latents(latents)
        # 与 MlxVAEEncoder / MlxVAEDecodeRawPIL 共用同一个键 → 全流程只驻留一份 VAE
        vae_module = pipeline.vae_component(vae, CACHE)
        return (_decode_with_vae(entry, vae_module, latents, arr, batch_index),)
