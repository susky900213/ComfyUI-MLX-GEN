"""MlxVAEDecodeRawPIL：从缓存取 latent 数组，用配置的 VAE 解码，返回 PIL。

只按「模型大类」（决定用哪个 VAE 类与权重定义）+「权重目录」（扫盘得到）
取组件；解码本身不需要 ModelConfig，因此没有配置变体可选。
"""

from __future__ import annotations

from .. import components, image, paths, runtime
from ..cache import CACHE
from ..types import MlxPilImage, entry_for, model_types

NO_PATH = "<无可用权重>"


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
        kind, resolved = paths.resolve("local", model_path, "vae")
        if kind == "missing":
            raise FileNotFoundError(f"未找到 vae 权重: {resolved}")

        arr, hit = CACHE.get("component_weights", latents.cache_key)
        if not hit:
            raise RuntimeError("latent 数组不在缓存里，请重新运行 MlxKSamplerMLX")

        key = runtime.cache_key(
            {"kind": "vae", "path": model_path, "precision": precision, "quantize": int(quantize)}
        )

        def build():
            instance, _bits = components.create_and_load(entry, "vae", kind, resolved, int(quantize))
            return instance

        vae, _hit = CACHE.get_or_create("module", key, build)

        # 与 _decode_latents 一致：每个 batch 的 latent 形状 [16, 1, h/8, w/8]
        if batch_index >= 0:
            arr = arr[[batch_index % arr.shape[0]]]

        from .. import pipeline

        pils = []
        for i in range(arr.shape[0]):
            decoded = pipeline.decode_latents(entry, {"vae": vae}, arr[i], latents.height, latents.width)
            pils.extend(image.to_pil(decoded, batch_index=-1))
        return (MlxPilImage(images=tuple(pils), batch_index=batch_index),)
