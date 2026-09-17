"""MlxVAELoader：只登记 VAE 配置，返回 MlxVaeHandle（不实例化组件、不加载权重）。

与 MlxTransformerLoader / MlxClipLoader 同语义：真正的 VAE 实例由消费它的节点
（MlxVAEEncoder / MlxVAEDecoder，以及兼容的 MlxVAEDecodeRawPIL）按
handle.cache_key 物化并缓存 —— 因此「同一个 VAE handle 同时接编码器与解码器」
时只会存在一份实例（编码用的 BN 统计量与解码用的权重必然同源）。

只选「模型大类」（决定用哪个 VAE 类与权重定义）+「vae/ 下的权重目录」，
**没有配置变体可选**：精度 / 量化档位与 MlxVAEDecodeRawPIL 的默认值一致
（bfloat16 / 8），因此新旧解码节点混用时也能命中同一份缓存。
"""

from __future__ import annotations

from .. import paths, runtime
from ..types import MlxVaeHandle, entry_for, model_types, vae

NO_WEIGHTS = "<无可用权重>"
PRECISIONS = ["bfloat16", "float16", "float32"]
QUANTIZE_OPTIONS = [0, 4, 8, 16]
# 图片链路只有 "vae"；MiniMax-H3 还有 "audio_vae"（权重通常也放在 vae/ 下，
# 因此 MlxVAELoader 的候选取两个目录的并集，解析时 audio_vae 缺失会退回 vae/）
ROLES = ["vae", "audio_vae"]


class MlxVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        types = model_types()
        # 两个候选目录取并集：H3 的音频 VAE 既可以放 audio_vae/，也可以放 vae/
        vae_paths = list(
            dict.fromkeys(
                paths.list_component_items("vae") + paths.list_component_items("audio_vae")
            )
        ) or [NO_WEIGHTS]
        return {
            "required": {
                "model_type": (types, {"default": types[0]}),
                "model_path": (vae_paths, {"default": vae_paths[0]}),
                "precision": (PRECISIONS, {"default": "bfloat16"}),
                "quantize": (QUANTIZE_OPTIONS, {"default": 8}),
                "role": (ROLES, {"default": "vae"}),
            }
        }

    RETURN_TYPES = (vae,)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "MLX/Gen"

    def load(self, model_type, model_path, precision, quantize, role="vae"):
        # 未知大类 → 直接报错；已知但未验证的大类提示还没实现（与 MlxTransformerLoader 一致）
        entry = entry_for(model_type)
        if not entry.supported:
            raise NotImplementedError(f"{model_type} 尚未实现：{entry.notes}")
        if precision not in PRECISIONS:
            raise ValueError(f"未知精度: {precision}")
        if role not in entry.components:
            raise ValueError(f"{model_type} 没有 {role} 组件（只有 {list(entry.components)}）")
        kind, resolved = paths.resolve("local", model_path, role)
        if kind == "missing":
            raise FileNotFoundError(f"未找到 {role} 权重: {resolved}")
        config = {
            "kind": "vae",
            "model_type": model_type,
            "path": model_path,
            "precision": precision,
            "quantize": int(quantize),
            "role": role,
        }
        handle = MlxVaeHandle(
            model_type=model_type,
            path=model_path,
            precision=precision,
            quantize=int(quantize),
            role=role,
            cache_key=runtime.cache_key(config),
        )
        return (handle,)
