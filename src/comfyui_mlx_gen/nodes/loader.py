"""MlxTransformerLoader：只登记配置，返回 MlxModelHandle（不实例化组件、不加载权重）。

只选「模型大类」（family，如 z_image / flux2）与「权重目录」，**没有配置变体
可选**：用哪套 ModelConfig 由 weights.config_for_path 按选中的目录名去 mflux 的
ModelConfig 注册表里匹配（z-image-turbo-* 命中 z_image_turbo、z-image-8bit 命中
z_image、qwen-image-2512-* 命中纯文生图的 qwen_image，匹配不到则用该大类的
兜底配置 default_config）。因此新增一份权重（如 z-image-turbo-4bit）放进
transformer/ 目录后即可直接选用，不必改代码。

Qwen-Image 2512 文生图时，三个加载器的 model_type 都选 `qwen_image`、权重目录
都选 `qwen-image-2512-8bit`；这是没有视觉塔的纯文本链路，不要连接采样器的
`ref_images`。需要参考图编辑时改用 `qwen_edit` + `qwen-image-edit-2511-8bit`。
LoRA 请单独连接 MlxModelLoraApply，不要在本节点上配置。
"""

from __future__ import annotations

from .. import paths, runtime
from ..types import MlxModelHandle, entry_for, model_types

# 0 = 保留磁盘精度（Ideogram 4 官方 checkpoint 已是 FP8，不应再做一次在线量化）。
QUANTIZE_OPTIONS = [0, 4, 8, 16]
NO_WEIGHTS = "<无可用权重>"


def path_options(component: str) -> list[str]:
    """该组件目录下可选的权重集（扫盘；目录为空时给占位项，避免下拉为空）。"""
    return paths.list_component_items(component) or [NO_WEIGHTS]


class MlxTransformerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        types = model_types()
        default_type = types[0]
        transformer_paths = path_options("transformer")
        return {
            "required": {
                "model_type": (types, {"default": default_type}),
                "model_path": (transformer_paths, {"default": transformer_paths[0]}),
                "quantize": (QUANTIZE_OPTIONS, {"default": 8}),
                "precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "compile": ([True, False], {"default": True}),
                "compile_cache_limit": ([0, 1, 2, 3], {"default": 2}),
            }
        }

    RETURN_TYPES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MLX/Gen"

    def load(self, model_type, model_path, quantize, precision, compile, compile_cache_limit):
        # 未知大类 → 直接报错（不静默回退，避免加载到错的类）；
        # 已知但尚未验证的大类（MODEL_DEFS 里 supported=False）提示还没实现
        entry = entry_for(model_type)
        if not entry.supported:
            raise NotImplementedError(f"{model_type} 尚未实现：{entry.notes}")
        kind, resolved = paths.resolve("local", model_path, "transformer")
        if kind == "missing":
            raise FileNotFoundError(f"未找到 transformer 权重: {resolved}")
        # 视频 / 音频家族（MiniMax-H3）：transformer 入参含逐行 timestep 与 int 索引，
        # 不支持 mx.compile；未量化约 108 GB 常驻，必须量化到 4 / 8 位
        if entry.media != "image":
            if int(quantize) not in (4, 8):
                raise ValueError(
                    f"{entry.family} 的 transformer 必须量化到 4 或 8 位（收到 {quantize}）；"
                    "不量化会超出 128 GB 机器的常驻预算"
                )
            if compile:
                print(f"[MlxTransformerLoader] {entry.family} 不支持 mx.compile，已自动关掉编译")
            compile, compile_cache_limit = False, 0
        config = {
            "model_type": model_type,
            "path": model_path,
            "kind": kind,
            "quantize": int(quantize),
            "precision": precision,
            "compile": bool(compile),
            "compile_cache_limit": int(compile_cache_limit),
        }
        handle = MlxModelHandle(
            model_type=model_type,
            model_path=model_path,
            quantize=int(quantize),
            precision=precision,
            compile=bool(compile),
            compile_cache_limit=int(compile_cache_limit),
            loras=(),
            cache_key=runtime.cache_key(config),
        )
        return (handle,)

