"""MiniMax-H3 各组件的构造参数：从组件目录的 `config.json` 读（不依赖 mflux 注册表）。

参考实现里这些参数由 `ModelConfig.minimax_h3()` 的 `transformer_overrides` 与三个
`*_kwargs()` 辅助函数拼出来；本插件直接读磁盘上的 `config.json`（与发布的权重一一
对应），因此换一份权重目录不需要改代码，也不会因为 mflux 版本里没有 `minimax-h3`
这一项而失败。

规则与参考实现一致：

- 参数名取自各组件的 `__init__` 形参，`config.json` 里同名的键直接透传（缺失的用
  类默认值），`patch_size` / `*_rates` / `block_out_channels` 这类列表要转成 tuple；
- 条件编码器只保留 Qwen3-VL 的**前 50 层**（`TEXT_ENCODER_LAYERS`，H3 只读
  `hidden_states[50]`），视觉塔按 `vision_config` 原样构造（本版的纯文生视频不用它，
  照搬是为了以后做图生视频）；
- 视频 / 音频 VAE 的 `latents_mean` / `latents_std` 由类自己转成 `mx.array`。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from comfyui_mlx_gen.h3.weights.h3_weight_mapping import TEXT_ENCODER_NUM_LAYERS

# config.json 里是 JSON 数组、而构造参数要 tuple 的键
_TUPLE_KEYS = (
    "patch_size",
    "block_out_channels",
    "spatial_downsample_factors",
    "temporal_downsample_factors",
    "encoder_rates",
    "decoder_rates",
    "decoder_kernel_sizes",
    "resblock_kernel_sizes",
    "deepstack_visual_indexes",
    "mrope_section",
)
# 需要 tuple[tuple[int, ...], ...] 的键
_NESTED_TUPLE_KEYS = ("resblock_dilation_sizes",)


def read_config(path: str | Path) -> dict[str, Any]:
    """读一个组件目录（或某个 .json 文件）的配置；缺失 / 损坏时给中文错误。"""
    root = Path(path)
    file = root if root.suffix == ".json" else root / "config.json"
    if not file.is_file():
        raise FileNotFoundError(
            f"缺少组件配置 {file}：MiniMax-H3 的构造参数只从 config.json 读"
            "（权重目录里应带一份 diffusers 的 config.json）"
        )
    try:
        return json.loads(file.read_text())
    except json.JSONDecodeError as exc:  # noqa: PERF203
        raise ValueError(f"组件配置 {file} 不是合法 JSON: {exc}") from exc


def _ctor_kwargs(cls: Any, config: dict[str, Any]) -> dict[str, Any]:
    """`config` 里与 `cls.__init__` 形参同名的键（顺带做列表 → tuple 的转换）。"""
    code = cls.__init__.__code__
    names = code.co_varnames[1 : code.co_argcount]
    kwargs = {name: config[name] for name in names if name in config}
    for key in _TUPLE_KEYS:
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(int(v) for v in kwargs[key])
    for key in _NESTED_TUPLE_KEYS:
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(tuple(int(v) for v in row) for row in kwargs[key])
    return kwargs


def _text_encoder_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """条件编码器：语言模型（截断到 50 层）+ 视觉塔两套参数，交给 Qwen3VLModel 组装。"""
    from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_text_model import Qwen3VLTextModel
    from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_vision_model import Qwen3VLVisionModel

    text_config = dict(config.get("text_config") or {})
    vision_config = dict(config.get("vision_config") or {})
    language = _ctor_kwargs(Qwen3VLTextModel, text_config)
    # M-RoPE 的 section 在 Qwen3-VL 里藏在 rope_scaling 下
    mrope = (text_config.get("rope_scaling") or {}).get("mrope_section")
    if mrope:
        language["mrope_section"] = tuple(int(v) for v in mrope)
    # H3 只读 hidden_states[50]（第 49 层之后的 pre-norm 输出），多的层不建也不加载
    language["num_hidden_layers"] = min(
        int(language.get("num_hidden_layers", TEXT_ENCODER_NUM_LAYERS)), TEXT_ENCODER_NUM_LAYERS
    )
    return {
        "language_model": language,
        "visual": _ctor_kwargs(Qwen3VLVisionModel, vision_config),
    }


def class_kwargs(role: str, path: str | Path) -> dict[str, Any]:
    """按 role 给出构造参数字典（条件编码器返回 `{language_model, visual}` 两套）。

    条件编码器允许目录里没有 `config.json`：参考实现也是「有就用、没有就用品类默认值」
    （Qwen3-VL 的类默认值就是发布配置）。两个 VAE 不行 —— 没有 `config.json` 就不知道
    latent 通道数与 `latents_mean/std`，那种情况仍由 `read_config` 直接报错。
    """
    root = Path(path)
    if role == "text_encoder" and not (root if root.suffix == ".json" else root / "config.json").is_file():
        print(f"[H3 配置] {role}: 目录里没有 config.json，用品类默认值（{path}）")
        return _text_encoder_kwargs({})
    config = read_config(path)
    if role == "text_encoder":
        return _text_encoder_kwargs(config)
    return _ctor_kwargs(_ctor_of(role), config)


def _ctor_of(role: str) -> Any:
    """按 role 取「用哪个类」（与 types.MINIMAX_H3.components 的 class_import 同源）。"""
    from comfyui_mlx_gen import runtime, types

    entry = types.entry_for("minimax_h3")
    return runtime.import_object(entry.components[role].class_import)


def build_model(role: str, path: str | Path) -> Any:
    """实例化一个组件（权重还没写进去；构造参数来自组件目录的 config.json）。"""
    cls = _ctor_of(role)
    kwargs = class_kwargs(role, path)
    if role == "text_encoder":
        # Qwen3VLModel(language_model=..., visual=...)：两个子模型各自按 config 构造
        from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_text_model import Qwen3VLTextModel
        from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_vision_model import Qwen3VLVisionModel

        return cls(
            language_model=Qwen3VLTextModel(**kwargs["language_model"]),
            visual=Qwen3VLVisionModel(**kwargs["visual"]),
        )
    return cls(**kwargs)
