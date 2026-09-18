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
import inspect
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

# PipeNetwork/minimax-h3-mlx 保留 MiniMax 原始 DiT 配置名；本地模块使用 diffusers 名。
_TRANSFORMER_ALIASES = {
    "token_refiner_num_layers": "num_refiner_layers",
    "ffn_hidden_size": "ffn_dim",
    "latents_dim": "in_channels",
    "audio_latents_dim": "audio_in_channels",
    "timestep_input_dim": "freq_dim",
    "time_embed_hidden_size": "time_embed_hidden_dim",
    "rope_inv_freq_len": "rope_freq_dim",
}


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
    names = tuple(
        name
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name != "self"
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
    kwargs = {name: config[name] for name in names if name in config}
    for key in _TUPLE_KEYS:
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(int(v) for v in kwargs[key])
    for key in _NESTED_TUPLE_KEYS:
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(tuple(int(v) for v in row) for row in kwargs[key])
    return kwargs


def transformer_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """把 diffusers 或 MiniMax 原始 transformer 配置转成本地构造参数并校验。

    原始配置还显式保存两个可由结构推导的输出宽度；本地模块不把它们暴露为构造参数，
    但仍校验，避免用一份结构不兼容的 config 静默构造默认模型。
    """
    from comfyui_mlx_gen.h3.model.h3_transformer.h3_transformer import MiniMaxH3Transformer

    converted = dict(config)
    for source, target in _TRANSFORMER_ALIASES.items():
        if source not in config:
            continue
        if target in config and config[target] != config[source]:
            raise ValueError(
                f"MiniMax-H3 transformer 配置冲突：{source}={config[source]!r}，"
                f"但 {target}={config[target]!r}"
            )
        converted[target] = config[source]

    kwargs = _ctor_kwargs(MiniMaxH3Transformer, converted)
    signature = inspect.signature(MiniMaxH3Transformer.__init__)
    values = {
        name: kwargs.get(name, parameter.default)
        for name, parameter in signature.parameters.items()
        if name != "self"
    }
    unresolved = [name for name, value in values.items() if value is inspect.Parameter.empty]
    if unresolved:
        raise ValueError(f"MiniMax-H3 transformer 配置缺少构造字段：{unresolved}")

    positive_ints = (
        "num_attention_heads",
        "attention_head_dim",
        "hidden_size",
        "num_layers",
        "num_refiner_layers",
        "ffn_dim",
        "in_channels",
        "audio_in_channels",
        "text_dim",
        "freq_dim",
        "time_embed_hidden_dim",
        "time_embed_dim",
        "rope_freq_dim",
    )
    invalid = [name for name in positive_ints if not isinstance(values[name], int) or values[name] <= 0]
    patch_size = values["patch_size"]
    if invalid:
        raise ValueError(f"MiniMax-H3 transformer 配置字段必须是正整数：{invalid}")
    if not isinstance(patch_size, tuple) or len(patch_size) != 3 or any(v <= 0 for v in patch_size):
        raise ValueError(f"MiniMax-H3 patch_size 必须是三个正整数，收到 {patch_size!r}")
    if 6 * int(values["rope_freq_dim"]) > int(values["attention_head_dim"]):
        raise ValueError(
            "MiniMax-H3 RoPE 宽度超过 attention head："
            f"6 * rope_freq_dim={6 * int(values['rope_freq_dim'])} > {values['attention_head_dim']}"
        )

    derived = {
        "adaln_out_features": 6 * 3 * int(values["hidden_size"]),
        "final_adaln_out_features": 2 * int(values["hidden_size"]),
    }
    for name, expected in derived.items():
        if name in config and int(config[name]) != expected:
            raise ValueError(
                f"MiniMax-H3 transformer 配置 {name}={config[name]!r}，"
                f"但按 hidden_size 推导应为 {expected}"
            )
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
    if role == "transformer":
        return transformer_kwargs(config)
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
