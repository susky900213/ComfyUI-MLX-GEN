"""Configuration and checkpoint identity checks for MOSS-Image-SFT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MOSSConfig:
    dim: int = 3072
    ffn_dim: int = 8192
    freq_dim: int = 256
    latent_channels: int = 128
    num_heads: int = 24
    num_layers: int = 42
    rope_base: float = 10_000.0
    text_dim: int = 4096

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MOSSConfig":
        allowed = {
            "dim", "ffn_dim", "freq_dim", "latent_channels", "num_heads",
            "num_layers", "rope_base", "text_dim",
        }
        kwargs = {key: value[key] for key in allowed if key in value}
        config = cls(**kwargs)
        validate_config(config)
        return config

    @classmethod
    def from_path(cls, path: str | Path) -> "MOSSConfig":
        file = Path(path)
        if file.is_dir():
            file = file / "config.json"
        if not file.is_file():
            raise FileNotFoundError(f"MOSS transformer 缺少 config.json: {file}")
        try:
            value = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"MOSS 配置不是合法 JSON: {file}: {exc}") from exc
        if not isinstance(value, dict) or value.get("_class_name") != "MOSSImageTransformer":
            raise ValueError(f"不是 MOSSImageTransformer 配置: {file}")
        return cls.from_dict(value)


def validate_config(config: MOSSConfig) -> MOSSConfig:
    expected = {
        "dim": 3072,
        "ffn_dim": 8192,
        "freq_dim": 256,
        "latent_channels": 128,
        "num_heads": 24,
        "num_layers": 42,
        "text_dim": 4096,
    }
    for name, value in expected.items():
        if int(getattr(config, name)) != value:
            raise ValueError(
                f"MOSS-SFT 只支持官方 dense 配置：{name}={getattr(config, name)!r}，"
                f"期望 {value}"
            )
    if float(config.rope_base) != 10_000.0:
        raise ValueError(f"MOSS rope_base 必须为 10000，收到 {config.rope_base!r}")
    if config.dim % config.num_heads or config.dim // config.num_heads % 8:
        raise ValueError("MOSS dim / num_heads 必须产生可用的 8-band RoPE head width")
    return config


def load_config(path: str | Path) -> MOSSConfig:
    return MOSSConfig.from_path(path)


def text_encoder_config(path: str | Path) -> dict[str, Any]:
    """读取并校验 MOSS 的 Qwen3.5-9B 文本配置。"""
    file = Path(path)
    if file.is_dir():
        file = file / "config.json"
    if not file.is_file():
        raise FileNotFoundError(f"MOSS text_encoder 缺少 config.json: {file}")
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"MOSS text encoder 配置不是合法 JSON: {file}: {exc}") from exc
    text = value.get("text_config") if isinstance(value, dict) else None
    if value.get("model_type") != "qwen3_5" or not isinstance(text, dict):
        raise ValueError("MOSS 需要 Qwen3.5 text encoder")
    for key, expected in (("hidden_size", 4096), ("num_hidden_layers", 32), ("vocab_size", 248320)):
        if int(text.get(key, -1)) != expected:
            raise ValueError(f"MOSS Qwen3.5 配置 {key}={text.get(key)!r}，期望 {expected}")
    return value