"""Configuration helpers for the official FLUX 3 Diffusers layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_config(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    file = root if root.name == "config.json" else root / "config.json"
    if not file.is_file():
        return {}
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"FLUX 3 配置不是合法 JSON: {file}: {exc}") from exc


def transformer_kwargs(path: str | Path, selection: str = "") -> dict[str, Any]:
    config = read_config(path)
    if not config:
        name = str(selection).lower()
        config = {"flow_map": not ("base" in name or "undistilled" in name)}
    config["flow_map"] = bool(config.get("flow_map", False))
    config["axes_dims_rope"] = tuple(config.get("axes_dims_rope", (32, 32, 32, 32)))
    return {
        key: config[key]
        for key in (
            "in_channels", "cond_in_channels", "num_layers", "num_single_layers",
            "attention_head_dim", "num_attention_heads", "joint_attention_dim",
            "timestep_channels", "mlp_ratio", "axes_dims_rope", "rope_theta", "eps", "flow_map",
        )
        if key in config
    }


def text_encoder_kwargs(path: str | Path) -> dict[str, Any]:
    config = read_config(path)
    return {
        "text_config": dict(config.get("text_config") or {}),
        "vision_config": dict(config.get("vision_config") or {}),
    }