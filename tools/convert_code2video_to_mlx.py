#!/usr/bin/env python3
"""Convert the SenseNova Code2Video Transformers checkpoint to MLX-VLM.

This is a thin wrapper around the official ``mlx_vlm.convert`` implementation.
It intentionally performs no weight renaming itself, so conversion and runtime
stay aligned with the Qwen3.5 implementation shipped by mlx-vlm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 SenseNova Code2Video Transformers 权重转换为 MLX-VLM 格式"
    )
    parser.add_argument("--hf-path", required=True, help="原始 Transformers checkpoint 目录")
    parser.add_argument("--mlx-path", required=True, help="输出 MLX-VLM checkpoint 目录")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="转换后浮点权重类型，默认 bfloat16",
    )
    parser.add_argument("--quantize", action="store_true", help="转换后进行 MLX 线性层量化")
    parser.add_argument("--q-bits", type=int, choices=(2, 3, 4, 6, 8), default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument(
        "--quant-predicate",
        choices=("mixed_2_6", "mixed_3_4", "mixed_3_5", "mixed_3_6", "mixed_3_8", "mixed_4_6", "mixed_4_8"),
        default=None,
        help="可选的混合 bit recipe；不指定时所有可量化文本线性层使用 q-bits",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    hf_path = Path(args.hf_path).expanduser().resolve()
    mlx_path = Path(args.mlx_path).expanduser().resolve()
    config_path = hf_path / "config.json"
    if not hf_path.is_dir():
        raise SystemExit(f"原始模型目录不存在：{hf_path}")
    if not config_path.is_file():
        raise SystemExit(f"原始模型目录缺少 config.json：{hf_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"无法读取 config.json：{exc}") from exc
    if config.get("model_type") != "qwen3_5" and "Qwen3_5ForConditionalGeneration" not in config.get(
        "architectures", []
    ):
        raise SystemExit(
            "输入目录不是已验证的 Qwen3.5/SenseNova Code2Video checkpoint："
            f"model_type={config.get('model_type')!r}, architectures={config.get('architectures')!r}"
        )
    if mlx_path == hf_path:
        raise SystemExit("--mlx-path 必须是独立于原始 Transformers 目录的输出目录")
    if mlx_path.exists() and any(mlx_path.iterdir()):
        raise SystemExit(f"输出目录不是空目录，为避免覆盖请先清理：{mlx_path}")

    try:
        from mlx_vlm.convert import convert
    except ImportError as exc:
        raise SystemExit(
            "当前 Python 环境没有 mlx-vlm；请先执行 pip install -r requirements.txt"
        ) from exc

    kwargs = {
        "hf_path": str(hf_path),
        "mlx_path": str(mlx_path),
        "dtype": args.dtype,
        "quantize": bool(args.quantize),
        "q_bits": int(args.q_bits),
        "q_group_size": int(args.q_group_size),
    }
    if args.quant_predicate:
        kwargs["quant_predicate"] = args.quant_predicate
    print(f"[Code2Video] 开始 MLX 转换：{hf_path} -> {mlx_path}", flush=True)
    convert(**kwargs)
    print(f"[Code2Video] MLX 转换完成：{mlx_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())