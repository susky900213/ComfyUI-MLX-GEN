#!/usr/bin/env python3
"""最小脚本：不启动 ComfyUI，照 workflows/z-image-turbo-512.json 的连线跑一遍出图。

本项目的节点只传「纯数据 handle」（目录名 + 精度 + 量化档位），真正的模块由消费它
的节点按需加载，因此脱离 ComfyUI 也能跑：按工作流里节点的 ``order`` 依次调用该节点
的 ``FUNCTION``，把上一个的输出喂给下一个就行。

工作流连线（编号即 z-image-turbo-512.json 里的 order，括号里是节点 id）::

    order 0  ① MlxClipLoader(1)         → CLIP 句柄（只登记配置，不加载权重）
    order 1  ② MlxTransformerLoader(2)  → model 句柄（同样只登记配置）
    order 2  ③ MlxTextEncoder(9) 正条件  → condition（此处才建 tokenizer + 编码器）
    order 3  ④ MlxTextEncoder(10) 负条件 → condition
    order 4  ⑤ MlxKSamplerMLX(3)         → latents（此处才加载 transformer 并采样）
    order 5  ⑥ MlxVAEDecodeRawPIL(4)     → images（PIL；此处才加载 VAE）
    order 6  ⑦ MlxSaveImage(5)           → 落盘
             ⑧⑨ MlxPilToTorch(6) → SaveImage(7) / PreviewImage(8) 是 ComfyUI 核心
             节点（要服务端），脚本里由第 ⑦ 步直接落盘代替

运行（要用装了 mflux / mlx 的那个环境，本机即 ComfyUI 的 venv）::

    python tools/gen_z_image_turbo.py --prompt "a red cube" --width 1024 --height 1024 --steps 8

``--cache-limit-gb`` 对应 Loader 的 ``compile_cache_limit``（默认 8，与 mlx-gen
按「内存 / 8、钳 1~8 GiB」推导出的默认一致；0 = 不设置，沿用 MLX 默认），
由 ``MlxKSamplerMLX`` 在加载权重前调 ``mx.set_cache_limit`` 设上。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:  # 与根目录 __init__.py 同样的做法：让包可导入
    sys.path.insert(0, str(SRC))

from comfyui_mlx_gen import paths, runtime  # noqa: E402
from comfyui_mlx_gen.nodes.clip_loader import MlxClipLoader  # noqa: E402
from comfyui_mlx_gen.nodes.loader import MlxTransformerLoader  # noqa: E402
from comfyui_mlx_gen.nodes.sampler import (  # noqa: E402
    HEIGHT_OPTIONS,
    WIDTH_OPTIONS,
    MlxKSamplerMLX,
)
from comfyui_mlx_gen.nodes.save import auto_filename  # noqa: E402
from comfyui_mlx_gen.nodes.text_encoder import MlxTextEncoder  # noqa: E402
from comfyui_mlx_gen.nodes.vae_decode import MlxVAEDecodeRawPIL  # noqa: E402

# 与 z-image-turbo-512.json 的 widget 值一一对应（工作流里存的就是这些值）
MODEL_TYPE = "z_image"
MODEL_PATH = "z-image-turbo-8bit"
PRECISION = "bfloat16"
QUANTIZE = 8
MAX_LENGTH = 512
COMPILE = True
# MLX 的 free-cache 上限（GB）：0 = 不设置（沿用 MLX 默认），由 MlxKSamplerMLX
# 在物化权重前调 mx.set_cache_limit。默认 8 与 mlx-gen 对齐：它按「机器内存 / 8」
# 推导并钳在 1~8 GiB（128 GB 机器即 8），单位按十进制 GB 计，与 mflux 的
# --mlx-cache-limit-gb 一致。
DEFAULT_CACHE_LIMIT_GB = 8
DEFAULT_STEPS = 4
DEFAULT_SEED = 0
DEFAULT_GUIDANCE = 1.0
DEFAULT_SCHEDULER = "linear"


def out(result):
    """节点函数一律返回元组（ComfyUI 的多输出约定），取第一个输出即可。"""
    return result[0]


def stage(label: str, fn):
    """打印每段耗时与内存水位（加载权重 / 编码 / 采样 / 解码都在各自节点里按需发生）。

    内存水位取自 ``runtime.memory_snapshot()``（活跃 / cache 占用 / 峰值），
    用来核对 cache 上限是否真的压住了 cache 占用。
    """
    started = time.perf_counter()
    value = fn()
    print(
        f"[script] {label}: 用时 {time.perf_counter() - started:.1f}s，"
        f"内存 {runtime.memory_snapshot()}",
        flush=True,
    )
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="最小 Z-Image-Turbo 出图脚本（模拟 ComfyUI 的 z-image-turbo-512 工作流）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", default="a red cube", help="正向提示词")
    parser.add_argument("--negative", default="", help="负向提示词（Turbo 不做 CFG，留空即可）")
    parser.add_argument("--width", type=int, default=512, choices=WIDTH_OPTIONS, help="画布宽")
    parser.add_argument("--height", type=int, default=512, choices=HEIGHT_OPTIONS, help="画布高")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="采样步数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE, help="CFG 强度")
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER, help="调度器（Z-Image 用 linear）")
    parser.add_argument("--model-type", default=MODEL_TYPE, help="模型大类（MODEL_DEFS 的键）")
    parser.add_argument("--model-path", default=MODEL_PATH, help="权重集目录名（各组件目录下同名）")
    parser.add_argument("--quantize", type=int, default=QUANTIZE, help="量化档位（0 = 用权重自带精度）")
    parser.add_argument("--precision", default=PRECISION, help="权重精度")
    parser.add_argument("--no-compile", action="store_true", help="关掉 mx.compile（更省内存但更慢）")
    parser.add_argument(
        "--cache-limit-gb",
        type=float,
        default=DEFAULT_CACHE_LIMIT_GB,
        help="MLX free-cache 上限（GB，0 = 不设置，沿用 MLX 默认）；采样器在加载权重前设上",
    )
    parser.add_argument("--out", default="output/z_image_turbo.png",
                       help="输出路径前缀（相对路径按仓库根目录解析；已存在则自动加 _1、_2 …）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        f"[script] MLX {runtime.check_mlx()}，模型根目录 {paths.MODEL_ROOT}",
        flush=True,
    )

    # ① MlxClipLoader：只登记「大类 + 组件 + 权重目录 + 精度」，不 import mflux、不建 tokenizer
    clip = stage(
        "① MlxClipLoader",
        lambda: out(
            MlxClipLoader().load(
                args.model_type, "text_encoder", args.model_path, args.precision, MAX_LENGTH, 0
            )
        ),
    )

    # ② MlxTransformerLoader：同样只登记配置，返回 model 句柄
    model = stage(
        "② MlxTransformerLoader",
        lambda: out(
            MlxTransformerLoader().load(
                args.model_type,
                args.model_path,
                args.quantize,
                args.precision,
                COMPILE and not args.no_compile,
                args.cache_limit_gb,
            )
        ),
    )

    # ③④ MlxTextEncoder：正、负条件各一条（对应 ComfyUI 的两个 CLIPTextEncode）
    encoder = MlxTextEncoder()
    positive = stage("③ MlxTextEncoder 正条件", lambda: out(encoder.encode(args.prompt, clip)))
    negative = stage("④ MlxTextEncoder 负条件", lambda: out(encoder.encode(args.negative, clip)))

    # ⑤ MlxKSamplerMLX：这里才真的加载 transformer 并跑采样循环
    latents = stage(
        "⑤ MlxKSamplerMLX",
        lambda: out(
            MlxKSamplerMLX().sample(
                model=model,
                positive=positive,
                negative=negative,
                seed=args.seed,
                steps=args.steps,
                width=args.width,
                height=args.height,
                batch_size=1,
                guidance=args.guidance,
                scheduler=args.scheduler,
            )
        ),
    )

    # ⑥ MlxVAEDecodeRawPIL：按 widget 里的权重目录物化 VAE，再解码成 PIL
    #    （工作流里用的就是这个自带 widget 的老节点；等价的新写法是
    #     MlxVAELoader().load(...) → MlxVAEDecoder().decode(vae, latents, -1)）
    images = stage(
        "⑥ MlxVAEDecodeRawPIL",
        lambda: out(
            MlxVAEDecodeRawPIL().decode(
                latents, args.model_type, args.model_path, args.precision, args.quantize, -1
            )
        ),
    )

    # ⑦ MlxSaveImage 的落盘语义（重名自动加 _1、_2 … 并写同名 metadata JSON），
    #    只是把文件写到 --out 指定的确切路径（相对路径按仓库根目录解析）
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    path, metadata = auto_filename(
        out_path,
        images.images,
        {
            "source": "tools/gen_z_image_turbo.py",
            "prompt": args.prompt,
            "negative": args.negative,
            "model_type": args.model_type,
            "model_path": args.model_path,
            "steps": args.steps,
            "guidance": args.guidance,
            "scheduler": args.scheduler,
        },
        images.images[0].size,
        args.seed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    images.images[0].save(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"[script] ⑦ 已保存: {path}（元数据 {path.with_suffix('.json')}）")


if __name__ == "__main__":
    main()

