#!/usr/bin/env python3
"""Real-checkpoint Qwen-Image 2.1 MLX smoke test.

Loads components sequentially (q8 text/vision tower -> q8 DiT -> full RGBA VAE),
runs a 32x32 one-step generation and writes an RGBA PNG.  The tiny resolution
validates execution only; use the workflow's normal 1024/2048 canvas for quality.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen import image as image_util, runtime  # noqa: E402
from comfyui_mlx_gen.qwen_image_21 import loader, sampling  # noqa: E402
from comfyui_mlx_gen.qwen_image_21.text_encoder import encode_prompt  # noqa: E402
from comfyui_mlx_gen.qwen_image_21.transformer import QwenImage21KVCache  # noqa: E402


def release(*values) -> None:
    del values
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def report(label: str, started: float) -> None:
    print(
        f"[{label}] {time.perf_counter() - started:.2f}s | {runtime.memory_snapshot()}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/qwen-image-2.1-mlx-smoke.png"))
    parser.add_argument("--prompt", default="a red cube")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional reference image; enables the native Qwen-Image 2.1 edit path.",
    )
    parser.add_argument("--quantize", type=int, choices=(4, 8), default=8)
    args = parser.parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    runtime.apply_cache_limit(2, label="qwen21-real-smoke")

    reference_images = []
    reference_latents = None
    reference_shapes = ()
    if args.reference is not None:
        source = Image.open(args.reference.expanduser().resolve()).convert("RGBA")
        # 256x256 is the Qwen3-VL processor's minimum area and remains cheap enough for a smoke test.
        source = source.resize((256, 256), Image.Resampling.LANCZOS)
        pixels = np.asarray(source, dtype=np.float32) / 127.5 - 1.0
        started = time.perf_counter()
        edit_vae = loader.load("vae", str(snapshot / "vae"), precision="bfloat16")
        encoded = edit_vae.module.encode(mx.array(pixels)[None])
        latent_h, latent_w = int(encoded.shape[2]), int(encoded.shape[3])
        reference_latents = encoded.transpose(0, 2, 3, 1).reshape(1, -1, 64)
        reference_shapes = ((latent_h, latent_w),)
        mx.eval(reference_latents)
        report("reference VAE encode", started)
        white = Image.new("RGB", source.size, (255, 255, 255))
        white.paste(source, mask=source.getchannel("A"))
        reference_images = [white]
        del edit_vae, encoded
        release()

    started = time.perf_counter()
    tokenizer = loader.load_tokenizer(snapshot / "processor")
    text = loader.load(
        "text_encoder",
        str(snapshot / "text_encoder"),
        quantize=args.quantize,
        precision="bfloat16",
    )
    condition = encode_prompt(
        text.module,
        tokenizer,
        args.prompt,
        max_length=512,
        images=reference_images,
    )
    mx.eval(*condition)
    report("text", started)
    print(f"condition={tuple(condition[0].shape)} {condition[0].dtype}", flush=True)
    del text, tokenizer
    release()
    print(f"[text released] {runtime.memory_snapshot()}", flush=True)

    started = time.perf_counter()
    dit = loader.load(
        "transformer",
        str(snapshot / "transformer"),
        quantize=args.quantize,
        precision="bfloat16",
    )
    report("dit load", started)
    latent = sampling.create_noise(7, 32, 32)
    sigmas = sampling.sigma_schedule(1, 4)
    prefix_cache = QwenImage21KVCache(dit.module.num_layers)
    started = time.perf_counter()
    noise = dit.module(
        latent,
        condition[0],
        mx.array([float(sigmas[0])], dtype=mx.float32),
        2,
        2,
        prefix_cache,
        "extract",
        reference_latents=reference_latents,
        reference_shapes=reference_shapes,
        image_pad_mask=condition[2],
    )
    latent = sampling.euler_step(noise, latent, sigmas[0], sigmas[1])
    mx.eval(latent)
    report("dit forward", started)
    print(f"latent={tuple(latent.shape)} {latent.dtype}", flush=True)
    del dit, prefix_cache, noise, condition
    release()
    print(f"[dit released] {runtime.memory_snapshot()}", flush=True)

    started = time.perf_counter()
    vae = loader.load("vae", str(snapshot / "vae"), precision="bfloat16")
    report("vae load", started)
    started = time.perf_counter()
    rgba = vae.module.decode(sampling.unpack_latents(latent, 32, 32))
    mx.eval(rgba)
    report("vae decode", started)
    print(
        f"rgba={tuple(rgba.shape)} {rgba.dtype} "
        f"range=[{float(mx.min(rgba).item()):.4f}, {float(mx.max(rgba).item()):.4f}]",
        flush=True,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pil = image_util.to_pil(rgba)[0]
    pil.save(output)
    print(f"saved={output} mode={pil.mode} size={pil.size}", flush=True)


if __name__ == "__main__":
    main()