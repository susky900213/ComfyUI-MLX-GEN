"""Smoke: qwen_image chain still samples after the compile-once refactor.

    PYTHONPATH=src python tests/test_qwen_image_smoke.py  (or run directly)

跑 4 步 / 512×512，两次（第二次应命中编译缓存与 latent 缓存）。
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from comfyui_mlx_gen.nodes.clip_loader import MlxClipLoader  # noqa: E402
from comfyui_mlx_gen.nodes.loader import MlxTransformerLoader  # noqa: E402
from comfyui_mlx_gen.nodes.sampler import MlxKSamplerMLX  # noqa: E402
from comfyui_mlx_gen.nodes.text_encoder import MlxTextEncoder  # noqa: E402
from comfyui_mlx_gen.nodes.vae_decode import MlxVAEDecodeRawPIL  # noqa: E402


def unwrap(result):
    return result[0] if isinstance(result, tuple) and len(result) == 1 else result


def run(tag, seed):
    t0 = time.perf_counter()
    clip = unwrap(MlxClipLoader().load("qwen_image", "text_encoder", "qwen-image-2512-8bit", "bfloat16", 1058, 0))
    pos = unwrap(MlxTextEncoder().encode("a red cube on a wooden table", clip))
    neg = unwrap(MlxTextEncoder().encode("blurry", clip))
    model = unwrap(MlxTransformerLoader().load("qwen_image", "qwen-image-2512-8bit", 8, "bfloat16", True, 2))
    handle = unwrap(
        MlxKSamplerMLX().sample(
            model, pos, neg, seed=seed, steps=4, width=512, height=512,
            batch_size=1, guidance=4.0, scheduler="flow_match_euler_discrete",
        )
    )
    img = unwrap(MlxVAEDecodeRawPIL().decode(handle, "qwen_image", "qwen-image-2512-8bit", "bfloat16", 8, -1))
    arr = np.asarray(img.images[0], dtype=np.float32)
    print(f"  {tag}: {time.perf_counter() - t0:.2f}s mean={arr.mean():.1f}", flush=True)
    return arr


if __name__ == "__main__":
    first = run("qwen_image seed=0", 0)
    second = run("qwen_image seed=7", 7)
    diff = float(np.abs(first - second).max())
    print(f"  seed0 vs seed7 max|diff| = {diff:.2f}（应 > 0，说明噪声没被烘进计算图）")
