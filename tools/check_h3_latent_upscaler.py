#!/usr/bin/env python3
"""H3 latent 放大网络的诊断脚本（只读；不改仓库状态）。

    python tools/check_h3_latent_upscaler.py [放大模型文件名]

全部在本机真实权重上跑（**不经过 27B transformer**）：

1. **可用性**：把同一段内容分别编码成低分 640×352 与高分 1280×704 的 latent（H3 视频 VAE），
   再把低分 latent 交给放大网络，与"高分真值"比**幅度比**与**逐通道相关**；同时给出
   「朴素 2× latent 插值」这条基线做对照。
   判据：幅度比应 ≈1.0、逐通道相关应高于基线；实测官方权重是幅度 ~4–6 倍、相关 ~0.1（不可用）。
2. **映射关系**：相关很低时，继续排查是不是"只差一个变换"——通道置换 / 空间转置翻转 /
   时间反转 / 逐通道仿射 / 空间尺度 / 频域特征，逐项给数字。

背景与完整结论见 docs/MINIMAX_H3_VIDEO_IMPLEMENTATION.md 附录 F。
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from comfyui_mlx_gen import pipeline  # noqa: E402
from comfyui_mlx_gen.cache import Cache  # noqa: E402
from comfyui_mlx_gen.h3 import pipeline as h3p  # noqa: E402
from comfyui_mlx_gen.h3.model.h3_latent_upscaler import latent_resizer_3d as lr  # noqa: E402
from comfyui_mlx_gen.nodes import h3_two_stage as node  # noqa: E402
from comfyui_mlx_gen.types import MlxVaeHandle  # noqa: E402

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FRAMES = 22  # 17n+5 → latent 7 帧


def load_test_image() -> Image.Image:
    """优先用参考项目自带的示例图（真实内容 + 真实细节），找不到就合成一张。

    只查几个已知位置（不做全盘递归扫描 —— 外置卷上会卡住）。
    """
    candidates: list[str] = []
    for root in ("/Volumes/extend/models/hub", str(Path.home() / ".cache/huggingface/hub")):
        candidates += glob.glob(
            f"{root}/models--LBH-123-AI--Minimax_h3_latent_Upscaler/snapshots/*/examples/*.jpg"
        )
    if candidates:
        print(f"测试图：参考项目示例图 {candidates[0]}")
        return Image.open(candidates[0]).convert("RGB")
    print("测试图：没找到参考示例图，用合成图案（渐变 + 方块）")
    w, h = 1280, 704
    xs = np.linspace(0.1, 0.9, w, dtype=np.float32)[None, :]
    ys = np.linspace(0.1, 0.9, h, dtype=np.float32)[:, None]
    arr = np.zeros((h, w, 3), np.float32)
    arr[..., 0] = xs
    arr[..., 1] = ys
    arr[..., 2] = 0.35
    arr[h // 2 - 60 : h // 2 + 60, w // 2 - 120 : w // 2 + 120] = 1.0
    return Image.fromarray((arr * 255).astype(np.uint8))


def clip(im: Image.Image):
    f = np.repeat(np.asarray(im, np.float32)[None] / 255.0, FRAMES, axis=0)
    return mx.array(((f - IM_MEAN) / IM_STD).transpose(3, 0, 1, 2)[None])


def chan_corr(a: np.ndarray, b: np.ndarray) -> float:
    """逐通道相关系数均值。"""
    a = a[0] if a.ndim == 5 else a
    b = b[0] if b.ndim == 5 else b
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return float(np.mean([np.corrcoef(a[c], b[c])[0, 1] for c in range(a.shape[0])]))


def load_comfy_latent(path: str):
    """读 ComfyUI `SaveLatent` 导出的 `.latent`（torch 保存的 dict）或 .safetensors。

    支持三种形态：`{"samples": tensor}`、裸 tensor、或 safetensors 里的单个张量。
    若是 NestedTensor（AV 合体），取 `unbind()` 的第一条（视频）。
    """
    import torch

    if path.endswith(".safetensors"):
        state = dict(mx.load(path))
        arr = next(iter(state.values()))
        t = np.array(arr.astype(mx.float32) if arr.dtype != mx.float32 else arr)
    else:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = obj.get("samples", obj)
        if hasattr(obj, "unbind") and type(obj).__name__ == "NestedTensor":
            obj = obj.unbind()[0]
        t = obj.detach().float().numpy() if hasattr(obj, "detach") else np.asarray(obj, np.float32)
    while t.ndim > 5:
        t = t[0]
    if t.ndim != 5:
        raise ValueError(f"latent 应该是 5D (B,C,T,H,W)，收到 {t.shape}")
    return t


def _same_channel_corr(a: np.ndarray, b: np.ndarray) -> float:
    """同一通道、把输入空间上采样到输出尺寸后比相关（判断"内容有没有保留"）。"""
    k = b.shape[3] // a.shape[3]
    up = np.repeat(np.repeat(a, k, axis=3), k, axis=4).reshape(a.shape[1], -1)
    bb = b.reshape(b.shape[1], -1)
    return float(np.mean([np.corrcoef(up[c], bb[c])[0, 1] for c in range(a.shape[1])]))


def main() -> int:
    names = [n for n in node.upscaler_items() if n != node.BUILTIN_UPSCALE]
    wanted = sys.argv[1] if len(sys.argv) > 1 else (names[0] if names else "")
    if not wanted or wanted == node.NO_UPSCALER:
        print("没有放大模型可测：请把 *_upscaler_3d_*.safetensors 放进 models/mlx/upscaler/")
        return 2
    print(f"被测模型：{wanted}")

    # 可选：直接拿 ComfyUI 导出的 video latent 测
    if len(sys.argv) > 2:
        raw = load_comfy_latent(sys.argv[2])
        flat = raw.reshape(raw.shape[1], -1)
        print(f"外部 latent：{sys.argv[2]} 形状 {tuple(raw.shape)}")
        print("  逐通道 mean（前 6）:", np.round(flat.mean(1)[:6], 3))
        print("  逐通道 std （前 6）:", np.round(flat.std(1)[:6], 3))
        print(
            "  说明：单条片段的通道统计量本身波动很大（归一化空间里也可能出现 std 0.7~2.7），"
            "所以**不能**只靠统计量判空间；下面的「幅度比」才是判据。"
        )
        b, c, t, h, w = raw.shape
        if c != 24:
            print(f"  通道数 {c} ≠ 24：这不是「分离后的视频 latent」，请把 SeparateAVLatent 的 video 输出存下来")
            return 2
        module = lr.load(node.resolve_upscaler_path(wanted), "bfloat16", Cache())
        for tag, arr in (("按原样喂（假定是原始空间）", raw), ):
            out = lr.upscale_latents(module, mx.array(arr.astype(np.float32)), (h * 2, w * 2), "bfloat16")
            mx.eval(out)
            o = np.array(out, np.float64)
            print(f"\n  【{tag}】输入 std {arr.std():.3f} → 输出 std {o.std():.3f}"
                  f"（倍率 {o.std() / arr.std():.2f}；本机对合法输入实测是 4~6 倍）")
            print(f"     逐通道相关（同一通道，输入上采样后 vs 输出）= {_same_channel_corr(arr, o):+.3f}")
        print(
            "\n  判读：倍率 ≈1 且同通道相关高 → 这个 latent 就是网络期待的空间（说明差异在输入空间，"
            "请把它连同 1280×704 的高分 latent 一起发我）；倍率 4~6 倍 → 与我们这边一致（权重/前向问题）。"
        )
        return 0

    # 高分真值 = 示例图裁到 1280×704；低分 = 同一张图缩到 640×352（细节真的丢了）
    img = load_test_image()
    scale = max(1280 / img.width, 704 / img.height)
    mid = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    left, top = (mid.width - 1280) // 2, (mid.height - 704) // 2
    hi_img = mid.crop((left, top, left + 1280, top + 704))
    lo_img = hi_img.resize((640, 352), Image.LANCZOS)

    cache = Cache()
    vae = pipeline.prepare_h3_vae(
        MlxVaeHandle(
            model_type="minimax_h3", path="MiniMax-H3", precision="bfloat16", quantize=8,
            role="vae", cache_key="vae",
        ),
        cache,
    )
    lo_z, _ = vae.encode(clip(lo_img).astype(h3p.component_dtype(vae)))
    hi_z, _ = vae.encode(clip(hi_img).astype(h3p.component_dtype(vae)))
    mx.eval(lo_z, hi_z)
    m = vae.latents_mean.reshape(1, -1, 1, 1, 1)
    s = vae.latents_std.reshape(1, -1, 1, 1, 1)
    lo5 = (lo_z.astype(mx.float32) - m) / s
    hi5 = (hi_z.astype(mx.float32) - m) / s
    mx.eval(lo5, hi5)

    module = lr.load(node.resolve_upscaler_path(wanted), "bfloat16", cache)
    up5 = lr.upscale_latents(module, lo5, (44, 80), "bfloat16")
    mx.eval(up5)
    up, ht, lt = np.array(up5, np.float64), np.array(hi5, np.float64), np.array(lo5, np.float64)
    naive = np.repeat(np.repeat(lt, 2, axis=3), 2, axis=4)

    print("\n=== 1. 可用性 ===")
    print(
        f"  幅度：低分 {lt.std():.3f} | 高分真值 {ht.std():.3f} | 放大输出 {up.std():.3f}"
        f"（输出/真值 = {up.std() / ht.std():.2f}，理想 ≈1.0）"
    )
    base_corr, net_corr = chan_corr(naive, ht), chan_corr(up, ht)
    print(f"  逐通道相关（vs 高分真值）：朴素 2× 插值 {base_corr:+.3f}（基线） | 放大网络 {net_corr:+.3f}")
    ok = abs(up.std() / ht.std() - 1) < 0.2 and net_corr > base_corr
    print(f"  判定：{'✅ 可用' if ok else '❌ 不可用（输出与目标不相关 / 幅度跑偏）'}")

    print("\n=== 2. 映射关系（逐项排除「只差一个变换」）===")
    best = []
    for c in range(24):
        corrs = [np.corrcoef(up[0, c].reshape(-1), ht[0, k].reshape(-1))[0, 1] for k in range(24)]
        j = int(np.argmax(np.abs(corrs)))
        best.append((j, corrs[j]))
    same = sum(1 for c, (j, r) in enumerate(best) if j == c and abs(r) > 0.5)
    print(f"  通道置换：与真值同号通道相关 >0.5 的有 {same}/24；最佳匹配是双射 {len({j for j, _ in best}) == 24}")
    for name, arr in (
        ("H↔W 转置", up.transpose(0, 1, 2, 4, 3)),
        ("H 翻转", up[:, :, :, ::-1, :]),
        ("W 翻转", up[:, :, :, :, ::-1]),
        ("时间反转", up[:, :, ::-1, :, :]),
    ):
        print(f"  {name:8s} 逐通道相关 {chan_corr(arr, ht):+.4f}")
    res = []
    for c in range(24):
        x = up[0, c].reshape(-1)
        y = ht[0, c].reshape(-1)
        A = np.stack([x, np.ones_like(x)], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res.append(np.corrcoef(A @ coef, y)[0, 1])
    print(f"  逐通道仿射拟合后的相关 {np.mean(res):+.3f}（≈1 才说明只是比例/偏移问题）")
    pool = up.reshape(1, 24, up.shape[2], 22, 2, 40, 2).mean(axis=(4, 6))
    print(f"  输出 2×2 池化后 vs 低分 {chan_corr(pool, lt):+.4f}（内容是否只留在低分层面）")

    def band(x):
        f = np.fft.rfft2(x[0], axes=(-2, -1))
        p = (np.abs(f) ** 2).mean(axis=0)
        n = p.shape[-1]
        b = np.array([p[:, i * n // 6 : (i + 1) * n // 6].mean() for i in range(6)])
        return (b / b.sum() * 100).round(1)

    print(f"  频带占比：真值 {band(ht)} | 放大 {band(up)}（放大更平滑 → 不是高频噪声型错误）")
    print(
        "\n完整排查表见 docs/MINIMAX_H3_VIDEO_IMPLEMENTATION.md 附录 F；"
        "可用替代：把 MlxH3LatentUpscaler 的 model_name 选成「内置：latent 三线性插值」。"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
