"""MLX 版 H3 Latent Upscaler（3D）的数值与接口回归。

运行：
    python tests/test_h3_latent_upscaler.py

守住三件事：
1. 与 torch 参考实现（同一套随机权重）逐点对拍 —— 覆盖 channels-last 布局、
   `conv1d(groups=C)` 的时序深度卷积、以及 `Upsample(linear, align_corners=False)`
   与 `F.interpolate(trilinear)` 的等价性（本机有 torch，所以能做这个对拍）；
2. 权重适配 / 架构探测（`adapt_state` / `detect_arch`）在真实键名上正确；
3. 有真实权重时：strict 加载通过、参数数 345.3 M、前向形状与耗时合理
   （权重不在就只跳过这一节）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen.h3.model import h3_latent_upscaler as up  # noqa: E402
from comfyui_mlx_gen.nodes import h3_two_stage as h3_two_stage_node  # noqa: E402

FAILED: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  {'OK   ' if condition else 'FAIL '}{label}")
    if not condition:
        FAILED.append(label)


# ------------------------------- torch 参考实现（与节点包 nodes/minimax_h3_latent_upscaler_3d.py 同构）
def _norm(channels: int) -> nn.Module:
    return nn.GroupNorm(32, channels)


class TorchResBlockEmb3D(nn.Module):
    def __init__(self, channels: int, emb: int = 64, dropout: float = 0.1):
        super().__init__()
        self.in_layers = nn.Sequential(_norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb, 2 * channels))
        self.out_norm = _norm(channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(p=dropout), nn.Conv3d(channels, channels, 3, padding=1))

    def forward(self, x, emb):
        h = self.in_layers(x)
        e = self.emb_layers(emb).type(h.dtype)
        while len(e.shape) < len(h.shape):
            e = e[..., None]
        scale, shift = torch.chunk(e, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return x + self.out_layers(h)


class TorchTemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.norm = _norm(channels)
        self.dwconv = nn.Conv3d(
            channels, channels, kernel_size=(kernel_size, 1, 1), padding=(pad, 0, 0), groups=channels
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class TorchLatentResizer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 24,
        in_blocks: int = 4,
        out_blocks: int = 4,
        channels: int = 32,
        dropout: float = 0.1,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(TorchResBlockEmb3D(channels, 64, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TorchTemporalConv(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(TorchResBlockEmb3D(channels, 64, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TorchTemporalConv(channels, temporal_kernel))
        self.norm_out = _norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale: float = 1.0, target_size=None):
        size = target_size or tuple(int(round(s * scale)) for s in x.shape[-3:])
        emb = self.embed(torch.tensor([[scale - 1.0]], dtype=x.dtype))
        y = self.conv_in(x)
        for block in self.in_blocks:
            y = block(y, emb) if isinstance(block, TorchResBlockEmb3D) else block(y)
        y = F.interpolate(y, size=size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            y = block(y, emb) if isinstance(block, TorchResBlockEmb3D) else block(y)
        return self.conv_out(F.silu(self.norm_out(y)))


# ------------------------------------------------ 1. torch ↔ MLX 对拍
print("1. torch 参考实现 ↔ MLX 实现（同一套随机权重，fp32）")
torch.manual_seed(0)
torch_ref = TorchLatentResizer3D().eval()
# 参考实现里的 zero_module 会把每个残差块的末卷积置零（那样对比就不敏感了），
# 这里统一改成同一套随机权重重跑，保证对拍真的在检验卷积 / 插值 / 归一化
with torch.no_grad():
    for parameter in torch_ref.parameters():
        nn.init.normal_(parameter, 0.0, 0.05)

mlx_model = up.LatentResizer3D(
    in_channels=24, channels=32, in_blocks=4, out_blocks=4, dropout=0.1, temporal_every=2, temporal_kernel=5
)
state = {key: mx.array(value.detach().numpy()) for key, value in torch_ref.state_dict().items()}
mlx_model.load_weights(up.adapt_state(state), strict=True)
check("同键名 strict 加载通过（含 conv1d 深度卷积的布局改写）", True)
# MLX 模块默认 training=True，Dropout 会真的丢激活（每次结果都不同）——见模块文档第 5 条
check("默认就是推理态（Dropout 恒等）", mlx_model.training is False)

torch.manual_seed(1)
x_torch = torch.randn(1, 24, 5, 8, 10)
target = (5, 12, 15)  # T 不变、H/W 非整数倍（1.5×）：顺带检验插值等价性
with torch.no_grad():
    reference = torch_ref(x_torch, scale=1.5, target_size=target).numpy()
out = mlx_model(mx.array(x_torch.numpy().transpose(0, 2, 3, 4, 1).astype(np.float32)), 1.5, (12, 15))
mlx_out = np.array(out.transpose(0, 4, 1, 2, 3))
check(f"输出形状一致 {mlx_out.shape} == {reference.shape}", mlx_out.shape == reference.shape)
max_diff = float(np.abs(mlx_out - reference).max())
scale_ref = float(np.abs(reference).max())
print(f"  max|Δ| = {max_diff:.3e}（参考输出幅度 {scale_ref:.3f}）")
check("逐点误差 < 1e-5（fp32；实测约 4e-7）", max_diff < 1e-5)

# 时间轴不变：T 方向不能被插值放大
check("时间轴长度不变", mlx_out.shape[2] == x_torch.shape[2])

# 官方 3D 节点的长视频路径：T>24 时分块，但时间长度 / 空间目标 / 数值有限性必须保持。
print("1b. 长视频 temporal chunking")
chunk_input = mx.random.normal((1, 24, 25, 8, 10)).astype(mx.float32)
chunk_out = mlx_model(
    chunk_input.transpose(0, 2, 3, 4, 1), 1.5, (12, 15), enable_chunking=True
)
check("chunking 输出形状正确", tuple(chunk_out.shape) == (1, 25, 12, 15, 24))
check("chunking 输出有限", bool(mx.all(mx.isfinite(chunk_out)).item()))

# ------------------------------------------------ 2. 权重适配 / 架构探测
print("2. adapt_state / detect_arch")
adapted = dict(up.adapt_state(state))
check(
    "5D 卷积核转成 (O, kD, kH, kW, I)",
    adapted["conv_in.weight"].shape == (32, 3, 3, 3, 24),
)
dwconv_key = next(k for k in adapted if k.endswith("dwconv.weight"))
check(f"深度卷积核转成 (C, k, 1)：{dwconv_key}", adapted[dwconv_key].shape == (32, 5, 1))
check("GroupNorm / Linear 布局不变", tuple(adapted["norm_out.weight"].shape) == (32,))

arch = up.detect_arch(state)
check(
    f"detect_arch 读出 {arch['channels']}ch × {arch['in_blocks']}+{arch['out_blocks']}"
    f"（temporal_every={arch['temporal_every']}, kernel={arch['temporal_kernel']}）",
    arch["in_channels"] == 24
    and arch["channels"] == 32
    and arch["in_blocks"] == 4
    and arch["out_blocks"] == 4
    and arch["temporal_every"] == 2
    and arch["temporal_kernel"] == 5,
)

# ------------------------------------------------ 3. channels-first 包装
print("3. upscale_latents 包装")
x5d = mx.random.normal((1, 24, 5, 8, 10)).astype(mx.float32)
y5d = up.upscale_latents(mlx_model, x5d, (16, 20), "float32")
check(
    f"(1,C,T,H,W) 进出：{tuple(y5d.shape)}",
    tuple(y5d.shape) == (1, 24, 5, 16, 20) and y5d.dtype == mx.float32,
)

print("3b. normalized latent 输出护栏")
safe = mx.random.normal((1, 24, 2, 4, 4)).astype(mx.float32)
safe_out = safe * 1.5
in_std, out_std, ratio = h3_two_stage_node.validate_neural_output(safe, safe_out, "test")
check("正常幅度输出通过", in_std > 0.0 and out_std > 0.0 and ratio < 4.0)
bad = safe + mx.random.normal(safe.shape).astype(mx.float32) * 6.0
try:
    h3_two_stage_node.validate_neural_output(safe, bad, "bad-checkpoint")
except RuntimeError as exc:
    check("失控输出在进入 VAE 前被拒绝", "输出幅度失控" in str(exc) and "std=" in str(exc))
else:
    check("失控输出在进入 VAE 前被拒绝", False)

items = h3_two_stage_node.upscaler_items()
check("神经模型优先于手动插值选项", not items or items[-1] == h3_two_stage_node.BUILTIN_UPSCALE)

# ------------------------------------------------ 4. 真实权重（没有就跳过）
print("4. 真实权重（models/mlx/upscaler/）")
from comfyui_mlx_gen import paths  # noqa: E402

real = sorted(paths.component_dir("upscaler").glob("*.safetensors"))
if not real:
    print("  SKIP 没有找到真实权重（把 minimax_h3_latent_upscaler_3d_*.safetensors 放到 models/mlx/upscaler/）")
else:
    t0 = time.perf_counter()
    module = up.build(str(real[0]), "bfloat16")
    build_seconds = time.perf_counter() - t0
    params = sum(v.size for _, v in mlx.utils.tree_flatten(module.parameters()))
    check(f"strict 加载 {real[0].name}（{params / 1e6:.1f} M 参数，{build_seconds:.1f}s）", params > 300e6)
    check("推理态", module.training is False)
    big = mx.random.normal((1, 24, 37, 22, 40)).astype(mx.float32)  # 124 帧 @640×352
    t0 = time.perf_counter()
    out = up.upscale_latents(module, big, (44, 80), "bfloat16")  # → 1280×704
    elapsed = time.perf_counter() - t0
    print(f"  640×352 → 1280×704：{elapsed:.1f}s，输出 {tuple(out.shape)}")
    check("输出形状 (1,24,37,44,80) 且有限", tuple(out.shape) == (1, 24, 37, 44, 80) and bool(mx.all(mx.isfinite(out)).item()))
    check("单次放大 < 30s（实测约 4s）", elapsed < 30.0)

if FAILED:
    raise SystemExit(f"\n{len(FAILED)} 项失败：{', '.join(FAILED)}")
print("\n对拍与接口检查通过。")

