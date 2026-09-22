"""MLX 版 MiniMax-H3 Latent Upscaler（3D）。

结构与 ComfyUI 节点包 `Comfyui_Minimax_h3_latent_Upscaler`（MIT，LBH-123-AI）的
`nodes/minimax_h3_latent_upscaler_3d.py` 逐块对应：

    conv_in → [ResBlockEmb3D, TemporalConv] × 6 → 三线性插值（仅 H/W）
            → [ResBlockEmb3D, TemporalConv] × 6 → norm_out → SiLU → conv_out

权重文件（已在真实权重上验证 strict 加载通过，345.3 M 参数）：
`minimax_h3_latent_upscaler_3d_{bf16,fp16}.safetensors`
（in_channels=24、channels=512、in/out_blocks=12、temporal_every=2、kernel=5、无 attn）

三处 MLX 适配（都实测确认过，务必别改回去）：

1. 激活与卷积核都是 **channels-last**：激活 `(N, T, H, W, C)`，torch 的
   `(O, I, kD, kH, kW)` 要 `transpose(0, 2, 3, 4, 1)` 成 `(O, kD, kH, kW, I)`；
2. **`nn.GroupNorm` 必须加 `pytorch_compatible=True`**：MLX 默认把通道按
   `reshape(batch, -1, num_groups)` 跨步分组，而 PyTorch 是按连续通道块分组 ——
   不加这个开关，权重能 strict 加载、形状全对，但每组统计量取错，输出直接偏 3e-1
   （实测：同权重同输入，torch 与 MLX 的 GroupNorm 相差 0.34，而 conv / 插值只差 1e-6）；
3. `mx.conv3d` 拒绝 `groups != 1`（报 "Can only handle groups != 1 in 1D or 2D"），
   时序深度卷积 `(k,1,1)` 走 `(B,T,H,W,C) → (B·H·W, T, C) → mx.conv1d(groups=C)`，
   权重 `(C,1,k,1,1) → (C,k,1)`；
4. `nn.Upsample` 只接受 `scale_factor`（没有 `size=`）且按 channels-last 插值；
   `align_corners=False` 的索引是 `(i+0.5)·step−0.5`，与 torch `F.interpolate`
   的三线性一致（实测差 2e-6）—— 时间轴必须给 `1.0`（只放大 H/W）；
5. **必须在加载后调 `module.eval()`**：MLX 的模块默认 `training=True`，而
   `nn.Dropout` 只在非训练态才恒等 —— 忘了 eval() 每次推理都会随机丢 10% 激活
   （实测输出偏差 1.6e-1，且每次结果都不同）。`Dropout` 本身要保留在层序里当占位
   （键名索引 `out_layers.2` 靠它对齐），所以不能换成 `Identity` 之外的东西。

输入是**归一化后**的 H3 视频 latent：本插件的 `video_rows` 已经是 `(z - mean) / std`
（`decode_video` 里做的是 `*std + mean`），与放大网络的训练空间一致，因此**不需要**
再做归一化/反归一化。
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

BUCKET = "h3_latent_upscaler"
DEFAULT_IN_CHANNELS = 24
DEFAULT_CHANNELS = 512
DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def group_norm(channels: int, groups: int = 32) -> nn.GroupNorm:
    """`GroupNorm(32, C)` —— **必须** `pytorch_compatible=True`。

    MLX 默认按 `reshape(batch, -1, num_groups)` 跨步分组，PyTorch 按连续通道块分组；
    参考权重是 PyTorch 训的，用默认分组会取错每组的均值 / 方差（实测偏差 3e-1）。
    """
    return nn.GroupNorm(groups, channels, pytorch_compatible=True)


class ResBlockEmb3D(nn.Module):
    """与 torch 参考同层序（含 Dropout 占位），否则 strict 加载时键名对不上。"""

    def __init__(self, channels: int, emb: int = 64, dropout: float = 0.1):
        super().__init__()
        self.in_layers = [group_norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1)]
        self.emb_layers = [nn.SiLU(), nn.Linear(emb, 2 * channels)]
        self.out_norm = group_norm(channels)
        self.out_layers = [nn.SiLU(), nn.Dropout(dropout), nn.Conv3d(channels, channels, 3, padding=1)]

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        h = x
        for layer in self.in_layers:
            h = layer(h)
        e = emb
        for layer in self.emb_layers:
            e = layer(e)
        scale, shift = mx.split(e.reshape(1, 1, 1, 1, -1), 2, axis=-1)
        h = self.out_norm(h) * (1 + scale) + shift
        for layer in self.out_layers:
            h = layer(h)
        return x + h


class DepthwiseTemporalConv3D(nn.Module):
    """groups=C 的 `(k,1,1)` 深度卷积：MLX 的 conv3d 不支持 groups，改走 conv1d。"""

    def __init__(self, channels: int, kernel: int, padding: int):
        super().__init__()
        self.weight = mx.zeros((channels, kernel, 1))  # (O, k, I/g)，等价 torch 的 (C,1,k,1,1)
        self.bias = mx.zeros((channels,))
        self.pad = padding
        self.groups = channels

    def __call__(self, x: mx.array) -> mx.array:
        b, t, h, w, c = x.shape
        y = x.transpose(0, 2, 3, 1, 4).reshape(b * h * w, t, c)
        y = mx.conv1d(y, self.weight, padding=self.pad, groups=self.groups) + self.bias
        return y.reshape(b, h, w, t, c).transpose(0, 3, 1, 2, 4)


class TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel: int = 5):
        super().__init__()
        self.norm = group_norm(channels)
        self.dwconv = DepthwiseTemporalConv3D(channels, kernel, kernel // 2)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.pwconv(self.dwconv(nn.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    """纯 3D 主干：只放大 H/W，时间长度不变。"""

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        channels: int = DEFAULT_CHANNELS,
        in_blocks: int = 12,
        out_blocks: int = 12,
        dropout: float = 0.1,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.channels = int(channels)
        self.conv_in = nn.Conv3d(self.in_channels, self.channels, 3, padding=1)
        self.embed = [nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64)]
        self.in_blocks: list[Any] = []
        for block in range(int(in_blocks)):
            self.in_blocks.append(ResBlockEmb3D(self.channels, dropout=dropout))
            if temporal_every > 0 and block % temporal_every == 0:
                self.in_blocks.append(TemporalConv(self.channels, temporal_kernel))
        self.out_blocks: list[Any] = []
        for block in range(int(out_blocks)):
            self.out_blocks.append(ResBlockEmb3D(self.channels, dropout=dropout))
            if temporal_every > 0 and block % temporal_every == 0:
                self.out_blocks.append(TemporalConv(self.channels, temporal_kernel))
        self.norm_out = group_norm(self.channels)
        self.conv_out = nn.Conv3d(self.channels, self.in_channels, 3, padding=1)
        # 推理专用模块：默认 `training=True` 会让 nn.Dropout 真的丢激活，必须切推理态
        self.eval()

    def _forward_segment(
        self,
        x: mx.array,
        scale: float,
        size: tuple[int, int, int],
    ) -> mx.array:
        """对一个时间段做完整前向；`size` 是 `(T, H, W)`。"""
        emb = mx.array([[float(scale) - 1.0]], dtype=x.dtype)
        for layer in self.embed:
            emb = layer(emb)
        y = self.conv_in(x)
        for block in self.in_blocks:
            y = block(y, emb) if isinstance(block, ResBlockEmb3D) else block(y)
        y = nn.Upsample(
            scale_factor=(size[0] / y.shape[1], size[1] / y.shape[2], size[2] / y.shape[3]),
            mode="linear",
            align_corners=False,
        )(y)
        for block in self.out_blocks:
            y = block(y, emb) if isinstance(block, ResBlockEmb3D) else block(y)
        return self.conv_out(nn.silu(self.norm_out(y)))

    def __call__(
        self,
        x: mx.array,
        scale: float,
        size_hw: tuple[int, int],
        enable_chunking: bool = True,
    ) -> mx.array:
        """`x` 是 `(1, T, H, W, C)`；只放大 H/W，时间长度不变。

        官方 3D 节点在超过 24 个 latent 帧时使用 replicate padding + overlap blend。
        MLX 端保持同一边界，避免长视频直接整段前向时 OOM 或出现末尾闪烁；短片仍走
        单段路径，数值与之前完全一致。
        """
        target = (int(x.shape[1]), int(size_hw[0]), int(size_hw[1]))
        if not enable_chunking or int(x.shape[1]) <= 24:
            return self._forward_segment(x, scale, target)

        temporal_kernel = 5
        for block in self.in_blocks:
            if isinstance(block, TemporalConv):
                temporal_kernel = int(block.dwconv.weight.shape[1])
                break
        overlap = max(1, temporal_kernel)
        chunk_size = 24
        source_t = int(x.shape[1])

        # 与 torch F.pad(mode="replicate") 等价。padded 的原始帧区间从 index=overlap 开始。
        padded = mx.concatenate(
            [mx.repeat(x[:, :1], overlap, axis=1), x, mx.repeat(x[:, -1:], overlap, axis=1)],
            axis=1,
        )
        output = mx.zeros((x.shape[0], source_t, target[1], target[2], x.shape[4]), dtype=x.dtype)
        weights = mx.zeros((1, source_t, 1, 1, 1), dtype=x.dtype)

        for start in range(0, source_t, chunk_size):
            end = min(source_t, start + chunk_size)
            output_start = max(0, start - overlap)
            output_end = min(source_t, end + overlap)
            # `low/high` are indices into the replicate-padded input, exactly as in
            # the PyTorch reference: the original frame zero lives at padded index
            # `overlap`, not at index zero.
            low = max(0, output_start - overlap)
            high = min(source_t + 2 * overlap, output_end + overlap)

            segment = padded[:, low:high]
            segment_out = self._forward_segment(
                segment,
                scale,
                (high - low, target[1], target[2]),
            )
            source_offset = (output_start + overlap) - low
            valid = segment_out[:, source_offset : source_offset + output_end - output_start]
            length = output_end - output_start

            blend = mx.ones((length,), dtype=x.dtype)
            if start > output_start:
                count = start - output_start
                left = mx.arange(1, count + 1, dtype=x.dtype) / (count + 1)
                blend = mx.concatenate([left, blend[count:]])
            if output_end > end:
                count = output_end - end
                right = mx.arange(count, 0, -1, dtype=x.dtype) / (count + 1)
                blend = mx.concatenate([blend[:-count], right])
            blend = blend.reshape(1, length, 1, 1, 1)

            output[:, output_start:output_end] = output[:, output_start:output_end] + valid * blend
            weights[:, output_start:output_end] = weights[:, output_start:output_end] + blend
            mx.eval(segment_out, output, weights)

        return output / mx.maximum(weights, mx.array(1e-8, dtype=x.dtype))


# --- 权重适配 / 架构探测 / 加载 / 前向包装 ----------------------------------------
def adapt_state(state: dict[str, Any]) -> list[tuple[str, mx.array]]:
    """torch state_dict → MLX 参数树（只改布局，不改键名）。"""
    pairs: list[tuple[str, mx.array]] = []
    for key, value in state.items():
        if key.endswith("dwconv.weight") and value.ndim == 5:
            value = value.reshape(value.shape[0], value.shape[2], 1)
        elif value.ndim == 5:
            value = value.transpose(0, 2, 3, 4, 1)
        pairs.append((key, value))
    return pairs


def detect_arch(state: dict[str, Any]) -> dict[str, Any]:
    """从实际权重里读架构（与参考实现的 `_detect_arch` 同口径，attn 一律视为关）。"""
    import re

    cfg: dict[str, Any] = {
        "in_channels": DEFAULT_IN_CHANNELS,
        "channels": DEFAULT_CHANNELS,
        "in_blocks": 12,
        "out_blocks": 12,
        "temporal_every": 2,
        "temporal_kernel": 5,
        "dropout": 0.1,
        "attn": False,
    }
    conv_in = state.get("conv_in.weight")
    if conv_in is not None:
        cfg["channels"] = int(conv_in.shape[0])
        cfg["in_channels"] = int(conv_in.shape[1])
    in_ids = {int(m.group(1)) for k in state if (m := re.match(r"in_blocks\.(\d+)\.in_layers\.", k))}
    out_ids = {int(m.group(1)) for k in state if (m := re.match(r"out_blocks\.(\d+)\.in_layers\.", k))}
    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)
    temporal = sorted(int(m.group(1)) for k in state if (m := re.match(r"in_blocks\.(\d+)\.dwconv\.weight", k)))
    cfg["temporal_every"] = (temporal[1] - temporal[0] - 1) if len(temporal) >= 2 else 0
    for key, value in state.items():
        if key.endswith("dwconv.weight") and value.ndim == 5:
            cfg["temporal_kernel"] = int(value.shape[2])
            break
    return cfg


def build(path: str, precision: str = "bfloat16") -> LatentResizer3D:
    """按权重文件构造并 strict 加载（键名 / 形状对不上会直接抛错，不静默降级）。"""
    state = dict(mx.load(str(path)))
    cfg = detect_arch(state)
    module = LatentResizer3D(
        in_channels=cfg["in_channels"],
        channels=cfg["channels"],
        in_blocks=cfg["in_blocks"],
        out_blocks=cfg["out_blocks"],
        dropout=cfg["dropout"],
        temporal_every=cfg["temporal_every"],
        temporal_kernel=cfg["temporal_kernel"],
    )
    module.load_weights(adapt_state(state), strict=True)
    # MLX 的模块默认是 training=True，而 nn.Dropout 只在 training 时才恒等；
    # 不 eval() 的话每次推理都会随机丢掉 10% 激活（实测输出偏差 1.6e-1、且每次不同）
    module.eval()
    dtype = DTYPES.get(str(precision))
    if dtype is None:
        raise ValueError(f"precision 只能是 {' / '.join(DTYPES)}，收到 {precision!r}")
    if dtype != mx.bfloat16:
        module.set_dtype(dtype)
    mx.eval(module.parameters())
    params = sum(v.size for _, v in mlx.utils.tree_flatten(module.parameters()))
    print(
        f"[MlxH3LatentUpscaler] 已加载 {path}（{params / 1e6:.1f} M 参数，{precision}，"
        f"{cfg['channels']}ch × {cfg['in_blocks']}+{cfg['out_blocks']} blocks）"
    )
    return module


def load(path: str, precision: str = "bfloat16", cache: Any | None = None) -> LatentResizer3D:
    """按 `(path, precision)` 复用模块实例（0.64 GB，常驻一份很划算）。"""
    if cache is None:
        return build(path, precision)
    from comfyui_mlx_gen import runtime

    key = runtime.cache_key({"kind": BUCKET, "path": str(path), "precision": str(precision)})
    module, hit = cache.get_or_create(BUCKET, key, lambda: build(path, precision))
    if hit:
        print(f"[MlxH3LatentUpscaler] 复用已加载的放大模型（{precision}）")
    return module


def release(path: str, precision: str = "bfloat16", cache: Any | None = None) -> None:
    """释放一次推理使用的放大器；输出 latent 已由调用方独立缓存。"""
    if cache is None:
        return
    from comfyui_mlx_gen import runtime

    key = runtime.cache_key({"kind": BUCKET, "path": str(path), "precision": str(precision)})
    if cache.evict(BUCKET, key):
        print("[MlxH3LatentUpscaler] 已释放放大模型（下次放大时重新懒加载）")


def upscale_latents(
    module: LatentResizer3D,
    latents: mx.array,
    size_hw: tuple[int, int],
    precision: str = "bfloat16",
    enable_chunking: bool = True,
) -> mx.array:
    """`(1, C, T, H, W)` → `(1, C, T, H_out, W_out)`（进出都是 channels-first）。

    `size_hw` 是目标 latent 的 `(H, W)`；`scale` 条件取两轴有效倍率的平均（与参考一致）。
    """
    dtype = DTYPES.get(str(precision), mx.bfloat16)
    scale = (float(size_hw[0]) / latents.shape[3] + float(size_hw[1]) / latents.shape[4]) / 2.0
    x = latents.astype(dtype).transpose(0, 2, 3, 4, 1)
    y = module(x, scale, (int(size_hw[0]), int(size_hw[1])), bool(enable_chunking))
    mx.eval(y)
    return y.transpose(0, 4, 1, 2, 3).astype(mx.float32)

