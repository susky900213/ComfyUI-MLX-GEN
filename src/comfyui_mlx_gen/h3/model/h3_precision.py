# 移植自 mlx-gen 0.37 的 mflux/models/minimax_h3/model/h3_precision.py（MIT，Filip Strand / AbstractVision）；
# 本插件只改导入路径与包名，算法逻辑逐行一致。
"""fp32 exactness on Apple M5-class GPUs.

MLX 0.31 runs fp32 GEMM-class kernels (matmul, batched matmul, fused attention, im2col convolutions)
in TF32 by default on M5 (`MLX_ENABLE_TF32`), a ~8e-4 relative error per product. MiniMax-H3 keeps its
input/output heads, the timestep MLP and the whole audio VAE in fp32 on purpose, and TF32 there turns
the audio decode's 1e-4 parity into a 0.26 max error after seven alias-free upsampling stages.
Disabling TF32 costs nothing for the bf16 block stack. Measured 2026-09-04 on an M5 Max, MLX 0.31.0.

MLX reads the variable at the fp32 GEMM dispatch, so call `disable_tf32()` before the fp32 matmuls it
protects; `mlxgen generate` does this when the MiniMax-H3 runtime initializes.

本插件把上游的「H3 运行时初始化时设置」收窄成**只在 H3 计算期间**生效（`exact_fp32()`
上下文管理器，调用点见 `pipeline.py` 的四处 H3 计算入口）：上游那份 0.31 的实测结论是
「关掉 TF32 对 bf16 部分没有代价」，但 **MLX 0.32.2 上不再成立** —— 进程级关掉 TF32 会让
bf16 出图链路慢 2.5 倍（实测见 ``disable_tf32`` 的 docstring），而 ComfyUI 是长驻进程，
所以在包导入时全局关掉会把 Z-Image / Flux.2 / Qwen 这些出图模型一起拖慢。

另外（2026-09-20 实测）**这个开关在进程第一次 GEMM 派发时就被锁死**：先跑一次 GEMM
（fp32 / bf16 / q8 都算）再把变量设成 0，之后新建的 fp32 kernel 结果仍与「全程开」
逐位相同。所以长驻进程里它是「先到先得」——`tf32_status()` 负责探测本次是否真的关上了，
关不上时由 `disable_tf32()` 打印可操作的告警。
"""

import contextlib
import os

import mlx.core as mx

# 进程最初的值（用户可能在环境里显式设过 MLX_ENABLE_TF32）——还原时按它来
_ORIGINAL_TF32: str | None = os.environ.get("MLX_ENABLE_TF32")
_tf32_off_by_us = False
# 「用户显式设过 → 插件不越权」只提示一次
_explicit_tf32_noticed = False
# 锁死告警只提示一次（H3 每次计算都会进出这个窗口，不刷屏）
_latch_notice_shown = False
# 开锁探测结果：None = 还没探过；之后是 "honored" / "latched_tf32" / "unknown"
_TF32_STATE: list[str | None] = [None]
# 探测阈值：TF32 的相对误差 ~1e-3，纯 fp32 ~1e-6，取中间值即可区分
_TF32_ACTIVE_THRESHOLD = 1e-4

# MLX 0.31 `quantized_matmul` returns wrong values for inputs with 32768 or more rows when the scales are bf16
# (relative error 0.6-1.0; the first `rows - 32768` rows collapse to ~0). fp32 scales are exact but ~4x slower
# and promote the output to fp32, so quantized linears are applied in row chunks instead. A 1344x768 clip packs
# 37,851 rows. Reproduces with K=64, N=8, M=32769 (`tests/minimax_h3/test_h3_precision.py`).
QUANTIZED_MATMUL_MAX_ROWS = 32767


def tf32_status() -> str:
    """探测「TF32 现在是否真的关掉了」（每个进程只探一次）。

    **MLX 在进程第一次 GEMM 派发时读 ``MLX_ENABLE_TF32`` 并锁死**（2026-09-20 实测：
    先随便跑一次 GEMM——fp32 / bf16 / q8 都算——再把它设成 0，之后**新建**的 fp32
    kernel 结果仍与「全程开」逐位相同：sum|out| 833964032 vs 关掉时的 834552640）。
    也就是说这个开关在长驻进程里「先到先得」：

    - 若本次调用是进程里第一次 GEMM 派发 → 探测本身就把开关锁在「关」，
      H3 的 fp32 计算从此精确，返回 ``"honored"``；
    - 若进程里更早跑过别的 GEMM（别的模型 / 别的节点）→ 开关已被锁在「开」，
      我们关不掉了，返回 ``"latched_tf32"``（调用方据此告警）；
    - 探测不出来（numpy 缺失等）→ ``"unknown"``。

    实现：128×128 的 fp32 matmul 与 numpy float64 参考比相对误差 —— TF32 ≈ 7e-4，
    纯 fp32 ≈ 2e-7，阈值取 1e-4。
    """
    if _TF32_STATE[0] is not None:
        return _TF32_STATE[0]
    try:
        import numpy as np

        rng = np.random.default_rng(0)
        a = (rng.standard_normal((128, 128), dtype=np.float32) * 1e3).astype(np.float32)
        b = (rng.standard_normal((128, 128), dtype=np.float32) * 1e-3).astype(np.float32)
        exact = a.astype(np.float64) @ b.astype(np.float64).T
        got = np.asarray(mx.matmul(mx.array(a), mx.array(b).T), dtype=np.float64)
        relative = float(np.max(np.abs(got - exact)) / np.max(np.abs(exact)))
        _TF32_STATE[0] = "latched_tf32" if relative > _TF32_ACTIVE_THRESHOLD else "honored"
    except Exception:  # noqa: BLE001
        _TF32_STATE[0] = "unknown"
    return _TF32_STATE[0]


def disable_tf32() -> bool:
    """关掉 TF32（H3 那些 fp32 计算的精度需要）；返回「是否由本次调用改动」。

    2026-09-20 实测（M5 Max / MLX 0.32.2，Z-Image-Turbo 8bit 1024²，mflux 0.19.1 自带 CLI
    改进程环境变量即可复现）：``MLX_ENABLE_TF32=0`` 时同一条采样链路从 **2.0s/步** 变成
    **5.1s/步**（4 步总耗时 15.7s → 25.3s；分模块计时里 QuantizedLinear 2.44s → 5.55s、
    RMSNorm 0.12s → 0.47s）。也就是说 MLX 0.32 上这个开关**不只影响 fp32**，bf16 的
    matmul/归一化也走不到快路径，所以只在 H3 计算期间关、用完立刻还原。

    ⚠️ 但这个开关**在进程第一次 GEMM 派发时就被锁死**（实测：先跑一次 GEMM 再设 0，
    之后新建的 fp32 kernel 结果仍与「全程开」逐位相同），所以「用完还原」只还原环境
    变量、不会解锁；能不能真的关掉由 `tf32_status()` 探测并告警。

    环境里显式设过 ``MLX_ENABLE_TF32`` 时以用户为准（``=0`` 本来就不用关；``=1`` 表示
    用户接受 H3 的 TF32 误差，这里不越权覆盖，只提示一次）。
    """
    global _tf32_off_by_us, _explicit_tf32_noticed
    current = os.environ.get("MLX_ENABLE_TF32")
    if current == "0":
        return False
    if _ORIGINAL_TF32 is not None and current == _ORIGINAL_TF32:
        if not _explicit_tf32_noticed:
            _explicit_tf32_noticed = True
            print(
                f"[H3] 环境里显式设了 MLX_ENABLE_TF32={current} → 尊重该设置，"
                "H3 计算期间不关 TF32（fp32 精度会略降；去掉该变量即可让插件按需开关）",
                flush=True,
            )
        return False
    os.environ["MLX_ENABLE_TF32"] = "0"
    _tf32_off_by_us = True
    mx.clear_cache()  # 丢掉按旧标志 JIT 出来的 kernel，让新标志对之后的计算生效
    status = tf32_status()
    print(
        f"[H3] MLX_ENABLE_TF32 → 0（H3 计算期间关闭 TF32，保 fp32 精度；算完还原为 {_ORIGINAL_TF32 or '默认'}）",
        flush=True,
    )
    global _latch_notice_shown
    if not _latch_notice_shown:
        _latch_notice_shown = True
        if status == "latched_tf32":
            print(
                "[H3] 注意：本进程在更早的时候已经跑过一次 GEMM，而 MLX 在那一刻就把 TF32 开关"
                "锁死了（实测：此后改 MLX_ENABLE_TF32 不再影响结果）——本次关闭**对 H3 的 fp32 "
                "精度无效**，输入/输出头、时间步 MLP 与音频 VAE 仍会按 TF32 算（音频误差可能被 "
                "7 级抗混叠上采样放大到 0.26 量级）。要拿到精确的 H3：① 给 ComfyUI 的启动环境"
                "设 MLX_ENABLE_TF32=0（代价：同进程的出图模型会慢约 2.5 倍）；或 ② 让 H3 在"
                "独立进程 / 独立一次会话里跑（H3 是进程里第一个跑 GEMM 的模型时，插件会自动"
                "把开关锁在「关」）。",
                flush=True,
            )
        elif status == "honored":
            print(
                "[H3] 已把 MLX 的 TF32 开关锁在「关」（本进程此前没跑过 GEMM），H3 的 fp32 计算精确。"
                "注意：这个锁是进程级的、撤不回来——本进程之后跑出图模型会比平时慢约 2.5 倍，"
                "想恢复请重启 ComfyUI（或把 H3 与出图分开跑）。",
                flush=True,
            )
    return True



def restore_tf32() -> bool:
    """还原进程最初的 TF32 设置（只还原我们改过的那一次）；返回是否改动。"""
    global _tf32_off_by_us
    if not _tf32_off_by_us:
        return False
    if _ORIGINAL_TF32 is None:
        os.environ.pop("MLX_ENABLE_TF32", None)
    else:
        os.environ["MLX_ENABLE_TF32"] = _ORIGINAL_TF32
    _tf32_off_by_us = False
    mx.clear_cache()
    print(
        f"[H3] MLX_ENABLE_TF32 环境变量已还原为 {_ORIGINAL_TF32 or '默认'}（注：MLX 的 TF32 开关是"
        "进程第一次 GEMM 派发时锁死的，还原环境变量不会改变已经锁定的状态）",
        flush=True,
    )
    return True


@contextlib.contextmanager
def exact_fp32():
    """H3 计算期间关闭 TF32，退出时还原（含抛异常的情况）。

    用户显式设过 ``MLX_ENABLE_TF32`` 时 ``disable_tf32()`` 什么都不做，
    本上下文也就不会去改动它。
    """
    changed = disable_tf32()
    try:
        yield
    finally:
        if changed:
            restore_tf32()


def linear_input_dtype(layer) -> mx.Dtype:
    """The floating dtype a (possibly quantized or LoRA-wrapped) linear expects its input in.

    `nn.QuantizedLinear.weight` is packed `uint32`; casting activations to it truncates them to
    integers, so the scales (or the bias) carry the layer's real compute dtype.
    """
    base = getattr(layer, "linear", getattr(layer, "base_linear", layer))
    scales = getattr(base, "scales", None)
    if scales is not None:
        return scales.dtype
    return base.weight.dtype


def rowwise(layer, x: mx.array) -> mx.array:
    """Apply a linear layer to `(B, S, C)` input in chunks of at most `QUANTIZED_MATMUL_MAX_ROWS` rows."""
    rows = x.shape[1]
    if rows <= QUANTIZED_MATMUL_MAX_ROWS:
        return layer(x)
    return mx.concatenate(
        [layer(x[:, start : start + QUANTIZED_MATMUL_MAX_ROWS]) for start in range(0, rows, QUANTIZED_MATMUL_MAX_ROWS)],
        axis=1,
    )
