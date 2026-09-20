"""MiniMax-H3（文本 → 视频 + 立体声）在本插件里的落地实现。

包内分三层：

- `model/`、`latent_creator/`、`scheduler/`：从 mlx-gen 0.37 的
  `mflux/models/minimax_h3` 移植过来的模型与几何算法（每个文件顶部有出处 banner，
  只改了导入路径），算法逐行一致 —— `rowwise` 分块、ATen 复刻的 sigma 网格、
  rope 的 float64 计算、音频 `weight_norm` 折叠、`17n+5` 吸附都不要「顺手优化」；
- `weights/`：本插件自写的权重定义与**流式加载器**（按 shard 读 → 过滤 → 改名 →
  逐张量量化 → 写进模块），绕开 mflux 0.19.1「一次读全量再量化」的加载路径
  （H3 的 54 GB transformer / 65 GB 条件编码器那样加载会直接 OOM）；
- `config.py` / `prompt.py` / `pipeline.py` / `video.py`：构造参数、提示词组装与
  编码、联合采样与解码、音频载荷（mp4 由 ComfyUI 自带的 CreateVideo + SaveVideo 落盘）。

导入本包**不再**在导入时全局关 TF32。MLX 在 fp32 GEMM 派发时读 `MLX_ENABLE_TF32`，
H3 的那些 fp32 计算必须在它们之前设置（否则音频解码的 7 级抗混叠上采样会把 1e-4 的
误差放大到 0.26）；但 MLX 0.32.2 上进程级关掉它会让 bf16 出图链路慢 2.5 倍（见
`h3_precision`）。因此改成按需开关：`comfyui_mlx_gen/pipeline.py` 里四处 H3 计算入口
用 `exact_fp32()` 包住，H3 算完立刻还原。
"""

from comfyui_mlx_gen.h3.model.h3_precision import (  # noqa: F401
    disable_tf32,
    exact_fp32,
    restore_tf32,
    tf32_status,
)

__all__ = ["disable_tf32", "exact_fp32", "restore_tf32", "tf32_status"]
