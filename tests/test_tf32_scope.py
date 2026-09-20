"""TF32 开关作用域的无模型测试。

背景（真实回归）：插件曾经在 `comfyui_mlx_gen.h3` 导入时就全局设
`MLX_ENABLE_TF32=0`。实测（M5 Max / MLX 0.32.2）这样会让**所有 bf16 出图模型**
慢 2.5 倍（Z-Image-Turbo 8bit 1024²：2.0s/步 → 5.1s/步）。现在改成只在 H3 计算
期间关、算完还原，本测试守住这三件事：导入不动环境、上下文内关/出还原、用户
显式设过时不越权。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import comfyui_mlx_gen  # noqa: F401,E402  导入整个插件包（会连带 h3 / 各节点）
from comfyui_mlx_gen.h3.model.h3_precision import (  # noqa: E402
    disable_tf32,
    exact_fp32,
    restore_tf32,
)

# ① 导入插件包本身不能动 MLX_ENABLE_TF32
assert os.environ.get("MLX_ENABLE_TF32") is None, (
    f"import comfyui_mlx_gen 之后不应设置 MLX_ENABLE_TF32，实际 {os.environ.get('MLX_ENABLE_TF32')!r}"
)

# ② H3 计算期间关，退出即还原
with exact_fp32():
    assert os.environ["MLX_ENABLE_TF32"] == "0", "exact_fp32() 内部应关闭 TF32"
assert os.environ.get("MLX_ENABLE_TF32") is None, "exact_fp32() 退出后应还原"

# ③ 抛异常也要还原
try:
    with exact_fp32():
        raise RuntimeError("boom")
except RuntimeError:
    pass
else:
    raise AssertionError("exact_fp32 没有把异常放出来")
assert os.environ.get("MLX_ENABLE_TF32") is None, "异常路径也必须还原"

# ④ 环境里显式设过（在导入之前设）时不越权：=1 保持 1、=0 保持 0
#    （`_ORIGINAL_TF32` 在模块导入时读取，所以这条只能在子进程里验）
CODE = (
    "import os, sys\n"
    "VALUE = {value!r}\n"
    "os.environ['MLX_ENABLE_TF32'] = VALUE\n"
    f"sys.path.insert(0, {str(SRC)!r})\n"
    "from comfyui_mlx_gen.h3.model.h3_precision import disable_tf32, restore_tf32\n"
    "assert disable_tf32() is False, '显式设过时不该覆盖'\n"
    "assert restore_tf32() is False, '没有改过就不该还原'\n"
    "assert os.environ['MLX_ENABLE_TF32'] == VALUE\n"
    "print('ok')\n"
)
for explicit in ("1", "0"):
    done = subprocess.run(
        [sys.executable, "-c", CODE.format(value=explicit)],
        env={**os.environ, "MLX_ENABLE_TF32": explicit},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, f"显式设 MLX_ENABLE_TF32={explicit} 的子进程失败：{done.stderr}"

# ⑤ 用户显式设 0（本进程内）：本来就不用关，也不需要还原
os.environ["MLX_ENABLE_TF32"] = "0"
assert disable_tf32() is False, "已经是 0 时 disable_tf32() 应返回 False"
assert restore_tf32() is False, "没改过就不该还原"
assert os.environ["MLX_ENABLE_TF32"] == "0"
os.environ.pop("MLX_ENABLE_TF32")

# ⑥ 嵌套调用：内层不提前还原，最外层退出后回到原状
with exact_fp32():
    with exact_fp32():
        assert os.environ["MLX_ENABLE_TF32"] == "0"
    assert os.environ["MLX_ENABLE_TF32"] == "0", "内层退出不应提前还原"
assert os.environ.get("MLX_ENABLE_TF32") is None

# ⑦ 开锁探测：返回三种状态之一，可重复调用（每个进程只真探一次）
from comfyui_mlx_gen.h3.model.h3_precision import tf32_status  # noqa: E402

status = tf32_status()
assert status in {"honored", "latched_tf32", "unknown"}, f"意外的探测结果 {status!r}"
assert tf32_status() == status, "探测结果应当在同一进程内保持稳定"

# ⑧ 模块导出（ComfyUI 之外的调用方也走这几个名字）
from comfyui_mlx_gen import h3  # noqa: E402

for name in ("exact_fp32", "disable_tf32", "restore_tf32", "tf32_status"):
    assert hasattr(h3, name), f"h3 应导出 {name}"

print("test_tf32_scope: ok")
