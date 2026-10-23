"""ComfyUI 采样进度适配。

插件自己的推理代码也会被仓库内的独立测试直接导入，此时并没有 ``comfy`` 包。
因此只在创建进度条时延迟导入 ComfyUI；脱离 ComfyUI 运行时保留计数语义，但不
发送 UI 事件。
"""

from __future__ import annotations

from typing import Any


class SamplingProgress:
    """按绝对步数驱动 ComfyUI ``ProgressBar`` 的轻量包装。"""

    def __init__(self, total: int) -> None:
        self.total = max(1, int(total))
        self.current = 0
        self._bar: Any | None = None
        try:
            from comfy.utils import ProgressBar

            self._bar = ProgressBar(self.total)
        except (ImportError, AttributeError):
            # 单元测试或从命令行独立运行 pipeline 时没有 ComfyUI。
            pass

    def update(self, amount: int = 1) -> None:
        """前进 ``amount`` 步，且不会超过当前总步数。"""
        self.update_absolute(self.current + int(amount))

    def update_absolute(self, current: int, total: int | None = None) -> None:
        """设置绝对进度；``total`` 可供运行中才知道工作量的采样器调整总量。"""
        if total is not None:
            self.total = max(1, int(total))
        self.current = min(max(0, int(current)), self.total)
        if self._bar is not None:
            self._bar.update_absolute(self.current, self.total)

    def complete(self) -> None:
        """缓存命中等没有逐步回调的路径也应在 UI 中显示完成。"""
        self.update_absolute(self.total)


def throw_if_interrupted() -> None:
    """在可安全停止的阶段边界响应 ComfyUI 的“取消当前任务”。

    MLX/Metal 的单次 kernel 执行期间不能从 Python 抢占；调用方应在进入昂贵阶段
    之前、以及每个采样 step 之间调用本函数。独立测试环境没有 ComfyUI 时静默跳过。
    """

    try:
        from comfy import model_management
    except (ImportError, AttributeError):
        return
    checker = getattr(model_management, "throw_exception_if_processing_interrupted", None)
    if callable(checker):
        checker()
