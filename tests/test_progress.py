"""采样进度适配器的无模型测试。"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


updates: list[tuple[int, int]] = []


class FakeProgressBar:
    def __init__(self, total: int) -> None:
        updates.append((0, total))

    def update_absolute(self, current: int, total: int) -> None:
        updates.append((current, total))


fake_comfy = types.ModuleType("comfy")
fake_utils = types.ModuleType("comfy.utils")
fake_utils.ProgressBar = FakeProgressBar
fake_comfy.utils = fake_utils
sys.modules["comfy"] = fake_comfy
sys.modules["comfy.utils"] = fake_utils

from comfyui_mlx_gen.progress import SamplingProgress  # noqa: E402


progress = SamplingProgress(4)
progress.update()
progress.update(2)
progress.update_absolute(7, 8)
progress.complete()

assert updates == [(0, 4), (1, 4), (3, 4), (7, 8), (8, 8)], updates
assert progress.current == 8
assert progress.total == 8
print("[PASS] ComfyUI 采样进度按绝对步数更新并可动态调整总量")


# 验证通用图片采样分派传入逐步回调，并按 batch_size × steps 计算总量。
from comfyui_mlx_gen import pipeline  # noqa: E402
from comfyui_mlx_gen.cache import Cache  # noqa: E402


class RecordingProgress:
    instances = []

    def __init__(self, total: int) -> None:
        self.total = total
        self.current = 0
        self.completed = False
        self.__class__.instances.append(self)

    def update(self, amount: int = 1) -> None:
        self.current += amount

    def complete(self) -> None:
        self.current = self.total
        self.completed = True


class FakeLatents:
    shape = (2, 4, 8)
    dtype = "float32"


original_progress = pipeline.SamplingProgress
original_config_for_path = pipeline.weights.config_for_path
original_sample_z_image = pipeline._sample_z_image
try:
    pipeline.SamplingProgress = RecordingProgress
    pipeline.weights.config_for_path = lambda *_args: SimpleNamespace(
        model_name="fake-z-image", supports_guidance=False
    )

    def fake_sample(*args):
        on_progress = args[-1]
        assert on_progress is not None
        for _ in range(6):
            on_progress()
        return FakeLatents()

    pipeline._sample_z_image = fake_sample
    definition = SimpleNamespace(family="z_image", supported=True, default_config="fake")
    model = SimpleNamespace(
        model_type="z_image",
        model_path="fake-model",
        precision="float32",
        quantize=None,
    )
    params = {
        "steps": 3,
        "batch_size": 2,
        "guidance": 1.0,
        "height": 64,
        "width": 64,
        "ref_cache_key": "",
    }
    cache = Cache()
    pipeline.run_sampler(definition, model, {}, params, cache)
    pipeline.run_sampler(definition, model, {}, params, cache)
finally:
    pipeline.SamplingProgress = original_progress
    pipeline.weights.config_for_path = original_config_for_path
    pipeline._sample_z_image = original_sample_z_image

first, cached = RecordingProgress.instances
assert (first.total, first.current, first.completed) == (6, 6, False)
assert (cached.total, cached.current, cached.completed) == (6, 6, True)
print("[PASS] 图片采样按 batch_size × steps 上报，缓存命中时直接完成")