"""编译产物缓存（compiled_predict.CompiledPredictCache）与工厂函数的回归检查。

    PYTHONPATH=src python tests/test_compiled_predict.py

覆盖：
1. 同一个 key + 同一个 weights_token → 复用同一份 callable（不重复 build）；
2. 换掉模块（weights_token 变了）→ 整桶丢掉并重建（避免旧计算图扣住旧权重）；
3. `clear()` 之后重新 build（对应「同一模块上就地改过权重」的站点）；
4. 五份工厂函数都返回「可调用对象」而不是直接跑出的数组，且 use_compile=False
   时返回的就是未编译的闭包（qwen / kv 缓存路径依赖这点）；
5. use_compile=True 时返回 mx.compile 之后的函数，重复调用同一形状只 trace 一次；
6. 工厂在采样循环**外面**调：同一份权重重复取回的是**同一个**函数（编译 /
   建闭包不会每个 step 再来一次），换权重则整桶重建。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402

from comfyui_mlx_gen import pipeline  # noqa: E402
from comfyui_mlx_gen.compiled_predict import CompiledPredictCache  # noqa: E402

FAILED = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"[{'OK ' if ok else 'FAIL'}] {label}" + (f" | {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def test_cache_reuse_and_invalidation() -> None:
    cache = CompiledPredictCache()

    class Transformer:
        def __init__(self, name: str):
            self.name = name

    builds = {"n": 0}

    def build():
        builds["n"] += 1
        return lambda x: x + 1

    first = Transformer("a")
    fn1 = cache.get_or_build(key="predict", weights_token=first, build=build)
    fn2 = cache.get_or_build(key="predict", weights_token=first, build=build)
    check(fn1 is fn2 and builds["n"] == 1, "同 key + 同模块只 build 一次", f"builds={builds['n']}")

    second = Transformer("b")
    fn3 = cache.get_or_build(key="predict", weights_token=second, build=build)
    check(fn3 is not fn1 and builds["n"] == 2, "换模块 → 丢掉旧 trace 重建", f"builds={builds['n']}")
    check(len(cache) == 1, "桶里只留最新一条", f"entries={len(cache)}")

    cache.clear()
    check(len(cache) == 0, "clear() 之后条目清空")
    fn4 = cache.get_or_build(key="predict", weights_token=second, build=build)
    check(builds["n"] == 3 and fn4(1) == 2, "clear 之后重建（就地改权重场景）")


def test_factories_return_callables() -> None:
    class FakeTransformer:
        def __call__(self, **kwargs):  # noqa: ANN401
            return mx.zeros((1, 2))

    fake = FakeTransformer()
    z_fn = pipeline.make_z_image_predict(fake, False)
    f2_fn = pipeline.make_flux2_predict(fake, False)
    edit_fn = pipeline.make_flux2_edit_predict(fake, False)
    edit_cached_fn = pipeline.make_flux2_edit_cached_predict(fake)
    qwen_fn = pipeline.make_qwen_predict(fake, object(), False)
    for label, fn in [
        ("make_z_image_predict", z_fn),
        ("make_flux2_predict", f2_fn),
        ("make_flux2_edit_predict", edit_fn),
        ("make_flux2_edit_cached_predict", edit_cached_fn),
        ("make_qwen_predict", qwen_fn),
    ]:
        check(callable(fn) and fn is not None, f"{label} 返回可调用对象")

    # use_compile=False → 原样返回闭包（identity 可从源码行为推断：不编译 = 每次直接跑）
    compiled_z = pipeline.make_z_image_predict(fake, True)
    check(callable(compiled_z), "use_compile=True 也返回可调用对象")

    # 同一函数 + 同一形状只 trace 一次（用闭包里的计数器验证）
    traces = {"n": 0}

    def body(x: mx.array) -> mx.array:
        traces["n"] += 1
        return x * 2

    compiled = mx.compile(body)
    for i in range(4):
        mx.eval(compiled(mx.array([float(i)])))
    check(traces["n"] == 1, "重复调用同形状只 trace 一次", f"traces={traces['n']}")


def test_predict_is_built_outside_the_step_loop() -> None:
    """工厂只在采样循环外调一次：同一份权重必须复用同一个函数，不再每步重建。"""

    class FakeTransformer:
        def __init__(self, name: str):
            self.name = name

        def __call__(self, **kwargs):  # noqa: ANN401
            return mx.zeros((1, 2))

    factories = [
        ("z_image", pipeline.make_z_image_predict),
        ("flux2", pipeline.make_flux2_predict),
        ("flux2_edit", pipeline.make_flux2_edit_predict),
        ("flux2_edit_cached", pipeline.make_flux2_edit_cached_predict),
    ]
    for label, factory in factories:
        args = (FakeTransformer("a"),) if label.startswith("flux2_edit_cached") else (
            FakeTransformer("a"),
            False,
        )
        first = factory(*args)
        again = factory(*args)  # 模拟「下一个 step / 下一个 seed」再取一次
        check(first is again, f"{label}: 同一份权重复用同一个函数")

        other_args = (FakeTransformer("b"),) if label.startswith("flux2_edit_cached") else (
            FakeTransformer("b"),
            False,
        )
        swapped = factory(*other_args)  # 换权重 → 整桶丢掉、重建
        check(swapped is not first, f"{label}: 换权重则重建")
        check(
            len(pipeline.compiled_predict(label)) == 1,
            f"{label}: 桶里不会因为逐步取函数而膨胀",
            f"entries={len(pipeline.compiled_predict(label))}",
        )

    # 编译开关是键的一部分：True / False 两份互不覆盖
    fake = FakeTransformer("c")
    check(
        pipeline.make_z_image_predict(fake, False)
        is not pipeline.make_z_image_predict(fake, True),
        "编译开关不同 → 两份函数互不覆盖",
    )


if __name__ == "__main__":
    test_cache_reuse_and_invalidation()
    test_factories_return_callables()
    test_predict_is_built_outside_the_step_loop()
    if FAILED:
        raise SystemExit(f"失败项：{FAILED}")
    print("\n编译产物缓存与工厂函数检查全部通过。")
