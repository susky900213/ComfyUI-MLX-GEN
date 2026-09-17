"""按配置实例化组件，并写入权重（不实例化未选中的组件）。"""

from __future__ import annotations

from dataclasses import replace
from inspect import signature
from typing import Any

from . import runtime, weights


def create_and_load(
    defn: Any,  # MlxModelEntry
    component: str,
    kind: str,
    path: str,
    quantize: int | None,
    raw_cache: dict[tuple, Any] | None = None,
    class_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """加载权重 → 创建实例 → 挂子模块 → 写入权重，返回 (实例, 实际量化位数)。"""
    loaded = weights.load_component(defn, component, kind, path, raw_cache)
    comp_def = defn.components[component]
    data_def = weights.component_def(defn, component)
    cls = runtime.import_object(comp_def.class_import)
    instance = cls(**(class_kwargs or {}))
    # 实例化之后、写权重之前跑挂钩（qwen 编辑的视觉塔）：
    # 必须早于 apply_weights —— Module.update(strict=False) 会静默丢弃模块树里没有的键。
    if comp_def.attach_import:
        runtime.import_object(comp_def.attach_import)(instance)
    bits = apply_weights(defn, data_def, instance, loaded, quantize)
    return instance, bits


def _stored_quantized(weights_dict: Any) -> bool:
    """这份组件权重在磁盘上是否已量化（看有没有成对的 weight + scales）。

    不能只看存档 metadata：`quantization_level` 是整个存档共用的一个值，
    而单个组件可能被定义成 skip_quantization（Qwen 的 text_encoder 就是这样，
    但本机这份 2511 存档里它确实带 327 个 scales）。
    """
    stack = [weights_dict]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "scales" in cur and "weight" in cur:
                return True
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
    return False


def apply_weights(
    defn: Any,
    data_def: Any,  # ComponentDefinition / TokenizerDefinition
    instance: Any,
    loaded: weights.LoadedComponent,
    quantize: int | None,
) -> int | None:
    """把已加载的权重写入实例（需要时量化）。

    ⚠️ 若磁盘上就是量化权重，**必须先量化模块再写权重**，即使权重定义标了
    `skip_quantization=True`：否则 q8 的打包权重（uint32）会被塞进普通 `Linear`，
    实测以 `addmm` 形状错误收场（更糟的情况是静默拿到垃圾输出）。
    """
    if data_def.skip_quantization and _stored_quantized(loaded.weights):
        print("[组件加载] 磁盘权重已量化 → 忽略 skip_quantization（否则打包权重会进未量化模块）")
        data_def = replace(data_def, skip_quantization=False)
    lw_mod = runtime.import_object("mflux.models.common.weights.loading.loaded_weights")
    wa = runtime.import_object("mflux.models.common.weights.loading.weight_applier:WeightApplier")
    loaded_weights = lw_mod.LoadedWeights(
        components={data_def.name: loaded.weights},
        meta_data=lw_mod.MetaData(
            quantization_level=loaded.quantization,
            mflux_version=loaded.version,
        ),
    )
    kwargs = dict(
        weights=loaded_weights,
        model=instance,
        component=data_def,
        quantize_arg=quantize,
    )
    # MFLUX 0.19.1 的新版接口允许模型族提供逐层量化 predicate（Ideogram 4 的
    # Fp8Linear 需要它）；兼容仓库既有环境中没有该参数的旧接口。
    if "quantization_predicate" in signature(wa.apply_and_quantize_single).parameters:
        kwargs["quantization_predicate"] = getattr(
            runtime.import_object(defn.weight_def), "quantization_predicate", None
        )
    return wa.apply_and_quantize_single(**kwargs)


def load_tokenizer(defn: Any, component: str, kind: str, path: str, max_length: int | None = None) -> Any:
    """加载 tokenizer（定义来自 weight def 的 tokenizers()）。"""
    loader = runtime.import_object("mflux.models.common.tokenizer.tokenizer_loader:TokenizerLoader")
    if kind == "missing":
        # 加固：原来会把不存在 / 断链的路径丢给 mflux，报出难懂的 FileNotFoundError
        raise FileNotFoundError(f"未找到 {component} 权重目录: {path}")
    if kind == "repo":
        raise RuntimeError(f"组件 {component} 指向 HF 仓库 {path}；请先下载到本地目录")
    data_def = replace(weights.component_def(defn, component), hf_subdir=".")
    over = {data_def.name: max_length} if max_length else None
    bundle = loader.load_all(
        definitions=[data_def],
        model_path=str(path),
        max_length_overrides=over,
    )
    return bundle.get(data_def.name)
