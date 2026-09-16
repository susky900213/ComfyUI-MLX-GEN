"""按配置实例化组件，并写入权重（不实例化未选中的组件）。"""

from __future__ import annotations

from dataclasses import replace
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
    """加载权重 → 创建实例 → 写入权重，返回 (实例, 实际量化位数)。"""
    loaded = weights.load_component(defn, component, kind, path, raw_cache)
    comp_def = defn.components[component]
    data_def = weights.component_def(defn, component)
    cls = runtime.import_object(comp_def.class_import)
    instance = cls(**(class_kwargs or {}))
    bits = apply_weights(defn, data_def, instance, loaded, quantize)
    return instance, bits


def apply_weights(
    defn: Any,
    data_def: Any,  # ComponentDefinition / TokenizerDefinition
    instance: Any,
    loaded: weights.LoadedComponent,
    quantize: int | None,
) -> int | None:
    """把已加载的权重写入实例（需要时量化）。"""
    lw_mod = runtime.import_object("mflux.models.common.weights.loading.loaded_weights")
    wa = runtime.import_object("mflux.models.common.weights.loading.weight_applier:WeightApplier")
    loaded_weights = lw_mod.LoadedWeights(
        components={data_def.name: loaded.weights},
        meta_data=lw_mod.MetaData(
            quantization_level=loaded.quantization,
            mflux_version=loaded.version,
        ),
    )
    return wa.apply_and_quantize_single(
        weights=loaded_weights,
        model=instance,
        component=data_def,
        quantize_arg=quantize,
    )


def load_tokenizer(defn: Any, component: str, kind: str, path: str, max_length: int | None = None) -> Any:
    """加载 tokenizer（定义来自 weight def 的 tokenizers()）。"""
    loader = runtime.import_object("mflux.models.common.tokenizer.tokenizer_loader:TokenizerLoader")
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
