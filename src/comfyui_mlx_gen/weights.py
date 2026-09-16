"""自建权重加载（参照 mflux 的 weight_loader，但不走它的整体加载路径）。

支持：
- 目录：多 shard（0.safetensors …）+ model.safetensors.index.json
- 单文件：xxx.safetensors / xxx.pth / xxx.bin
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any

from . import paths, runtime
from .types import MlxModelEntry


@dataclass(frozen=True)
class LoadedComponent:
    weights: dict[str, Any]
    quantization: int | None
    version: str | None
    cache_key: str


def component_def(defn: MlxModelEntry, component: str) -> Any:
    """从权重定义里找组件（含 mapping_getter 等）。"""
    cd = defn.components[component]
    wd = runtime.import_object(defn.weight_def)
    for source in (wd.get_components(), wd.get_tokenizers()):
        for item in source:
            if getattr(item, "name", None) == cd.name:
                return item
    raise KeyError(f"权重定义中没有组件 {component}（name={cd.name}）")


# 权重集名字里常见的「量化 / 精度」后缀（-8bit、-4bit、-q4、-mx4、-bf16 …），
# 匹配 ModelConfig 之前要先去掉，否则 z-image-turbo-8bit 对不上 z-image-turbo。
_QUANT_SUFFIX = re.compile(
    r"(?:[-_.](?:\d+(?:\.\d+)?bit|q\d+(?:\.\d+)?|mx\d+|fp(?:16|32|8|4)|bf16))+$",
    re.IGNORECASE,
)


def normalize_model_key(name: str) -> str:
    """权重集名 → 用来匹配 ModelConfig 的键（小写、去掉量化后缀与所有分隔符）。

    例：z-image-turbo-8bit → zimageturbo、flux.2-klein-9b-8bit → flux2klein9b、
    与注册表里的 "z-image-turbo" / "flux2-klein-9b" 归一化后一致。
    """
    key = str(name).strip().lower().replace("_", "-")
    key = _QUANT_SUFFIX.sub("", key)
    return re.sub(r"[^a-z0-9]", "", key)


def available_configs() -> dict[str, Any]:
    """mflux 的 ModelConfig 注册表（键 = 规范名，如 "z-image-turbo"）。"""
    return runtime.import_object("mflux.models.common.config.model_config:AVAILABLE_MODELS")


def config_for_path(name_or_path: str, fallback_key: str) -> Any:
    """按「权重集目录名 / 文件名 / repo id」挑一套 ModelConfig（不预先登记模型名）。

    取名字的规范形式（去掉 -8bit / -4bit 这类量化后缀）后，在 AVAILABLE_MODELS
    里找能前缀匹配上的**最长**键：z-image-turbo-8bit → z-image-turbo、
    z-image-8bit → z-image、z-image-turbo-4bit → 仍是 z-image-turbo。
    全都匹配不上时退回该大类的兜底配置（fallback_key = ModelConfig 工厂方法名）。
    """
    registry = available_configs()
    key = normalize_model_key(Path(name_or_path).name)
    if key:
        matches = [
            name
            for name in registry
            if key == normalize_model_key(name) or key.startswith(normalize_model_key(name))
        ]
        if matches:
            return registry[max(matches, key=len)]
    mc = runtime.import_object("mflux.models.common.config.model_config:ModelConfig")
    factory = getattr(mc, fallback_key, None)
    if factory is None:
        raise RuntimeError(
            f"无法从 {name_or_path} 判断用哪套配置，且 ModelConfig 没有兜底工厂方法 {fallback_key}"
        )
    return factory()


def load_component(
    defn: MlxModelEntry,
    component: str,
    kind: str,
    path: str,
    raw_cache: dict[tuple, dict] | None = None,
) -> LoadedComponent:
    """加载一个组件的权重（目录或单文件）。"""
    wl = runtime.import_object("mflux.models.common.weights.loading.weight_loader:WeightLoader")
    comp = component_def(defn, component)
    mapping = getattr(comp, "mapping_getter", None)
    config = {
        "kind": kind,
        "path": str(path),
        "component": component,
        "mapping": getattr(mapping, "__qualname__", None) if mapping else None,
        "loading_mode": comp.loading_mode,
    }
    key = runtime.cache_key(config)

    if kind in {"dir", "repo"}:
        root = Path(paths.MODEL_ROOT)
        if kind == "dir":
            target = Path(path)
            if not str(target).startswith(str(root)):
                root = target.parent
            rel = paths.relative_to_root(str(path))
        else:  # repo：需要预先下载到本地，这里不再联网
            raise RuntimeError(f"组件 {component} 指向 HF 仓库 {path}；请先下载到本地目录")
        weights, quantization, version = wl._load_component(root, replace(comp, hf_subdir=rel), raw_cache)
    elif kind == "file":
        weights, quantization, version = _load_single_file(wl, comp, Path(path))
    else:
        raise ValueError(f"未知 kind: {kind}")
    return LoadedComponent(weights=weights, quantization=quantization, version=version, cache_key=key)


def _load_single_file(wl: Any, comp: Any, file: Path) -> tuple[dict[str, Any], int | None, str | None]:
    """与 WeightLoader._load_component 尾部逻辑一致，但直接加载单个文件。"""
    mode = "torch_checkpoint" if file.suffix in {".pt", ".bin"} else "single"
    raw: dict[str, Any] = wl._load_weights_file(file, mode)

    if comp.weight_prefix_filters:
        prefixes = tuple(comp.weight_prefix_filters)
        raw = {k: v for k, v in raw.items() if k.startswith(prefixes)}
    if comp.precision is not None:
        raw = wl._convert_precision(raw, comp.precision, precision_override=comp.precision_override)

    if comp.mapping_getter is None:
        if comp.bulk_transform is not None:
            raw = {k: comp.bulk_transform(v) for k, v in raw.items()}
        from mlx.utils import tree_unflatten  # type: ignore

        return tree_unflatten(list(raw.items())), None, None

    mapped = runtime.import_object(
        "mflux.models.common.weights.mapping.weight_mapper:WeightMapper"
    ).apply_mapping(
        hf_weights=raw,
        mapping=comp.mapping_getter(),
        num_blocks=comp.num_blocks,
        num_layers=comp.num_layers,
    )
    return mapped, None, None
