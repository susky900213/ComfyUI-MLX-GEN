"""自建权重加载（参照 mflux 的 weight_loader，但不走它的整体加载路径）。

支持：
- 目录：多 shard（0.safetensors …）+ model.safetensors.index.json
- 单文件：xxx.safetensors / xxx.pth / xxx.bin
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from inspect import signature
import json
from pathlib import Path
import re
from typing import Any

import mlx.core as mx

from . import paths, runtime
from .types import MlxModelEntry, QWEN_IMAGE_21_FAMILY, detect_model_family


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
    # 2.1 不使用 mflux ModelConfig；它由 qwen_image_21.loader 从 checkpoint 的
    # config.json 构造。若调用方误把 2.1 丢进这个 legacy 入口，仍要硬拦截。
    detected_family = detect_model_family(name_or_path)
    if detected_family == QWEN_IMAGE_21_FAMILY:
        raise NotImplementedError(
            "检测到 qwen_image_21（Qwen-Image 2.1）权重；该模型必须走独立的 "
            "qwen_image_21 配置/加载路径，不会回退到 legacy qwen_image ModelConfig"
        )

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
        # mflux's ``multi_json`` loader historically assumes the index is named
        # ``model.safetensors.index.json``.  Diffusers transformer components use
        # ``diffusion_pytorch_model.safetensors.index.json`` instead.  Handle that
        # layout locally while preserving mflux's normal path for its canonical
        # index name and for every other component.
        component_path = Path(path)
        alternate_index = _alternate_safetensors_index(component_path, comp.loading_mode)
        if alternate_index is not None:
            weights, quantization, version = _load_indexed_component(
                wl, comp, component_path, alternate_index, raw_cache
            )
        else:
            weights, quantization, version = wl._load_component(root, replace(comp, hf_subdir=rel), raw_cache)
    elif kind == "file":
        weights, quantization, version = _load_single_file(wl, comp, Path(path))
    else:
        raise ValueError(f"未知 kind: {kind}")
    return LoadedComponent(weights=weights, quantization=quantization, version=version, cache_key=key)


def _alternate_safetensors_index(path: Path, loading_mode: str) -> Path | None:
    """Return a non-mflux index name that the local loader can read.

    Diffusers exports transformer weights under ``diffusion_pytorch_model`` while
    mflux's ``multi_json`` implementation only looks for ``model``.  Do not use a
    broad glob here: selecting an arbitrary index could silently load the wrong
    variant when a directory contains more than one checkpoint.
    """

    if loading_mode != "multi_json":
        return None
    canonical = path / "model.safetensors.index.json"
    if canonical.is_file():
        return None
    alternate = path / "diffusion_pytorch_model.safetensors.index.json"
    return alternate if alternate.is_file() else None


def _indexed_safetensor_files(index_path: Path) -> tuple[Path, ...]:
    """Validate a safetensors index and return its declared shard paths.

    The checkpoint is local-only, but the index is still treated as untrusted
    input.  In particular, a malformed or path-traversing ``weight_map`` must not
    make the loader read files outside the selected component directory.
    """

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 safetensors 索引 {index_path}: {exc}") from exc

    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"safetensors 索引缺少非空 weight_map：{index_path}")

    # Keep the component path itself resolved, but do not resolve shard paths:
    # Hugging Face snapshots commonly store each file as a symlink to a blob
    # outside ``snapshots/<revision>/<component>``.
    component_dir = index_path.parent.resolve()
    raw_names = list(weight_map.values())
    if not all(isinstance(name, str) and name for name in raw_names):
        raise ValueError(f"safetensors 索引包含非字符串 shard 名称：{index_path}")
    names = sorted(set(raw_names))

    files: list[Path] = []
    for name in names:
        relative_name = Path(name)
        if relative_name.is_absolute() or ".." in relative_name.parts:
            raise ValueError(
                f"safetensors 索引 shard 路径非法：{index_path}: {name}"
            )
        candidate = component_dir / name
        try:
            candidate.relative_to(component_dir)
        except ValueError as exc:
            raise ValueError(
                f"safetensors 索引 shard 超出组件目录：{index_path}: {name}"
            ) from exc
        if not candidate.is_file():
            raise FileNotFoundError(
                f"safetensors 索引声明的 shard 不存在：{candidate}（索引：{index_path}）"
            )
        files.append(candidate)
    return tuple(files)


def _load_indexed_component(
    wl: Any,
    comp: Any,
    component_path: Path,
    index_path: Path,
    raw_cache: dict[tuple, dict] | None = None,
) -> tuple[dict[str, Any], int | None, str | None]:
    """Load a Diffusers-indexed component and apply its normal mflux mapping."""

    shard_files = _indexed_safetensor_files(index_path)
    cache_key = (str(component_path), comp.loading_mode, tuple(str(path) for path in shard_files))
    if raw_cache is not None and cache_key in raw_cache:
        raw = raw_cache[cache_key]
    else:
        raw: dict[str, Any] = {}
        for shard in shard_files:
            shard_weights, _metadata = mx.load(str(shard), return_metadata=True)
            for name, value in dict(shard_weights).items():
                if name in raw:
                    raise ValueError(f"safetensors 分片包含重复权重键 {name!r}：{index_path}")
                raw[name] = value
        if raw_cache is not None:
            raw_cache[cache_key] = raw

    if comp.weight_prefix_filters is not None:
        raw = {
            key: value
            for key, value in raw.items()
            if any(key.startswith(prefix) for prefix in comp.weight_prefix_filters)
        }

    if comp.key_transform is not None:
        transformed = {}
        for key, value in raw.items():
            transformed_key = comp.key_transform(key)
            if transformed_key is not None:
                transformed[transformed_key] = value
        raw = transformed

    if comp.weight_transform is not None:
        raw = {key: comp.weight_transform(key, value) for key, value in raw.items()}

    if comp.precision is not None:
        raw = wl._convert_precision(raw, comp.precision)

    if comp.mapping_getter is None:
        if comp.bulk_transform is not None:
            raw = {key: comp.bulk_transform(value) for key, value in raw.items()}
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


def _load_single_file(wl: Any, comp: Any, file: Path) -> tuple[dict[str, Any], int | None, str | None]:
    """与 WeightLoader._load_component 尾部逻辑一致，但直接加载单个文件。"""
    mode = "torch_checkpoint" if file.suffix in {".pt", ".bin"} else "single"
    raw: dict[str, Any] = wl._load_weights_file(file, mode)

    if comp.weight_prefix_filters:
        prefixes = tuple(comp.weight_prefix_filters)
        raw = {k: v for k, v in raw.items() if k.startswith(prefixes)}
    if comp.precision is not None:
        convert_kwargs = {}
        if "precision_override" in signature(wl._convert_precision).parameters:
            convert_kwargs["precision_override"] = getattr(comp, "precision_override", None)
        raw = wl._convert_precision(raw, comp.precision, **convert_kwargs)

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
