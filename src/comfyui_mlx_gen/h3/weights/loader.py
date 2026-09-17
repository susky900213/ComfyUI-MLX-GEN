"""MiniMax-H3 的流式权重加载器（本插件自写，替代 mflux 的 WeightLoader + WeightApplier）。

为什么不能直接用 `comfyui_mlx_gen.components.create_and_load`：它走的是「一次读全量 →
再量化」（mflux 0.19.1 的 `_load_multi_glob` + `apply_and_quantize_single`），而 H3 的
两大组件在 128 GB 机器上那样加载必然 OOM —— 条件编码器会先把 65 GB bf16 全读进来再叠
25 GB q8 副本，transformer 要叠 54 GB bf16 + 27 GB q8。这里按 shard「读 → 过滤 →
改名 → 逐张量量化 → 写进模块 → 立刻释放」，峰值只有「最终足迹 + 单个 shard」。

流程与参考实现的 `StreamingWeightLoader.load_into` 一致：

1. 按组件目录的 `config.json` 构造模块（还没有权重）；
2. 用 `nn.quantize` 先把**结构**换成量化层（此时参数仍是未求值的懒图，不占内存）；
3. 逐个 shard：`mx.load` → 键过滤（条件编码器只留前 50 层与视觉塔）→ 改名
   （`WeightMapper`）/ 逐张量变换（3D 核转置、`weight_norm` 折叠）→ 精度转换 →
   对量化模块调 `mx.quantize` 生成 weight/scales/biases → `model.update`；
4. 全部写完做一次形状 / 缺键校验（**必须**：`update(strict=False)` 会静默丢弃未知键）。
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from comfyui_mlx_gen.h3 import config
from comfyui_mlx_gen.h3.weights import h3_weight_definition as h3def
from comfyui_mlx_gen.h3.weights.h3_weight_mapping import MiniMaxH3WeightMapping

_DTYPE_NAMES = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}
_FLOAT_DTYPES = (mx.float32, mx.float16, mx.bfloat16)


@dataclass(frozen=True)
class LoadedH3Component:
    """一个已物化的组件（模块 + 量化档位 + 生效精度）。"""

    role: str
    module: Any
    bits: int | None
    dtype: str
    path: str = ""
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def parameter_bytes(self) -> int:
        """常驻参数字节数（内存预检用）。"""
        return sum(int(value.size) * int(value.itemsize) for _, value in tree_flatten(self.module.parameters()))


def effective_dtype(definition: h3def.H3ComponentDef, precision: str) -> str:
    """组件的生效精度：组件规则里的强制精度优先（音频 VAE = float32）。"""
    return definition.precision or precision


def load(
    role: str,
    kind: str,
    path: str,
    quantize: int | None = None,
    precision: str = "bfloat16",
    log: Callable[[str], None] | None = print,
) -> LoadedH3Component:
    """把一个 H3 组件从磁盘流式加载进内存（kind 由 paths.resolve 给出，当前只支持目录 / 单文件）。"""
    if kind == "missing":
        raise FileNotFoundError(f"未找到 {role} 权重: {path}")
    definition = h3def.MiniMaxH3WeightDefinition.component(role)
    dtype_name = effective_dtype(definition, precision)
    if dtype_name not in _DTYPE_NAMES:
        raise ValueError(f"未知精度 {dtype_name}（可选：{', '.join(_DTYPE_NAMES)}）")
    base_dtype = _DTYPE_NAMES[dtype_name]
    bits = None if definition.skip_quantization else (int(quantize) if quantize else None)

    module = config.build_model(role, path)
    # 量化前的参数形状：量化会把 (out, in) 换成 (out, in * 32 / bits)，校验要用原始形状
    expected = {key: tuple(value.shape) for key, value in tree_flatten(module.parameters())}
    if bits:
        # 只改结构：此时参数还是未求值的懒初始化，nn.quantize 不会真的算一遍
        nn.quantize(
            module,
            group_size=h3def.GROUP_SIZE,
            bits=bits,
            class_predicate=h3def.MiniMaxH3WeightDefinition.quantization_predicate,
        )
    quantized_modules = {
        name: sub
        for name, sub in module.named_modules()
        if isinstance(sub, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    }
    if log:
        quantized_note = f"量化 q{bits}（{len(quantized_modules)} 个模块）" if bits else "不量化"
        log(f"[H3 加载] {role}: {len(expected)} 个参数，{quantized_note}，精度 {dtype_name}")
        log(f"[H3 加载] {role}: 目录 {path}")
    if bits and log:
        log(
            "[H3 加载] %s: 敏感路径保持原精度（%s）"
            % (role, " / ".join(h3def.MiniMaxH3WeightDefinition.TRANSFORMER_QUANTIZATION_SENSITIVE_FRAGMENTS))
        )

    loaded: dict[str, tuple[int, ...]] = {}
    for shard in _shard_files(role, path):
        written = _load_shard(
            module, role, shard, base_dtype, bits, quantized_modules, expected, loaded, log
        )
        if log:
            log(f"[H3 加载] {role}: {Path(shard).name} → 写入 {written} 个张量")
        gc.collect()
        mx.clear_cache()

    _validate(role, expected, loaded)
    # 写进去的张量都已 mx.eval；这里再走一次只是让结构完全物化，不产生新计算
    mx.eval(module.parameters())
    return LoadedH3Component(
        role=role,
        module=module,
        bits=bits,
        dtype=dtype_name,
        path=str(path),
        shapes={key: tuple(value.shape) for key, value in tree_flatten(module.parameters())},
    )


# --- 内部实现 ---
def _shard_files(role: str, path: str) -> list[str]:
    """组件目录下的权重分片（条件编码器按 index.json 只取需要的那些）。"""
    root = Path(path)
    if root.is_file():
        return [str(root)]
    if not root.is_dir():
        raise FileNotFoundError(f"未找到 {role} 权重目录: {root}")
    if role == "text_encoder":
        needed = _text_encoder_shards(root)
        if needed:
            return needed
    shards = sorted(str(item) for item in root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"{root} 下没有 .safetensors（{role}）")
    return shards


def _text_encoder_shards(root: Path) -> list[str] | None:
    """只读含「embed_tokens / 前 50 层 / 视觉塔」的 shard；其余 Qwen3-VL 的层永不读盘。"""
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return None
    try:
        weight_map = (json.loads(index.read_text()) or {}).get("weight_map") or {}
    except (json.JSONDecodeError, OSError):
        return None
    files = sorted(
        {name for key, name in weight_map.items() if MiniMaxH3WeightMapping.text_encoder_key_is_needed(key)}
    )
    found = [str(root / name) for name in files if (root / name).is_file()]
    return found or None


def _load_shard(
    module: nn.Module,
    role: str,
    shard: str,
    base_dtype: mx.Dtype,
    bits: int | None,
    quantized_modules: dict[str, Any],
    expected: dict[str, tuple[int, ...]],
    loaded: dict[str, tuple[int, ...]],
    log: Callable[[str], None] | None,
) -> int:
    """读一个 shard：过滤 → 改名 / 变换 → 精度转换 →（量化）写入模块，返回写入的张量数。"""
    definition = h3def.MiniMaxH3WeightDefinition.component(role)
    raw = dict(mx.load(shard).items())
    if definition.weight_prefix_filters:
        prefixes = tuple(definition.weight_prefix_filters)
        raw = {key: value for key, value in raw.items() if key.startswith(prefixes)}
        if not raw:
            raise ValueError(
                f"{role} 的 {Path(shard).name} 里没有任何以 {' / '.join(prefixes)} 开头的键："
                "这份权重的键名和 H3 的映射对不上（社区那份 NVFP4 用 model.layers.* / "
                "visual.*，映射要的是 model.language_model.* / model.visual.*）。"
                "请改选官方 HF 版的组件目录：text_encoder 指到 …/MiniMax-H3/text_encoder_back"
            )
    if not raw:  # 无 prefix 过滤的组件（transformer / vae）遇到空 shard 直接跳过
        return 0

    if definition.mapping_getter is not None:
        nested = _weight_mapper().apply_mapping(
            hf_weights=raw,
            mapping=definition.mapping_getter(),
            num_blocks=definition.num_blocks,
            num_layers=definition.num_layers,
        )
        mapped = dict(tree_flatten(nested))
    elif definition.bulk_transform is not None:
        mapped = {key: definition.bulk_transform(value) for key, value in raw.items()}
    else:
        mapped = raw
    if role == "audio_vae":
        # torch 的 weight_norm 折叠 + 1D 核转 MLX 布局，必须在写入之前做完
        mapped = MiniMaxH3WeightMapping.convert_audio_vae_state(mapped)

    flat: list[tuple[str, mx.array]] = []
    skipped = 0
    for key, tensor in mapped.items():
        if key not in expected:
            # mapping 之外的键（例如 Qwen3-VL 第 50 层以后）在模块里没有对应参数
            skipped += 1
            continue
        tensor = _to_dtype(role, key, tensor, base_dtype)
        flat.extend(_quantize_entry(key, tensor, bits, quantized_modules))
        loaded[key] = tuple(tensor.shape)
    if skipped and log:
        log(f"[H3 加载] {role}: 跳过 {skipped} 个模块里没有的键")
    if not flat:
        return 0
    module.update(tree_unflatten(flat), strict=False)
    mx.eval(*[value for _, value in flat])
    count = len(flat)
    del raw, mapped, flat
    return count


def _to_dtype(role: str, key: str, tensor: mx.array, base_dtype: mx.Dtype) -> mx.array:
    """按组件精度转换；transformer 的 fp32 保留集（输入/输出头、时间步 MLP）保持 float32。"""
    if tensor.dtype not in _FLOAT_DTYPES:
        return tensor
    if role == "transformer" and h3def.MiniMaxH3WeightDefinition.is_transformer_fp32_path(key):
        return tensor.astype(mx.float32)
    return tensor.astype(base_dtype)


def _quantize_entry(
    key: str, tensor: mx.array, bits: int | None, quantized_modules: dict[str, Any]
) -> list[tuple[str, mx.array]]:
    """量化模块的 `weight` 要连同 `scales` / `biases` 一起写；其余张量原样。"""
    parent, _, leaf = key.rpartition(".")
    module = quantized_modules.get(parent)
    if module is None or leaf != "weight":
        return [(key, tensor)]
    weight, scales, biases = mx.quantize(tensor, group_size=h3def.GROUP_SIZE, bits=bits)
    entries = [(key, weight), (f"{parent}.scales", scales)]
    if biases is not None:
        entries.append((f"{parent}.biases", biases))
    return entries


def _validate(role: str, expected: dict[str, tuple[int, ...]], loaded: dict[str, tuple[int, ...]]) -> None:
    """缺键 / 形状不符都要直接报错（`model.update(strict=False)` 不会替我们挡）。"""
    missing = sorted(set(expected) - set(loaded))
    mismatched = [
        (key, loaded[key], expected[key])
        for key in sorted(set(expected) & set(loaded))
        if tuple(loaded[key]) != tuple(expected[key])
    ]
    if not (missing or mismatched):
        return
    raise ValueError(
        f"MiniMax-H3 的 {role} 权重对不上模块结构：缺 {len(missing)} 个 {missing[:3]}；"
        f"形状不符 {len(mismatched)} 个 {mismatched[:3]}；"
        "请确认这份权重就是 MiniMax-H3（键名与 config.json 要与发布版本一致）"
    )


def _weight_mapper() -> Any:
    """mflux 的 WeightTarget 展开器（只用来改名，不碰加载路径）。"""
    from comfyui_mlx_gen import runtime

    return runtime.import_object("mflux.models.common.weights.mapping.weight_mapper:WeightMapper")


