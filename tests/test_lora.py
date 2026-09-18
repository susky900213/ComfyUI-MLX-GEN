"""Transformer LoRA 接入的合成回归测试（不加载任何真实大模型）。

运行：
    /opt/anaconda3/envs/py313/bin/python tests/test_lora.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen import paths, pipeline, transformer_lora  # noqa: E402
from comfyui_mlx_gen.h3.weights.h3_lora import (  # noqa: E402
    decode_comfy_quantized_state_dict,
    dequantize_int8_tensorwise,
)
from comfyui_mlx_gen.h3.weights.h3_lora_mapping import MiniMaxH3LoRAMapping  # noqa: E402
from comfyui_mlx_gen.nodes.lora import MlxClipLoraApply, MlxModelLoraApply  # noqa: E402
from comfyui_mlx_gen.types import LoraRef, MlxClipHandle, MlxModelHandle, entry_for  # noqa: E402

FAILED: list[str] = []


def check(label, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {label}" + (f" | {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def check_raises(label, exc_type, message_part, call):
    try:
        call()
    except exc_type as exc:
        check(label, message_part in str(exc), str(exc))
    except Exception as exc:  # noqa: BLE001
        check(label, False, f"异常类型不对：{type(exc).__name__}: {exc}")
    else:
        check(label, False, f"没有抛出 {exc_type.__name__}")


# ---------------------------------------------------------------- 1. 支持矩阵和正式 mapping
expected_mapping_names = {
    "z_image": "ZImageLoRAMapping",
    "flux2": "Flux2LoRAMapping",
    "qwen_image": "QwenLoRAMapping",
    "qwen_edit": "QwenLoRAMapping",
    "ideogram4": "Ideogram4LoRAMapping",
}
for family, class_name in expected_mapping_names.items():
    mapping = transformer_lora.mapping_for_family(family)
    check(
        f"{family} 使用 mflux {class_name}",
        bool(mapping) and type(mapping[0]).__name__ == "LoRATarget",
        f"targets={len(mapping)}",
    )
    patterns = [pattern for target in mapping for pattern in target.possible_down_patterns]
    check(
        f"{family} mapping 补齐标准 PEFT base_model/default 别名",
        any(pattern.startswith("base_model.model.") for pattern in patterns)
        and any(".lora_A.default.weight" in pattern for pattern in patterns),
    )
check(
    "支持矩阵包含全部图片家族和 MiniMax-H3",
    set(transformer_lora.supported_families()) == {*expected_mapping_names, "minimax_h3"},
    str(transformer_lora.supported_families()),
)
check_raises(
    "音频家族 LoRA 明确拒绝",
    ValueError,
    "尚不支持 Transformer LoRA",
    lambda: transformer_lora.mapping_for_family("yue2"),
)


# ------------------------------------------------------------- 2. 图片调用参数与相对路径
class FakeLoader:
    call = None

    @staticmethod
    def load_and_apply_lora(**kwargs):
        FakeLoader.call = kwargs
        return kwargs["lora_paths"], kwargs["lora_scales"]


class EmptyTransformer(nn.Module):
    pass


with tempfile.TemporaryDirectory() as temporary:
    old_root = paths.MODEL_ROOT
    paths.MODEL_ROOT = Path(temporary)
    (paths.MODEL_ROOT / "lora").mkdir()
    lora_file = paths.MODEL_ROOT / "lora" / "style.safetensors"
    mx.save_safetensors(
        str(lora_file),
        {
            "transformer.layers.0.attention.to_q.lora_A.weight": mx.zeros((2, 4)),
            "transformer.layers.0.attention.to_q.lora_B.weight": mx.zeros((4, 2)),
        },
    )
    original_mapping = transformer_lora.mapping_for_family
    original_import = transformer_lora.runtime.import_object
    transformer_lora.mapping_for_family = lambda family: [object()]

    def import_with_fake_loader(spec):
        if spec.endswith(":LoRALoader"):
            return FakeLoader
        return original_import(spec)

    transformer_lora.runtime.import_object = import_with_fake_loader
    try:
        result = transformer_lora.apply_transformer_loras(
            "z_image",
            EmptyTransformer(),
            (LoraRef("style.safetensors", 0.75),),
            role="transformer",
        )
    finally:
        transformer_lora.mapping_for_family = original_mapping
        transformer_lora.runtime.import_object = original_import
        paths.MODEL_ROOT = old_root

check(
    "图片 LoRA 解析绝对路径并以非 bake 模式调用 mflux",
    result == ([str(lora_file.absolute())], [0.75])
    and FakeLoader.call["bake_lora"] is False
    and FakeLoader.call["role"] == "transformer",
    repr(FakeLoader.call),
)

with tempfile.TemporaryDirectory() as temporary:
    old_root = paths.MODEL_ROOT
    paths.MODEL_ROOT = Path(temporary)
    lora_root = paths.MODEL_ROOT / "lora"
    blob_root = paths.MODEL_ROOT / "hub" / "blobs"
    snapshot_root = paths.MODEL_ROOT / "hub" / "snapshots" / "revision"
    lora_root.mkdir()
    blob_root.mkdir(parents=True)
    snapshot_root.mkdir(parents=True)
    blob = blob_root / "0123456789abcdef"
    mx.save_safetensors(str(blob_root / "source.safetensors"), {"weight": mx.ones((1,))})
    (blob_root / "source.safetensors").rename(blob)
    snapshot_link = snapshot_root / "linked.safetensors"
    snapshot_link.symlink_to(blob)
    selected_link = lora_root / "linked.safetensors"
    selected_link.symlink_to(snapshot_link)
    try:
        linked_result = transformer_lora.resolve_lora_path("linked.safetensors")
        linked_weights = mx.load(linked_result)
        mx.eval(linked_weights["weight"])
        linked_is_symlink = Path(linked_result).is_symlink()
    finally:
        paths.MODEL_ROOT = old_root

check(
    "LoRA 路径保留 .safetensors 符号链接，不解引用为 HF 无扩展名 blob",
    linked_result == str(selected_link.absolute())
    and linked_is_symlink
    and "weight" in linked_weights,
    linked_result,
)


# --------------------------------------------------------------- 3. H3 映射的布局变换
fused = mx.arange(18, dtype=mx.float32).reshape(6, 3)
thirds = [MiniMaxH3LoRAMapping.qkv_rows_third(i)(fused) for i in range(3)]
check(
    "H3 融合 QKV 的 lora_B 按输出行三等分",
    all(np.array_equal(np.asarray(value), np.asarray(fused[i * 2 : (i + 1) * 2])) for i, value in enumerate(thirds)),
)
swiglu = mx.arange(12, dtype=mx.float32).reshape(4, 3)
swapped = MiniMaxH3LoRAMapping.swap_row_halves(swiglu)
check(
    "H3 原始 [gate; value] lora_B 转为 [value; gate]",
    np.array_equal(np.asarray(swapped), np.asarray(mx.concatenate([swiglu[2:], swiglu[:2]], axis=0))),
)
peft_key = "base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight"
normalized = MiniMaxH3LoRAMapping.normalize_state_dict({peft_key: mx.zeros((2, 4))})
check(
    "H3 在 mflux 0.19.1 上补齐 PEFT 前缀/default 中缀规范化",
    set(normalized) == {"transformer_blocks.0.attn.to_q.lora_A.weight"},
    str(normalized.keys()),
)


# --------------------------------------------------------- 4. Comfy INT8-ConvRot 黄金值
qdata = mx.array(
    [[1, 2, 3, 4, -1, -2, -3, -4], [4, 3, 2, 1, -4, -3, -2, -1]],
    dtype=mx.int8,
)
scales = mx.array([[0.5], [0.25]], dtype=mx.float32)
h4 = np.asarray(
    [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
    dtype=np.float32,
) / 2.0
plain = np.asarray(qdata, dtype=np.float32) * np.asarray(scales)
expected = (plain.reshape(2, 2, 4) @ h4.T).reshape(2, 8)
actual = dequantize_int8_tensorwise(
    qdata, scales, convrot=True, convrot_groupsize=4
)
actual_fp32 = np.asarray(actual.astype(mx.float32))
check(
    "int8_tensorwise 先乘逐行 scale，再按 regular Hadamard 旋回",
    np.allclose(actual_fp32, expected, atol=2e-2, rtol=2e-2),
    f"max_err={np.max(np.abs(actual_fp32 - expected))}",
)

config = mx.array(
    list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}).encode()),
    dtype=mx.uint8,
)
base = "diffusion_model.blocks.0.attn.out_proj.lora_A"
decoded, count = decode_comfy_quantized_state_dict(
    {
        f"{base}.comfy_quant": config,
        f"{base}.weight": qdata,
        f"{base}.weight_scale": scales,
        "diffusion_model.blocks.0.attn.out_proj.alpha": mx.array(2.0),
    }
)
check(
    "comfy_quant 三元组折叠为普通 weight 且保留 alpha",
    count == 1
    and set(decoded) == {f"{base}.weight", "diffusion_model.blocks.0.attn.out_proj.alpha"}
    and np.allclose(
        np.asarray(decoded[f"{base}.weight"].astype(mx.float32)),
        expected,
        atol=2e-2,
        rtol=2e-2,
    ),
    str(decoded.keys()),
)


# --------------------------------------------------------- 5. H3 小模型实际应用与多 LoRA
class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_out = [nn.Linear(4, 4, bias=False)]


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()


class TinyH3(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = [TinyBlock()]


with tempfile.TemporaryDirectory() as temporary:
    old_root = paths.MODEL_ROOT
    paths.MODEL_ROOT = Path(temporary)
    (paths.MODEL_ROOT / "lora").mkdir()
    path = paths.MODEL_ROOT / "lora" / "tiny-h3.safetensors"
    down = mx.array([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, -0.5]])
    up = mx.array([[0.2, 0.0], [0.0, 0.3], [0.1, 0.0], [0.0, -0.2]])
    mx.save_safetensors(
        str(path),
        {
            "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": down,
            "diffusion_model.blocks.0.attn.out_proj.lora_B.weight": up,
            "diffusion_model.blocks.0.attn.out_proj.alpha": mx.array(2.0),
        },
    )
    tiny = TinyH3()
    mx.eval(tiny.parameters())
    base_weight = mx.array(tiny.transformer_blocks[0].attn.to_out[0].weight)
    x = mx.array([[1.0, 2.0, 3.0, 4.0]])
    try:
        resolved, strengths = transformer_lora.apply_transformer_loras(
            "minimax_h3", tiny, (LoraRef("tiny-h3.safetensors", 0.5),)
        )
        output = tiny.transformer_blocks[0].attn.to_out[0](x)
    finally:
        paths.MODEL_ROOT = old_root
    expected_output = x @ base_weight.T + 0.5 * ((x @ down.T) @ up.T)

check(
    "H3 LoRA 真正包装目标 Linear 并按 strength/alpha 影响输出",
    resolved == [str(path.absolute())]
    and strengths == [0.5]
    and np.allclose(np.asarray(output), np.asarray(expected_output), atol=2e-3, rtol=2e-3),
    f"resolved={resolved}, strengths={strengths}",
)


class BadShapeLoader:
    @staticmethod
    def load_and_apply_lora(**kwargs):
        linear = kwargs["transformer"].transformer_blocks[0].attn.to_out[0]
        from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear

        wrapped = LoRALinear.from_linear(linear, r=2)
        wrapped.lora_A = mx.zeros((3, 2))  # base input=4，故意错一维
        wrapped.lora_B = mx.zeros((2, 4))
        kwargs["transformer"].transformer_blocks[0].attn.to_out[0] = wrapped
        return kwargs["lora_paths"], kwargs["lora_scales"]


with tempfile.TemporaryDirectory() as temporary:
    old_root = paths.MODEL_ROOT
    paths.MODEL_ROOT = Path(temporary)
    (paths.MODEL_ROOT / "lora").mkdir()
    bad_path = paths.MODEL_ROOT / "lora" / "bad.safetensors"
    mx.save_safetensors(
        str(bad_path),
        {
            "transformer.layers.0.attention.to_q.lora_A.weight": mx.zeros((2, 4)),
            "transformer.layers.0.attention.to_q.lora_B.weight": mx.zeros((4, 2)),
        },
    )
    original_mapping = transformer_lora.mapping_for_family
    original_import = transformer_lora.runtime.import_object
    transformer_lora.mapping_for_family = lambda family: [object()]

    def import_with_bad_loader(spec):
        if spec.endswith(":LoRALoader"):
            return BadShapeLoader
        return original_import(spec)

    transformer_lora.runtime.import_object = import_with_bad_loader
    try:
        check_raises(
            "LoRA A/B 与基础 Linear 形状不符时在采样前报错",
            RuntimeError,
            "形状与目标层",
            lambda: transformer_lora.apply_transformer_loras(
                "z_image", TinyH3(), (LoraRef("bad.safetensors", 1.0),)
            ),
        )
    finally:
        transformer_lora.mapping_for_family = original_mapping
        transformer_lora.runtime.import_object = original_import
        paths.MODEL_ROOT = old_root


# ----------------------------------------------------- 6. strength=0、缓存隔离、CLIP 拒绝
check(
    "strength=0 被严格跳过",
    transformer_lora.active_loras((LoraRef("a.safetensors", 0.0),)) == (),
)
base_model = MlxModelHandle(
    model_type="z_image",
    model_path="dummy",
    quantize=8,
    precision="bfloat16",
    compile=False,
    compile_cache_limit=0,
    cache_key="base",
)
check(
    "节点首次登记 strength=0 直接复用基础 handle 与缓存键",
    MlxModelLoraApply().apply(base_model, "unused.safetensors", 0.0)[0] is base_model,
)
entry = entry_for("minimax_h3")
base_key = pipeline.h3_component_cache_key(entry, "transformer", "/model", 8, "bfloat16")
lora_key_1 = pipeline.h3_component_cache_key(
    entry, "transformer", "/model", 8, "bfloat16", (LoraRef("a", 1.0),)
)
lora_key_2 = pipeline.h3_component_cache_key(
    entry, "transformer", "/model", 8, "bfloat16", (LoraRef("a", 0.5),)
)
check(
    "H3 基础模型、LoRA 强度与组合使用隔离缓存键",
    len({base_key, lora_key_1, lora_key_2}) == 3,
)
clip = MlxClipHandle(
    model_type="z_image",
    component="text_encoder",
    source="local",
    path="dummy",
    precision="bfloat16",
)
check_raises(
    "CLIP LoRA 不再静默登记",
    NotImplementedError,
    "尚不支持",
    lambda: MlxClipLoraApply().apply(clip, "style.safetensors", 1.0),
)


if FAILED:
    raise SystemExit(f"\n{len(FAILED)} 项失败：{', '.join(FAILED)}")
print("\n全部 Transformer LoRA 合成检查通过。")