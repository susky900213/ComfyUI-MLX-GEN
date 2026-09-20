"""MiniMax-H3 PipeNetwork 预量化 checkpoint 适配的合成回归测试。

运行：
    /opt/anaconda3/envs/py313/bin/python tests/test_h3_pipenetwork.py

不读取 33B 权重；多 shard 用一个微型 QKV 模块和真实 safetensors 文件验证。
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

from comfyui_mlx_gen import paths  # noqa: E402
from comfyui_mlx_gen.h3 import config as h3_config  # noqa: E402
from comfyui_mlx_gen.h3.weights import loader  # noqa: E402
from comfyui_mlx_gen.h3.weights.pipenetwork_adapter import (  # noqa: E402
    adapt_transformer_state,
    reorder_mlp_gate_value,
    split_head_interleaved_qkv,
)

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


# --------------------------------------------------------- 1. Hugging Face cache 外壳解析
original_root = paths.MODEL_ROOT
with tempfile.TemporaryDirectory() as temporary:
    model_root = Path(temporary) / "models"
    component_root = model_root / "transformer"
    cache = Path(temporary) / "models--PipeNetwork--MiniMax-H3-MLX-8bit"
    revision = "0123456789abcdef"
    snapshot = cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text(revision, encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    component_root.mkdir(parents=True)
    (component_root / "h3-q8").symlink_to(cache, target_is_directory=True)
    paths.MODEL_ROOT = model_root
    try:
        kind, resolved = paths.resolve("local", "h3-q8", "transformer")
    finally:
        paths.MODEL_ROOT = original_root
    check(
        "HF cache 根目录经 refs/main 解析到 snapshot",
        kind == "dir" and resolved == str(snapshot.resolve()),
        f"{kind} / {resolved}",
    )

    broken = Path(temporary) / "models--PipeNetwork--broken"
    (broken / "snapshots").mkdir(parents=True)
    check_raises(
        "HF cache 缺 refs/main 时给出明确错误",
        FileNotFoundError,
        "refs/main",
        lambda: paths.normalize_hf_cache_path(broken),
    )


# --------------------------------------------------------- 2. PipeNetwork 配置别名与校验
pipe_config = {
    "hidden_size": 96,
    "num_layers": 2,
    "token_refiner_num_layers": 1,
    "num_attention_heads": 2,
    "attention_head_dim": 48,
    "ffn_hidden_size": 128,
    "latents_dim": 8,
    "audio_latents_dim": 4,
    "patch_size": [1, 2, 2],
    "text_dim": 64,
    "timestep_input_dim": 32,
    "time_embed_hidden_size": 64,
    "time_embed_dim": 32,
    "adaln_out_features": 1728,
    "final_adaln_out_features": 192,
    "rope_inv_freq_len": 8,
}
converted = h3_config.transformer_kwargs(pipe_config)
check(
    "上游 transformer 字段完整转换为本地构造参数",
    converted["num_refiner_layers"] == 1
    and converted["ffn_dim"] == 128
    and converted["in_channels"] == 8
    and converted["audio_in_channels"] == 4
    and converted["freq_dim"] == 32
    and converted["time_embed_hidden_dim"] == 64
    and converted["rope_freq_dim"] == 8
    and converted["patch_size"] == (1, 2, 2),
    str(converted),
)
bad_config = dict(pipe_config, final_adaln_out_features=191)
check_raises(
    "派生 AdaLN 宽度不一致时拒绝配置",
    ValueError,
    "final_adaln_out_features",
    lambda: h3_config.transformer_kwargs(bad_config),
)
with tempfile.TemporaryDirectory() as temporary:
    malformed_recipe = Path(temporary)
    (malformed_recipe / "quant_config.json").write_text(
        json.dumps({"bits": "8", "group_size": 32}),
        encoding="utf-8",
    )
    mx.save_safetensors(
        str(malformed_recipe / "model.safetensors"),
        {"synthetic.scales": mx.ones((1,), dtype=mx.float32)},
    )
    check_raises(
        "预量化 recipe 的位宽必须是 JSON 整数",
        ValueError,
        "JSON 整数",
        lambda: loader._prequantized_recipe("transformer", str(malformed_recipe)),
    )


# --------------------------------------------------------- 3. fused QKV / MLP 的量化张量布局
qkv = mx.arange(24 * 3).reshape(24, 3)
q, k, v = split_head_interleaved_qkv(qkv, num_heads=2)
check(
    "head-interleaved QKV 按每个 head 的 q/k/v 行拆分",
    np.array_equal(np.asarray(q)[:, 0], np.array([0, 3, 6, 9, 36, 39, 42, 45]))
    and np.array_equal(np.asarray(k)[:, 0], np.array([12, 15, 18, 21, 48, 51, 54, 57]))
    and np.array_equal(np.asarray(v)[:, 0], np.array([24, 27, 30, 33, 60, 63, 66, 69])),
)
mlp = mx.arange(12).reshape(12, 1)
check(
    "fused MLP 从 [gate; value] 换为 [value; gate]",
    np.array_equal(np.asarray(reorder_mlp_gate_value(mlp))[:, 0], np.r_[6:12, 0:6]),
)
adapted = adapt_transformer_state(
    {
        "blocks.0.attn.qkv_proj.scales": qkv,
        "blocks.0.mlp.fc1.biases": mlp,
        "token_refiner.blocks.0.attn.out_proj.weight": mx.ones((2, 2)),
        "final_layer.video_out.bias": mx.ones((2,)),
    },
    num_heads=2,
)
check(
    "weight/scales/biases 使用同一布局变换并映射到本地模块名",
    set(adapted)
    == {
        "transformer_blocks.0.attn.to_q.scales",
        "transformer_blocks.0.attn.to_k.scales",
        "transformer_blocks.0.attn.to_v.scales",
        "transformer_blocks.0.ff.net.0.proj.biases",
        "token_refiner.refiner_blocks.0.attn.to_out.0.weight",
        "proj_out.bias",
    },
    str(sorted(adapted)),
)


# --------------------------------------------------------- 4. core 与 AdaLN 使用各自 bit width
class TinyAttention(nn.Module):
    def __init__(self, outputs=32):
        super().__init__()
        self.to_q = nn.Linear(32, outputs, bias=False)
        self.to_k = nn.Linear(32, outputs, bias=False)
        self.to_v = nn.Linear(32, outputs, bias=False)
        self.to_out = [nn.Linear(outputs, 32, bias=False)]


class TinyProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 64, bias=False)


class TinyFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = [TinyProjection(), nn.Identity(), nn.Linear(32, 32, bias=False)]


class TinyAdaLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 64, bias=True)


class TinyBlock(nn.Module):
    def __init__(self, full=True):
        super().__init__()
        self.attn = TinyAttention()
        if full:
            self.ff = TinyFeedForward()
            self.adaln_proj = TinyAdaLN()


class TinyRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.refiner_blocks = [TinyBlock()]


class TinyMixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = [TinyBlock()]
        self.token_refiner = TinyRefiner()
        self.norm_out = TinyAdaLN()


mixed = TinyMixedModel()
loader._reconstruct_prequantized_modules(
    mixed,
    loader.PrequantizedRecipe(bits=4, group_size=32, quantize_adaln=True, adaln_bits=8),
)
check(
    "core q4、block AdaLN q8、final AdaLN 保持浮点",
    isinstance(mixed.transformer_blocks[0].attn.to_q, nn.QuantizedLinear)
    and mixed.transformer_blocks[0].attn.to_q.bits == 4
    and isinstance(mixed.token_refiner.refiner_blocks[0].ff.net[0].proj, nn.QuantizedLinear)
    and mixed.token_refiner.refiner_blocks[0].ff.net[0].proj.bits == 4
    and isinstance(mixed.transformer_blocks[0].adaln_proj.linear, nn.QuantizedLinear)
    and mixed.transformer_blocks[0].adaln_proj.linear.bits == 8
    and isinstance(mixed.norm_out.linear, nn.Linear),
)


# --------------------------------------------------------- 5. 两个 shard 增量加载且不二次量化
class TinyLoadBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyLoadAttention()


class TinyLoadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.to_q = nn.Linear(32, 32, bias=False)
        self.to_k = nn.Linear(32, 32, bias=False)
        self.to_v = nn.Linear(32, 32, bias=False)


class TinyLoadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = [TinyLoadBlock()]


with tempfile.TemporaryDirectory() as temporary:
    model_dir = Path(temporary)
    (model_dir / "quant_config.json").write_text(
        json.dumps({"bits": 8, "group_size": 32, "quantize_adaln": False, "adaln_bits": 8}),
        encoding="utf-8",
    )
    # 输出行是 [head0 q,k,v][head1 q,k,v]；每一行单独量化，不改变第一维布局。
    source = mx.arange(96 * 32, dtype=mx.float32).reshape(96, 32) / 100.0
    packed, scales, biases = mx.quantize(source, group_size=32, bits=8)
    mx.save_safetensors(
        str(model_dir / "model-00001-of-00002.safetensors"),
        {"blocks.0.attn.qkv_proj.weight": packed},
    )
    mx.save_safetensors(
        str(model_dir / "model-00002-of-00002.safetensors"),
        {
            "blocks.0.attn.qkv_proj.scales": scales,
            "blocks.0.attn.qkv_proj.biases": biases,
        },
    )

    original_build = loader.config.build_model
    original_quantize_entry = loader._quantize_entry
    loader.config.build_model = lambda role, path: TinyLoadModel()

    def fail_if_runtime_quantized(*args, **kwargs):
        raise AssertionError("预量化 checkpoint 不得进入 _quantize_entry")

    loader._quantize_entry = fail_if_runtime_quantized
    try:
        loaded = loader.load(
            "transformer", "dir", str(model_dir), quantize=4, precision="float32", log=None
        )
    finally:
        loader.config.build_model = original_build
        loader._quantize_entry = original_quantize_entry

    expected_q, expected_k, expected_v = split_head_interleaved_qkv(packed, 2)
    check(
        "预量化 recipe 覆盖 Loader 位宽且两个 shard 写入同一量化模块",
        loaded.bits == 8
        and loaded.module.transformer_blocks[0].attn.to_q.bits == 8
        and np.array_equal(
            np.asarray(loaded.module.transformer_blocks[0].attn.to_q.weight), np.asarray(expected_q)
        )
        and np.array_equal(
            np.asarray(loaded.module.transformer_blocks[0].attn.to_k.weight), np.asarray(expected_k)
        )
        and np.array_equal(
            np.asarray(loaded.module.transformer_blocks[0].attn.to_v.weight), np.asarray(expected_v)
        )
        and all(name in loaded.shapes for name in (
            "transformer_blocks.0.attn.to_q.scales",
            "transformer_blocks.0.attn.to_k.biases",
            "transformer_blocks.0.attn.to_v.scales",
        )),
        str(loaded.shapes),
    )


# --- 分块求值 / Metal 故障提示（整条 shard 一次 mx.eval 会被 macOS 的 GPU 看门狗掐掉）
chunk_probe = [("a", mx.array([1.0])), ("b", mx.array([2.0])), ("c", mx.array([3.0]))]
loader._eval_in_chunks(chunk_probe, 2)
check(
    "分块求值：小批 mx.eval 后张量已物化，块大小非法也不报错",
    loader.EVAL_CHUNK >= 1
    and float(chunk_probe[2][1][0]) == 3.0
    and loader._eval_in_chunks(chunk_probe, 0) is None,
    str([float(value[0]) for _, value in chunk_probe]),
)
metal_exc = RuntimeError(
    "[METAL] Command buffer execution failed: Ignored (for causing prior/excessive GPU errors) "
    "(00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)."
)
check(
    "Metal GPU 故障（超时 / 上下文已废）会被翻译成「重启 ComfyUI」的可操作提示",
    (loader._metal_fault_hint(metal_exc) or "").find("重启 ComfyUI") >= 0
    and loader._metal_fault_hint(RuntimeError(
        "Command buffer execution failed: Caused GPU Timeout Error"
        " (00000002:kIOGPUCommandBufferCallbackErrorTimeout)."
    )) is not None
    and loader._metal_fault_hint(ValueError("键名和 H3 的映射对不上")) is None,
)


if FAILED:
    raise SystemExit(f"\n{len(FAILED)} 项失败：{', '.join(FAILED)}")
print("\n全部 MiniMax-H3 PipeNetwork 合成检查通过。")