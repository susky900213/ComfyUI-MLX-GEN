"""Contract tests for the native MOSS-Image-SFT LoRA paths."""

from __future__ import annotations

import sys
import json
import struct
from pathlib import Path

import mlx.core as mx
from mlx import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen.moss import lora  # noqa: E402
from comfyui_mlx_gen.nodes.lora import MlxModelLoraApply, MlxMoss4StepLoraApply  # noqa: E402
from comfyui_mlx_gen.types import LoraRef, MlxModelHandle  # noqa: E402


LORA_PATH = Path.home() / "ComfyUI-Shared/models/mlx/lora/moss_image_sft_4_steps_lora_v2_600_avg_rank_72_bf16.safetensors"
STANDARD_LORA_PATH = Path.home() / "ComfyUI-Shared/models/mlx/lora/moss-sft-nsfw-pytorch_lora_weights.safetensors"


def _handle(*, loras=(), quantize=0, precision="bfloat16"):
    return MlxModelHandle(
        model_type="moss",
        model_path="OpenMOSS-Team--MOSS-Image-SFT",
        quantize=quantize,
        precision=precision,
        compile=False,
        compile_cache_limit=0,
        loras=tuple(loras),
        cache_key="test",
    )


def test_official_header_is_dynamic_rank_and_covers_all_moss_blocks():
    targets = lora.validate_header(LORA_PATH)
    assert len(targets) == 42 * 6
    assert {target.layer for target in targets.values()} == set(range(42))
    assert {target.projection for target in targets.values()} == set(lora.MOSS_LORA_PROJECTIONS)
    assert min(target.rank for target in targets.values()) < 72
    assert max(target.rank for target in targets.values()) > 72
    assert all(target.alpha is None for target in targets.values())


def test_standard_header_is_fixed_rank_and_covers_all_moss_blocks():
    targets = lora.validate_standard_header(STANDARD_LORA_PATH)
    assert len(targets) == 42 * 6
    assert {target.layer for target in targets.values()} == set(range(42))
    assert {target.projection for target in targets.values()} == set(lora.MOSS_LORA_PROJECTIONS)
    assert {target.rank for target in targets.values()} == {16}
    assert {target.alpha for target in targets.values()} == {16.0}
    assert targets[(0, "ffn_down")].down_shape == (16, 8192)
    assert targets[(0, "ffn_down")].up_shape == (3072, 16)


def _write_header(path: Path, *, omit: str | None = None, bad_shape: str | None = None):
    tensors = {}
    for layer in range(42):
        for projection in lora.MOSS_LORA_PROJECTIONS:
            if projection == "ffn_down":
                input_dim, output_dim = 8192, 3072
            elif projection == "ffn_up_gate":
                input_dim, output_dim = 3072, 16384
            else:
                input_dim = output_dim = 3072
            rank = 17 + (layer + len(projection)) % 5
            values = {
                "alpha": {"dtype": "BF16", "shape": [], "data_offsets": [0, 2]},
                "lora_down.weight": {
                    "dtype": "BF16",
                    "shape": [rank, input_dim],
                    "data_offsets": [0, 2],
                },
                "lora_up.weight": {
                    "dtype": "BF16",
                    "shape": [output_dim, rank],
                    "data_offsets": [0, 2],
                },
            }
            for suffix, info in values.items():
                key = lora._target_key(layer, projection, suffix)
                if key != omit:
                    tensors[key] = info
    if bad_shape is not None:
        tensors[bad_shape]["shape"][0] += 1
    header = {
        "__metadata__": {"ss_network_dim": "Dynamic", "ss_network_alpha": "Dynamic"},
        **tensors,
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def test_header_rejects_incomplete_groups_and_wrong_dynamic_rank_shape(tmp_path):
    missing = tmp_path / "moss_image_4_steps_missing.safetensors"
    _write_header(missing, omit=lora._target_key(0, "q", "alpha"))
    try:
        lora.validate_header(missing)
    except ValueError as exc:
        assert "missing=" in str(exc)
    else:
        raise AssertionError("incomplete MOSS LoRA group was accepted")

    bad_shape = tmp_path / "moss_image_4_steps_bad.safetensors"
    bad_key = lora._target_key(0, "q", "lora_down.weight")
    _write_header(bad_shape, bad_shape=bad_key)
    try:
        lora.validate_header(bad_shape)
    except ValueError as exc:
        assert "shape 不匹配" in str(exc)
    else:
        raise AssertionError("invalid dynamic rank shape was accepted")


def test_lora_linear_keeps_base_and_adds_scaled_low_rank_branch():
    base = nn.Linear(3, 2, bias=False)
    base.weight = mx.zeros((2, 3), dtype=mx.float32)
    down = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=mx.float32)
    up = mx.array([[2.0, 0.0], [0.0, 3.0]], dtype=mx.float32)
    module = lora.MOSSLoRALinear(base, down, up, 0.5)
    output = module(mx.array([[4.0, 5.0, 9.0]], dtype=mx.float32))
    mx.eval(output)
    assert output.tolist() == [[4.0, 7.5]]


def test_sampling_contract_keeps_scheduler_and_guidance_constraints_only():
    plain = _handle()
    assert not lora.validate_sampling_request(plain, 50, 3.0, "moss_flow_match_euler")
    adapted = _handle(loras=(LoraRef(str(LORA_PATH), 1.0),))
    assert lora.validate_sampling_request(adapted, 4, 1.0, "moss_flow_match_euler")
    assert lora.validate_sampling_request(adapted, 50, 1.0, "moss_flow_match_euler")
    for steps, guidance, scheduler, message in (
        (4, 2.0, "moss_flow_match_euler", "CFG/guidance=1.0"),
        (4, 1.0, "Euler", "moss_flow_match_euler"),
    ):
        try:
            lora.validate_sampling_request(adapted, steps, guidance, scheduler)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("invalid MOSS 4-step sampling settings were accepted")


def test_moss_adapter_rejects_non_unit_strength_quantization_and_generic_node():
    with_strength = _handle(loras=(LoraRef(str(LORA_PATH), 0.5),))
    try:
        lora.validate_sampling_request(with_strength, 4, 1.0, "moss_flow_match_euler")
    except ValueError as exc:
        assert "强度固定为 1.0" in str(exc)
    else:
        raise AssertionError("non-unit MOSS LoRA strength was accepted")

    quantized = _handle(quantize=8)
    try:
        MlxMoss4StepLoraApply().apply(quantized, str(LORA_PATH))
    except ValueError as exc:
        assert "未量化 dense" in str(exc)
    else:
        raise AssertionError("quantized MOSS transformer accepted 4-step LoRA")

    try:
        MlxModelLoraApply().apply(_handle(), str(LORA_PATH), 1.0)
    except ValueError as exc:
        assert "不能接通用" in str(exc)
    else:
        raise AssertionError("generic MOSS LoRA node accepted the adapter")


def test_standard_moss_lora_uses_generic_node_and_allows_strength():
    model = _handle()
    adapted, = MlxModelLoraApply().apply(model, str(STANDARD_LORA_PATH), 0.5)
    assert adapted.loras == (LoraRef(str(STANDARD_LORA_PATH), 0.5),)
    assert lora.validate_sampling_request(adapted, 20, 1.0, "moss_flow_match_euler")


def test_standard_moss_lora_still_rejects_quantized_transformer():
    quantized = _handle(quantize=8)
    try:
        MlxModelLoraApply().apply(quantized, str(STANDARD_LORA_PATH), 1.0)
    except ValueError as exc:
        assert "未量化 dense" in str(exc)
    else:
        raise AssertionError("quantized MOSS PEFT LoRA was accepted")


def test_moss_lora_nodes_reject_mixing_four_step_and_standard_adapters():
    four_step = _handle(loras=(LoraRef(str(LORA_PATH), 1.0),))
    try:
        MlxModelLoraApply().apply(four_step, str(STANDARD_LORA_PATH), 1.0)
    except ValueError as exc:
        assert "4-step LoRA 节点和普通 PEFT LoRA 节点之间二选一" in str(exc)
    else:
        raise AssertionError("MOSS 4-step and standard PEFT LoRAs were mixed")

    standard = _handle(loras=(LoraRef(str(STANDARD_LORA_PATH), 0.5),))
    try:
        MlxMoss4StepLoraApply().apply(standard, str(LORA_PATH))
    except ValueError as exc:
        assert "4-step LoRA 节点和普通 PEFT LoRA 节点之间二选一" in str(exc)
    else:
        raise AssertionError("MOSS standard PEFT and 4-step LoRAs were mixed")


def test_moss_lora_node_allows_idempotent_same_selection():
    model = _handle(loras=(LoraRef(str(STANDARD_LORA_PATH), 0.5),))
    adapted, = MlxModelLoraApply().apply(model, str(STANDARD_LORA_PATH), 0.5)
    assert adapted is model