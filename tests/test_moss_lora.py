"""Contract tests for the native MOSS-Image-SFT LoRA paths."""

from __future__ import annotations

import sys
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen.moss import lora  # noqa: E402
from comfyui_mlx_gen.nodes.lora import MlxModelLoraApply, MlxMoss4StepLoraApply  # noqa: E402
from comfyui_mlx_gen.nodes.moss_image_edit_lora import (  # noqa: E402
    MlxMossImageEditLoraApply,
    scan_moss_image_edit_loras,
)
from comfyui_mlx_gen.types import LoraRef, MlxModelHandle  # noqa: E402


LORA_PATH = Path.home() / "ComfyUI-Shared/models/mlx/lora/moss_image_sft_4_steps_lora_v2_600_avg_rank_72_bf16.safetensors"
STANDARD_LORA_PATH = Path.home() / "ComfyUI-Shared/models/mlx/lora/moss-sft-nsfw-pytorch_lora_weights.safetensors"
IMAGE_EDIT_LORA_DIR = Path.home() / "ComfyUI-Shared/models/mlx/lora/moss_image_edit_lora"
IMAGE_EDIT_LORA_PATH = IMAGE_EDIT_LORA_DIR / "adapter_model.safetensors"


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


def test_image_edit_header_uses_directory_config_and_covers_all_targets():
    targets = lora.validate_image_edit_header(IMAGE_EDIT_LORA_PATH)
    assert len(targets) == 42 * 6
    assert {target.layer for target in targets.values()} == set(range(42))
    assert {target.projection for target in targets.values()} == set(lora.MOSS_LORA_PROJECTIONS)
    assert {target.rank for target in targets.values()} == {64}
    assert {target.alpha for target in targets.values()} == {64.0}
    assert targets[(0, "ffn_down")].down_shape == (64, 8192)
    assert targets[(0, "ffn_down")].up_shape == (3072, 64)


def test_image_edit_scan_only_returns_valid_peft_directories():
    assert "moss_image_edit_lora" in scan_moss_image_edit_loras()


def test_image_edit_node_registers_adapter_file_and_is_idempotent():
    model = _handle()
    node = MlxMossImageEditLoraApply()
    adapted, = node.apply(model, "moss_image_edit_lora", 0.75)
    assert adapted.loras == (LoraRef(str(IMAGE_EDIT_LORA_PATH.absolute()), 0.75),)
    repeated, = node.apply(adapted, "moss_image_edit_lora", 0.75)
    assert repeated is adapted


def test_image_edit_adapter_is_not_accepted_by_the_shared_model_lora_node():
    try:
        MlxModelLoraApply().apply(_handle(), str(IMAGE_EDIT_LORA_PATH), 1.0)
    except ValueError as exc:
        assert "MOSS Image Edit LoRA" in str(exc)
    else:
        raise AssertionError("image-edit MOSS LoRA was accepted by the shared model-LoRA node")


def test_standard_key_normalization_accepts_peft_prefix_default_and_weight_variants():
    variants = (
        "layers.0.proj.q.lora_A",
        "layers.0.proj.q.lora_B.weight",
        "transformer.layers.1.proj.k.lora_A.weight",
        "base_model.model.layers.2.proj.v.lora_B.default",
        "base_model.model.transformer.layers.3.proj.o.lora_A.default.weight",
        "diffusion_model.layers.4.proj.ffn_down.lora_B.weight",
    )
    normalized = lora._standard_key_map({key: {} for key in variants})
    assert set(normalized) == {
        (0, "q", "A"),
        (0, "q", "B"),
        (1, "k", "A"),
        (2, "v", "B"),
        (3, "o", "A"),
        (4, "ffn_down", "B"),
    }


def test_unprefixed_peft_payload_is_injected(monkeypatch):
    base = nn.Linear(3, 2, bias=False)
    base.weight = mx.zeros((2, 3), dtype=mx.float32)
    transformer = SimpleNamespace(
        layers=[SimpleNamespace(proj={"q": base})],
        parameters=lambda: {},
    )
    down = mx.array([[1.0, 0.0, 0.0]], dtype=mx.float32)
    up = mx.array([[2.0], [0.0]], dtype=mx.float32)
    monkeypatch.setattr(lora, "MOSS_LORA_LAYERS", 1)
    monkeypatch.setattr(lora, "MOSS_LORA_PROJECTIONS", ("q",))
    monkeypatch.setattr(lora, "_expected_shapes", lambda projection: (3, 2))
    monkeypatch.setattr(lora, "resolve_lora_path", lambda path: path)
    monkeypatch.setattr(lora.mx, "load", lambda path, format=None: {
        "layers.0.proj.q.lora_A": down,
        "layers.0.proj.q.lora_B": up,
    })
    path = "image_edit/adapter_model.safetensors"
    monkeypatch.setattr(
        lora,
        "_header",
        lambda ignored: (
            {
                "layers.0.proj.q.lora_A": {"dtype": "BF16", "shape": [1, 3]},
                "layers.0.proj.q.lora_B": {"dtype": "BF16", "shape": [2, 1]},
            },
            {"lora_adapter_metadata": '{"r": 1, "lora_alpha": 1}'},
        ),
    )
    model = _handle(loras=(LoraRef(path, 1.0),))

    lora.apply_lora(transformer, model)
    output = transformer.layers[0].proj["q"](mx.array([[4.0, 5.0, 9.0]], dtype=mx.float32))
    mx.eval(output)
    assert output.tolist() == [[8.0, 0.0]]


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


def test_lora_linear_combines_multiple_low_rank_branches():
    base = nn.Linear(3, 2, bias=False)
    base.weight = mx.zeros((2, 3), dtype=mx.float32)
    first_down = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=mx.float32)
    first_up = mx.array([[2.0, 0.0], [0.0, 3.0]], dtype=mx.float32)
    second_down = mx.array([[0.0, 0.0, 1.0]], dtype=mx.float32)
    second_up = mx.array([[1.0], [1.0]], dtype=mx.float32)
    module = lora.MOSSLoRALinear(base, first_down, first_up, 0.5)
    assert module.add_branch(second_down, second_up, 0.25)
    output = module(mx.array([[4.0, 5.0, 9.0]], dtype=mx.float32))
    mx.eval(output)
    assert output.tolist() == [[6.25, 9.75]]


def test_lora_linear_deduplicates_repeated_adapter_key():
    base = nn.Linear(3, 2, bias=False)
    base.weight = mx.zeros((2, 3), dtype=mx.float32)
    down = mx.array([[1.0, 0.0, 0.0]], dtype=mx.float32)
    up = mx.array([[1.0], [1.0]], dtype=mx.float32)
    module = lora.MOSSLoRALinear(base, down, up, 0.5, adapter_key="adapter.safetensors")
    assert not module.add_branch(down, up, 0.5, adapter_key="adapter.safetensors")
    assert len(module.branches) == 1


def test_apply_lora_composes_standard_and_four_step_targets(monkeypatch):
    base = nn.Linear(3, 2, bias=False)
    base.weight = mx.zeros((2, 3), dtype=mx.float32)
    transformer = SimpleNamespace(
        layers=[SimpleNamespace(proj={"q": base})],
        parameters=lambda: {},
    )
    standard_target = lora.MOSSLoRATarget(0, "q", 1, 1.0, (1, 3), (2, 1))
    four_step_target = lora.MOSSLoRATarget(0, "q", 1, None, (1, 3), (2, 1))
    standard_down = mx.array([[1.0, 0.0, 0.0]], dtype=mx.float32)
    standard_up = mx.array([[1.0], [0.0]], dtype=mx.float32)
    four_step_down = mx.array([[0.0, 1.0, 0.0]], dtype=mx.float32)
    four_step_up = mx.array([[0.0], [1.0]], dtype=mx.float32)

    monkeypatch.setattr(lora, "MOSS_LORA_LAYERS", 1)
    monkeypatch.setattr(lora, "_expected_shapes", lambda projection: (3, 2))
    monkeypatch.setattr(lora, "resolve_lora_path", lambda path: path)
    monkeypatch.setattr(lora, "_is_standard_layout", lambda path: "standard" in path)
    monkeypatch.setattr(
        lora,
        "validate_standard_header",
        lambda path: {(0, "q"): standard_target},
    )
    monkeypatch.setattr(
        lora,
        "validate_header",
        lambda path: {(0, "q"): four_step_target},
    )

    def fake_load(path, format=None):
        if "standard" in path:
            return {
                lora._standard_target_key(0, "q", "A"): standard_down,
                lora._standard_target_key(0, "q", "B"): standard_up,
            }
        return {
            lora._target_key(0, "q", "lora_down.weight"): four_step_down,
            lora._target_key(0, "q", "lora_up.weight"): four_step_up,
            lora._target_key(0, "q", "alpha"): mx.array(1.0, dtype=mx.float32),
        }

    monkeypatch.setattr(lora.mx, "load", fake_load)
    model = _handle(
        loras=(
            LoraRef("four.safetensors", 1.0),
            LoraRef("standard.safetensors", 0.5),
        )
    )

    lora.apply_lora(transformer, model)
    wrapped = transformer.layers[0].proj["q"]
    assert isinstance(wrapped, lora.MOSSLoRALinear)
    assert len(wrapped.branches) == 2
    output = wrapped(mx.array([[2.0, 3.0, 4.0]], dtype=mx.float32))
    mx.eval(output)
    assert output.tolist() == [[1.0, 3.0]]

    # Re-materializing against the same module is idempotent at the injection
    # layer as well as at the model-handle node layer.
    lora.apply_lora(transformer, model)
    assert len(wrapped.branches) == 2


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


def test_moss_lora_nodes_allow_mixing_four_step_and_standard_adapters():
    four_step = _handle(loras=(LoraRef(str(LORA_PATH), 1.0),))
    mixed_from_four_step, = MlxModelLoraApply().apply(
        four_step, str(STANDARD_LORA_PATH), 0.5
    )
    assert mixed_from_four_step.loras == (
        LoraRef(str(LORA_PATH), 1.0),
        LoraRef(str(STANDARD_LORA_PATH), 0.5),
    )

    standard = _handle(loras=(LoraRef(str(STANDARD_LORA_PATH), 0.5),))
    mixed_from_standard, = MlxMoss4StepLoraApply().apply(standard, str(LORA_PATH))
    assert mixed_from_standard.loras == (
        LoraRef(str(STANDARD_LORA_PATH), 0.5),
        LoraRef(str(LORA_PATH), 1.0),
    )


def test_moss_sampling_accepts_mixed_standard_and_four_step_adapters():
    mixed = _handle(
        loras=(
            LoraRef(str(LORA_PATH), 1.0),
            LoraRef(str(STANDARD_LORA_PATH), 0.5),
        )
    )
    assert lora.validate_sampling_request(mixed, 4, 1.0, "moss_flow_match_euler")


def test_moss_lora_node_allows_idempotent_same_selection():
    model = _handle(loras=(LoraRef(str(STANDARD_LORA_PATH), 0.5),))
    adapted, = MlxModelLoraApply().apply(model, str(STANDARD_LORA_PATH), 0.5)
    assert adapted is model