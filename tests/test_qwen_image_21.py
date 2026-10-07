"""Qwen-Image 2.1 detection, architecture and tiny end-to-end tests.

The tests construct tiny random models and never load the official 30 GiB
checkpoint. Run directly (pytest is not required)::

    PYTHONPATH=src python tests/test_qwen_image_21.py
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen import paths, pipeline, weights  # noqa: E402
from comfyui_mlx_gen.nodes.clip_loader import (  # noqa: E402
    MlxClipLoader,
    normalize_max_length,
    normalize_precision,
)
from comfyui_mlx_gen.nodes.loader import MlxTransformerLoader  # noqa: E402
from comfyui_mlx_gen.nodes.sampler import MlxKSamplerMLX  # noqa: E402
from comfyui_mlx_gen.nodes.vae_loader import MlxVAELoader  # noqa: E402
from comfyui_mlx_gen.qwen_image_21 import sampling  # noqa: E402
from comfyui_mlx_gen.qwen_image_21 import loader as qwen21_loader  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from comfyui_mlx_gen.qwen_image_21.text_encoder import (  # noqa: E402
    QwenImage21TextEncoder,
    encode_prompt,
)
from comfyui_mlx_gen.qwen_image_21.transformer import (  # noqa: E402
    QwenImage21KVCache,
    QwenImage21Transformer,
)
from comfyui_mlx_gen.qwen_image_21.vae import QwenImage21VAE  # noqa: E402
from comfyui_mlx_gen.types import (  # noqa: E402
    QWEN_IMAGE_21_FAMILY,
    QWEN_IMAGE_21_MAX_LENGTH,
    MlxClipHandle,
    MlxConditioning,
    MlxModelHandle,
    MlxReferenceImages,
    detect_model_family,
    entry_for,
    model_types,
    validate_model_family,
)

QWEN_21_NAME = "Qwen-Image-2.1"
LEGACY_T2I_NAME = "qwen-image-2512-8bit"
LEGACY_EDIT_NAME = "qwen-image-edit-2511-8bit"


def _tiny_transformer(blocks: int = 1) -> QwenImage21Transformer:
    """Return a small model whose parameter tree still has the production schema."""
    return QwenImage21Transformer(
        in_channels=4,
        out_channels=4,
        num_layers=blocks,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=8,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )


def _source_key(local_key: str) -> str:
    prefix = "model.diffusion_model."
    key = local_key
    if key.endswith(".img_mlp.proj.weight"):
        key = key[: -len(".proj.weight")] + ".net.0.proj.weight"
    elif key.endswith(".img_mlp.out.weight"):
        key = key[: -len(".out.weight")] + ".net.2.weight"
    return prefix + key


def _bf16_bytes(tensor: mx.array) -> bytes:
    """Safetensors stores BF16 as its raw two-byte representation."""
    float32 = np.asarray(tensor.astype(mx.float32))
    bits = float32.view(np.uint32) >> 16
    return np.asarray(bits, dtype="<u2").tobytes()


def _write_safetensors(
    path: Path,
    tensors: dict[str, mx.array],
    *,
    metadata: dict[str, str] | None = None,
    dtype: str = "BF16",
    offsets: dict[str, list[int]] | None = None,
    payload: bytes | None = None,
    header_override: dict | None = None,
) -> dict:
    """Write a deliberately small safetensors file for loader validation tests."""
    if offsets is None and payload is None and header_override is None and dtype == "BF16":
        arrays = {key: value.astype(mx.bfloat16) for key, value in tensors.items()}
        mx.save_safetensors(str(path), arrays, metadata=metadata)
        return qwen21_loader._read_safetensors_header(path)[0]
    raw_payload = bytearray()
    header: dict[str, object] = {}
    for key, tensor in tensors.items():
        value = _bf16_bytes(tensor)
        start = len(raw_payload)
        raw_payload.extend(value)
        header[key] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [start, len(raw_payload)],
        }
    if offsets:
        for key, value in offsets.items():
            header[key]["data_offsets"] = value  # type: ignore[index]
    if metadata is not None:
        header["__metadata__"] = metadata
    if header_override is not None:
        header = header_override
    data = bytes(raw_payload) if payload is None else payload
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    return header


def _single_file_fixture(
    root: Path,
    *,
    blocks: int = 1,
    metadata: dict[str, str] | None = None,
    remove: str | None = None,
    extra: dict[str, mx.array] | None = None,
    overrides: dict[str, mx.array] | None = None,
) -> tuple[Path, QwenImage21Transformer, dict[str, mx.array], dict]:
    root.mkdir(parents=True, exist_ok=True)
    model = _tiny_transformer(blocks)
    local = dict(tree_flatten(model.parameters()))
    if overrides:
        local.update(overrides)
    source = {_source_key(key): value for key, value in local.items()}
    if remove is not None:
        del source[_source_key(remove)]
    if extra:
        source.update(extra)
    path = root / "transformer.safetensors"
    header = _write_safetensors(path, source, metadata=metadata or {"format": "mlx", "dtype": "BF16"})
    return path, model, local, header


def _fused_single_file_fixture(root: Path) -> tuple[Path, QwenImage21Transformer, dict[str, mx.array]]:
    """Build a single-file fixture using Turbo's fused SwiGLU input weights."""
    root.mkdir(parents=True, exist_ok=True)
    model = _tiny_transformer()
    local = dict(tree_flatten(model.parameters()))
    source: dict[str, mx.array] = {}
    for key, tensor in local.items():
        if key.endswith(".img_mlp.gate_layer.weight"):
            proj_key = key[: -len(".gate_layer.weight")] + ".proj.weight"
            fused = mx.concatenate([tensor, local[proj_key]], axis=0)
            source["model.diffusion_model." + key[: -len(".gate_layer.weight")] + ".gate_up.weight"] = fused
        elif key.endswith(".img_mlp.proj.weight"):
            continue
        else:
            source[_source_key(key)] = tensor
    path = root / "QwenImage_2.1_turbo_8Steps_bf16.safetensors"
    _write_safetensors(path, source, metadata={"format": "mlx", "dtype": "BF16"})
    return path, model, local


def _directory_fixture(root: Path, *, blocks: int = 1) -> tuple[Path, QwenImage21Transformer, dict[str, mx.array]]:
    model = _tiny_transformer(blocks)
    local = dict(tree_flatten(model.parameters()))
    root.mkdir()
    mx.save_safetensors(
        str(root / "000.safetensors"),
        {key: value.astype(mx.bfloat16) for key, value in local.items()},
        metadata={"format": "mlx", "dtype": "BF16"},
    )
    return root, model, local


def _rewrite_header(path: Path, update) -> dict:
    header, header_size = qwen21_loader._read_safetensors_header(path)
    raw = path.read_bytes()
    payload = raw[8 + header_size :]
    update(header)
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return header


def _expect_value_error(action, contains: str) -> None:
    try:
        action()
    except ValueError as exc:
        assert contains in str(exc), (contains, str(exc))
    else:
        raise AssertionError(f"expected ValueError containing {contains!r}")


def _load_single_with_model(path: Path, model: QwenImage21Transformer):
    original = qwen21_loader.build_model
    qwen21_loader.build_model = lambda role, _path: model
    try:
        return qwen21_loader.load("transformer", str(path), quantize=0, precision="bfloat16", log=None)
    finally:
        qwen21_loader.build_model = original


def _load_directory_with_model(path: Path, model: QwenImage21Transformer):
    original = qwen21_loader.build_model
    qwen21_loader.build_model = lambda role, _path: model
    try:
        return qwen21_loader.load("transformer", str(path), quantize=0, precision="bfloat16", log=None)
    finally:
        qwen21_loader.build_model = original


def test_model_root_uses_comfyui_registered_mlx_directory(tmp_path):
    registered_root = tmp_path / "shared-models" / "mlx"

    class FakeFolderPaths:
        models_dir = str(tmp_path / "unused-default-models")

        @staticmethod
        def get_folder_paths(folder_name):
            assert folder_name == "mlx"
            return [str(registered_root), str(tmp_path / "secondary-mlx")]

        @staticmethod
        def add_model_folder_path(*_args):
            raise AssertionError("已有 mlx 注册目录时不应再次注册")

    assert paths._model_root_from_comfyui(FakeFolderPaths) == registered_root.resolve()


def test_model_root_registers_shared_directory_when_mlx_is_not_configured(tmp_path):
    calls = []

    class FakeFolderPaths:
        @staticmethod
        def get_folder_paths(folder_name):
            assert folder_name == "mlx"
            raise KeyError(folder_name)

        @staticmethod
        def add_model_folder_path(*args):
            calls.append(args)

    expected = Path.home() / "ComfyUI-Shared" / "models" / "mlx"
    assert paths._model_root_from_comfyui(FakeFolderPaths) == expected
    assert calls == [("mlx", str(expected), True)]


def test_qwen_image_21_is_an_isolated_supported_family():
    assert QWEN_IMAGE_21_FAMILY in model_types()
    entry = entry_for(QWEN_IMAGE_21_FAMILY)
    assert entry.family == QWEN_IMAGE_21_FAMILY
    assert entry.supported is True
    assert set(entry.components) == {"transformer", "vae", "text_encoder", "tokenizer"}
    assert entry.weight_def == ""
    assert entry.default_steps == 40
    assert entry.default_scheduler == "flow_match_euler_discrete"
    assert entry.default_guidance == 1.0
    assert "legacy qwen_image" in entry.notes


def test_qwen_image_21_conditioning_length_uses_the_model_context_limit():
    precision_options = MlxClipLoader.INPUT_TYPES()["required"]["precision"][0]
    assert "MLX 16bit" in precision_options
    assert normalize_precision("MLX 16bit") == normalize_precision("float16") == "float16"

    max_length = MlxClipLoader.INPUT_TYPES()["required"]["max_length"][1]
    assert (
        max_length["default"]
        == max_length["max"]
        == QWEN_IMAGE_21_MAX_LENGTH
        == 262_144
    )

    clip = MlxClipLoader().load(
        QWEN_IMAGE_21_FAMILY,
        "text_encoder",
        QWEN_21_NAME,
        "MLX 16bit",
        QWEN_IMAGE_21_MAX_LENGTH,
        8,
    )[0]
    assert clip.precision == "float16"
    assert clip.max_length == QWEN_IMAGE_21_MAX_LENGTH
    assert normalize_max_length(QWEN_IMAGE_21_FAMILY, 512) == QWEN_IMAGE_21_MAX_LENGTH
    assert normalize_max_length(QWEN_IMAGE_21_FAMILY, 4096) == 4096
    assert normalize_max_length("qwen_image", 512) == 512

    legacy_clip = MlxClipLoader().load(
        QWEN_IMAGE_21_FAMILY,
        "text_encoder",
        QWEN_21_NAME,
        "bfloat16",
        512,
        8,
    )[0]
    assert legacy_clip.max_length == QWEN_IMAGE_21_MAX_LENGTH


def test_name_detection_prioritizes_qwen_image_21_and_keeps_legacy_families_separate():
    assert detect_model_family("Qwen/Qwen-Image-2.1") == QWEN_IMAGE_21_FAMILY
    assert detect_model_family("qwen_image_2_1-8bit") == QWEN_IMAGE_21_FAMILY
    assert detect_model_family("qwen-image-21-fp16") == QWEN_IMAGE_21_FAMILY
    assert detect_model_family(LEGACY_T2I_NAME) == "qwen_image"
    assert detect_model_family(LEGACY_EDIT_NAME) == "qwen_edit"
    assert detect_model_family("some-unrelated-checkpoint") is None


def test_config_detection_identifies_a_local_qwen_image_21_checkpoint(tmp_path):
    checkpoint = tmp_path / "opaque-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["QwenImage21Transformer"]}), encoding="utf-8"
    )
    assert detect_model_family(checkpoint) == QWEN_IMAGE_21_FAMILY


def test_validation_keeps_qwen_image_21_out_of_legacy_families():
    validate_model_family("qwen_image", LEGACY_T2I_NAME)
    validate_model_family("qwen_edit", LEGACY_EDIT_NAME)
    validate_model_family(QWEN_IMAGE_21_FAMILY, QWEN_21_NAME)
    try:
        validate_model_family("qwen_image", QWEN_21_NAME)
    except ValueError as exc:
        assert QWEN_IMAGE_21_FAMILY in str(exc)
        assert "不会把 Qwen-Image 2.1 路由到 legacy" in str(exc)
    else:
        raise AssertionError("legacy qwen_image accepted a Qwen-Image 2.1 checkpoint")


def test_qwen_image_21_requires_an_explicitly_detectable_checkpoint():
    try:
        validate_model_family(QWEN_IMAGE_21_FAMILY, "opaque-checkpoint")
    except ValueError as exc:
        assert "config.json" in str(exc) and "无法确认" in str(exc)
    else:
        raise AssertionError("qwen_image_21 accepted an unidentifiable checkpoint")


def test_legacy_model_config_lookup_retains_a_hard_qwen_image_21_boundary():
    try:
        weights.config_for_path(QWEN_21_NAME, "qwen_image")
    except NotImplementedError as exc:
        assert "独立" in str(exc) and "legacy" in str(exc)
    else:
        raise AssertionError("Qwen-Image 2.1 entered the legacy mflux ModelConfig registry")


def test_public_loaders_create_qwen_image_21_handles_without_loading_weights(tmp_path):
    old_root = paths.MODEL_ROOT
    paths.MODEL_ROOT = tmp_path
    try:
        for role in ("transformer", "text_encoder", "tokenizer", "vae"):
            (tmp_path / role / QWEN_21_NAME).mkdir(parents=True)
        model = MlxTransformerLoader().load(
            QWEN_IMAGE_21_FAMILY, QWEN_21_NAME, 8, "bfloat16", False, 0
        )[0]
        clip = MlxClipLoader().load(
            QWEN_IMAGE_21_FAMILY, "text_encoder", QWEN_21_NAME, "bfloat16", 512, 8
        )[0]
        vae = MlxVAELoader().load(
            QWEN_IMAGE_21_FAMILY, QWEN_21_NAME, "bfloat16", 0, "vae"
        )[0]
        assert (model.model_type, model.quantize) == (QWEN_IMAGE_21_FAMILY, 8)
        assert (clip.model_type, clip.quantize) == (QWEN_IMAGE_21_FAMILY, 8)
        assert (vae.model_type, vae.role) == (QWEN_IMAGE_21_FAMILY, "vae")
        kind, resolved = pipeline.component_path(
            entry_for(QWEN_IMAGE_21_FAMILY), "transformer", QWEN_21_NAME
        )
        assert kind == "dir" and Path(resolved).is_dir()
    finally:
        paths.MODEL_ROOT = old_root


def test_qwen21_single_file_bf16_loads_and_maps_each_feed_forward_weight_independently(tmp_path):
    model = _tiny_transformer()
    local = dict(tree_flatten(model.parameters()))
    selected = {
        "transformer_blocks.0.img_mlp.proj.weight": 1.0,
        "transformer_blocks.0.img_mlp.out.weight": 2.0,
        "transformer_blocks.0.img_mlp.gate_layer.weight": 3.0,
    }
    overrides = {
        key: mx.full(local[key].shape, value, dtype=mx.bfloat16)
        for key, value in selected.items()
    }
    path, _unused, _local, header = _single_file_fixture(tmp_path, overrides=overrides)
    assert header["__metadata__"] == {"dtype": "BF16", "format": "mlx"}
    assert all(
        header[_source_key(key)]["dtype"] == "BF16"
        for key in selected
    )

    loaded = _load_single_with_model(path, model)
    actual = dict(tree_flatten(loaded.module.parameters()))
    mx.eval(*(actual[key] for key in selected))
    assert loaded.bits is None
    assert loaded.dtype == "bfloat16"
    assert Path(loaded.path) == path
    for key, value in overrides.items():
        assert float(mx.max(mx.abs(actual[key] - value)).item()) == 0.0


def test_qwen21_single_file_splits_turbo_fused_gate_up_weight(tmp_path):
    path, model, local = _fused_single_file_fixture(tmp_path)
    gate_key = "transformer_blocks.0.img_mlp.gate_layer.weight"
    proj_key = "transformer_blocks.0.img_mlp.proj.weight"
    gate = mx.full(local[gate_key].shape, 3.0, dtype=mx.bfloat16)
    proj = mx.full(local[proj_key].shape, 5.0, dtype=mx.bfloat16)
    source_key = "model.diffusion_model." + gate_key[: -len("gate_layer.weight")] + "gate_up.weight"
    source = {source_key: mx.concatenate([gate, proj], axis=0)}
    # Rewrite only this block's fused input tensor while retaining every other
    # source tensor from the complete Turbo-shaped fixture.
    raw = path.read_bytes()
    header, header_size = qwen21_loader._read_safetensors_header(path)
    payload = raw[8 + header_size :]
    fused_key = next(key for key in header if key.endswith(".img_mlp.gate_up.weight"))
    start, end = header[fused_key]["data_offsets"]
    replacement = _bf16_bytes(source[fused_key])
    assert len(replacement) == end - start
    payload = payload[:start] + replacement + payload[end:]
    path.write_bytes(raw[: 8 + header_size] + payload)

    loaded = _load_single_with_model(path, model)
    actual = dict(tree_flatten(loaded.module.parameters()))
    mx.eval(actual[gate_key], actual[proj_key])
    assert float(mx.max(mx.abs(actual[gate_key] - gate)).item()) == 0.0
    assert float(mx.max(mx.abs(actual[proj_key] - proj)).item()) == 0.0


def test_qwen21_single_file_and_directory_loading_follow_dynamic_block_schema(tmp_path):
    single_path, single_model, expected, _header = _single_file_fixture(tmp_path / "single", blocks=2)
    loaded_single = _load_single_with_model(single_path, single_model)
    assert set(dict(tree_flatten(loaded_single.module.parameters()))) == set(expected)
    assert "transformer_blocks.1.img_mlp.gate_layer.weight" in expected

    directory, directory_model, directory_expected = _directory_fixture(tmp_path / "directory", blocks=2)
    loaded_directory = _load_directory_with_model(directory, directory_model)
    assert set(dict(tree_flatten(loaded_directory.module.parameters()))) == set(directory_expected)
    assert loaded_directory.path == str(directory)


def test_qwen21_single_file_accepts_an_extensionless_valid_safetensors_blob(tmp_path):
    source, model, _local, _header = _single_file_fixture(tmp_path / "source")
    blob = tmp_path / "blob"
    blob.write_bytes(source.read_bytes())

    loaded = _load_single_with_model(blob, model)

    assert loaded.path == str(blob)


def test_qwen21_single_file_rejects_missing_and_unexpected_keys(tmp_path):
    missing_path, missing_model, _local, _header = _single_file_fixture(
        tmp_path / "missing",
        remove="transformer_blocks.0.img_mlp.gate_layer.weight",
    )
    _expect_value_error(
        lambda: _load_single_with_model(missing_path, missing_model),
        "缺 1 个 transformer 参数",
    )

    extra_path, extra_model, _local, _header = _single_file_fixture(
        tmp_path / "extra",
        extra={"model.diffusion_model.unexpected.weight": mx.ones((1,), dtype=mx.bfloat16)},
    )
    _expect_value_error(
        lambda: _load_single_with_model(extra_path, extra_model),
        "本地模块没有的键",
    )


def test_qwen21_single_file_validates_metadata_dtype_offsets_overlap_bounds_and_size(tmp_path):
    malformed_metadata, metadata_model, _local, _header = _single_file_fixture(
        tmp_path / "metadata"
    )
    _rewrite_header(
        malformed_metadata,
        lambda header: header.__setitem__("__metadata__", {"format": 1}),
    )
    _expect_value_error(
        lambda: _load_single_with_model(malformed_metadata, metadata_model),
        "metadata 必须是 string object",
    )

    unsupported_dtype, dtype_model, _local, _header = _single_file_fixture(tmp_path / "dtype")
    first_key = next(key for key in _rewrite_header(unsupported_dtype, lambda _header: None) if key != "__metadata__")
    _rewrite_header(
        unsupported_dtype,
        lambda header: header[first_key].__setitem__("dtype", "F8"),
    )
    _expect_value_error(
        lambda: _load_single_with_model(unsupported_dtype, dtype_model),
        "只支持 F16/BF16/F32",
    )

    bad_size, size_model, _local, _header = _single_file_fixture(tmp_path / "size")
    first_key = next(key for key in _rewrite_header(bad_size, lambda _header: None) if key != "__metadata__")
    _rewrite_header(
        bad_size,
        lambda header: header[first_key].__setitem__(
            "data_offsets", [
                header[first_key]["data_offsets"][0],
                header[first_key]["data_offsets"][1] - 1,
            ]
        ),
    )
    _expect_value_error(lambda: _load_single_with_model(bad_size, size_model), "字节数不符")

    out_of_bounds, bounds_model, _local, _header = _single_file_fixture(tmp_path / "bounds")
    first_key = next(key for key in _rewrite_header(out_of_bounds, lambda _header: None) if key != "__metadata__")
    _rewrite_header(
        out_of_bounds,
        lambda header: header[first_key].__setitem__("data_offsets", [0, 10**9]),
    )
    _expect_value_error(lambda: _load_single_with_model(out_of_bounds, bounds_model), "header 非法")

    overlap, overlap_model, _local, _header = _single_file_fixture(tmp_path / "overlap")
    header = _rewrite_header(overlap, lambda _header: None)
    norm_keys = [
        key
        for key in header
        if key.endswith(".attn.norm_k.weight") or key.endswith(".attn.norm_q.weight")
    ]
    assert len(norm_keys) == 2
    first_range = header[norm_keys[0]]["data_offsets"]
    _rewrite_header(
        overlap,
        lambda current: current[norm_keys[1]].__setitem__("data_offsets", first_range),
    )
    _expect_value_error(lambda: _load_single_with_model(overlap, overlap_model), "data_offsets 重叠")


def test_qwen21_single_file_rejects_malformed_header_and_non_transformer_roles(tmp_path):
    empty = tmp_path / "empty"
    empty.touch()
    _expect_value_error(
        lambda: _load_single_with_model(empty, _tiny_transformer()),
        "无法读取 safetensors header",
    )

    non_safetensors = tmp_path / "not-a-safetensors-file"
    non_safetensors.write_bytes(b"not a safetensors file")
    _expect_value_error(
        lambda: _load_single_with_model(non_safetensors, _tiny_transformer()),
        "无法读取 safetensors header",
    )

    malformed = tmp_path / "malformed.safetensors"
    malformed.write_bytes(struct.pack("<Q", 1) + b"{" + b"\x00")
    model = _tiny_transformer()
    _expect_value_error(lambda: _load_single_with_model(malformed, model), "无法读取 safetensors header")

    valid, _model, _local, _header = _single_file_fixture(tmp_path / "valid")
    _expect_value_error(
        lambda: qwen21_loader.load("vae", str(valid), quantize=0, precision="bfloat16", log=None),
        "只支持 transformer role",
    )


def test_qwen21_pipeline_dispatches_single_transformer_and_falls_back_to_sibling_components(tmp_path):
    root = tmp_path / "Qwen-Image-2.1"
    transformer_dir = root / "transformer"
    transformer_dir.mkdir(parents=True)
    transformer = transformer_dir / "Qwen-Image-2.1.safetensors"
    transformer.write_bytes(b"fixture")
    (root / "vae").mkdir()

    entry = entry_for(QWEN_IMAGE_21_FAMILY)
    kind, resolved = pipeline.component_path(entry, "transformer", str(transformer))
    assert kind == "file" and Path(resolved).resolve() == transformer.resolve()
    kind, resolved = pipeline.component_path(entry, "vae", str(transformer))
    assert kind == "dir" and Path(resolved).resolve() == (root / "vae").resolve()

    (root / "vae").rmdir()
    kind, resolved = pipeline.component_path(entry, "vae", str(transformer))
    assert kind == "missing"
    assert Path(resolved) != transformer


def test_qwen21_pipeline_preserves_safetensors_suffix_for_hf_blob_symlink(tmp_path):
    root = tmp_path / "Qwen-Image-2.1"
    transformer_dir = root / "transformer"
    transformer_dir.mkdir(parents=True)
    blob = tmp_path / "blobs" / "c5b30c48037b085fee9a582835bd823509abee9b08754bf212cc4085278ec64f"
    blob.parent.mkdir()
    blob.write_bytes(b"fixture")
    transformer = transformer_dir / "QwenImage_2.1_turbo_8Steps_bf16.safetensors"
    transformer.symlink_to(blob)

    kind, resolved = pipeline.component_path(
        entry_for(QWEN_IMAGE_21_FAMILY), "transformer", str(transformer)
    )

    assert kind == "file"
    assert Path(resolved) == transformer
    assert Path(resolved).suffix == ".safetensors"
    assert Path(resolved).is_symlink()


def test_dynamic_schedule_and_latent_contract():
    sigmas = sampling.sigma_schedule(4, 256)
    assert sigmas.shape == (5,)
    assert abs(float(sigmas[0]) - 1.0) < 1e-6
    assert abs(float(sigmas[-2]) - 0.02) < 1e-5
    assert float(sigmas[-1]) == 0.0
    assert all(float(sigmas[i]) > float(sigmas[i + 1]) for i in range(4))
    latent = sampling.create_noise(7, 32, 32, dtype=mx.float32)
    assert latent.shape == (1, 4, 64)
    assert sampling.unpack_latents(latent, 32, 32).shape == (1, 64, 1, 2, 2)


def test_sampler_requires_the_same_qwen21_reference_on_both_condition_branches():
    clip = MlxClipHandle(
        model_type=QWEN_IMAGE_21_FAMILY,
        component="text_encoder",
        source="local",
        path=QWEN_21_NAME,
        precision="bfloat16",
        cache_key="clip:qwen21",
    )
    model = MlxModelHandle(
        model_type=QWEN_IMAGE_21_FAMILY,
        model_path=QWEN_21_NAME,
        quantize=8,
        precision="bfloat16",
        compile=False,
        compile_cache_limit=0,
    )
    reference = MlxReferenceImages(
        model_type=QWEN_IMAGE_21_FAMILY,
        vae_path=QWEN_21_NAME,
        count=1,
        height=256,
        width=256,
        cache_key="ref:qwen21",
        edit_kind="qwen_image_21",
        seq_len=256,
    )
    positive = MlxConditioning(clip, "edit", "positive")
    negative = MlxConditioning(clip, " ", "negative", ref_cache_key=reference.cache_key)
    try:
        MlxKSamplerMLX().sample(
            model,
            positive,
            negative,
            seed=0,
            steps=1,
            width=256,
            height=256,
            batch_size=1,
            guidance=1.0,
            scheduler="flow_match_euler_discrete",
            ref_images=reference,
        )
    except ValueError as exc:
        assert "正向条件" in str(exc) and "同一个 ref_images" in str(exc)
    else:
        raise AssertionError("Qwen-Image 2.1 sampler accepted mismatched visual/latent references")


def test_tiny_multimodal_prompt_has_one_slot_per_2x2_latent_block():
    class TinyTokenizer:
        specials = {
            "<|image_pad|>": 151655,
            "<|vision_start|>": 151652,
            "<|vision_end|>": 151653,
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
        }

        def __call__(self, text, add_special_tokens=False):
            ids = []
            cursor = 0
            ordered = sorted(self.specials, key=len, reverse=True)
            while cursor < len(text):
                match = next((token for token in ordered if text.startswith(token, cursor)), None)
                if match is not None:
                    ids.append(self.specials[match])
                    cursor += len(match)
                else:
                    ids.append(100 + ord(text[cursor]) % 100)
                    cursor += 1
            return {"input_ids": ids}

    encoder = QwenImage21TextEncoder(
        vocab_size=151936,
        hidden_size=128,
        num_hidden_layers=3,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        intermediate_size=256,
        mrope_section=(24, 20, 20),
        vision_config={
            "hidden_size": 16,
            "depth": 3,
            "num_heads": 4,
            "intermediate_size": 32,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 256,
            "out_hidden_size": 128,
            "deepstack_visual_indexes": (0, 1, 2),
        },
    )
    image = Image.new("RGB", (256, 256), (30, 60, 90))
    hidden, attention_mask, image_pad_mask = encode_prompt(
        encoder, TinyTokenizer(), "change the background", images=[image]
    )
    mx.eval(hidden, attention_mask, image_pad_mask)
    # 256/16 = 16 visual patches per side; Qwen3-VL merges 2x2 patches, yielding
    # 8x8=64 VLM slots. VAE latent is 16x16=256 tokens, exactly four per slot.
    assert int(mx.sum(image_pad_mask).item()) == 64
    assert hidden.shape[:2] == attention_mask.shape == image_pad_mask.shape
    assert hidden.shape[-1] == 128
    assert bool(mx.all(mx.isfinite(hidden)).item())


def test_prefix_cache_matches_full_tiny_transformer_forward():
    model = QwenImage21Transformer(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=8,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    latent = mx.random.normal((1, 4, 4))
    text = mx.random.normal((1, 5, 8))
    timestep = mx.array([0.8], dtype=mx.float32)
    cache = QwenImage21KVCache(2)
    prefill = model(latent, text, timestep, 2, 2, cache, "extract")
    cached = model(latent, text, timestep, 2, 2, cache, "cached")
    full = model(latent, text, timestep, 2, 2)
    mx.eval(prefill, cached, full)
    assert float(mx.max(mx.abs(prefill - cached)).item()) < 1e-5
    assert float(mx.max(mx.abs(prefill - full)).item()) < 1e-5


def test_multireference_edit_prefix_cache_matches_full_forward():
    model = QwenImage21Transformer(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=8,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    target = mx.random.normal((1, 4, 4))
    # Two references: 2x2 and 2x4 latent grids. Each VLM slot expands to 2x2
    # latent tokens, hence one + two True positions in the image-pad mask.
    references = mx.random.normal((1, 12, 4))
    text = mx.random.normal((1, 8, 8))
    image_pad_mask = mx.array([[False, True, False, False, True, True, False, False]])
    timestep = mx.array([0.8], dtype=mx.float32)
    kwargs = {
        "reference_latents": references,
        "reference_shapes": ((2, 2), (2, 4)),
        "image_pad_mask": image_pad_mask,
    }
    cache = QwenImage21KVCache(2)
    prefill = model(target, text, timestep, 2, 2, cache, "extract", **kwargs)
    cached = model(target, text, timestep, 2, 2, cache, "cached", **kwargs)
    full = model(target, text, timestep, 2, 2, **kwargs)
    mx.eval(prefill, cached, full)
    assert prefill.shape == target.shape
    assert float(mx.max(mx.abs(prefill - cached)).item()) < 1e-5
    assert float(mx.max(mx.abs(prefill - full)).item()) < 1e-5

    try:
        model(
            target,
            text,
            timestep,
            2,
            2,
            reference_latents=references,
            reference_shapes=((2, 2), (2, 4)),
            image_pad_mask=mx.zeros((1, 8), dtype=mx.bool_),
        )
    except ValueError as exc:
        assert "2×2" in str(exc)
    else:
        raise AssertionError("Qwen-Image 2.1 accepted reference latents without matching visual slots")


def test_tiny_rgba_vae_reference_encode_contract():
    vae = QwenImage21VAE(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=64,
        dim_mult=(1, 1, 1, 1, 1),
        num_res_blocks=1,
        temperal_downsample=(False, True, True, True),
        out_channels=4,
        in_channels=4,
        latents_mean=[0.0] * 64,
        latents_std=[1.0] * 64,
    )
    rgba = mx.random.uniform(low=-1.0, high=1.0, shape=(1, 32, 32, 4))
    latents = vae.encode(rgba)
    mx.eval(latents)
    assert latents.shape == (1, 64, 2, 2)
    assert bool(mx.all(mx.isfinite(latents)).item())


def _workflow(name: str) -> dict:
    return json.loads((ROOT / "workflows" / name).read_text(encoding="utf-8"))


def test_official_t2i_and_edit_workflows_keep_their_conditioning_paths_separate():
    t2i = _workflow("qwen-image-2.1.json")
    t2i_clip = next(node for node in t2i["nodes"] if node["type"] == "MlxClipLoader")
    assert t2i_clip["widgets_values"][4] == QWEN_IMAGE_21_MAX_LENGTH
    assert all(node["type"] != "MlxVAEEncoder" for node in t2i["nodes"])
    t2i_sampler = next(node for node in t2i["nodes"] if node["type"] == "MlxKSamplerMLX")
    assert all(item["name"] != "ref_images" for item in t2i_sampler["inputs"])
    assert t2i_sampler["widgets_values"][2:8] == [40, 1024, 1024, 1, 1.0, "flow_match_euler_discrete"]

    edit = _workflow("qwen-image-2.1-edit.json")
    edit_clip = next(node for node in edit["nodes"] if node["type"] == "MlxClipLoader")
    assert edit_clip["widgets_values"][4] == QWEN_IMAGE_21_MAX_LENGTH
    node_types = [node["type"] for node in edit["nodes"]]
    assert node_types.count("MlxVAEEncoder") == 1
    assert node_types.count("MlxTextEncoder") == 2
    assert "MlxQwenEditEncoder" not in node_types
    loaders = [
        node for node in edit["nodes"]
        if node["type"] in {"MlxClipLoader", "MlxTransformerLoader", "MlxVAELoader"}
    ]
    assert all(node["widgets_values"][0] == QWEN_IMAGE_21_FAMILY for node in loaders)

    encoder = next(node for node in edit["nodes"] if node["type"] == "MlxVAEEncoder")
    assert encoder["widgets_values"] == [1, "auto", 1024, 1024]
    ref_link_ids = set(encoder["outputs"][0]["links"])
    assert len(ref_link_ids) == 3
    nodes = {node["id"]: node for node in edit["nodes"]}
    destinations = {
        (nodes[target]["type"], nodes[target]["inputs"][slot]["name"])
        for link_id, _source, _source_slot, target, slot, _type in edit["links"]
        if link_id in ref_link_ids
    }
    assert destinations == {
        ("MlxTextEncoder", "ref_images"),
        ("MlxKSamplerMLX", "ref_images"),
    }
    assert sum(
        1
        for link_id, _source, _source_slot, target, slot, _type in edit["links"]
        if link_id in ref_link_ids
        and nodes[target]["type"] == "MlxTextEncoder"
        and nodes[target]["inputs"][slot]["name"] == "ref_images"
    ) == 2
    sampler = next(node for node in edit["nodes"] if node["type"] == "MlxKSamplerMLX")
    assert sampler["widgets_values"][2:8] == [40, 1024, 1024, 1, 1.0, "flow_match_euler_discrete"]

    multi = _workflow("qwen-image-2.1-edit-multi.json")
    multi_clip = next(node for node in multi["nodes"] if node["type"] == "MlxClipLoader")
    assert multi_clip["widgets_values"][4] == QWEN_IMAGE_21_MAX_LENGTH
    multi_types = [node["type"] for node in multi["nodes"]]
    assert multi_types.count("LoadImage") == 3
    assert multi_types.count("MlxRefImageSet") == 1
    assert multi_types.count("MlxVAEEncoder") == 1
    assert multi_types.count("MlxTextEncoder") == 2
    assert "BatchImagesNode" not in multi_types
    assert "MlxQwenEditEncoder" not in multi_types

    multi_loaders = [
        node for node in multi["nodes"]
        if node["type"] in {"MlxClipLoader", "MlxTransformerLoader", "MlxVAELoader"}
    ]
    assert all(node["widgets_values"][0] == QWEN_IMAGE_21_FAMILY for node in multi_loaders)
    assert all(
        node["widgets_values"][2 if node["type"] == "MlxClipLoader" else 1] == QWEN_21_NAME
        for node in multi_loaders
    )

    ref_set = next(node for node in multi["nodes"] if node["type"] == "MlxRefImageSet")
    assert [item["name"] for item in ref_set["inputs"]] == [
        f"image{index}" for index in range(1, 11)
    ]
    assert all(item["link"] is not None for item in ref_set["inputs"][:3])
    assert all(item["link"] is None for item in ref_set["inputs"][3:])

    multi_encoder = next(node for node in multi["nodes"] if node["type"] == "MlxVAEEncoder")
    assert multi_encoder["widgets_values"] == [10, "auto", 1024, 1024]
    encoder_inputs = {item["name"]: item["link"] for item in multi_encoder["inputs"]}
    assert encoder_inputs["images"] is None
    assert encoder_inputs["ref_source"] is not None
    multi_nodes = {node["id"]: node for node in multi["nodes"]}
    ref_destinations = [
        (multi_nodes[target]["type"], multi_nodes[target]["inputs"][slot]["name"])
        for link_id, _source, _source_slot, target, slot, _type in multi["links"]
        if link_id in set(multi_encoder["outputs"][0]["links"])
    ]
    assert ref_destinations.count(("MlxTextEncoder", "ref_images")) == 2
    assert ref_destinations.count(("MlxKSamplerMLX", "ref_images")) == 1
    multi_sampler = next(node for node in multi["nodes"] if node["type"] == "MlxKSamplerMLX")
    assert multi_sampler["widgets_values"][2:8] == [
        40, 1024, 1024, 1, 1.0, "flow_match_euler_discrete"
    ]


def test_tiny_dit_to_rgba_vae_end_to_end():
    transformer = QwenImage21Transformer(
        in_channels=64,
        out_channels=64,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=8,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    vae = QwenImage21VAE(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=64,
        dim_mult=(1, 1, 1, 1, 1),
        num_res_blocks=1,
        temperal_downsample=(False, True, True, True),
        out_channels=4,
        latents_mean=[0.0] * 64,
        latents_std=[1.0] * 64,
    )
    latent = sampling.create_noise(9, 32, 32, dtype=mx.float32)
    text = mx.random.normal((1, 4, 8))
    sigmas = sampling.sigma_schedule(2, 4)
    cache = QwenImage21KVCache(1)
    for index in range(2):
        noise = transformer(
            latent,
            text,
            mx.array([float(sigmas[index])], dtype=mx.float32),
            2,
            2,
            cache,
            "extract" if index == 0 else "cached",
        )
        latent = sampling.euler_step(noise, latent, sigmas[index], sigmas[index + 1])
    rgba = vae.decode(sampling.unpack_latents(latent, 32, 32))
    mx.eval(rgba)
    assert rgba.shape == (1, 4, 1, 32, 32)
    assert bool(mx.all(mx.isfinite(rgba)).item())
    assert float(mx.min(rgba).item()) >= -1.0
    assert float(mx.max(rgba).item()) <= 1.0


if __name__ == "__main__":
    tests = [
        test_qwen_image_21_is_an_isolated_supported_family,
        test_name_detection_prioritizes_qwen_image_21_and_keeps_legacy_families_separate,
        test_validation_keeps_qwen_image_21_out_of_legacy_families,
        test_qwen_image_21_requires_an_explicitly_detectable_checkpoint,
        test_legacy_model_config_lookup_retains_a_hard_qwen_image_21_boundary,
        test_dynamic_schedule_and_latent_contract,
        test_sampler_requires_the_same_qwen21_reference_on_both_condition_branches,
        test_tiny_multimodal_prompt_has_one_slot_per_2x2_latent_block,
        test_prefix_cache_matches_full_tiny_transformer_forward,
        test_multireference_edit_prefix_cache_matches_full_forward,
        test_tiny_rgba_vae_reference_encode_contract,
        test_official_t2i_and_edit_workflows_keep_their_conditioning_paths_separate,
        test_tiny_dit_to_rgba_vae_end_to_end,
    ]
    for test in tests:
        test()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        test_model_root_uses_comfyui_registered_mlx_directory(root)
        test_model_root_registers_shared_directory_when_mlx_is_not_configured(root)
        test_config_detection_identifies_a_local_qwen_image_21_checkpoint(root)
        test_public_loaders_create_qwen_image_21_handles_without_loading_weights(root)
    parameterized_tests = [
        test_qwen21_single_file_bf16_loads_and_maps_each_feed_forward_weight_independently,
        test_qwen21_single_file_splits_turbo_fused_gate_up_weight,
        test_qwen21_single_file_and_directory_loading_follow_dynamic_block_schema,
        test_qwen21_single_file_rejects_missing_and_unexpected_keys,
        test_qwen21_single_file_validates_metadata_dtype_offsets_overlap_bounds_and_size,
        test_qwen21_single_file_rejects_malformed_header_and_non_transformer_roles,
        test_qwen21_pipeline_dispatches_single_transformer_and_falls_back_to_sibling_components,
    ]
    for test in parameterized_tests:
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))
    print("全部通过：Qwen-Image 2.1 隔离、T2I/多图编辑 KV、64 通道 latent 与 RGBA VAE 契约有效。")