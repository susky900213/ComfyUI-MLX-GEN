"""Qwen-Image 2.1 detection, architecture and tiny end-to-end tests.

The tests construct tiny random models and never load the official 30 GiB
checkpoint. Run directly (pytest is not required)::

    PYTHONPATH=src python tests/test_qwen_image_21.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
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
    print("全部通过：Qwen-Image 2.1 隔离、T2I/多图编辑 KV、64 通道 latent 与 RGBA VAE 契约有效。")