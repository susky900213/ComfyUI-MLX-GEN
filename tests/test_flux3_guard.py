"""FLUX 3 family and implementation-contract tests.

These tests are dependency-light and do not load multi-gigabyte weights.  They
verify that FLUX 3 is an independent family, that legacy Flux.2 routing is
rejected, and that the MLX implementation exposes the official tensor shapes.

Run directly::

    PYTHONPATH=src python tests/test_flux3_guard.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen.types import (  # noqa: E402
    FLUX3_FAMILY,
    detect_model_family,
    model_types,
    validate_model_family,
)
from comfyui_mlx_gen.flux3.dit import (  # noqa: E402
    Flux3Packed,
    Flux3Transformer,
    _join_single_stream_modulations,
)
from comfyui_mlx_gen.flux3.latent_creator import Flux3LatentCreator  # noqa: E402
from comfyui_mlx_gen.pipeline import (  # noqa: E402
    _flux3_pack_sample,
    prepare_flux3_multi_reference_image,
    prepare_flux3_reference_image,
)
import mlx.core as mx  # noqa: E402


def _assert_flux3_rejected(model_type, checkpoint):
    try:
        validate_model_family(model_type, checkpoint)
    except NotImplementedError as exc:
        message = str(exc)
        assert "不会把 FLUX 3 路由到 Flux.2" in message
    else:
        raise AssertionError("FLUX 3 checkpoint was accepted by a legacy family")


def test_flux3_is_exposed_as_a_separate_supported_model_type():
    assert FLUX3_FAMILY in model_types()
    assert detect_model_family("black-forest-labs/flux-3-image-klein") == FLUX3_FAMILY
    assert detect_model_family("flux-3-image-klein-base") == FLUX3_FAMILY
    assert detect_model_family("flux3-image-klein-comfy") == FLUX3_FAMILY
    assert detect_model_family("flux.2-klein-9b-8bit") is None


def test_flux3_architecture_in_local_config_is_detected():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "opaque-checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(
            json.dumps({"architectures": ["Flux3ImageKleinTransformer"]}), encoding="utf-8"
        )
        assert detect_model_family(checkpoint) == FLUX3_FAMILY


def test_flux3_is_rejected_instead_of_being_routed_to_flux2():
    _assert_flux3_rejected("flux2", "flux-3-image-klein")


def test_flux3_can_be_selected_without_loading_weights():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "flux-3-image-klein"
        checkpoint.mkdir()
        validate_model_family("flux3", checkpoint)


def test_flux3_latent_contract_and_transformer_defaults():
    packed, ids, latent_h, latent_w = Flux3LatentCreator.prepare_packed_latents(1, 256, 256)
    assert tuple(packed.shape) == (1, 256, 128)
    assert tuple(ids.shape) == (1, 256, 4)
    assert (latent_h, latent_w) == (16, 16)
    model = Flux3Transformer()
    assert model.flow_map is False
    assert len(model.mode_blocks["image"]) == 5
    assert len(model.single_transformer_blocks) == 28


def test_flux3_single_stream_modulation_is_joined_once():
    single = {
        "image": mx.arange(2 * 9).reshape(2, 9),
        "image_cond": mx.arange(3 * 9).reshape(3, 9) + 100,
        "text": mx.arange(4 * 9).reshape(4, 9) + 200,
    }
    placement = [
        ("image", 0, 2, 0),
        ("image_cond", 0, 3, 2),
        ("text", 0, 4, 5),
    ]

    joined = _join_single_stream_modulations(single, placement)

    assert tuple(joined.shape) == (9, 9)
    assert joined[0, 0].item() == 0
    assert joined[2, 0].item() == 100
    assert joined[5, 0].item() == 200


def test_flux3_joint_sequence_uses_official_text_image_reference_order():
    packed = Flux3Packed(
        data={
            "image": mx.zeros((2, 128)),
            "image_cond": mx.zeros((5, 128)),
            "text": mx.zeros((3, 20_480)),
        },
        ids={
            "image": mx.zeros((2, 4)),
            "image_cond": mx.zeros((5, 4)),
            "text": mx.zeros((3, 4)),
        },
        timesteps={
            "image": mx.zeros((2,)),
            "image_cond": mx.zeros((5,)),
            "text": mx.zeros((3,)),
        },
        mode_indices={
            "image": [(0, 2)],
            "image_cond": [(0, 5)],
            "text": [(0, 3)],
        },
    )

    _bounds, placement = Flux3Transformer._ranges_for_joint(packed)
    assert [mode for mode, _left, _right, _offset in placement] == [
        "image",
        "image_cond",
        "text",
    ]
    assert [offset for _mode, _left, _right, offset in placement] == [0, 2, 7]


def test_flux3_reference_preprocessing_keeps_normal_sized_images():
    image = Image.new("RGBA", (1600, 800), (255, 0, 0, 128))
    prepared = prepare_flux3_reference_image(image, "aspect_area_crop")
    assert prepared.mode == "RGB"
    assert prepared.size == (1600, 800)

    cropped = prepare_flux3_reference_image(Image.new("RGB", (1611, 803)), "auto")
    assert cropped.size == (1600, 800)


def test_flux3_multi_reference_preprocessing_uses_one_megapixel_per_reference():
    square = prepare_flux3_multi_reference_image(Image.new("RGB", (2048, 2048)), "auto")
    assert square.size == (1024, 1024)

    wide = prepare_flux3_multi_reference_image(Image.new("RGB", (4000, 2000)), "auto")
    assert wide.size == (1440, 720)
    assert wide.width * wide.height <= 1024 * 1024

    kept = prepare_flux3_multi_reference_image(Image.new("RGB", (2048, 2048)), "keep_resolution")
    assert kept.size == (2048, 2048)


def test_flux3_reference_ids_use_input_order_on_negative_layer_axis():
    first = mx.zeros((1, 128, 2, 3))
    second = mx.zeros((1, 128, 1, 2))
    first_ids = Flux3LatentCreator.prepare_grid_ids(first, layer_coord=-1)
    second_ids = Flux3LatentCreator.prepare_grid_ids(second, layer_coord=-2)
    ids = mx.concatenate([first_ids, second_ids], axis=1)
    assert tuple(ids.shape) == (1, 8, 4)
    assert np.all(np.asarray(ids[0, :6, 3]) == -1)
    assert np.all(np.asarray(ids[0, 6:, 3]) == -2)


def test_flux3_sampler_pack_keeps_all_multi_reference_tokens():
    text = (mx.zeros((1, 2, 20_480)), mx.zeros((1, 2, 4)))
    image = (mx.zeros((1, 4, 128)), mx.zeros((1, 4, 4)))
    references = (
        mx.zeros((1, 3, 128)),
        mx.zeros((1, 3, 4)),
        1024,
        1024,
    )

    packed = _flux3_pack_sample(text, references, image, 0)

    assert tuple(packed.data["image_cond"].shape) == (3, 128)
    assert packed.mode_indices["image_cond"] == [(0, 3)]
    assert tuple(packed.ids["image_cond"].shape) == (3, 4)


def test_flux3_edit_workflows_keep_reference_images_out_of_text_encoder():
    for workflow_name in ("flux3-klein-edit.json", "flux3-klein-edit-multi.json"):
        workflow = json.loads(
            (ROOT / "workflows" / workflow_name).read_text(encoding="utf-8")
        )
        nodes = {node["id"]: node for node in workflow["nodes"]}

        text_encoder_ref_inputs = [
            (node["id"], item)
            for node in workflow["nodes"]
            if node["type"] == "MlxTextEncoder"
            for item in node.get("inputs", [])
            if item["name"] == "ref_images" and item.get("link") is not None
        ]
        assert not text_encoder_ref_inputs, (
            f"{workflow_name} 把 FLUX 3 参考图接入了 MlxTextEncoder: "
            f"{text_encoder_ref_inputs}"
        )

        reference_links = [
            link for link in workflow["links"] if link[-1] == "mlx_ref_images"
        ]
        assert reference_links, f"{workflow_name} 缺少 FLUX 3 参考图到采样器的连线"
        assert all(nodes[link[3]]["type"] == "MlxKSamplerMLX" for link in reference_links)


if __name__ == "__main__":
    tests = [
        test_flux3_is_exposed_as_a_separate_supported_model_type,
        test_flux3_architecture_in_local_config_is_detected,
        test_flux3_is_rejected_instead_of_being_routed_to_flux2,
        test_flux3_can_be_selected_without_loading_weights,
        test_flux3_latent_contract_and_transformer_defaults,
        test_flux3_single_stream_modulation_is_joined_once,
        test_flux3_joint_sequence_uses_official_text_image_reference_order,
        test_flux3_reference_preprocessing_keeps_normal_sized_images,
        test_flux3_multi_reference_preprocessing_uses_one_megapixel_per_reference,
        test_flux3_reference_ids_use_input_order_on_negative_layer_axis,
        test_flux3_edit_workflows_keep_reference_images_out_of_text_encoder,
    ]
    for test in tests:
        test()
    print(f"[OK] {len(tests)} FLUX 3 guard tests passed")

