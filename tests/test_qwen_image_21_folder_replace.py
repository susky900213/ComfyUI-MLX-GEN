from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from comfyui_mlx_gen import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from comfyui_mlx_gen.nodes import qwen_image_21_folder_replace as replace_module
from comfyui_mlx_gen.types import MlxClipHandle, MlxLatentHandle, MlxModelHandle, MlxVaeHandle


def _handles():
    family = "qwen_image_21"
    return (
        MlxVaeHandle(
            model_type=family,
            path="Qwen-Image-2.1",
            precision="bfloat16",
            quantize=0,
            role="vae",
            cache_key="test-vae",
        ),
        MlxClipHandle(
            model_type=family,
            component="text_encoder",
            source="local",
            path="Qwen-Image-2.1",
            precision="bfloat16",
            max_length=262144,
            cache_key="test-clip",
        ),
        MlxModelHandle(
            model_type=family,
            model_path="Qwen-Image-2.1",
            quantize=8,
            precision="bfloat16",
            compile=False,
            compile_cache_limit=0,
            cache_key="test-model",
        ),
    )


def _patch_fake_pipeline(monkeypatch):
    reference_calls: list[tuple[Image.Image, ...]] = []
    text_calls: list[tuple[str, tuple[Image.Image, ...]]] = []
    sample_calls: list[dict] = []

    monkeypatch.setattr(replace_module.runtime, "apply_cache_limit", lambda *args, **kwargs: None)
    monkeypatch.setattr(replace_module.runtime, "flush_caches", lambda: None)
    monkeypatch.setattr(replace_module.pipeline.qwen21_sampling, "validate_dimensions", lambda h, w: (h, w))
    monkeypatch.setattr(replace_module.pipeline, "vae_component", lambda handle, cache: object())
    monkeypatch.setattr(replace_module.pipeline, "release_vae_component", lambda handle, cache: None)
    monkeypatch.setattr(
        replace_module.pipeline,
        "reference_image_cache_key",
        lambda _vae, digest, params: f"ref-{digest[:8]}-{params['count']}-{params['width']}x{params['height']}",
    )

    def fake_encode_reference(
        _vae_module, images, cache, cache_key, max_images, width, height, resize_mode
    ):
        assert max_images == 2
        assert resize_mode == "keep_resolution"
        reference_calls.append(tuple(image.copy() for image in images))
        cache.get_or_create(
            "ref_encoding",
            cache_key,
            lambda: {
                "images": tuple(image.convert("RGB") for image in images),
                "latents": None,
                "shapes": (),
            },
        )

    monkeypatch.setattr(
        replace_module.pipeline, "encode_qwen21_reference_images", fake_encode_reference
    )
    monkeypatch.setattr(replace_module.pipeline, "prepare_encoder", lambda *args: {"fake": True})
    monkeypatch.setattr(replace_module.pipeline, "release_encoder", lambda *args: None)
    monkeypatch.setattr(
        replace_module.pipeline,
        "prompt_encoding_key",
        lambda _clip, text, ref_key: f"{text}:{ref_key}",
    )

    def fake_encode_text(entry, components, text, cache, cache_key, **kwargs):
        text_calls.append((text, tuple(image.copy() for image in kwargs["reference_images"])))
        cache.get_or_create("prompt_encoding", cache_key, lambda: (None, None))

    monkeypatch.setattr(replace_module.pipeline, "encode_text", fake_encode_text)
    monkeypatch.setattr(replace_module.pipeline, "prepare_sampler_components", lambda *args: {})
    monkeypatch.setattr(replace_module.pipeline, "clear_compiled_predict_caches", lambda: None)
    monkeypatch.setattr(replace_module.pipeline, "release_sampler_components", lambda *args: None)

    def fake_run_sampler(entry, model_handle, components, params, cache):
        sample_calls.append(dict(params))
        key = f"latent-{len(sample_calls)}"
        cache.get_or_create("component_weights", key, lambda: np.zeros((1, 1, 64), dtype=np.float32))
        return MlxLatentHandle(
            kind="noise",
            shape=(1, 1, 64),
            dtype="float32",
            cache_key=key,
            model="qwen_image_21",
            height=params["height"],
            width=params["width"],
        )

    monkeypatch.setattr(replace_module.pipeline, "run_sampler", fake_run_sampler)
    monkeypatch.setattr(
        replace_module.pipeline,
        "decode_latents",
        lambda _entry, _components, cached, _height, _width: int(cached.shape[0]),
    )
    monkeypatch.setattr(
        replace_module.image_mod,
        "to_pil",
        lambda decoded, batch_index=-1: (Image.new("RGB", (32, 32), (decoded, 0, 0)),),
    )
    return reference_calls, text_calls, sample_calls


def test_folder_replace_orders_source_then_person_and_saves_to_images(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    input_dir = comfy_input / "mlx_qwen_image_21" / "replace-batch"
    input_dir.mkdir(parents=True)
    monkeypatch.setattr(replace_module, "_comfy_input_directory", lambda: comfy_input)

    Image.new("RGBA", (64, 48), (10, 0, 0, 255)).save(input_dir / "image10.png")
    Image.new("RGBA", (48, 64), (20, 0, 0, 255)).save(input_dir / "image2.png")
    Image.new("RGBA", (40, 40), (30, 0, 0, 255)).save(input_dir / "image1.png")
    (input_dir / "not-an-image.txt").write_text("ignore", encoding="utf-8")
    (input_dir / "images").mkdir()
    Image.new("RGBA", (10, 10), (99, 0, 0, 255)).save(input_dir / "images" / "old.png")

    reference_calls, text_calls, sample_calls = _patch_fake_pipeline(monkeypatch)
    progress_updates: list[tuple[int, int]] = []

    class FakeProgress:
        def __init__(self, total):
            self.total = total
            self.current = 0

        def update(self, amount=1):
            self.current += amount
            progress_updates.append((self.current, self.total))

    monkeypatch.setattr(replace_module, "SamplingProgress", FakeProgress)
    person_image = np.full((1, 24, 24, 3), (0, 0, 200), dtype=np.float32) / 255.0
    vae, clip, model = _handles()

    result = replace_module.MlxQwenImage21FolderReplace().replace_folder(
        "mlx_qwen_image_21/replace-batch",
        person_image,
        vae,
        clip,
        model,
        "positive replacement prompt",
        "negative replacement prompt",
        100,
        2,
        1.0,
        "flow_match_euler_discrete",
        "off",
    )

    assert [[image.getpixel((0, 0))[0] for image in call] for call in reference_calls] == [
        [30, 0],
        [20, 0],
        [10, 0],
    ]
    assert all(len(call) == 2 for call in reference_calls)
    assert [text for text, _images in text_calls] == [
        "positive replacement prompt",
        "negative replacement prompt",
    ] * 3
    assert all([image.getpixel((0, 0))[0] for image in images] == [30, 0] for _text, images in text_calls[:2])
    assert all(len(images) == 2 for _text, images in text_calls)
    assert [call["seed"] for call in sample_calls] == [100, 101, 102]
    assert all(call["batch_size"] == 1 and call["ref_count"] == 2 for call in sample_calls)

    saved = result[1].splitlines()
    assert [Path(path).name for path in saved] == [
        "0000_image1.png",
        "0001_image2.png",
        "0002_image10.png",
    ]
    assert all(str(input_dir / "images") in path for path in saved)
    assert sorted(path.name for path in (input_dir / "images").glob("*.png")) == [
        "0000_image1.png",
        "0001_image2.png",
        "0002_image10.png",
        "old.png",
    ]
    assert result[3] == 3
    assert [record["reference_order"] for record in json.loads(result[2])] == [
        ["source", "person"],
        ["source", "person"],
        ["source", "person"],
    ]
    assert progress_updates == [(1, 3), (2, 3), (3, 3)]


def test_folder_replace_continues_after_one_invalid_source(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    input_dir = comfy_input / "mlx_qwen_image_21" / "with-failure"
    input_dir.mkdir(parents=True)
    monkeypatch.setattr(replace_module, "_comfy_input_directory", lambda: comfy_input)
    Image.new("RGBA", (32, 32), (10, 0, 0, 255)).save(input_dir / "image1.png")
    (input_dir / "image2.png").write_text("not an image", encoding="utf-8")
    Image.new("RGBA", (32, 32), (30, 0, 0, 255)).save(input_dir / "image3.png")
    reference_calls, _text_calls, sample_calls = _patch_fake_pipeline(monkeypatch)

    class FakeProgress:
        def __init__(self, total):
            self.total = total
            self.current = 0

        def update(self, amount=1):
            self.current += amount

    monkeypatch.setattr(replace_module, "SamplingProgress", FakeProgress)
    vae, clip, model = _handles()
    result = replace_module.MlxQwenImage21FolderReplace().replace_folder(
        "mlx_qwen_image_21/with-failure",
        np.full((1, 16, 16, 3), 0.5, dtype=np.float32),
        vae,
        clip,
        model,
        "prompt",
        "negative",
        0,
        1,
        1.0,
        "flow_match_euler_discrete",
        "off",
    )
    assert len(reference_calls) == 2
    assert len(sample_calls) == 2
    assert result[3] == 2
    assert [record["status"] for record in json.loads(result[2])] == ["success", "error", "success"]


def test_folder_replace_is_registered_with_independent_picker_and_workflow():
    assert NODE_CLASS_MAPPINGS["MlxQwenImage21FolderReplace"] is replace_module.MlxQwenImage21FolderReplace
    assert NODE_DISPLAY_NAME_MAPPINGS["MlxQwenImage21FolderReplace"] == "Qwen-Image 2.1 文件夹人物替换"

    picker = (
        Path(__file__).resolve().parents[1] / "web" / "qwen_image_21_folder_replace_picker.js"
    ).read_text(encoding="utf-8")
    assert 'const TARGET_NODE = "MlxQwenImage21FolderReplace";' in picker
    assert "${subfolder}/images" in picker
    assert "输出：images/" in picker

    workflow = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "workflows"
            / "qwen-image-2.1-folder-replace.json"
        ).read_text(encoding="utf-8")
    )
    node = next(item for item in workflow["nodes"] if item["type"] == "MlxQwenImage21FolderReplace")
    assert node["widgets_values"][0] == ""
    assert node["widgets_values"][3:7] == [42, 40, 1.0, "flow_match_euler_discrete"]
    assert any(link[2] == 0 and link[3] == node["id"] and link[5] == "IMAGE" for link in workflow["links"])
