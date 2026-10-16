from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from comfyui_mlx_gen.nodes import qwen_image_21_folder_edit as folder_module
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


def _patch_fake_pipeline(monkeypatch, decoded_size=(32, 32)):
    reference_calls: list[tuple[str, int, int, int, str, int, tuple[int, int]]] = []
    text_calls: list[tuple[str, str, int]] = []
    sample_calls: list[dict] = []
    encoder_lifecycle_calls: list[str] = []

    monkeypatch.setattr(folder_module.runtime, "apply_cache_limit", lambda *args, **kwargs: None)
    monkeypatch.setattr(folder_module.runtime, "flush_caches", lambda: None)
    monkeypatch.setattr(folder_module.pipeline, "vae_component", lambda handle, cache: object())
    monkeypatch.setattr(folder_module.pipeline, "release_vae_component", lambda handle, cache: None)
    monkeypatch.setattr(
        folder_module.pipeline,
        "reference_image_cache_key",
        lambda _vae, digest, params: f"ref-{digest[:8]}-{params['width']}x{params['height']}",
    )

    def fake_encode_reference(_vae_module, images, cache, cache_key, max_images, width, height, resize_mode):
        reference_calls.append(
            (
                cache_key,
                len(images),
                width,
                height,
                resize_mode,
                images[0].getpixel((0, 0))[0],
                images[0].size,
            )
        )
        cache.get_or_create(
            "ref_encoding",
            cache_key,
            lambda: {"images": tuple(image.convert("RGB") for image in images), "latents": None, "shapes": ()},
        )

    monkeypatch.setattr(folder_module.pipeline, "encode_qwen21_reference_images", fake_encode_reference)
    monkeypatch.setattr(
        folder_module.pipeline,
        "prepare_encoder",
        lambda entry, clip_handle, cache, cache_key: (
            encoder_lifecycle_calls.append("prepare") or {"fake": True}
        ),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "release_encoder",
        lambda clip_handle, cache: encoder_lifecycle_calls.append("release"),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "prompt_encoding_key",
        lambda clip_handle, text, ref_key: f"{text}:{ref_key}",
    )

    def fake_encode_text(entry, components, text, cache, cache_key, **kwargs):
        reference_images = kwargs["reference_images"]
        text_calls.append((text, cache_key, len(reference_images)))
        cache.get_or_create("prompt_encoding", cache_key, lambda: (None, None))

    monkeypatch.setattr(folder_module.pipeline, "encode_text", fake_encode_text)
    monkeypatch.setattr(folder_module.pipeline, "prepare_sampler_components", lambda *args: {})
    monkeypatch.setattr(folder_module.pipeline, "clear_compiled_predict_caches", lambda: None)
    monkeypatch.setattr(folder_module.pipeline, "release_sampler_components", lambda *args: None)

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

    monkeypatch.setattr(folder_module.pipeline, "run_sampler", fake_run_sampler)
    monkeypatch.setattr(
        folder_module.pipeline,
        "decode_latents",
        lambda _entry, _components, cached, _height, _width: int(cached.shape[0]),
    )
    monkeypatch.setattr(
        folder_module.image_mod,
        "to_pil",
        lambda decoded, batch_index=-1: (Image.new("RGB", decoded_size, (decoded, 0, 0)),),
    )
    return reference_calls, text_calls, sample_calls, encoder_lifecycle_calls


def test_folder_edit_processes_natural_sorted_images_one_at_a_time(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    input_dir = comfy_input / "mlx_qwen_image_21" / "test-batch"
    input_dir.mkdir(parents=True)
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    # Natural sort must be image1, image2, image10 rather than lexical image1,
    # image10, image2.  The text file must be ignored.
    Image.new("RGBA", (128, 80), (10, 0, 0, 255)).save(input_dir / "image10.png")
    Image.new("RGBA", (96, 64), (20, 0, 0, 255)).save(input_dir / "image2.png")
    Image.new("RGBA", (64, 96), (30, 0, 0, 255)).save(input_dir / "image1.png")
    (input_dir / "not-an-image.txt").write_text("ignore", encoding="utf-8")

    reference_calls, text_calls, sample_calls, encoder_lifecycle_calls = _patch_fake_pipeline(monkeypatch)
    progress_updates: list[tuple[int, int]] = []

    class FakeProgress:
        def __init__(self, total):
            self.total = total
            self.current = 0

        def update(self, amount=1):
            self.current += amount
            progress_updates.append((self.current, self.total))

    monkeypatch.setattr(folder_module, "SamplingProgress", FakeProgress)
    vae, clip, model = _handles()
    result = folder_module.MlxQwenImage21FolderEdit().edit_folder(
        "mlx_qwen_image_21/test-batch",
        vae,
        clip,
        model,
        folder_module.DEFAULT_PROMPT,
        folder_module.DEFAULT_NEGATIVE_PROMPT,
        100,
        2,
        1.0,
        "flow_match_euler_discrete",
        "off",
    )

    assert [call[5] for call in reference_calls] == [30, 20, 10]
    assert all(call[1] == 1 for call in reference_calls)
    assert [(call[2], call[3]) for call in reference_calls] == [(64, 96), (96, 64), (128, 80)]
    assert all(call[4] == "keep_resolution" for call in reference_calls)
    assert len(text_calls) == 6  # positive + negative for every source image
    assert all(call[2] == 1 for call in text_calls)
    assert encoder_lifecycle_calls == ["prepare", "release"]
    assert len(sample_calls) == 3
    assert [call["seed"] for call in sample_calls] == [100, 101, 102]
    assert [(call["width"], call["height"]) for call in sample_calls] == [
        (64, 96),
        (96, 64),
        (128, 80),
    ]
    assert all(call["batch_size"] == 1 and call["ref_count"] == 1 for call in sample_calls)

    saved = result[1].splitlines()
    assert [path.split("/")[-1] for path in saved] == [
        "0000_image1.png",
        "0001_image2.png",
        "0002_image10.png",
    ]
    assert all(str(input_dir / "image") in path for path in saved)
    for path in saved:
        destination = Path(path)
        assert destination.is_file()
        with Image.open(destination) as generated:
            assert generated.format == "PNG"
            assert generated.size == (32, 32)
    assert sorted(destination.name for destination in (input_dir / "image").glob("*.png")) == [
        "0000_image1.png",
        "0001_image2.png",
        "0002_image10.png",
    ]
    assert result[3] == 3
    report = json.loads(result[2])
    assert [record["status"] for record in report] == ["success", "success", "success"]
    assert progress_updates == [(1, 3), (2, 3), (3, 3)]


def test_folder_edit_counts_failed_images_in_progress_and_continues(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    input_dir = comfy_input / "mlx_qwen_image_21" / "with-failure"
    input_dir.mkdir(parents=True)
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    Image.new("RGBA", (64, 64), (10, 0, 0, 255)).save(input_dir / "image1.png")
    (input_dir / "image2.png").write_text("not a valid image", encoding="utf-8")
    Image.new("RGBA", (64, 64), (30, 0, 0, 255)).save(input_dir / "image3.png")

    reference_calls, _text_calls, sample_calls, encoder_lifecycle_calls = _patch_fake_pipeline(monkeypatch)
    progress_updates: list[tuple[int, int]] = []

    class FakeProgress:
        def __init__(self, total):
            self.total = total
            self.current = 0

        def update(self, amount=1):
            self.current += amount
            progress_updates.append((self.current, self.total))

    monkeypatch.setattr(folder_module, "SamplingProgress", FakeProgress)
    vae, clip, model = _handles()
    result = folder_module.MlxQwenImage21FolderEdit().edit_folder(
        "mlx_qwen_image_21/with-failure",
        vae,
        clip,
        model,
        folder_module.DEFAULT_PROMPT,
        folder_module.DEFAULT_NEGATIVE_PROMPT,
        100,
        2,
        1.0,
        "flow_match_euler_discrete",
        "off",
    )

    assert len(reference_calls) == 2
    assert len(sample_calls) == 2
    assert result[3] == 2
    assert progress_updates == [(1, 3), (2, 3), (3, 3)]
    assert encoder_lifecycle_calls == ["prepare", "release"]
    report = json.loads(result[2])
    assert [record["status"] for record in report] == ["success", "error", "success"]


def test_folder_edit_skips_existing_outputs_and_keeps_original_indexes(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    input_dir = comfy_input / "mlx_qwen_image_21" / "resume"
    input_dir.mkdir(parents=True)
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    Image.new("RGBA", (64, 64), (10, 0, 0, 255)).save(input_dir / "image1.png")
    Image.new("RGBA", (64, 64), (20, 0, 0, 255)).save(input_dir / "image2.png")
    Image.new("RGBA", (64, 64), (30, 0, 0, 255)).save(input_dir / "image3.png")
    output = input_dir / "image"
    output.mkdir()
    Image.new("RGBA", (32, 32), (99, 0, 0, 255)).save(output / "0001_image2.png")

    reference_calls, _text_calls, sample_calls, encoder_lifecycle_calls = _patch_fake_pipeline(monkeypatch)
    progress_updates: list[tuple[int, int]] = []

    class FakeProgress:
        def __init__(self, total):
            self.total = total

        def update(self, amount=1):
            progress_updates.append((len(progress_updates) + amount, self.total))

    monkeypatch.setattr(folder_module, "SamplingProgress", FakeProgress)
    vae, clip, model = _handles()
    result = folder_module.MlxQwenImage21FolderEdit().edit_folder(
        "mlx_qwen_image_21/resume",
        vae,
        clip,
        model,
        "保持主体",
        "模糊",
        100,
        2,
        1.0,
        "flow_match_euler_discrete",
        "off",
    )

    assert len(reference_calls) == 2
    assert [call["seed"] for call in sample_calls] == [100, 102]
    assert encoder_lifecycle_calls == ["prepare", "release"]
    assert [Path(path).name for path in result[1].splitlines()] == [
        "0000_image1.png",
        "0001_image2.png",
        "0002_image3.png",
    ]
    assert result[3] == 3
    report = json.loads(result[2])
    assert [record["status"] for record in report] == ["success", "skipped", "success"]
    assert report[1]["output"].endswith("image/0001_image2.png")
    assert progress_updates == [(1, 3), (2, 3), (3, 3)]


def test_folder_edit_all_existing_outputs_skip_model_loading(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    input_dir = comfy_input / "mlx_qwen_image_21" / "already-done"
    input_dir.mkdir(parents=True)
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    Image.new("RGBA", (64, 64), (10, 0, 0, 255)).save(input_dir / "image1.png")
    Image.new("RGBA", (64, 64), (20, 0, 0, 255)).save(input_dir / "image2.png")
    output = input_dir / "image"
    output.mkdir()
    Image.new("RGBA", (32, 32), (90, 0, 0, 255)).save(output / "0000_image1.png")
    Image.new("RGBA", (32, 32), (80, 0, 0, 255)).save(output / "0001_image2.png")

    _reference_calls, _text_calls, sample_calls, encoder_lifecycle_calls = _patch_fake_pipeline(monkeypatch)
    monkeypatch.setattr(
        folder_module,
        "SamplingProgress",
        lambda total: type("Progress", (), {"update": lambda self: None})(),
    )
    vae, clip, model = _handles()
    result = folder_module.MlxQwenImage21FolderEdit().edit_folder(
        "mlx_qwen_image_21/already-done",
        vae,
        clip,
        model,
        "保持主体",
        "模糊",
        0,
        1,
        1.0,
        "flow_match_euler_discrete",
        "off",
    )

    assert sample_calls == []
    assert encoder_lifecycle_calls == []
    assert result[3] == 2
    assert [record["status"] for record in json.loads(result[2])] == ["skipped", "skipped"]


def test_folder_edit_reprocesses_corrupt_existing_output(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    input_dir = comfy_input / "mlx_qwen_image_21" / "corrupt-output"
    input_dir.mkdir(parents=True)
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    Image.new("RGBA", (64, 64), (10, 0, 0, 255)).save(input_dir / "source.png")
    output = input_dir / "image"
    output.mkdir()
    (output / "0000_source.png").write_text("incomplete", encoding="utf-8")

    _reference_calls, _text_calls, sample_calls, _encoder_lifecycle_calls = _patch_fake_pipeline(monkeypatch)
    monkeypatch.setattr(
        folder_module,
        "SamplingProgress",
        lambda total: type("Progress", (), {"update": lambda self: None})(),
    )
    vae, clip, model = _handles()
    result = folder_module.MlxQwenImage21FolderEdit().edit_folder(
        "mlx_qwen_image_21/corrupt-output",
        vae,
        clip,
        model,
        "保持主体",
        "模糊",
        0,
        1,
        1.0,
        "flow_match_euler_discrete",
        "off",
    )

    assert len(sample_calls) == 1
    assert json.loads(result[2])[0]["status"] == "success"
    with Image.open(output / "0000_source.png") as generated:
        assert generated.format == "PNG"


def test_folder_edit_keeps_source_size_and_scales_sampling_canvas(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    input_dir = comfy_input / "mlx_qwen_image_21" / "scaled"
    input_dir.mkdir(parents=True)
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    Image.new("RGBA", (128, 80), (10, 0, 0, 255)).save(input_dir / "source.png")

    reference_calls, _text_calls, sample_calls, _encoder_lifecycle_calls = _patch_fake_pipeline(
        monkeypatch,
        decoded_size=(128, 80),
    )
    monkeypatch.setattr(folder_module, "SamplingProgress", lambda total: type("Progress", (), {"update": lambda self: None})())
    vae, clip, model = _handles()
    result = folder_module.MlxQwenImage21FolderEdit().edit_folder(
        "mlx_qwen_image_21/scaled",
        vae,
        clip,
        model,
        "保持主体",
        "模糊",
        0,
        1,
        1.0,
        "flow_match_euler_discrete",
        "off",
        0.5,
    )

    assert reference_calls[0][2:5] == (128, 80, "keep_resolution")
    assert reference_calls[0][6] == (128, 80)
    assert sample_calls[0]["width"] == 64
    assert sample_calls[0]["height"] == 40
    output_path = Path(result[1])
    with Image.open(output_path) as generated:
        # 采样时降分辨率不对解码结果再做一次 PIL 缩放；输出尺寸由解码链路决定。
        assert generated.size == (128, 80)
    report = json.loads(result[2])
    assert report[0]["resolution_scale"] == 0.5
    assert report[0]["width"] == 64
    assert report[0]["height"] == 40


def test_before_timing_downscales_reference_and_sampling_canvas(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    input_dir = comfy_input / "mlx_qwen_image_21" / "legacy-timing"
    input_dir.mkdir(parents=True)
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    Image.new("RGBA", (128, 80), (10, 0, 0, 255)).save(input_dir / "source.png")

    reference_calls, _text_calls, sample_calls, _encoder_lifecycle_calls = _patch_fake_pipeline(monkeypatch)
    monkeypatch.setattr(
        folder_module,
        "SamplingProgress",
        lambda total: type("Progress", (), {"update": lambda self: None})(),
    )
    vae, clip, model = _handles()
    result = folder_module.MlxQwenImage21FolderEdit().edit_folder(
        "mlx_qwen_image_21/legacy-timing",
        vae,
        clip,
        model,
        "保持主体",
        "模糊",
        0,
        1,
        1.0,
        "flow_match_euler_discrete",
        "off",
        0.5,
        "先降分辨率",
    )

    assert reference_calls[0][2:5] == (64, 40, "keep_resolution")
    assert reference_calls[0][6] == (64, 40)
    assert (sample_calls[0]["width"], sample_calls[0]["height"]) == (64, 40)
    assert json.loads(result[2])[0]["resolution_timing"] == "先降分辨率"


def test_folder_edit_scales_final_image_after_generation(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    input_dir = comfy_input / "mlx_qwen_image_21" / "scaled-after"
    input_dir.mkdir(parents=True)
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    Image.new("RGBA", (128, 80), (10, 0, 0, 255)).save(input_dir / "source.png")

    reference_calls, _text_calls, sample_calls, _encoder_lifecycle_calls = _patch_fake_pipeline(
        monkeypatch,
        decoded_size=(128, 80),
    )
    monkeypatch.setattr(
        folder_module,
        "SamplingProgress",
        lambda total: type("Progress", (), {"update": lambda self: None})(),
    )
    vae, clip, model = _handles()
    result = folder_module.MlxQwenImage21FolderEdit().edit_folder(
        "mlx_qwen_image_21/scaled-after",
        vae,
        clip,
        model,
        "保持主体",
        "模糊",
        0,
        1,
        1.0,
        "flow_match_euler_discrete",
        "off",
        0.5,
        "后降分辨率",
    )

    assert reference_calls[0][2:5] == (128, 80, "keep_resolution")
    assert reference_calls[0][6] == (128, 80)
    assert sample_calls[0]["width"] == 128
    assert sample_calls[0]["height"] == 80
    output_path = Path(result[1])
    with Image.open(output_path) as generated:
        assert generated.size == (64, 40)
    report = json.loads(result[2])
    assert report[0]["resolution_scale"] == 0.5
    assert report[0]["resolution_timing"] == "后降分辨率"
    assert report[0]["width"] == 64
    assert report[0]["height"] == 40


def test_folder_edit_rejects_non_positive_resolution_scale(tmp_path, monkeypatch):
    vae, clip, model = _handles()
    node = folder_module.MlxQwenImage21FolderEdit()
    with pytest.raises(ValueError, match="分辨率系数必须是大于 0"):
        node.edit_folder(
            "mlx_qwen_image_21/not-used",
            vae,
            clip,
            model,
            "保持主体",
            "模糊",
            0,
            1,
            1.0,
            "flow_match_euler_discrete",
            "off",
            0,
        )


def test_folder_edit_rejects_unknown_resolution_timing():
    vae, clip, model = _handles()
    node = folder_module.MlxQwenImage21FolderEdit()
    with pytest.raises(ValueError, match="未知分辨率缩放时机"):
        node.edit_folder(
            "mlx_qwen_image_21/not-used",
            vae,
            clip,
            model,
            "保持主体",
            "模糊",
            0,
            1,
            1.0,
            "flow_match_euler_discrete",
            "off",
            1.0,
            "invalid",
        )


def test_folder_edit_uses_input_image_subdirectory():
    inputs = folder_module.MlxQwenImage21FolderEdit.INPUT_TYPES()["required"]
    assert "input_directory" in inputs
    input_type, options = inputs["input_directory"]
    assert input_type == "STRING"
    assert options["default"] == ""
    assert "选择输入文件夹" in options["tooltip"]
    assert "output_directory" not in inputs
    assert "width" not in inputs
    assert "height" not in inputs
    assert "resize_mode" not in inputs
    assert inputs["resolution_scale"] == (
        "FLOAT",
        {
            "default": 1.0,
            "min": 0.01,
            "max": 4.0,
            "step": 0.01,
            "tooltip": "按每张图片原始宽高乘以此系数；例如 0.5 表示目标尺寸减半，具体作用取决于缩放时机",
        },
    )
    assert inputs["resolution_timing"] == (
        ["采样时降分辨率", "先降分辨率", "后降分辨率"],
        {
            "default": "采样时降分辨率",
            "tooltip": "选择只缩小采样画布、先缩小参考图后采样，或在生成完成后缩小最终图片",
        },
    )


def test_folder_workflow_serializes_prompt_and_sampler_values_in_order():
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / "workflows"
        / "qwen-image-2.1-folder-batch-edit.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    node = next(item for item in workflow["nodes"] if item["type"] == "MlxQwenImage21FolderEdit")

    # control_after_generate is a frontend-only widget (serialize=false). The
    # serialized values must therefore map directly to the Python signature:
    # input_directory, prompt, negative_prompt, seed, steps, guidance,
    # scheduler, kv_cache, resolution_scale, resolution_timing.
    assert node["widgets_values"] == [
        "",
        folder_module.DEFAULT_PROMPT,
        folder_module.DEFAULT_NEGATIVE_PROMPT,
        42,
        40,
        1.0,
        "flow_match_euler_discrete",
        "auto",
        1.0,
        "采样时降分辨率",
    ]
    assert "randomize" not in node["widgets_values"]


def test_output_directory_is_input_image_subdirectory(tmp_path):
    output = folder_module._output_directory(str(tmp_path / "source"))
    assert output == (tmp_path / "source" / "image").resolve()
    assert output.is_dir()


def test_folder_picker_keeps_selected_folder_name_and_image_output():
    picker_path = Path(__file__).resolve().parents[1] / "web" / "qwen_image_21_folder_picker.js"
    picker = picker_path.read_text(encoding="utf-8")
    assert "const subfolder = `${UPLOAD_ROOT}/${folderName}`;" in picker
    assert "node.properties.mlx_qwen_image_21_output_directory = `${subfolder}/image`;" in picker
    assert "输出：image/" in picker


def test_folder_token_resolves_only_below_comfy_input(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    selected = comfy_input / "mlx_qwen_image_21" / "batch"
    selected.mkdir(parents=True)
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)

    assert folder_module._resolve_input_directory("mlx_qwen_image_21/batch") == selected
    with pytest.raises(ValueError, match=r"不能使用绝对路径或 \.\."):
        folder_module._resolve_input_directory(str(selected))
    with pytest.raises(ValueError, match=r"不能使用绝对路径或 \.\."):
        folder_module._resolve_input_directory("mlx_qwen_image_21/../outside")
    with pytest.raises(ValueError, match="必须来自.*选择输入文件夹"):
        folder_module._resolve_input_directory("other-folder/batch")


def test_folder_edit_rejects_empty_directory(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    input_dir = comfy_input / "mlx_qwen_image_21" / "empty"
    input_dir.mkdir(parents=True)
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    vae, clip, model = _handles()
    node = folder_module.MlxQwenImage21FolderEdit()

    monkeypatch.setattr(folder_module.pipeline.qwen21_sampling, "validate_dimensions", lambda h, w: (h, w))
    with pytest.raises(ValueError, match="没有可读取的图片"):
        node.edit_folder(
            "mlx_qwen_image_21/empty",
            vae,
            clip,
            model,
            "保持主体",
            "模糊",
            0,
            1,
            1.0,
            "flow_match_euler_discrete",
        )


def test_default_prompt_requires_chinese_text_replacement():
    assert "文字全部替换为准确、自然、清晰、可读的简体中文" in folder_module.DEFAULT_PROMPT
    assert "没有文字时不要额外添加文字" in folder_module.DEFAULT_PROMPT