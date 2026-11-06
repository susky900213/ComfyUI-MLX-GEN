from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen import pipeline
from comfyui_mlx_gen.cache import Cache
from comfyui_mlx_gen.nodes import moss_folder_edit as folder_module
from comfyui_mlx_gen.types import (
    MlxClipHandle,
    MlxLatentHandle,
    MlxModelHandle,
    MlxVaeHandle,
)


def _handles(model_path: str = "OpenMOSS-Team--MOSS-Image-SFT"):
    family = "moss"
    return (
        MlxVaeHandle(
            model_type=family,
            path=model_path,
            precision="bfloat16",
            quantize=0,
            role="vae",
            cache_key="test-vae",
        ),
        MlxClipHandle(
            model_type=family,
            component="text_encoder",
            source="local",
            path=model_path,
            precision="bfloat16",
            max_length=4096,
            cache_key="test-clip",
        ),
        MlxModelHandle(
            model_type=family,
            model_path=model_path,
            quantize=8,
            precision="bfloat16",
            compile=False,
            compile_cache_limit=0,
            cache_key="test-model",
        ),
    )


def _make_input(tmp_path: Path, name: str = "batch") -> tuple[Path, Path]:
    comfy_input = tmp_path / "comfy-input"
    input_dir = comfy_input / folder_module.UPLOAD_ROOT / name
    input_dir.mkdir(parents=True)
    return comfy_input, input_dir


def _run(node, input_token, handles, **kwargs):
    return node.edit_folder(
        input_token,
        *handles,
        kwargs.pop("prompt", "保持主体，只按提示编辑"),
        kwargs.pop("negative_prompt", "模糊"),
        kwargs.pop("seed", 100),
        kwargs.pop("steps", 4),
        kwargs.pop("guidance", 1.0),
        kwargs.pop("resolution_scale", 1.0),
        kwargs.pop("resolution_timing", folder_module.RESOLUTION_TIMINGS[0]),
    )


def _patch_fake_pipeline(monkeypatch, *, fail_on_sample: set[int] | None = None):
    fail_on_sample = set(fail_on_sample or ())
    reference_calls: list[dict] = []
    text_calls: list[tuple[str, str]] = []
    sample_calls: list[dict] = []
    cleanup_calls: list[str] = []
    cache_instances: list[Cache] = []

    class RecordingCache(Cache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            cache_instances.append(self)

    monkeypatch.setattr(folder_module, "Cache", RecordingCache)
    monkeypatch.setattr(
        folder_module.runtime,
        "apply_cache_limit",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        folder_module.runtime,
        "flush_caches",
        lambda: cleanup_calls.append("flush"),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "prepare_sampler_components",
        lambda *args: cleanup_calls.append("sampler_prepare") or {"transformer": object()},
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "prepare_encoder",
        lambda *args: cleanup_calls.append("encoder_prepare") or {"fake": True},
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "vae_component",
        lambda *args: cleanup_calls.append("vae_prepare") or object(),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "prepare_moss_reference_image",
        lambda image: image.convert("RGB"),
    )

    def fake_encode_reference(_vae, image, cache, cache_key):
        reference_calls.append(
            {
                "key": cache_key,
                "size": image.size,
                "pixel": image.getpixel((0, 0)),
            }
        )
        cache.get_or_create(
            "moss_ref_encoding",
            cache_key,
            lambda: np.zeros((1, 128, 4, 4), dtype=np.float32),
        )

    monkeypatch.setattr(
        folder_module.pipeline,
        "encode_moss_reference_image",
        fake_encode_reference,
    )

    def fake_encode_text(_entry, _components, text, cache, cache_key, **kwargs):
        text_calls.append((text, cache_key))
        cache.get_or_create("prompt_encoding", cache_key, lambda: (None, None))

    monkeypatch.setattr(folder_module.pipeline, "encode_text", fake_encode_text)
    monkeypatch.setattr(
        folder_module.pipeline,
        "moss_reference_kv_cache_key",
        lambda _model, reference_key, conditioning_key: (
            f"kv:{reference_key}:{conditioning_key}"
        ),
    )

    def fake_run_sampler(_entry, _model, _components, params, cache):
        sample_calls.append(dict(params))
        index = len(sample_calls) - 1
        if index in fail_on_sample:
            raise RuntimeError(f"sample failure {index}")
        positive_kv = params["moss_kv_cache_key"]
        negative_kv = f"kv:{params['moss_ref_cache_key']}:{params['negative_encoding_key']}"
        cache.get_or_create("moss_ref_kv", positive_kv, lambda: object())
        cache.get_or_create("moss_ref_kv", negative_kv, lambda: object())
        latent_key = f"latent-{index}"
        cache.get_or_create(
            "component_weights",
            latent_key,
            lambda: np.zeros((1, 1, 128), dtype=np.float32),
        )
        return MlxLatentHandle(
            kind="noise",
            shape=(1, 1, 128),
            dtype="float32",
            cache_key=latent_key,
            model="moss",
            height=params["height"],
            width=params["width"],
        )

    monkeypatch.setattr(folder_module.pipeline, "run_sampler", fake_run_sampler)
    monkeypatch.setattr(
        folder_module.pipeline,
        "decode_latents",
        lambda _entry, _components, _latent, _height, _width: len(sample_calls),
    )
    monkeypatch.setattr(
        folder_module.image_mod,
        "to_pil",
        lambda decoded, batch_index=-1: (
            Image.new("RGB", (32, 32), (int(decoded), 80, 120)),
        ),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "clear_compiled_predict_caches",
        lambda: cleanup_calls.append("compiled"),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "release_sampler_components",
        lambda *args: cleanup_calls.append("sampler_release"),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "release_encoder",
        lambda *args: cleanup_calls.append("encoder_release"),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "release_vae_component",
        lambda *args: cleanup_calls.append("vae_release"),
    )
    return reference_calls, text_calls, sample_calls, cleanup_calls, cache_instances


def test_moss_folder_edit_natural_sort_numbering_and_per_image_cache_cleanup(
    tmp_path, monkeypatch
):
    comfy_input, input_dir = _make_input(tmp_path)
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(
        folder_module,
        "_comfy_output_directory",
        lambda: tmp_path / "comfy-output",
    )
    Image.new("RGB", (64, 64), (10, 0, 0)).save(input_dir / "image10.png")
    Image.new("RGB", (64, 64), (20, 0, 0)).save(input_dir / "image2.jpg")
    Image.new("RGB", (64, 64), (30, 0, 0)).save(input_dir / "image1.png")
    (input_dir / "notes.txt").write_text("ignored", encoding="utf-8")

    reference_calls, text_calls, sample_calls, cleanup_calls, cache_instances = (
        _patch_fake_pipeline(monkeypatch)
    )
    result = _run(
        folder_module.MlxMossFolderEdit(),
        "mlx_moss/batch",
        _handles(),
    )

    assert [call["pixel"][0] for call in reference_calls] == [30, 20, 10]
    assert [Path(path).name for path in result[1].splitlines()] == [
        "0000_image1.png",
        "0001_image2.png",
        "0002_image10.png",
    ]
    assert [call["seed"] for call in sample_calls] == [100, 101, 102]
    assert all(call["ref_cache_key"] == "" for call in sample_calls)
    assert all(call["moss_ref_cache_key"] for call in sample_calls)
    assert len({call["moss_ref_cache_key"] for call in sample_calls}) == 3
    assert len({key for _text, key in text_calls}) == 6
    assert [record["status"] for record in json.loads(result[2])] == [
        "success",
        "success",
        "success",
    ]
    assert result[3] == 3
    assert len(result[0].images) == 3
    assert cleanup_calls[-5:] == [
        "compiled",
        "sampler_release",
        "encoder_release",
        "vae_release",
        "flush",
    ]
    assert cleanup_calls.count("flush") >= 4
    assert all(
        not cache.keys(bucket)
        for cache in cache_instances
        for bucket in (
            "moss_ref_encoding",
            "moss_ref_kv",
            "prompt_encoding",
            "component_weights",
        )
    )


@pytest.mark.parametrize("timing", folder_module.RESOLUTION_TIMINGS)
def test_moss_folder_edit_resolution_timing_controls_sampling_and_output(
    tmp_path, monkeypatch, timing
):
    comfy_input, input_dir = _make_input(tmp_path, f"resolution-{timing}")
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(
        folder_module,
        "_comfy_output_directory",
        lambda: tmp_path / "comfy-output",
    )
    Image.new("RGB", (128, 80), (10, 0, 0)).save(input_dir / "source.png")

    reference_calls, _text_calls, sample_calls, _cleanup_calls, _cache_instances = (
        _patch_fake_pipeline(monkeypatch)
    )
    result = _run(
        folder_module.MlxMossFolderEdit(),
        f"mlx_moss/resolution-{timing}",
        _handles(),
        resolution_scale=0.5,
        resolution_timing=timing,
    )

    # The fake MOSS preparation preserves the input dimensions, so the
    # expected 16-pixel aligned target is deterministic here.
    if timing == "采样时降分辨率":
        assert reference_calls[0]["size"] == (128, 80)
        assert (sample_calls[0]["width"], sample_calls[0]["height"]) == (64, 32)
    elif timing == "先降分辨率":
        assert reference_calls[0]["size"] == (64, 32)
        assert (sample_calls[0]["width"], sample_calls[0]["height"]) == (64, 32)
    else:
        assert reference_calls[0]["size"] == (128, 80)
        assert (sample_calls[0]["width"], sample_calls[0]["height"]) == (128, 80)

    output_path = Path(result[1])
    with Image.open(output_path) as generated:
        assert generated.size == ((64, 32) if timing == "后降分辨率" else (32, 32))
    report = json.loads(result[2])[0]
    assert report["resolution_scale"] == 0.5
    assert report["resolution_timing"] == timing
    assert (report["width"], report["height"]) == (64, 32)


def test_moss_folder_edit_resolution_scale_is_16_pixel_aligned():
    image = Image.new("RGB", (130, 98))
    assert folder_module.MlxMossFolderEdit._scaled_dimensions(image, 1.0) == (128, 96)
    assert folder_module.MlxMossFolderEdit._scaled_dimensions(image, 0.01) == (16, 16)


def test_moss_folder_edit_rejects_invalid_resolution_parameters():
    vae, clip, model = _handles()
    node = folder_module.MlxMossFolderEdit()
    with pytest.raises(ValueError, match="分辨率系数必须是大于 0"):
        node.edit_folder(
            "mlx_moss/not-used",
            vae,
            clip,
            model,
            "保持主体",
            "模糊",
            0,
            1,
            1.0,
            0,
        )
    with pytest.raises(ValueError, match="未知分辨率缩放时机"):
        node.edit_folder(
            "mlx_moss/not-used",
            vae,
            clip,
            model,
            "保持主体",
            "模糊",
            0,
            1,
            1.0,
            1.0,
            "invalid",
        )


def test_moss_folder_edit_input_types_include_resolution_parameters():
    inputs = folder_module.MlxMossFolderEdit.INPUT_TYPES()["required"]
    assert inputs["input_directory"] == (
        "STRING",
        {
            "default": "",
            "multiline": False,
            "tooltip": "可点击「选择输入文件夹」上传图片，也可直接输入 ComfyUI 服务器上的文件夹绝对路径",
        },
    )
    assert inputs["resolution_scale"] == (
        "FLOAT",
        {
            "default": 1.0,
            "min": 0.01,
            "max": 4.0,
            "step": 0.01,
            "tooltip": "按 MOSS 参考图预处理后的宽高乘以此系数；例如 0.5 表示目标尺寸减半，具体作用取决于缩放时机",
        },
    )
    assert inputs["resolution_timing"] == (
        list(folder_module.RESOLUTION_TIMINGS),
        {
            "default": folder_module.RESOLUTION_TIMINGS[0],
            "tooltip": "选择只缩小采样画布、先缩小参考图后采样，或在生成完成后缩小最终图片",
        },
    )


def test_moss_folder_workflow_serializes_resolution_values_in_order():
    workflow_path = (
        Path(__file__).resolve().parents[1] / "workflows" / "moss-folder-edit.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    node = next(item for item in workflow["nodes"] if item["type"] == "MlxMossFolderEdit")
    assert node["widgets_values"] == [
        "",
        folder_module.DEFAULT_PROMPT,
        folder_module.DEFAULT_NEGATIVE_PROMPT,
        42,
        50,
        3.0,
        1.0,
        folder_module.RESOLUTION_TIMINGS[0],
    ]


def test_moss_folder_picker_supports_upload_and_manual_path():
    picker_path = Path(__file__).resolve().parents[1] / "web" / "moss_folder_picker.js"
    picker = picker_path.read_text(encoding="utf-8")
    assert 'const TARGET_NODE = "MlxMossFolderEdit";' in picker
    assert "const subfolder = `${UPLOAD_ROOT}/${folderName}`;" in picker
    assert "node.properties.mlx_moss_output_directory = `output/moss-folder-edit/${subfolder}`;" in picker
    assert "directoryWidget.hidden = false;" in picker
    assert "node.widgets.splice(directoryIndex + 1, 0, button);" in picker


def test_moss_folder_token_and_manual_path_resolve_to_directories(tmp_path, monkeypatch):
    comfy_input, selected = _make_input(tmp_path, "selected")
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)

    assert folder_module._resolve_input_directory("mlx_moss/selected") == selected
    manual = tmp_path / "manually-entered"
    manual.mkdir()
    assert folder_module._resolve_input_directory(str(manual)) == manual.resolve()
    with pytest.raises(ValueError, match="手动输入的文件夹不存在或不是目录"):
        folder_module._resolve_input_directory(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match=r"不能包含 \.\."):
        folder_module._resolve_input_directory("mlx_moss/../outside")
    with pytest.raises(ValueError, match="必须来自.*选择输入文件夹"):
        folder_module._resolve_input_directory("other-folder/batch")


def test_moss_manual_path_outputs_inside_comfy_output(tmp_path, monkeypatch):
    comfy_input = tmp_path / "comfy-input"
    comfy_input.mkdir()
    comfy_output = tmp_path / "comfy-output"
    manual = tmp_path / "manual-source"
    manual.mkdir()
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(folder_module, "_comfy_output_directory", lambda: comfy_output)

    output = folder_module._output_directory(manual.resolve())
    assert output.parent == (comfy_output / folder_module.OUTPUT_ROOT / "manual").resolve()
    assert output.is_dir()


def test_moss_folder_edit_resume_skips_valid_png_and_retries_corrupt_output(
    tmp_path, monkeypatch
):
    comfy_input, input_dir = _make_input(tmp_path, "resume")
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(
        folder_module,
        "_comfy_output_directory",
        lambda: tmp_path / "comfy-output",
    )
    for index in range(1, 4):
        Image.new("RGB", (64, 64), (index, 0, 0)).save(input_dir / f"image{index}.png")

    output = (
        tmp_path
        / "comfy-output"
        / folder_module.OUTPUT_ROOT
        / folder_module.UPLOAD_ROOT
        / "resume"
    )
    output.mkdir(parents=True)
    Image.new("RGB", (20, 20), "white").save(output / "0001_image2.png")
    (output / "0002_image3.png").write_bytes(b"not a png")

    _reference_calls, _text_calls, sample_calls, _cleanup_calls, _cache_instances = (
        _patch_fake_pipeline(monkeypatch)
    )
    result = _run(
        folder_module.MlxMossFolderEdit(),
        "mlx_moss/resume",
        _handles(),
    )

    assert [call["seed"] for call in sample_calls] == [100, 102]
    report = json.loads(result[2])
    assert [record["status"] for record in report] == ["success", "skipped", "success"]
    assert result[3] == 3
    assert (output / "0002_image3.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_moss_folder_edit_continues_after_one_image_failure(tmp_path, monkeypatch):
    comfy_input, input_dir = _make_input(tmp_path, "failure")
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(
        folder_module,
        "_comfy_output_directory",
        lambda: tmp_path / "comfy-output",
    )
    for index in range(1, 4):
        Image.new("RGB", (64, 64), (index, 0, 0)).save(input_dir / f"image{index}.png")

    _reference_calls, _text_calls, sample_calls, cleanup_calls, cache_instances = (
        _patch_fake_pipeline(monkeypatch, fail_on_sample={1})
    )
    result = _run(
        folder_module.MlxMossFolderEdit(),
        "mlx_moss/failure",
        _handles(),
    )

    assert len(sample_calls) == 3
    assert result[3] == 2
    report = json.loads(result[2])
    assert [record["status"] for record in report] == ["success", "error", "success"]
    assert "sample failure 1" in report[1]["error"]
    assert cleanup_calls.count("flush") >= 4
    assert all(
        not cache.keys(bucket)
        for cache in cache_instances
        for bucket in ("moss_ref_encoding", "moss_ref_kv", "prompt_encoding", "component_weights")
    )


def test_moss_folder_edit_all_skipped_does_not_initialize_pipeline(tmp_path, monkeypatch):
    comfy_input, input_dir = _make_input(tmp_path, "all-skipped")
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(
        folder_module,
        "_comfy_output_directory",
        lambda: tmp_path / "comfy-output",
    )
    for index in range(1, 3):
        Image.new("RGB", (24, 24), (index, 0, 0)).save(input_dir / f"image{index}.png")
    output = (
        tmp_path
        / "comfy-output"
        / folder_module.OUTPUT_ROOT
        / folder_module.UPLOAD_ROOT
        / "all-skipped"
    )
    output.mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (24, 24), "white").save(output / f"000{index}_image{index + 1}.png")

    monkeypatch.setattr(
        folder_module.pipeline,
        "prepare_sampler_components",
        lambda *args: pytest.fail("all-skipped folder must not prepare sampler components"),
    )
    result = _run(
        folder_module.MlxMossFolderEdit(),
        "mlx_moss/all-skipped",
        _handles(),
    )

    report = json.loads(result[2])
    assert [record["status"] for record in report] == ["skipped", "skipped"]
    assert result[3] == 2
    assert len(result[0].images) == 2


def test_moss_folder_edit_validates_folder_scope_and_empty_images(tmp_path, monkeypatch):
    comfy_input, input_dir = _make_input(tmp_path, "validation")
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(
        folder_module,
        "_comfy_output_directory",
        lambda: tmp_path / "comfy-output",
    )
    node = folder_module.MlxMossFolderEdit()

    manual = tmp_path / "manual-empty"
    manual.mkdir()
    with pytest.raises(ValueError, match="没有可读取的图片"):
        _run(node, str(manual), _handles())
    with pytest.raises(ValueError, match=r"不能包含 \.\."):
        _run(node, "mlx_moss/../validation", _handles())
    with pytest.raises(ValueError, match="必须来自"):
        _run(node, "other-folder/validation", _handles())
    with pytest.raises(ValueError, match="没有可读取的图片"):
        _run(node, "mlx_moss/validation", _handles())

    outside = tmp_path / "outside.png"
    Image.new("RGB", (16, 16), "black").save(outside)
    linked = input_dir / "linked.png"
    linked.symlink_to(outside)
    with pytest.raises(ValueError, match="不允许符号链接"):
        _run(node, "mlx_moss/validation", _handles())


def test_moss_folder_edit_initialization_failure_still_releases_resources(tmp_path, monkeypatch):
    comfy_input, input_dir = _make_input(tmp_path, "init-failure")
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(
        folder_module,
        "_comfy_output_directory",
        lambda: tmp_path / "comfy-output",
    )
    Image.new("RGB", (64, 64), "white").save(input_dir / "image.png")
    cleanup_calls: list[str] = []
    monkeypatch.setattr(folder_module.runtime, "apply_cache_limit", lambda *args, **kwargs: None)
    monkeypatch.setattr(folder_module.runtime, "flush_caches", lambda: cleanup_calls.append("flush"))
    monkeypatch.setattr(
        folder_module.pipeline,
        "prepare_sampler_components",
        lambda *args: (_ for _ in ()).throw(RuntimeError("init failure")),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "clear_compiled_predict_caches",
        lambda: cleanup_calls.append("compiled"),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "release_sampler_components",
        lambda *args: cleanup_calls.append("sampler"),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "release_encoder",
        lambda *args: cleanup_calls.append("encoder"),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "release_vae_component",
        lambda *args: cleanup_calls.append("vae"),
    )

    with pytest.raises(RuntimeError, match="init failure"):
        _run(node=folder_module.MlxMossFolderEdit(), input_token="mlx_moss/init-failure", handles=_handles())

    assert cleanup_calls == ["compiled", "sampler", "encoder", "vae", "flush"]


def test_moss_folder_edit_cache_keys_separate_reference_and_condition_branches():
    node = folder_module.MlxMossFolderEdit()
    reference_a = node._cache_key(
        "moss_folder_reference",
        source="image-a.png",
        digest="same-digest",
        seed=1,
    )
    reference_b = node._cache_key(
        "moss_folder_reference",
        source="image-b.png",
        digest="same-digest",
        seed=1,
    )
    positive = node._cache_key(
        "moss_folder_prompt",
        source="image-a.png",
        branch="positive",
        digest="same-digest",
    )
    negative = node._cache_key(
        "moss_folder_prompt",
        source="image-a.png",
        branch="negative",
        digest="same-digest",
    )
    assert len({reference_a, reference_b, positive, negative}) == 4


def _sampler_params(**overrides):
    params = {
        "seed": 7,
        "steps": 2,
        "height": 64,
        "width": 64,
        "batch_size": 1,
        "guidance": 1.0,
        "scheduler_name": "moss_flow_match_euler",
        "positive_encoding_key": "positive",
        "negative_encoding_key": "negative",
    }
    params.update(overrides)
    return params


def _sampler_model(family: str, model_path: str) -> MlxModelHandle:
    return MlxModelHandle(
        model_type=family,
        model_path=model_path,
        quantize=8,
        precision="bfloat16",
        compile=False,
        compile_cache_limit=0,
        cache_key=f"{family}-model",
    )


def _sampler_defn(family: str, default_config: str = "flux2"):
    return SimpleNamespace(
        family=family,
        supported=True,
        notes="",
        default_config=default_config,
    )


def _fake_sample(calls, name):
    def sample(*args):
        calls.append((name, args))
        return np.zeros((1, 1, 8), dtype=np.float32)

    return sample


def test_run_sampler_routes_moss_reference_sampling_to_moss_path(monkeypatch):
    calls: list[tuple[str, tuple]] = []
    monkeypatch.setattr(pipeline, "_sample_moss", _fake_sample(calls, "moss"))
    monkeypatch.setattr(
        pipeline.weights,
        "config_for_path",
        lambda *args: SimpleNamespace(model_name="legacy", supports_guidance=True),
    )
    params = _sampler_params(
        moss_ref_cache_key="reference-cache",
        moss_kv_cache_key="reference-kv",
    )
    cache = Cache()
    pipeline.run_sampler(
        _sampler_defn("moss"),
        _sampler_model("moss", "OpenMOSS-Team--MOSS-Image-SFT"),
        {"transformer": object()},
        params,
        cache,
    )

    assert [name for name, _args in calls] == ["moss"]
    assert calls[0][1][4] is cache
    assert calls[0][1][5] == 1.0


def test_run_sampler_routes_moss_without_reference_to_same_moss_path(monkeypatch):
    calls: list[tuple[str, tuple]] = []
    monkeypatch.setattr(pipeline, "_sample_moss", _fake_sample(calls, "moss"))
    cache = Cache()

    pipeline.run_sampler(
        _sampler_defn("moss"),
        _sampler_model("moss", "OpenMOSS-Team--MOSS-Image-SFT"),
        {"transformer": object()},
        _sampler_params(),
        cache,
    )

    assert [name for name, _args in calls] == ["moss"]
    assert calls[0][1][3].get("moss_ref_cache_key", "") == ""


def test_run_sampler_rejects_moss_reference_key_for_non_moss_family(monkeypatch):
    monkeypatch.setattr(
        pipeline.weights,
        "config_for_path",
        lambda *args: SimpleNamespace(model_name="flux2", supports_guidance=True),
    )
    with pytest.raises(ValueError, match="moss_ref_cache_key 只支持 MOSS"):
        pipeline.run_sampler(
            _sampler_defn("flux2"),
            _sampler_model("flux2", "flux2-test"),
            {"transformer": object()},
            _sampler_params(moss_ref_cache_key="wrong-family"),
            Cache(),
        )


@pytest.mark.parametrize(
    ("family", "model_path", "config_name", "sample_name", "ref_key"),
    [
        ("flux2", "flux2-test", "flux2", "flux2", ""),
        ("flux3", "FLUX.3-Klein", "flux3", "flux3", ""),
        ("qwen_image", "Qwen-Image", "qwen_image", "qwen_image", ""),
        ("qwen_edit", "Qwen-Image-Edit", "qwen_edit", "qwen_edit", "legacy-ref"),
    ],
)
def test_run_sampler_preserves_legacy_dispatch(
    monkeypatch, family, model_path, config_name, sample_name, ref_key
):
    calls: list[tuple[str, tuple]] = []
    target = getattr(pipeline, f"_sample_{sample_name}")
    monkeypatch.setattr(pipeline, target.__name__, _fake_sample(calls, sample_name))
    monkeypatch.setattr(
        pipeline.weights,
        "config_for_path",
        lambda *args: SimpleNamespace(model_name=config_name, supports_guidance=True),
    )
    pipeline.run_sampler(
        _sampler_defn(family, config_name),
        _sampler_model(family, model_path),
        {"transformer": object()},
        _sampler_params(ref_cache_key=ref_key),
        Cache(),
    )
    assert [name for name, _args in calls] == [sample_name]


def test_moss_folder_edit_is_registered():
    import comfyui_mlx_gen

    assert comfyui_mlx_gen.NODE_CLASS_MAPPINGS["MlxMossFolderEdit"] is folder_module.MlxMossFolderEdit