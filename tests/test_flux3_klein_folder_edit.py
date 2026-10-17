from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen.cache import Cache
from comfyui_mlx_gen.nodes import flux3_klein_folder_edit as folder_module
from comfyui_mlx_gen.types import (
    MlxClipHandle,
    MlxLatentHandle,
    MlxModelHandle,
    MlxVaeHandle,
)


def _handles(model_path: str = "FLUX.3-Klein"):
    family = "flux3"
    return (
        MlxVaeHandle(
            model_type=family,
            path="FLUX.3-Klein",
            precision="bfloat16",
            quantize=0,
            role="vae",
            cache_key="test-vae",
        ),
        MlxClipHandle(
            model_type=family,
            component="text_encoder",
            source="local",
            path="FLUX.3-Klein",
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

    monkeypatch.setattr(folder_module.runtime, "apply_cache_limit", lambda *args, **kwargs: None)
    monkeypatch.setattr(folder_module.runtime, "flush_caches", lambda: cleanup_calls.append("flush"))
    monkeypatch.setattr(
        folder_module.pipeline,
        "reference_image_cache_key",
        lambda _vae, digest, params: f"ref-{digest[:8]}-{params['family']}",
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "vae_component",
        lambda _handle, _cache: cleanup_calls.append("vae_prepare") or object(),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "encode_flux3_reference_images",
        lambda _vae, images, cache, cache_key, max_images, resize_mode: (
            reference_calls.append(
                {
                    "key": cache_key,
                    "count": len(images),
                    "max_images": max_images,
                    "resize_mode": resize_mode,
                    "size": images[0].size,
                }
            )
            or (np.zeros((1, 4, 128), dtype=np.float32), np.zeros((1, 4, 4)), 64, 48)
        ),
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "prepare_sampler_components",
        lambda *args: cleanup_calls.append("sampler_prepare") or {"fake": True},
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "prepare_encoder",
        lambda *args: cleanup_calls.append("encoder_prepare") or {"fake": True},
    )
    monkeypatch.setattr(
        folder_module.pipeline,
        "prompt_encoding_key",
        lambda _clip, text, ref_key: f"{text}|{ref_key}",
    )

    def fake_encode_text(_entry, _components, text, cache, cache_key):
        text_calls.append((text, cache_key))
        cache.get_or_create("prompt_encoding", cache_key, lambda: (None, None))

    monkeypatch.setattr(folder_module.pipeline, "encode_text", fake_encode_text)
    monkeypatch.setattr(folder_module.pipeline, "clear_compiled_predict_caches", lambda: cleanup_calls.append("compiled"))
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

    def fake_run_sampler(_entry, _model, _components, params, cache):
        sample_calls.append(dict(params))
        index = len(sample_calls) - 1
        if index in fail_on_sample:
            raise RuntimeError(f"sample failure {index}")
        key = f"latent-{index}"
        cache.get_or_create("component_weights", key, lambda: np.zeros((1, 1, 128), dtype=np.float32))
        return MlxLatentHandle(
            kind="packed",
            shape=(1, 1, 128),
            dtype="float32",
            cache_key=key,
            model="flux3",
            height=params["height"],
            width=params["width"],
        )

    monkeypatch.setattr(folder_module.pipeline, "run_sampler", fake_run_sampler)
    monkeypatch.setattr(folder_module.pipeline, "decode_latents", lambda *args: None)
    monkeypatch.setattr(
        folder_module.image_mod,
        "to_pil",
        lambda _decoded, batch_index=-1: (Image.new("RGB", (20, 20), (40, 80, 120)),),
    )
    return reference_calls, text_calls, sample_calls, cleanup_calls, cache_instances


def _run(node, input_token, handles, **kwargs):
    return node.edit_folder(
        input_token,
        *handles,
        kwargs.pop("prompt", "保持主体，只按提示编辑"),
        kwargs.pop("negative_prompt", "模糊"),
        kwargs.pop("seed", 100),
        kwargs.pop("steps", 4),
        kwargs.pop("resize_mode", "auto"),
    )


def test_folder_edit_orders_images_and_uses_shared_reference_key_for_conditions(tmp_path, monkeypatch):
    comfy_input, input_dir = _make_input(tmp_path)
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    Image.new("RGB", (80, 64), (10, 0, 0)).save(input_dir / "image10.png")
    Image.new("RGB", (64, 80), (20, 0, 0)).save(input_dir / "image2.png")
    Image.new("RGB", (48, 64), (30, 0, 0)).save(input_dir / "image1.jpg")
    (input_dir / "ignored.txt").write_text("ignore", encoding="utf-8")

    reference_calls, text_calls, sample_calls, cleanup_calls, cache_instances = _patch_fake_pipeline(monkeypatch)
    monkeypatch.setattr(folder_module, "_comfy_output_directory", lambda: tmp_path / "comfy-output")
    conditioning_calls = []
    real_conditioning = folder_module.MlxConditioning

    def record_conditioning(*args, **kwargs):
        conditioning = real_conditioning(*args, **kwargs)
        conditioning_calls.append(conditioning)
        return conditioning

    monkeypatch.setattr(folder_module, "MlxConditioning", record_conditioning)
    vae, clip, model = _handles()

    result = _run(folder_module.MlxFlux3KleinFolderEdit(), "mlx_flux3_klein/batch", (vae, clip, model))

    assert [call["size"] for call in reference_calls] == [(48, 64), (64, 80), (80, 64)]
    assert [call["count"] for call in reference_calls] == [1, 1, 1]
    assert len(text_calls) == 6
    for positive, negative in zip(text_calls[::2], text_calls[1::2]):
        assert positive[0] == "保持主体，只按提示编辑"
        assert negative[0] == "模糊"
        assert positive[1].split("|")[1] == negative[1].split("|")[1]
    for positive, negative in zip(conditioning_calls[::2], conditioning_calls[1::2]):
        assert positive.ref_cache_key == negative.ref_cache_key
        assert positive.ref_cache_key
    assert len(sample_calls) == 3
    assert [call["seed"] for call in sample_calls] == [100, 101, 102]
    assert all(call["positive_encoding_key"] == text_calls[index * 2][1] for index, call in enumerate(sample_calls))
    assert all(call["negative_encoding_key"] == text_calls[index * 2 + 1][1] for index, call in enumerate(sample_calls))
    assert all(call["ref_cache_key"] == reference_calls[index]["key"] for index, call in enumerate(sample_calls))
    assert all(call["ref_count"] == 1 and call["scheduler_name"] == "flux3_flow_map" for call in sample_calls)
    assert [Path(path).name for path in result[1].splitlines()] == [
        "0000_image1.png",
        "0001_image2.png",
        "0002_image10.png",
    ]
    assert result[3] == 3
    assert [record["status"] for record in json.loads(result[2])] == ["success"] * 3
    assert "sampler_prepare" in cleanup_calls
    assert cleanup_calls[-5:] == ["compiled", "sampler_release", "encoder_release", "vae_release", "flush"]
    assert len(cache_instances) == 1
    assert all(not cache.keys(bucket) for bucket in ("ref_encoding", "prompt_encoding", "component_weights") for cache in cache_instances)


def test_folder_edit_resumes_valid_outputs_and_reprocesses_corrupt_outputs(tmp_path, monkeypatch):
    comfy_input, input_dir = _make_input(tmp_path, "resume")
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(folder_module, "_comfy_output_directory", lambda: tmp_path / "comfy-output")
    for index in range(3):
        Image.new("RGB", (64, 64), (index, 0, 0)).save(input_dir / f"image{index + 1}.png")
    output = tmp_path / "comfy-output" / folder_module.OUTPUT_ROOT / "mlx_flux3_klein" / "resume"
    output.mkdir(parents=True)
    Image.new("RGB", (20, 20), (99, 0, 0)).save(output / "0001_image2.png")
    (output / "0002_image3.png").write_bytes(b"corrupt png")

    _reference_calls, _text_calls, sample_calls, _cleanup_calls, _cache_instances = _patch_fake_pipeline(monkeypatch)
    result = _run(folder_module.MlxFlux3KleinFolderEdit(), "mlx_flux3_klein/resume", _handles())

    assert [call["seed"] for call in sample_calls] == [100, 102]
    report = json.loads(result[2])
    assert [record["status"] for record in report] == ["success", "skipped", "success"]
    assert result[3] == 3
    assert (output / "0002_image3.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_folder_edit_continues_after_one_image_sampling_failure(tmp_path, monkeypatch):
    comfy_input, input_dir = _make_input(tmp_path, "failure")
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(folder_module, "_comfy_output_directory", lambda: tmp_path / "comfy-output")
    for index in range(3):
        Image.new("RGB", (64, 64), (index, 0, 0)).save(input_dir / f"image{index + 1}.png")

    _reference_calls, _text_calls, sample_calls, cleanup_calls, cache_instances = _patch_fake_pipeline(
        monkeypatch, fail_on_sample={1}
    )
    result = _run(folder_module.MlxFlux3KleinFolderEdit(), "mlx_flux3_klein/failure", _handles())

    assert len(sample_calls) == 3
    assert result[3] == 2
    report = json.loads(result[2])
    assert [record["status"] for record in report] == ["success", "error", "success"]
    assert "sample failure 1" in report[1]["error"]
    # Per-image cleanup flushes after every attempt, and batch cleanup still runs.
    assert cleanup_calls.count("flush") >= 4
    assert all(not cache.keys(bucket) for bucket in ("ref_encoding", "prompt_encoding", "component_weights") for cache in cache_instances)


def test_folder_edit_rejects_unsafe_directories_and_symlinks(tmp_path, monkeypatch):
    comfy_input, input_dir = _make_input(tmp_path, "safe")
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(folder_module, "_comfy_output_directory", lambda: tmp_path / "comfy-output")
    Image.new("RGB", (32, 32), "white").save(input_dir / "image.png")
    outside = tmp_path / "outside.png"
    Image.new("RGB", (32, 32), "black").save(outside)

    with pytest.raises(ValueError, match="相对目录"):
        folder_module._resolve_input_directory("/tmp/absolute")
    with pytest.raises(ValueError, match="相对目录"):
        folder_module._resolve_input_directory("mlx_flux3_klein/../outside")
    with pytest.raises(ValueError, match="必须来自"):
        folder_module._resolve_input_directory("other-folder/safe")

    directory_link = comfy_input / "mlx_flux3_klein" / "link"
    directory_link.symlink_to(input_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        folder_module._resolve_input_directory("mlx_flux3_klein/link")

    file_link = input_dir / "linked.png"
    file_link.symlink_to(outside)
    with pytest.raises(ValueError, match="不允许符号链接"):
        folder_module._secure_image_paths(input_dir)


def test_folder_edit_cleans_resources_when_initialization_fails(tmp_path, monkeypatch):
    comfy_input, input_dir = _make_input(tmp_path, "init-failure")
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(folder_module, "_comfy_output_directory", lambda: tmp_path / "comfy-output")
    Image.new("RGB", (64, 64), "white").save(input_dir / "image.png")

    cleanup_calls: list[str] = []
    monkeypatch.setattr(folder_module.runtime, "apply_cache_limit", lambda *args, **kwargs: None)
    monkeypatch.setattr(folder_module.runtime, "flush_caches", lambda: cleanup_calls.append("flush"))
    monkeypatch.setattr(folder_module.pipeline, "prepare_sampler_components", lambda *args: (_ for _ in ()).throw(RuntimeError("init failure")))
    monkeypatch.setattr(folder_module.pipeline, "clear_compiled_predict_caches", lambda: cleanup_calls.append("compiled"))
    monkeypatch.setattr(folder_module.pipeline, "release_sampler_components", lambda *args: cleanup_calls.append("sampler"))
    monkeypatch.setattr(folder_module.pipeline, "release_encoder", lambda *args: cleanup_calls.append("encoder"))
    monkeypatch.setattr(folder_module.pipeline, "release_vae_component", lambda *args: cleanup_calls.append("vae"))

    with pytest.raises(RuntimeError, match="init failure"):
        _run(folder_module.MlxFlux3KleinFolderEdit(), "mlx_flux3_klein/init-failure", _handles())

    assert cleanup_calls == ["compiled", "sampler", "encoder", "vae", "flush"]


def test_folder_edit_keeps_cleaning_after_individual_cleanup_errors(tmp_path, monkeypatch):
    comfy_input, input_dir = _make_input(tmp_path, "cleanup-errors")
    monkeypatch.setattr(folder_module, "_comfy_input_directory", lambda: comfy_input)
    monkeypatch.setattr(folder_module, "_comfy_output_directory", lambda: tmp_path / "comfy-output")
    Image.new("RGB", (64, 64), "white").save(input_dir / "image.png")

    _reference_calls, _text_calls, _sample_calls, _cleanup_calls, cache_instances = _patch_fake_pipeline(monkeypatch)
    cleanup_calls: list[str] = []

    def failing_cleanup(name):
        def cleanup(*args):
            cleanup_calls.append(name)
            raise RuntimeError(f"{name} cleanup failure")

        return cleanup

    monkeypatch.setattr(folder_module.pipeline, "clear_compiled_predict_caches", failing_cleanup("compiled"))
    monkeypatch.setattr(folder_module.pipeline, "release_sampler_components", failing_cleanup("sampler"))
    monkeypatch.setattr(folder_module.pipeline, "release_encoder", failing_cleanup("encoder"))
    monkeypatch.setattr(folder_module.pipeline, "release_vae_component", failing_cleanup("vae"))
    monkeypatch.setattr(folder_module.runtime, "flush_caches", lambda: cleanup_calls.append("flush"))

    result = _run(folder_module.MlxFlux3KleinFolderEdit(), "mlx_flux3_klein/cleanup-errors", _handles())

    assert result[3] == 1
    assert cleanup_calls[-5:] == ["compiled", "sampler", "encoder", "vae", "flush"]
    assert cleanup_calls.count("flush") >= 2
    assert all(not cache.keys(bucket) for bucket in ("ref_encoding", "prompt_encoding", "component_weights") for cache in cache_instances)