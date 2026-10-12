from __future__ import annotations

import numpy as np
from PIL import Image

from comfyui_mlx_gen.cache import Cache
from comfyui_mlx_gen.nodes import storyboard_batch_edit as batch_module
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


def test_storyboard_batch_edit_runs_one_reference_and_sample_per_panel(monkeypatch):
    vae, clip, model = _handles()
    node = batch_module.MlxQwenImage21StoryboardBatchEdit()
    panels = np.zeros((3, 32, 32, 3), dtype=np.float32)
    panels[0, :, :, 0] = 1.0
    panels[1, :, :, 1] = 1.0
    panels[2, :, :, 2] = 1.0

    reference_calls: list[tuple[str, int]] = []
    text_calls: list[tuple[str, str]] = []
    sample_calls: list[dict] = []
    decode_calls: list[str] = []

    monkeypatch.setattr(batch_module.runtime, "apply_cache_limit", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_module.pipeline, "vae_component", lambda handle, cache: object())
    monkeypatch.setattr(batch_module.pipeline, "release_vae_component", lambda handle, cache: None)
    monkeypatch.setattr(batch_module.pipeline, "reference_image_cache_key", lambda *args: f"ref-{len(reference_calls)}")

    def fake_encode_reference(vae_module, images, cache, cache_key, *args):
        reference_calls.append((cache_key, len(images)))
        cache.get_or_create(
            "ref_encoding",
            cache_key,
            lambda: {"images": tuple(images), "latents": None, "shapes": ()},
        )

    monkeypatch.setattr(batch_module.pipeline, "encode_qwen21_reference_images", fake_encode_reference)
    monkeypatch.setattr(
        batch_module.pipeline,
        "prepare_encoder",
        lambda entry, clip_handle, cache, cache_key: {"fake": True},
    )
    monkeypatch.setattr(batch_module.pipeline, "release_encoder", lambda clip_handle, cache: None)
    monkeypatch.setattr(
        batch_module.pipeline,
        "prompt_encoding_key",
        lambda clip_handle, text, ref_key: f"{text}:{ref_key}",
    )

    def fake_encode_text(entry, components, text, cache, cache_key, **kwargs):
        text_calls.append((text, kwargs["reference_images"][0].getpixel((0, 0))))
        cache.get_or_create("prompt_encoding", cache_key, lambda: (None, None, None))

    monkeypatch.setattr(batch_module.pipeline, "encode_text", fake_encode_text)
    monkeypatch.setattr(batch_module.pipeline, "prepare_sampler_components", lambda *args: {})
    monkeypatch.setattr(batch_module.pipeline, "clear_compiled_predict_caches", lambda: None)
    monkeypatch.setattr(batch_module.pipeline, "release_sampler_components", lambda *args: None)

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

    monkeypatch.setattr(batch_module.pipeline, "run_sampler", fake_run_sampler)
    monkeypatch.setattr(batch_module.pipeline, "decode_latents", lambda *args: None)
    monkeypatch.setattr(
        batch_module.image_mod,
        "to_pil",
        lambda decoded, batch_index=-1: (Image.new("RGB", (32, 32), "white"),),
    )

    result, = node.edit(
        panels,
        vae,
        clip,
        model,
        "保持每格内容",
        "模糊",
        100,
        2,
        32,
        32,
        1.0,
        "flow_match_euler_discrete",
        "auto",
    )

    assert len(reference_calls) == 3
    assert all(count == 1 for _key, count in reference_calls)
    assert len(text_calls) == 6  # positive + negative for each panel
    assert len(sample_calls) == 3
    assert [call["seed"] for call in sample_calls] == [100, 101, 102]
    assert all(call["batch_size"] == 1 and call["ref_count"] == 1 for call in sample_calls)
    assert len(result.images) == 3
