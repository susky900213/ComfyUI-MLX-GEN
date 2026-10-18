"""Dependency-light MOSS-Image-SFT MLX contract tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen.moss import transformer as transformer_module  # noqa: E402
from comfyui_mlx_gen.moss.config import MOSSConfig, text_encoder_config  # noqa: E402
from comfyui_mlx_gen.moss.latent import create_noise, unpack_latents  # noqa: E402
from comfyui_mlx_gen.moss.loader import validate_index  # noqa: E402
from comfyui_mlx_gen.moss.prompt import format_prompt, pad_conditioning  # noqa: E402
from comfyui_mlx_gen.moss.scheduler import MOSSImageScheduler  # noqa: E402
from comfyui_mlx_gen.types import (  # noqa: E402
    MOSS_FAMILY,
    detect_model_family,
    entry_for,
    model_types,
    validate_model_family,
)


MODEL_ROOT = Path.home() / "ComfyUI-Shared" / "models" / "mlx"
MOSS_ROOT = MODEL_ROOT / "transformer" / "OpenMOSS-Team--MOSS-Image-SFT"


def test_moss_is_an_independent_family_and_matches_official_config():
    assert MOSS_FAMILY in model_types()
    assert entry_for(MOSS_FAMILY).default_steps == 50
    assert detect_model_family("OpenMOSS-Team--MOSS-Image-SFT") == MOSS_FAMILY
    config = MOSSConfig.from_path(MOSS_ROOT)
    assert config == MOSSConfig()
    validate_model_family(MOSS_FAMILY, "OpenMOSS-Team--MOSS-Image-SFT")
    try:
        validate_model_family("flux2", "OpenMOSS-Team--MOSS-Image-SFT")
    except NotImplementedError as exc:
        assert "不会把 MOSS 路由到 Flux.2" in str(exc)
    else:
        raise AssertionError("MOSS checkpoint was accepted by Flux.2")


def test_moss_text_encoder_contract():
    config = text_encoder_config(MODEL_ROOT / "text_encoder" / "OpenMOSS-Team--MOSS-Image-SFT")
    assert config["model_type"] == "qwen3_5"
    assert config["text_config"]["hidden_size"] == 4096
    assert config["text_config"]["num_hidden_layers"] == 32
    assert config["text_config"]["vocab_size"] == 248320


def test_moss_shard_indexes_are_header_consistent():
    transformer_shards, transformer_map = validate_index(MOSS_ROOT, "diffusion_pytorch_model")
    text_shards, text_map = validate_index(
        MODEL_ROOT / "text_encoder" / "OpenMOSS-Team--MOSS-Image-SFT", "model"
    )
    assert len(transformer_shards) == 3
    assert len(transformer_map) == 463
    assert len(text_shards) == 4
    assert len(text_map) == 759


def test_moss_latent_is_spatial_128_channel_and_seeded():
    first = create_noise(123, 256, 512)
    second = create_noise(123, 256, 512)
    assert tuple(first.shape) == (1, 128, 16, 32)
    assert tuple(unpack_latents(first, 256, 512).shape) == tuple(first.shape)
    mx.eval(first, second)
    assert bool(mx.all(first == second).item())


def test_moss_prompt_and_conditioning_contract():
    assert format_prompt("cat") == (
        "<|im_start|>user\ncat<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    condition = pad_conditioning(
        [
            mx.zeros((1, 3, 4096), dtype=mx.bfloat16),
            mx.zeros((1, 5, 4096), dtype=mx.bfloat16),
        ]
    )
    assert tuple(condition.embeddings.shape) == (2, 5, 4096)
    assert condition.attention_mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]


def test_moss_scheduler_preserves_nearest_grid_denoising_order():
    scheduler = MOSSImageScheduler(shift=5.0, training_shift=1.0)
    timesteps = scheduler.set_timesteps(50)
    assert len(timesteps) == 50
    assert float(scheduler.sigmas[0]) == 1.0
    assert float(scheduler.sigmas[-2]) > 0.0
    assert float(scheduler.sigmas[-1]) == 0.0
    assert bool((scheduler.sigmas[:-1] >= scheduler.sigmas[1:]).all())
    sample = mx.zeros((1, 128, 2, 2), dtype=mx.float32)
    output = scheduler.step(mx.ones_like(sample), timesteps[0], sample)
    mx.eval(output)
    assert tuple(output.shape) == tuple(sample.shape)


def test_moss_transformer_forward_and_negative_skip_layer(monkeypatch):
    # Keep this smoke test tiny while exercising the exact forward topology.
    monkeypatch.setattr(transformer_module, "validate_config", lambda config: config)
    config = MOSSConfig(
        dim=64,
        ffn_dim=128,
        freq_dim=16,
        latent_channels=8,
        num_heads=8,
        num_layers=3,
        text_dim=32,
    )
    model = transformer_module.MOSSImageTransformer(config)
    latent = mx.random.normal((1, 8, 2, 3), key=mx.random.key(1))
    text = mx.random.normal((1, 4, 32), key=mx.random.key(2))
    mask = mx.array([[True, True, True, False]])
    normal = model(latent, 0.5, text, mask)
    skipped = model(latent, 0.5, text, mask, skip_layers=(1,))
    mx.eval(normal, skipped)
    assert tuple(normal.shape) == (1, 8, 2, 3)
    assert tuple(skipped.shape) == (1, 8, 2, 3)
    assert not bool(mx.all(normal == skipped).item())