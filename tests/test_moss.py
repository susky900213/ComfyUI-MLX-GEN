"""Dependency-light MOSS-Image-SFT MLX contract tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import mlx.core as mx
from mlx import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen.moss import transformer as transformer_module  # noqa: E402
from comfyui_mlx_gen.moss.config import MOSSConfig, text_encoder_config  # noqa: E402
from comfyui_mlx_gen.moss.latent import (  # noqa: E402
    create_noise,
    create_zero,
    normalize_flux2_latents,
    patchify_flux2_latents,
    unpack_latents,
)
from comfyui_mlx_gen.moss.loader import MOSSQwenRMSNorm, _text_key, validate_index  # noqa: E402
from comfyui_mlx_gen.moss.prompt import format_prompt, pad_conditioning  # noqa: E402
from comfyui_mlx_gen.moss.scheduler import MOSSImageScheduler, to_mlx_timestep  # noqa: E402
from comfyui_mlx_gen.moss.vae import MOSSVAE  # noqa: E402
from comfyui_mlx_gen.types import (  # noqa: E402
    MlxClipHandle,
    MlxConditioning,
    MlxMossReferenceImage,
    MlxModelHandle,
    MlxVaeHandle,
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


def test_moss_uses_the_official_vae_decoder_widths():
    assert entry_for(MOSS_FAMILY).components["vae"].class_import == (
        "comfyui_mlx_gen.moss.vae:MOSSVAE"
    )
    assert MOSSVAE.DECODER_BLOCK_OUT_CHANNELS == (96, 192, 384, 384)


def test_moss_text_encoder_contract():
    config = text_encoder_config(MODEL_ROOT / "text_encoder" / "OpenMOSS-Team--MOSS-Image-SFT")
    assert config["model_type"] == "qwen3_5"
    assert config["text_config"]["hidden_size"] == 4096
    assert config["text_config"]["num_hidden_layers"] == 32
    assert config["text_config"]["vocab_size"] == 248320


def test_moss_text_encoder_converts_pytorch_depthwise_conv_layout():
    checkpoint = mx.arange(8, dtype=mx.float32).reshape(2, 1, 4)
    converted = _text_key("language_model.layers.5.linear_attn.conv1d.weight", checkpoint)
    assert converted is not None
    key, value = converted
    assert key == "layers.5.linear_attn.conv1d.weight"
    assert tuple(value.shape) == (2, 4, 1)
    assert value.tolist() == [[[0], [1], [2], [3]], [[4], [5], [6], [7]]]
    already_mlx = mx.zeros((2, 4, 1), dtype=mx.float32)
    assert _text_key("language_model.layers.5.linear_attn.conv1d.weight", already_mlx)[1] is already_mlx

    ordinary = mx.zeros((2, 3), dtype=mx.float32)
    ordinary_result = _text_key("language_model.layers.5.input_layernorm.weight", ordinary)
    assert ordinary_result is not None
    assert ordinary_result[0] == "layers.5.input_layernorm.weight"
    assert tuple(ordinary_result[1].shape) == (2, 3)
    assert _text_key("visual.patch_embed.weight", ordinary) is None


def test_moss_qwen_rmsnorm_is_zero_centered_like_transformers():
    norm = MOSSQwenRMSNorm(4)
    norm.weight = mx.array([0.0, 0.1, -0.1, 0.0], dtype=mx.float32)
    value = mx.ones((1, 1, 4), dtype=mx.float32)
    output = norm(value)
    mx.eval(output)
    assert bool(mx.allclose(output, mx.array([[[1.0, 1.1, 0.9, 1.0]]]), atol=1e-5).item())


def test_moss_text_encoder_replaces_qwen_zero_centered_norms():
    from comfyui_mlx_gen.moss.loader import build_text_encoder

    module = build_text_encoder(MODEL_ROOT / "text_encoder" / "OpenMOSS-Team--MOSS-Image-SFT")
    assert isinstance(module.norm, MOSSQwenRMSNorm)
    assert isinstance(module.layers[0].input_layernorm, MOSSQwenRMSNorm)
    assert isinstance(module.layers[3].self_attn.q_norm, MOSSQwenRMSNorm)


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
    assert float(timesteps[0]) == 1000.0
    assert float(timesteps[-1]) > 0.0
    assert float(scheduler.sigmas[-2]) > 0.0
    assert float(scheduler.sigmas[-1]) == 0.0
    assert bool((scheduler.sigmas[:-1] >= scheduler.sigmas[1:]).all())
    sample = mx.zeros((1, 128, 2, 2), dtype=mx.float32)
    output = scheduler.step(mx.ones_like(sample), timesteps[0], sample)
    mx.eval(output)
    assert tuple(output.shape) == tuple(sample.shape)


def test_moss_numpy_timestep_converts_to_an_mlx_tensor():
    scheduler = MOSSImageScheduler()
    timestep = scheduler.set_timesteps(4)[0]
    tensor = to_mlx_timestep(timestep)
    mx.eval(tensor)
    assert tuple(tensor.shape) == (1,)
    assert tensor.dtype == mx.float32
    assert float(tensor.item()) == float(timestep)


def test_moss_timestep_embedding_uses_direct_scheduler_timestep_in_fp32():
    values = transformer_module._timestep_embedding(mx.array([1000.0]), 16)
    half = 8
    frequency = mx.exp(-mx.log(mx.array(10_000.0)) * mx.arange(half, dtype=mx.float32) / half)
    phase = 1000.0 * frequency
    reference = mx.concatenate([mx.cos(phase), mx.sin(phase)])[None]
    mx.eval(values)
    assert values.dtype == mx.float32
    assert tuple(values.shape) == (1, 16)
    assert bool(mx.allclose(values, reference, atol=1e-6, rtol=1e-6).item())


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


def test_moss_quantized_linear_keeps_floating_activation_dtype():
    linear = nn.Linear(32, 64)
    quantized = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=8)
    assert quantized.weight.dtype not in (mx.float16, mx.bfloat16, mx.float32)
    assert transformer_module._activation_dtype(quantized, mx.bfloat16) == mx.bfloat16


def test_moss_transformer_ignores_right_padding_in_text_sequence(monkeypatch):
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
    latent = mx.random.normal((1, 8, 2, 3), key=mx.random.key(3))
    text = mx.random.normal((1, 4, 32), key=mx.random.key(4))
    prefix = model(latent, 0.5, text[:, :2], mx.array([[True, True]]))
    padded = model(latent, 0.5, text, mx.array([[True, True, False, False]]))
    mx.eval(prefix, padded)
    assert bool(mx.all(prefix == padded).item())


def test_moss_flux2_reference_latent_patchify_and_normalize():
    plain = mx.arange(128, dtype=mx.float32).reshape(1, 32, 2, 2)
    patched = patchify_flux2_latents(plain)
    assert tuple(patched.shape) == (1, 128, 1, 1)
    assert patched[0, :8, 0, 0].tolist() == list(range(8))
    normalized = normalize_flux2_latents(
        patched, mean=mx.zeros((128,)), variance=mx.full((128,), 3.0)
    )
    mx.eval(normalized)
    assert bool(mx.allclose(normalized, patched / mx.sqrt(mx.array(3.0001))).item())


def test_moss_reference_kv_cache_is_injected_per_block(monkeypatch):
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
    reference = mx.random.normal((1, 8, 2, 3), key=mx.random.key(11))
    target = mx.zeros_like(reference)
    text = mx.random.normal((1, 4, 32), key=mx.random.key(12))
    mask = mx.array([[True, True, True, False]])
    cache = model.build_reference_kv(reference, 0.0, text, mask)
    injected = model(target, 0.5, text, mask, reference_kv=cache)
    plain = model(target, 0.5, text, mask)
    mx.eval(injected, plain, *[x for layer in cache.layers for x in (layer.keys, layer.values)])
    assert len(cache.layers) == 3
    # Official ref-prefix-only-t0 caches image tokens only: 2×3, not
    # image tokens plus the prompt's three valid text tokens.
    assert tuple(cache.layers[0].keys.shape) == (1, 6, 64)
    assert tuple(cache.layers[0].values.shape) == (1, 6, 64)
    assert cache.text_tokens == 0
    assert tuple(injected.shape) == tuple(plain.shape)
    assert not bool(mx.all(injected == plain).item())


def test_moss_rope_uses_official_split_half_pairs(monkeypatch):
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
    rope = transformer_module.MOSSImageRoPE(config.dim // config.num_heads, config.rope_base)(1, 2, 0)
    x = mx.arange(16, dtype=mx.float32).reshape(1, 2, 1, 8)
    rotated = transformer_module._rope_rotate(x, rope)
    half = x.shape[-1] // 2
    pairs = x.reshape(*x.shape[:-1], 2, half).transpose(0, 1, 2, 4, 3)
    matrices = rope[:, :, None, ...]
    expected = (
        matrices[..., 0] * pairs[..., 0][..., None]
        + matrices[..., 1] * pairs[..., 1][..., None]
    ).transpose(0, 1, 2, 4, 3).reshape(x.shape)
    mx.eval(rope, rotated)
    assert tuple(rope.shape) == (1, 2, 4, 2, 2)
    assert bool(mx.allclose(rotated, expected, atol=1e-6, rtol=1e-6).item())
    # The first rotary pair is (x[0], x[4]), not (x[0], x[1]).
    assert not bool(mx.allclose(rotated, x, atol=1e-6, rtol=1e-6).item())


def test_moss_rope_rotates_in_rope_dtype_before_casting_back():
    rope = transformer_module.MOSSImageRoPE(8, 10_000.0)(1, 2, 0)
    x = mx.arange(16, dtype=mx.bfloat16).reshape(1, 2, 1, 8)
    rotated = transformer_module._rope_rotate(x, rope)
    mx.eval(rotated)
    assert rotated.dtype == x.dtype


def test_moss_rope_uses_comfyui_axis_frequency_endpoint():
    rope = transformer_module.MOSSImageRoPE(128, 10_000.0)
    positions = mx.array([[1.0, 0.0, 0.0, 0.0]], dtype=mx.float32)
    matrices = rope._matrices(positions)
    mx.eval(matrices)
    # Four axes are concatenated, so the first axis occupies the first 16
    # matrices.  Its final exponent is (32 - 2) / 32 = 30 / 32.
    expected_phase = 10_000.0 ** (-30.0 / 32.0)
    expected = mx.array(expected_phase, dtype=mx.float32)
    assert bool(mx.allclose(mx.arctan2(matrices[0, 0, 15, 1, 0], matrices[0, 0, 15, 0, 0]), expected).item())


def test_moss_edit_target_latent_uses_fp32_noise():
    latent = create_noise(42, 32, 48)
    mx.eval(latent)
    assert tuple(latent.shape) == (1, 128, 2, 3)
    assert latent.dtype == mx.float32
    assert not bool(mx.all(latent == 0).item())


def test_moss_reference_kv_cache_key_includes_transformer_and_condition():
    from comfyui_mlx_gen import pipeline

    model = MlxModelHandle(
        model_type="moss",
        model_path="OpenMOSS-Team--MOSS-Image-SFT",
        quantize=0,
        precision="bfloat16",
        compile=False,
        compile_cache_limit=0,
        cache_key="model",
    )
    identity = pipeline.moss_reference_kv_identity_key("reference")
    assert identity
    assert identity != pipeline.moss_reference_kv_identity_key("other-reference")
    key = pipeline.moss_reference_kv_cache_key(model, "reference", "positive")
    assert key
    assert key != pipeline.moss_reference_kv_cache_key(model, "reference", "negative")
    changed_model = MlxModelHandle(
        model_type=model.model_type,
        model_path=model.model_path,
        quantize=model.quantize,
        precision=model.precision,
        compile=model.compile,
        compile_cache_limit=model.compile_cache_limit,
        cache_key="different-model",
    )
    assert key != pipeline.moss_reference_kv_cache_key(changed_model, "reference", "positive")


def test_moss_reference_encoder_populates_reference_kv_metadata(monkeypatch):
    from comfyui_mlx_gen.nodes import moss_reference_encoder as encoder_module
    from comfyui_mlx_gen import pipeline

    vae = MlxVaeHandle(
        model_type="moss",
        path="OpenMOSS-Team--MOSS-Image-SFT",
        precision="float32",
        quantize=0,
        cache_key="vae",
    )
    reference_image = object()
    prepared = SimpleNamespace(width=512, height=512)
    packed = SimpleNamespace(shape=(1, 128, 32, 32))
    monkeypatch.setattr(encoder_module.image_mod, "to_pil_batch", lambda images: (reference_image,))
    monkeypatch.setattr(encoder_module.image_mod, "digest", lambda images: "digest")
    monkeypatch.setattr(pipeline, "prepare_moss_reference_image", lambda image: prepared)
    monkeypatch.setattr(pipeline, "vae_component", lambda handle, cache: object())
    monkeypatch.setattr(pipeline, "encode_moss_reference_image", lambda *args: packed)
    monkeypatch.setattr(pipeline, "release_vae_component", lambda *args: None)

    reference, width, height = encoder_module.MlxMossReferenceImageEncoder().encode(
        vae, object()
    )

    assert (width, height) == (512, 512)
    assert reference.cache_key
    assert reference.kv_cache_key == pipeline.moss_reference_kv_identity_key(reference.cache_key)


def test_moss_sampler_passes_reference_kv_cache_key(monkeypatch):
    from comfyui_mlx_gen.nodes import sampler as sampler_module

    clip = MlxClipHandle(
        model_type="moss",
        component="text_encoder",
        source="local",
        path="OpenMOSS-Team--MOSS-Image-SFT",
        precision="bfloat16",
        max_length=4096,
        cache_key="clip",
    )
    model = MlxModelHandle(
        model_type="moss",
        model_path="OpenMOSS-Team--MOSS-Image-SFT",
        quantize=0,
        precision="bfloat16",
        compile=False,
        compile_cache_limit=0,
        cache_key="model",
    )
    positive = MlxConditioning(clip=clip, text="prompt", encoding_key="positive")
    negative = MlxConditioning(clip=clip, text="", encoding_key="negative")
    reference = MlxMossReferenceImage(
        model_type="moss",
        vae_path="moss-vae",
        count=1,
        height=512,
        width=512,
        cache_key="reference",
        latent_height=32,
        latent_width=32,
    )
    captured = {}
    monkeypatch.setattr(sampler_module.moss_lora, "has_image_edit_adapter", lambda refs: False)
    monkeypatch.setattr(sampler_module.runtime, "apply_cache_limit", lambda *args, **kwargs: None)
    monkeypatch.setattr(sampler_module.pipeline, "prepare_sampler_components", lambda *args: {})
    monkeypatch.setattr(
        sampler_module.pipeline,
        "run_sampler",
        lambda entry, model_handle, comps, params, cache: captured.setdefault("params", params)
        or SimpleNamespace(),
    )
    monkeypatch.setattr(sampler_module.pipeline, "release_sampler_components", lambda *args: None)

    sampler_module.MlxKSamplerMLX().sample(
        model,
        positive,
        negative,
        0,
        4,
        512,
        512,
        1,
        1.0,
        "moss_flow_match_euler",
        moss_reference=reference,
    )

    assert captured["params"]["moss_ref_cache_key"] == "reference"
    assert captured["params"]["moss_kv_cache_key"] == sampler_module.pipeline.moss_reference_kv_cache_key(
        model, "reference", "positive"
    )


def test_moss_sampling_reports_missing_reference_kv_key():
    from comfyui_mlx_gen import pipeline

    class Cache:
        def get(self, type_, key):
            assert type_ == "prompt_encoding"
            return SimpleNamespace(embeddings=None, attention_mask=None), True

    params = {
        "positive_encoding_key": "positive",
        "negative_encoding_key": "negative",
        "moss_ref_cache_key": "reference",
        "batch_size": 1,
    }
    try:
        pipeline._sample_moss(
            object(),
            MlxModelHandle(
                model_type="moss",
                model_path="OpenMOSS-Team--MOSS-Image-SFT",
                quantize=0,
                precision="bfloat16",
                compile=False,
                compile_cache_limit=0,
                cache_key="model",
            ),
            {"transformer": object()},
            params,
            Cache(),
            1.0,
        )
    except RuntimeError as exc:
        assert "moss_kv_cache_key 缺失" in str(exc)
    else:
        raise AssertionError("MOSS sampling did not report a missing reference KV key")


def test_moss_image_edit_lora_requires_a_reference_image(monkeypatch):
    from comfyui_mlx_gen.nodes import sampler as sampler_module

    clip = MlxClipHandle(
        model_type="moss",
        component="text_encoder",
        source="local",
        path="OpenMOSS-Team--MOSS-Image-SFT",
        precision="bfloat16",
        max_length=4096,
        cache_key="clip",
    )
    model = MlxModelHandle(
        model_type="moss",
        model_path="OpenMOSS-Team--MOSS-Image-SFT",
        quantize=0,
        precision="bfloat16",
        compile=False,
        compile_cache_limit=0,
        cache_key="model",
    )
    condition = MlxConditioning(clip=clip, text="prompt", encoding_key="positive")
    negative = MlxConditioning(clip=clip, text="", encoding_key="negative")
    monkeypatch.setattr(sampler_module.moss_lora, "has_image_edit_adapter", lambda refs: True)
    try:
        sampler_module.MlxKSamplerMLX().sample(
            model,
            condition,
            negative,
            0,
            4,
            512,
            512,
            1,
            1.0,
            "moss_flow_match_euler",
        )
    except ValueError as exc:
        assert "必须连接 MlxMossReferenceImageEncoder" in str(exc)
    else:
        raise AssertionError("MOSS Image Edit LoRA was allowed without a reference image")