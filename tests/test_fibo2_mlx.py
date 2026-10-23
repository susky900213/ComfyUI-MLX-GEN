"""Small dependency-local tests for the native Fibo-2 MLX numerical path."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from comfyui_mlx_gen.fibo2 import mlx_backend  # noqa: E402
from comfyui_mlx_gen.fibo2.backend import Fibo2ClipHandle, Fibo2ModelHandle  # noqa: E402
from comfyui_mlx_gen.fibo2.mlx_scheduler import Fibo2Scheduler  # noqa: E402
from comfyui_mlx_gen.fibo2.mlx_text_encoder import Fibo2TextEncoder  # noqa: E402
from comfyui_mlx_gen.fibo2.mlx_transformer import Fibo2Transformer, compute_gist_count  # noqa: E402
from comfyui_mlx_gen.fibo2.weight_definition import Fibo2WeightDefinition  # noqa: E402
from comfyui_mlx_gen.fibo2.weight_mapping import Fibo2WeightMapping  # noqa: E402
from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_vision_model import preprocess_image  # noqa: E402


def test_fibo2_gist_count_matches_official_linear_quantized_rule():
    assert compute_gist_count([256, 2048, 300], 256, 512, 64, 256, 2048) == [256, 512, 320]


def test_fibo2_transformer_mapping_loads_perceiver_output_norm():
    targets = Fibo2WeightMapping.get_transformer_mapping()
    assert any(target.to_pattern == "perceiver.norm_out.weight" for target in targets)


def test_fibo2_quantization_keeps_timestep_embedder_in_bfloat16():
    class Quantizable:
        def to_quantized(self):
            raise AssertionError("predicate test should not quantize a module")

    module = Quantizable()
    assert Fibo2WeightDefinition.quantization_predicate("t_embedder.mlp.0", module) is False
    assert Fibo2WeightDefinition.quantization_predicate("t_embedder.mlp.2", module) is False
    assert Fibo2WeightDefinition.quantization_predicate("layers.0.attention.to_q", module) is True


def test_fibo2_scheduler_appends_terminal_zero_and_euler_step():
    scheduler = Fibo2Scheduler()
    timesteps = scheduler.set_timesteps(4, 1024)
    assert timesteps.shape == (4,)
    assert scheduler.sigmas.shape == (5,)
    assert float(scheduler.sigmas[0]) == 1.0
    current = mx.ones((1, 2), dtype=mx.float32)
    result = scheduler.step(mx.ones_like(current), current)
    assert result.shape == current.shape
    assert float(result[0, 0]) < 1.0


def test_fibo2_scheduler_keeps_float32_latent_trajectory_for_bfloat16_input():
    scheduler = Fibo2Scheduler()
    scheduler.set_timesteps(2, 1024)
    current = mx.ones((1, 2), dtype=mx.bfloat16)
    result = scheduler.step(mx.ones_like(current), current)
    mx.eval(result)
    assert result.dtype == mx.float32


def test_fibo2_bundle_build_does_not_require_removed_dtype_mapping(monkeypatch, tmp_path):
    class FakeModule:
        def parameters(self):
            return {}

    loaded_components = []
    text_encoder_kwargs = []
    tokenizer_lengths = []

    def fake_create_and_load(*args, **kwargs):
        loaded_components.append(args[1])
        if args[1] == "text_encoder":
            text_encoder_kwargs.append(kwargs.get("class_kwargs"))
        return FakeModule(), None

    def fake_load_tokenizer(*args, **kwargs):
        tokenizer_lengths.append(kwargs.get("max_length"))
        return object()

    class FakeCache:
        def get_or_create(self, _type, _key, factory):
            return factory(), False

    for role in ("transformer", "text_encoder", "processor", "vae"):
        (tmp_path / role).mkdir()
    (tmp_path / "transformer" / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(mlx_backend, "_local_root", lambda _path: tmp_path)
    monkeypatch.setattr(mlx_backend.components, "create_and_load", fake_create_and_load)
    monkeypatch.setattr(mlx_backend.components, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(mlx_backend.runtime, "cache_key", lambda _value: "fibo2-test-bundle")

    bundle = mlx_backend.load_bundle(
        Fibo2ModelHandle(model_path="fibo2-test", dtype="bfloat16"),
        cache=FakeCache(),
        clip=Fibo2ClipHandle(model_path="fibo2-test", max_length=8192),
    )

    assert isinstance(bundle, mlx_backend.Fibo2MLXBundle)
    assert loaded_components == ["transformer", "text_encoder"]
    assert text_encoder_kwargs == [{"max_length": 4096}]
    assert tokenizer_lengths == [262144]
    assert not hasattr(bundle, "vae")


def test_fibo2_module_materialization_is_chunked_and_synchronized(monkeypatch):
    class FakeModule:
        def parameters(self):
            return {f"weight_{index}": mx.array([index]) for index in range(5)}

    eval_sizes = []
    synchronizations = []
    monkeypatch.setattr(mlx_backend.mx, "eval", lambda *values: eval_sizes.append(len(values)))
    monkeypatch.setattr(
        mlx_backend.mx, "synchronize", lambda: synchronizations.append(True)
    )
    monkeypatch.setattr(mlx_backend.runtime, "memory_snapshot", lambda: "test-memory")

    mlx_backend._materialize_module(FakeModule(), "test", chunk=2)

    assert eval_sizes == [2, 2, 1]
    assert synchronizations == [True, True, True]


def test_fibo2_bundle_cache_identity_excludes_runtime_only_options(monkeypatch, tmp_path):
    class FakeModule:
        def parameters(self):
            return {}

    class RecordingCache:
        def __init__(self):
            self.keys = []

        def get_or_create(self, _type, key, factory):
            self.keys.append(key)
            return factory(), False

    for role in ("transformer", "text_encoder", "processor"):
        (tmp_path / role).mkdir()
    (tmp_path / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mlx_backend, "_local_root", lambda _path: tmp_path)
    monkeypatch.setattr(
        mlx_backend.components,
        "create_and_load",
        lambda *args, **kwargs: (FakeModule(), None),
    )
    monkeypatch.setattr(
        mlx_backend.components, "load_tokenizer", lambda *args, **kwargs: object()
    )
    cache = RecordingCache()
    mlx_backend.load_bundle(
        Fibo2ModelHandle(model_path="fibo2-test", compile_cache_limit=2),
        cache,
        clip=Fibo2ClipHandle(model_path="fibo2-test", max_length=4096),
    )
    mlx_backend.load_bundle(
        Fibo2ModelHandle(model_path="fibo2-test", compile_cache_limit=4),
        cache,
        clip=Fibo2ClipHandle(model_path="fibo2-test", max_length=8192),
    )

    assert len(cache.keys) == 2
    assert cache.keys[0] == cache.keys[1]


def test_fibo2_text_encoder_can_exceed_diffusers_default_when_configured():
    class TinyEncoder(Fibo2TextEncoder):
        def __init__(self, max_length):
            self.max_length = max_length

        def _hidden_states(self, token_ids, image_patches, image_grids):
            state = mx.zeros((1, len(token_ids), 1), dtype=mx.float16)
            return [state] * 37

    class LongTokenizer:
        def __call__(self, _text, **_kwargs):
            return {"input_ids": list(range(4097))}

    tokenizer = LongTokenizer()
    with pytest.raises(ValueError, match="max_length=4096"):
        TinyEncoder(4096).encode(tokenizer, "ignored")

    embeds, mask = TinyEncoder(4096).encode(
        tokenizer, "ignored", max_length=8192
    )

    assert embeds.shape == (1, 6, 4097, 3)
    assert mask.shape == (1, 4097)


def test_fibo2_text_encoder_context_bounds_high_resolution_source_to_official_size():
    source = Image.new("RGB", (2112, 2112))
    old_patches, old_grid = preprocess_image(source)
    assert old_patches.shape[0] // 4 == 4356
    assert old_grid == (1, 132, 132)

    (prepared,) = mlx_backend.preprocess_text_encoder_context(
        [source], width=512, height=512
    )
    patches, grid = preprocess_image(prepared)

    assert prepared.size == (512, 512)
    assert grid == (1, 32, 32)
    assert patches.shape[0] // 4 == 256


def test_fibo2_3000_source_reports_each_effective_image_size():
    source = Image.new("RGB", (3000, 3000))
    vae_size = mlx_backend.context_target_sizes([source], width=512, height=512)[0]
    qwen_image = mlx_backend.preprocess_text_encoder_context(
        [source], width=512, height=512
    )[0]
    route = (
        source.width,
        source.height,
        vae_size[0],
        vae_size[1],
        qwen_image.width,
        qwen_image.height,
    )

    assert vae_size == (512, 512)
    assert qwen_image.size == (512, 512)
    assert mlx_backend.format_size_routes([route]) == (
        "图 1：原图 3000×3000 → VAE 512×512 → Qwen3-VL 512×512"
    )


def test_fibo2_text_encoder_context_caps_trained_multi_image_crops_at_768():
    sources = [Image.new("RGB", (1600, 900)), Image.new("RGB", (500, 500))]

    prepared = mlx_backend.preprocess_text_encoder_context(
        sources, width=512, height=512
    )

    assert len(prepared) == 2
    assert max(prepared[0].size) == mlx_backend.FIBO2_TEXT_ENCODER_IMAGE_SIZE
    assert prepared[1].size == (mlx_backend.FIBO2_TEXT_ENCODER_IMAGE_SIZE,) * 2


def test_fibo2_run_has_finite_steps_and_reports_each_long_stage(monkeypatch, capsys):
    class TinyTextEncoder:
        def __init__(self):
            self.calls = 0

        def encode(self, _tokenizer, _prompt, _images=(), **_kwargs):
            self.calls += 1
            return mx.zeros((1, 6, 3, 4)), mx.ones((1, 3), dtype=mx.bool_)

    class TinyTransformer:
        def __init__(self):
            self.calls = 0

        def __call__(self, sample, _timestep, _cond, _mask, _context):
            self.calls += 1
            return mx.zeros_like(sample)

    class FakeCache:
        def __init__(self):
            self.value = None

        def get_or_create(self, _bucket, _key, factory):
            if self.value is None:
                self.value = factory()
                return self.value, False
            return self.value, True

    text_encoder = TinyTextEncoder()
    transformer = TinyTransformer()
    bundle = mlx_backend.Fibo2MLXBundle(transformer, text_encoder, object())

    def fake_load_bundle(*_args, **kwargs):
        callback = kwargs.get("on_component_loaded")
        for label in ("加载 Transformer", "加载 Qwen3-VL", "加载 tokenizer / processor"):
            callback(label)
        return bundle

    monkeypatch.setattr(mlx_backend, "load_bundle", fake_load_bundle)
    monkeypatch.setattr(mlx_backend.runtime, "memory_snapshot", lambda: "test-memory")
    events = []

    latent = mlx_backend.run(
        Fibo2ModelHandle(model_path="fibo2-test"),
        clip=Fibo2ClipHandle(model_path="fibo2-test"),
        prompt="test",
        negative_prompt=None,
        images=None,
        seed=1,
        steps=4,
        guidance=1.0,
        width=16,
        height=16,
        batch_size=1,
        cache=FakeCache(),
        on_progress=lambda label, current, total: events.append((label, current, total)),
    )

    assert latent.shape == (1, 32, 2, 2)
    assert text_encoder.calls == 1
    assert transformer.calls == 4
    assert [current for _label, current, _total in events] == list(range(1, 9))
    assert {total for _label, _current, total in events} == {8}
    assert [label for label, _current, _total in events[-4:]] == [
        "去噪 step 1/4",
        "去噪 step 2/4",
        "去噪 step 3/4",
        "去噪 step 4/4",
    ]
    output = capsys.readouterr().out
    assert "[Fibo-2 去噪] 计划：共 4 步" in output
    assert "[Fibo-2 去噪] 步数 1/4 开始" in output
    assert "[Fibo-2 去噪] 步数 4/4 完成" in output
    assert "预计剩余 0.00s" in output
    assert "Transformer 前向累计 4/4" in output


def test_fibo2_high_resolution_metrics_include_reference_attention_cost():
    context = [mx.zeros((1, 32, 424, 290), dtype=mx.bfloat16)]

    target, reference, joint, token_ratio, attention_ratio = (
        mlx_backend._denoise_sequence_metrics(2320, 3392, context)
    )

    assert target == 30_740
    assert reference == 30_740
    assert joint == 61_480
    assert token_ratio == pytest.approx(30.01953125)
    assert attention_ratio == pytest.approx(901.1722564697266)


def test_fibo2_user_canvas_aligns_to_nearest_16_without_upscaling_ties():
    assert mlx_backend.aligned_dimensions(689, 981) == (688, 976)
    assert mlx_backend.aligned_dimensions(697, 984) == (704, 976)
    assert mlx_backend.aligned_dimensions(960, 960) == (960, 960)
    assert mlx_backend.aligned_dimensions(1, 1) == (16, 16)


def test_fibo2_denoise_heartbeat_reports_liveness_and_stops(monkeypatch):
    messages = []
    monkeypatch.setattr(mlx_backend, "_log", messages.append)
    started = time.perf_counter()

    with mlx_backend._denoise_heartbeat(1, 4, started, interval=0.01):
        time.sleep(0.035)
    count_after_exit = len(messages)
    time.sleep(0.025)

    assert len(messages) == count_after_exit
    assert any("步数 1/4 仍在计算" in message for message in messages)
    assert any("无逐层百分比回调" in message for message in messages)


def test_fibo2_rejects_3000_target_before_loading_bundle(monkeypatch):
    monkeypatch.setattr(
        mlx_backend,
        "load_bundle",
        lambda *_args, **_kwargs: pytest.fail("invalid size must fail before model loading"),
    )

    with pytest.raises(
        ValueError,
        match=r"收到 3000×3000.*width 可用 2992 或 3008.*height 可用 2992 或 3008",
    ):
        mlx_backend.run(
            Fibo2ModelHandle(model_path="fibo2-test"),
            clip=Fibo2ClipHandle(model_path="fibo2-test"),
            prompt="test",
            negative_prompt=None,
            images=None,
            seed=1,
            steps=4,
            guidance=1.0,
            width=3000,
            height=3000,
            batch_size=1,
        )


def test_fibo2_tiny_transformer_forward_has_official_latent_shape():
    model = Fibo2Transformer(
        in_channels=4,
        patch_size=2,
        dim=32,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=4,
        cap_feat_dim=24,
        injection_layer_ids=(1,),
        perceiver_num_layers=1,
        min_num_gist_tokens=2,
        max_num_gist_tokens=4,
        gist_step=2,
        gist_min_text_len=1,
        gist_max_text_len=8,
        axes_dims=(2, 2, 4),
        axes_lens=(32, 32, 32),
    )
    sample = mx.random.normal((1, 4, 4, 4), key=mx.random.key(1))
    context = mx.random.normal((1, 2, 3, 24), key=mx.random.key(2))
    output = model(sample, mx.array([0.5]), context, mx.array([[True, True, True]]))
    mx.eval(output)
    assert output.shape == sample.shape