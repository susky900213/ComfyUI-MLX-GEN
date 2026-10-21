"""Small dependency-local tests for the native Fibo-2 MLX numerical path."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from comfyui_mlx_gen.fibo2 import mlx_backend  # noqa: E402
from comfyui_mlx_gen.fibo2.backend import Fibo2ModelHandle  # noqa: E402
from comfyui_mlx_gen.fibo2.mlx_scheduler import Fibo2Scheduler  # noqa: E402
from comfyui_mlx_gen.fibo2.mlx_transformer import Fibo2Transformer, compute_gist_count  # noqa: E402
from comfyui_mlx_gen.fibo2.weight_definition import Fibo2WeightDefinition  # noqa: E402
from comfyui_mlx_gen.fibo2.weight_mapping import Fibo2WeightMapping  # noqa: E402


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

    def fake_create_and_load(*args, **kwargs):
        loaded_components.append(args[1])
        return FakeModule(), None

    class FakeCache:
        def get_or_create(self, _type, _key, factory):
            return factory(), False

    for role in ("transformer", "text_encoder", "processor", "vae"):
        (tmp_path / role).mkdir()
    (tmp_path / "transformer" / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(mlx_backend, "_local_root", lambda _path: tmp_path)
    monkeypatch.setattr(mlx_backend.components, "create_and_load", fake_create_and_load)
    monkeypatch.setattr(mlx_backend.components, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(mlx_backend.runtime, "cache_key", lambda _value: "fibo2-test-bundle")

    bundle = mlx_backend.load_bundle(
        Fibo2ModelHandle(model_path="fibo2-test", dtype="bfloat16"),
        cache=FakeCache(),
    )

    assert isinstance(bundle, mlx_backend.Fibo2MLXBundle)
    assert loaded_components == ["transformer", "text_encoder", "vae"]


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