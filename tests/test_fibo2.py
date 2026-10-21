"""Fibo 2.0 isolation and request-contract tests.

These tests do not download the gated checkpoint.  Pipeline execution uses a
small fake Diffusers class, while the conversion tests exercise only the
ComfyUI tensor/PIL boundary.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from comfyui_mlx_gen import NODE_CLASS_MAPPINGS  # noqa: E402
from comfyui_mlx_gen.fibo2 import backend  # noqa: E402
from comfyui_mlx_gen.nodes.clip_loader import MlxClipLoader  # noqa: E402
from comfyui_mlx_gen.nodes.fibo2 import (  # noqa: E402
    _image_batch_to_pil,
    _mask_to_pil,
    _to_comfy_image,
)
from comfyui_mlx_gen.nodes.loader import MlxTransformerLoader  # noqa: E402
from comfyui_mlx_gen.nodes.lora import MlxClipLoraApply, MlxModelLoraApply, MlxVaeLoraApply  # noqa: E402
from comfyui_mlx_gen.nodes.sampler import MlxKSamplerMLX  # noqa: E402
from comfyui_mlx_gen.nodes.sampler import SCHEDULERS  # noqa: E402
from comfyui_mlx_gen.nodes.text_encoder import MlxTextEncoder  # noqa: E402
from comfyui_mlx_gen.nodes.vae_decode import MlxVAEDecoder  # noqa: E402
from comfyui_mlx_gen.nodes.vae_loader import MlxVAELoader  # noqa: E402
from comfyui_mlx_gen.types import MlxConditioning, MlxPilImage  # noqa: E402
from comfyui_mlx_gen import weights as weight_loader  # noqa: E402


def _checkpoint(tmp_path: Path, *, legacy: bool = False) -> Path:
    root = tmp_path / ("legacy" if legacy else "Fibo-2.0-edit-turbo")
    for name in ("transformer", "text_encoder", "vae"):
        (root / name).mkdir(parents=True)
    if legacy:
        transformer = {"_class_name": "LegacyFiboTransformer", "in_channels": 48}
        text = {"architectures": ["SmolLM3ForCausalLM"]}
        vae = {"_class_name": "AutoencoderKLWan"}
    else:
        transformer = {"_class_name": "BriaFibo2Transformer2DModel", "in_channels": 32}
        text = {"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl"}
        vae = {"_class_name": "AutoencoderKLFlux2", "latent_channels": 32}
    (root / "model_index.json").write_text('{"_class_name":"BriaFibo2Pipeline"}', encoding="utf-8")
    (root / "transformer/config.json").write_text(json.dumps(transformer), encoding="utf-8")
    (root / "text_encoder/config.json").write_text(json.dumps(text), encoding="utf-8")
    (root / "vae/config.json").write_text(json.dumps(vae), encoding="utf-8")
    return root


def test_fibo2_diffusers_transformer_index_name_is_supported(tmp_path):
    component = tmp_path / "transformer"
    component.mkdir()
    first = component / "diffusion_pytorch_model-00001-of-00002.safetensors"
    second = component / "diffusion_pytorch_model-00002-of-00002.safetensors"
    extra = component / "not-listed.safetensors"
    mx.save_safetensors(str(first), {"first.weight": mx.ones((2,))})
    mx.save_safetensors(str(second), {"second.weight": mx.zeros((2,))})
    mx.save_safetensors(str(extra), {"must.not.load": mx.full((2,), 9.0)})
    (component / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {
                    "first.weight": first.name,
                    "second.weight": second.name,
                },
            }
        ),
        encoding="utf-8",
    )

    assert weight_loader._alternate_safetensors_index(component, "multi_json").name == (
        "diffusion_pytorch_model.safetensors.index.json"
    )
    assert weight_loader._indexed_safetensor_files(
        component / "diffusion_pytorch_model.safetensors.index.json"
    ) == (first, second)

    definition = types.SimpleNamespace(
        loading_mode="multi_json",
        weight_prefix_filters=None,
        key_transform=None,
        weight_transform=None,
        precision=None,
        mapping_getter=None,
        bulk_transform=None,
    )
    loaded, _, _ = weight_loader._load_indexed_component(
        types.SimpleNamespace(),
        definition,
        component,
        component / "diffusion_pytorch_model.safetensors.index.json",
    )
    assert set(loaded) == {"first", "second"}
    assert set(loaded["first"]) == {"weight"}
    assert set(loaded["second"]) == {"weight"}
    assert mx.all(loaded["first"]["weight"] == 1.0).item()
    assert mx.all(loaded["second"]["weight"] == 0.0).item()


def test_fibo2_diffusers_index_rejects_path_traversal(tmp_path):
    index = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="路径非法"):
        weight_loader._indexed_safetensor_files(index)


def test_fibo2_uses_unified_nodes_and_keeps_backend_unregistered():
    assert "Fibo2ModelLoader" not in NODE_CLASS_MAPPINGS
    assert "Fibo2Sampler" not in NODE_CLASS_MAPPINGS
    assert NODE_CLASS_MAPPINGS["MlxTransformerLoader"] is MlxTransformerLoader
    assert NODE_CLASS_MAPPINGS["MlxClipLoader"] is MlxClipLoader
    assert NODE_CLASS_MAPPINGS["MlxVAELoader"] is MlxVAELoader
    assert NODE_CLASS_MAPPINGS["MlxKSamplerMLX"] is MlxKSamplerMLX
    assert NODE_CLASS_MAPPINGS["MlxVaeLoraApply"] is MlxVaeLoraApply
    assert MlxKSamplerMLX.RETURN_TYPES == ("latents", "images")


def test_fibo2_backend_import_has_no_eager_diffusers_import():
    assert "diffusers" not in sys.modules


def test_fibo2_legacy_diffusers_error_is_actionable(monkeypatch):
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.__version__ = "0.38.0"
    fake_diffusers.__path__ = []
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr(backend, "_legacy_fibo_module_available", lambda _: True)

    with pytest.raises(backend.Fibo2BackendUnavailable) as error:
        backend._pipeline_class()

    message = str(error.value)
    assert "0.38.0" in message
    assert "legacy" in message
    assert backend.FIBO2_DIFFUSERS_COMMIT in message
    assert sys.executable in message


def test_fibo2_scheduler_alias_is_kept_for_old_workflows():
    assert "fibo_flow_match_euler" in SCHEDULERS
    assert "fibo_flow_match_euler" in MlxKSamplerMLX.INPUT_TYPES()["required"]["scheduler"][0]


def test_fibo2_local_checkpoint_is_identified_and_component_paths_resolve(tmp_path):
    root = _checkpoint(tmp_path)
    info = backend.inspect_model_path(root / "transformer")
    assert info["family"] == "fibo2"
    assert info["path"] == str(root.resolve())
    assert info["pipeline"] is True
    assert info["transformer"] is True
    assert backend.is_fibo2_model_path(root)


def test_unified_loaders_and_lora_nodes_create_fibo2_handles(tmp_path, monkeypatch):
    root = _checkpoint(tmp_path)
    monkeypatch.setattr(backend.paths, "MODEL_ROOT", tmp_path)
    model = MlxTransformerLoader().load("fibo2", str(root), 8, "bfloat16", True, 2)[0]
    clip = MlxClipLoader().load("fibo2", "text_encoder", str(root), "bfloat16", 512, 0)[0]
    vae = MlxVAELoader().load("fibo2", str(root), "bfloat16", 8, "vae")[0]
    assert isinstance(model, backend.Fibo2ModelHandle)
    assert isinstance(clip, backend.Fibo2ClipHandle)
    assert isinstance(vae, backend.Fibo2VaeHandle)
    assert model.model_type == clip.model_type == vae.model_type == "fibo2"

    model_lora = MlxModelLoraApply().apply(model, "model.safetensors", 0.7)[0]
    clip_lora = MlxClipLoraApply().apply(clip, "clip.safetensors", 0.4)[0]
    with pytest.raises(NotImplementedError, match="不会回退到 Torch/Diffusers"):
        MlxVaeLoraApply().apply(vae, "vae.safetensors", 0.5)
    assert model_lora.loras[0].strength == 0.7
    assert clip_lora.loras[0].strength == 0.4


def test_fibo2_text_encoder_only_forwards_condition_text(tmp_path):
    root = _checkpoint(tmp_path)
    clip = MlxClipLoader().load("fibo2", "text_encoder", str(root), "bfloat16", 512, 0)[0]
    condition = MlxTextEncoder().encode("a red cube", clip)[0]
    assert condition.text == "a red cube"
    assert condition.clip is clip


def test_legacy_fibo_checkpoint_is_rejected_by_new_backend(tmp_path):
    with pytest.raises(backend.Fibo2ValidationError, match="legacy FIBO"):
        backend.inspect_model_path(_checkpoint(tmp_path, legacy=True))


def test_only_approved_remote_repositories_are_accepted():
    assert backend.inspect_model_path("briaai/Fibo-2.0-dev")["kind"] == "repo"
    assert backend.inspect_model_path("briaai/Fibo-2.0-edit-turbo")["kind"] == "repo"
    with pytest.raises(backend.Fibo2ValidationError, match="官方仓库"):
        backend.inspect_model_path("some-user/Fibo-2.0-edit-turbo")
    with pytest.raises(backend.Fibo2ValidationError):
        backend.inspect_model_path("briaai/Fibo-1.0")


def test_edit_turbo_requires_source_four_steps_and_guidance_one():
    with pytest.raises(backend.Fibo2ValidationError, match="source image"):
        backend.validate_request("briaai/Fibo-2.0-edit-turbo", [], None, 4, 1.0)
    with pytest.raises(backend.Fibo2ValidationError, match="固定 4 步"):
        backend.validate_request("briaai/Fibo-2.0-edit-turbo", [object()], None, 8, 1.0)
    with pytest.raises(backend.Fibo2ValidationError, match="guidance"):
        backend.validate_request("briaai/Fibo-2.0-edit-turbo", [object()], None, 4, 2.0)


def test_fibo2_mask_is_single_source_and_context_images_are_limited():
    backend.validate_request("briaai/Fibo-2.0-edit-turbo", [object()] * 5, None, 4, 1.0)
    with pytest.raises(backend.Fibo2ValidationError, match="单张"):
        backend.validate_request("briaai/Fibo-2.0-edit-turbo", [object(), object()], object(), 4, 1.0)
    with pytest.raises(backend.Fibo2ValidationError, match="最多接收 5"):
        backend.validate_request("briaai/Fibo-2.0-edit-turbo", [object()] * 6, None, 4, 1.0)


def test_loader_normalizes_dtype_and_device_for_a_lazy_handle(tmp_path):
    handle = MlxTransformerLoader().load(
        "fibo2", str(_checkpoint(tmp_path)), 8, "bf16", False, 0
    )[0]
    assert handle.model_path.endswith("Fibo-2.0-edit-turbo")
    assert handle.dtype == "bfloat16"
    assert handle.device == "auto"
    assert '"family": "fibo2"' in handle.cache_key


def test_edit_turbo_node_does_not_treat_reference_as_source(monkeypatch):
    model = backend.Fibo2ModelHandle("briaai/Fibo-2.0-edit-turbo", device="cpu")
    with pytest.raises(ValueError, match="必须连接 image source"):
        MlxKSamplerMLX().sample(
            model=model,
            positive=None,
            negative=None,
            seed=0,
            steps=4,
            width=512,
            height=512,
            batch_size=1,
            guidance=1.0,
            scheduler="flow_match_euler_discrete",
            reference_1=None,
        )


def test_unified_sampler_routes_fibo_to_native_mlx_latent(monkeypatch):
    model = backend.Fibo2ModelHandle("briaai/Fibo-2.0-dev", device="cpu")
    clip = backend.Fibo2ClipHandle("briaai/Fibo-2.0-dev", device="cpu")
    positive = MlxConditioning(clip=clip, text="a red cube")
    negative = MlxConditioning(clip=clip, text="blurry")
    from comfyui_mlx_gen.fibo2 import mlx_backend
    from comfyui_mlx_gen.cache import CACHE
    import mlx.core as mx

    latent = SimpleNamespace(cache_key="fibo-latent", shape=(1, 32, 64, 64))
    captured = {}

    def fake_run(handle, **kwargs):
        captured.update(kwargs)
        assert handle is model
        CACHE.get_or_create("component_weights", latent.cache_key, lambda: mx.zeros(latent.shape))
        return latent

    monkeypatch.setattr(mlx_backend, "run", fake_run)
    monkeypatch.setattr(mlx_backend, "decode_latent", lambda *_args, **_kwargs: mx.zeros((1, 3, 512, 512)))
    latents, images = MlxKSamplerMLX().sample(
        model=model,
        positive=positive,
        negative=negative,
        seed=3,
        steps=4,
        width=512,
        height=512,
        batch_size=1,
        guidance=1.0,
        scheduler="flow_match_euler_discrete",
        clip=clip,
    )
    assert latents is latent
    assert isinstance(images, MlxPilImage)
    assert len(images.images) == 1
    assert captured["prompt"] == "a red cube"
    assert captured["negative_prompt"] == "blurry"


def test_fibo_vae_decoder_reuses_sampler_decoded_images(monkeypatch):
    from comfyui_mlx_gen.cache import Cache
    from comfyui_mlx_gen.nodes import vae_decode as vae_decode_node

    cache = Cache()
    monkeypatch.setattr(vae_decode_node, "CACHE", cache)
    latent = SimpleNamespace(model="fibo2", cache_key="fibo-decoded-images")
    payload = MlxPilImage(images=("decoded-image",))
    cache.get_or_create("fibo2_images", latent.cache_key, lambda: payload)
    vae = backend.Fibo2VaeHandle("/tmp/Fibo-2.0-dev")

    result = MlxVAEDecoder().decode(vae, latent, -1)

    assert result == (payload,)


def test_fake_pipeline_receives_list_image_single_mask_and_is_cached(monkeypatch):
    calls = []

    class FakeGenerator:
        def __init__(self, device):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class FakeTorch:
        float32 = "float32"
        bfloat16 = "bfloat16"
        float16 = "float16"
        Generator = FakeGenerator

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("load", path, kwargs))
            return cls()

        def to(self, device):
            calls.append(("to", device))
            return self

        def __call__(self, **kwargs):
            calls.append(("run", kwargs))
            return SimpleNamespace(images=["fake-pil-image"])

    monkeypatch.setattr(backend, "_PIPELINE_CACHE", {})
    monkeypatch.setattr(backend, "_pipeline_class", lambda: FakePipeline)
    monkeypatch.setattr(backend, "_import_torch", lambda: FakeTorch)
    monkeypatch.setattr(backend, "resolve_device", lambda device: "cpu")
    handle = backend.Fibo2ModelHandle("briaai/Fibo-2.0-edit-turbo", device="cpu")
    source = object()
    mask = object()

    assert backend.run_pipeline(
        handle,
        prompt="caption",
        images=[source],
        mask=mask,
        seed=123,
        steps=4,
        guidance=1.0,
    ) == ["fake-pil-image"]
    assert backend.run_pipeline(
        handle,
        prompt="caption",
        images=[source],
        mask=mask,
        seed=456,
        steps=4,
        guidance=1.0,
    ) == ["fake-pil-image"]

    assert [call[0] for call in calls].count("load") == 1
    load_kwargs = next(call[2] for call in calls if call[0] == "load")
    assert load_kwargs["dtype"] == FakeTorch.bfloat16
    run_kwargs = next(call[1] for call in calls if call[0] == "run")
    assert run_kwargs["image"] == [source]
    assert run_kwargs["mask"] is mask
    assert run_kwargs["generator"].seed == 123
    assert run_kwargs["output_type"] == "pil"


def test_image_mask_and_output_conversion_contract():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")
    image = torch.tensor([[[[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]]], dtype=torch.float32)
    source, = _image_batch_to_pil(image)
    assert source.mode == "RGB"
    assert source.size == (2, 1)

    mask = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
    mask_image = _mask_to_pil(mask, source)
    assert mask_image.mode == "L"
    assert list(mask_image.getdata()) == [0, 255]

    output = _to_comfy_image([Image.new("RGBA", (2, 1), (255, 0, 128, 255))])
    assert tuple(output.shape) == (1, 1, 2, 3)
    assert output.dtype == torch.float32
    np.testing.assert_allclose(output[0, 0, 0].numpy(), [1.0, 0.0, 128 / 255], atol=1e-6)
