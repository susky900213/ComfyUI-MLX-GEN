from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import comfyui_mlx_gen.code2video as code2video
from comfyui_mlx_gen.nodes.code2video import MlxCode2Video


def test_request_prompt_contains_fixed_render_contract():
    request = code2video.VideoRequest("画一个圆", 320, 180, 12, 2)
    prompt = code2video.build_generation_prompt(request)
    assert "320x180" in prompt
    assert "12 fps" in prompt
    assert "out/final.mp4" in prompt
    assert "shell=True" in prompt


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", 1800), (" 1800 ", 1800), (1800, 1800), ("1800.0", 1800)],
)
def test_parse_render_timeout_accepts_legacy_empty_and_numeric_values(value, expected):
    assert code2video.parse_render_timeout(value) == expected


@pytest.mark.parametrize("value", ["abc", "1800.5", "0", "86401", True])
def test_parse_render_timeout_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        code2video.parse_render_timeout(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "mlx_vlm"),
        ("", "mlx_vlm"),
        ("MLX-VLM", "mlx_vlm"),
        ("local-transformers", "transformers"),
        ("transformers", "transformers"),
    ],
)
def test_normalize_backend_accepts_current_and_legacy_values(value, expected):
    assert code2video.normalize_backend(value) == expected


@pytest.mark.parametrize("value", ["", "metal", "gpu", "MPS", "cuda", "cpu"])
def test_normalize_device_accepts_current_and_legacy_values(value):
    expected = {"": "auto", "metal": "mps", "gpu": "auto"}.get(value, value.lower())
    assert code2video.normalize_device(value) == expected


@pytest.mark.parametrize("normalizer, value", [(code2video.normalize_backend, "bad"), (code2video.normalize_device, "bad")])
def test_normalize_backend_and_device_reject_unknown_values(normalizer, value):
    with pytest.raises(ValueError):
        normalizer(value)


@pytest.mark.parametrize(
    ("backend", "device", "render_timeout", "expected"),
    [
        ("mlx_vlm", "auto", "1800", ("mlx_vlm", "auto", 1800)),
        ("auto", 1800, "", ("mlx_vlm", "auto", 1800)),
        ("mps", "900", "", ("mlx_vlm", "mps", 900)),
    ],
)
def test_normalize_node_options_handles_current_and_stale_widget_layouts(
    backend, device, render_timeout, expected
):
    assert code2video.normalize_node_options(backend, device, render_timeout) == expected


def test_normalize_node_options_does_not_hide_invalid_current_values():
    with pytest.raises(ValueError, match="不支持的 backend"):
        code2video.normalize_node_options("bad", "auto", "1800")

    with pytest.raises(ValueError, match="不支持的 device"):
        code2video.normalize_node_options("mlx_vlm", "bad", "1800")


def test_extract_python_code_accepts_one_python_block():
    code = code2video.extract_python_code(
        "说明文字\n```python\nfrom pathlib import Path\nPath('out').mkdir()\nPath('out/final.mp4').touch()\n```"
    )
    assert code.startswith("from pathlib import Path")
    assert code.endswith("\n")


def test_extract_python_code_accepts_path_join_output():
    code = code2video.extract_python_code(
        "```py\nfrom pathlib import Path\nPath('out').mkdir()\noutput = Path('out') / 'final.mp4'\n```"
    )
    assert "final.mp4" in code


def test_extract_python_code_accepts_unfenced_self_contained_python():
    code = code2video.extract_python_code(
        "from pathlib import Path\n"
        "output_dir = Path('out')\n"
        "output_dir.mkdir(exist_ok=True)\n"
        "output_path = output_dir / 'final.mp4'\n"
        "output_path.touch()\n"
    )
    assert code.startswith("from pathlib import Path")
    assert "output_dir / 'final.mp4'" in code


def test_extract_python_code_accepts_direct_out_file_path():
    code = code2video.extract_python_code(
        "```python\n"
        "from pathlib import Path\n"
        "Path('out/final.mp4').touch()\n"
        "```"
    )
    assert "out/final.mp4" in code


@pytest.mark.parametrize(
    "response",
    [
        "no code",
        "```python\npass\n```\n```python\npass\n```",
        "```python\nthis is not valid python\n```",
        "```python\nprint('no output')\n```",
    ],
)
def test_extract_python_code_rejects_ambiguous_or_invalid_blocks(response):
    with pytest.raises(ValueError):
        code2video.extract_python_code(response)


def test_run_renderer_writes_task_artifacts_and_validates_video(tmp_path):
    pytest.importorskip("av")
    code = """
from pathlib import Path
import av
import numpy as np

out = Path('out')
out.mkdir(exist_ok=True)
container = av.open(str(out / 'final.mp4'), 'w')
stream = container.add_stream('mpeg4', rate=4)
stream.width = 32
stream.height = 24
stream.pix_fmt = 'yuv420p'
for index in range(4):
    frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index * 40, dtype=np.uint8), format='rgb24')
    for packet in stream.encode(frame):
        container.mux(packet)
for packet in stream.encode():
    container.mux(packet)
container.close()
"""
    task = tmp_path / "task"
    final, metadata = code2video.run_renderer(
        task,
        code,
        python_executable=sys.executable,
        timeout=30,
        expected_width=32,
        expected_height=24,
        expected_fps=4,
        expected_duration=1,
    )
    assert final == task / "out" / "final.mp4"
    assert final.stat().st_size > 0
    assert metadata["width"] == 32
    assert metadata["height"] == 24
    assert (task / "render.py").is_file()
    assert (task / "render.log").is_file()
    assert (task / "VERIFY.md").is_file()


def test_run_renderer_normalizes_reversed_pillow_rectangle_at_boundary(tmp_path):
    pytest.importorskip("av")
    pytest.importorskip("PIL")
    code = """
from pathlib import Path
import av
from PIL import Image, ImageDraw

out = Path('out')
out.mkdir(exist_ok=True)
container = av.open(str(out / 'final.mp4'), 'w')
stream = container.add_stream('mpeg4', rate=2)
stream.width = 32
stream.height = 24
stream.pix_fmt = 'yuv420p'
for _ in range(2):
    image = Image.new('RGB', (32, 24), 'black')
    draw = ImageDraw.Draw(image)
    # This is the same zero-progress shape that Pillow rejects without the
    # adapter compatibility preamble.
    draw.rectangle([22, 4, 22 - int(18 * 0.0), 18], fill='blue')
    frame = av.VideoFrame.from_image(image)
    for packet in stream.encode(frame):
        container.mux(packet)
for packet in stream.encode():
    container.mux(packet)
container.close()
"""
    final, metadata = code2video.run_renderer(
        tmp_path / "task",
        code,
        python_executable=sys.executable,
        timeout=30,
        expected_width=32,
        expected_height=24,
        expected_fps=2,
        expected_duration=1,
    )
    assert final.is_file()
    assert metadata["width"] == 32
    assert metadata["height"] == 24


def test_run_renderer_removes_stale_output_before_failed_render(tmp_path):
    task = tmp_path / "task"
    stale = task / "out" / "final.mp4"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale failed output")
    with pytest.raises(RuntimeError, match="退出码"):
        code2video.run_renderer(
            task,
            "raise RuntimeError('new render failure')\n",
            python_executable=sys.executable,
            timeout=30,
        )
    assert not stale.exists()
    assert "new render failure" in (task / "render.log").read_text(encoding="utf-8")


def test_run_renderer_timeout_kills_process_group(tmp_path):
    code = """
import time
time.sleep(60)
"""
    with pytest.raises(RuntimeError, match="超时"):
        code2video.run_renderer(tmp_path / "task", code, timeout=1)
    assert (tmp_path / "task" / "render.log").is_file()


def test_run_renderer_rejects_output_symlink_outside_task(tmp_path):
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"not a video")
    code = f"""
from pathlib import Path
Path('out').mkdir(exist_ok=True)
Path('out/final.mp4').symlink_to({str(outside)!r})
"""
    with pytest.raises(RuntimeError, match="任务目录内"):
        code2video.run_renderer(tmp_path / "task", code, timeout=30)


def test_node_contract_is_native_video_and_save_video_compatible():
    definition = MlxCode2Video.INPUT_TYPES()
    assert list(definition["required"]) == [
        "model_path",
        "prompt",
        "task_root",
        "width",
        "height",
        "fps",
        "duration",
        "max_new_tokens",
        "temperature",
        "backend",
        "device",
        "render_timeout",
    ]
    assert isinstance(definition["required"]["model_path"][0], list)
    assert definition["required"]["model_path"][1]["default"] in definition["required"]["model_path"][0]
    assert definition["required"]["backend"][0] == list(code2video.SUPPORTED_BACKENDS)
    assert definition["required"]["backend"][1]["default"] == "mlx_vlm"
    assert definition["required"]["device"][0] == list(code2video.SUPPORTED_DEVICES)
    assert definition["required"]["device"][1]["default"] == "auto"
    assert definition["required"]["render_timeout"][0] == "STRING"
    assert definition["required"]["render_timeout"][1]["default"] == "1800"
    assert MlxCode2Video.RETURN_TYPES == ("VIDEO", "STRING", "STRING")


def test_local_models_exposes_valid_transformers_and_mlx_directories(monkeypatch, tmp_path):
    monkeypatch.setattr(code2video.paths, "MODEL_ROOT", tmp_path)
    transformer_root = tmp_path / "transformer"
    transformer_root.mkdir()

    for name in ("Code2Video", "Code2Video-MLX"):
        model = transformer_root / name
        model.mkdir()
        (model / "config.json").write_text(
            '{"model_type":"qwen3_5",'
            '"architectures":["Qwen3_5ForConditionalGeneration"]}',
            encoding="utf-8",
        )
        (model / "model-00001.safetensors").touch()

    monkeypatch.setattr(
        code2video,
        "DEFAULT_MODEL_PATH",
        str(transformer_root / "Code2Video-MLX"),
    )
    assert code2video.local_models() == ("Code2Video-MLX", "Code2Video")
    assert code2video.resolve_model_selection("Code2Video-MLX") == str(
        (transformer_root / "Code2Video-MLX").resolve()
    )


def test_model_choices_keeps_legacy_transformers_absolute_path(monkeypatch, tmp_path):
    legacy = tmp_path / "Code2Video"
    legacy.mkdir()
    (legacy / "config.json").write_text(
        '{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}',
        encoding="utf-8",
    )
    (legacy / "model.safetensors").touch()
    monkeypatch.setattr(code2video, "local_models", lambda: ("Code2Video-MLX", "Code2Video"))
    monkeypatch.setattr(code2video, "DEFAULT_TRANSFORMERS_MODEL_PATH", str(legacy))
    choices = code2video.model_choices()
    assert choices[:2] == ("Code2Video-MLX", "Code2Video")
    assert str(legacy) in choices
    assert str(legacy.resolve()) in choices


def test_node_model_path_is_a_local_model_combo(monkeypatch):
    monkeypatch.setattr(code2video, "local_models", lambda: ("Code2Video-MLX",))
    monkeypatch.setattr(code2video, "DEFAULT_TRANSFORMERS_MODEL_PATH", "/missing/Code2Video")
    definition = MlxCode2Video.INPUT_TYPES()
    model_path = definition["required"]["model_path"]
    assert model_path[0][0] == "Code2Video-MLX"
    assert "Code2Video-MLX" in model_path[0]
    assert model_path[1]["default"] == "Code2Video-MLX"


def test_node_generate_wires_model_renderer_and_native_video(monkeypatch, tmp_path):
    request_seen = {}
    renderer_seen = {}
    task = tmp_path / "task"
    final = task / "out" / "final.mp4"
    task.mkdir()
    (task / "out").mkdir()
    final.write_bytes(b"fake")

    def fake_generate_code(model_path, request, **kwargs):
        request_seen["request"] = request
        request_seen["kwargs"] = kwargs
        return "print('render')\n", "raw model response"

    monkeypatch.setattr(code2video, "create_task_directory", lambda _root: task)
    monkeypatch.setattr(code2video, "generate_code", fake_generate_code)
    monkeypatch.setattr(code2video, "write_spec", lambda *_args: None)
    monkeypatch.setattr(
        code2video,
        "run_renderer",
        lambda task_dir, code, **kwargs: (
            renderer_seen.update(kwargs) or (final, {"width": 320, "height": 180})
        ),
    )
    monkeypatch.setattr(code2video, "video_from_file", lambda path: ("VIDEO", path))

    result = MlxCode2Video().generate(
        "/model",
        "画一个圆",
        str(tmp_path),
        320,
        180,
        12,
        2,
        128,
        0.0,
        "auto",
        1800,
        "",
    )
    assert result[0] == ("VIDEO", final)
    assert result[1] == "print('render')\n"
    assert result[2] == str(task)
    assert request_seen["request"].width == 320
    assert request_seen["kwargs"]["backend"] == "mlx_vlm"
    assert request_seen["kwargs"]["device"] == "auto"
    assert request_seen["kwargs"]["response_path"] == task / "model_response.txt"
    assert renderer_seen["timeout"] == 1800


def test_node_generation_failure_writes_task_diagnostics(monkeypatch, tmp_path):
    task = tmp_path / "task"
    monkeypatch.setattr(code2video, "create_task_directory", lambda _root: task)

    def fail_generate(*_args, **_kwargs):
        raise ValueError("checkpoint load failed")

    monkeypatch.setattr(code2video, "generate_code", fail_generate)
    with pytest.raises(RuntimeError, match="error.md"):
        MlxCode2Video().generate(
            "/model",
            "画一个圆",
            str(tmp_path),
            320,
            180,
            12,
            2,
            128,
            0.0,
            "mlx_vlm",
            "cpu",
            30,
        )
    assert "model_load_or_code_generation" in (task / "error.md").read_text()


def test_generate_code_uses_processor_and_model_without_loading_real_checkpoint(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = {}

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["template_kwargs"] = kwargs
            return {
                "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }

        def batch_decode(self, generated, **kwargs):
            calls["decoded_tokens"] = generated
            return [
                "```python\nfrom pathlib import Path\nPath('out').mkdir()\n"
                "Path('out') / 'final.mp4'\n```"
            ]

    class FakeModel:
        def parameters(self):
            yield torch.zeros(1)

        def generate(self, **kwargs):
            calls["generation_kwargs"] = kwargs
            return torch.tensor([[11, 12, 13, 14]], dtype=torch.long)

    monkeypatch.setattr(
        code2video,
        "_load_transformers_model",
        lambda _path, _device: (FakeProcessor(), FakeModel(), torch.device("cpu")),
    )
    code2video._MODEL_CACHE.clear()
    code, response = code2video.generate_code(
        "/fake/model",
        code2video.VideoRequest("画一个圆", 64, 64, 8, 1),
        backend="transformers",
        device="cpu",
        max_new_tokens=128,
        temperature=0,
    )
    assert "final.mp4" in code
    assert response.startswith("```python")
    assert calls["generation_kwargs"]["do_sample"] is False
    assert "temperature" not in calls["generation_kwargs"]
    assert calls["decoded_tokens"].shape == (1, 2)
    code2video._MODEL_CACHE.clear()


def test_generate_code_uses_native_mlx_vlm_backend(monkeypatch, tmp_path):
    calls = {}

    def fake_load(model_path, device):
        calls["load"] = (model_path, device)
        return "processor", "model", device

    def fake_generate(processor, model, request, **kwargs):
        calls["generate"] = (processor, model, request, kwargs)
        return (
            "```python\nfrom pathlib import Path\nPath('out').mkdir()\n"
            "Path('out/final.mp4').touch()\n```",
            {
                "backend": "mlx_vlm",
                "generation_tokens": 12,
                "generation_tokens_per_second": 3.0,
            },
        )

    monkeypatch.setattr(code2video, "resolve_mlx_device", lambda _device: "metal")
    monkeypatch.setattr(code2video, "_load_mlx_vlm_model", fake_load)
    monkeypatch.setattr(code2video, "_generate_with_mlx", fake_generate)
    code2video._MODEL_CACHE.clear()
    metrics = tmp_path / "generation_metrics.json"

    code, response = code2video.generate_code(
        str(tmp_path / "converted-model"),
        code2video.VideoRequest("画一个圆", 64, 64, 8, 1),
        backend="mlx_vlm",
        device="auto",
        max_new_tokens=128,
        temperature=0,
        metrics_path=metrics,
    )

    assert "final.mp4" in code
    assert response.startswith("```python")
    assert calls["load"] == (str((tmp_path / "converted-model").resolve()), "metal")
    assert calls["generate"][0:2] == ("processor", "model")
    code2video._MODEL_CACHE.clear()


def test_mlx_generation_applies_qwen_chat_template_before_generate(monkeypatch):
    import mlx_vlm

    calls = {}

    class FakeModel:
        config = {"model_type": "qwen3_5"}

    def fake_template(processor, config, prompt, **kwargs):
        calls["template"] = (processor, config, prompt, kwargs)
        return "<formatted-qwen-prompt>"

    def fake_generate(model, processor, prompt, **kwargs):
        calls["generate"] = (model, processor, prompt, kwargs)
        return type(
            "Result",
            (),
            {
                "text": "```python\nfrom pathlib import Path\n"
                "Path('out').mkdir()\nPath('out/final.mp4').touch()\n```",
                "prompt_tokens": 3,
                "generation_tokens": 4,
                "total_tokens": 7,
                "prompt_tps": 1.0,
                "generation_tps": 2.0,
                "peak_memory": 1.5,
                "finish_reason": "stop",
            },
        )()

    monkeypatch.setattr(mlx_vlm, "apply_chat_template", fake_template)
    monkeypatch.setattr(mlx_vlm, "generate", fake_generate)
    response, stats = code2video._generate_with_mlx(
        "processor",
        FakeModel(),
        code2video.VideoRequest("画一个圆", 64, 64, 8, 1),
        max_new_tokens=128,
        temperature=0,
    )

    assert response.startswith("```python")
    assert calls["template"][0:2] == ("processor", {"model_type": "qwen3_5"})
    assert calls["generate"][2] == "<formatted-qwen-prompt>"
    assert stats["generation_tokens"] == 4


def test_mlx_backend_rejects_original_transformers_checkpoint(tmp_path):
    model = tmp_path / "Code2Video"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}',
        encoding="utf-8",
    )
    (model / "model-00001-of-00012.safetensors").write_bytes(b"not-a-real-shard")
    with pytest.raises(ValueError, match="Transformers 原始 Code2Video checkpoint"):
        code2video._validate_mlx_checkpoint(model)