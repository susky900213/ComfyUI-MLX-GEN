"""Raon 节点的无模型契约测试；不下载或实例化 21B checkpoint。"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from comfyui_mlx_gen import raon  # noqa: E402
from comfyui_mlx_gen.nodes.raon import (  # noqa: E402
    RaonModelLoader,
    RaonSpeechChat,
    RaonSpeechChatAudio,
    RaonSTT,
    RaonTTS,
)


def test_raon_runtime_import_is_lazy():
    assert raon.RAON_MODEL == "RAON_MODEL"
    assert "raon.models" not in sys.modules
    assert "raon.utils.processor" not in sys.modules


def test_raon_nodes_are_registered_and_have_contracts():
    assert RaonModelLoader.RETURN_TYPES == (raon.RAON_MODEL,)
    assert RaonTTS.RETURN_TYPES == ("AUDIO",)
    assert RaonSTT.RETURN_TYPES == ("STRING",)
    assert RaonSpeechChat.RETURN_TYPES == ("STRING",)
    assert RaonSpeechChatAudio.RETURN_TYPES == ("AUDIO",)
    assert RaonTTS.INPUT_TYPES()["optional"]["speaker_audio"][0] == "AUDIO"


def test_normalize_audio_downmixes_first_batch_and_clips():
    audio = {
        "waveform": torch.tensor(
            [
                [[-2.0, -0.5, 0.5], [2.0, 0.5, 1.5]],
                [[99.0, 99.0, 99.0], [99.0, 99.0, 99.0]],
            ]
        ),
        "sample_rate": 24000,
    }
    waveform, sample_rate = raon.normalize_audio(audio)
    assert sample_rate == 24000
    assert waveform.dtype == np.float32
    np.testing.assert_allclose(waveform, [-0.0, 0.0, 1.0])


def test_audio_path_writes_mono_pcm16_wav_and_cleans_up():
    audio = {
        "waveform": torch.tensor([[[0.0, 0.5, -1.0]]]),
        "sample_rate": 16000,
    }
    with raon.audio_path(audio) as path:
        wav_path = Path(path)
        assert wav_path.is_file()
        with wave.open(path, "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 16000
            assert wav.getnframes() == 3
            assert wav.readframes(3) == b"\x00\x00\x00@\x01\x80"
    assert not wav_path.exists()


def test_audio_output_is_comfy_shape():
    output = raon._audio_output(torch.tensor([0.0, 0.25, -0.5]), 22050)
    assert output["sample_rate"] == 22050
    assert tuple(output["waveform"].shape) == (1, 1, 3)
    assert output["waveform"].dtype == torch.float32


def test_model_loader_rejects_missing_local_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(raon.paths, "MODEL_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="本地 Raon Speech 模型目录"):
        raon.make_handle("A.X-K2-Raon-Speech-21B-A3B")


def test_model_options_have_default_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(raon.paths, "MODEL_ROOT", tmp_path)
    assert raon.model_options() == (raon.DEFAULT_MODEL,)


def test_pipeline_constructor_receives_upstream_options(monkeypatch):
    calls = []

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(raon, "_import_pipeline_class", lambda model_path: FakePipeline)
    handle = raon.RaonModelHandle(
        "/local/raon",
        device="cpu",
        dtype="float16",
        attn_implementation="fa",
    )

    pipeline = raon._load_pipeline(handle)

    assert isinstance(pipeline, FakePipeline)
    assert calls == [
        (
            ("/local/raon",),
            {"device": "cpu", "dtype": "float16", "attn_implementation": "fa"},
        )
    ]


def test_local_pipeline_file_is_loaded_without_network(monkeypatch, tmp_path):
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    (model_path / "pipeline.py").write_text(
        "class RaonPipeline:\n    pass\n",
        encoding="utf-8",
    )
    original_import = raon.importlib.import_module

    def import_module(name):
        if name in {
            "raon",
            "raon.pipeline",
            "raon_pipeline",
            "transformers.dynamic_module_utils",
        }:
            raise ImportError(f"blocked optional import: {name}")
        return original_import(name)

    monkeypatch.setattr(raon.importlib, "import_module", import_module)

    pipeline_class = raon._import_pipeline_class(str(model_path))

    assert pipeline_class.__name__ == "RaonPipeline"


class _FakePipeline:
    def __init__(self):
        self.tts_calls = []
        self.stt_calls = []
        self.chat_calls = []

    def tts(self, text, speaker_audio=None):
        self.tts_calls.append((text, speaker_audio))
        if speaker_audio is not None:
            assert Path(speaker_audio).is_file()
        return torch.tensor([0.0, 0.25, -0.5]), 22050

    def stt(self, audio, prompt=None):
        self.stt_calls.append((audio, prompt))
        assert Path(audio).is_file()
        return "转录结果"

    def chat(self, messages, *, force_audio_output=False):
        self.chat_calls.append((messages, force_audio_output))
        audio_path = messages[0]["content"][0]["audio"]
        assert Path(audio_path).is_file()
        if force_audio_output:
            return torch.tensor([0.1, -0.1]), 16000
        return "文本回答"


class _RefAudioPipeline:
    def __init__(self):
        self.reference_path = None

    def tts(self, text, *, ref_audio):
        self.reference_path = ref_audio
        assert Path(ref_audio).is_file()
        return torch.tensor([0.0]), 24000


def _patch_pipeline(monkeypatch, pipeline):
    raon._PIPELINES.clear()
    monkeypatch.setattr(raon, "_load_pipeline", lambda handle: pipeline)
    return raon.RaonModelHandle("/local/raon", device="cpu")


def test_synthesize_uses_mock_pipeline_and_cleans_reference_wav(monkeypatch):
    pipeline = _FakePipeline()
    handle = _patch_pipeline(monkeypatch, pipeline)
    audio = {"waveform": torch.tensor([[[0.0, 0.5, -1.0]]]), "sample_rate": 16000}

    output = raon.synthesize(handle, "你好", audio)

    assert tuple(output["waveform"].shape) == (1, 1, 3)
    assert pipeline.tts_calls[0][0] == "你好"
    assert not Path(pipeline.tts_calls[0][1]).exists()


def test_synthesize_supports_ref_audio_pipeline_variant(monkeypatch):
    pipeline = _RefAudioPipeline()
    handle = _patch_pipeline(monkeypatch, pipeline)
    audio = {"waveform": torch.tensor([[[0.0, 0.5]]]), "sample_rate": 16000}

    output = raon.synthesize(handle, "hello", audio)

    assert output["sample_rate"] == 24000
    assert not Path(pipeline.reference_path).exists()


def test_transcribe_uses_mock_pipeline_and_default_prompt(monkeypatch):
    pipeline = _FakePipeline()
    handle = _patch_pipeline(monkeypatch, pipeline)
    audio = {"waveform": torch.tensor([[[0.0, 0.5]]]), "sample_rate": 16000}

    assert raon.transcribe(handle, audio) == "转录结果"

    input_path, prompt = pipeline.stt_calls[0]
    assert prompt is None
    assert not Path(input_path).exists()


def test_chat_text_and_audio_outputs_use_mock_pipeline(monkeypatch):
    pipeline = _FakePipeline()
    handle = _patch_pipeline(monkeypatch, pipeline)
    audio = {"waveform": torch.tensor([[[0.0, 0.5]]]), "sample_rate": 16000}

    assert raon.speech_chat(handle, audio) == "文本回答"
    audio_output = raon.chat(handle, audio, force_audio_output=True)

    assert audio_output["sample_rate"] == 16000
    assert tuple(audio_output["waveform"].shape) == (1, 1, 2)
    assert pipeline.chat_calls[0][1] is False
    assert pipeline.chat_calls[1][1] is True
    for messages, _ in pipeline.chat_calls:
        assert not Path(messages[0]["content"][0]["audio"]).exists()
