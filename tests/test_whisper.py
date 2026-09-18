"""MlxWhisperTranscribe 的静态与小波形回归测试（不加载真实大权重）。

运行：
    /opt/anaconda3/envs/py313/bin/python tests/test_whisper.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen import NODE_CLASS_MAPPINGS  # noqa: E402
from comfyui_mlx_gen import paths, runtime, whisper_asr  # noqa: E402

FAILED: list[str] = []


def check(label, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {label}" + (f" | {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def check_raises(label, exc_type, message_part, call):
    try:
        call()
    except exc_type as exc:
        check(label, message_part in str(exc), str(exc))
    except Exception as exc:  # noqa: BLE001
        check(label, False, f"异常类型不对：{type(exc).__name__}: {exc}")
    else:
        check(label, False, f"没有抛出 {exc_type.__name__}")


# -------------------------------------------------------------- 1. 节点与依赖契约
check("MlxWhisperTranscribe 已注册", "MlxWhisperTranscribe" in NODE_CLASS_MAPPINGS)
node_class = NODE_CLASS_MAPPINGS["MlxWhisperTranscribe"]
inputs = node_class.INPUT_TYPES()["required"]
check(
    "ASR 节点输入输出契约完整",
    list(inputs)
    == [
        "audio", "model_path", "language", "temperature", "condition_on_previous_text",
        "initial_prompt",
    ]
    and inputs["audio"][0] == "AUDIO"
    and inputs["language"][0] == whisper_asr.LANGUAGES
    and node_class.RETURN_TYPES == ("STRING", "STRING")
    and node_class.RETURN_NAMES == ("text", "language")
    and node_class.FUNCTION == "transcribe",
    str(inputs),
)
requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
check("requirements 固定 mlx-whisper==0.4.3", "mlx-whisper==0.4.3" in requirements)


# -------------------------------------------------------- 2. 本地 checkpoint 发现与校验
original_root = paths.MODEL_ROOT
with tempfile.TemporaryDirectory() as temporary:
    model_root = Path(temporary)
    transformer_root = model_root / "transformer"
    valid = transformer_root / whisper_asr.DEFAULT_MODEL
    invalid = transformer_root / "not-whisper"
    valid.mkdir(parents=True)
    invalid.mkdir()
    (valid / "config.json").write_text('{"model_type":"whisper"}', encoding="utf-8")
    (valid / "weights.npz").write_bytes(b"test")
    (invalid / "config.json").write_text('{"model_type":"other"}', encoding="utf-8")
    (invalid / "weights.npz").write_bytes(b"test")
    paths.MODEL_ROOT = model_root
    try:
        discovered = whisper_asr.local_models()
        resolved = whisper_asr.resolve_model(whisper_asr.DEFAULT_MODEL)
    finally:
        paths.MODEL_ROOT = original_root
check(
    "仅发现 config.model_type=whisper 且带权重的目录",
    discovered == (whisper_asr.DEFAULT_MODEL,) and resolved == str(valid.resolve()),
    f"{discovered} / {resolved}",
)


# ---------------------------------------------------------- 3. AUDIO 规范化与重采样
left = np.array([0.0, 0.5, 2.0, -2.0], dtype=np.float32)
right = np.array([0.0, -0.5, 0.0, 0.0], dtype=np.float32)
audio = {"waveform": np.stack([left, right])[None, :, :], "sample_rate": 48000}
mono, sample_rate = whisper_asr.normalize_audio(audio)
check(
    "ComfyUI AUDIO 取首 batch、声道均值并 clip",
    sample_rate == 48000 and np.allclose(mono, [0.0, 0.0, 1.0, -1.0]),
    str(mono.tolist()),
)
one_second = np.sin(2 * np.pi * 440 * np.arange(48000, dtype=np.float32) / 48000)
resampled = whisper_asr.resample_to_whisper(one_second, 48000)
check(
    "48 kHz 波形重采样为 mlx-whisper 要求的 16 kHz float32",
    resampled.shape == (16000,) and resampled.dtype == np.float32 and np.all(np.isfinite(resampled)),
    f"{resampled.shape} {resampled.dtype}",
)
check_raises(
    "拒绝非 [batch,channels,samples] 波形",
    ValueError,
    "[batch, channels, samples]",
    lambda: whisper_asr.normalize_audio(
        {"waveform": np.zeros((2, 100), dtype=np.float32), "sample_rate": 16000}
    ),
)


# -------------------------------------------------------- 4. mlx-whisper 调用与释放契约
with tempfile.TemporaryDirectory() as temporary:
    model_root = Path(temporary)
    model_dir = model_root / "transformer" / whisper_asr.DEFAULT_MODEL
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type":"whisper"}', encoding="utf-8")
    (model_dir / "weights.npz").write_bytes(b"test")
    paths.MODEL_ROOT = model_root
    captured = {}
    holder = SimpleNamespace(model=object(), model_path="loaded-model")

    def fake_transcribe(waveform, **kwargs):
        captured["waveform"] = waveform
        captured.update(kwargs)
        return {"text": "  这是逐字转写。  ", "language": "zh", "segments": []}

    fake_package = SimpleNamespace(transcribe=fake_transcribe)
    fake_transcribe_module = SimpleNamespace(ModelHolder=holder)
    original_import = whisper_asr.importlib.import_module
    original_flush = runtime.flush_caches

    def fake_import(name):
        if name == "mlx_whisper":
            return fake_package
        if name == "mlx_whisper.transcribe":
            return fake_transcribe_module
        return original_import(name)

    whisper_asr.importlib.import_module = fake_import
    runtime.flush_caches = lambda: captured.setdefault("flushed", True)
    try:
        result = node_class().transcribe(
            audio={
                "waveform": np.ones((1, 1, 48000), dtype=np.float32) * 0.1,
                "sample_rate": 48000,
            },
            model_path=whisper_asr.DEFAULT_MODEL,
            language="auto",
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt="",
        )
    finally:
        whisper_asr.importlib.import_module = original_import
        runtime.flush_caches = original_flush
        paths.MODEL_ROOT = original_root

check(
    "节点强制逐字转写、关闭时间戳并输出纯 STRING",
    result == ("这是逐字转写。", "zh")
    and captured["task"] == "transcribe"
    and captured["language"] is None
    and captured["temperature"] == 0.0
    and captured["condition_on_previous_text"] is False
    and captured["initial_prompt"] is None
    and captured["word_timestamps"] is False
    and captured["verbose"] is None
    and captured["fp16"] is True
    and captured["path_or_hf_repo"] == str(model_dir.resolve())
    and captured["waveform"].shape == (16000,),
    str(captured),
)
check(
    "转写结束释放 mlx-whisper 模型与 MLX cache",
    holder.model is None and holder.model_path is None and captured.get("flushed") is True,
)


# ---------------------------------------------------------- 5. 自动克隆工作流契约
workflow = json.loads(
    (ROOT / "workflows" / "breeze-tts2-voice-clone-asr.json").read_text(encoding="utf-8")
)
nodes = workflow["nodes"]
links = workflow["links"]
nodes_by_id = {node["id"]: node for node in nodes}
counts = Counter(node["type"] for node in nodes)
check(
    "自动转写声音克隆工作流节点种类与数量完整",
    counts
    == Counter(
        {
            "PrimitiveStringMultiline": 1,
            "LoadAudio": 1,
            "MlxWhisperTranscribe": 1,
            "MlxTransformerLoader": 1,
            "MlxVAELoader": 1,
            "MlxBreezeSampler": 1,
            "MlxVAEDecoder": 1,
            "MlxPilToTorch": 1,
            "SaveAudio": 1,
            "PreviewAudio": 1,
        }
    ),
    str(counts),
)
asr_node = next(node for node in nodes if node["type"] == "MlxWhisperTranscribe")
audio_node = next(node for node in nodes if node["type"] == "LoadAudio")
sampler_node = next(node for node in nodes if node["type"] == "MlxBreezeSampler")
check(
    "参考 AUDIO 同时连接 ASR 与 Breeze，ASR text 连接 ref_text",
    asr_node["widgets_values"] == [whisper_asr.DEFAULT_MODEL, "zh", 0.0, False, ""]
    and audio_node["outputs"][0]["links"] == [2, 4]
    and [item["link"] for item in sampler_node["inputs"]] == [5, 1, 3, None, 4]
    and sampler_node["widgets_values"][1] == "voice_clone",
    f"{asr_node} / {sampler_node}",
)

bad_links = []
for link_id, src_id, src_slot, dst_id, dst_slot, link_type in links:
    src_output = nodes_by_id[src_id]["outputs"][src_slot]
    dst_input = nodes_by_id[dst_id]["inputs"][dst_slot]
    if link_id not in (src_output.get("links") or []):
        bad_links.append(f"link {link_id} 不在源输出")
    if dst_input.get("link") != link_id:
        bad_links.append(f"link {link_id} 不在目标输入")
    if src_output["type"] != link_type or dst_input["type"] != link_type:
        bad_links.append(f"link {link_id} 类型不一致")
check("自动转写工作流 link 与 socket 元数据一致", not bad_links, "; ".join(bad_links))

if FAILED:
    print(f"\n{len(FAILED)} 项失败：{', '.join(FAILED)}")
    raise SystemExit(1)
print("\nMLX Whisper 回归测试全部通过。")