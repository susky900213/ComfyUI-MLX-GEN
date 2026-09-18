"""Breeze-TTS-2 与 ComfyUI AUDIO 之间的无状态适配层。

``mlx-audio`` 接受路径或没有采样率元数据的 ``mx.array``。ComfyUI AUDIO 自带
原始采样率，因此克隆模式必须先写成短生命周期 WAV，让 Breeze runtime 正确重采样。
本模块不在导入期依赖 mlx-audio，缺少可选 runtime 时不会影响其它节点注册。
"""

from __future__ import annotations

import hashlib
import math
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

SAMPLE_RATE = 24000
MODES = ("speaker", "voice_clone", "voice_design")
SPEAKERS = tuple(f"S{index}" for index in range(10))


def normalize_reference(audio: Any) -> tuple[np.ndarray, int, str]:
    """校验 ComfyUI AUDIO，取首个 batch、声道求均值并返回内容摘要。"""
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("参考音频必须是 ComfyUI AUDIO（含 waveform 与 sample_rate）")
    waveform = audio["waveform"]
    try:
        # torch Tensor 先搬到 CPU；numpy / 测试替身则直接走 asarray。
        if hasattr(waveform, "detach"):
            # ComfyUI 通常给 float32，但 bfloat16 Tensor 不能直接转 NumPy。
            waveform = waveform.detach().float().cpu().numpy()
        array = np.asarray(waveform, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法读取参考 AUDIO waveform：{exc}") from exc
    if array.ndim != 3:
        raise ValueError(
            f"参考 AUDIO waveform 必须是 [batch, channels, samples]，收到 {array.shape}"
        )
    if array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"参考 AUDIO waveform 不能为空，收到 {array.shape}")
    try:
        sample_rate = int(audio["sample_rate"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"参考 AUDIO sample_rate 必须是正整数：{audio['sample_rate']!r}") from exc
    if sample_rate <= 0:
        raise ValueError(f"参考 AUDIO sample_rate 必须大于 0，收到 {sample_rate}")
    mono = np.mean(array[0], axis=0, dtype=np.float32)
    if not np.all(np.isfinite(mono)):
        raise ValueError("参考 AUDIO 含 NaN 或无穷值")
    mono = np.ascontiguousarray(np.clip(mono, -1.0, 1.0), dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(sample_rate.to_bytes(8, "little", signed=False))
    digest.update(mono.tobytes())
    return mono, sample_rate, digest.hexdigest()


@contextmanager
def temporary_reference_wav(mono: np.ndarray, sample_rate: int) -> Iterator[Path]:
    """写 PCM16 单声道 WAV；上下文退出（包括生成异常）时立即删除。"""
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="comfyui-mlx-breeze-", suffix=".wav", delete=False) as file:
            path = Path(file.name)
        pcm = np.round(np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(int(sample_rate))
            output.writeframes(pcm.tobytes())
        yield path
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def validate_params(params: dict[str, Any]) -> None:
    mode = params["mode"]
    if mode not in MODES:
        raise ValueError(f"未知 Breeze 模式：{mode!r}")
    if not str(params["text"]).strip():
        raise ValueError("Breeze 目标文本不能为空（请填写 positive 文本）")
    if params["speaker"] not in SPEAKERS:
        raise ValueError(f"Breeze 内置说话人必须是 S0–S9，收到 {params['speaker']!r}")
    has_reference = params.get("reference") is not None
    if mode == "voice_clone":
        if not has_reference:
            raise ValueError("Breeze voice_clone 模式必须连接 ref_audio")
        if not str(params["ref_text"]).strip():
            raise ValueError("Breeze voice_clone 模式必须填写与参考音频逐字对应的 breeze_ref_text")
    elif has_reference:
        raise ValueError(f"Breeze {mode} 模式不使用参考音频，请断开 ref_audio")
    if mode == "voice_design" and not str(params["instruction"]).strip():
        raise ValueError("Breeze voice_design 模式必须填写 breeze_instruction")
    if not math.isfinite(float(params["temperature"])) or float(params["temperature"]) < 0:
        raise ValueError("Breeze temperature 必须是大于等于 0 的有限数")
    if not 0 <= float(params["top_p"]) <= 1:
        raise ValueError("Breeze top_p 必须在 [0, 1] 内")
    if int(params["top_k"]) < 0 or int(params["max_tokens"]) <= 0:
        raise ValueError("Breeze top_k 必须非负且 max_tokens 必须为正数")
    if float(params["repetition_penalty"]) <= 0:
        raise ValueError("Breeze repetition_penalty 必须大于 0")
    if not math.isfinite(float(params["cfg_scale"])):
        raise ValueError("Breeze cfg_scale 必须是有限数")


def generate_waveform(model: Any, params: dict[str, Any]) -> tuple[Any, int]:
    """调用 mlx-audio 0.5.1，并返回一维 waveform 与 runtime 报告的采样率。"""
    validate_params(params)
    mode = params["mode"]
    kwargs = {
        "voice": params["speaker"],
        "instruct": params["instruction"] if mode == "voice_design" else None,
        "ref_text": params["ref_text"] if mode == "voice_clone" else None,
        "cfg_scale": float(params["cfg_scale"]),
        "max_tokens": int(params["max_tokens"]),
        "temperature": float(params["temperature"]),
        "top_p": float(params["top_p"]),
        "top_k": int(params["top_k"]),
        "repetition_penalty": float(params["repetition_penalty"]),
        "seed": int(params["seed"]),
        "stream": False,
    }

    def consume(ref_audio=None):
        kwargs["ref_audio"] = ref_audio
        results = list(model.generate(params["text"], **kwargs))
        if not results:
            raise RuntimeError("Breeze-TTS-2 没有返回 GenerationResult")
        result = results[-1]
        if not hasattr(result, "audio"):
            raise RuntimeError("Breeze-TTS-2 GenerationResult 缺少 audio")
        return result.audio, int(getattr(result, "sample_rate", SAMPLE_RATE))

    if mode != "voice_clone":
        return consume()
    mono, sample_rate, _digest = params["reference"]
    with temporary_reference_wav(mono, sample_rate) as path:
        return consume(path)