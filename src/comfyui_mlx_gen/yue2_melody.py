"""从 ComfyUI AUDIO 提取 YuE2 可用的单旋律 ABC。

YuE2 的原生接口接收的是文本 / ABC 乐谱，不是 WAV 或 MP3。这个模块提供一个
轻量的「音频 → 单旋律近似 ABC」桥接：它不把原曲音频送进 YuE2，而是从原曲中
估计主音高、按八分音符量化，再把结果作为 YuE2 的外部 ABC 条件。

复杂编曲、鼓组、和弦或多人声会让主旋律估计出现误差；输出应理解为旋律提示，
而不是专业级扒谱。用户仍可在生成前修改 BPM、音域和门限，或直接替换节点输出
为手写 ABC。
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


TARGET_SAMPLE_RATE = 16_000
FRAME_SIZE = 2_048
HOP_SIZE = 512
MIN_BPM = 40.0
MAX_BPM = 220.0


def validate_abc(text: str) -> str:
    """校验并返回可供 YuE2 使用的最小 ABC 文本。"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("ABC 内容不能为空")
    normalized = re.sub(r"\r\n?", "\n", text)
    lines = normalized.splitlines()
    if not any(re.match(r"^\s*X\s*:\s*\d+", line) for line in lines):
        raise ValueError("ABC 缺少 X: 编号头部")
    key_index = None
    for index, line in enumerate(lines):
        if re.match(r"^\s*K\s*:", line):
            key_index = index
            break
    if key_index is None:
        raise ValueError("ABC 缺少 K: 调性头部")
    # K: 之后允许继续出现 V:/L:/Q: 等字段；这里只检查是否确实有音符或休止符。
    body = "\n".join(lines[key_index + 1 :])
    body = re.sub(r"%.*", "", body)
    if not re.search(r"(?:\^|_|=)?[A-Ga-gzZ]", body):
        raise ValueError("ABC 的 K: 头部之后没有音符或休止符")
    return normalized


def _audio_to_mono(audio: Any) -> tuple[np.ndarray, int]:
    """校验并读取 ComfyUI AUDIO，取第一批并混合为单声道。"""
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("旋律提取输入必须是 ComfyUI AUDIO（含 waveform 与 sample_rate）")
    waveform = audio["waveform"]
    try:
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().float().cpu().numpy()
        array = np.asarray(waveform, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法读取 AUDIO waveform：{exc}") from exc
    if array.ndim != 3:
        raise ValueError(f"AUDIO waveform 必须是 [batch, channels, samples]，收到 {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"AUDIO waveform 不能为空，收到 {array.shape}")
    try:
        sample_rate = int(audio["sample_rate"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AUDIO sample_rate 必须是正整数：{audio['sample_rate']!r}") from exc
    if sample_rate <= 0:
        raise ValueError(f"AUDIO sample_rate 必须大于 0，收到 {sample_rate}")
    mono = np.mean(array[0], axis=0, dtype=np.float32)
    if not np.all(np.isfinite(mono)):
        raise ValueError("AUDIO 含 NaN 或无穷值")
    return np.ascontiguousarray(np.clip(mono, -1.0, 1.0)), sample_rate


def _resample_linear(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """不引入额外依赖的线性重采样；旋律提取不需要保留音频高频细节。"""
    if source_rate == target_rate:
        return np.ascontiguousarray(signal, dtype=np.float32)
    target_length = max(1, round(signal.size * target_rate / source_rate))
    source_x = np.arange(signal.size, dtype=np.float64)
    target_x = np.linspace(0, signal.size - 1, target_length, dtype=np.float64)
    return np.ascontiguousarray(np.interp(target_x, source_x, signal), dtype=np.float32)


def _estimate_pitch(frame: np.ndarray, sample_rate: int, min_hz: float, max_hz: float):
    """用归一化自相关估计一个短帧的基频，返回 ``(Hz, confidence)`` 或 ``None``。"""
    frame = frame.astype(np.float32, copy=False)
    frame = frame - np.mean(frame, dtype=np.float32)
    rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
    if rms < 1e-4:
        return None
    windowed = frame * np.hanning(frame.size).astype(np.float32)
    fft_size = 1 << (2 * frame.size - 1).bit_length()
    spectrum = np.fft.rfft(windowed, n=fft_size)
    correlation = np.fft.irfft(np.abs(spectrum) ** 2, n=fft_size)[: frame.size]
    zero = float(correlation[0])
    if zero <= 1e-12:
        return None
    correlation = correlation / zero
    min_lag = max(2, int(sample_rate / max_hz))
    max_lag = min(frame.size - 2, int(sample_rate / min_hz))
    if min_lag >= max_lag:
        return None
    region = correlation[min_lag : max_lag + 1]
    lag = min_lag + int(np.argmax(region))
    confidence = float(correlation[lag])
    if confidence < 0.22:
        return None
    # 抛物线插值减少音高量化抖动。
    if 1 <= lag < correlation.size - 1:
        left, center, right = correlation[lag - 1 : lag + 2]
        denominator = float(left - 2 * center + right)
        if abs(denominator) > 1e-8:
            lag = lag + float(0.5 * (left - right) / denominator)
    if lag <= 0:
        return None
    return float(sample_rate / lag), confidence


def _midi_to_abc(midi: int) -> str:
    """将 MIDI 音高转换为以 C 大调记谱的 ABC 音符（升号表示黑键）。"""
    midi = max(24, min(108, int(midi)))
    names = ("C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B")
    name = names[midi % 12]
    octave = midi // 12
    if octave < 5:
        return name + "," * (5 - octave)
    if octave == 5:
        return name
    return name.lower() + "'" * (octave - 6)


def _build_abc(notes: list[int | None], bpm: float) -> str:
    """把八分音符序列压缩成带小节线的 ABC。"""
    if not notes:
        notes = [None] * 8
    tokens: list[str] = []
    index = 0
    while index < len(notes):
        bar_end = min(len(notes), ((index // 8) + 1) * 8)
        while index < bar_end:
            value = notes[index]
            end = index + 1
            while end < bar_end and notes[end] == value:
                end += 1
            duration = end - index
            token = "z" if value is None else _midi_to_abc(value)
            if duration > 1:
                token += str(duration)
            tokens.append(token)
            index = end
        if index < len(notes):
            tokens.append("|")
    body = " ".join(tokens)
    return (
        "X:1\n"
        "T:Extracted melody reference\n"
        "M:4/4\n"
        "L:1/8\n"
        f"Q:1/4={int(round(bpm))}\n"
        "K:C\n"
        f"{body}\n"
    )


def audio_to_abc(
    audio: Any,
    bpm: float = 100.0,
    max_seconds: int = 60,
    min_note_hz: float = 65.0,
    max_note_hz: float = 1_000.0,
    gate: float = 0.20,
) -> str:
    """从 ComfyUI AUDIO 生成外部 YuE2 ABC 条件。"""
    bpm = float(bpm)
    max_seconds = int(max_seconds)
    min_note_hz = float(min_note_hz)
    max_note_hz = float(max_note_hz)
    gate = float(gate)
    if not MIN_BPM <= bpm <= MAX_BPM:
        raise ValueError(f"BPM 必须在 {MIN_BPM:g}..{MAX_BPM:g} 之间")
    if not 1 <= max_seconds <= 600:
        raise ValueError("旋律参考最长截取时间必须在 1..600 秒之间")
    if not 20 <= min_note_hz < max_note_hz <= 4_000:
        raise ValueError("旋律提取音域必须满足 20 <= min_note_hz < max_note_hz <= 4000")
    if not 0 <= gate <= 1:
        raise ValueError("静音门限 gate 必须在 0..1 之间")

    signal, source_rate = _audio_to_mono(audio)
    signal = signal[: int(source_rate * max_seconds)]
    signal = _resample_linear(signal, source_rate, TARGET_SAMPLE_RATE)
    if signal.size < FRAME_SIZE:
        signal = np.pad(signal, (0, FRAME_SIZE - signal.size))

    rms_values = []
    pitches: list[tuple[float, float] | None] = []
    window = np.hanning(FRAME_SIZE).astype(np.float32)
    for start in range(0, max(1, signal.size - FRAME_SIZE + 1), HOP_SIZE):
        frame = signal[start : start + FRAME_SIZE]
        if frame.size < FRAME_SIZE:
            frame = np.pad(frame, (0, FRAME_SIZE - frame.size))
        rms_values.append(float(np.sqrt(np.mean(frame * frame) + 1e-12)))
        pitches.append(_estimate_pitch(frame * window, TARGET_SAMPLE_RATE, min_note_hz, max_note_hz))
    peak_rms = max(rms_values, default=0.0)
    amplitude_gate = max(1e-4, peak_rms * gate)
    for index, rms in enumerate(rms_values):
        if rms < amplitude_gate:
            pitches[index] = None

    step_seconds = 30.0 / bpm  # 一个八分音符
    units = max(1, int(math.ceil(signal.size / TARGET_SAMPLE_RATE / step_seconds)))
    notes: list[int | None] = []
    for unit in range(units):
        begin = unit * step_seconds
        end = (unit + 1) * step_seconds
        values = [
            pitch
            for frame_index, pitch in enumerate(pitches)
            if pitch is not None
            and begin <= (frame_index * HOP_SIZE + FRAME_SIZE / 2) / TARGET_SAMPLE_RATE < end
        ]
        if not values or len(values) < 1:
            notes.append(None)
            continue
        frequencies = np.asarray([item[0] for item in values], dtype=np.float32)
        confidences = np.asarray([item[1] for item in values], dtype=np.float32)
        if float(np.mean(confidences >= 0.22)) < 0.35:
            notes.append(None)
            continue
        frequency = float(np.median(frequencies))
        notes.append(int(round(69 + 12 * math.log2(frequency / 440.0))))

    # 消除单个八分音符的检测跳变，避免 ABC 中出现大量不自然的碎音。
    for index in range(1, len(notes) - 1):
        if notes[index - 1] == notes[index + 1] and notes[index] != notes[index - 1]:
            notes[index] = notes[index - 1]
    return _build_abc(notes, bpm)
