"""MLX Whisper 与 ComfyUI AUDIO 之间的无状态适配层。"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import paths, runtime

SAMPLE_RATE = 16000
DEFAULT_MODEL = "whisper-large-v3-mlx"
LANGUAGES = ("auto", "zh", "yue", "en", "ja", "ko")


def _is_whisper_directory(directory: Path) -> bool:
    config_path = directory / "config.json"
    if not config_path.is_file():
        return False
    if not ((directory / "weights.npz").is_file() or (directory / "weights.safetensors").is_file()):
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return config.get("model_type") == "whisper"


def local_models() -> tuple[str, ...]:
    """列出 transformer/ 中可由 mlx-whisper 直接加载的目录。"""
    root = paths.component_dir("transformer")
    found = [name for name in paths.list_component_items("transformer") if _is_whisper_directory(root / name)]
    found.sort(key=lambda name: (name != DEFAULT_MODEL, name.casefold()))
    # 保留默认项可让缺权重时节点仍能注册；执行时会给出明确的目录错误。
    return tuple(found or [DEFAULT_MODEL])


def resolve_model(selection: str) -> str:
    """解析并校验本地 MLX Whisper checkpoint，禁止静默联网下载。"""
    kind, resolved = paths.resolve("local", str(selection), "transformer")
    directory = Path(resolved)
    if kind != "dir":
        raise ValueError(
            f"找不到本地 MLX Whisper 模型目录：{resolved}；"
            f"请把完整 checkpoint 放到 {paths.component_dir('transformer')}"
        )
    if not _is_whisper_directory(directory):
        raise ValueError(
            f"不是有效的 MLX Whisper checkpoint：{directory}；"
            "目录必须含 model_type=whisper 的 config.json 以及 weights.npz/weights.safetensors"
        )
    return str(directory)


def normalize_audio(audio: Any) -> tuple[np.ndarray, int]:
    """校验 ComfyUI AUDIO，并把第一批的所有声道平均成 float32 单声道。"""
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Whisper 输入必须是 ComfyUI AUDIO（含 waveform 与 sample_rate）")
    waveform = audio["waveform"]
    try:
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().float().cpu().numpy()
        array = np.asarray(waveform, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法读取 Whisper AUDIO waveform：{exc}") from exc
    if array.ndim != 3:
        raise ValueError(f"Whisper AUDIO waveform 必须是 [batch, channels, samples]，收到 {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"Whisper AUDIO waveform 不能为空，收到 {array.shape}")
    try:
        sample_rate = int(audio["sample_rate"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Whisper AUDIO sample_rate 必须是正整数：{audio['sample_rate']!r}") from exc
    if sample_rate <= 0:
        raise ValueError(f"Whisper AUDIO sample_rate 必须大于 0，收到 {sample_rate}")
    mono = np.mean(array[0], axis=0, dtype=np.float32)
    if not np.all(np.isfinite(mono)):
        raise ValueError("Whisper AUDIO 含 NaN 或无穷值")
    return np.ascontiguousarray(np.clip(mono, -1.0, 1.0), dtype=np.float32), sample_rate


def resample_to_whisper(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """使用 polyphase 抗混叠重采样，把波形变成 mlx-whisper 要求的 16 kHz。"""
    if int(sample_rate) == SAMPLE_RATE:
        return np.ascontiguousarray(mono, dtype=np.float32)
    try:
        from scipy.signal import resample_poly
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("MLX Whisper 重采样需要 scipy；请重新安装本插件 requirements.txt") from exc
    divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
    output = resample_poly(mono, SAMPLE_RATE // divisor, int(sample_rate) // divisor)
    if output.size < 1 or not np.all(np.isfinite(output)):
        raise RuntimeError("Whisper 音频重采样失败：输出为空或含非有限值")
    return np.ascontiguousarray(np.clip(output, -1.0, 1.0), dtype=np.float32)


def transcribe(
    audio: Any,
    model_path: str,
    language: str = "auto",
    temperature: float = 0.0,
    condition_on_previous_text: bool = False,
    initial_prompt: str = "",
) -> tuple[str, str]:
    """逐字转写 ComfyUI AUDIO；第二项返回实际语言代码。"""
    if language not in LANGUAGES:
        raise ValueError(f"不支持的 Whisper language：{language!r}")
    if not math.isfinite(float(temperature)) or float(temperature) < 0:
        raise ValueError("Whisper temperature 必须是大于等于 0 的有限数")
    resolved = resolve_model(model_path)
    mono, source_rate = normalize_audio(audio)
    waveform = resample_to_whisper(mono, source_rate)
    try:
        mlx_whisper = importlib.import_module("mlx_whisper")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "无法导入 mlx-whisper；请在 ComfyUI 的 Python 环境执行 "
            "pip install -r <ComfyUI-MLX-GEN>/requirements.txt"
        ) from exc

    print(
        f"[MlxWhisperTranscribe] {Path(resolved).name}: "
        f"{waveform.size / SAMPLE_RATE:.2f}s, language={language}, task=transcribe"
    )
    try:
        result = mlx_whisper.transcribe(
            waveform,
            path_or_hf_repo=resolved,
            language=None if language == "auto" else language,
            task="transcribe",
            temperature=float(temperature),
            condition_on_previous_text=bool(condition_on_previous_text),
            initial_prompt=str(initial_prompt).strip() or None,
            word_timestamps=False,
            verbose=None,
            fp16=True,
        )
    finally:
        # 后续通常立即加载 Breeze；不要让约 3 GB 的 large-v3 与 TTS 权重同时常驻。
        release_model()
    if not isinstance(result, dict):
        raise RuntimeError(f"mlx-whisper 返回了无效结果：{type(result).__name__}")
    text = str(result.get("text", "")).strip()
    if not text:
        raise RuntimeError("Whisper 没有识别到语音；请检查参考音频是否包含清晰人声")
    detected_language = str(result.get("language") or ("" if language == "auto" else language))
    print(f"[MlxWhisperTranscribe] 转写完成（{detected_language or 'unknown'}）：{text}")
    return text, detected_language


def release_model() -> None:
    """释放 mlx-whisper 的进程级单模型缓存以及 MLX Metal cache。"""
    try:
        module = importlib.import_module("mlx_whisper.transcribe")
        holder = getattr(module, "ModelHolder", None)
        if holder is not None:
            holder.model = None
            holder.model_path = None
    except Exception:  # noqa: BLE001
        pass
    runtime.flush_caches()