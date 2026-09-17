"""H3 的音频载荷与帧 / 音频 → ComfyUI 原生张量的转换。

本模块**不写编码代码**：mp4 交给 ComfyUI 自带的 `CreateVideo` + `SaveVideo`，
这里只负责把 H3 解码出来的 `(2, samples)` 波形包成 ComfyUI 的 `AUDIO`
（`dict(waveform=torch.Tensor[batch, channels, samples], sample_rate=int)`），
好让它能接核心的 `CreateVideo` / `SaveAudio`。

图片链路不产生音频：`to_comfy_audio(None)` 给 0 采样的空音频（下游不会崩，
但也不会写出有内容的声音）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

DEFAULT_SAMPLE_RATE = 32000  # H3 的音频 VAE 输出（config.json 里的 sampling_rate）


@dataclass(frozen=True)
class AudioTrack:
    """一段立体声波形：`waveform` 形状 `(2, samples)` float32（H3 用「两个 batch 项」表示立体声）。"""

    waveform: np.ndarray
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @property
    def channels(self) -> int:
        return int(self.waveform.shape[0]) if self.waveform.ndim == 2 else 1

    @property
    def samples(self) -> int:
        return int(self.waveform.shape[-1])

    @property
    def duration(self) -> float:
        return self.samples / float(self.sample_rate) if self.sample_rate else 0.0

    def trimmed(self, seconds: float) -> "AudioTrack":
        """按目标时长截断（H3 解码出来的长度会比 `帧数 / 24` 略长）。"""
        limit = int(round(float(seconds) * self.sample_rate))
        if limit <= 0 or limit >= self.samples:
            return self
        return replace(self, waveform=self.waveform[:, :limit])


def to_comfy_audio(track: AudioTrack | None, sample_rate: int = DEFAULT_SAMPLE_RATE) -> dict:
    """`AudioTrack` → ComfyUI 的 AUDIO（`None` → 0 采样的空音频）。"""
    import torch  # 延迟导入：只在真正要交给 ComfyUI 时才需要 torch

    if track is None:
        return {
            "waveform": torch.zeros((1, 2, 0), dtype=torch.float32),
            "sample_rate": int(sample_rate),
        }
    waveform = np.ascontiguousarray(track.waveform, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    return {
        "waveform": torch.from_numpy(waveform)[None],
        "sample_rate": int(track.sample_rate),
    }
