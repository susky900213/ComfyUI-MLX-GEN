"""YuE2 旋律参考节点。"""

from __future__ import annotations

from ..yue2_melody import audio_to_abc


class MlxYue2MelodyFromAudio:
    """把原曲 AUDIO 转成 YuE2 外部 ABC 旋律条件。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "bpm": (
                    "FLOAT",
                    {"default": 100.0, "min": 40.0, "max": 220.0, "step": 1.0},
                ),
                "max_seconds": ("INT", {"default": 60, "min": 1, "max": 600}),
                "min_note_hz": (
                    "FLOAT",
                    {"default": 65.0, "min": 20.0, "max": 2_000.0, "step": 1.0},
                ),
                "max_note_hz": (
                    "FLOAT",
                    {"default": 1_000.0, "min": 100.0, "max": 4_000.0, "step": 1.0},
                ),
                "gate": (
                    "FLOAT",
                    {
                        "default": 0.20,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "相对峰值的静音门限；伴奏较响时可提高",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("abc",)
    FUNCTION = "extract"
    CATEGORY = "MLX/Audio"

    def extract(
        self,
        audio,
        bpm=100.0,
        max_seconds=60,
        min_note_hz=65.0,
        max_note_hz=1_000.0,
        gate=0.20,
    ):
        abc = audio_to_abc(
            audio,
            bpm=bpm,
            max_seconds=max_seconds,
            min_note_hz=min_note_hz,
            max_note_hz=max_note_hz,
            gate=gate,
        )
        print(
            f"[MlxYue2MelodyFromAudio] 已提取约 {max_seconds}s 内的单旋律，"
            "将作为 YuE2 外部 ABC 条件"
        )
        return (abc,)
