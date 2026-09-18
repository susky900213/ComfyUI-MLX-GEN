"""MlxWhisperTranscribe：用本地 MLX Whisper 把 ComfyUI AUDIO 转成纯文本。"""

from __future__ import annotations

from .. import whisper_asr


class MlxWhisperTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_path": (whisper_asr.local_models(),),
                "language": (
                    whisper_asr.LANGUAGES,
                    {"default": "auto", "tooltip": "auto 自动检测；中文参考音频可固定为 zh"},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "condition_on_previous_text": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "长音频可开启；短参考音频关闭更不易出现重复循环",
                    },
                ),
                "initial_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "可选：填写人名、术语或产品名；不是转写指令",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "language")
    FUNCTION = "transcribe"
    CATEGORY = "MLX/Audio"

    def transcribe(
        self,
        audio,
        model_path,
        language="auto",
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt="",
    ):
        return whisper_asr.transcribe(
            audio=audio,
            model_path=model_path,
            language=language,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
        )