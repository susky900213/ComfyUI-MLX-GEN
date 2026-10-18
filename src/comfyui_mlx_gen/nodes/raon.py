"""Raon Speech 的独立 PyTorch/Transformers ComfyUI 节点。"""

from __future__ import annotations

from .. import raon


class RaonModelLoader:
    """只返回轻量配置 handle；完整 21B 模型在 TTS/STT 节点首次执行时加载。"""

    @classmethod
    def INPUT_TYPES(cls):
        models = raon.model_options()
        return {
            "required": {
                "model_path": (list(models), {"default": models[0]}),
                "device": (list(raon.DEVICES), {"default": "auto"}),
                "dtype": (list(raon.DTYPES), {"default": "bfloat16"}),
                "attn_implementation": (
                    list(raon.ATTENTION_IMPLEMENTATIONS),
                    {"default": "sdpa"},
                ),
            }
        }

    RETURN_TYPES = (raon.RAON_MODEL,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Raon/Speech"

    def load(self, model_path, device="auto", dtype="bfloat16", attn_implementation="sdpa"):
        return (raon.make_handle(model_path, device, dtype, attn_implementation),)


class RaonTTS:
    """Raon Speech 文本转语音，支持可选参考音频音色条件。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (raon.RAON_MODEL,),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "forceInput": True,
                    },
                ),
            },
            "optional": {"speaker_audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "synthesize"
    CATEGORY = "Raon/Speech"

    def synthesize(self, model, text, speaker_audio=None):
        return (raon.synthesize(model, text, speaker_audio),)


class RaonSTT:
    """Raon Speech 语音转文字。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (raon.RAON_MODEL,),
                "audio": ("AUDIO",),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "可选转录指令；留空使用官方默认提示词",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "Raon/Speech"

    def transcribe(self, model, audio, prompt=""):
        return (raon.transcribe(model, audio, prompt),)


class RaonSpeechChat:
    """Raon Speech 语音对话，当前固定返回文本回答。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (raon.RAON_MODEL,),
                "audio": ("AUDIO",),
            },
            "optional": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "可选附加文字问题",
                    },
                )
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "chat"
    CATEGORY = "Raon/Speech"

    def chat(self, model, audio, prompt=""):
        return (raon.speech_chat(model, audio, prompt),)


class RaonSpeechChatAudio:
    """Raon Speech 语音对话，强制生成语音回答。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (raon.RAON_MODEL,),
                "audio": ("AUDIO",),
            },
            "optional": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "可选附加文字问题",
                    },
                )
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "chat"
    CATEGORY = "Raon/Speech"

    def chat(self, model, audio, prompt=""):
        return (raon.chat(model, audio, prompt=prompt, force_audio_output=True),)


__all__ = [
    "RaonModelLoader",
    "RaonTTS",
    "RaonSTT",
    "RaonSpeechChat",
    "RaonSpeechChatAudio",
]
