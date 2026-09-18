"""Breeze-TTS-2 专用采样器。

目标文本、克隆转写和音色设计指令都声明为 ``forceInput`` STRING socket，节点本身
不显示文本输入框。完整 checkpoint 仍由 MlxTransformerLoader 提供；生成后的最终
waveform 继续沿用 latents → MlxVAEDecoder → MlxPilToTorch.audio 链路。

本节点与通用 MlxKSamplerMLX 完全独立，避免改变已有图片、视频、YuE2 或旧 Breeze
工作流的输入签名和执行行为。
"""

from __future__ import annotations

from .. import pipeline
from ..breeze import MODES, SPEAKERS, normalize_reference, validate_params
from ..cache import CACHE
from ..types import MlxModelHandle, entry_for, latents, model


class MlxBreezeSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (model, {}),
                # 只提供 socket，不在采样器节点内创建文本 widget。
                "text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "连接外部文本节点；内容是要朗读的目标文本",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**63 - 1}),
                "mode": (list(MODES), {"default": "speaker"}),
                "speaker": (list(SPEAKERS), {"default": "S0"}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 2051}),
                "cfg_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "max_tokens": ("INT", {"default": 750, "min": 1, "max": 750}),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 3.0, "step": 0.05},
                ),
            },
            "optional": {
                "ref_text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "voice_clone 模式：连接参考音频的逐字转写",
                    },
                ),
                "instruction": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "voice_design 模式：连接音色、情绪或语速描述",
                    },
                ),
                "ref_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = (latents,)
    FUNCTION = "sample"
    CATEGORY = "MLX/Gen"

    def sample(
        self,
        model,
        text,
        seed,
        mode,
        speaker,
        temperature,
        top_p,
        top_k,
        cfg_scale,
        max_tokens,
        repetition_penalty,
        ref_text="",
        instruction="",
        ref_audio=None,
    ):
        if model is None:
            raise ValueError("必须连接 MlxTransformerLoader 的输出")
        if not isinstance(model, MlxModelHandle):
            raise ValueError(
                "model 必须是 MlxTransformerLoader 产出的 MLX handle；"
                "ComfyUI 原生 MODEL 不能被 Breeze 采样器使用"
            )
        entry = entry_for(model.model_type)
        if entry.family != "breeze_tts2":
            raise ValueError(
                f"MLX Breeze 采样器只能使用 breeze_tts2 模型，收到 {model.model_type}"
            )
        if not entry.supported:
            raise NotImplementedError(f"{model.model_type} 尚未实现：{entry.notes}")

        reference = normalize_reference(ref_audio) if ref_audio is not None else None
        params = {
            "seed": int(seed),
            "text": str(text or ""),
            "mode": str(mode),
            "speaker": str(speaker),
            "ref_text": str(ref_text or ""),
            "instruction": str(instruction or ""),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "cfg_scale": float(cfg_scale),
            "max_tokens": int(max_tokens),
            "repetition_penalty": float(repetition_penalty),
            "reference": reference,
        }

        # 无效文本或模式必须在缓存探测和加载 4+ GB checkpoint 之前报错。
        validate_params(params)
        if pipeline.has_breeze_waveform(model, params, CACHE):
            return (pipeline.run_breeze_sampler(entry, model, None, params, CACHE),)

        breeze_model = pipeline.prepare_breeze_model(entry, model, CACHE)
        try:
            handle = pipeline.run_breeze_sampler(entry, model, breeze_model, params, CACHE)
        finally:
            # 最终 waveform 已进独立缓存；解码阶段不再需要完整模型。
            pipeline.release_breeze_model(model, CACHE)
        return (handle,)