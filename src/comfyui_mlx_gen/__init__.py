"""ComfyUI-MLX-GEN 包入口：注册节点。"""

import sys
from pathlib import Path

# 把 src 目录加入 sys.path，使 "comfyui_mlx_gen" 可被顶层导入
SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from comfyui_mlx_gen.nodes import (  # noqa: E402
    breeze_sampler,
    clip_loader,
    loader,
    lora,
    pil_to_torch,
    qwen_edit,
    sampler,
    save,
    text_encoder,
    vae_decode,
    vae_encoder,
    vae_loader,
    whisper_transcribe,
)

NODE_CLASS_MAPPINGS = {
    "MlxClipLoader": clip_loader.MlxClipLoader,
    "MlxBreezeSampler": breeze_sampler.MlxBreezeSampler,
    "MlxTextEncoder": text_encoder.MlxTextEncoder,
    "MlxQwenEditEncoder": qwen_edit.MlxQwenEditEncoder,
    "MlxTransformerLoader": loader.MlxTransformerLoader,
    "MlxModelLoraApply": lora.MlxModelLoraApply,
    "MlxClipLoraApply": lora.MlxClipLoraApply,
    "MlxKSamplerMLX": sampler.MlxKSamplerMLX,
    "MlxVAELoader": vae_loader.MlxVAELoader,
    "MlxVAEEncoder": vae_encoder.MlxVAEEncoder,
    "MlxVAEDecoder": vae_decode.MlxVAEDecoder,
    "MlxVAEDecodeRawPIL": vae_decode.MlxVAEDecodeRawPIL,
    "MlxSaveImage": save.MlxSaveImage,
    "MlxPilToTorch": pil_to_torch.MlxPilToTorch,
    "MlxWhisperTranscribe": whisper_transcribe.MlxWhisperTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MlxClipLoader": "MLX CLIP 加载",
    "MlxBreezeSampler": "MLX Breeze-TTS-2 采样器",
    "MlxTextEncoder": "MLX 文本编码器",
    "MlxQwenEditEncoder": "MLX Qwen 编辑条件（带参考图）",
    "MlxTransformerLoader": "MLX 模型加载",
    "MlxModelLoraApply": "MLX 模型 LoRA",
    "MlxClipLoraApply": "MLX CLIP LoRA",
    "MlxKSamplerMLX": "MLX 采样器",
    "MlxVAELoader": "MLX VAE 加载",
    "MlxVAEEncoder": "MLX VAE 编码",
    "MlxVAEDecoder": "MLX VAE 解码",
    "MlxVAEDecodeRawPIL": "MLX VAE 解码（PIL）",
    "MlxSaveImage": "MLX 保存图片",
    "MlxPilToTorch": "MLX PIL → 张量",
    "MlxWhisperTranscribe": "MLX Whisper 语音转文字",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
