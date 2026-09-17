"""YuE2-3B 的纯 MLX 本地音乐生成实现。

算法移植自本机 ``/Users/apple/workspace/YuE2-3B-MLX`` 快照；这里只保留
ComfyUI 进程内推理需要的模型、采样与 VAE 解码代码，不依赖其 FastAPI 服务。
"""

from .pipeline import SAMPLE_RATE, Sampling, Tokenizer, generate_music_latents

__all__ = ["SAMPLE_RATE", "Sampling", "Tokenizer", "generate_music_latents"]