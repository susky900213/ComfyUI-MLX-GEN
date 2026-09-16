"""MlxKSamplerMLX：实现完整生成流程（自己写采样循环，不依赖 config 节点）。

采样器不直接接受文本输入，而是像 ComfyUI 原生 KSampler 那样有「正/负两
条条件」两个入口（`positive` / `negative`，类型都是 condition，由两
个独立的 MlxTextEncoder 节点分别产出，各自只带一条提示词）。

本节点只加载 transformer + vae；编码已由 MlxTextEncoder 算好并按 key 存在
cache.py 里，这里按 key 取用，不再加载文本编码器。

widget 里的 steps / scheduler / guidance 只是工作流自己填的值（初始默认取
「第一个大类」登记的 default_steps / default_scheduler）；真正用哪套
ModelConfig 由连进来的 handle 的权重目录名现算，因此新增权重目录不用改这里。
"""

from __future__ import annotations

from .. import pipeline
from ..cache import CACHE
from ..types import condition, entry_for, model, model_types

SCHEDULERS = ["linear", "flow_match_euler_discrete", "seedvr2_euler"]


class MlxKSamplerMLX:
    @classmethod
    def INPUT_TYPES(cls):
        entry = entry_for(model_types()[0])
        return {
            "required": {
                "model": (model, {}),
                "positive": (condition, {}),
                "negative": (condition, {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**63 - 1}),
                "steps": ("INT", {"default": entry.default_steps, "min": 1, "max": 100}),
                "width": ([256, 512, 768, 1024, 1536, 2048], {"default": 512}),
                "height": ([256, 512, 768, 1024, 1536, 2048], {"default": 512}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0}),
                "scheduler": (SCHEDULERS, {"default": entry.default_scheduler}),
            }
        }

    RETURN_TYPES = ("latents",)
    FUNCTION = "sample"
    CATEGORY = "MLX/Gen"

    def sample(
        self,
        model,
        positive,
        negative,
        seed,
        steps,
        width,
        height,
        batch_size,
        guidance,
        scheduler,
    ):
        if model is None:
            raise ValueError("必须连接 MlxTransformerLoader 的输出")
        if positive is None or negative is None:
            raise ValueError(
                "必须分别连接 MlxTextEncoder 的输出作为正/负条件，"
                "采样器不直接接受文本输入"
            )
        if positive.clip != negative.clip:
            raise ValueError(
                "正/负条件必须接在同一个 MlxClipLoader 上（组件配置不能混用）"
            )
        if positive.clip.model_type != model.model_type:
            raise ValueError(
                f"文本编码器选的是 {positive.clip.model_type}，与采样器的 {model.model_type} 不匹配"
            )
        if positive.clip.path != model.model_path:
            raise ValueError(
                "文本编码器与采样器选的权重不是同一套，"
                f"请把两边选成同一份（当前 {positive.clip.path} 与 {model.model_path}）"
            )
        # 配置由「大类 + 权重目录名」现取；未知大类直接报错（不静默回退）
        entry = entry_for(model.model_type)
        if not entry.supported:
            raise NotImplementedError(f"{model.model_type} 尚未实现：{entry.notes}")
        params = {
            "seed": int(seed),
            "steps": int(steps),
            "height": int(height),
            "width": int(width),
            "batch_size": int(batch_size),
            "positive_encoding_key": positive.encoding_key,
            "negative_encoding_key": negative.encoding_key,
            "guidance": float(guidance),
            "scheduler_name": scheduler,
            "compile_model": bool(model.compile),
        }
        # 只加载 transformer + vae；编码按 key 从 cache.py 取，不加载文本编码器
        comps = pipeline.prepare_sampler_components(entry, model, CACHE, model.cache_key)
        handle = pipeline.run_sampler(entry, model, comps, params, CACHE)
        return (handle,)
