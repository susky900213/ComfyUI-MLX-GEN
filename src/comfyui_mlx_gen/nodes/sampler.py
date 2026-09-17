"""MlxKSamplerMLX：实现完整生成流程（自己写采样循环，不依赖 config 节点）。

采样器不直接接受文本输入，而是像 ComfyUI 原生 KSampler 那样有「正/负两
条条件」两个入口（`positive` / `negative`，类型都是 condition，由两
个独立的 MlxTextEncoder 节点分别产出，各自只带一条提示词）。

本节点只加载 transformer；编码已由 MlxTextEncoder 算好并按 key 存在
cache.py 里，这里按 key 取用，不再加载文本编码器；VAE 也不在这里加载
（采样阶段不需要，交给 VAE 编码 / 解码节点按 handle 物化）。

Flux.2 参考图编辑（edit）：把「MLX VAE 编码」的输出接到可选入口 `ref_images`
上即可 —— 采样时会按 mflux `Flux2KleinEdit` 的语义，把参考图 latent 拼在目标
latent 之后一起过 transformer，并只取回目标段；不接 `ref_images` 时行为与
文生图完全一致。

widget 里的 steps / scheduler / guidance 只是工作流自己填的值（初始默认取
「第一个大类」登记的 default_steps / default_scheduler）；真正用哪套
ModelConfig 由连进来的 handle 的权重目录名现算，因此新增权重目录不用改这里。

注意（选 flux2 时）：默认值取自第一个大类（z_image）的 `linear`，而 Flux2 走
flow-match，请把 scheduler 改成 `flow_match_euler_discrete`（`workflows/
flux2-klein-9b-*.json` 已填好）；steps 取 4、guidance 保持 1.0（Klein 是蒸馏
模型，不开 CFG，只有 guidance > 1.0 时才会用负面条件）。
"""

from __future__ import annotations

from .. import pipeline
from ..cache import CACHE
from ..types import condition, entry_for, model, model_types, ref_images

SCHEDULERS = ["linear", "flow_match_euler_discrete", "seedvr2_euler"]
KV_CACHE_MODES = ["auto", "off"]


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
            },
            "optional": {
                # 连上 → Flux.2 参考图编辑（edit）；不连 → 文生图（行为与以前一致）
                "ref_images": (ref_images, {}),
                # 只有 kv 版权重（配置 supports_kv_cache=True）才会有实际作用
                "kv_cache": (KV_CACHE_MODES, {"default": "auto"}),
            },
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
        ref_images=None,
        kv_cache="auto",
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

        # 参考图（可选）：edit 分支。参考图由「MLX VAE 编码」产出（来源是 VAE handle，
        # 与 transformer 权重无关），所以这里只校验大类一致 + 本大类确实支持参考图编辑
        ref_key = ""
        ref_count = 0
        if ref_images is not None:
            if ref_images.model_type != model.model_type:
                raise ValueError(
                    f"参考图是用 {ref_images.model_type} 编码的，"
                    f"与采样器的 {model.model_type} 不匹配"
                )
            if entry.family != "flux2":
                raise NotImplementedError(
                    f"{model.model_type} 暂不支持参考图编辑（目前只有 flux2 的 edit 路径）"
                )
            ref_key = ref_images.cache_key
            ref_count = int(ref_images.count)

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
            # edit 用：进 latent 缓存键 → 换参考图必然重新采样
            "ref_cache_key": ref_key,
            "ref_count": ref_count,
            "kv_cache": kv_cache,
        }
        # 只加载 transformer；编码按 key 从 cache.py 取，VAE 由编码/解码节点按 handle 物化
        comps = pipeline.prepare_sampler_components(entry, model, CACHE)
        handle = pipeline.run_sampler(entry, model, comps, params, CACHE)
        return (handle,)
