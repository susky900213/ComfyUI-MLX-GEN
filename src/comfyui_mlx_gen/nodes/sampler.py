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

Qwen-Image 文生图（txt2img）：选 `qwen_image` 大类 + `qwen-image-2512-8bit`
权重，条件用「MLX 文本编码器」（纯文本；**不要**接 ref_images —— 那套权重里
没有视觉塔，本大类也没有 edit 分支，接了会直接报错），默认 20 步 +
flow_match_euler_discrete + guidance 4.0（与 mflux `QwenImage.generate_image`
的默认值一致；`supports_compile=False` 会强制关掉编译）。

Qwen-Image-Edit 多图编辑（edit）：同样接 `ref_images`，但条件必须由
`MlxQwenEditEncoder`（带参考图）产出，且参考图 latent 是按**目标尺寸**编码的，
因此本节点会校验「参考图宽高 == 采样器 width/height」，不一致直接报错
（把「MLX VAE 编码」的 width/height 输出接到这里即可跟随）。Qwen 的 transformer
入参含 Config 对象与步号 int，不支持 `mx.compile`（`entry.supports_compile=False`
会强制关掉编译开关），CFG 用 `qwen_guided_noise`（普通 CFG 之后再按条件范数重标定）。

widget 里的 steps / scheduler / guidance 只是工作流自己填的值（初始默认取
「第一个大类」登记的 default_steps / default_scheduler / default_guidance）；真正用哪套
ModelConfig 由连进来的 handle 的权重目录名现算，因此新增权重目录不用改这里。

注意（选 flux2 时）：默认值取自第一个大类（z_image）的 `linear`，而 Flux2 走
flow-match，请把 scheduler 改成 `flow_match_euler_discrete`（`workflows/
flux2-klein-9b-*.json` 已填好）；steps 取 4、guidance 保持 1.0（Klein 是蒸馏
模型，不开 CFG，只有 guidance > 1.0 时才会用负面条件）。
"""

from __future__ import annotations

from .. import pipeline
from ..cache import CACHE
from ..h3.latent_creator.h3_layout import valid_frame_counts
from ..types import condition, entry_for, model, model_types, ref_images

SCHEDULERS = ["linear", "flow_match_euler_discrete", "seedvr2_euler"]
KV_CACHE_MODES = ["auto", "off"]
# 画布档位：图片链路原有的几档 + MiniMax-H3 的推荐档位（H3 必须是 32 的倍数，
# 且宽高比要在 [1/4, 4] 之内；见 h3/pipeline.make_plan）
WIDTH_OPTIONS = [256, 352, 384, 512, 544, 640, 768, 960, 1024, 1280, 1344, 1536, 2048]
HEIGHT_OPTIONS = [256, 352, 384, 512, 544, 640, 768, 960, 1024, 1280, 1344, 1536]
# MiniMax-H3 只接受 17n+5 的帧数（124 ~ 345，其余值直接不在下拉里）
FRAME_OPTIONS = list(valid_frame_counts())


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
                "width": (WIDTH_OPTIONS, {"default": 512}),
                "height": (HEIGHT_OPTIONS, {"default": 512}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                # 初值取「第一个大类」的 default_guidance（本机是 z_image 的 1.0）；
                # Qwen 编辑请在 2.5~4.0（该大类登记的是 2.5，工作流里显式写 2.5）
                "guidance": (
                    "FLOAT",
                    {"default": entry.default_guidance, "min": 0.0, "max": 20.0},
                ),
                "scheduler": (SCHEDULERS, {"default": entry.default_scheduler}),
                # --- 仅 MiniMax-H3 用到（图片链路默认收在「extras」里，不影响既有工作流）---
                # 帧数只给 17n+5 的合法值；两条整流流的 shift（视频 12.0 / 音频 3.0）
                "num_frames": (FRAME_OPTIONS, {"default": FRAME_OPTIONS[0], "advanced": True}),
                "video_shift": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.0, "max": 30.0, "step": 0.5, "advanced": True},
                ),
                "audio_shift": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.5, "advanced": True},
                ),
            },
            "optional": {
                # 连上 → 编辑（flux2 单图编辑 / qwen_edit 多图编辑）；不连 → 文生图
                # （z_image / flux2；qwen_edit 必须有参考图，不连会直接报错）
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
        num_frames=FRAME_OPTIONS[0],
        video_shift=12.0,
        audio_shift=3.0,
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

        # MiniMax-H3：一条打包序列里同时去噪视频与音频（guidance 蒸馏模型，每步只有
        # 一次前向），所以 scheduler / guidance / 负向条件三个 widget 都不看；
        # 画布必须是 32 的倍数、帧数必须是 17n+5（下拉里只给合法值）
        if entry.media != "image":
            if ref_images is not None:
                raise ValueError(f"{entry.family} 不支持参考图编辑，请把 ref_images 断开")
            if scheduler != entry.default_scheduler or float(guidance) != 1.0:
                print(
                    f"[MlxKSamplerMLX] {entry.family} 不看 scheduler / guidance / 负向条件"
                    "（guidance 蒸馏模型，每步一次前向），已忽略这三项"
                )
            params = {
                "seed": int(seed),
                "steps": int(steps),
                "height": int(height),
                "width": int(width),
                "num_frames": int(num_frames),
                "video_shift": float(video_shift),
                "audio_shift": float(audio_shift),
                "positive_encoding_key": positive.encoding_key,
                "prompt_digest": positive.text,
            }
            comps = pipeline.prepare_h3_sampler_components(entry, model, CACHE)
            handle = pipeline.run_h3_sampler(entry, model, comps, params, CACHE)
            # 潜变量已进 h3_latents 桶：transformer（q8 约 35 GB）用完就丢，
            # 别和解码要用的 VAE 一起占着（换 seed / 步数重跑时会重新懒加载）
            pipeline.release_h3_sampler_components(entry, model, CACHE)
            return (handle,)

        # 参考图（可选）：edit 分支。参考图由「MLX VAE 编码」产出（来源是 VAE handle，
        # 与 transformer 权重无关），所以这里只校验大类一致 + 本大类确实支持参考图编辑
        ref_key = ""
        ref_count = 0
        if ref_images is None and entry.family == "qwen_edit":
            raise ValueError(
                "qwen_edit 必须有参考图：请用「MLX VAE 编码」产出 ref_images，"
                "并把它接到本节点的 ref_images 入口（条件也要用「MLX Qwen 编辑条件」节点）"
            )
        if ref_images is not None:
            if entry.family == "qwen_image":
                raise ValueError(
                    "qwen_image 是文生图大类，本大类不接参考图（qwen-image-2512 这类"
                    "权重里没有视觉塔，参考图 latent 与带图的编辑条件都编不出来）；"
                    "要用参考图编辑，请把三个加载器都改成 qwen_edit 大类 + "
                    "qwen-image-edit-2511-8bit 权重"
                )
            if ref_images.model_type != model.model_type:
                raise ValueError(
                    f"参考图是用 {ref_images.model_type} 编码的，"
                    f"与采样器的 {model.model_type} 不匹配"
                )
            if entry.family == "qwen_edit":
                if ref_images.edit_kind != "qwen_edit":
                    raise ValueError(
                        "参考图不是按 qwen_edit 语义编码的（请把「MLX VAE 编码」的 model_type "
                        "也选成 qwen_edit 的那套权重）"
                    )
                if (int(ref_images.width), int(ref_images.height)) != (int(width), int(height)):
                    raise ValueError(
                        f"qwen_edit 的参考图 latent 是按目标尺寸编码的："
                        f"参考图 {ref_images.width}×{ref_images.height} 与采样器 "
                        f"{int(width)}×{int(height)} 不一致；"
                        "请把「MLX VAE 编码」的 width/height 输出接到采样器的 width/height"
                    )
            elif entry.family != "flux2":
                raise NotImplementedError(
                    f"{model.model_type} 暂不支持参考图编辑（目前只有 flux2 / qwen_edit）"
                )
            elif ref_images.edit_kind != "flux2":
                raise ValueError("参考图不是按 flux2 语义编码的")
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
            # qwen_edit 的 transformer 入参含 Config 对象与 int 步号 → 不支持 mx.compile
            "compile_model": bool(model.compile) and entry.supports_compile,
            # edit 用：进 latent 缓存键 → 换参考图必然重新采样
            "ref_cache_key": ref_key,
            "ref_count": ref_count,
            "kv_cache": kv_cache,
        }
        # 只加载 transformer；编码按 key 从 cache.py 取，VAE 由编码/解码节点按 handle 物化
        comps = pipeline.prepare_sampler_components(entry, model, CACHE)
        handle = pipeline.run_sampler(entry, model, comps, params, CACHE)
        return (handle,)
