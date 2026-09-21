"""MlxKSamplerMLX：实现完整生成流程（自己写采样循环，不依赖 config 节点）。

采样器不直接接受文本输入，而是像 ComfyUI 原生 KSampler 那样有「正/负两
条条件」两个入口（`positive` / `negative`，类型都是 condition，由两
个独立的 MlxTextEncoder 节点分别产出，各自只带一条提示词）。

普通图片家族在本节点只加载 transformer；编码已由 MlxTextEncoder 算好并按 key
存在 cache.py 里，这里按 key 取用。Ideogram 4 的条件依赖目标尺寸，是唯一例外：
本节点先延迟编码并释放文本编码器，再加载两套 transformer。VAE 一律不在采样阶段
加载，交给 VAE 编码 / 解码节点按 handle 物化。

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

Qwen-Image 2.1 原生统一编辑：三个加载器选择 `qwen_image_21`，同一个
`MlxVAEEncoder.ref_images` 必须接到正向/负向 `MlxTextEncoder.ref_images` 和本节点；
参考 latent 按视觉槽插入单流 DiT，首步执行 block-causal prefill，后续复用 prefix KV。

Ideogram 4 本地文生图：三个加载器选 `ideogram4` + `ideogram-4-fp8`，仍沿用本节点
的 model / positive / negative 三条标准连线；负向文本会被忽略，scheduler 必须选择
ideogram4_default / quality / turbo 之一，步数和 guidance schedule 由预设决定。

widget 里的 steps / scheduler / guidance 只是工作流自己填的值（初始默认取
「第一个大类」登记的 default_steps / default_scheduler / default_guidance）；真正用哪套
ModelConfig 由连进来的 handle 的权重目录名现算，因此新增权重目录不用改这里。

注意（选 flux2 时）：默认值取自第一个大类（z_image）的 `linear`，而 Flux2 走
flow-match，请把 scheduler 改成 `flow_match_euler_discrete`（`workflows/
flux2-klein-9b-*.json` 已填好）；steps 取 4、guidance 保持 1.0（Klein 是蒸馏
模型，不开 CFG，只有 guidance > 1.0 时才会用负面条件）。
"""

from __future__ import annotations

from .. import pipeline, runtime
from ..cache import CACHE
from ..h3 import pipeline as h3_pipeline
from ..h3.latent_creator.h3_layout import valid_frame_counts
from ..types import condition, entry_for, model, model_types, ref_images, validate_model_family

SCHEDULERS = [
    "linear",
    "flow_match_euler_discrete",
    "seedvr2_euler",
    "ideogram4_default",
    "ideogram4_quality",
    "ideogram4_turbo",
    "minimax_h3",
    "yue2_midpoint",
]
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
                # 必须声明 control_after_generate：ComfyUI 前端对任何名叫 seed /
                # noise_seed 的 widget 都会自动插一个「控制方式」伴随 widget
                # （fixed / increment / decrement / randomize），工作流 JSON 的
                # widgets_values 也必须在 seed 后面写上它的值 —— 漏写会让后面所有
                # widget 错位一格（提交时报 scheduler=124 / steps=640 这类怪错）。
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**63 - 1,
                        "control_after_generate": True,
                    },
                ),
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
                # 仅 YuE2：off 直接生成 codec；melody/full 先生成 ABC 规划。
                "cot": (["off", "melody", "full"], {"default": "full", "advanced": True}),
                # 仅 YuE2：语义 codec token 上限；每帧约 40 ms，9000 上限约 6 分钟。
                "max_tokens": (
                    "INT",
                    {"default": 9000, "min": 200, "max": 9000, "step": 100, "advanced": True},
                ),
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
        cot="full",
        max_tokens=9000,
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
        # if positive.clip.path != model.model_path:
        #     raise ValueError(
        #         "文本编码器与采样器选的权重不是同一套，"
        #         f"请把两边选成同一份（当前 {positive.clip.path} 与 {model.model_path}）"
        #     )
        # 配置由「大类 + 权重目录名」现取；未知大类直接报错（不静默回退）
        entry = entry_for(model.model_type)
        validate_model_family(model.model_type, model.model_path)
        if not entry.supported:
            raise NotImplementedError(f"{model.model_type} 尚未实现：{entry.notes}")

        # MiniMax-H3：一条打包序列里同时去噪视频与音频（guidance 蒸馏模型，每步只有
        # 一次前向），所以 scheduler / guidance / 负向条件三个 widget 都不看；
        # 画布必须是 32 的倍数、帧数必须是 17n+5（下拉里只给合法值）
        if entry.media == "video":
            if ref_images is not None:
                raise ValueError(f"{entry.family} 不支持参考图编辑，请把 ref_images 断开")
            if scheduler != entry.default_scheduler or float(guidance) != 1.0:
                print(
                    f"[MlxKSamplerMLX] {entry.family} 不看 scheduler / guidance / 负向条件"
                    "（guidance 蒸馏模型，每步一次前向），已忽略这三项"
                )
            keyframes = positive.h3_keyframes
            if keyframes is not None:
                # 参考生视频（Ref2VA）与首 / 尾锚点（I2VA / FL2VA）是两套不同的
                # transformer 权重：拿 Base 跑参考图，模型只会读到文本里的图片描述
                checkpoint_note = h3_pipeline.check_visual_condition_checkpoint(model, keyframes)
                if checkpoint_note:
                    print(f"[MlxKSamplerMLX] {checkpoint_note}")
                if keyframes.vae is None:
                    # 纯参考（anchor=none）：图只进 Qwen3-VL 的 presentation，没有 latent 锚点
                    print(
                        f"[MlxKSamplerMLX] {keyframes.source_label or '视觉条件'}："
                        "没有首 / 尾帧锚点，图片只当视觉提示（presentation）"
                    )
                elif (
                    keyframes.vae.model_type != model.model_type
                    or keyframes.vae.role != "vae"
                ):
                    raise ValueError(
                        "H3 视觉条件必须由当前模型大类的视频 VAE（role=vae）编码："
                        f"条件是 model_type={keyframes.vae.model_type!r}, "
                        f"role={keyframes.vae.role!r}，采样器是 {model.model_type!r}"
                    )
                if keyframes.width != int(width) or keyframes.height != int(height):
                    raise ValueError(
                        "H3 视觉条件的目标画布必须与采样器一致："
                        f"条件是 {keyframes.width}×{keyframes.height}，"
                        f"采样器是 {int(width)}×{int(height)}"
                    )
            expected_key = pipeline.h3_prompt_encoding_key(positive.clip, positive.text, keyframes)
            if positive.encoding_key != expected_key:
                raise RuntimeError(
                    "H3 关键帧与 Qwen3-VL presentation 编码不匹配；"
                    "请重新运行当前连线下的「MLX 文本编码器」节点"
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
                "keyframe_digest": keyframes.digest if keyframes is not None else "",
                "keyframe_anchors": keyframes.anchors if keyframes is not None else (),
            }
            # 相同 latent 仍在缓存时，不能为重建同一 handle 再加载 5 GB VAE 或 35 GB transformer。
            if pipeline.has_h3_latents(model, params, CACHE):
                return (pipeline.run_h3_sampler(entry, model, None, params, CACHE),)
            # 锚点 latent 与 params["keyframe_anchors"] 一一对应（纯参考时两个都是空）
            keyframe_latents = (
                pipeline.encode_h3_keyframes(keyframes, CACHE)[1] if keyframes is not None else ()
            )
            try:
                comps = pipeline.prepare_h3_sampler_components(entry, model, CACHE)
                handle = pipeline.run_h3_sampler(
                    entry,
                    model,
                    comps,
                    params,
                    CACHE,
                    keyframe_latents=keyframe_latents,
                )
            finally:
                # 潜变量已进 h3_latents 桶：transformer（q8 约 35 GB）用完或异常都释放。
                pipeline.release_h3_sampler_components(entry, model, CACHE)
            return (handle,)

        # YuE2：正向文本 = style，负向文本 = lyrics（空 lyrics = 纯音乐）。模型内部
        # tokenizer 同时参与 ABC / codec AR，因此文本节点只透传原文，不提前编码。
        if entry.media == "audio":
            if entry.family != "yue2":
                raise ValueError(
                    f"{entry.family} 不使用通用 MLX KSampler；"
                    "请改用「MLX Breeze Sampler」专用节点"
                )
            if ref_images is not None:
                raise ValueError("YuE2 不支持参考图，请把 ref_images 断开")
            if int(batch_size) != 1:
                raise ValueError("YuE2 当前一次生成一首音乐，请把 batch_size 设为 1")
            if scheduler != entry.default_scheduler:
                print(
                    f"[MlxKSamplerMLX] YuE2 固定使用 midpoint ODE，已忽略 scheduler={scheduler!r}"
                )
            params = {
                "seed": int(seed),
                "steps": int(steps),
                "guidance": float(guidance),
                "style": positive.text,
                "lyrics": negative.text,
                "cot": str(cot),
                "max_tokens": int(max_tokens),
            }
            # ComfyUI 可能因下游变化而重跑本节点；相同参数的 latent 若仍在缓存，绝不
            # 为构造同一个 handle 再加载一次 3B 主模型。
            if pipeline.has_yue2_latents(model, params, CACHE):
                return (pipeline.run_yue2_sampler(entry, model, None, params, CACHE),)
            comps = pipeline.prepare_yue2_sampler_components(entry, model, CACHE)
            try:
                handle = pipeline.run_yue2_sampler(entry, model, comps, params, CACHE)
            finally:
                # 采样结束后声学 latent 已独立缓存；释放约 3B 主模型，给 VAE 解码腾空间。
                pipeline.release_yue2_sampler_components(entry, model, CACHE)
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
        if ref_images is None and entry.family == "qwen_image_21" and (
            positive.ref_cache_key or negative.ref_cache_key
        ):
            raise ValueError(
                "Qwen-Image 2.1 文本条件已经带参考图，但采样器没有连接 ref_images；"
                "请把同一个「MLX VAE 编码」输出也接到采样器，或从正/负文本编码器断开它"
            )
        if ref_images is not None:
            if entry.family == "ideogram4":
                raise NotImplementedError(
                    "Ideogram 4 本地权重目前只支持文生图，不支持 Remix、参考图或蒙版编辑；"
                    "请断开 ref_images"
                )
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
            if entry.family == "qwen_image_21":
                if ref_images.edit_kind != "qwen_image_21":
                    raise ValueError("参考图不是按 Qwen-Image 2.1 原生统一编辑语义编码的")
                if positive.ref_cache_key != ref_images.cache_key:
                    raise ValueError(
                        "Qwen-Image 2.1 正向条件没有连接同一个 ref_images："
                        "请把「MLX VAE 编码」输出同时接到正向文本编码器和采样器"
                    )
                if negative.ref_cache_key != ref_images.cache_key:
                    raise ValueError(
                        "Qwen-Image 2.1 负向条件没有连接同一个 ref_images："
                        "请把「MLX VAE 编码」输出同时接到负向文本编码器和采样器"
                    )
                if (int(ref_images.width), int(ref_images.height)) != (int(width), int(height)):
                    print(
                        f"[MlxKSamplerMLX] Qwen-Image 2.1 第一张参考图是 "
                        f"{ref_images.width}×{ref_images.height}，目标画布是 {int(width)}×{int(height)}；"
                        "模型允许不同尺寸，但保持同一宽高比通常能减少编辑构图偏移"
                    )
            elif entry.family == "qwen_edit":
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
                    f"{model.model_type} 暂不支持参考图编辑（目前只有 flux2 / qwen_edit / qwen_image_21）"
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
            # 编译缓存上限（GB）：进 latent 缓存键 → 改档位必然重新采样，
            # 真正生效在下面的 runtime.apply_cache_limit（权重加载之前）
            "cache_limit_gb": int(model.compile_cache_limit),
            # edit 用：进 latent 缓存键 → 换参考图必然重新采样
            "ref_cache_key": ref_key,
            "ref_count": ref_count,
            "kv_cache": kv_cache,
        }
        if entry.family == "ideogram4":
            if not positive.text.strip():
                raise ValueError("Ideogram 4 的正向 caption 不能为空")
            preset = pipeline.ideogram4_preset(scheduler)
            if int(steps) != preset.num_steps or float(guidance) != preset.guidance_schedule[-1]:
                print(
                    f"[MlxKSamplerMLX] Ideogram 4 的 {scheduler} 预设固定使用 "
                    f"{preset.num_steps} 步及内置 guidance schedule；已忽略 steps / guidance widget"
                )
            params["steps"] = int(preset.num_steps)
            params["guidance"] = float(preset.guidance_schedule[-1])
            params["positive_encoding_key"] = pipeline.prepare_ideogram4_conditioning(
                entry,
                positive.clip,
                positive.text,
                int(width),
                int(height),
                CACHE,
            )
            params["negative_encoding_key"] = ""
            if negative.text.strip():
                print("[MlxKSamplerMLX] Ideogram 4 固定使用空无条件分支，已忽略 negative 文本")
        # 只加载 transformer；编码按 key 从 cache.py 取，VAE 由编码/解码节点按 handle 物化
        if entry.family == "qwen_image_21":
            # 正/负条件共享同一份 Qwen3-VL-8B；两条都编码完以后、加载 7B DiT 之前
            # 再统一释放，避免每条提示词各重载一次，又不让文本塔与 DiT 同时常驻。
            pipeline.release_encoder(positive.clip, CACHE)
            if negative.clip.cache_key != positive.clip.cache_key:
                pipeline.release_encoder(negative.clip, CACHE)
        # 先在权重物化之前把 MLX 的 free-cache 上限设好：这件事与 mx.compile 无关
        # （mflux 也是在建模型前调 apply_runtime_memory_options，编不编译都设），
        # 所以 compile 关闭时照样设。0 = 沿用默认不动，之前设过则还原。
        runtime.apply_cache_limit(model.compile_cache_limit, label=entry.family)
        comps = pipeline.prepare_sampler_components(entry, model, CACHE)
        handle = pipeline.run_sampler(entry, model, comps, params, CACHE)
        return (handle,)
