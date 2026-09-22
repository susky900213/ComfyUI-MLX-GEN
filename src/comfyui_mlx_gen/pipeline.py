"""生成实现（我们自己写采样循环，不调用 mflux 的 generate_image / generate_output）。

按「模型大类」（family）分派，借用 mflux 的数据工具（latent creator、prompt
encoder、scheduler、vae 等），采样循环与权重加载由我们自己实现：

- z_image：`create_noise` → 单数组条件 → `transformer(timestep, x, cap_feats, sigmas)`
  → `scheduler.step(noise, t, latents)` → `unpack_latents` + `VAEUtil.decode`；
- flux2：`prepare_packed_latents` → 条件为 `(embeds, ids)` → `transformer(hidden_states,
  encoder_hidden_states, timestep, img_ids, txt_ids)` → `scheduler.step(..., sigmas)`
  → `unpack_latents` + `Flux2VAE.decode_packed_latents`；
- flux2 参考图编辑（edit）：参考图 latent 拼在目标 latent 之后一起过 transformer，
  每个 step 只取回目标段（见 `_sample_flux2_edit`），目标 latent 形状与 txt2img 相同；
- qwen_edit 多图编辑（edit）：条件必须带参考图（`MlxQwenEditEncoder` 走
  `prepare_encoder` + `encode_edit_conditioning`），参考图 latent 按**目标尺寸**编码后
  拼在目标 latent 之后；CFG 用 `qwen_guided_noise`（普通 CFG 之后再按条件范数重标定），
  transformer 的 `t` 传步号 int（内部取 `config.scheduler.sigmas[t]`），不编译
  （见 `_sample_qwen_edit` / `decode_latents`）；
- qwen_image 文生图（txt2img）：条件由「MLX 文本编码器」编成 `(embeds, mask)`
  （`encode_text` 走 `QwenPromptEncoder`，签名是
  `encode_prompt(prompt, negative_prompt, prompt_cache, qwen_tokenizer,
  qwen_text_encoder)`，prompt_cache 在本项目用不上 → 给空 dict），latent 由
  `QwenLatentCreator.create_noise` 纯噪声起（`[1, seq, 64]`，**不接参考图**），
  默认 20 步 + flow_match_euler_discrete + guidance 4.0（见 `_sample_qwen_image`；
  latent 形状与 edit 一致，因此解码仍走同一个 `decode_latents`）。
- qwen_image_21 统一生成/编辑：Qwen3-VL 条件为 `(pre_norm_embeds, attention_mask,
  image_pad_mask)`；编辑时每个视觉槽扩展为 2×2 个参考 VAE latent token，文本和各参考图
  组成 block-causal prefix，目标图块双向注意并在后续步骤复用 prefix KV cache；不接
  `ref_images` 时同一前向自然退化为 T2I。
- ideogram4 本地文生图：采样器拿到目标宽高后才编码 JSON caption；文本编码器用完
  立即释放，再加载 conditional / unconditional 两套 FP8 transformer，按官方
  Ideogram4Scheduler 预设执行双模型 CFG；latent 用 Ideogram4LatentCreator 解包并
  交给 Flux2VAE.decode。

组件分工（延迟装配）：普通图片家族的采样器只加载 transformer；text_encoder + tokenizer 由
MlxTextEncoder 自己加载并编码，VAE 由 MlxVAEEncoder / MlxVAEDecoder 按
MlxVaeHandle 物化（编码与解码共用同一份实例）。采样器按 handle 里的缓存键
直接取编码结果，不再碰文本编码器。Ideogram 4 是例外：条件依赖目标宽高，所以由
采样器延迟编码；采样阶段同时物化两套 transformer。
"""

from __future__ import annotations

import json
import math
import time
from inspect import signature
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx

from . import breeze, components, image, paths, runtime, transformer_lora, weights
from .compiled_predict import CompiledPredictCache
from .h3 import pipeline as h3_pipeline, prompt as h3_prompt
from .h3.model.h3_precision import exact_fp32 as h3_exact_fp32
from .h3.weights import loader as h3_loader
from .progress import SamplingProgress
from .qwen_image_21 import loader as qwen21_loader, sampling as qwen21_sampling
from .qwen_image_21.text_encoder import encode_prompt as encode_qwen21_prompt
from .qwen_image_21.transformer import QwenImage21KVCache
from .types import (
    MlxH3VisualCondition,
    MlxLatentHandle,
    MlxModelEntry,
    MlxPilImage,
    entry_for,
    validate_model_family,
)
from .yue2 import pipeline as yue2_pipeline
from .yue2.model import load_model as load_yue2_model
from .yue2.vae import load_vae as load_yue2_vae

# 各节点负责的组件 role（采样器不加载文本编码 / VAE，编码与解码节点各自按 handle 物化）。
# 采样阶段只需要 transformer —— 这里是采样器的「声明式」role 清单，
# prepare_sampler_components 只装配其中的 transformer；将来若采样阶段真的要用 VAE，
# 改这一处 + 那个函数即可。
SAMPLER_ROLES: tuple[str, ...] = ("transformer",)
ENCODER_ROLES: tuple[str, ...] = ("text_encoder", "tokenizer")


def component_path(entry, role: str, selection: str) -> tuple[str, str]:
    """解析某 role 的路径（由调用方给出权重集目录名，不在配置里写死）。

    同一个「权重集名」在各组件目录下同名（如 z-image-turbo-8bit 同时存在于
    transformer/ 与 vae/ 等目录下），因此 handle 里只存目录名就够了。
    """
    if not selection:
        raise ValueError(f"没有为 {role} 指定权重目录（请在节点上重新选择）")
    validate_model_family(entry.family, selection)
    if not entry.supported:
        raise NotImplementedError(f"{entry.family} 尚未实现：{entry.notes}")
    comp = entry.components[role]
    kind, resolved = paths.resolve(comp.source, selection, role)
    # Ideogram 4 的权重根目录有 transformer/ 与 unconditional_transformer/ 两个
    # 同级目录。现有 ComfyUI 模型目录只要求用户给 transformer 建一个软链；若没有
    # 单独的 unconditional_transformer 软链，就从它解析后的真实目录寻找同级组件。
    if kind == "missing" and role == "unconditional_transformer":
        cond_kind, cond_path = paths.resolve(comp.source, selection, "transformer")
        if cond_kind == "dir":
            sibling = Path(cond_path).resolve().parent / "unconditional_transformer"
            if sibling.is_dir():
                return "dir", str(sibling)
    return kind, resolved


def resolve_class_kwargs(entry, kind: str, model_config) -> dict[str, Any]:
    """该组件构造参数（只保留构造函数真正接受的键，全部按目录名现算的配置里取）。

    ModelConfig 的 `{kind}_overrides` 里混着两类东西：① 真正的构造参数
    （`num_layers` / `in_channels` / `supports_kv_cache` / `use_src_mask` …，
    少了就会建出维度对不上的实例）；② 只在**采样期**生效的开关（flux2 的
    `finetune_ratio`、Qwen 的 `qwen_edit_plus` / `qwen_edit_2509`）—— 后者在
    mflux 里由 variant 类（`QwenEditPlus` / `QwenEdit2509`）自己判断，
    从来不是 `X(**overrides)` 的入参，原样透传会得到
    `TypeError: ... unexpected keyword argument 'qwen_edit_plus'`
    （qwen-image-edit-2511-8bit 之前就是这样挂的）。

    另外 `qwen_image`（2512）这一族 `transformer_overrides` 是空的
    （mflux 的 `QwenImageInitializer` 对 generic「qwen-image」配置返回 `{}`），
    所以过滤后仍然是 `{}` → `QwenTransformer(**{})`，与参考实现一致。
    """
    overrides = dict(getattr(model_config, f"{kind}_overrides", {}) or {})
    comp = entry.components.get(kind, {})
    import_str = getattr(comp, "class_import", "")
    if not import_str:
        return overrides

    params = signature(runtime.import_object(import_str)).parameters
    class_kwargs = {k: v for k, v in overrides.items() if k in params}
    dropped = sorted(set(overrides) - set(class_kwargs))
    if dropped:
        print(
            f"[resolve_class_kwargs] {kind} 的构造函数不接受 {dropped}，已跳过"
            "（这些是运行时开关，不是 __init__ 入参；采样语义由我们自己按"
            "权重集实现，见 `_sample_qwen_edit` 的 zero_cond_t）"
        )
    return class_kwargs


def component_class_kwargs(entry, role: str, path: str, model_config) -> dict[str, Any]:
    """按组件实际目录生成构造参数；Ideogram 4 从本地 config.json 读取尺寸。"""
    if entry.family == "ideogram4":
        initializer = runtime.import_object(
            "mflux.models.ideogram4.ideogram4_initializer:Ideogram4Initializer"
        )
        if role in ("transformer", "unconditional_transformer"):
            return {"config": initializer._transformer_config(Path(path))}
        if role == "text_encoder":
            return initializer._text_encoder_kwargs(Path(path))
    return resolve_class_kwargs(entry, role, model_config)


def prepare_components(
    entry,
    roles: tuple[str, ...],
    cache,
    cache_key: str,
    selections: dict[str, str],
    quantize: int | None = None,
    max_length: int | None = None,
    precision: str = "bfloat16",
) -> dict[str, Any]:
    """只加载 roles 里列出的组件（命中缓存则复用整组实例）。

    selections 是「role -> 权重集目录名」；目录不写在配置里，因此配置里
    不用为每个变体登记路径，新权重目录丢进磁盘就能用。
    """

    # 这个入口也可能被测试或扩展节点直接调用；不能只依赖上层节点的 guard。
    validate_model_family(entry.family, next(iter(selections.values()), ""))
    if not entry.supported:
        raise NotImplementedError(f"{entry.family} 尚未实现：{entry.notes}")

    def build() -> dict[str, Any]:
        raw: dict[tuple, dict] = {}
        comps: dict[str, Any] = {}
        for role in roles:
            selection = selections.get(role, "")
            kind, path = component_path(entry, role, selection)
            if kind == "missing":
                raise FileNotFoundError(f"未找到 {role} 权重: {path}")
            if entry.family == "qwen_image_21":
                if role == "tokenizer":
                    comps[role] = qwen21_loader.load_tokenizer(path)
                else:
                    comps[role] = qwen21_loader.load(
                        role,
                        path,
                        quantize=quantize,
                        precision=precision,
                    ).module
            elif role == "tokenizer":
                comps[role] = components.load_tokenizer(entry, role, kind, path, max_length)
            else:
                # 构造组件需要的 class_kwargs（如 flux2 的 transformer/text_encoder
                # 必须按 ModelConfig 的 overrides 建，否则用默认构造参数会建出
                # 错误维度、加载权重对不上）；按当前 role 的目录名现算 config，
                # 再只留构造函数接受的键（overrides 里混着运行时开关）。
                model_config = weights.config_for_path(selection, entry.default_config)
                class_kwargs = component_class_kwargs(entry, role, path, model_config)
                # quantize=None → 用权重自带的档位（QuantizationResolution.resolve）
                instance, _bits = components.create_and_load(
                    entry, role, kind, path, quantize, raw, class_kwargs=class_kwargs
                )
                comps[role] = instance
        return comps

    instance, _hit = cache.get_or_create("module", cache_key, build)
    return instance


def prepare_encoder(entry, clip, cache, cache_key: str) -> dict[str, Any]:
    """加载 text_encoder + tokenizer（由 MlxTextEncoder / MlxQwenEditEncoder 负责）。

    同一 handle 第二次执行命中同一条 module 缓存（两个节点共用这个键），
    因此 Qwen 编辑的正/负两个条件节点只会驻留一份文本编码器。
    qwen_edit 还要在 bundle 上挂 VL tokenizer × 2 与 VL 编码器包装（`_attach_vl`）：
    它们是「对象包装」，不含新权重，随 bundle 一起缓存，不额外占 module 桶位。

    MiniMax-H3（media="video"）不走这里，用 prepare_h3_encoder。
    """
    _ensure_image_family(entry, "MLX 文本编码器")
    # 两个 role 都用 handle 里的那个权重集名（各组件目录下同名）
    selections = {role: clip.path for role in ENCODER_ROLES}
    comps = prepare_components(
        entry,
        ENCODER_ROLES,
        cache,
        cache_key,
        selections,
        quantize=clip.quantize,
        max_length=clip.max_length,
        precision=clip.precision,
    )
    # qwen_edit 的条件编码还要 VL 那两层；命中缓存时 bundle 里已经有了（幂等）
    if entry.family == "qwen_edit" and "vl_encoder" not in comps:
        _attach_vl(comps)
    return comps


# --- Qwen-Image-Edit（多图编辑） ---
QWEN_EDIT_VL_MAX_LENGTH = 1024  # 与 mflux QwenImageInitializer.init_edit 一致
QWEN_EDIT_TARGET_MULTIPLE = 16  # 参考图 latent 的目标尺寸必须是 16 的倍数
QWEN_EDIT_RESIZE_MODES = ("stretch", "aspect_fit_pad")


def _attach_vl(comps: dict[str, Any]) -> None:
    """给 Qwen 编辑的编码器 bundle 挂上 VL tokenizer 与 VL 编码器包装（都是对象，不含新权重）。

    - `vl_encoder = QwenVisionLanguageEncoder(encoder=bundle 里那个 encoder)`：
      与 text_encoder 共用同一份权重（视觉塔已在 attach_import 里挂到 `encoder.visual`）；
    - 两个 VL tokenizer 只是模板不同：单图用官方 plain 模板（= mflux 现在的行为），
      多图用官方 `Picture N:` 模板（mflux 默认的 plain 在多图时会丢图，见文档 §2.4）；
    - 挂进 bundle 后随 bundle 一起缓存，不额外占 module 桶位。
    """
    proc_cls = runtime.import_object(
        "mflux.models.qwen.tokenizer.qwen_vision_language_processor:QwenVisionLanguageProcessor"
    )
    tok_cls = runtime.import_object(
        "mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer:QwenVisionLanguageTokenizer"
    )
    enc_cls = runtime.import_object(
        "mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder"
        ":QwenVisionLanguageEncoder"
    )
    processor = proc_cls(tokenizer=comps["tokenizer"].tokenizer)
    comps["vl_tokenizer_single"] = tok_cls(
        processor=processor, max_length=QWEN_EDIT_VL_MAX_LENGTH, use_picture_prefix=False
    )
    comps["vl_tokenizer_multi"] = tok_cls(
        processor=processor, max_length=QWEN_EDIT_VL_MAX_LENGTH, use_picture_prefix=True
    )
    comps["vl_encoder"] = enc_cls(encoder=comps["text_encoder"].encoder)


def transformer_cache_key(model_handle, role: str = "transformer") -> str:
    """transformer 单独一项（Ideogram 4 的两套 transformer 按 role 分键）。"""
    return runtime.cache_key({"kind": "transformer", "role": role, "model": model_handle})


def vae_component(handle, cache) -> Any:
    """按 MlxVaeHandle 的键物化 VAE（编码器 / 解码器 / RawPIL 三处共用同一份实例）。

    handle.cache_key 由 MlxVAELoader 算好（内容为
    `{"kind":"vae","model_type","path","precision","quantize"}`），因此「谁先跑谁物化、
    另一个命中同一条缓存」。采样阶段不使用 VAE，所以这里不参与采样器。

    MiniMax-H3 的两个 VAE 由 prepare_h3_vae 物化（媒体守卫会在这里挡住它）。
    """
    entry = entry_for(handle.model_type)
    validate_model_family(handle.model_type, handle.path)
    if not entry.supported:
        raise NotImplementedError(f"{entry.family} 尚未实现：{entry.notes}")
    _ensure_image_family(entry, "MLX VAE 编码 / 解码")
    kind, resolved = paths.resolve("local", handle.path, "vae")
    if kind == "missing":
        raise FileNotFoundError(f"未找到 vae 权重: {resolved}")

    def build():
        if entry.family == "qwen_image_21":
            return qwen21_loader.load(
                "vae",
                resolved,
                quantize=None,
                precision=handle.precision,
            ).module
        instance, _bits = components.create_and_load(
            entry, "vae", kind, resolved, int(handle.quantize) or None
        )
        return instance

    instance, _hit = cache.get_or_create("module", handle.cache_key, build)
    return instance


def prepare_sampler_components(entry, model_handle, cache) -> dict[str, Any]:
    """只物化 transformer（采样阶段根本不需要 VAE）。

    VAE 交给 MlxVAEEncoder / MlxVAEDecoder 按 MlxVaeHandle 的键物化；
    这里顺带加载 VAE 属于白加载（采样循环从来没用过它）。

    MiniMax-H3（media="video"）走 prepare_h3_sampler_components，不查 mflux 的
    ModelConfig 注册表（0.19.1 里没有 minimax-h3，兜底配置也是图片模型的那套）。
    """
    validate_model_family(entry.family, model_handle.model_path)
    if not entry.supported:
        raise NotImplementedError(f"{entry.family} 尚未实现：{entry.notes}")
    _ensure_image_family(entry, "MLX 采样器")
    model_config = (
        None
        if entry.family == "qwen_image_21"
        else weights.config_for_path(model_handle.model_path, entry.default_config)
    )

    def build_transformer(role: str = "transformer"):
        kind, path = component_path(entry, role, model_handle.model_path)
        if kind == "missing":
            raise FileNotFoundError(f"未找到 {role} 权重: {path}")
        # 构造参数必须按 ModelConfig 的 overrides 给（否则会建出维度对不上的实例），
        # 但 overrides 里还混着运行时开关（qwen_edit_plus 之类）→ 要按签名过滤
        if entry.family == "qwen_image_21":
            instance = qwen21_loader.load(
                role,
                path,
                quantize=model_handle.quantize or None,
                precision=model_handle.precision,
            ).module
        else:
            instance, _bits = components.create_and_load(
                entry,
                role,
                kind,
                path,
                model_handle.quantize or None,
                class_kwargs=component_class_kwargs(entry, role, path, model_config),
            )
        transformer_lora.apply_transformer_loras(
            entry.family,
            instance,
            model_handle.loras,
            role=role,
        )
        return instance

    roles = ("transformer", "unconditional_transformer") if entry.family == "ideogram4" else ("transformer",)
    return {
        role: cache.get_or_create(
            "module",
            transformer_cache_key(model_handle, role),
            lambda role=role: build_transformer(role),
        )[0]
        for role in roles
    }


def create_latents(defn, seed, height, width, batch_size) -> list[Any]:
    """每个 batch 的噪声 latent（按大类不同，形状与附带元数据不同）：

    - z_image：每项 `[16, 1, h/8, w/8]`，采样用 `transformer(timestep, x, cap_feats, sigmas)`；
    - flux2：每项 `(latents[1, seq, C], latent_ids[1, seq, 4], latent_h, latent_w)`，
      采样用打包 latent + grid ids，`Flux2LatentCreator` 没有 `create_noise`，
      只有 `prepare_packed_latents`；
    - qwen_image / qwen_edit：每项 `[1, (h/16)*(w/16), 64]`（`QwenLatentCreator`
      直接借 `FluxLatentCreator.create_noise`，给的就是打包好的纯噪声 latent，
      与「是否接参考图」无关 —— edit 那边另外还要参考图 latent，见
      `_sample_qwen_edit`）。
    """
    if defn.family == "qwen_image_21":
        return [
            qwen21_sampling.create_noise(seed + i, height, width)
            for i in range(batch_size)
        ]
    latent_creator = runtime.import_object(defn.latent_creator)
    if defn.family == "flux2":
        return [
            latent_creator.prepare_packed_latents(
                seed=seed + i, height=height, width=width, batch_size=1
            )
            for i in range(batch_size)
        ]
    return [latent_creator.create_noise(seed + i, height, width) for i in range(batch_size)]


def prompt_encoding_key(clip, text: str, ref_cache_key: str = "") -> str:
    """某条提示词的编码缓存键（同一 handle + 同一文本 → 同一键，可命中缓存）。"""
    return runtime.cache_key(
        {"kind": "prompt_encoding", "clip": clip, "text": text, "ref": ref_cache_key}
    )


def release_encoder(clip, cache) -> None:
    """释放一条普通图片文本编码器 bundle；编码结果仍留在 prompt cache。"""
    cache.evict("module", runtime.cache_key({"kind": "module", "clip": clip}))


def encode_text(
    defn,
    comps,
    text: str,
    cache,
    cache_key: str,
    reference_images: Sequence[Any] = (),
    max_length: int | None = None,
) -> Any:
    """用已加载的 tokenizer + text_encoder 编码一条文本（M1 不支持 prompt_cache）。

    - z_image：`encode_prompt` 返回单个 `cap_feats` 数组；
    - flux2：`encode_prompt` 返回 `(prompt_embeds, text_ids)` 二元组，且需要
      `num_images_per_prompt / max_sequence_length / text_encoder_out_layers`
      （取自 entry.prompt_encoder_args，`cache` 这个键 M1 不支持，跳过）；
    - qwen_image：`QwenPromptEncoder.encode_prompt` 是另一套签名（正/负一起编，
      还要显式给 `prompt_cache` 字典），本项目只要正向外，因此取返回值的
      `[0] / [1]` 存成 `(embeds, mask)`；`prompt_cache` 用不上（缓存由
      cache.py 的 prompt_encoding 桶负责）→ 给空 dict。
    缓存里就存返回值原样（数组或二元组），采样端按大类取用。
    """

    def build():
        if defn.family == "qwen_image_21":
            return encode_qwen21_prompt(
                comps["text_encoder"],
                comps["tokenizer"],
                text,
                max_length=max_length,
                images=reference_images,
            )
        prompt_encoder = runtime.import_object(defn.prompt_encoder)
        if defn.family == "flux2":
            args = {k: v for k, v in defn.prompt_encoder_args.items() if k != "cache"}
            return prompt_encoder.encode_prompt(
                prompt=text,
                tokenizer=comps["tokenizer"],
                text_encoder=comps["text_encoder"],
                **args,
            )
        if defn.family in ("qwen_image", "qwen_edit"):
            # 两条 Qwen 链路共用这套签名：negative 由另一个 MlxTextEncoder 单独编，
            # 因此这里只取正向外（第 3/4 个返回值恒为 None）
            embeds, mask, _neg_embeds, _neg_mask = prompt_encoder.encode_prompt(
                prompt=text,
                negative_prompt=None,
                prompt_cache={},
                qwen_tokenizer=comps["tokenizer"],
                qwen_text_encoder=comps["text_encoder"],
            )
            return (embeds, mask)
        return prompt_encoder.encode_prompt(
            prompt=text,
            tokenizer=comps["tokenizer"],
            text_encoder=comps["text_encoder"],
        )

    encoding, _hit = cache.get_or_create("prompt_encoding", cache_key, build)
    return encoding


def cached_encoding(cache, key: str, label: str) -> Any:
    """按缓存键取出 MlxTextEncoder / MlxQwenEditEncoder 算好的编码；缺失就提示重新运行。"""
    if not key:
        raise RuntimeError(f"未连接{label}向条件（MlxTextEncoder / MlxQwenEditEncoder 的输出）")
    encoding, hit = cache.get("prompt_encoding", key)
    if not hit or encoding is None:
        raise RuntimeError(f"{label}向条件的编码已失效，请重新运行「MLX 文本编码器」节点")
    return encoding


# --- Ideogram 4：条件编码依赖目标宽高，因此由采样器在加载 transformer 前完成 ---
IDEOGRAM4_SCHEDULERS: dict[str, str] = {
    "ideogram4_default": "V4_DEFAULT_20",
    "ideogram4_quality": "V4_QUALITY_48",
    "ideogram4_turbo": "V4_TURBO_12",
}
IDEOGRAM4_PROMPT_BUCKET = "ideogram4_prompt"


def ideogram4_preset(scheduler_name: str):
    """ComfyUI scheduler widget 值 → MFLUX 的 Ideogram 4 官方预设。"""
    preset_name = IDEOGRAM4_SCHEDULERS.get(scheduler_name)
    if preset_name is None:
        choices = ", ".join(IDEOGRAM4_SCHEDULERS)
        raise ValueError(
            f"Ideogram 4 必须选择自己的采样预设（{choices}），收到 {scheduler_name!r}"
        )
    scheduler = runtime.import_object(
        "mflux.models.ideogram4.model.ideogram4_scheduler.scheduler:Ideogram4Scheduler"
    )
    return scheduler.get_preset(preset_name)


def ideogram4_encoding_key(clip, prompt: str, width: int, height: int) -> str:
    return runtime.cache_key(
        {
            "kind": "ideogram4_prompt",
            "clip": clip,
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
        }
    )


def prepare_ideogram4_conditioning(entry, clip, prompt: str, width: int, height: int, cache) -> str:
    """编码 Ideogram 4 caption，并在返回前释放文本编码器。

    缓存值是 ``(inputs, text_features)``。其中 inputs 同时包含目标图像 token 的
    position/segment/indicator，因此宽高是缓存键的一部分，不能在 MlxTextEncoder
    节点（尚不知道采样尺寸）提前计算。
    """
    if entry.family != "ideogram4":
        raise ValueError(f"{entry.family} 不能走 Ideogram 4 条件编码")
    latent_creator = runtime.import_object(entry.latent_creator)
    latent_creator.validate_dimensions(width=int(width), height=int(height))
    prompt_encoder = runtime.import_object(entry.prompt_encoder)
    prompt = prompt_encoder.resolve_prompt(
        prompt,
        strict_caption_validation=False,
        warn_on_caption_issues=True,
    )
    key = ideogram4_encoding_key(clip, prompt, width, height)
    module_key = runtime.cache_key({"kind": "module", "clip": clip})

    def build():
        comps = prepare_encoder(entry, clip, cache, module_key)
        inputs = prompt_encoder.build_inputs(
            comps["tokenizer"], [prompt], height=int(height), width=int(width)
        )
        max_text_tokens = int(inputs["max_text_tokens"])
        # 官方 encode_prompt 把图像占位 token 一起送进 36 层 Qwen；它们位于文本之后且
        # attention mask 为 0，因果注意力下不会影响前面的文本输出。这里只编码真实文本段，
        # 与官方文本特征等价，同时避免在 1024² 时为 4096 个占位 token 做无效前向。
        token_ids = inputs["token_ids"][:, :max_text_tokens]
        attention_mask = mx.ones(token_ids.shape, dtype=mx.int32)
        position_ids = inputs["text_position_ids"][:, :max_text_tokens, 0]
        features = comps["text_encoder"].get_prompt_embeds(
            token_ids, attention_mask, position_ids
        )
        # Transformer 入口立即把官方 float32 输出转成 ModelConfig.precision；提前转换后缓存，
        # 数值等价且把条件缓存减半。图像 token 的零特征在采样阶段按目标尺寸补回。
        model_config_cls = runtime.import_object(
            "mflux.models.common.config.model_config:ModelConfig"
        )
        features = features.astype(model_config_cls.precision)
        mx.eval(features, inputs["position_ids"], inputs["segment_ids"], inputs["indicator"])
        print(
            f"[Ideogram 4 条件] {int(inputs['max_text_tokens'])} 个文本 token + "
            f"{int(inputs['num_image_tokens'])} 个图像 token | features {tuple(features.shape)}"
        )
        return inputs, features

    try:
        cache.get_or_create(IDEOGRAM4_PROMPT_BUCKET, key, build)
    finally:
        # 采样要同时驻留两套约 8.7 GB transformer；编码完成或失败后都立即让出文本编码器。
        cache.evict("module", module_key)
    return key


def edit_prompt_encoding_key(clip, text: str, image_digest: str, count: int) -> str:
    """Qwen 编辑条件的缓存键：handle + 文本 + **图片摘要** + 张数（换图必换键）。"""
    return runtime.cache_key(
        {
            "kind": "qwen_edit_prompt",
            "clip": clip,
            "text": text,
            "image": image_digest,
            "count": int(count),
        }
    )


def encode_edit_conditioning(entry, comps, text: str, images: Sequence[Any], cache, cache_key: str):
    """Qwen 编辑的文本条件（带参考图）→ `(embeds, mask)`，数组存 prompt_encoding 桶。

    等价 mflux `QwenImageEdit._encode_prompts_with_images` 的**一半**（正/负各调一次本函数）：
    VL tokenizer（按张数选模板）→ `tokenize_with_image` → 视觉塔 + 语言塔 →
    最后按 mflux 的约定转 fp16（`final_prompt_embeds.astype(mx.float16)`）。
    """
    if entry.family != "qwen_edit":
        raise NotImplementedError(f"{entry.family} 不支持带参考图的文本编码（目前只有 qwen_edit）")

    def build():
        tokenizer = comps["vl_tokenizer_multi"] if len(images) > 1 else comps["vl_tokenizer_single"]
        prompt = text if text and text.strip() else " "  # 空提示词按空格（与 MlxTextEncoder 一致）
        input_ids, attention_mask, pixel_values, grid_thw = tokenizer.tokenize_with_image(
            prompt, list(images)
        )
        embeds, embeds_mask = comps["vl_encoder"](
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
        )
        embeds = embeds.astype(mx.float16)
        embeds_mask = embeds_mask.astype(mx.float16)
        mx.eval(embeds, embeds_mask)
        print(
            f"[Qwen 编辑条件] {len(images)} 张参考图 | embeds {tuple(embeds.shape)} "
            f"{embeds.dtype} | mask {int(mx.sum(embeds_mask).item())}"
        )
        return (embeds, embeds_mask)

    value, _hit = cache.get_or_create("prompt_encoding", cache_key, build)
    return value


# --- Flux.2 参考图（edit） ---

# --- 多图参考图源（MlxRefImageSet → MlxVAEEncoder 的 ref_source 入口） ---
def ref_source_key(digest: str, sizes: Sequence[tuple[int, int]]) -> str:
    """参考图集的缓存键：有序摘要 + 每张尺寸（换图 / 换顺序 / 换张数都会换键）。"""
    return runtime.cache_key(
        {"kind": "ref_source", "image": digest, "sizes": [list(s) for s in sizes]}
    )


def store_ref_source(pils: Sequence[Any], cache, cache_key: str) -> tuple[Any, ...]:
    """把有序 PIL 元组存进 "ref_source" 桶（图片本体不进 handle）。"""
    value, _hit = cache.get_or_create("ref_source", cache_key, lambda: tuple(pils))
    return value


def load_ref_source(cache, cache_key: str) -> tuple[Any, ...]:
    """取出参考图集；桶里没有说明它已被淘汰（提示重跑图集节点，而不是静默少几张图）。"""
    if not cache_key:
        raise RuntimeError("ref_source 没有缓存键（MlxRefImageSet 的输出不完整）")
    value, hit = cache.get("ref_source", cache_key)
    if not hit or value is None:
        raise RuntimeError("参考图集已失效，请重新运行「MLX 参考图集」节点")
    return value


# --- Flux.2 参考图（edit）：编码与预处理（每张各自尺寸） ---
REFERENCE_MIN_DIM = 16  # 参考图每条边必须是 16 的倍数（vae scale 8 × patch 2）
REFERENCE_T_COORD_BASE = 10  # 第 i 张参考图的 grid ids t 坐标 = 10 + 10 * i（目标图是 0）
REFERENCE_T_COORD_STEP = 10
RESIZE_MODES = ("aspect_area_crop", "keep_resolution")


def reference_image_cache_key(vae_handle, image_digest: str, params: dict[str, Any]) -> str:
    """参考图条件的缓存键（换图 / 换预处理参数 / 换 VAE 都会换键）。"""
    return runtime.cache_key(
        {
            "kind": "reference_image",
            "vae": vae_handle,
            "image": image_digest,
            "params": params,
        }
    )


def _reference_dims(pil) -> tuple[int, int]:
    """把宽高向下取整到 16 的倍数（越界时会在调用处中心裁剪）。"""
    width = pil.width - pil.width % REFERENCE_MIN_DIM
    height = pil.height - pil.height % REFERENCE_MIN_DIM
    if width == 0 or height == 0:
        raise ValueError(f"参考图太小（每边至少 {REFERENCE_MIN_DIM} 像素）: {pil.size}")
    return width, height


def prepare_reference_image(pil, resize_mode: str):
    """参考图预处理。

    - `aspect_area_crop`（默认）：等价 mflux `_Flux2KleinEditHelpers.prepare_reference_image`
      —— 等比缩放到面积 ≤ 1024×1024，再中心裁剪到 16 的倍数（不拉伸）；
    - `keep_resolution`：保留原分辨率，只把宽高裁到 16 的倍数（更慢、更吃内存，慎用大图）。
    """
    if resize_mode == "aspect_area_crop":
        helpers = runtime.import_object(
            "mflux.models.flux2.variants.edit.flux2_klein_edit_helpers:_Flux2KleinEditHelpers"
        )
        return helpers.prepare_reference_image(pil)
    width, height = _reference_dims(pil)
    if (pil.width, pil.height) == (width, height):
        return pil
    left = (pil.width - width) // 2
    top = (pil.height - height) // 2
    return pil.crop((left, top, left + width, top + height))


def encode_reference_images(
    entry,
    vae,
    images: Sequence[Any],
    cache,
    cache_key: str,
    max_reference_images: int,
    resize_mode: str,
):
    """参考图批次 → `(packed [1, seq, 128], ids [1, seq, 4], width, height)`。

    与 mflux `_Flux2KleinEditHelpers.prepare_reference_image_conditioning` 逐步一致：
    预处理 → `VAEUtil.encode`（PIL 先经 `ImageUtil.to_array` 变成 [-1,1] 的 CHW）
    → `ensure_4d_latents` / `crop_to_even_spatial` → `patchify_latents`
    → `bn_normalize_vae_encoded_latents`（用 `vae.bn` 的统计量，漏掉会得到噪声图）
    → `pack_latents` → `prepare_grid_ids(t_coord=10 + 10 * i)`；多张沿 seq 维 concat。
    返回的 width/height 是首张参考图预处理后的尺寸（作为「建议生成尺寸」）。
    """
    helpers = runtime.import_object(
        "mflux.models.flux2.variants.edit.flux2_klein_edit_helpers:_Flux2KleinEditHelpers"
    )
    image_util = runtime.import_object("mflux.utils.image_util:ImageUtil")
    vae_util = runtime.import_object("mflux.models.common.vae.vae_util:VAEUtil")
    latent_creator = runtime.import_object(entry.latent_creator)
    selected = list(images)[: max(1, int(max_reference_images))]
    if not selected:
        raise ValueError("参考图为空")

    def build():
        packed: list[Any] = []
        ids: list[Any] = []
        width = height = 0
        for i, pil in enumerate(selected):
            prepared = prepare_reference_image(pil, resize_mode)
            if i == 0:
                width, height = prepared.width, prepared.height
            encoded = vae_util.encode(
                vae=vae, image=image_util.to_array(prepared), tiling_config=None
            )
            encoded = helpers.ensure_4d_latents(encoded)
            encoded = helpers.crop_to_even_spatial(encoded)
            encoded = latent_creator.patchify_latents(encoded)
            encoded = helpers.bn_normalize_vae_encoded_latents(encoded, vae=vae)
            packed.append(latent_creator.pack_latents(encoded))
            ids.append(
                latent_creator.prepare_grid_ids(
                    encoded,
                    t_coord=REFERENCE_T_COORD_BASE + REFERENCE_T_COORD_STEP * i,
                )
            )
        return mx.concatenate(packed, axis=1), mx.concatenate(ids, axis=1), width, height

    value, _hit = cache.get_or_create("ref_encoding", cache_key, build)
    return value


def cached_reference(cache, key: str):
    """取出参考图条件；缺失时提示重跑编码节点。

    - flux2：`(packed, ids, width, height)`；
    - qwen_edit：`(packed, ids, cond_h_patches, cond_w_patches)`（`cond_image_grid` 由
      调用方按参考图张数拼出）。
    """
    if not key:
        raise RuntimeError("未连接参考图（MlxVAEEncoder 的输出）")
    value, hit = cache.get("ref_encoding", key)
    if not hit or value is None:
        raise RuntimeError("参考图条件已失效，请重新运行「MLX VAE 编码」节点")
    return value


QWEN21_REFERENCE_MULTIPLE = 32
QWEN21_REFERENCE_DEFAULT_AREA = 1024 * 1024


def _qwen21_reference_dimensions(pil, width: int, height: int, keep_resolution: bool) -> tuple[int, int]:
    """Official 2.1 aspect-preserving reference resize, rounded to 32-pixel multiples."""
    if keep_resolution:
        use_w = round(int(pil.width) / QWEN21_REFERENCE_MULTIPLE) * QWEN21_REFERENCE_MULTIPLE
        use_h = round(int(pil.height) / QWEN21_REFERENCE_MULTIPLE) * QWEN21_REFERENCE_MULTIPLE
    else:
        area = (
            int(width) * int(height)
            if int(width) > 0 and int(height) > 0
            else QWEN21_REFERENCE_DEFAULT_AREA
        )
        ratio = float(pil.width) / float(pil.height)
        use_w = round(math.sqrt(area * ratio) / QWEN21_REFERENCE_MULTIPLE) * QWEN21_REFERENCE_MULTIPLE
        use_h = round(math.sqrt(area / ratio) / QWEN21_REFERENCE_MULTIPLE) * QWEN21_REFERENCE_MULTIPLE
    from comfyui_mlx_gen.h3.model.h3_text_encoder.qwen3_vl_vision_model import smart_resize

    # The Qwen3-VL processor enforces 256²..4096² pixels. Apply the same rule here so
    # its merged visual slots remain exactly one quarter of the VAE latent token count.
    use_h, use_w = smart_resize(
        max(QWEN21_REFERENCE_MULTIPLE, use_h),
        max(QWEN21_REFERENCE_MULTIPLE, use_w),
        QWEN21_REFERENCE_MULTIPLE,
        65536,
        16777216,
    )
    return use_w, use_h


def encode_qwen21_reference_images(
    vae,
    images: Sequence[Any],
    cache,
    cache_key: str,
    max_images: int,
    width: int,
    height: int,
    resize_mode: str,
):
    """Encode 2.1 reference pixels once for both Qwen3-VL and the block-causal DiT."""
    from PIL import Image as PILImage
    import numpy as np

    selected = list(images)[: min(10, max(1, int(max_images)))]
    if not selected:
        raise ValueError("Qwen-Image 2.1 参考图为空")
    if resize_mode not in ("auto", "aspect_area_crop", "keep_resolution"):
        raise ValueError(
            "qwen_image_21 只支持 auto / aspect_area_crop（保持比例、按面积缩放）或 "
            f"keep_resolution（只对齐 32 倍数），收到 {resize_mode}"
        )

    def build():
        prepared: list[Any] = []
        packed: list[Any] = []
        shapes: list[tuple[int, int]] = []
        for source in selected:
            use_w, use_h = _qwen21_reference_dimensions(
                source, width, height, resize_mode == "keep_resolution"
            )
            image_rgba = source.convert("RGBA")
            if image_rgba.size != (use_w, use_h):
                image_rgba = image_rgba.resize((use_w, use_h), PILImage.Resampling.LANCZOS)
            pixels = np.asarray(image_rgba, dtype=np.float32) / 127.5 - 1.0
            encoded = vae.encode(mx.array(pixels)[None])
            latent_h, latent_w = int(encoded.shape[2]), int(encoded.shape[3])
            packed.append(encoded.transpose(0, 2, 3, 1).reshape(1, latent_h * latent_w, 64))
            shapes.append((latent_h, latent_w))
            white = PILImage.new("RGB", image_rgba.size, (255, 255, 255))
            white.paste(image_rgba, mask=image_rgba.getchannel("A"))
            prepared.append(white)
        latents = mx.concatenate(packed, axis=1)
        mx.eval(latents)
        return {
            "latents": latents,
            "shapes": tuple(shapes),
            "images": tuple(prepared),
            "width": int(prepared[0].width),
            "height": int(prepared[0].height),
        }

    value, _hit = cache.get_or_create("ref_encoding", cache_key, build)
    return value


def cached_qwen21_reference(cache, key: str) -> dict[str, Any]:
    value = cached_reference(cache, key)
    if not isinstance(value, dict) or not {"latents", "shapes", "images"}.issubset(value):
        raise ValueError("参考图缓存不是 Qwen-Image 2.1 原生编辑条件")
    return value


# --- Qwen-Image-Edit 参考图（edit：按目标尺寸编码，拼在目标 latent 之后） ---
def _qwen_target_dims(width: int, height: int, pil) -> tuple[int, int]:
    """Qwen 编辑的目标尺寸：显式给就用给的，否则用首张参考图尺寸（都向下取到 16 的倍数）。"""
    use_w = int(width) or int(pil.width)
    use_h = int(height) or int(pil.height)
    use_w -= use_w % QWEN_EDIT_TARGET_MULTIPLE
    use_h -= use_h % QWEN_EDIT_TARGET_MULTIPLE
    if use_w <= 0 or use_h <= 0:
        raise ValueError(
            f"目标尺寸非法（每边至少 {QWEN_EDIT_TARGET_MULTIPLE} 像素）: {use_w}x{use_h}"
        )
    return use_w, use_h


def prepare_qwen_edit_reference_image(pil, width: int, height: int, resize_mode: str):
    """参考图 → 精确目标尺寸。

    - `stretch`（默认）：LANCZOS 直接拉伸，等价 mflux `ImageUtil.scale_to_dimensions`
      —— 与目标长宽比不同时主体会变形，但这是参考实现的既有行为；
    - `aspect_fit_pad`：等比缩放后居中 pad 黑边（不拉伸主体，多出来的边交给模型当背景）。
    """
    if resize_mode == "stretch":
        image_util = runtime.import_object("mflux.utils.image_util:ImageUtil")
        return image_util.scale_to_dimensions(pil, target_width=width, target_height=height)
    if resize_mode != "aspect_fit_pad":
        raise ValueError(f"未知的 resize_mode: {resize_mode}")
    from PIL import Image as PILImage  # 局部 import（pipeline 其余部分不依赖 PIL）

    ratio = min(width / pil.width, height / pil.height)
    new_w = max(1, int(round(pil.width * ratio)))
    new_h = max(1, int(round(pil.height * ratio)))
    resized = pil if (new_w, new_h) == pil.size else pil.resize((new_w, new_h), PILImage.LANCZOS)
    canvas = PILImage.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


def encode_edit_reference_latents(
    entry,
    vae,
    images: Sequence[Any],
    cache,
    cache_key: str,
    max_images: int,
    width: int,
    height: int,
    resize_mode: str,
):
    """qwen_edit 参考图条件 → `(packed, ids, cond_grid, use_w, use_h)`，数组存 ref_encoding 桶。

    与 mflux `QwenEditUtil.create_image_conditioning_latents` 逐步一致：每张图都缩放到
    **目标编辑尺寸**（不是各自原始尺寸）→ `VAEUtil.encode` → `QwenLatentCreator.pack_latents`
    → 沿 seq 维 concat。桶里存的是 `(packed, ids, cond_h_patches, cond_w_patches)`，
    `cond_grid` 由调用方按张数拼（多图是**列表**、单图是**元组**：transformer 内部按
    `isinstance(list)` 分支，单图给列表会把目标段算两遍）。
    """
    if entry.family != "qwen_edit":
        raise NotImplementedError(f"{entry.family} 不用这条参考图编码路径（只有 qwen_edit）")
    latent_creator = runtime.import_object(entry.latent_creator)
    image_util = runtime.import_object("mflux.utils.image_util:ImageUtil")
    vae_util = runtime.import_object("mflux.models.common.vae.vae_util:VAEUtil")
    selected = list(images)[: max(1, int(max_images))]
    if not selected:
        raise ValueError("参考图为空")
    use_w, use_h = _qwen_target_dims(width, height, selected[0])

    def build():
        packed = []
        for image in selected:
            prepared = prepare_qwen_edit_reference_image(image, use_w, use_h, resize_mode)
            encoded = vae_util.encode(
                vae=vae, image=image_util.to_array(prepared), tiling_config=None
            )
            packed.append(
                latent_creator.pack_latents(encoded, use_h, use_w, num_channels_latents=16)
            )
        latents = mx.concatenate(packed, axis=1)
        # 官方 ids：本机 mflux 的 transformer 不读它，仍按参考实现算出并缓存
        ids_util = runtime.import_object(
            "mflux.models.qwen.variants.edit.qwen_edit_util:QwenEditUtil"
        )
        ids = mx.concatenate(
            [ids_util._create_image_ids(height=use_h, width=use_w) for _ in selected], axis=1
        )
        mx.eval(latents, ids)
        print(
            f"[Qwen 参考图] {len(selected)} 张 → {use_w}x{use_h} | packed {tuple(latents.shape)} "
            f"{latents.dtype} | ids {tuple(ids.shape)}"
        )
        return latents, ids, use_h // 16, use_w // 16

    (latents, ids, h_patches, w_patches), _hit = cache.get_or_create(
        "ref_encoding", cache_key, build
    )
    cond_grid: Any = [(1, h_patches, w_patches)] * len(selected)
    if len(selected) == 1:
        cond_grid = (1, h_patches, w_patches)
    return latents, ids, cond_grid, use_w, use_h


def make_sampler_config(entry, model_config, params) -> Any:
    """Config 里需要的字段（num_inference_steps / height / width / guidance / scheduler）。

    model_config 由调用方按权重目录名解析出来（weights.config_for_path）；
    步数 / 调度器 / guidance 一律取自工作流的 widget。
    """
    cfg_cls = runtime.import_object("mflux.models.common.config.config:Config")
    return cfg_cls(
        model_config=model_config,
        num_inference_steps=params["steps"],
        height=params["height"],
        width=params["width"],
        guidance=params["guidance"],
        scheduler=params["scheduler_name"],
    )


def build_scheduler(scheduler_name: str, config: Any) -> Any:
    """按名字创建调度器（不在注册表里时回退到 linear）。"""
    reg = runtime.import_object("mflux.models.common.schedulers:SCHEDULER_REGISTRY")
    cls = reg.get(scheduler_name) or runtime.import_object(
        "mflux.models.common.schedulers.linear_scheduler:LinearScheduler"
    )
    scheduler = cls(config)
    if hasattr(scheduler, "set_image_seq_len") and config.model_config.requires_sigma_shift:
        scheduler.set_image_seq_len(config.image_seq_len)
    return scheduler


# --- 单步预测（编译产物在采样循环「外面」建一次）------------------------------
# 每个 predict 变体各占一份 CompiledPredictCache（z_image / flux2 / flux2_edit /
# flux2_edit_cached 互不挤占）：键只覆盖 Python 层的差异（是否开编译），
# 数组的形状与数值都由调用方当参数传进来，`mx.compile` 自己按签名重 trace，
# 因此换 prompt / 换尺寸 / 换步数都不会撞上旧计算图。
COMPILED_PREDICT: dict[str, CompiledPredictCache] = {}


def compiled_predict(variant: str) -> CompiledPredictCache:
    """取某 predict 变体的编译缓存（没有就新建一份，之后一直复用）。"""
    cache = COMPILED_PREDICT.get(variant)
    if cache is None:
        cache = CompiledPredictCache()
        COMPILED_PREDICT[variant] = cache
    return cache


def make_z_image_predict(transformer, use_compile: bool):
    """造 Z-Image 的单步预测（含 CFG；与 ZImage._predict 一致），返回可复用的 callable。

    工厂只在采样循环**外面**调一次：`mx.compile` 因此只发生一次，
    所有 seed、所有 step 共用同一份 trace（旧写法每步都重新 `mx.compile` 一次）；
    编译产物还按「模块 + 编译开关」存在 `COMPILED_PREDICT` 里，
    同一份权重的第二次采样连 trace 都不用重建。

    返回的签名：`predict(latents, timestep, sigmas, encodings, negative, guidance)`。
    """

    def predict(x, t, s, feats, neg_feats, scale):
        noise = transformer(timestep=t, x=x, cap_feats=feats, sigmas=s)
        if neg_feats is None:
            return noise
        negative = transformer(timestep=t, x=x, cap_feats=neg_feats, sigmas=s)
        return negative + scale * (noise - negative)

    return compiled_predict("z_image").get_or_build(
        key=("t2i", bool(use_compile)),
        weights_token=transformer,
        build=lambda: mx.compile(predict) if use_compile else predict,
    )


def make_flux2_predict(transformer, use_compile: bool):
    """造 Flux2 的单步预测（含 CFG；与 Flux2Klein._predict 一致），返回可复用的 callable。

    与 z_image 同理：编译只做一次，整段采样（含多个 batch 项）复用同一份 trace。
    latents 是打包形式 `[B, seq, C]`，配套 `latent_ids`（grid ids `[B, seq, 4]`）；
    encodings / negative 都是 `(prompt_embeds, text_ids)` 二元组（negative 可为 None）。
    CFG 组合顺序与参考实现一致：`negative + guidance * (noise - negative)`。

    返回的签名：`predict(latents, latent_ids, timestep, encodings, negative, guidance)`。
    """

    def predict(x, img_ids, t, enc, neg, scale):
        p_embeds, p_ids = enc
        noise = transformer(
            hidden_states=x,
            encoder_hidden_states=p_embeds,
            timestep=t,
            img_ids=img_ids,
            txt_ids=p_ids,
            guidance=None,
        )
        if neg is None:
            return noise
        n_embeds, n_ids = neg
        negative_noise = transformer(
            hidden_states=x,
            encoder_hidden_states=n_embeds,
            timestep=t,
            img_ids=img_ids,
            txt_ids=n_ids,
            guidance=None,
        )
        return negative_noise + scale * (noise - negative_noise)

    return compiled_predict("flux2").get_or_build(
        key=("t2i", bool(use_compile)),
        weights_token=transformer,
        build=lambda: mx.compile(predict) if use_compile else predict,
    )


def _sample_z_image(defn, comps, params, cache, model_config, guidance, on_progress=None):
    """Z-Image 采样循环（与 ZImage.generate_image 一致：单数组条件 + linear 调度器）。"""
    config = make_sampler_config(defn, model_config, params)
    scheduler = build_scheduler(params["scheduler_name"], config)
    per_seed = create_latents(
        defn, params["seed"], params["height"], params["width"], params["batch_size"]
    )
    encodings = cached_encoding(cache, params["positive_encoding_key"], "正")
    negative = (
        cached_encoding(cache, params["negative_encoding_key"], "负")
        if guidance > 1.0
        else None
    )
    transformer = comps["transformer"]
    # 单步预测（含编译）在循环外只建一次：所有 seed / 所有 step 复用同一份 trace
    predict = make_z_image_predict(transformer, bool(params.get("compile_model", True)))
    final = []
    for latents in per_seed:
        current = latents
        for t in range(params["steps"]):
            sigma_t = scheduler.sigmas[t].reshape((1,))
            timestep = mx.ones_like(sigma_t) - sigma_t
            noise = predict(current, timestep, scheduler.sigmas, encodings, negative, guidance)
            current = scheduler.step(noise=noise, timestep=t, latents=current)
            # 计时必须跨过 mx.eval：MLX 惰性求值，eval 之前只建了计算图，
            # 时间都在 eval 物化时才花掉（在 eval 前取时间只会打出约 0）
            started = time.perf_counter()
            mx.eval(current)
            print(
                f"[Z-Image 采样] step {t + 1}/{params['steps']} "
                f"耗时 {time.perf_counter() - started:.2f}s"
            )
            if on_progress is not None:
                on_progress()
        final.append(current)
    stacked = mx.stack(final, axis=0)  # [B, 16, 1, h/8, w/8]
    mx.eval(stacked)
    return stacked


def _sample_flux2(defn, comps, params, cache, model_config, guidance, on_progress=None):
    """Flux2 采样循环（与 Flux2Klein.generate_image 的 txt2img 一致）。

    与 Z-Image 的差别：
    - latent 是打包形式 `[1, seq, C]`，配套 grid ids（不会 unpack 后才进采样）；
    - 条件编码是 `(prompt_embeds, text_ids)` 二元组；
    - transformer 用 `hidden_states / encoder_hidden_states / img_ids / txt_ids`；
    - timestep 取 `scheduler.timesteps[t]`，且 `step` 要显式传 `sigmas`。
    每个 batch 项用 seed+i 单独采样（batch=1），最后沿 batch 维拼接成 `[B, seq, C]`。
    """
    config = make_sampler_config(defn, model_config, params)
    scheduler = build_scheduler(params["scheduler_name"], config)
    per_seed = create_latents(
        defn, params["seed"], params["height"], params["width"], params["batch_size"]
    )
    encodings = cached_encoding(cache, params["positive_encoding_key"], "正")
    negative = (
        cached_encoding(cache, params["negative_encoding_key"], "负")
        if guidance > 1.0
        else None
    )
    apple_silicon = runtime.import_object("mflux.utils.apple_silicon:AppleSiliconUtil")
    use_compile = bool(params.get("compile_model", True)) and not apple_silicon.is_m1_or_m2()
    transformer = comps["transformer"]
    # 同 z_image：编译只发生在循环外，整段采样（含 batch 内多个 seed）复用同一份 trace
    predict = make_flux2_predict(transformer, use_compile)
    final = []
    for latents, latent_ids, _latent_h, _latent_w in per_seed:
        current = latents  # [1, seq, C]
        for t in range(params["steps"]):
            timestep = scheduler.timesteps[t]
            noise = predict(current, latent_ids, timestep, encodings, negative, guidance)
            current = scheduler.step(
                noise=noise, timestep=t, latents=current, sigmas=scheduler.sigmas
            )
            mx.eval(current)
            if on_progress is not None:
                on_progress()
        final.append(current)
    stacked = mx.concatenate(final, axis=0)  # [B, seq, C]
    mx.eval(stacked)
    return stacked


def _is_m1_or_m2() -> bool:
    """与 mflux 一致：M1/M2 上不编译 edit 的预测函数（取不到就当 False）。"""
    try:
        util = runtime.import_object("mflux.utils.apple_silicon:AppleSiliconUtil")
        return bool(util.is_m1_or_m2())
    except Exception:  # noqa: BLE001
        return False


def _new_kv_cache(transformer):
    """与 mflux `Flux2KleinEdit._new_kv_cache` 一致（层数从 transformer 现取）。"""
    kv_cls = runtime.import_object(
        "mflux.models.flux2.model.flux2_transformer.flux2_kv_cache:Flux2KVCache"
    )
    return kv_cls(
        num_double_layers=len(transformer.transformer_blocks),
        num_single_layers=len(transformer.single_transformer_blocks),
    )


def make_flux2_edit_predict(transformer, use_compile: bool):
    """造 Flux2 edit 的单步预测（含 CFG；与 mflux `Flux2KleinEdit._predict` 一致）。

    目标 token 在前、参考图 token 在后；`noise` 只取回目标段
    `[:, :latents.shape[1]]`；CFG 组合顺序同样是 `neg + scale * (pos - neg)`。
    工厂只在采样循环**外面**调一次，所有 step / seed 复用同一份编译产物。
    带 kv 缓存时只有第 1 步（`extract`）走这里 —— 之后由
    `make_flux2_edit_cached_predict` 只送目标 token，因此启用 kv 时本工厂
    始终收到 `use_compile=False`（与 mflux 一致：kv 缓存对象要在步之间改状态，
    既不能烘进计算图，也不是数组 / 标量常量，不能当编译入参）。

    返回的签名：
    `predict(latents, ref_latents, latent_ids, ref_ids, timestep, encodings, negative,
    guidance, kv_cache, negative_kv_cache)`。
    """

    def predict(x, refs, ids, rids, t, enc, neg, scale, kvc, nkvc):
        p_embeds, p_ids = enc
        hidden = mx.concatenate([x, refs], axis=1)
        img_ids = mx.concatenate([ids, rids], axis=1)
        noise = transformer(
            hidden_states=hidden,
            encoder_hidden_states=p_embeds,
            timestep=t,
            img_ids=img_ids,
            txt_ids=p_ids,
            guidance=None,
            kv_cache=kvc,
        )
        noise = noise[:, : x.shape[1]]
        if neg is None:
            return noise
        n_embeds, n_ids = neg
        negative_noise = transformer(
            hidden_states=hidden,
            encoder_hidden_states=n_embeds,
            timestep=t,
            img_ids=img_ids,
            txt_ids=n_ids,
            guidance=None,
            kv_cache=nkvc or kvc,
        )
        negative_noise = negative_noise[:, : x.shape[1]]
        return negative_noise + scale * (noise - negative_noise)

    return compiled_predict("flux2_edit").get_or_build(
        key=("extract", bool(use_compile)),
        weights_token=transformer,
        build=lambda: mx.compile(predict) if use_compile else predict,
    )


def make_flux2_edit_cached_predict(transformer):
    """kv 缓存命中（`mode="cached"`）时的 edit 单步预测工厂：只送目标 token。

    与 mflux `Flux2KleinEdit._cached_predict` 一致：参考图 token 的 K/V 已在第 1 步
    抽进 kv 缓存，这里不再 concat 参考图；与 mflux 一样**不编译**，
    工厂仍只在循环外调一次，省掉每步重建函数的开销。

    返回的签名：
    `predict(latents, latent_ids, timestep, encodings, negative, guidance,
    kv_cache, negative_kv_cache)`。
    """

    def predict(x, ids, t, enc, neg, scale, kvc, nkvc):
        p_embeds, p_ids = enc
        noise = transformer(
            hidden_states=x,
            encoder_hidden_states=p_embeds,
            timestep=t,
            img_ids=ids,
            txt_ids=p_ids,
            guidance=None,
            kv_cache=kvc,
        )
        noise = noise[:, : x.shape[1]]
        if neg is None:
            return noise
        n_embeds, n_ids = neg
        negative_noise = transformer(
            hidden_states=x,
            encoder_hidden_states=n_embeds,
            timestep=t,
            img_ids=ids,
            txt_ids=n_ids,
            guidance=None,
            kv_cache=nkvc or kvc,
        )
        negative_noise = negative_noise[:, : x.shape[1]]
        return negative_noise + scale * (noise - negative_noise)

    # 始终不编译（见上）；仍然走缓存，让同一份权重反复采样时复用同一个函数对象
    return compiled_predict("flux2_edit_cached").get_or_build(
        key=("cached", False),
        weights_token=transformer,
        build=lambda: predict,
    )


def _sample_flux2_edit(defn, comps, params, cache, model_config, guidance, on_progress=None):
    """Flux.2 参考图编辑采样：循环与 `_sample_flux2` 相同，只换单步预测为 edit 版。

    - 参考图 latent 按缓存键从 cache.py 的 `ref_encoding` 桶取（编码节点已算好）；
    - kv 缓存只在 `model_config.supports_kv_cache`（kv 版权重才有）且没有关掉时启用：
      第 1 步 `extract`（拼参考图）、之后 `cached`（只送目标 token），此时不走
      `mx.compile`（与 mflux 一致）；negative 有独立 cache；
    - 目标 latent 每个 seed 单独采（batch=1），参考图也是 batch 1，直接 concat；
    - 目标 latent 形状与 txt2img 完全相同 → 下游解码/保存链路无需改动。
    """
    config = make_sampler_config(defn, model_config, params)
    scheduler = build_scheduler(params["scheduler_name"], config)
    ref_latents, ref_ids, _ref_w, _ref_h = cached_reference(cache, params["ref_cache_key"])
    per_seed = create_latents(
        defn, params["seed"], params["height"], params["width"], params["batch_size"]
    )
    encodings = cached_encoding(cache, params["positive_encoding_key"], "正")
    negative = (
        cached_encoding(cache, params["negative_encoding_key"], "负") if guidance > 1.0 else None
    )
    kv_enabled = (
        params.get("kv_cache", "auto") == "auto"
        and bool(model_config.supports_kv_cache)
        and ref_latents.shape[1] > 0
    )
    use_compile = bool(params.get("compile_model", True)) and not kv_enabled and not _is_m1_or_m2()
    transformer = comps["transformer"]
    # 两份单步预测（第 1 步的 extract、之后的 cached）都在循环外建一次
    predict_extract = make_flux2_edit_predict(transformer, use_compile)
    predict_cached = make_flux2_edit_cached_predict(transformer)
    final = []
    for latents, latent_ids, _latent_h, _latent_w in per_seed:
        current = latents  # [1, seq, C]
        kv_cache = _new_kv_cache(transformer) if kv_enabled else None
        negative_kv_cache = (
            _new_kv_cache(transformer) if (kv_enabled and negative is not None) else None
        )
        for t in range(params["steps"]):
            if kv_cache is not None:
                mode = "extract" if t == 0 else "cached"
                kv_cache.configure(mode=mode, num_ref_tokens=ref_latents.shape[1])
                if negative_kv_cache is not None:
                    negative_kv_cache.configure(mode=mode, num_ref_tokens=ref_latents.shape[1])
            timestep = scheduler.timesteps[t]
            if kv_cache is not None and t > 0:
                noise = predict_cached(
                    current,
                    latent_ids,
                    timestep,
                    encodings,
                    negative,
                    guidance,
                    kv_cache,
                    negative_kv_cache,
                )
            else:
                noise = predict_extract(
                    current,
                    ref_latents,
                    latent_ids,
                    ref_ids,
                    timestep,
                    encodings,
                    negative,
                    guidance,
                    kv_cache,
                    negative_kv_cache,
                )
            current = scheduler.step(
                noise=noise, timestep=t, latents=current, sigmas=scheduler.sigmas
            )
            mx.eval(current)
            if on_progress is not None:
                on_progress()
        final.append(current)
    stacked = mx.concatenate(final, axis=0)  # [B, seq, C]
    mx.eval(stacked)
    return stacked


QWEN_EDIT_EPS = 1e-12


def qwen_guided_noise(noise: Any, noise_negative: Any, guidance: float) -> Any:
    """与 `QwenImage.compute_guided_noise` 逐行一致：CFG 之后再按条件范数重标定。

    普通写法 `neg + g·(pos - neg)` 在 Qwen 上偏差明显（这就是它单独实现的原因）。
    """
    combined = noise_negative + guidance * (noise - noise_negative)
    cond_norm = mx.sqrt(mx.sum(noise * noise, axis=-1, keepdims=True) + QWEN_EDIT_EPS)
    noise_norm = mx.sqrt(mx.sum(combined * combined, axis=-1, keepdims=True) + QWEN_EDIT_EPS)
    return combined * (cond_norm / noise_norm)


def _sample_qwen_edit(defn, comps, params, cache, model_config, guidance, on_progress=None):
    """Qwen-Image-Edit 采样：目标 latent 与参考 latent 拼 seq，每个 step 只取回目标段。

    与 mflux `QwenImageEdit.generate_image` 的循环逐步一致：
    - 参考图条件从 `ref_encoding` 桶按 `ref_cache_key` 取（`(packed, ids, cond_h, cond_w)`），
      `cond_image_grid` 按参考图张数拼（多图列表 / 单图元组，与 mflux 一致）；
    - transformer 的 `t` 传**步号 int**（内部取 `config.scheduler.sigmas[t]`）；
    - CFG 用 `qwen_guided_noise`；`guidance<=1.0` 时跳过负向分支；
    - 不做 `mx.compile`（入参含 Config 对象与 int，`entry.supports_compile=False`）；
    - 每个 batch 项单独采样（batch=1），最后沿 batch 维拼接，形状与 txt2img 一致
      → 下游 `MlxVAEDecoder` / `MlxSaveImage` 不用改。
    """
    config = make_sampler_config(defn, model_config, params)
    scheduler = build_scheduler(params["scheduler_name"], config)
    ref_latents, ref_ids, h_patches, w_patches = cached_reference(cache, params["ref_cache_key"])
    ref_count = max(1, int(params.get("ref_count") or 1))
    cond_grid: Any = [(1, h_patches, w_patches)] * ref_count
    if ref_count == 1:
        cond_grid = (1, h_patches, w_patches)
    encodings = cached_encoding(cache, params["positive_encoding_key"], "正")
    negative = (
        cached_encoding(cache, params["negative_encoding_key"], "负") if guidance > 1.0 else None
    )
    latent_creator = runtime.import_object(defn.latent_creator)
    transformer = comps["transformer"]
    final = []
    for i in range(params["batch_size"]):
        latents = latent_creator.create_noise(
            params["seed"] + i, params["height"], params["width"]
        )  # [1, (h/16)(w/16), 64]
        for t in range(params["steps"]):
            target_len = latents.shape[1]
            hidden = mx.concatenate([latents, ref_latents], axis=1)  # 目标段在前，参考段在后
            noise = transformer(
                t=t,
                config=config,
                hidden_states=hidden,
                encoder_hidden_states=encodings[0],
                encoder_hidden_states_mask=encodings[1],
                qwen_image_ids=ref_ids,  # 当前 mflux 不读它，按参考实现传入
                cond_image_grid=cond_grid,
            )[:, :target_len]
            if negative is not None:
                negative_noise = transformer(
                    t=t,
                    config=config,
                    hidden_states=hidden,
                    encoder_hidden_states=negative[0],
                    encoder_hidden_states_mask=negative[1],
                    qwen_image_ids=ref_ids,
                    cond_image_grid=cond_grid,
                )[:, :target_len]
                noise = qwen_guided_noise(noise, negative_noise, guidance)
            latents = scheduler.step(noise=noise, timestep=t, latents=latents)
            mx.eval(latents)
            if on_progress is not None:
                on_progress()
        final.append(latents)
    stacked = mx.concatenate(final, axis=0)  # [B, seq, C]
    mx.eval(stacked)
    return stacked


def make_qwen_predict(transformer, config, use_compile: bool):
    """造 Qwen 的单步预测（含 CFG；与 mflux `QwenImage._predict` 一致），返回可复用的 callable。

    latents 是打包形式 `[1, seq, 64]`，条件 `encodings` 是 `(embeds, mask)`；
    `step` 是**步号 int**（transformer 内部按 `config.scheduler.sigmas[step]`
    反查 sigma，与 mflux 的 `for t in config.time_steps` 一致），因此
    CFG 组合用 `qwen_guided_noise`（普通 CFG 之后再按条件范数重标定）。

    与 flux2 / z_image 的工厂不同，这里**不编译**，因此也不进
    `CompiledPredictCache`：transformer 要整份 `config` 和步号 int，
    两者都不是数组 / 标量常量，当编译入参会直接把计算图烘坏（把建函数时
    的 Config 留在计算图里，换了步数 / 尺寸 / 调度器就会算错）；
    `entry.supports_compile=False` 也已经把 `use_compile` 关掉，
    工厂仍然只在采样循环**外面**调一次，循环里复用同一个闭包。

    返回的签名：`predict(latents, step, encodings, negative, guidance)`。
    """

    def predict(x, step, enc, neg, scale):
        p_embeds, p_mask = enc
        noise = transformer(
            t=step,
            config=config,
            hidden_states=x,
            encoder_hidden_states=p_embeds,
            encoder_hidden_states_mask=p_mask,
        )
        if neg is None:
            return noise
        n_embeds, n_mask = neg
        negative_noise = transformer(
            t=step,
            config=config,
            hidden_states=x,
            encoder_hidden_states=n_embeds,
            encoder_hidden_states_mask=n_mask,
        )
        return qwen_guided_noise(noise, negative_noise, scale)

    return predict


def _sample_qwen_image(defn, comps, params, cache, model_config, guidance, on_progress=None):
    """Qwen-Image 文生图（与 mflux `QwenImage.generate_image` 的 t2i 分支一致）。

    与 `_sample_qwen_edit` 的三点差别：

    - latent 由 `create_noise` 纯噪声起，**不接参考图**，也就没有
      `zero_cond_t` / `cond_image_grid` 这两个 edit 专用入参；
    - 条件是「MLX 文本编码器」算好的 `(embeds, mask)`（`encode_text` 的 Qwen 分支），
      不是带图片 token 的编辑条件；
    - 默认调度器是 `flow_match_euler_discrete`、guidance 4.0（mflux 的
      `ModelConfig.qwen_image` 与 `QwenImage.generate_image` 的默认值）。

    其余与 edit 相同：`t` 传步号 int、`scale_model_input` 恒等、CFG 走
    `qwen_guided_noise`、每个 batch 项单独采样后沿 batch 拼接 →
    latent 形状同样是 `[B, seq, 64]`，因此 `decode_latents` 与下游不用改。
    """
    config = make_sampler_config(defn, model_config, params)
    scheduler = build_scheduler(params["scheduler_name"], config)
    per_seed = create_latents(
        defn, params["seed"], params["height"], params["width"], params["batch_size"]
    )
    encodings = cached_encoding(cache, params["positive_encoding_key"], "正")
    negative = (
        cached_encoding(cache, params["negative_encoding_key"], "负") if guidance > 1.0 else None
    )
    transformer = comps["transformer"]
    # 单步预测在循环外建一次，所有 seed / 所有 step 复用同一个闭包
    # （Qwen 这条链路始终不编译，见 make_qwen_predict）
    predict = make_qwen_predict(transformer, config, bool(params.get("compile_model", True)))
    final = []
    for latents in per_seed:
        current = latents  # [1, (h/16)(w/16), 64]
        for t in range(params["steps"]):
            # 与参考实现一致地过一遍 scale_model_input（Qwen 用的两个调度器都继承
            # BaseScheduler 的恒等实现，因此这里等价于直接用 current）
            scaled = scheduler.scale_model_input(current, t)
            noise = predict(scaled, t, encodings, negative, guidance)
            current = scheduler.step(noise=noise, timestep=t, latents=scaled)
            mx.eval(current)
            if on_progress is not None:
                on_progress()
        final.append(current)
    stacked = mx.concatenate(final, axis=0)  # [B, seq, 64]
    mx.eval(stacked)
    return stacked


def _sample_qwen_image_21(defn, comps, params, cache, guidance, on_progress=None):
    """Qwen-Image 2.1 unified T2I/edit sampling with block-causal prefix KV cache.

    条件编码缓存是 ``(pre_norm_embeds, attention_mask, image_pad_mask)``。T2I 的
    T2I 的 ``image_pad_mask`` 全 False；编辑条件中的视觉槽与参考 VAE latent 来自同一
    个 ref cache。每个视觉槽在 DiT 中扩展成一个 2×2 latent block。
    """
    if params["scheduler_name"] != "flow_match_euler_discrete":
        raise ValueError(
            "Qwen-Image 2.1 只能使用 flow_match_euler_discrete scheduler，"
            f"收到 {params['scheduler_name']!r}"
        )
    height, width = qwen21_sampling.validate_dimensions(params["height"], params["width"])
    latent_h, latent_w = height // 16, width // 16
    positive = cached_encoding(cache, params["positive_encoding_key"], "正")
    negative = (
        cached_encoding(cache, params["negative_encoding_key"], "负")
        if guidance > 1.0
        else None
    )
    transformer = comps["transformer"]
    reference = (
        cached_qwen21_reference(cache, params["ref_cache_key"])
        if params.get("ref_cache_key")
        else None
    )
    reference_latents = reference["latents"] if reference is not None else None
    reference_shapes = reference["shapes"] if reference is not None else ()
    sigmas = qwen21_sampling.sigma_schedule(params["steps"], latent_h * latent_w)
    use_kv = params.get("kv_cache", "auto") != "off"
    final = []
    for batch_index in range(int(params["batch_size"])):
        current = qwen21_sampling.create_noise(
            int(params["seed"]) + batch_index, height, width
        )
        positive_cache = QwenImage21KVCache(transformer.num_layers) if use_kv else None
        negative_cache = (
            QwenImage21KVCache(transformer.num_layers) if use_kv and negative is not None else None
        )
        for step in range(int(params["steps"])):
            mode = ("extract" if step == 0 else "cached") if use_kv else None
            timestep = mx.full((1,), float(sigmas[step]), dtype=mx.float32)
            noise = transformer(
                hidden_states=current,
                encoder_hidden_states=positive[0],
                timestep=timestep,
                height=latent_h,
                width=latent_w,
                kv_cache=positive_cache,
                kv_cache_mode=mode,
                reference_latents=reference_latents,
                reference_shapes=reference_shapes,
                image_pad_mask=positive[2],
            )
            if negative is not None:
                negative_noise = transformer(
                    hidden_states=current,
                    encoder_hidden_states=negative[0],
                    timestep=timestep,
                    height=latent_h,
                    width=latent_w,
                    kv_cache=negative_cache,
                    kv_cache_mode=mode,
                    reference_latents=reference_latents,
                    reference_shapes=reference_shapes,
                    image_pad_mask=negative[2],
                )
                noise = negative_noise + float(guidance) * (noise - negative_noise)
            current = qwen21_sampling.euler_step(
                noise, current, float(sigmas[step]), float(sigmas[step + 1])
            )
            started = time.perf_counter()
            mx.eval(current)
            print(
                f"[Qwen-Image 2.1 采样] step {step + 1}/{params['steps']} "
                f"耗时 {time.perf_counter() - started:.2f}s"
            )
            if on_progress is not None:
                on_progress()
        final.append(current)
    result = mx.concatenate(final, axis=0)
    mx.eval(result)
    return result


def _sample_ideogram4(defn, comps, params, cache, on_progress=None):
    """Ideogram 4 FP8 文生图；与 MFLUX Ideogram4.generate_image 的去噪语义一致。"""
    preset = ideogram4_preset(params["scheduler_name"])
    num_steps = int(preset.num_steps)
    prompt_data, hit = cache.get(IDEOGRAM4_PROMPT_BUCKET, params["positive_encoding_key"])
    if not hit or prompt_data is None:
        raise RuntimeError("Ideogram 4 正向条件已失效，请重新运行「MLX 文本编码器」节点")
    inputs, text_features = prompt_data
    scheduler = runtime.import_object(
        "mflux.models.ideogram4.model.ideogram4_scheduler.scheduler:Ideogram4Scheduler"
    )
    t_values, s_values = scheduler.make_timesteps(
        num_steps=num_steps,
        height=int(params["height"]),
        width=int(params["width"]),
        mu=float(preset.mu),
        std=float(preset.std),
    )
    latent_creator = runtime.import_object(defn.latent_creator)
    conditional = comps["transformer"]
    unconditional = comps["unconditional_transformer"]
    max_text_tokens = int(inputs["max_text_tokens"])
    num_image_tokens = int(inputs["num_image_tokens"])
    llm_features = mx.concatenate(
        [
            text_features,
            mx.zeros(
                (text_features.shape[0], num_image_tokens, text_features.shape[-1]),
                dtype=text_features.dtype,
            ),
        ],
        axis=1,
    )
    text_padding = mx.zeros(
        (1, max_text_tokens, conditional.config.in_channels), dtype=mx.float32
    )
    prompt_encoder = runtime.import_object(defn.prompt_encoder)
    negative_inputs = prompt_encoder.negative_inputs(inputs, llm_features)
    apple_silicon = runtime.import_object("mflux.utils.apple_silicon:AppleSiliconUtil")
    use_compile = bool(params.get("compile_model", True)) and not apple_silicon.is_m1_or_m2()

    def predict_conditional(z, t, padding, features):
        joined = mx.concatenate([padding, z], axis=1)
        output = conditional(
            llm_features=features,
            x=joined,
            t=t,
            position_ids=inputs["position_ids"],
            segment_ids=inputs["segment_ids"],
            indicator=inputs["indicator"],
        )
        return output[:, max_text_tokens:, :]

    def predict_unconditional(z, t):
        return unconditional(
            llm_features=negative_inputs["llm_features"],
            x=z,
            t=t,
            position_ids=negative_inputs["position_ids"],
            segment_ids=negative_inputs["segment_ids"],
            indicator=negative_inputs["indicator"],
        )

    if use_compile:
        predict_conditional = mx.compile(predict_conditional)
        predict_unconditional = mx.compile(predict_unconditional)

    final = []
    for batch_index in range(int(params["batch_size"])):
        z = latent_creator.create_noise(
            seed=int(params["seed"]) + batch_index,
            width=int(params["width"]),
            height=int(params["height"]),
            latent_dim=conditional.config.in_channels,
        )
        # 官方循环正序 step、反序读取 schedule（高噪声 → 低噪声）。
        for step_index in range(num_steps):
            schedule_index = num_steps - 1 - step_index
            t_value = float(t_values[schedule_index])
            s_value = float(s_values[schedule_index])
            guide = float(preset.guidance_schedule[schedule_index])
            t = mx.full((1,), t_value, dtype=mx.float32)
            pos_v = predict_conditional(z, t, text_padding, llm_features)
            if abs(guide - 1.0) < 1e-6:
                velocity = pos_v
            else:
                neg_v = predict_unconditional(z, t)
                velocity = guide * pos_v + (1.0 - guide) * neg_v
            z = z + velocity * (s_value - t_value)
            mx.eval(z)
            if on_progress is not None:
                on_progress()
        final.append(z)
    stacked = mx.concatenate(final, axis=0)
    mx.eval(stacked)
    return stacked


def run_sampler(defn, model_handle, comps, params, cache):
    """跑完整采样，返回带缓存数组的 latent 句柄。

    正/负条件的编码由 MlxTextEncoder / MlxQwenEditEncoder 提前存进 cache 的
    "prompt_encoding" 桶，params 里只有编码键；按键取不到就报错（说明编码已换掉，
    需要重新运行）。

    配置按「权重集目录名」在 mflux 的 ModelConfig 注册表里现取（不依赖预先
    登记的模型名）：z-image-turbo-8bit / z-image-turbo-4bit 都会命中
    z-image-turbo，z-image-8bit 命中 z_image，因此新增权重目录不用改这里；
    步数 / 调度器 / guidance 由工作流的 widget 决定。
    """
    validate_model_family(defn.family, model_handle.model_path)
    if not defn.supported:
        raise NotImplementedError(f"{defn.family} 尚未实现：{defn.notes}")
    model_config = (
        None
        if defn.family == "qwen_image_21"
        else weights.config_for_path(model_handle.model_path, defn.default_config)
    )
    # 只有**明确声明**不支持 CFG（False，如 z-image-turbo）才清零；
    # None（qwen-image-edit）表示「未声明」，保留 widget 上的值 —— Qwen 编辑必须有 CFG。
    guidance = (
        float(params["guidance"])
        if model_config is None or model_config.supports_guidance is not False
        else 0.0
    )
    progress_steps = (
        int(ideogram4_preset(params["scheduler_name"]).num_steps)
        if defn.family == "ideogram4"
        else int(params["steps"])
    )
    progress = SamplingProgress(progress_steps * int(params["batch_size"]))

    def sample():
        # 接了参考图 → 编辑（edit）分支；按大类分派，不静默降级
        # （qwen_image 是文生图大类，接参考图在「MLX KSampler」里就被挡掉了）
        if params.get("ref_cache_key"):
            if defn.family == "flux2":
                return _sample_flux2_edit(
                    defn, comps, params, cache, model_config, guidance, progress.update
                )
            if defn.family == "qwen_edit":
                return _sample_qwen_edit(
                    defn, comps, params, cache, model_config, guidance, progress.update
                )
            if defn.family == "qwen_image_21":
                return _sample_qwen_image_21(
                    defn, comps, params, cache, guidance, progress.update
                )
            raise NotImplementedError(
                f"{defn.family} 暂不支持参考图编辑（目前只有 flux2 / qwen_edit / "
                "qwen_image_21 的 edit 路径）"
            )
        if defn.family == "flux2":
            return _sample_flux2(
                defn, comps, params, cache, model_config, guidance, progress.update
            )
        if defn.family == "qwen_image":
            return _sample_qwen_image(
                defn, comps, params, cache, model_config, guidance, progress.update
            )
        if defn.family == "qwen_image_21":
            return _sample_qwen_image_21(
                defn, comps, params, cache, guidance, progress.update
            )
        if defn.family == "ideogram4":
            return _sample_ideogram4(defn, comps, params, cache, progress.update)
        return _sample_z_image(
            defn, comps, params, cache, model_config, guidance, progress.update
        )

    key = runtime.cache_key(
        {
            "kind": "noise",
            "params": params,
            "config": "qwen-image-2.1" if model_config is None else model_config.model_name,
        }
    )
    latents, hit = cache.get_or_create("component_weights", key, sample)
    if hit:
        progress.complete()
    # 句柄里只存「大类 + 权重集名」，供下游按同一路径继续解析
    return MlxLatentHandle(
        kind="noise",
        shape=tuple(latents.shape),
        dtype=str(latents.dtype),
        cache_key=key,
        model=model_handle.model_type,
        source="local",
        path=model_handle.model_path,
        precision=model_handle.precision,
        quantize=model_handle.quantize,
        model_cache_key=runtime.cache_key(model_handle),
        height=params["height"],
        width=params["width"],
    )


def decode_latents(defn, comps, latents, height, width):
    """解码 latent（按大类分派）：

    - z_image：`unpack_latents([16,1,h/8,w/8])` → `[1,16,h/8,w/8]`，走 `VAEUtil.decode`；
    - flux2：latent 是打包形式 `[seq, C]`（或 `[1, seq, C]`），先补 batch 维再用
      `Flux2LatentCreator.unpack_latents` 还原成 `[1, C, latent_h, latent_w]`，
      最后交给 `Flux2VAE.decode_packed_latents`（内部含 bn 反归一化与 unpatchify）；
    - qwen_edit：同样先 unpack 成 `[1, 16, h/8, w/8]` 再走 `VAEUtil.decode`
      （QwenVAE 内部处理 mean/std 与 5D 维度）；两条 Qwen 链路的 latent 都是
      打包形式，因此 unpack 分支共用；
    """
    if not defn.supported:
        raise NotImplementedError(f"{defn.family} 尚未实现：{defn.notes}")
    latent_creator = runtime.import_object(defn.latent_creator)
    vae = comps["vae"]
    if defn.family == "ideogram4":
        if latents.ndim == 2:
            latents = latents[None, ...]
        unpacked = latent_creator.unpack_latents(latents, height, width)
        return vae.decode(unpacked)
    if defn.family == "qwen_image_21":
        unpacked = qwen21_sampling.unpack_latents(latents, height, width)
        return vae.decode(unpacked)
    if defn.family == "flux2":
        if latents.ndim == 2:
            latents = latents[None, ...]  # [seq, C] → [1, seq, C]
        unpacked = latent_creator.unpack_latents(latents, height, width)
        return vae.decode_packed_latents(unpacked)
    if defn.family in ("qwen_edit", "qwen_image") and latents.ndim == 2:
        latents = latents[None, ...]  # [seq, C] → [1, seq, C]
    unpacked = latent_creator.unpack_latents(latents, height, width)
    vae_util = runtime.import_object("mflux.models.common.vae.vae_util:VAEUtil")
    return vae_util.decode(vae=vae, latent=unpacked, tiling_config=None)


# === MiniMax-H3（文生视频 + 立体声）================================================
# H3 绕开 mflux 的两条基础设施：① 构造参数不查 ModelConfig 注册表（0.19.1 里
# 没有 minimax-h3），从组件目录的 config.json 现读；② 权重不「一次读全量再
# 量化」，按 shard 流式加载（否则 128 GB 机器上光加载就 OOM）。因此全部走
# h3/weights/loader.py，缓存也另起桶，不与图片链路互相挤占。

H3_MODULE_BUCKET = "h3_module"
H3_PROMPT_BUCKET = "h3_prompt"
H3_LATENT_BUCKET = "h3_latents"
# 视觉条件（首 / 尾帧锚点 + 多图参考）的有序 PIL 元组；键 = MlxH3VisualCondition.images_key，
# 值 = tuple[PIL.Image, ...]，顺序就是 presentation 里 <Picture i> 的编号顺序
H3_VISUAL_BUCKET = "h3_keyframe_source"


def _ensure_image_family(entry, node: str) -> None:
    """图片侧入口的媒体守卫：H3 / YuE2 在这里被明确挡住。

    句柄类型名是通用的（model / CLIP / mlx_vae），连线期挡不住错接，所以放在
    每个图片侧入口的最前面 —— 必须早于 `weights.config_for_path`，否则只会拿到
    一份对不上 H3 权重的图片配置，报错也很难懂。
    """
    if entry.media != "image":
        raise ValueError(
            f"{entry.family} 是视频 / 音频家族，不能在「{node}」里使用；"
            "它的条件编码 / 采样 / 解码走 MlxTextEncoder / MlxKSamplerMLX / "
            "MlxVAEDecodeRawPIL 的专用媒体分支"
        )


# === YuE2（风格 + 歌词 → 48 kHz 立体声音乐）===================================
YUE2_MODULE_BUCKET = "yue2_module"


def _raw_component_selection(selection: str, role: str) -> Path:
    """按 paths.resolve 的查找次序返回未 resolve 的路径，以保留第一层软链接目标。"""
    selected = Path(str(selection)).expanduser()
    if selected.is_absolute():
        candidates = [selected]
    elif "/" in str(selection):
        candidates = [paths.MODEL_ROOT / selected, paths.component_dir(role) / selected]
    else:
        candidates = [paths.component_dir(role) / selected, paths.MODEL_ROOT / selected]
    for candidate in candidates:
        if candidate.exists() or candidate.is_symlink():
            return candidate
    return candidates[0]


def yue2_variant_dir(selection: str, role: str) -> Path:
    """把 YuE2 目录或单文件软链接还原成包含配置的精度变体目录。

    推荐直接把 snapshot 的 ``4bit/`` / ``8bit/`` / ``bf16/`` 目录软链接到对应组件
    目录；也兼容 ``transformer/foo.safetensors -> .../8bit/model.safetensors`` 这种
    单文件软链接。不能直接使用通用 ``paths.resolve``，因为它会继续解析到 HF blob，
    从而丢失同目录的 config、tokenizer 与生成配置。
    """
    raw = _raw_component_selection(selection, role)
    expected = "model.safetensors" if role == "transformer" else "vae.safetensors"
    required = (
        ("config.json", "qwen.tiktoken", "yue2_generation_config.json", expected)
        if role == "transformer"
        else ("vae_config.json", expected)
    )

    candidates: list[Path] = []
    if raw.is_dir():
        candidates.append(raw)
    else:
        # 直接选择 snapshot 内的 HF 文件时，先试它的逻辑父目录，不跟随文件软链接。
        candidates.append(raw.parent)
    if raw.is_symlink():
        target = raw.readlink()
        if not target.is_absolute():
            target = raw.parent / target
        candidates.append(target if target.is_dir() else target.parent)

    checked: list[str] = []
    for candidate in candidates:
        directory = candidate.resolve()
        missing = [name for name in required if not (directory / name).is_file()]
        checked.append(f"{directory}（缺 {', '.join(missing) or '无'}）")
        if not missing:
            return directory
    raise FileNotFoundError(
        f"无法从 {role}/{selection} 找到完整 YuE2 变体目录；检查过：{'；'.join(checked)}。"
        "请把完整 4bit/、8bit/ 或 bf16/ 目录软链接进组件目录，或让单文件软链接"
        "直接指向该目录中的 model.safetensors / vae.safetensors。"
    )


def yue2_component_cache_key(role: str, directory: Path) -> str:
    return runtime.cache_key(
        {"kind": "yue2_module", "role": role, "directory": str(directory)}
    )


def prepare_yue2_sampler_components(entry, model_handle, cache) -> dict[str, Any]:
    """加载 YuE2 主模型、固定 tokenizer 与生成配置；采样阶段不加载 VAE。"""
    if entry.family != "yue2":
        raise ValueError(f"prepare_yue2_sampler_components 收到非 YuE2 大类：{entry.family}")
    directory = yue2_variant_dir(model_handle.model_path, "transformer")
    key = yue2_component_cache_key("transformer", directory)

    def build():
        print(f"[YuE2 组件] 从 {directory} 加载主模型与 tokenizer")
        return {
            "transformer": load_yue2_model(directory),
            "tokenizer": yue2_pipeline.Tokenizer(directory / "qwen.tiktoken"),
            "generation_config": yue2_pipeline.load_generation_config(directory),
            "directory": directory,
        }

    bundle, _hit = cache.get_or_create(YUE2_MODULE_BUCKET, key, build)
    return bundle


def release_yue2_sampler_components(entry, model_handle, cache) -> None:
    """采样完成即释放 3B 主模型；latent 留在独立缓存中供 VAE 节点解码。"""
    directory = yue2_variant_dir(model_handle.model_path, "transformer")
    key = yue2_component_cache_key("transformer", directory)
    if cache.evict(YUE2_MODULE_BUCKET, key):
        print("[YuE2 组件] 已释放主模型（下次采样时重新懒加载）")


def yue2_latent_cache_key(model_handle, params: dict[str, Any]) -> str:
    return runtime.cache_key(
        {"kind": "yue2_latents", "params": params, "model": model_handle}
    )


def has_yue2_latents(model_handle, params: dict[str, Any], cache) -> bool:
    """在加载主模型前探测相同参数的声学 latent 是否还在缓存。"""
    _latents, hit = cache.get("component_weights", yue2_latent_cache_key(model_handle, params))
    return hit


def run_yue2_sampler(entry, model_handle, comps, params: dict[str, Any], cache) -> MlxLatentHandle:
    """ABC 规划 → 语义 codec AR → NAR 声学 latent。

    ``comps`` 在缓存命中时允许为 ``None``；factory 不会执行，因此无需重新加载主模型。
    """
    key = yue2_latent_cache_key(model_handle, params)
    progress = SamplingProgress(int(params["steps"]))

    def sample():
        if comps is None:
            raise RuntimeError("YuE2 latent 缓存已失效，请重新执行采样器")
        latents, _info = yue2_pipeline.generate_music_latents(
            comps["transformer"],
            comps["tokenizer"],
            comps["generation_config"],
            params["style"],
            params["lyrics"],
            cot=params["cot"],
            seed=int(params["seed"]),
            steps=int(params["steps"]),
            max_tokens=int(params["max_tokens"]),
            cfg_scale=float(params["guidance"]),
            log=print,
            on_progress=progress.update_absolute,
        )
        return latents

    latents, hit = cache.get_or_create("component_weights", key, sample)
    if hit:
        progress.complete()
    duration = yue2_pipeline.audio_samples(latents.shape[0]) / yue2_pipeline.SAMPLE_RATE
    return MlxLatentHandle(
        kind="yue2_audio",
        shape=tuple(latents.shape),
        dtype=str(latents.dtype),
        cache_key=key,
        model=model_handle.model_type,
        source="local",
        path=model_handle.model_path,
        precision=model_handle.precision,
        quantize=model_handle.quantize,
        model_cache_key=runtime.cache_key(model_handle),
        duration=float(duration),
        audio_num_rows=int(latents.shape[0]),
        prompt_digest=runtime.cache_key(
            {"style": params["style"], "lyrics": params["lyrics"], "cot": params["cot"]}
        ),
    )


def prepare_yue2_vae(handle, cache):
    """按 YuE2 VAE handle 延迟物化 Oobleck decoder。"""
    if handle.role != "vae":
        raise ValueError(f"YuE2 只有 role='vae'，收到 {handle.role!r}")
    directory = yue2_variant_dir(handle.path, "vae")
    key = yue2_component_cache_key("vae", directory)
    vae, _hit = cache.get_or_create(
        YUE2_MODULE_BUCKET,
        key,
        lambda: load_yue2_vae(directory),
    )
    return vae


def release_yue2_vae(handle, cache) -> None:
    directory = yue2_variant_dir(handle.path, "vae")
    key = yue2_component_cache_key("vae", directory)
    if cache.evict(YUE2_MODULE_BUCKET, key):
        print("[YuE2 组件] 已释放 VAE（下次解码时重新懒加载）")


def decode_yue2_latents(latents, vae_handle, cache, batch_index: int) -> MlxPilImage:
    """声学 latent → AudioTrack；交给 MlxPilToTorch 包成 ComfyUI 原生 AUDIO。"""
    import numpy as np

    from .h3.video import AudioTrack

    if batch_index not in (-1, 0):
        print("[YuE2 VAE] 一次只有一首音乐，batch_index 已按 0 处理")
    vae = prepare_yue2_vae(vae_handle, cache)
    waveform = mx.clip(vae.decode_tiled(latents), -1, 1)
    mx.eval(waveform)
    # Oobleck 输出 [samples, 2]；AudioTrack / ComfyUI AUDIO 使用 [channels, samples]。
    stereo = np.ascontiguousarray(np.array(waveform, dtype=np.float32).T)
    track = AudioTrack(waveform=stereo, sample_rate=yue2_pipeline.SAMPLE_RATE)
    return MlxPilImage(images=(), batch_index=-1, audio=track)


# === Breeze-TTS-2（完整 checkpoint → 24 kHz 单声道语音）========================
BREEZE_MODULE_BUCKET = "breeze_module"
BREEZE_WAVEFORM_BUCKET = "breeze_waveform"


def breeze_model_dir(selection: str) -> Path:
    """解析 transformer/ 下的完整 Breeze checkpoint，并验证格式。"""
    _kind, resolved = paths.resolve("local", selection, "transformer")
    directory = Path(resolved)
    if directory.is_file():
        directory = directory.parent
    directory = directory.resolve()
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Breeze-TTS-2 需要完整模型目录，{directory} 缺少 config.json"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 Breeze 配置 {config_path}：{exc}") from exc
    if config.get("model_type") != "breeze_tts":
        raise ValueError(
            f"{directory} 不是 Breeze-TTS-2 checkpoint："
            f"model_type={config.get('model_type')!r}（应为 'breeze_tts'）"
        )
    return directory


def breeze_model_cache_key(model_handle) -> str:
    directory = breeze_model_dir(model_handle.model_path)
    return runtime.cache_key({"kind": "breeze_model", "directory": str(directory)})


def prepare_breeze_model(entry, model_handle, cache):
    """用 mlx-audio 一次装配主干、文本编码器和 audio tokenizer。"""
    if entry.family != "breeze_tts2":
        raise ValueError(f"prepare_breeze_model 收到非 Breeze 大类：{entry.family}")
    directory = breeze_model_dir(model_handle.model_path)
    key = breeze_model_cache_key(model_handle)

    def build():
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            requirements = Path(__file__).resolve().parents[2] / "requirements.txt"
            raise RuntimeError(
                "Breeze-TTS-2 需要 mlx-audio==0.5.1；请在 ComfyUI 使用的 Python 环境中运行 "
                f"pip install -r {requirements}"
            ) from exc
        print(f"[Breeze-TTS-2] 从 {directory} 加载完整 checkpoint")
        return load_model(directory)

    model, _hit = cache.get_or_create(BREEZE_MODULE_BUCKET, key, build)
    return model


def release_breeze_model(model_handle, cache) -> None:
    key = breeze_model_cache_key(model_handle)
    if cache.evict(BREEZE_MODULE_BUCKET, key):
        print("[Breeze-TTS-2] 已释放完整模型（最终波形继续保留供解码）")


def breeze_waveform_cache_key(model_handle, params: dict[str, Any]) -> str:
    """只对规范化音频内容取 digest，绝不把随机临时文件名写入缓存键。"""
    cache_params = {key: value for key, value in params.items() if key != "reference"}
    reference = params.get("reference")
    if reference is not None:
        _mono, sample_rate, digest = reference
        cache_params["reference"] = {"sample_rate": sample_rate, "digest": digest}
    return runtime.cache_key(
        {"kind": "breeze_waveform", "model": model_handle, "params": cache_params}
    )


def has_breeze_waveform(model_handle, params: dict[str, Any], cache) -> bool:
    _waveform, hit = cache.get(
        BREEZE_WAVEFORM_BUCKET, breeze_waveform_cache_key(model_handle, params)
    )
    return hit


def run_breeze_sampler(entry, model_handle, model, params: dict[str, Any], cache) -> MlxLatentHandle:
    """生成或复用最终语音波形，并返回现有 latent 工作流可传递的轻量句柄。"""
    if entry.family != "breeze_tts2":
        raise ValueError(f"run_breeze_sampler 收到非 Breeze 大类：{entry.family}")
    breeze.validate_params(params)
    key = breeze_waveform_cache_key(model_handle, params)
    progress = SamplingProgress(int(params["max_tokens"]))

    def sample():
        if model is None:
            raise RuntimeError("Breeze waveform 缓存已失效，请重新执行采样器")
        waveform, sample_rate = breeze.generate_waveform(
            model, params, on_progress=progress.update_absolute
        )
        if int(sample_rate) <= 0:
            raise RuntimeError(f"Breeze 返回了无效采样率：{sample_rate}")
        array = mx.asarray(waveform, dtype=mx.float32).reshape(-1)
        array = mx.clip(array, -1, 1)
        mx.eval(array)
        return array, int(sample_rate)

    cached, hit = cache.get_or_create(BREEZE_WAVEFORM_BUCKET, key, sample)
    if hit:
        progress.complete()
    waveform, sample_rate = cached
    samples = int(waveform.shape[0])
    return MlxLatentHandle(
        kind="breeze_audio",
        shape=tuple(waveform.shape),
        dtype=str(waveform.dtype),
        cache_key=key,
        model=model_handle.model_type,
        source="local",
        path=model_handle.model_path,
        precision=model_handle.precision,
        quantize=model_handle.quantize,
        model_cache_key=breeze_model_cache_key(model_handle),
        duration=samples / sample_rate,
        audio_num_rows=samples,
        sample_rate=sample_rate,
        prompt_digest=runtime.cache_key(
            {"text": params["text"], "mode": params["mode"], "speaker": params["speaker"]}
        ),
    )


def decode_breeze_waveform(latents, batch_index: int) -> MlxPilImage:
    """最终 waveform → AudioTrack；Breeze 不需要、也不会加载第二个 VAE。"""
    import numpy as np

    from .h3.video import AudioTrack

    if batch_index not in (-1, 0):
        print("[Breeze-TTS-2] 一次只有一条语音，batch_index 已按 0 处理")
    waveform, sample_rate = latents
    mono = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32).reshape(1, -1))
    track = AudioTrack(waveform=mono, sample_rate=int(sample_rate))
    return MlxPilImage(images=(), batch_index=-1, audio=track)


def prepare_h3_components(
    entry,
    roles: tuple[str, ...],
    cache,
    selections: dict[str, str],
    quantize: int | None = None,
    precision: str = "bfloat16",
    loras=(),
) -> dict[str, Any]:
    """只装 roles 里列出的组件；同一「role + 目录 + 精度 + 量化档位 + LoRA」复用同一份。

    每个 role 单独一条缓存（都在 h3_module 桶里，上限 5）：MlxTextEncoder 只要
    text_encoder + tokenizer，MlxKSamplerMLX 只要 transformer，解码节点要 VAE。
    """

    def build(role: str, kind: str, path: str) -> Any:
        if role == "tokenizer":
            if kind == "missing":
                raise FileNotFoundError(
                    f"没有 {role}：{path}"
                    "（请在 <模型根>/tokenizer/ 下补一个与权重集同名的目录或软链）"
                )
            return h3_prompt.load_tokenizer(path)
        if kind == "missing":
            raise FileNotFoundError(f"未找到 {role} 权重：{path}")
        comp = h3_loader.load(role, kind, path, quantize, precision)
        if role == "transformer":
            transformer_lora.apply_transformer_loras(
                entry.family,
                comp.module,
                loras,
                role=role,
            )
        bits = f"q{comp.bits}" if comp.bits else "不量化"
        print(
            f"[H3 组件] {role}: 常驻参数约 {comp.parameter_bytes() / 1e9:.1f} GB"
            f"（{comp.dtype}, {bits}）"
        )
        return comp

    comps: dict[str, Any] = {}
    for role in roles:
        selection = selections.get(role, "")
        kind, resolved = component_path(entry, role, selection)
        if role == "audio_vae" and kind == "missing":
            # 音频 VAE 既可以放 audio_vae/，也可以继续放 vae/（本仓库的软链就在 vae/）
            kind, resolved = paths.resolve("local", selection, "vae")
        key = h3_component_cache_key(
            entry,
            role,
            resolved,
            quantize,
            precision,
            loras if role == "transformer" else (),
        )
        comps[role], _hit = cache.get_or_create(
            H3_MODULE_BUCKET, key, lambda role=role, kind=kind, path=resolved: build(role, kind, path)
        )
    return comps


def prepare_h3_encoder(entry, clip, cache, cache_key: str) -> dict[str, Any]:
    """加载 H3 的条件编码器 + tokenizer（都在 handle 里的那个权重集名下解析）。"""
    selections = {role: clip.path for role in ENCODER_ROLES}
    return prepare_h3_components(
        entry,
        ENCODER_ROLES,
        cache,
        selections,
        quantize=clip.quantize,
        precision=clip.precision,
    )


def prepare_h3_sampler_components(entry, model_handle, cache) -> dict[str, Any]:
    """只物化 transformer（H3 的采样阶段不需要 VAE，也不需要用条件编码器）。"""
    return prepare_h3_components(
        entry,
        SAMPLER_ROLES,
        cache,
        {"transformer": model_handle.model_path},
        quantize=int(model_handle.quantize) or None,
        precision=model_handle.precision,
        loras=model_handle.loras,
    )


def prepare_h3_vae(handle, cache) -> Any:
    """按 MlxVaeHandle 物化某个 VAE（role = "vae" 视频 / "audio_vae" 音频）。"""
    entry = entry_for(handle.model_type)
    comps = prepare_h3_components(
        entry,
        (handle.role,),
        cache,
        {handle.role: handle.path},
        quantize=int(handle.quantize) or None,
        precision=handle.precision,
    )
    # 同上：解码用的 decode_video / decode_audio 直接调 vae.decode() 与
    # vae.latent_channels，因此交出去的是包装器里的模块而不是 LoadedH3Component
    return comps[handle.role].module


def h3_component_dir(entry, role: str, selection: str) -> str:
    """role 实际解析到的目录（audio_vae 缺失时回落到 vae/，与 prepare_h3_components 一致）。"""
    kind, path = component_path(entry, role, selection)
    if kind == "missing" and role == "audio_vae":
        _kind, path = paths.resolve("local", selection, "vae")
    return path


def h3_component_cache_key(entry, role: str, path: str, quantize, precision, loras=()) -> str:
    """组件在 h3_module 桶里的键：prepare / release 两边共用，保证键完全对得上。"""
    return runtime.cache_key(
        {
            "kind": "h3_module",
            "role": role,
            "path": path,
            "quantize": quantize,
            "precision": precision,
            "loras": loras if role == "transformer" else (),
        }
    )


def release_h3_components(
    entry,
    roles: tuple[str, ...],
    cache,
    selections: dict[str, str],
    quantize: int | None = None,
    precision: str = "bfloat16",
    loras=(),
) -> None:
    """「用完即释放」：把这一段刚用过的组件从 h3_module 桶里丢掉，下次需要时再懒加载。

    键与 prepare_h3_components 共用 h3_component_cache_key，所以丢掉的就是那一段
    刚加载的那份；键不在桶里（没加载过 / 已换配置）就静默跳过。
    """
    for role in roles:
        path = h3_component_dir(entry, role, selections.get(role, ""))
        key = h3_component_cache_key(
            entry,
            role,
            path,
            quantize,
            precision,
            loras if role == "transformer" else (),
        )
        if cache.evict(H3_MODULE_BUCKET, key):
            print(f"[H3 组件] 已释放 {role}（下次需要时重新懒加载）")


def release_h3_encoder(entry, clip, cache) -> None:
    """编码用完就丢条件编码器 + tokenizer（q8 约 30 GB，采样阶段用不到）。"""
    release_h3_components(
        entry,
        ENCODER_ROLES,
        cache,
        {role: clip.path for role in ENCODER_ROLES},
        quantize=clip.quantize,
        precision=clip.precision,
    )


def release_h3_sampler_components(entry, model_handle, cache) -> None:
    """采样用完就丢 transformer（q8 约 35 GB，解码阶段用不到）。"""
    release_h3_components(
        entry,
        SAMPLER_ROLES,
        cache,
        {"transformer": model_handle.model_path},
        quantize=int(model_handle.quantize) or None,
        precision=model_handle.precision,
        loras=model_handle.loras,
    )


def release_h3_vae(handle, cache) -> None:
    """解码用完就丢这个 VAE（视频约 5.2 GB / 音频约 0.6 GB，与 prepare_h3_vae 对称）。"""
    entry = entry_for(handle.model_type)
    release_h3_components(
        entry,
        (handle.role,),
        cache,
        {handle.role: handle.path},
        quantize=int(handle.quantize) or None,
        precision=handle.precision,
    )


# --- H3 提示词（MlxTextEncoder 负责；数组只进 h3_prompt 桶）---------------------
def compose_h3_prompt(text: str) -> str:
    """把提示词组装成 H3 的三段式；已经自带标签的提示词原样透传。"""
    return h3_prompt.compose_prompt(text)


def h3_prompt_encoding_key(
    clip, prompt: str, visual: MlxH3VisualCondition | None = None
) -> str:
    """H3 条件的编码键（换大类 / 换目录 / 换精度 / 换量化 / 换文本都会换键）。"""
    return runtime.cache_key(
        {
            "kind": "h3_prompt",
            "clip": clip,
            "text": prompt,
            "visual": visual.digest if visual is not None else "",
        }
    )


def encode_h3_prompt(
    entry,
    comps,
    prompt: str,
    cache,
    cache_key: str,
    visual: MlxH3VisualCondition | None = None,
) -> tuple[Any, Any]:
    """编码 presentation → `(embeds (1,L,hidden), tags (L,) int32)`（存进缓存）。

    接了视觉条件时，桶里那 N 张图会按顺序编成 ``<Picture 1>`` … ``<Picture N>``
    （``h3/prompt.py`` 的 ``encode_presentation`` 已支持任意张数；上游只用过 1 张）。

    编码器前向用 `exact_fp32()` 包住（H3 的输入/输出头与时间步 MLP 是 fp32，见
    `h3/model/h3_precision.py`）：TF32 只在 H3 计算期间关，出图链路仍走快路径。
    """

    def build() -> tuple[Any, Any]:
        images: tuple[Any, ...] = ()
        motion_frames: tuple[Any, ...] = ()
        motion_timestamps: tuple[float, ...] = ()
        if visual is not None:
            if visual.source in {"motion_reference", "motion_reference_with_picture"}:
                source, hit = cache.get("h3_motion_source", visual.motion_key)
                if not hit or source is None:
                    raise RuntimeError("H3 动作参考缓存已失效，请重新运行「MLX H3 动作参考条件」节点")
                motion_frames = tuple(source["presentation_frames"])
                motion_timestamps = tuple(float(v) for v in source["timestamps"])
            if visual.source not in {"motion_reference"}:
                found, hit = cache.get(H3_VISUAL_BUCKET, visual.images_key)
                if not hit or found is None:
                    raise RuntimeError(
                        "H3 的参考图缓存已失效，请重新运行「MLX H3 关键帧 / 视觉条件」节点"
                    )
                images = tuple(found)
                if len(images) != int(visual.picture_count):
                    raise RuntimeError(
                        "H3 的参考图数量与 handle 不一致，请重新运行「MLX H3 关键帧 / 视觉条件」节点"
                    )
        encode = runtime.import_object(entry.prompt_encoder)
        # 保留 entry.prompt_encoder 的可注入契约（测试 / 第三方扩展会替换它）。
        # 旧的四参数 encoder 继续用于图片 / 纯文本条件；只有动作参考才使用
        # 新增的两个可选参数。
        with h3_exact_fp32():
            if motion_frames:
                return encode(
                    comps["text_encoder"].module,
                    comps["tokenizer"],
                    prompt,
                    images,
                    motion_frames=motion_frames,
                    motion_timestamps=motion_timestamps,
                )
            return encode(comps["text_encoder"].module, comps["tokenizer"], prompt, images)
    return cache.get_or_create(H3_PROMPT_BUCKET, cache_key, build)[0]


def cached_h3_prompt(cache, key: str, label: str = "H3") -> tuple[Any, Any]:
    """取 H3 的 presentation 编码；不在缓存里就提示重跑「MLX 文本编码器」。"""
    if not key:
        raise RuntimeError(f"未连接{label} 条件（MlxTextEncoder 的输出）")
    encoding, hit = cache.get(H3_PROMPT_BUCKET, key)
    if not hit or encoding is None:
        raise RuntimeError(f"{label} 条件的编码已失效，请重新运行「MLX 文本编码器」节点")
    return encoding


# --- H3 联合采样（MlxKSamplerMLX 负责；latent 行只进 h3_latents 桶）------------
def _mlx_memory_bytes() -> int:
    """MLX 当前占用（活跃 + 缓存）字节数；取不到就算 0（不让预检把能跑的任务挡掉）。"""
    total = 0
    for name in ("get_active_memory", "get_cache_memory"):
        fn = getattr(mx, name, None)
        if fn is None:
            continue
        try:
            total += int(fn())
        except Exception:  # noqa: BLE001
            pass
    return total


def h3_latent_cache_key(params: dict[str, Any], model_handle) -> str:
    """H3 latent 的缓存键（几何 / 调度 / 提示词任一项变化都会换键）。"""
    return runtime.cache_key({"kind": "h3_latents", "params": params, "model": model_handle})


def has_h3_latents(model_handle, params: dict[str, Any], cache) -> bool:
    """在加载 Video VAE / transformer 前探测相同 H3 latent 是否仍在缓存。"""
    _latents, hit = cache.get(H3_LATENT_BUCKET, h3_latent_cache_key(params, model_handle))
    return bool(hit)


def encode_h3_keyframes(
    visual: MlxH3VisualCondition, cache
) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    """按锚点顺序编码 latent 锚点，返回 `(锚点名, 对应的单帧 latent)`。

    纯参考（``anchors`` 为空）时直接返回空 —— 那些图只进 Qwen3-VL 的
    presentation，不占 latent 行，因此也不用物化 Video VAE。
    """
    if not visual.anchors:
        return (), ()
    images, hit = cache.get(H3_VISUAL_BUCKET, visual.images_key)
    if not hit or images is None:
        raise RuntimeError(
            "H3 的参考图缓存已失效，请重新运行「MLX H3 关键帧 / 视觉条件」节点"
        )
    if len(images) < int(visual.picture_count):
        raise RuntimeError(
            "H3 的参考图比 handle 里记的少（缓存被挤掉了？），"
            "请重新运行「MLX H3 关键帧 / 视觉条件」节点"
        )
    try:
        vae = prepare_h3_vae(visual.vae, cache)
        with h3_exact_fp32():
            latents = tuple(
                h3_pipeline.encode_keyframe_latents((images[int(index)],), vae)[0]
                for index in visual.anchor_images
            )
    finally:
        release_h3_vae(visual.vae, cache)
    return tuple(visual.anchors), latents


def encode_h3_motion_reference(
    visual: MlxH3VisualCondition, cache
) -> tuple[Any, ...]:
    """物化完整动作参考视频的 normalized Video VAE latent。"""
    if visual.source not in {"motion_reference", "motion_reference_with_picture"}:
        return ()
    source, hit = cache.get("h3_motion_source", visual.motion_key)
    if not hit or source is None:
        raise RuntimeError("H3 动作参考缓存已失效，请重新运行「MLX H3 动作参考条件」节点")
    try:
        vae = prepare_h3_vae(visual.vae, cache)
        with h3_exact_fp32():
            latent = h3_pipeline.encode_motion_reference_latent(tuple(source["frames"]), vae)
        return (latent,)
    finally:
        release_h3_vae(visual.vae, cache)


def run_h3_sampler(
    entry,
    model_handle,
    comps,
    params: dict[str, Any],
    cache,
    keyframe_latents: tuple[Any, ...] = (),
    reference_latents: tuple[Any, ...] = (),
) -> MlxLatentHandle:
    """规划 → 内存预检 → 联合去噪 → 把 `(视频行, 音频行, 计划)` 存进 h3_latents 桶。"""
    plan = h3_pipeline.make_plan(
        int(params["width"]),
        int(params["height"]),
        int(params["num_frames"]),
        int(params["steps"]),
        float(params["video_shift"]),
        float(params["audio_shift"]),
        log=print,
    )
    print(f"[H3 采样] {plan.summary()}")
    progress = SamplingProgress(plan.steps)

    def sample() -> tuple[Any, Any, Any]:
        embeds, tags = cached_h3_prompt(cache, params["positive_encoding_key"])
        loaded = comps["transformer"]
        # 预检按「已物化组件 + MLX 缓存 + 每行标定值」估（参考机实测口径，见 h3/pipeline）
        h3_pipeline.preflight(
            plan,
            int(loaded.parameter_bytes()),
            int(runtime.system_total_memory()),
            text_tokens=int(tags.shape[0]),
            cache_bytes=_mlx_memory_bytes(),
            keyframe_count=len(params.get("keyframe_anchors", ())),
            reference_rows=sum(
                int(latent.shape[2]) * (int(latent.shape[3]) // 2) * (int(latent.shape[4]) // 2)
                for latent in reference_latents
            ),
            log=print,
        )
        with h3_exact_fp32():
            video_rows, audio_rows, _layout = h3_pipeline.sample(
                loaded.module,
                embeds,
                tags,
                plan,
                int(params["seed"]),
                keyframe_latents=keyframe_latents,
                keyframe_anchors=tuple(params.get("keyframe_anchors", ())),
                reference_latents=reference_latents,
                log=print,
                on_progress=progress.update_absolute,
            )
        return video_rows, audio_rows, plan

    key = h3_latent_cache_key(params, model_handle)
    (video_rows, audio_rows, plan), hit = cache.get_or_create(H3_LATENT_BUCKET, key, sample)
    if hit:
        progress.complete()
    return MlxLatentHandle(
        kind="h3_video",
        shape=tuple(video_rows.shape),
        dtype=str(video_rows.dtype),
        cache_key=key,
        model=model_handle.model_type,
        source="local",
        path=model_handle.model_path,
        precision=model_handle.precision,
        quantize=model_handle.quantize,
        model_cache_key=runtime.cache_key(model_handle),
        height=plan.height,
        width=plan.width,
        num_frames=plan.num_frames,
        duration=plan.duration_seconds,
        video_shift=plan.video_shift,
        audio_shift=plan.audio_shift,
        num_latent_frames=plan.num_latent_frames,
        latent_height=plan.latent_height,
        latent_width=plan.latent_width,
        audio_num_rows=int(audio_rows.shape[0]),
        prompt_digest=str(params.get("prompt_digest", "")),
    )


# --- H3 解码（MlxVAEDecodeRawPIL / MlxVAEDecoder 负责）-------------------------
def h3_state(cache, key: str) -> tuple[Any, Any, Any]:
    """取「视频行 + 音频行 + 计划」；不在缓存里就提示重跑采样器。"""
    state, hit = cache.get(H3_LATENT_BUCKET, key)
    if not hit or state is None:
        raise RuntimeError("H3 的 latent 不在缓存里，请重新运行「MLX 采样器」")
    return state


def decode_h3_latents(latents, vae_handle, cache, batch_index: int) -> MlxPilImage:
    """按 VAE 的 role 解 H3 latent：vae → 帧序列，audio_vae → 音频轨。

    帧序列是「一段视频」而不是「一批图片」，所以 `batch_index` 只用来取其中一帧
    （-1 = 全部）；音频给整段 `AudioTrack`，由 MlxPilToTorch 转成 ComfyUI 的
    AUDIO，再接核心的 CreateVideo / SaveAudio 落盘。
    """
    video_rows, audio_rows, plan = h3_state(cache, latents.cache_key)
    vae = prepare_h3_vae(vae_handle, cache)
    # 解码也走 fp32（音频 VAE 的 7 级抗混叠上采样是 TF32 最敏感的地方），
    # 因此与编码 / 采样一样用 exact_fp32() 包住
    if vae_handle.role == "vae":
        with h3_exact_fp32():
            frames = h3_pipeline.decode_video(video_rows, plan, vae)
        pils = image.to_pil_uint8(frames, batch_index=batch_index)
        return MlxPilImage(images=pils, batch_index=batch_index, fps=float(plan.fps), audio=None)
    if vae_handle.role == "audio_vae":
        with h3_exact_fp32():
            track = h3_pipeline.decode_audio(audio_rows, plan, vae)
        return MlxPilImage(images=(), batch_index=batch_index, fps=float(plan.fps), audio=track)
    raise ValueError(f"未知 VAE role：{vae_handle.role}")

