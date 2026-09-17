"""生成实现（我们自己写采样循环，不调用 mflux 的 generate_image / generate_output）。

按「模型大类」（family）分派，借用 mflux 的数据工具（latent creator、prompt
encoder、scheduler、vae 等），采样循环与权重加载由我们自己实现：

- z_image：`create_noise` → 单数组条件 → `transformer(timestep, x, cap_feats, sigmas)`
  → `scheduler.step(noise, t, latents)` → `unpack_latents` + `VAEUtil.decode`；
- flux2：`prepare_packed_latents` → 条件为 `(embeds, ids)` → `transformer(hidden_states,
  encoder_hidden_states, timestep, img_ids, txt_ids)` → `scheduler.step(..., sigmas)`
  → `unpack_latents` + `Flux2VAE.decode_packed_latents`。

组件分工（延迟装配）：采样器只加载 transformer + vae，text_encoder +
tokenizer 由 MlxTextEncoder 节点自己加载并编码；采样器按 handle 里的缓存键
直接取编码结果，不再碰文本编码器。
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import components, paths, runtime, weights
from .types import MlxLatentHandle, MlxModelEntry

# 各节点负责的组件 role（采样器不加载文本编码相关组件，反之亦然）
SAMPLER_ROLES: tuple[str, ...] = ("transformer", "vae")
ENCODER_ROLES: tuple[str, ...] = ("text_encoder", "tokenizer")


def component_path(entry, role: str, selection: str) -> tuple[str, str]:
    """解析某 role 的路径（由调用方给出权重集目录名，不在配置里写死）。

    同一个「权重集名」在各组件目录下同名（如 z-image-turbo-8bit 同时存在于
    transformer/ 与 vae/ 等目录下），因此 handle 里只存目录名就够了。
    """
    comp = entry.components[role]
    if not selection:
        raise ValueError(f"没有为 {role} 指定权重目录（请在节点上重新选择）")
    return paths.resolve(comp.source, selection, role)


def prepare_components(
    entry,
    roles: tuple[str, ...],
    cache,
    cache_key: str,
    selections: dict[str, str],
    quantize: int | None = None,
    max_length: int | None = None,
) -> dict[str, Any]:
    """只加载 roles 里列出的组件（命中缓存则复用整组实例）。

    selections 是「role -> 权重集目录名」；目录不写在配置里，因此配置里
    不用为每个变体登记路径，新权重目录丢进磁盘就能用。
    """

    def build() -> dict[str, Any]:
        raw: dict[tuple, dict] = {}
        comps: dict[str, Any] = {}
        for role in roles:
            selection = selections.get(role, "")
            kind, path = component_path(entry, role, selection)
            if kind == "missing":
                raise FileNotFoundError(f"未找到 {role} 权重: {path}")
            if role == "tokenizer":
                comps[role] = components.load_tokenizer(entry, role, kind, path, max_length)
            else:
                # 构造组件需要的 class_kwargs（如 flux2 的 transformer/text_encoder
                # 必须按 ModelConfig 的 overrides 建，否则用默认构造参数会建出
                # 错误维度、加载权重对不上）；按当前 role 的目录名现算 config。
                model_config = weights.config_for_path(selection, entry.default_config)
                if role == "transformer":
                    class_kwargs = dict(model_config.transformer_overrides)
                elif role == "text_encoder":
                    class_kwargs = dict(model_config.text_encoder_overrides)
                else:  # vae：构造参数不依赖 overrides（Flux2VAE()/VAE() 无参）
                    class_kwargs = {}
                # quantize=None → 用权重自带的档位（QuantizationResolution.resolve）
                instance, _bits = components.create_and_load(
                    entry, role, kind, path, quantize, raw, class_kwargs=class_kwargs
                )
                comps[role] = instance
        return comps

    instance, _hit = cache.get_or_create("module", cache_key, build)
    return instance


def prepare_encoder(entry, clip, cache, cache_key: str) -> dict[str, Any]:
    """加载 text_encoder + tokenizer（由 MlxTextEncoder 负责，M1 不额外量化）。"""
    # 两个 role 都用 handle 里的那个权重集名（各组件目录下同名）
    selections = {role: clip.path for role in ENCODER_ROLES}
    return prepare_components(
        entry, ENCODER_ROLES, cache, cache_key, selections, max_length=clip.max_length
    )


def prepare_sampler_components(entry, model_handle, cache, cache_key: str) -> dict[str, Any]:
    """加载 transformer + vae（文本编码已在 MlxTextEncoder 里做完，这里不碰）。"""
    return prepare_components(
        entry,
        SAMPLER_ROLES,
        cache,
        cache_key,
        {role: model_handle.model_path for role in SAMPLER_ROLES},
        quantize=model_handle.quantize,
    )


def create_latents(defn, seed, height, width, batch_size) -> list[Any]:
    """每个 batch 的噪声 latent（按大类不同，形状与附带元数据不同）：

    - z_image：每项 `[16, 1, h/8, w/8]`，采样用 `transformer(timestep, x, cap_feats, sigmas)`；
    - flux2：每项 `(latents[1, seq, C], latent_ids[1, seq, 4], latent_h, latent_w)`，
      采样用打包 latent + grid ids，`Flux2LatentCreator` 没有 `create_noise`，
      只有 `prepare_packed_latents`。
    """
    latent_creator = runtime.import_object(defn.latent_creator)
    if defn.family == "flux2":
        return [
            latent_creator.prepare_packed_latents(
                seed=seed + i, height=height, width=width, batch_size=1
            )
            for i in range(batch_size)
        ]
    return [latent_creator.create_noise(seed + i, height, width) for i in range(batch_size)]


def prompt_encoding_key(clip, text: str) -> str:
    """某条提示词的编码缓存键（同一 handle + 同一文本 → 同一键，可命中缓存）。"""
    return runtime.cache_key({"kind": "prompt_encoding", "clip": clip, "text": text})


def encode_text(defn, comps, text: str, cache, cache_key: str) -> Any:
    """用已加载的 tokenizer + text_encoder 编码一条文本（M1 不支持 prompt_cache）。

    - z_image：`encode_prompt` 返回单个 `cap_feats` 数组；
    - flux2：`encode_prompt` 返回 `(prompt_embeds, text_ids)` 二元组，且需要
      `num_images_per_prompt / max_sequence_length / text_encoder_out_layers`
      （取自 entry.prompt_encoder_args，`cache` 这个键 M1 不支持，跳过）。
    缓存里就存返回值原样（数组或二元组），采样端按大类取用。
    """

    def build():
        prompt_encoder = runtime.import_object(defn.prompt_encoder)
        if defn.family == "flux2":
            args = {k: v for k, v in defn.prompt_encoder_args.items() if k != "cache"}
            return prompt_encoder.encode_prompt(
                prompt=text,
                tokenizer=comps["tokenizer"],
                text_encoder=comps["text_encoder"],
                **args,
            )
        return prompt_encoder.encode_prompt(
            prompt=text,
            tokenizer=comps["tokenizer"],
            text_encoder=comps["text_encoder"],
        )

    encoding, _hit = cache.get_or_create("prompt_encoding", cache_key, build)
    return encoding


def cached_encoding(cache, key: str, label: str) -> Any:
    """按缓存键取出 MlxTextEncoder 算好的编码；缺失就提示重新运行该节点。"""
    if not key:
        raise RuntimeError(f"未连接{label}向条件（MlxTextEncoder 的输出）")
    encoding, hit = cache.get("prompt_encoding", key)
    if not hit or encoding is None:
        raise RuntimeError(f"{label}向条件的编码已失效，请重新运行「MLX 文本编码器」节点")
    return encoding


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


def _predict(transformer, latents, timestep, sigmas, encodings, negative, guidance, use_compile):
    """单步预测（含 CFG；与 ZImage._predict 一致）。"""

    def predict(x, t, s, feats, neg_feats, scale):
        noise = transformer(timestep=t, x=x, cap_feats=feats, sigmas=s)
        if neg_feats is None:
            return noise
        negative = transformer(timestep=t, x=x, cap_feats=neg_feats, sigmas=s)
        return negative + scale * (noise - negative)

    fn = mx.compile(predict) if use_compile else predict
    return fn(latents, timestep, sigmas, encodings, negative, guidance)


def _predict_flux2(
    transformer, latents, latent_ids, timestep, encodings, negative, guidance, use_compile
):
    """Flux2 单步预测（含 CFG；与 Flux2Klein._predict 一致）。

    latents 是打包形式 `[B, seq, C]`，配套 `latent_ids`（grid ids `[B, seq, 4]`）；
    encodings / negative 都是 `(prompt_embeds, text_ids)` 二元组（negative 可为 None）。
    CFG 组合顺序与参考实现一致：`negative + guidance * (noise - negative)`。
    """
    pos_embeds, pos_ids = encodings
    if negative is not None:
        neg_embeds, neg_ids = negative
    else:
        neg_embeds, neg_ids = None, None

    def predict(x, img_ids, p_embeds, p_ids, n_embeds, n_ids, scale, t):
        noise = transformer(
            hidden_states=x,
            encoder_hidden_states=p_embeds,
            timestep=t,
            img_ids=img_ids,
            txt_ids=p_ids,
            guidance=None,
        )
        if n_embeds is None:
            return noise
        negative_noise = transformer(
            hidden_states=x,
            encoder_hidden_states=n_embeds,
            timestep=t,
            img_ids=img_ids,
            txt_ids=n_ids,
            guidance=None,
        )
        return negative_noise + scale * (noise - negative_noise)

    fn = mx.compile(predict) if use_compile else predict
    return fn(latents, latent_ids, pos_embeds, pos_ids, neg_embeds, neg_ids, guidance, timestep)


def _sample_z_image(defn, comps, params, cache, model_config, guidance):
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
    use_compile = bool(params.get("compile_model", True))
    transformer = comps["transformer"]
    final = []
    for latents in per_seed:
        current = latents
        for t in range(params["steps"]):
            sigma_t = scheduler.sigmas[t].reshape((1,))
            timestep = mx.ones_like(sigma_t) - sigma_t
            noise = _predict(
                transformer, current, timestep, scheduler.sigmas,
                encodings, negative, guidance, use_compile,
            )
            current = scheduler.step(noise=noise, timestep=t, latents=current)
            mx.eval(current)
        final.append(current)
    stacked = mx.stack(final, axis=0)  # [B, 16, 1, h/8, w/8]
    mx.eval(stacked)
    return stacked


def _sample_flux2(defn, comps, params, cache, model_config, guidance):
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
    use_compile = bool(params.get("compile_model", True))
    transformer = comps["transformer"]
    final = []
    for latents, latent_ids, _latent_h, _latent_w in per_seed:
        current = latents  # [1, seq, C]
        for t in range(params["steps"]):
            timestep = scheduler.timesteps[t]
            noise = _predict_flux2(
                transformer, current, latent_ids, timestep,
                encodings, negative, guidance, use_compile,
            )
            current = scheduler.step(
                noise=noise, timestep=t, latents=current, sigmas=scheduler.sigmas
            )
            mx.eval(current)
        final.append(current)
    stacked = mx.concatenate(final, axis=0)  # [B, seq, C]
    mx.eval(stacked)
    return stacked


def run_sampler(defn, model_handle, comps, params, cache):
    """跑完整采样，返回带缓存数组的 latent 句柄。

    正/负条件的编码由 MlxTextEncoder 提前存进 cache 的 "prompt_encoding" 桶，
    params 里只有编码键；按键取不到就报错（说明编码已换掉，需要重新运行）。

    配置按「权重集目录名」在 mflux 的 ModelConfig 注册表里现取（不依赖预先
    登记的模型名）：z-image-turbo-8bit / z-image-turbo-4bit 都会命中
    z-image-turbo，z-image-8bit 命中 z_image，因此新增权重目录不用改这里；
    步数 / 调度器 / guidance 由工作流的 widget 决定。
    """
    if not defn.supported:
        raise NotImplementedError(f"{defn.family} 尚未实现：{defn.notes}")
    model_config = weights.config_for_path(model_handle.model_path, defn.default_config)
    # 配置说这套模型没有 CFG（如 z-image-turbo）就强制关掉，与 mflux 行为一致
    guidance = float(params["guidance"]) if model_config.supports_guidance else 0.0

    def sample():
        if defn.family == "flux2":
            return _sample_flux2(defn, comps, params, cache, model_config, guidance)
        return _sample_z_image(defn, comps, params, cache, model_config, guidance)

    key = runtime.cache_key(
        {"kind": "noise", "params": params, "config": model_config.model_name}
    )
    latents, _hit = cache.get_or_create("component_weights", key, sample)
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
      最后交给 `Flux2VAE.decode_packed_latents`（内部含 bn 反归一化与 unpatchify）。
    """
    latent_creator = runtime.import_object(defn.latent_creator)
    vae = comps["vae"]
    if defn.family == "flux2":
        if latents.ndim == 2:
            latents = latents[None, ...]  # [seq, C] → [1, seq, C]
        unpacked = latent_creator.unpack_latents(latents, height, width)
        return vae.decode_packed_latents(unpacked)
    unpacked = latent_creator.unpack_latents(latents, height, width)
    vae_util = runtime.import_object("mflux.models.common.vae.vae_util:VAEUtil")
    return vae_util.decode(vae=vae, latent=unpacked, tiling_config=None)

