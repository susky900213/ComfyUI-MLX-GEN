"""纯数据契约：自定义类型、handle、模型清单（不 import mlx、不 import _impl）。

MODEL_DEFS 按「模型大类」（family）组织：键与 MlxTransformerLoader 的
model_type 下拉一一对应（"z_image" 覆盖 z-image-turbo-* 与 z-image-* 等
Z-Image 权重，"flux2" 覆盖 flux.2-klein-*）。这里**不登记具体模型名，也没有
「配置变体」可选**：

- 权重目录：节点下拉由 paths.list_component_items 扫磁盘得到（widget 存的只是
  目录名，如 "z-image-turbo-8bit"），往对应目录里放 "z-image-turbo-4bit"
  之类的新权重，不必改本文件；
- 用哪套 ModelConfig：由 weights.config_for_path 拿目录名去 mflux 的
  AVAILABLE_MODELS 里做前缀匹配（先去掉 -8bit / -4bit 这类量化后缀，
  "z-image-turbo-8bit" → "z-image-turbo"、"z-image-8bit" → "z-image"），
  匹配不到才退回本大类登记的 default_config；
- 步数 / 调度器 / guidance 等由工作流里的 widget 决定，本文件只登记每个大类
  的「默认值」（供节点 widget 初始显示，用户随时可改）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- 自定义类型名（ComfyUI 连线用的类型标识） ---
# CLIP 复用 ComfyUI 原生类型名（与原生 CLIPTextEncode 等节点的 clip 入口一致）；
# 上面只跑纯数据 handle（组件 + 路径 + 精度），真正的模块由 MlxTextEncoder 按需创建。
CLIP = "CLIP"
model = "model"
latents = "latents"
images = "images"
condition = "condition"  # 单条提示词 + 编码它所用的组件配置
vae = "mlx_vae"  # MLX VAE handle（MlxVAELoader → MlxVAEEncoder / MlxVAEDecoder）
ref_images = "mlx_ref_images"  # Flux.2 参考图条件（打包好的参考图 latent + grid ids）
ref_source = "mlx_ref_image_src"  # 有序参考图源（多图，各自保留原尺寸；MlxRefImageSet → MlxVAEEncoder）


# --- 纯数据 handle ---
@dataclass(frozen=True)
class LoraRef:
    path: str
    strength: float


@dataclass(frozen=True)
class MlxClipHandle:
    """只存数据（模型大类 + 组件 + 权重目录），实际模块在 MlxTextEncoder 里创建。

    MlxClipLoader 自己没有文本输入，只填组件配置；正/负提示词分别由两个
    独立的 MlxTextEncoder 节点（对应 ComfyUI 原生 CLIPTextEncode +
    KSampler 的 positive/negative 两条连线）填入，不放在本 handle 上。

    用哪套 ModelConfig 不登记在这里：由 weights.config_for_path 按 path
    （目录名 / 文件名 / repo id）去 AVAILABLE_MODELS 里匹配。
    """

    model_type: str  # 模型大类（MODEL_DEFS 的键，如 "z_image"），决定 class_import
    component: str  # "text_encoder" | "tokenizer"
    source: str  # "local" | "hf_repo"
    path: str  # 权重集目录名（各组件目录下同名，如 "z-image-turbo-8bit"）或 repo id
    precision: str  # "bfloat16" | "float16" | "float32"
    max_length: int | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)
    loras: tuple[LoraRef, ...] = ()  # 只登记，M1 不实际应用（与 MlxModelHandle 一致）
    cache_key: str = ""


@dataclass(frozen=True)
class MlxConditioning:
    """单条提示词 + 编码它的 CLIP 配置与编码结果键（相当于 ComfyUI 的 CONDITIONING）。

    正、负条件分别由两个 MlxTextEncoder 节点产出，再接到 MlxKSamplerMLX 的
    positive / negative 两个入口；本 handle 不合并两者，每个只带自己的一条文本
    （对应 mflux 里 generate_image(prompt=..., negative_prompt=...) 的两个独立参数）。

    编码结果按 encoding_key 存在 cache.py 的 "prompt_encoding" 桶里；数组不放
    进 handle（ComfyUI 的输出缓存会长期持有节点输出）。
    """

    clip: MlxClipHandle
    text: str
    encoding_key: str = ""  # 编码数组的缓存键（数组本身留在 cache.py）


@dataclass(frozen=True)
class MlxModelHandle:
    model_type: str  # 模型大类（MODEL_DEFS 的键，如 "z_image"）
    model_path: str  # 权重集目录名（如 "z-image-turbo-8bit"；扫盘得到，不必预先登记）
    quantize: int  # 量化档位；配置由 model_path 现算（见 weights.config_for_path）
    precision: str
    compile: bool
    compile_cache_limit: int
    loras: tuple[LoraRef, ...] = ()
    cache_key: str = ""


@dataclass(frozen=True)
class MlxLatentHandle:
    kind: str  # "noise" | "packed"
    shape: tuple[int, ...]
    dtype: str
    cache_key: str  # 缓存在 cache.py 中的条目键（数组不放 handle 里）
    model: str = ""  # 产生它的模型大类（MODEL_DEFS 的键）
    source: str = ""  # 同 MlxModelHandle.source
    path: str = ""  # 权重集目录名
    precision: str = ""  # 同 MlxModelHandle.precision
    quantize: int = 0
    model_cache_key: str = ""  # 用于复用 MlxModelHandle 的缓存
    height: int = 0  # 生成尺寸（解码时重建 latent 用）
    width: int = 0


@dataclass(frozen=True)
class MlxVaeHandle:
    """只存「用哪套 VAE + 精度 + 量化档位」；同一 cache_key → 同一份 VAE 实例。

    由 MlxVAELoader 产出（不加载权重），MlxVAEEncoder / MlxVAEDecoder（以及
    兼容的 MlxVAEDecodeRawPIL）按 cache_key 物化它，因此编码器与解码器必然
    共用同一份 VAE —— 编码用的统计量与解码用的权重必须是同一套。
    """

    model_type: str  # 模型大类（MODEL_DEFS 的键，如 "flux2"）
    path: str  # vae/ 目录下的权重集目录名（如 "flux.2-klein-9b-8bit"）
    precision: str  # "bfloat16" | "float16" | "float32"
    quantize: int  # 4 / 8 / 16
    cache_key: str = ""  # 编码器与解码器共用的缓存键


@dataclass(frozen=True)
class MlxRefImageSource:
    """有序参考图源（多图，各自尺寸）：只带键与元信息，图片本体留在 cache.py 的 "ref_source" 桶。

    由 MlxRefImageSet 产出、只接 MlxVAEEncoder 的可选入口 ref_source。
    顺序 = 编码顺序 = 参考图 grid ids 的 t 坐标顺序（第 i 张 → t = 10 + 10 * i），
    因此 sizes / digest 都是**有序**的：换图、换顺序、换张数都会换键。
    """

    count: int  # 张数
    sizes: tuple[tuple[int, int], ...]  # 每张图的原始 (宽, 高)
    digest: str  # 有序图片摘要（image_mod.digest 对 PIL 序列）
    cache_key: str = ""  # "ref_source" 桶里的 PIL 元组键


@dataclass(frozen=True)
class MlxReferenceImages:
    """参考图条件：只带缓存键，数组留在 cache.py 的 "ref_encoding" 桶里。

    - flux2（单图编辑）：`(packed, ids, width, height)` —— packed 与 grid ids，
      拼在目标 latent 之后（每个 step 只取回目标段）；
    - qwen_edit（多图编辑）：`(packed, ids, cond_h_patches, cond_w_patches)` —— 参考图
      latent（按目标尺寸编码）+ 官方 ids（当前 mflux 的 transformer 不读它）+
      `cond_image_grid` 的 patch 尺寸。

    采样器按 `edit_kind` 分派取用方式，两个大类的缓存键互不干扰。
    """

    model_type: str  # 产出它的模型大类（"flux2" | "qwen_edit"）
    vae_path: str  # 编码它的 VAE 权重集目录名（信息性）
    count: int  # 实际参与编码的参考图张数
    height: int  # qwen_edit：目标编辑高度（= 采样器该用的高度）；flux2：建议生成高度
    width: int  # 同上（宽度）
    cache_key: str  # 打包数组的缓存键
    vae_cache_key: str = ""  # 与 MlxVaeHandle.cache_key 同源（诊断用）
    precision: str = ""
    quantize: int = 0
    edit_kind: str = "flux2"  # "flux2" | "qwen_edit"
    cond_grid: tuple[int, int, int] | None = None  # qwen_edit：(1, height//16, width//16)
    seq_len: int = 0  # 参考图 token 数（诊断/打印用）


@dataclass(frozen=True)
class MlxPilImage:
    images: tuple[Any, ...] = ()  # tuple[PIL.Image.Image, ...]
    batch_index: int = -1  # -1 = 整批


# --- 模型配置（数据） ---
@dataclass(frozen=True)
class ComponentDef:
    """某 role 用哪个类、从哪里取；路径不写在这里（由节点 widget 决定）。"""

    name: str  # 与权重定义里的 ComponentDefinition.name / TokenizerDefinition.name 对齐
    class_import: str  # "module:Class"；"" = 由 weight_def 的 tokenizers() 决定
    source: str  # "local" | "hf_repo" | "model_card"
    kind: str = "dir"  # "dir" | "file" | "repo"
    # 实例化之后、写权重之前要执行的挂钩（"module:func"，func(instance)->None）：
    # Qwen 编辑的文本编码器必须先把视觉塔挂到 encoder.visual 上，否则
    # Module.update(strict=False) 会把 encoder.visual.* 的权重静默丢掉。
    attach_import: str = ""


@dataclass(frozen=True)
class MlxModelEntry:
    """一个大类的装配信息：只写「用哪些类 + 兜底配置 + 默认采样参数」。

    不写具体权重路径（由节点 widget 选、扫盘得到），也不列「配置变体」：
    实际用哪套 ModelConfig 由 weights.config_for_path 按选中的目录名推断，
    匹配不到才用 default_config 兜底。
    """

    family: str  # 与 MODEL_DEFS 的键一致（"z_image" | "flux2" | "qwen_edit"），即 model_type
    weight_def: str  # 权重定义类（含 components()/tokenizers()，同大类内共用）
    components: dict[str, ComponentDef]  # role -> 定义；role 见 paths.COMPONENT_DIRS
    default_config: str  # 目录名匹配不到配置时用的兜底（ModelConfig 工厂方法名）
    default_steps: int  # 节点 widget 的默认步数（工作流里可改）
    default_scheduler: str  # 节点 widget 的默认调度器（工作流里可改）
    default_guidance: float = 1.0  # 节点 guidance widget 的初始值（qwen_edit = 2.5）
    supports_compile: bool = True  # False = 采样循环不使用 mx.compile（qwen_edit 的入参含 Config/int）
    prompt_encoder: str = ""  # 提示编码入口（"module:Class.method"）
    prompt_encoder_args: dict[str, Any] = field(default_factory=dict)  # 编码入口的额外参数
    latent_creator: str = ""  # "module:Class"
    supported: bool = True  # False = 尚未验证，选中时节点拒绝执行
    notes: str = ""


def _components(
    *,
    transformer: str,
    vae: str,
    text_encoder: str,
    tokenizer_name: str,
    text_encoder_attach: str = "",
) -> dict[str, ComponentDef]:
    """构造四个 role 的组件定义（只写「用哪个类」，路径来自 widget）。"""
    return {
        "transformer": ComponentDef("transformer", transformer, "local"),
        "vae": ComponentDef("vae", vae, "local"),
        "text_encoder": ComponentDef(
            "text_encoder", text_encoder, "local", attach_import=text_encoder_attach
        ),
        # tokenizer 没有 class_import，由 weight_def 的 TokenizerDefinition 决定
        "tokenizer": ComponentDef(tokenizer_name, "", "local"),
    }


Z_IMAGE = MlxModelEntry(
    family="z_image",
    weight_def="mflux.models.z_image.weights.z_image_weight_definition:ZImageWeightDefinition",
    components=_components(
        transformer="mflux.models.z_image.model.z_image_transformer.transformer:ZImageTransformer",
        vae="mflux.models.z_image.model.z_image_vae.vae:VAE",
        text_encoder="mflux.models.z_image.model.z_image_text_encoder.text_encoder:TextEncoder",
        tokenizer_name="z_image",
    ),
    # 目录名匹配不到时的兜底；z-image-turbo-* 与 z-image-* 都会各自匹配到
    # 同名配置（z_image_turbo / z_image），一般走不到这里。
    default_config="z_image_turbo",
    default_steps=9,
    default_scheduler="linear",
    prompt_encoder="mflux.models.z_image.model.z_image_text_encoder.prompt_encoder:PromptEncoder",
    prompt_encoder_args={"cache": True},
    latent_creator="mflux.models.z_image.latent_creator.z_image_latent_creator:ZImageLatentCreator",
)

FLUX2_KLEIN = MlxModelEntry(
    family="flux2",
    weight_def="mflux.models.flux2.weights.flux2_weight_definition:Flux2KleinWeightDefinition",
    components=_components(
        transformer="mflux.models.flux2.model.flux2_transformer.transformer:Flux2Transformer",
        vae="mflux.models.flux2.model.flux2_vae.vae:Flux2VAE",
        text_encoder="mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder:Qwen3TextEncoder",
        tokenizer_name="qwen3",
    ),
    default_config="flux2_klein_9b",
    default_steps=4,
    default_scheduler="flow_match_euler_discrete",
    prompt_encoder="mflux.models.flux2.model.flux2_text_encoder.prompt_encoder:Flux2PromptEncoder",
    prompt_encoder_args={
        "num_images_per_prompt": 1,
        "max_sequence_length": 512,
        "text_encoder_out_layers": [9, 18, 27],
        "cache": True,
    },
    latent_creator="mflux.models.flux2.latent_creator.flux2_latent_creator:Flux2LatentCreator",
    supported=True,
    notes=(
        "已对接文生图（txt2img）；权重定义只写了 9b（flux.2-klein-9b-*），"
        "4b / base 需要另加配置。默认 guidance=1.0（klein 蒸馏模型不开 CFG），"
        "调度器请用 flow_match_euler_discrete。"
    ),
)

# 键 = 模型大类（= model_type 下拉）；大类内用哪套配置由权重目录名决定
QWEN_EDIT = MlxModelEntry(
    family="qwen_edit",
    weight_def="mflux.models.qwen.weights.qwen_weight_definition:QwenWeightDefinition",
    components=_components(
        transformer="mflux.models.qwen.model.qwen_transformer.qwen_transformer:QwenTransformer",
        vae="mflux.models.qwen.model.qwen_vae.qwen_vae:QwenVAE",
        text_encoder=(
            "mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder:QwenTextEncoder"
        ),
        tokenizer_name="qwen",
        # 视觉塔：Qwen 编辑的条件编码要用 encoder.visual（权重名 encoder.visual.*）
        text_encoder_attach="comfyui_mlx_gen.qwen_edit:attach_vision_tower",
    ),
    # 目录名匹配不到时的兜底（qwen-image-edit-2511-8bit 会命中 qwen-image-edit）
    default_config="qwen_image_edit",
    default_steps=20,
    default_scheduler="linear",
    default_guidance=2.5,
    supports_compile=False,  # transformer 入参含 Config 对象与 int 步号，不编译
    prompt_encoder=(
        "mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder:QwenPromptEncoder"
    ),
    prompt_encoder_args={},  # edit 路径不用它（走 MlxQwenEditEncoder）
    latent_creator="mflux.models.qwen.latent_creator.qwen_latent_creator:QwenLatentCreator",
    supported=True,
    notes=(
        "已对接多图编辑（edit）：条件必须用「MLX Qwen 编辑条件」节点（带参考图），"
        "采样用 linear + guidance 2.5~4.0（负向条件参与 CFG），"
        "参考图 latent 按目标尺寸编码；文生图（qwen-image）未对接。"
    ),
)


MODEL_DEFS: dict[str, MlxModelEntry] = {
    "z_image": Z_IMAGE,
    "flux2": FLUX2_KLEIN,
    "qwen_edit": QWEN_EDIT,
}


# --- 供节点使用的查询入口（只依赖大类，不依赖具体模型名 / 配置名） ---
def model_types() -> list[str]:
    """所有模型大类（= MODEL_DEFS 的键，如 "z_image"、"flux2"、"qwen_edit"）。"""
    return list(MODEL_DEFS)


def entry_for(model_type: str) -> MlxModelEntry:
    """按大类取配置；未知大类直接报错（提示重新选择，或在 MODEL_DEFS 里登记）。"""
    entry = MODEL_DEFS.get(model_type)
    if entry is None:
        raise ValueError(f"未知模型大类: {model_type}（请在 MODEL_DEFS 里登记）")
    return entry
