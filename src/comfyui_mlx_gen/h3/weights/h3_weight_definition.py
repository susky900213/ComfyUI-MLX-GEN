# 依据 mlx-gen 0.37 的 mflux/models/minimax_h3/weights/h3_weight_definition.py（MIT，
# Filip Strand / AbstractVision）重写：去掉 ModelConfig 注册表（0.19.1 里没有 minimax-h3），
# 改成「组件 role → 权重规则」的纯数据表，供本插件自己的 h3/weights/loader.py 逐张量使用。
"""MiniMax-H3 四个组件的权重规则：映射、精度、量化谓词。

规则与参考实现逐条对齐（这几条都是「错了不会报错、只会出垃圾」的类型，别省）：

- transformer 的 `proj_in. / audio_proj_in. / time_embedder. / proj_out. / audio_proj_out.`
  必须 **fp32**（对应 diffusers 的 `_keep_in_fp32_modules`）；
- `.adaln_proj.` 与 `norm_out.` 在 q8/q4 下必须保持 bf16（量化后输出直接变噪声）；
- 音频 VAE 固定 **fp32** 且不量化；视频 VAE 不量化（精度随主精度）；
- 条件编码器只读 Qwen3-VL 的 `embed_tokens` 与前 50 层 + 27 层视觉塔，其余 shard 不读。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from comfyui_mlx_gen.h3.weights.h3_weight_mapping import (
    TEXT_ENCODER_NUM_LAYERS,
    VISION_NUM_BLOCKS,
    MiniMaxH3WeightMapping,
)

# 量化分组大小（H3 的 hidden 5376 / 5120 / 14336 / 25600 都能被 64 整除）
GROUP_SIZE = 64
# 条件编码器的 tokenizer 名（= MlxClipLoader 里 tokenizer 组件的 name）
TOKENIZER_NAME = "minimax_h3"
# tokenizer 原样喂提示词：不加特殊 token、不加 chat 模板
TOKENIZER_MAX_LENGTH = 8192


@dataclass(frozen=True)
class H3ComponentDef:
    """一个 role 的权重规则（不含路径：路径来自节点 widget）。"""

    name: str  # role：transformer / text_encoder / vae / audio_vae
    skip_quantization: bool = False  # True = 即使传了 quantize 也不量化
    precision: str | None = None  # None = 用节点 widget 的精度；"float32" = 强制
    mapping_getter: Callable[[], list[Any]] | None = None  # 需要改名时给 WeightTarget 列表
    bulk_transform: Callable[[Any], Any] | None = None  # 逐张量变换（3D 核转置等）
    num_layers: int | None = None
    num_blocks: int | None = None
    weight_prefix_filters: tuple[str, ...] = field(default_factory=tuple)


class MiniMaxH3WeightDefinition:
    """组件 / 精度 / 量化谓词的登记表（静态方法，等价于参考实现的同名类）。"""

    # diffusers 的 `_keep_in_fp32_modules`：输入输出头与时间步 MLP
    TRANSFORMER_FP32_PREFIXES = ("proj_in.", "audio_proj_in.", "time_embedder.", "proj_out.", "audio_proj_out.")
    # q8/q4 下必须保持 bf16 的敏感路径（量化后输出直接变噪声）
    TRANSFORMER_QUANTIZATION_SENSITIVE_FRAGMENTS = (".adaln_proj.", "norm_out.")
    TOKENIZER_NAME = TOKENIZER_NAME
    GROUP_SIZE = GROUP_SIZE

    @staticmethod
    def is_transformer_fp32_path(path: str) -> bool:
        """模块路径（`proj_out`）或参数键（`proj_out.weight`）命中 fp32 保留集。"""
        return any(
            path == prefix[:-1] or path.startswith(prefix)
            for prefix in MiniMaxH3WeightDefinition.TRANSFORMER_FP32_PREFIXES
        )

    @staticmethod
    def is_transformer_quantization_sensitive_path(path: str) -> bool:
        dotted = f"{path}."
        return any(
            fragment in dotted
            for fragment in MiniMaxH3WeightDefinition.TRANSFORMER_QUANTIZATION_SENSITIVE_FRAGMENTS
        )

    @staticmethod
    def quantization_predicate(path: str, module: Any, bits: int | None = None) -> bool:
        """这个模块该不该量化（`nn.quantize` 的 class_predicate）。"""
        if not hasattr(module, "to_quantized"):
            return False
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2 or weight.shape[-1] % GROUP_SIZE:
            return False
        if MiniMaxH3WeightDefinition.is_transformer_fp32_path(path):
            return False
        return not MiniMaxH3WeightDefinition.is_transformer_quantization_sensitive_path(path)

    @staticmethod
    def get_components() -> list[H3ComponentDef]:
        """四个组件的规则（顺序 = 加载顺序：小的先来，便于按需验证）。"""
        return [_COMPONENTS[name] for name in ("audio_vae", "vae", "text_encoder", "transformer")]

    @staticmethod
    def component(name: str) -> H3ComponentDef:
        try:
            return _COMPONENTS[name]
        except KeyError as exc:
            raise KeyError(f"MiniMax-H3 没有组件 {name}（可选：{', '.join(_COMPONENTS)}）") from exc


_COMPONENTS: dict[str, H3ComponentDef] = {
    # 条件编码器：Qwen3-VL 的 embed_tokens + 前 50 层 + 视觉塔（键名要按 WeightTarget 改名）
    "text_encoder": H3ComponentDef(
        name="text_encoder",
        mapping_getter=MiniMaxH3WeightMapping.get_text_encoder_mapping,
        num_layers=TEXT_ENCODER_NUM_LAYERS,
        num_blocks=VISION_NUM_BLOCKS,
        weight_prefix_filters=("model.language_model.", "model.visual."),
    ),
    # omni transformer：diffusers 键名逐键透传（模块属性名与 checkpoint 一致）
    "transformer": H3ComponentDef(name="transformer"),
    # 视频 VAE：逐键透传 + 3D 核转成 MLX 的 channel-last
    "vae": H3ComponentDef(
        name="vae",
        skip_quantization=True,
        bulk_transform=MiniMaxH3WeightMapping.video_vae_transform,
    ),
    # 音频 VAE：固定 fp32（bf16 会留下可听的量化噪声）+ weight_norm 折叠
    "audio_vae": H3ComponentDef(name="audio_vae", skip_quantization=True, precision="float32"),
}
