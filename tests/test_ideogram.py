"""Ideogram 4 本地 MLX 接入的静态与小张量回归测试（不加载真实大权重）。

运行：
    /opt/anaconda3/envs/py313/bin/python tests/test_ideogram.py
"""

from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# 当前开发环境里另有一份较旧、没有 Ideogram 4 的手工 MFLUX 源码；专项测试使用
# 仓库的 0.19.1 参考 checkout。生产安装直接来自 requirements.txt 的同版本 wheel。
REFERENCE_MFLUX = ROOT / "docs" / "reference_code" / "mflux" / "src"
mflux_spec = importlib.util.find_spec("mflux")
installed_mflux = (
    Path(next(iter(mflux_spec.submodule_search_locations)))
    if mflux_spec is not None and mflux_spec.submodule_search_locations
    else Path("/__missing_mflux__")
)
if not (installed_mflux / "models" / "ideogram4").is_dir() and REFERENCE_MFLUX.is_dir():
    sys.path.insert(0, str(REFERENCE_MFLUX))

import mlx.core as mx  # noqa: E402

from comfyui_mlx_gen import NODE_CLASS_MAPPINGS, paths, pipeline, runtime, weights  # noqa: E402
from comfyui_mlx_gen.cache import Cache  # noqa: E402
from comfyui_mlx_gen.types import (  # noqa: E402
    MlxClipHandle,
    MlxConditioning,
    MlxModelHandle,
    MlxReferenceImages,
    entry_for,
    model_types,
)

MODEL_KEY = "ideogram-4-fp8"
FAILED: list[str] = []


def check(label, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {label}" + (f" | {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def check_raises(label, exc_type, message_part, call):
    try:
        call()
    except exc_type as exc:
        message = str(exc)
        check(label, message_part in message, message)
    except Exception as exc:  # noqa: BLE001
        check(label, False, f"异常类型不对：{type(exc).__name__}: {exc}")
    else:
        check(label, False, f"没有抛出 {exc_type.__name__}")


# ---------------------------------------------------------------- 1. 本地模型大类
entry = entry_for("ideogram4")
check("MODEL_DEFS 含 ideogram4", "ideogram4" in model_types(), str(model_types()))
check("family 与 MODEL_DEFS 键一致", entry.family == "ideogram4", entry.family)
check("Ideogram 4 已启用", entry.supported is True)
check(
    "默认参数 = 20 步 / ideogram4_default / guidance 7.0",
    (entry.default_steps, entry.default_scheduler, entry.default_guidance)
    == (20, "ideogram4_default", 7.0),
    f"{entry.default_steps} / {entry.default_scheduler} / {entry.default_guidance}",
)
check("兜底配置为 ideogram4_fp8", entry.default_config == "ideogram4_fp8")
check(
    "登记五个本地组件 role",
    set(entry.components)
    == {"transformer", "unconditional_transformer", "vae", "text_encoder", "tokenizer"},
    str(entry.components),
)
check(
    "条件 / 无条件 transformer 对应正确的权重组件名",
    entry.components["transformer"].name == "conditional_transformer"
    and entry.components["unconditional_transformer"].name == "unconditional_transformer",
)
check(
    "使用 Ideogram 4 prompt encoder / latent creator",
    "ideogram4" in entry.prompt_encoder and "ideogram4" in entry.latent_creator,
    f"{entry.prompt_encoder} | {entry.latent_creator}",
)

model_config = weights.config_for_path(MODEL_KEY, entry.default_config)
check(
    "ideogram-4-fp8 命中官方 ModelConfig",
    model_config.model_name == "ideogram-ai/ideogram-4-fp8",
    model_config.model_name,
)

weight_def = runtime.import_object(entry.weight_def)
components = {item.name: item for item in weight_def.get_components()}
check(
    "MFLUX 权重定义含 VAE、双 transformer、文本编码器",
    set(components)
    == {"vae", "conditional_transformer", "unconditional_transformer", "text_encoder"},
    str(components),
)
check(
    "三份 FP8 组件使用 fp8_safetensors 加载模式",
    all(
        components[name].loading_mode == "fp8_safetensors"
        for name in ("conditional_transformer", "unconditional_transformer", "text_encoder")
    ),
)
tokenizer_defs = weight_def.get_tokenizers()
check(
    "Ideogram tokenizer = AutoTokenizer + 2048 token",
    len(tokenizer_defs) == 1
    and tokenizer_defs[0].name == "ideogram4"
    and tokenizer_defs[0].max_length == 2048,
)

# 只读本机 checkpoint 的小型配置与文件头，不读取任何 tensor payload。
local_roots = [
    paths.component_dir(role) / MODEL_KEY
    for role in ("transformer", "text_encoder", "tokenizer", "vae")
]
if all(path.exists() for path in local_roots):
    transformer_root = local_roots[0].resolve().parent
    expected_files = [
        transformer_root / "transformer/config.json",
        transformer_root / "transformer/diffusion_pytorch_model.safetensors",
        transformer_root / "unconditional_transformer/config.json",
        transformer_root / "unconditional_transformer/diffusion_pytorch_model.safetensors",
        local_roots[1] / "config.json",
        local_roots[1] / "model.safetensors",
        local_roots[2] / "tokenizer.json",
        local_roots[3] / "config.json",
        local_roots[3] / "diffusion_pytorch_model.safetensors",
    ]
    check(
        "本机 Ideogram 4 五组件文件完整",
        all(path.is_file() for path in expected_files),
        str([str(path) for path in expected_files if not path.is_file()]),
    )
    text_config = json.loads((local_roots[1] / "config.json").read_text())
    check(
        "本机文本编码器是官方 FP8 weight-only layout",
        text_config.get("ideogram_fp8_weight_only") is True,
        str(text_config.get("ideogram_fp8_weight_only")),
    )
else:
    print("[SKIP] 本机未放置 ideogram-4-fp8，跳过 checkpoint 文件完整性检查")


# ------------------------------------------------------------- 2. 双 transformer 路径
old_root = paths.MODEL_ROOT
try:
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        model_root = temp_path / "mlx"
        snapshot = temp_path / "snapshot"
        (model_root / "transformer").mkdir(parents=True)
        (snapshot / "transformer").mkdir(parents=True)
        (snapshot / "unconditional_transformer").mkdir()
        (model_root / "transformer" / MODEL_KEY).symlink_to(
            snapshot / "transformer", target_is_directory=True
        )
        paths.MODEL_ROOT = model_root
        cond_kind, cond_path = pipeline.component_path(entry, "transformer", MODEL_KEY)
        uncond_kind, uncond_path = pipeline.component_path(
            entry, "unconditional_transformer", MODEL_KEY
        )
        check(
            "conditional transformer 从标准 transformer/ 软链解析",
            cond_kind == "dir" and Path(cond_path) == (snapshot / "transformer").resolve(),
            cond_path,
        )
        check(
            "未建第二条软链时自动找到 snapshot 同级 unconditional_transformer",
            uncond_kind == "dir"
            and Path(uncond_path) == (snapshot / "unconditional_transformer").resolve(),
            uncond_path,
        )
finally:
    paths.MODEL_ROOT = old_root


# ------------------------------------------------------------- 3. 官方采样预设与几何
expected_presets = {
    "ideogram4_default": (20, 3.0, 7.0),
    "ideogram4_quality": (48, 3.0, 7.0),
    "ideogram4_turbo": (12, 3.0, 7.0),
}
for name, (steps, first_guidance, last_guidance) in expected_presets.items():
    preset = pipeline.ideogram4_preset(name)
    check(
        f"{name} 映射官方预设",
        preset.num_steps == steps
        and preset.guidance_schedule[0] == first_guidance
        and preset.guidance_schedule[-1] == last_guidance,
        repr(preset),
    )
check_raises(
    "拒绝把普通 linear scheduler 用于 Ideogram 4",
    ValueError,
    "必须选择自己的采样预设",
    lambda: pipeline.ideogram4_preset("linear"),
)

scheduler_cls = runtime.import_object(
    "mflux.models.ideogram4.model.ideogram4_scheduler.scheduler:Ideogram4Scheduler"
)
t_values, s_values = scheduler_cls.make_timesteps(
    num_steps=20, height=1024, width=1024, mu=0.0, std=1.75
)
check(
    "官方 logit-normal schedule 生成 20 对区间端点",
    t_values.shape == (20,) and s_values.shape == (20,),
    f"{t_values.shape} / {s_values.shape}",
)

latent_creator = runtime.import_object(entry.latent_creator)
noise = latent_creator.create_noise(seed=42, height=256, width=256)
unpacked = latent_creator.unpack_latents(noise, height=256, width=256)
check("256² 噪声形状 = [1, 256, 128]", tuple(noise.shape) == (1, 256, 128), noise.shape)
check(
    "Ideogram latent unpack = [1, 32, 32, 32]",
    tuple(unpacked.shape) == (1, 32, 32, 32),
    unpacked.shape,
)
check_raises(
    "拒绝小于 256 的画布",
    ValueError,
    "[256, 2048]",
    lambda: latent_creator.validate_dimensions(width=128, height=256),
)
check_raises(
    "拒绝非 16 倍数画布",
    ValueError,
    "multiple of 16",
    lambda: latent_creator.validate_dimensions(width=257, height=256),
)


# --------------------------------------------------------- 4. Prompt 输入与 FP8 小层
class FakeTokenizer:
    def tokenize_one(self, prompt, max_length=None):  # noqa: ARG002
        return np.asarray([11, 22, 33], dtype=np.int64)


prompt_encoder = runtime.import_object(entry.prompt_encoder)
inputs = prompt_encoder.build_inputs(FakeTokenizer(), ["test"], height=256, width=256)
check(
    "Prompt inputs 包含 3 文本 token + 256 图像 token",
    inputs["max_text_tokens"] == 3
    and inputs["num_image_tokens"] == 256
    and tuple(inputs["position_ids"].shape) == (1, 259, 3),
    str({k: getattr(v, "shape", v) for k, v in inputs.items()}),
)
fake_features = mx.zeros((1, 3, 53248), dtype=mx.float32)
negative_inputs = prompt_encoder.negative_inputs(inputs, fake_features)
check(
    "无条件分支只保留 256 个图像 token，LLM features 全零",
    tuple(negative_inputs["llm_features"].shape) == (1, 256, 53248)
    and float(mx.sum(negative_inputs["llm_features"]).item()) == 0.0,
)

fp8_cls = runtime.import_object(
    "mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear:Fp8Linear"
)
fp8 = fp8_cls(4, 3, bias=True)
fp8_out = fp8(mx.ones((2, 4), dtype=mx.bfloat16))
check(
    "MLX FP8 线性层可执行小张量前向",
    hasattr(mx, "from_fp8") and tuple(fp8_out.shape) == (2, 3),
    f"from_fp8={hasattr(mx, 'from_fp8')} shape={fp8_out.shape}",
)


# --------------------------------------------------------- 5. Fake 双 transformer 采样与解码
class FakeTransformer:
    config = type("Config", (), {"in_channels": 128})()

    def __call__(self, **kwargs):
        return mx.zeros_like(kwargs["x"])


cache = Cache()
encoding_key = "ideogram4:test-encoding"
cache.get_or_create(
    pipeline.IDEOGRAM4_PROMPT_BUCKET,
    encoding_key,
    lambda: (inputs, fake_features[:, : int(inputs["max_text_tokens"]), :]),
)
sampled = pipeline._sample_ideogram4(
    entry,
    {"transformer": FakeTransformer(), "unconditional_transformer": FakeTransformer()},
    {
        "scheduler_name": "ideogram4_turbo",
        "positive_encoding_key": encoding_key,
        "height": 256,
        "width": 256,
        "seed": 7,
        "batch_size": 1,
        "compile_model": False,
    },
    cache,
)
check(
    "Fake 双 transformer 完成 12 步采样并返回标准 latent",
    tuple(sampled.shape) == (1, 256, 128),
    sampled.shape,
)


class FakeVAE:
    def decode(self, latents):
        return mx.zeros((latents.shape[0], 3, 256, 256), dtype=mx.float32)


decoded = pipeline.decode_latents(entry, {"vae": FakeVAE()}, sampled[0], 256, 256)
check("Ideogram latent 走 Flux2VAE.decode 直解分支", tuple(decoded.shape) == (1, 3, 256, 256))


# --------------------------------------------------------- 6. ComfyUI 节点边界
clip = MlxClipHandle(
    model_type="ideogram4",
    component="text_encoder",
    source="local",
    path=MODEL_KEY,
    precision="bfloat16",
    max_length=2048,
    quantize=None,
    cache_key="clip:ideogram4",
)
model = MlxModelHandle(
    model_type="ideogram4",
    model_path=MODEL_KEY,
    quantize=0,
    precision="bfloat16",
    compile=False,
    compile_cache_limit=0,
    cache_key="model:ideogram4",
)
positive = MlxConditioning(clip=clip, text='{"high_level_description":"test"}')
negative = MlxConditioning(clip=clip, text="")
dummy_ref = MlxReferenceImages(
    model_type="ideogram4",
    vae_path=MODEL_KEY,
    count=1,
    height=256,
    width=256,
    cache_key="must-not-be-read",
)
sampler_cls = NODE_CLASS_MAPPINGS["MlxKSamplerMLX"]
check_raises(
    "Ideogram 4 明确拒绝 ref_images",
    NotImplementedError,
    "只支持文生图",
    lambda: sampler_cls().sample(
        model=model,
        positive=positive,
        negative=negative,
        seed=0,
        steps=20,
        width=256,
        height=256,
        batch_size=1,
        guidance=7.0,
        scheduler="ideogram4_default",
        ref_images=dummy_ref,
    ),
)
check(
    "没有注册任何 Ideogram API 专属节点",
    not any("Ideogram" in name for name in NODE_CLASS_MAPPINGS),
    str(sorted(NODE_CLASS_MAPPINGS)),
)


# --------------------------------------------------------- 7. 示例工作流契约
workflow = json.loads((ROOT / "workflows" / "ideogram-4-fp8.json").read_text(encoding="utf-8"))
nodes = workflow["nodes"]
links = workflow["links"]
nodes_by_id = {node["id"]: node for node in nodes}
links_by_id = {link[0]: link for link in links}
expected_counts = Counter(
    {
        "MlxClipLoader": 1,
        "MlxTextEncoder": 2,
        "MlxTransformerLoader": 1,
        "MlxVAELoader": 1,
        "MlxKSamplerMLX": 1,
        "MlxVAEDecoder": 1,
        "MlxSaveImage": 1,
        "MlxPilToTorch": 1,
        "SaveImage": 1,
        "PreviewImage": 1,
    }
)
actual_counts = Counter(node["type"] for node in nodes)
check("工作流沿用标准 MLX 节点链", actual_counts == expected_counts, str(actual_counts))
custom_types = {name for name in actual_counts if name.startswith("Mlx")}
check("工作流 MLX 节点均已注册", custom_types <= NODE_CLASS_MAPPINGS.keys())

loader_indices = {
    "MlxClipLoader": (0, 2),
    "MlxTransformerLoader": (0, 1),
    "MlxVAELoader": (0, 1),
}
bad_loaders = []
for node_type, (type_index, path_index) in loader_indices.items():
    values = next(node for node in nodes if node["type"] == node_type)["widgets_values"]
    if values[type_index] != "ideogram4" or values[path_index] != MODEL_KEY:
        bad_loaders.append(f"{node_type}: {values}")
check("三个 Loader 都选 ideogram4 + 本地 FP8 权重", not bad_loaders, "; ".join(bad_loaders))

transformer_values = next(
    node for node in nodes if node["type"] == "MlxTransformerLoader"
)["widgets_values"]
vae_values = next(node for node in nodes if node["type"] == "MlxVAELoader")["widgets_values"]
sampler_node = next(node for node in nodes if node["type"] == "MlxKSamplerMLX")
check("Transformer 保留原生 FP8（quantize=0）", transformer_values[2] == 0, str(transformer_values))
check("VAE 不做在线量化（quantize=0）", vae_values[3] == 0, str(vae_values))
check(
    "采样器使用 1024² + 官方 default 预设（seed 后面跟 control_after_generate）",
    sampler_node["widgets_values"][:8]
    == [42, "randomize", 20, 1024, 1024, 1, 7.0, "ideogram4_default"],
    str(sampler_node["widgets_values"]),
)
check(
    "Ideogram 工作流没有 ref_images 连线",
    [item["name"] for item in sampler_node["inputs"]] == ["model", "positive", "negative"],
)

bad_signatures = []
for node in nodes:
    cls = NODE_CLASS_MAPPINGS.get(node["type"])
    if cls is None:
        continue
    declared = cls.INPUT_TYPES()
    declared_inputs = {**declared.get("required", {}), **declared.get("optional", {})}
    for item in node.get("inputs", []):
        if item["name"] not in declared_inputs:
            bad_signatures.append(f"{node['type']} 缺输入 {item['name']}")
        elif declared_inputs[item["name"]][0] != item["type"]:
            bad_signatures.append(
                f"{node['type']}.{item['name']}: {item['type']} != "
                f"{declared_inputs[item['name']][0]}"
            )
    output_types = tuple(item["type"] for item in node.get("outputs", []))
    if output_types != tuple(cls.RETURN_TYPES):
        bad_signatures.append(f"{node['type']} 输出 {output_types} != {cls.RETURN_TYPES}")
check("工作流 socket 与节点签名一致", not bad_signatures, "; ".join(bad_signatures))

bad_links = []
for link_id, src_id, src_slot, dst_id, dst_slot, link_type in links:
    src_output = nodes_by_id[src_id]["outputs"][src_slot]
    dst_input = nodes_by_id[dst_id]["inputs"][dst_slot]
    if link_id not in (src_output.get("links") or []):
        bad_links.append(f"link {link_id} 不在源输出")
    if dst_input.get("link") != link_id:
        bad_links.append(f"link {link_id} 不在目标输入")
    if src_output["type"] != link_type or dst_input["type"] != link_type:
        bad_links.append(f"link {link_id} 类型不一致")
check("工作流 link 与 socket 元数据一致", not bad_links, "; ".join(bad_links))

preview = next(node for node in nodes if node["type"] == "PreviewImage")
preview_link = links_by_id[preview["inputs"][0]["link"]]
check(
    "PreviewImage 接 MlxPilToTorch 的 IMAGE 输出",
    nodes_by_id[preview_link[1]]["type"] == "MlxPilToTorch"
    and preview_link[2] == 0
    and preview_link[5] == "IMAGE",
    str(preview_link),
)


# --------------------------------------------------------- 8. 依赖与错误实现清理
requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
check("依赖要求 MLX 0.32.x", "mlx>=0.32.0,<0.33.0" in requirements)
check("依赖固定 MFLUX 0.19.1", "mflux==0.19.1" in requirements)
check("错误的 API 客户端文件已删除", not (ROOT / "src/comfyui_mlx_gen/ideogram.py").exists())
check("错误的 API 节点文件已删除", not (ROOT / "src/comfyui_mlx_gen/nodes/ideogram.py").exists())


if FAILED:
    print(f"\n{len(FAILED)} 项检查失败：")
    for label in FAILED:
        print(f"  - {label}")
    sys.exit(1)

print(f"\n全部通过：Ideogram 4 本地节点链与工作流契约有效（{len(nodes)} 个节点）。")