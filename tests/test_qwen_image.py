"""qwen_image 大类的静态校验 + 边界检查（不加载真实权重，跑完约 1 秒）。

    /opt/anaconda3/envs/py313/bin/python tests/test_qwen_image.py

覆盖：
1. `qwen_image` 已在 MODEL_DEFS 里登记，组件与 latent/prompt 入口齐全，
   且**不挂视觉塔**（attach_import 为空）；
2. `qwen-image-2512-8bit` 命中的是 generic「Qwen/Qwen-Image」（T2I）配置，
   `transformer_overrides` 是空的（构造 `QwenTransformer()` 走默认参数）；
3. **2511 回归**：`qwen-image-edit-2511-8bit` 的 `transformer_overrides` 里那个
   `qwen_edit_plus` 不是构造参数，`resolve_class_kwargs` 必须把它过滤掉，
   否则 `QwenTransformer(**class_kwargs)` 会 TypeError；
4. `QwenPromptEncoder.encode_prompt` 的入参名与 `encode_text` 里的调用一致；
5. `QwenLatentCreator` 的 create_noise / unpack 形状与采样、解码约定一致；
6. 采样器边界：qwen_image 不接参考图、qwen_edit 必须有参考图；
7. `workflows/qwen-image-2512.json` 的节点与 widget 值对得上节点签名。
"""

import inspect
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen import NODE_CLASS_MAPPINGS, pipeline, runtime, weights  # noqa: E402
from comfyui_mlx_gen.types import (  # noqa: E402
    MlxClipHandle,
    MlxConditioning,
    MlxModelHandle,
    MlxReferenceImages,
    entry_for,
    model_types,
)

T2I_KEY = "qwen-image-2512-8bit"
EDIT_KEY = "qwen-image-edit-2511-8bit"
FAILED = []


def check(label, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {label}" + (f" | {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def check_raises(label, exc_type, message_part, call):
    """边界检查：既要抛指定异常，也要给出能指导用户修工作流的信息。"""
    try:
        call()
    except exc_type as exc:
        message = str(exc)
        check(label, message_part in message, message)
    except Exception as exc:  # noqa: BLE001
        check(label, False, f"异常类型不对：{type(exc).__name__}: {exc}")
    else:
        check(label, False, f"没有抛出 {exc_type.__name__}")


# ---------------------------------------------------------------- 1. 大类已登记
entry = entry_for("qwen_image")
check("MODEL_DEFS 含 qwen_image", "qwen_image" in model_types(), str(model_types()))
check("family 与 MODEL_DEFS 的键一致", entry.family == "qwen_image", entry.family)
comps = entry.components
check(
    "组件齐全（transformer / vae / text_encoder 都有 class_import，tokenizer 名对得上）",
    all(comps[r].class_import for r in ("transformer", "vae", "text_encoder"))
    and comps["tokenizer"].name == "qwen",
    str({k: (v.class_import or v.name) for k, v in comps.items()}),
)
check(
    "不挂视觉塔（attach_import 为空，与 mflux 的 generic 配置一致）",
    comps["text_encoder"].attach_import == "",
    repr(comps["text_encoder"].attach_import),
)
check("兜底配置 = qwen_image", entry.default_config == "qwen_image", entry.default_config)
check(
    "默认 20 步 / flow_match_euler_discrete / guidance 4.0",
    (entry.default_steps, entry.default_scheduler, entry.default_guidance)
    == (20, "flow_match_euler_discrete", 4.0),
    f"{entry.default_steps} / {entry.default_scheduler} / {entry.default_guidance}",
)
check("supports_compile = False（入参含 Config 与 int 步号）", entry.supports_compile is False)
check("supported = True", entry.supported is True)
check(
    "latent_creator / prompt_encoder 已登记",
    bool(entry.latent_creator) and bool(entry.prompt_encoder),
    f"{entry.latent_creator} , {entry.prompt_encoder}",
)

# ------------------------------------------------- 2. 权重集 → 配置（T2I 那一套）
t2i_config = weights.config_for_path(T2I_KEY, entry.default_config)
check(
    "qwen-image-2512-8bit → Qwen-Image 文生图配置",
    t2i_config.model_name in {"Qwen/Qwen-Image", "Qwen/Qwen-Image-2512"},
    f"{t2i_config.model_name}，num_train_steps={t2i_config.num_train_steps}，"
    f"shift {t2i_config.sigma_base_shift}~{t2i_config.sigma_max_shift}",
)
check(
    "该配置的 transformer_overrides 是空的（构造走默认参数）",
    dict(t2i_config.transformer_overrides) == {},
    str(dict(t2i_config.transformer_overrides)),
)
check(
    "支持 CFG（supports_guidance 不是 False）",
    t2i_config.supports_guidance is not False,
    str(t2i_config.supports_guidance),
)

# ------------------------------------ 3. 2511 回归：class_kwargs 要按签名过滤
edit_entry = entry_for("qwen_edit")
edit_config = weights.config_for_path(EDIT_KEY, edit_entry.default_config)
raw_overrides = dict(edit_config.transformer_overrides)
filtered = pipeline.resolve_class_kwargs(edit_entry, "transformer", edit_config)
check(
    "2511 的运行时标记不会泄漏进 transformer 构造参数",
    "qwen_edit_plus" not in filtered,
    str(raw_overrides),
)
check(
    "resolve_class_kwargs 过滤掉非构造参数（2511 才能建出 QwenTransformer）",
    "qwen_edit_plus" not in filtered,
    f"过滤后 {filtered}",
)
transformer_cls = runtime.import_object(edit_entry.components["transformer"].class_import)
ctor_signature = inspect.signature(transformer_cls)
ctor_params = ctor_signature.parameters
check(
    "过滤后剩下的键都是构造参数",
    all(k in ctor_params for k in filtered),
    str(sorted(ctor_params)),
)
check(
    "QwenTransformer(**class_kwargs) 可按签名绑定（不在静态测试里分配 57 层模型）",
    ctor_signature.bind(**filtered) is not None,
    str(filtered),
)

# ------------------------------------ 4. prompt encoder 签名与 encode_text 一致
prompt_encoder = runtime.import_object(entry.prompt_encoder)
pe_params = set(inspect.signature(prompt_encoder.encode_prompt).parameters)
check(
    "QwenPromptEncoder.encode_prompt 的入参名与 encode_text 的调用一致",
    {"prompt", "negative_prompt", "prompt_cache", "qwen_tokenizer", "qwen_text_encoder"}
    <= pe_params,
    str(sorted(pe_params)),
)

# ------------------------------------ 5. latent creator 形状（采样 / 解码约定）
latent_creator = runtime.import_object(entry.latent_creator)
noise = latent_creator.create_noise(0, 256, 256)
check(
    "create_noise 给 [1, (h/16)*(w/16), 64]",
    tuple(noise.shape) == (1, 256, 64),
    str(tuple(noise.shape)),
)
unpacked = latent_creator.unpack_latents(noise, 256, 256)
check(
    "unpack_latents 给 [1, 16, h/8, w/8]（decode_latents 依赖这个形状）",
    tuple(unpacked.shape) == (1, 16, 32, 32),
    str(tuple(unpacked.shape)),
)
created = pipeline.create_latents(entry, 0, 256, 256, 2)
check(
    "create_latents 每个 batch 项都是 [1, 256, 64]",
    len(created) == 2 and all(tuple(a.shape) == (1, 256, 64) for a in created),
    str([tuple(a.shape) for a in created]),
)

# ------------------------------------------------------- 6. 采样器 edit / t2i 边界
sampler_cls = NODE_CLASS_MAPPINGS["MlxKSamplerMLX"]


def make_clip(model_type, path):
    return MlxClipHandle(
        model_type=model_type,
        component="text_encoder",
        source="local",
        path=path,
        precision="bfloat16",
        max_length=1058,
        quantize=None,
        cache_key=f"clip:{model_type}",
    )


def make_model(model_type, path):
    return MlxModelHandle(
        model_type=model_type,
        model_path=path,
        quantize=8,
        precision="bfloat16",
        compile=False,
        compile_cache_limit=0,
        cache_key=f"model:{model_type}",
    )


def call_sampler(model_type, path, ref=None):
    clip = make_clip(model_type, path)
    positive = MlxConditioning(clip=clip, text="test", encoding_key="positive")
    negative = MlxConditioning(clip=clip, text=" ", encoding_key="negative")
    return sampler_cls().sample(
        model=make_model(model_type, path),
        positive=positive,
        negative=negative,
        seed=0,
        steps=1,
        width=256,
        height=256,
        batch_size=1,
        guidance=4.0,
        scheduler="flow_match_euler_discrete",
        ref_images=ref,
    )


dummy_ref = MlxReferenceImages(
    model_type="qwen_image",
    vae_path=T2I_KEY,
    count=1,
    height=256,
    width=256,
    cache_key="must-not-be-read",
    edit_kind="qwen_edit",
)
check_raises(
    "qwen_image + ref_images 明确拒绝（不会静默走编辑路径）",
    ValueError,
    "不接参考图",
    lambda: call_sampler("qwen_image", T2I_KEY, dummy_ref),
)
check_raises(
    "qwen_edit 缺少 ref_images 明确拒绝（原有编辑语义不变）",
    ValueError,
    "必须有参考图",
    lambda: call_sampler("qwen_edit", EDIT_KEY),
)

# --------------------------------------------------------- 7. 工作流静态契约
workflow_path = ROOT / "workflows" / "qwen-image-2512.json"
workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
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
check(
    "工作流节点种类与数量完整",
    actual_counts == expected_counts,
    f"实际 {dict(actual_counts)}",
)

custom_types = {name for name in actual_counts if name.startswith("Mlx")}
check(
    "工作流里的 MLX 节点都已注册",
    custom_types <= NODE_CLASS_MAPPINGS.keys(),
    f"未注册 {sorted(custom_types - NODE_CLASS_MAPPINGS.keys())}",
)

expected_widget_counts = {
    "MlxClipLoader": 6,
    "MlxTextEncoder": 1,
    "MlxTransformerLoader": 6,
    "MlxVAELoader": 5,
    "MlxKSamplerMLX": 11,
    "MlxVAEDecoder": 1,
    "MlxSaveImage": 2,
    "MlxPilToTorch": 0,
    "SaveImage": 1,
    "PreviewImage": 0,
}
bad_widget_counts = [
    f"{node['type']}#{node['id']}: {len(node.get('widgets_values', []))}"
    for node in nodes
    if len(node.get("widgets_values", [])) != expected_widget_counts[node["type"]]
]
check(
    "每个节点的 widgets_values 数量与当前节点签名一致",
    not bad_widget_counts,
    "; ".join(bad_widget_counts),
)

# Loader 的路径 widget 不在同一索引：CLIP=2，Transformer/VAE=1。
loader_expectations = {
    "MlxClipLoader": (0, 2),
    "MlxTransformerLoader": (0, 1),
    "MlxVAELoader": (0, 1),
}
bad_loaders = []
for node_type, (type_index, path_index) in loader_expectations.items():
    loader_node = next(node for node in nodes if node["type"] == node_type)
    values = loader_node["widgets_values"]
    if values[type_index] != "qwen_image" or values[path_index] != T2I_KEY:
        bad_loaders.append(f"{node_type}: {values}")
check(
    "三个 Loader 都选 qwen_image + qwen-image-2512-8bit（按各自真实索引）",
    not bad_loaders,
    "; ".join(bad_loaders),
)

clip_values = next(node for node in nodes if node["type"] == "MlxClipLoader")["widgets_values"]
transformer_values = next(
    node for node in nodes if node["type"] == "MlxTransformerLoader"
)["widgets_values"]
vae_values = next(node for node in nodes if node["type"] == "MlxVAELoader")["widgets_values"]
sampler_node = next(node for node in nodes if node["type"] == "MlxKSamplerMLX")
check(
    "CLIP Loader 使用 text_encoder / bf16 / max_length 1058 / 不二次量化",
    clip_values[1:] == ["text_encoder", T2I_KEY, "bfloat16", 1058, 0],
    str(clip_values),
)
check(
    "Transformer Loader 使用 q8 / bf16，并关闭不受支持的 compile",
    transformer_values[2:] == [8, "bfloat16", False, 0],
    str(transformer_values),
)
check(
    "VAE Loader 使用 bf16 / q8 / vae role",
    vae_values[2:] == ["bfloat16", 8, "vae"],
    str(vae_values),
)
check(
    "采样默认值 = seed 0 / 20 steps / 1024² / batch 1 / CFG 4.0 / flow-match",
    sampler_node["widgets_values"][:7]
    == [0, 20, 1024, 1024, 1, 4.0, "flow_match_euler_discrete"],
    str(sampler_node["widgets_values"]),
)
check(
    "Qwen 文生图工作流没有 ref_images 输入",
    [item["name"] for item in sampler_node["inputs"]] == ["model", "positive", "negative"],
    str([item["name"] for item in sampler_node["inputs"]]),
)

# 自定义节点的已序列化 socket 必须仍存在于 INPUT_TYPES / RETURN_TYPES。
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
check(
    "工作流里的 MLX 输入/输出 socket 与当前节点类签名一致",
    not bad_signatures,
    "; ".join(bad_signatures),
)

# link 表与每个 input/output 上的元数据双向一致；重点防止 IMAGE 预览误接 MASK。
bad_links = []
for link_id, src_id, src_slot, dst_id, dst_slot, link_type in links:
    src = nodes_by_id[src_id]
    dst = nodes_by_id[dst_id]
    src_output = src["outputs"][src_slot]
    dst_input = dst["inputs"][dst_slot]
    if link_id not in (src_output.get("links") or []):
        bad_links.append(f"link {link_id} 不在源输出元数据")
    if dst_input.get("link") != link_id:
        bad_links.append(f"link {link_id} 不在目标输入元数据")
    if src_output["type"] != link_type or dst_input["type"] != link_type:
        bad_links.append(
            f"link {link_id}: {src_output['type']} -> {link_type} -> {dst_input['type']}"
        )
for node in nodes:
    for output in node.get("outputs", []):
        for link_id in output.get("links") or []:
            if link_id not in links_by_id:
                bad_links.append(f"{node['type']}#{node['id']} 引用了不存在的 link {link_id}")
check("工作流 link 与 socket 元数据双向一致", not bad_links, "; ".join(bad_links))

preview_node = next(node for node in nodes if node["type"] == "PreviewImage")
preview_link = links_by_id[preview_node["inputs"][0]["link"]]
pil_node = nodes_by_id[preview_link[1]]
check(
    "PreviewImage 接 MlxPilToTorch 的 IMAGE（slot 0），不是 MASK",
    pil_node["type"] == "MlxPilToTorch"
    and preview_link[2] == 0
    and preview_link[5] == "IMAGE",
    str(preview_link),
)


if FAILED:
    print(f"\n{len(FAILED)} 项检查失败：")
    for label in FAILED:
        print(f"  - {label}")
    sys.exit(1)

print(f"\n全部通过：qwen_image 静态回归与工作流契约有效（{len(nodes)} 个节点）。")
