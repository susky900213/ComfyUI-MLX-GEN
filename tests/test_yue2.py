"""YuE2-3B 原生 MLX 接入的静态与小张量回归测试（不加载真实大权重）。

运行：
    /opt/anaconda3/envs/py313/bin/python tests/test_yue2.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402

from comfyui_mlx_gen import NODE_CLASS_MAPPINGS, paths, pipeline  # noqa: E402
from comfyui_mlx_gen.cache import Cache  # noqa: E402
from comfyui_mlx_gen.nodes import sampler as sampler_module  # noqa: E402
from comfyui_mlx_gen.nodes import vae_decode as vae_decode_module  # noqa: E402
from comfyui_mlx_gen.types import (  # noqa: E402
    MlxClipHandle,
    MlxConditioning,
    MlxModelHandle,
    MlxVaeHandle,
    entry_for,
    model_types,
)
from comfyui_mlx_gen.yue2 import pipeline as yue2_pipeline  # noqa: E402

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


def make_clip(path="YuE2-test"):
    return MlxClipHandle(
        model_type="yue2",
        component="text_encoder",
        source="local",
        path=path,
        precision="bfloat16",
        max_length=9000,
        quantize=None,
        cache_key="clip:yue2",
    )


def make_model(path="YuE2-test"):
    return MlxModelHandle(
        model_type="yue2",
        model_path=path,
        quantize=8,
        precision="bfloat16",
        compile=False,
        compile_cache_limit=0,
        cache_key="model:yue2",
    )


def make_vae(path="YuE2-test"):
    return MlxVaeHandle(
        model_type="yue2",
        role="vae",
        path=path,
        precision="bfloat16",
        quantize=8,
        cache_key="vae:yue2",
    )


# ---------------------------------------------------------------- 1. 模型登记
entry = entry_for("yue2")
check("MODEL_DEFS 含 yue2", "yue2" in model_types(), str(model_types()))
check("YuE2 是已启用的音频大类", entry.supported and entry.media == "audio", str(entry))
check(
    "默认参数 = 32 步 / yue2_midpoint / guidance 1.0",
    (entry.default_steps, entry.default_scheduler, entry.default_guidance)
    == (32, "yue2_midpoint", 1.0),
)
check(
    "登记 transformer / vae / tokenizer / text_encoder",
    set(entry.components) == {"transformer", "vae", "tokenizer", "text_encoder"},
    str(entry.components),
)
check(
    "YuE2 复用既有节点、没有 HTTP 专属节点",
    {"MlxClipLoader", "MlxTextEncoder", "MlxTransformerLoader", "MlxKSamplerMLX",
     "MlxVAELoader", "MlxVAEDecoder", "MlxPilToTorch"} <= set(NODE_CLASS_MAPPINGS)
    and not any("YuE" in name for name in NODE_CLASS_MAPPINGS),
)


# ---------------------------------------------------------- 2. 变体目录与 tokenizer
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    variant = tmp_path / "snapshot" / "8bit"
    variant.mkdir(parents=True)
    for name in (
        "config.json", "qwen.tiktoken", "yue2_generation_config.json",
        "model.safetensors", "vae_config.json", "vae.safetensors",
    ):
        (variant / name).touch()
    transformer_root = tmp_path / "transformer"
    vae_root = tmp_path / "vae"
    transformer_root.mkdir()
    vae_root.mkdir()
    (transformer_root / "YuE2-dir").symlink_to(variant, target_is_directory=True)
    (transformer_root / "YuE2-file.safetensors").symlink_to(variant / "model.safetensors")
    (vae_root / "YuE2-file.safetensors").symlink_to(variant / "vae.safetensors")

    old_root = paths.MODEL_ROOT
    paths.MODEL_ROOT = tmp_path
    try:
        check(
            "完整变体目录软链接可解析",
            pipeline.yue2_variant_dir("YuE2-dir", "transformer") == variant.resolve(),
        )
        check(
            "transformer 单文件软链接可恢复 sibling 配置",
            pipeline.yue2_variant_dir("YuE2-file.safetensors", "transformer")
            == variant.resolve(),
        )
        check(
            "VAE 单文件软链接可恢复 sibling 配置",
            pipeline.yue2_variant_dir("YuE2-file.safetensors", "vae") == variant.resolve(),
        )
    finally:
        paths.MODEL_ROOT = old_root

local_variant = pipeline.yue2_variant_dir("YuE2-3B-MLX-8bit.safetensors", "transformer")
tokenizer = yue2_pipeline.Tokenizer(local_variant / "qwen.tiktoken")
prefix = yue2_pipeline.token_prefix(tokenizer, "indie pop, warm vocal", "", "off")
check(
    "当前本机 8bit 单文件软链接与 tokenizer 可用",
    local_variant.name == "8bit" and len(prefix) > 4 and prefix[-1] == yue2_pipeline.MUSIC_START,
    f"{local_variant} / {len(prefix)} tokens",
)


# -------------------------------------------------- 3. 文本透传与 sampler 分派
clip = make_clip()
text_node = NODE_CLASS_MAPPINGS["MlxTextEncoder"]()
positive = text_node.encode("cinematic synthwave", clip)[0]
negative = text_node.encode("[Verse]\nHello world", clip)[0]
check(
    "文本节点不提前编码：正向 style / 负向 lyrics 保留原文",
    positive.text == "cinematic synthwave"
    and negative.text == "[Verse]\nHello world"
    and bool(positive.encoding_key)
    and bool(negative.encoding_key)
    and positive.encoding_key != negative.encoding_key,
)

fake_cache = Cache()
old_cache = sampler_module.CACHE
old_prepare = pipeline.prepare_yue2_sampler_components
old_run = pipeline.run_yue2_sampler
old_release = pipeline.release_yue2_sampler_components
captured = {}
sentinel = object()
try:
    sampler_module.CACHE = fake_cache
    pipeline.prepare_yue2_sampler_components = lambda *_args: {"fake": True}

    def fake_run(_entry, _model, comps, params, _cache):
        captured.update(params)
        captured["comps"] = comps
        return sentinel

    pipeline.run_yue2_sampler = fake_run
    pipeline.release_yue2_sampler_components = lambda *_args: captured.setdefault("released", True)
    result = NODE_CLASS_MAPPINGS["MlxKSamplerMLX"]().sample(
        model=make_model(),
        positive=positive,
        negative=negative,
        seed=17,
        steps=2,
        width=512,
        height=512,
        batch_size=1,
        guidance=1.5,
        scheduler="yue2_midpoint",
        cot="melody",
        max_tokens=200,
    )
finally:
    sampler_module.CACHE = old_cache
    pipeline.prepare_yue2_sampler_components = old_prepare
    pipeline.run_yue2_sampler = old_run
    pipeline.release_yue2_sampler_components = old_release

check(
    "采样器把 style / lyrics / cot / max_tokens / NAR steps 映射到 YuE2",
    result == (sentinel,)
    and captured == {
        "seed": 17, "steps": 2, "guidance": 1.5,
        "style": "cinematic synthwave", "lyrics": "[Verse]\nHello world",
        "cot": "melody", "max_tokens": 200, "comps": {"fake": True}, "released": True,
    },
    str(captured),
)

released_after_error = {}
try:
    sampler_module.CACHE = Cache()
    pipeline.prepare_yue2_sampler_components = lambda *_args: {"fake": True}

    def fail_run(*_args):
        raise RuntimeError("synthetic generation failure")

    pipeline.run_yue2_sampler = fail_run
    pipeline.release_yue2_sampler_components = (
        lambda *_args: released_after_error.setdefault("released", True)
    )
    check_raises(
        "YuE2 生成失败时保留原异常",
        RuntimeError,
        "synthetic generation failure",
        lambda: NODE_CLASS_MAPPINGS["MlxKSamplerMLX"]().sample(
            model=make_model(), positive=positive, negative=negative, seed=0, steps=2,
            width=512, height=512, batch_size=1, guidance=1.0,
            scheduler="yue2_midpoint", max_tokens=200,
        ),
    )
finally:
    sampler_module.CACHE = old_cache
    pipeline.prepare_yue2_sampler_components = old_prepare
    pipeline.run_yue2_sampler = old_run
    pipeline.release_yue2_sampler_components = old_release
check(
    "YuE2 生成失败时仍释放主模型",
    released_after_error == {"released": True},
    str(released_after_error),
)

check_raises(
    "YuE2 明确拒绝 batch_size > 1",
    ValueError,
    "一次生成一首",
    lambda: NODE_CLASS_MAPPINGS["MlxKSamplerMLX"]().sample(
        model=make_model(), positive=positive, negative=negative, seed=0, steps=2,
        width=512, height=512, batch_size=2, guidance=1.0, scheduler="yue2_midpoint",
    ),
)


# -------------------------------------------------------- 4. latent 缓存与解码契约
cache = Cache()
model = make_model()
params = {
    "seed": 1, "steps": 2, "guidance": 1.0, "style": "test", "lyrics": "",
    "cot": "off", "max_tokens": 200,
}
fake_latents = mx.zeros((7, 64), dtype=mx.float32)
old_generate = yue2_pipeline.generate_music_latents
try:
    yue2_pipeline.generate_music_latents = lambda *_args, **_kwargs: (
        fake_latents, {"duration": 0.28}
    )
    latent_handle = pipeline.run_yue2_sampler(
        entry,
        model,
        {"transformer": object(), "tokenizer": object(), "generation_config": {}},
        params,
        cache,
    )
finally:
    yue2_pipeline.generate_music_latents = old_generate

cached, hit = cache.get("component_weights", latent_handle.cache_key)
check(
    "latent 缓存值直接是 [frames,64] 数组（不是 metadata tuple）",
    hit and hasattr(cached, "shape") and tuple(cached.shape) == (7, 64),
    repr(type(cached)),
)
check(
    "latent handle 带 audio kind / 帧数 / 精确时长",
    latent_handle.kind == "yue2_audio"
    and latent_handle.audio_num_rows == 7
    and yue2_pipeline.audio_samples(7) == 13376
    and yue2_pipeline.audio_samples(200) == 383936
    and abs(latent_handle.duration - 13376 / 48000) < 1e-9,
    str(latent_handle),
)
check("可在加载主模型前识别 latent 命中", pipeline.has_yue2_latents(model, params, cache))


class FakeVAE:
    def decode_tiled(self, latents):
        assert tuple(latents.shape) == (7, 64)
        # 故意越界，覆盖解码出口 clip。
        return mx.array([[-2.0, 2.0], [-0.5, 0.5], [0.5, -0.5], [2.0, -2.0]])


old_prepare_vae = pipeline.prepare_yue2_vae
try:
    pipeline.prepare_yue2_vae = lambda _handle, _cache: FakeVAE()
    decoded = pipeline.decode_yue2_latents(fake_latents, make_vae(), cache, -1)
finally:
    pipeline.prepare_yue2_vae = old_prepare_vae

check(
    "VAE 解码返回 48 kHz 双声道并裁剪到 [-1,1]",
    decoded.images == ()
    and decoded.audio is not None
    and decoded.audio.sample_rate == 48000
    and tuple(decoded.audio.waveform.shape) == (2, 4)
    and float(np.min(decoded.audio.waveform)) == -1.0
    and float(np.max(decoded.audio.waveform)) == 1.0,
    str(decoded.audio),
)

old_node_cache = vae_decode_module.CACHE
old_decode_yue2 = pipeline.decode_yue2_latents
old_release_vae = pipeline.release_yue2_vae
captured_decode = {}
try:
    vae_decode_module.CACHE = cache

    def fake_decode(arr, _vae, _cache, batch_index):
        captured_decode["shape"] = tuple(arr.shape)
        captured_decode["batch_index"] = batch_index
        return decoded

    pipeline.decode_yue2_latents = fake_decode
    pipeline.release_yue2_vae = lambda *_args: captured_decode.setdefault("released", True)
    node_decoded = NODE_CLASS_MAPPINGS["MlxVAEDecoder"]().decode(
        make_vae(), latent_handle, -1
    )[0]
finally:
    vae_decode_module.CACHE = old_node_cache
    pipeline.decode_yue2_latents = old_decode_yue2
    pipeline.release_yue2_vae = old_release_vae

check(
    "VAE 节点从缓存取数组并在解码后释放 YuE2 VAE",
    node_decoded is decoded
    and captured_decode == {"shape": (7, 64), "batch_index": -1, "released": True},
    str(captured_decode),
)


# ----------------------------------------------------------- 5. 配置与依赖契约
generation_config = yue2_pipeline.load_generation_config(local_variant)
check(
    "生成配置含 AR / NAR / ODE 必需字段",
    {"abc", "semantic", "ode_steps", "ode_method", "context"} <= set(generation_config)
    and generation_config["ode_method"] == "midpoint"
    and generation_config["semantic"]["min_tokens"] == 200,
    json.dumps(generation_config, ensure_ascii=False),
)
requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
check("requirements 声明 tiktoken>=0.9", "tiktoken>=0.9" in requirements)


# --------------------------------------------------------- 6. 示例工作流序列化契约
workflow = json.loads((ROOT / "workflows" / "yue2-3b.json").read_text(encoding="utf-8"))
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
        "MlxPilToTorch": 1,
        "SaveAudio": 1,
        "PreviewAudio": 1,
    }
)
actual_counts = Counter(node["type"] for node in nodes)
check("工作流节点种类与数量完整", actual_counts == expected_counts, str(actual_counts))
custom_types = {name for name in actual_counts if name.startswith("Mlx")}
check("工作流 MLX 节点均已注册", custom_types <= NODE_CLASS_MAPPINGS.keys())

model_key = "YuE2-3B-MLX-4bit"
loader_indices = {
    "MlxClipLoader": (0, 2),
    "MlxTransformerLoader": (0, 1),
    "MlxVAELoader": (0, 1),
}
bad_loaders = []
for node_type, (type_index, path_index) in loader_indices.items():
    values = next(node for node in nodes if node["type"] == node_type)["widgets_values"]
    if values[type_index] != "yue2" or values[path_index] != model_key:
        bad_loaders.append(f"{node_type}: {values}")
check("三个 Loader 都选择 yue2 + 4-bit 变体目录", not bad_loaders, "; ".join(bad_loaders))

text_nodes = [node for node in nodes if node["type"] == "MlxTextEncoder"]
style_node = next(node for node in text_nodes if "style" in node.get("title", ""))
lyrics_node = next(node for node in text_nodes if "lyrics" in node.get("title", ""))
sampler_node = next(node for node in nodes if node["type"] == "MlxKSamplerMLX")
check(
    "正向文本明确作为 style、负向文本明确作为 lyrics",
    bool(style_node["widgets_values"][0].strip())
    and "[Verse]" in lyrics_node["widgets_values"][0]
    and style_node["outputs"][0]["links"] == [3]
    and lyrics_node["outputs"][0]["links"] == [5],
)
check(
    "采样器序列化 32 步 midpoint / batch 1 / guidance 1 / full CoT / 200 token",
    sampler_node["widgets_values"]
    == [42, 32, 512, 512, 1, 1.0, "yue2_midpoint", 124, 12.0, 3.0,
        "auto", "full", 200],
    str(sampler_node["widgets_values"]),
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
check("工作流 MLX socket 与节点签名一致", not bad_signatures, "; ".join(bad_signatures))

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

pil_node = next(node for node in nodes if node["type"] == "MlxPilToTorch")
save_audio = next(node for node in nodes if node["type"] == "SaveAudio")
preview_audio = next(node for node in nodes if node["type"] == "PreviewAudio")
audio_links = pil_node["outputs"][2]
check(
    "MlxPilToTorch 的第三个输出是分叉到保存与试听的 AUDIO",
    audio_links["type"] == "AUDIO"
    and audio_links["slot_index"] == 2
    and audio_links["links"] == [9, 10]
    and links_by_id[9][1:3] == [pil_node["id"], 2]
    and links_by_id[10][1:3] == [pil_node["id"], 2],
    str(audio_links),
)
check(
    "SaveAudio 使用当前内置契约：audio + filename_prefix widget + AUDIO 输出",
    save_audio["inputs"]
    == [
        {"name": "audio", "type": "AUDIO", "link": 9, "slot_index": 0},
        {
            "name": "filename_prefix",
            "type": "STRING",
            "widget": {"name": "filename_prefix"},
            "link": None,
        },
    ]
    and save_audio["widgets_values"] == ["audio/ComfyUI-MLX-GEN-yue2"]
    and save_audio["outputs"]
    == [{"name": "audio", "type": "AUDIO", "links": None, "slot_index": 0}]
    and links_by_id[9] == [9, pil_node["id"], 2, save_audio["id"], 0, "AUDIO"],
    str(save_audio),
)
check(
    "PreviewAudio 使用当前内置契约：仅 audio 输入并透传 AUDIO 输出",
    preview_audio["inputs"]
    == [{"name": "audio", "type": "AUDIO", "link": 10, "slot_index": 0}]
    and preview_audio["widgets_values"] == []
    and preview_audio["outputs"]
    == [{"name": "audio", "type": "AUDIO", "links": None, "slot_index": 0}]
    and links_by_id[10] == [10, pil_node["id"], 2, preview_audio["id"], 0, "AUDIO"],
    str(preview_audio),
)

if FAILED:
    print(f"\n{len(FAILED)} 项失败：{', '.join(FAILED)}")
    raise SystemExit(1)
print("\nYuE2 回归测试全部通过。")