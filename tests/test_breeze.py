"""Breeze-TTS-2 接入的静态与小波形回归测试（不加载真实大权重）。

运行：
    /opt/anaconda3/envs/py313/bin/python tests/test_breeze.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import wave
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402

from comfyui_mlx_gen import NODE_CLASS_MAPPINGS, paths, pipeline  # noqa: E402
from comfyui_mlx_gen import breeze  # noqa: E402
from comfyui_mlx_gen.cache import Cache  # noqa: E402
from comfyui_mlx_gen.nodes import breeze_sampler as breeze_sampler_module  # noqa: E402
from comfyui_mlx_gen.nodes import vae_decode as vae_decode_module  # noqa: E402
from comfyui_mlx_gen.types import (  # noqa: E402
    MlxClipHandle,
    MlxConditioning,
    MlxModelHandle,
    MlxVaeHandle,
    entry_for,
    model_types,
)

FAILED: list[str] = []


def check(label, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {label}" + (f" | {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def check_raises(label, exc_type, message_part, call):
    try:
        call()
    except exc_type as exc:
        check(label, message_part in str(exc), str(exc))
    except Exception as exc:  # noqa: BLE001
        check(label, False, f"异常类型不对：{type(exc).__name__}: {exc}")
    else:
        check(label, False, f"没有抛出 {exc_type.__name__}")


def make_model(path="Breeze-TTS-2-test"):
    return MlxModelHandle(
        model_type="breeze_tts2",
        model_path=path,
        quantize=4,
        precision="bfloat16",
        compile=False,
        compile_cache_limit=0,
        cache_key="model:breeze",
    )


def make_clip(path="Breeze-TTS-2-test"):
    return MlxClipHandle(
        model_type="breeze_tts2",
        component="text_encoder",
        source="local",
        path=path,
        precision="bfloat16",
        max_length=512,
        quantize=None,
        cache_key="clip:breeze",
    )


def make_vae(path="Breeze-TTS-2-test"):
    return MlxVaeHandle(
        model_type="breeze_tts2",
        path=path,
        precision="bfloat16",
        quantize=4,
        role="vae",
        cache_key="vae:breeze",
    )


def params(**overrides):
    value = {
        "seed": 7,
        "text": "你好，這是一段測試語音。",
        "mode": "speaker",
        "speaker": "S0",
        "ref_text": "",
        "instruction": "",
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "cfg_scale": 1.0,
        "max_tokens": 8,
        "repetition_penalty": 1.0,
        "reference": None,
    }
    value.update(overrides)
    return value


# ---------------------------------------------------------------- 1. 模型与节点登记
entry = entry_for("breeze_tts2")
check("MODEL_DEFS 含 breeze_tts2", "breeze_tts2" in model_types(), str(model_types()))
check(
    "Breeze 是已启用的音频大类",
    entry.supported and entry.media == "audio" and not entry.supports_compile,
    str(entry),
)
check(
    "Breeze 专用采样器与通用采样器节点均已注册",
    {
        "MlxBreezeSampler", "MlxTransformerLoader", "MlxKSamplerMLX", "MlxVAELoader",
        "MlxVAEDecoder", "MlxPilToTorch",
    } <= set(NODE_CLASS_MAPPINGS)
)
inputs = NODE_CLASS_MAPPINGS["MlxBreezeSampler"].INPUT_TYPES()
check(
    "目标文本/克隆转写/设计指令都是外部 STRING socket",
    inputs["required"]["text"] == (
        "STRING",
        {"forceInput": True, "tooltip": "连接外部文本节点；内容是要朗读的目标文本"},
    )
    and inputs["optional"]["ref_text"][1]["forceInput"] is True
    and inputs["optional"]["instruction"][1]["forceInput"] is True
    and inputs["optional"]["ref_audio"][0] == "AUDIO",
    str(inputs),
)
check(
    "专用采样器只暴露 Breeze 控件而不复用通用图像采样控件",
    list(inputs["required"])
    == [
        "model", "text", "seed", "mode", "speaker", "temperature", "top_p", "top_k",
        "cfg_scale", "max_tokens", "repetition_penalty",
    ]
    and list(inputs["optional"]) == ["ref_text", "instruction", "ref_audio"]
    and not {"positive", "negative", "steps", "width", "height", "scheduler"}
    & set(inputs["required"]),
    str(inputs),
)


# --------------------------------------------------------- 2. AUDIO 规范化、摘要与临时 WAV
left = np.array([0.0, 0.5, 2.0, -2.0], dtype=np.float32)
right = np.array([0.0, -0.5, 0.0, 0.0], dtype=np.float32)
audio = {"waveform": np.stack([left, right])[None, :, :], "sample_rate": 16000}
mono, sample_rate, digest = breeze.normalize_reference(audio)
mono2, _, digest2 = breeze.normalize_reference(
    {"waveform": audio["waveform"].copy(), "sample_rate": 16000}
)
check(
    "参考 AUDIO 取首 batch、声道均值、clip，并按内容稳定摘要",
    sample_rate == 16000
    and np.allclose(mono, [0.0, 0.0, 1.0, -1.0])
    and np.array_equal(mono, mono2)
    and digest == digest2,
    f"{mono.tolist()} {digest[:12]}",
)
_, _, different_rate_digest = breeze.normalize_reference(
    {"waveform": audio["waveform"], "sample_rate": 24000}
)
check("摘要包含原始采样率", digest != different_rate_digest)
check_raises(
    "拒绝非 [batch,channels,samples] 参考波形",
    ValueError,
    "[batch, channels, samples]",
    lambda: breeze.normalize_reference({"waveform": np.zeros((2, 8)), "sample_rate": 16000}),
)

temporary_path = None
with breeze.temporary_reference_wav(mono, sample_rate) as path:
    temporary_path = path
    with wave.open(str(path), "rb") as wav:
        wav_info = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes())
    check("临时参考音频是原采样率单声道 PCM16 WAV", wav_info == (1, 2, 16000, 4), str(wav_info))
check("临时 WAV 在上下文退出后删除", temporary_path is not None and not temporary_path.exists())


# --------------------------------------------------------------- 3. 三模式参数校验
breeze.validate_params(params())
breeze.validate_params(params(mode="voice_design", instruction="溫柔、低沉、語速慢"))
breeze.validate_params(
    params(mode="voice_clone", ref_text="參考音頻內容", reference=(mono, sample_rate, digest))
)
check_raises(
    "克隆模式必须连接 AUDIO",
    ValueError,
    "ref_audio",
    lambda: breeze.validate_params(params(mode="voice_clone", ref_text="有轉寫")),
)
check_raises(
    "克隆模式必须填写逐字转写",
    ValueError,
    "breeze_ref_text",
    lambda: breeze.validate_params(params(mode="voice_clone", reference=(mono, sample_rate, digest))),
)
check_raises(
    "设计模式必须填写 instruction",
    ValueError,
    "breeze_instruction",
    lambda: breeze.validate_params(params(mode="voice_design")),
)


# ----------------------------------------------------------- 4. checkpoint 路径与 waveform 缓存
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    model_dir = root / "transformer" / "Breeze-TTS-2-test"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "breeze_tts", "sampling_rate": 24000}), encoding="utf-8"
    )
    old_root = paths.MODEL_ROOT
    paths.MODEL_ROOT = root
    try:
        check(
            "完整 checkpoint 目录按 transformer 选择解析",
            pipeline.breeze_model_dir(model_dir.name) == model_dir.resolve(),
        )
        model_handle = make_model()
        key1 = pipeline.breeze_waveform_cache_key(model_handle, params())
        key2 = pipeline.breeze_waveform_cache_key(
            model_handle,
            params(mode="voice_clone", ref_text="參考音頻內容", reference=(mono, sample_rate, digest)),
        )
        key3 = pipeline.breeze_waveform_cache_key(
            model_handle,
            params(mode="voice_clone", ref_text="參考音頻內容", reference=(mono.copy(), sample_rate, digest)),
        )
        check("参考音频缓存键只看内容摘要，不看数组身份/临时文件名", key2 == key3 and key1 != key2)

        calls = []
        progress_updates = []

        class RecordingProgress:
            def __init__(self, total):
                self.total = total

            def update_absolute(self, current, total=None):
                progress_updates.append((current, total or self.total))

            def complete(self):
                progress_updates.append((self.total, self.total))

        class FakeModel:
            def generate(self, text, **kwargs):
                calls.append((text, kwargs))
                if kwargs["ref_audio"] is not None:
                    assert Path(kwargs["ref_audio"]).is_file()
                yield SimpleNamespace(
                    audio=mx.array([-2.0, -0.25]), sample_rate=24000, token_count=3
                )
                yield SimpleNamespace(
                    audio=mx.array([0.5, 2.0]), sample_rate=24000, token_count=2
                )

        cache = Cache()
        old_progress = pipeline.SamplingProgress
        pipeline.SamplingProgress = RecordingProgress
        try:
            handle = pipeline.run_breeze_sampler(entry, model_handle, FakeModel(), params(), cache)
            reused = pipeline.run_breeze_sampler(entry, model_handle, None, params(), cache)
        finally:
            pipeline.SamplingProgress = old_progress
        cached, hit = cache.get("breeze_waveform", handle.cache_key)
        check(
            "流式 token chunk 驱动进度、拼接并缓存 24 kHz waveform",
            hit
            and handle.kind == "breeze_audio"
            and handle.sample_rate == 24000
            and handle.audio_num_rows == 4
            and tuple(cached[0].shape) == (4,)
            and np.allclose(np.asarray(cached[0]), [-1.0, -0.25, 0.5, 1.0])
            and calls[0][1]["stream"] is True
            and progress_updates == [(3, 8), (5, 8), (8, 8), (8, 8)],
            f"{handle} / progress={progress_updates}",
        )
        check("相同参数命中 waveform 后无需模型", reused.cache_key == handle.cache_key and len(calls) == 1)

        non_stream_calls = []

        class NonStreamingModel:
            def generate(self, text, **kwargs):
                non_stream_calls.append((text, kwargs))
                yield SimpleNamespace(
                    audio=mx.array([0.25, -0.5]), sample_rate=24000, token_count=2
                )

        waveform, sample_rate = breeze.generate_waveform(
            NonStreamingModel(), params(mode="speaker")
        )
        check(
            "未传进度回调时保留 mlx-audio 非流式生成行为",
            non_stream_calls[0][1]["stream"] is False
            and sample_rate == 24000
            and np.allclose(np.asarray(waveform), [0.25, -0.5]),
            str(non_stream_calls),
        )

        clone_params = params(
            mode="voice_clone", ref_text="參考音頻內容", reference=(mono, sample_rate, digest)
        )
        pipeline.run_breeze_sampler(entry, model_handle, FakeModel(), clone_params, cache)
        ref_path = Path(calls[-1][1]["ref_audio"])
        check(
            "克隆生成收到短生命周期 Path 且结束后文件已删除",
            calls[-1][1]["ref_text"] == "參考音頻內容" and not ref_path.exists(),
            str(ref_path),
        )

        decoded = pipeline.decode_breeze_waveform(cached, -1)
        check(
            "解码只包装 24 kHz 单声道 AudioTrack",
            decoded.images == ()
            and decoded.audio.sample_rate == 24000
            and decoded.audio.waveform.shape == (1, 4),
            str(decoded.audio),
        )

        old_cache = vae_decode_module.CACHE
        vae_decode_module.CACHE = cache
        try:
            node_decoded = NODE_CLASS_MAPPINGS["MlxVAEDecoder"]().decode(
                make_vae(), handle, -1
            )[0]
        finally:
            vae_decode_module.CACHE = old_cache
        check(
            "VAE Decode 对 Breeze 不加载第二份模型而直接取 waveform",
            node_decoded.audio.sample_rate == 24000
            and node_decoded.audio.waveform.shape == (1, 4),
        )
    finally:
        paths.MODEL_ROOT = old_root


# ------------------------------------------------------ 5. 通用 sampler 明确拒绝 Breeze
clip = make_clip()
positive = MlxConditioning(text="這是目標台詞", clip=clip, encoding_key="positive")
negative = MlxConditioning(text="ignored", clip=clip, encoding_key="negative")
check_raises(
    "通用 sampler 明确提示 Breeze 使用专用节点",
    ValueError,
    "MLX Breeze Sampler",
    lambda: NODE_CLASS_MAPPINGS["MlxKSamplerMLX"]().sample(
        model=make_model(), positive=positive, negative=negative, seed=19, steps=1,
        width=512, height=512, batch_size=1, guidance=1.0, scheduler="breeze_tts"
    ),
)


# ----------------------------------------------- 6. 专用 sampler 外部文本分派/释放
fake_cache = Cache()
old_cache = breeze_sampler_module.CACHE
old_has = pipeline.has_breeze_waveform
old_prepare = pipeline.prepare_breeze_model
old_run = pipeline.run_breeze_sampler
old_release = pipeline.release_breeze_model
captured = {}
sentinel = object()
try:
    breeze_sampler_module.CACHE = fake_cache
    pipeline.has_breeze_waveform = lambda *_args: False
    pipeline.prepare_breeze_model = lambda *_args: "complete-model"

    def fake_breeze_run(_entry, _handle, complete_model, sampler_params, _cache):
        captured.update(sampler_params)
        captured["complete_model"] = complete_model
        return sentinel

    pipeline.run_breeze_sampler = fake_breeze_run
    pipeline.release_breeze_model = lambda *_args: captured.setdefault("released", True)
    result = NODE_CLASS_MAPPINGS["MlxBreezeSampler"]().sample(
        model=make_model(),
        text="外部節點傳入的目標台詞",
        seed=23,
        mode="voice_design",
        speaker="S4",
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        cfg_scale=1.5,
        max_tokens=24,
        repetition_penalty=1.1,
        instruction="沉穩、清晰",
    )
finally:
    breeze_sampler_module.CACHE = old_cache
    pipeline.has_breeze_waveform = old_has
    pipeline.prepare_breeze_model = old_prepare
    pipeline.run_breeze_sampler = old_run
    pipeline.release_breeze_model = old_release

check(
    "专用 sampler 接收外部文本并始终释放完整模型",
    result == (sentinel,)
    and captured["text"] == "外部節點傳入的目標台詞"
    and captured["seed"] == 23
    and captured["mode"] == "voice_design"
    and captured["speaker"] == "S4"
    and captured["instruction"] == "沉穩、清晰"
    and captured["max_tokens"] == 24
    and captured["complete_model"] == "complete-model"
    and captured["released"] is True,
    str(captured),
)


# ---------------------------------------------------------------- 7. 依赖契约
requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
check("requirements 精确声明 mlx-audio==0.5.1", "mlx-audio==0.5.1" in requirements)


# --------------------------------------------------------- 8. 示例工作流序列化契约
workflow = json.loads((ROOT / "workflows" / "breeze-tts2.json").read_text(encoding="utf-8"))
nodes = workflow["nodes"]
links = workflow["links"]
nodes_by_id = {node["id"]: node for node in nodes}
expected_counts = Counter(
    {
        "PrimitiveStringMultiline": 1,
        "MlxTransformerLoader": 1,
        "MlxVAELoader": 1,
        "MlxBreezeSampler": 1,
        "MlxVAEDecoder": 1,
        "MlxPilToTorch": 1,
        "SaveAudio": 1,
        "PreviewAudio": 1,
    }
)
actual_counts = Counter(node["type"] for node in nodes)
check("Breeze 工作流节点种类与数量完整", actual_counts == expected_counts, str(actual_counts))

model_key = "Breeze-TTS-2-mlx-4bit"
loader_indices = {
    "MlxTransformerLoader": (0, 1),
    "MlxVAELoader": (0, 1),
}
bad_loaders = []
for node_type, (type_index, path_index) in loader_indices.items():
    values = next(node for node in nodes if node["type"] == node_type)["widgets_values"]
    if values[type_index] != "breeze_tts2" or values[path_index] != model_key:
        bad_loaders.append(f"{node_type}: {values}")
check("两个 Loader 都选择 breeze_tts2 + 4-bit 目录", not bad_loaders, "; ".join(bad_loaders))

sampler_node = next(node for node in nodes if node["type"] == "MlxBreezeSampler")
text_node = next(node for node in nodes if node["type"] == "PrimitiveStringMultiline")
check(
    "Breeze 工作流默认 speaker/S0/750 tokens 且文本参数只从 socket 输入",
    sampler_node["widgets_values"]
    == [42, "speaker", "S0", 0.9, 1.0, 50, 1.0, 750, 1.0]
    and [item["name"] for item in sampler_node["inputs"]]
    == ["model", "text", "ref_text", "instruction", "ref_audio"]
    and sampler_node["inputs"][1]["link"] == 1
    and all(item["link"] is None for item in sampler_node["inputs"][2:])
    and text_node["outputs"][0]["links"] == [1],
    str(sampler_node),
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
check("Breeze 工作流 MLX socket 与节点签名一致", not bad_signatures, "; ".join(bad_signatures))

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
check("Breeze 工作流 link 与 socket 元数据一致", not bad_links, "; ".join(bad_links))

pil_node = next(node for node in nodes if node["type"] == "MlxPilToTorch")
check(
    "Breeze 工作流从 MlxPilToTorch AUDIO 分叉到保存与试听",
    pil_node["outputs"][2]["type"] == "AUDIO"
    and pil_node["outputs"][2]["links"] == [6, 7],
    str(pil_node["outputs"][2]),
)

# ---------------------------------------------------------------- 9. 声音克隆工作流
clone_workflow = json.loads(
    (ROOT / "workflows" / "breeze-tts2-voice-clone.json").read_text(encoding="utf-8")
)
clone_nodes = clone_workflow["nodes"]
clone_links = clone_workflow["links"]
clone_nodes_by_id = {node["id"]: node for node in clone_nodes}
clone_counts = Counter(node["type"] for node in clone_nodes)
check(
    "Breeze 声音克隆工作流节点种类与数量完整",
    clone_counts
    == Counter(
        {
            "PrimitiveStringMultiline": 2,
            "LoadAudio": 1,
            "MlxTransformerLoader": 1,
            "MlxVAELoader": 1,
            "MlxBreezeSampler": 1,
            "MlxVAEDecoder": 1,
            "MlxPilToTorch": 1,
            "SaveAudio": 1,
            "PreviewAudio": 1,
        }
    ),
    str(clone_counts),
)

clone_sampler = next(node for node in clone_nodes if node["type"] == "MlxBreezeSampler")
clone_audio = next(node for node in clone_nodes if node["type"] == "LoadAudio")
clone_transformer = next(node for node in clone_nodes if node["type"] == "MlxTransformerLoader")
clone_vae = next(node for node in clone_nodes if node["type"] == "MlxVAELoader")
check(
    "声音克隆工作流选择本机 BF16 Breeze checkpoint",
    clone_transformer["widgets_values"]
    == ["breeze_tts2", "Breeze-TTS-2-mlx", 0, "bfloat16", False, 0]
    and clone_vae["widgets_values"]
    == ["breeze_tts2", "Breeze-TTS-2-mlx", "bfloat16", 0, "vae"],
    f"{clone_transformer['widgets_values']} / {clone_vae['widgets_values']}",
)
check(
    "声音克隆工作流默认 voice_clone 并连接目标文本、逐字稿与参考音频",
    clone_sampler["widgets_values"]
    == [42, "voice_clone", "S0", 0.9, 1.0, 50, 1.0, 750, 1.0]
    and [item["name"] for item in clone_sampler["inputs"]]
    == ["model", "text", "ref_text", "instruction", "ref_audio"]
    and [item["link"] for item in clone_sampler["inputs"]] == [4, 1, 2, None, 3]
    and clone_audio["inputs"]
    == [{"name": "audio", "type": "COMBO", "widget": {"name": "audio"}, "link": None}]
    and clone_audio["outputs"][0]["type"] == "AUDIO"
    and clone_audio["outputs"][0]["links"] == [3],
    str(clone_sampler),
)

clone_bad_signatures = []
for node in clone_nodes:
    cls = NODE_CLASS_MAPPINGS.get(node["type"])
    if cls is None:
        continue
    declared = cls.INPUT_TYPES()
    declared_inputs = {**declared.get("required", {}), **declared.get("optional", {})}
    for item in node.get("inputs", []):
        if item["name"] not in declared_inputs:
            clone_bad_signatures.append(f"{node['type']} 缺输入 {item['name']}")
        elif declared_inputs[item["name"]][0] != item["type"]:
            clone_bad_signatures.append(
                f"{node['type']}.{item['name']}: {item['type']} != "
                f"{declared_inputs[item['name']][0]}"
            )
    output_types = tuple(item["type"] for item in node.get("outputs", []))
    if output_types != tuple(cls.RETURN_TYPES):
        clone_bad_signatures.append(f"{node['type']} 输出 {output_types} != {cls.RETURN_TYPES}")
check(
    "声音克隆工作流 MLX socket 与节点签名一致",
    not clone_bad_signatures,
    "; ".join(clone_bad_signatures),
)

clone_bad_links = []
for link_id, src_id, src_slot, dst_id, dst_slot, link_type in clone_links:
    src_output = clone_nodes_by_id[src_id]["outputs"][src_slot]
    dst_input = clone_nodes_by_id[dst_id]["inputs"][dst_slot]
    if link_id not in (src_output.get("links") or []):
        clone_bad_links.append(f"link {link_id} 不在源输出")
    if dst_input.get("link") != link_id:
        clone_bad_links.append(f"link {link_id} 不在目标输入")
    if src_output["type"] != link_type or dst_input["type"] != link_type:
        clone_bad_links.append(f"link {link_id} 类型不一致")
check(
    "声音克隆工作流 link 与 socket 元数据一致",
    not clone_bad_links,
    "; ".join(clone_bad_links),
)

if FAILED:
    print(f"\n{len(FAILED)} 项失败：{', '.join(FAILED)}")
    raise SystemExit(1)
print("\nBreeze-TTS-2 回归测试全部通过。")