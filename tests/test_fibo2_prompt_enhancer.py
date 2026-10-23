"""FIBO 2.0 VGL enhancer tests that do not require a running LLM."""

from __future__ import annotations

import base64
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen import NODE_CLASS_MAPPINGS  # noqa: E402
from comfyui_mlx_gen.fibo2.memory import Fibo2MemoryStore  # noqa: E402
from comfyui_mlx_gen.fibo2 import prompt_enhancer as enhancer  # noqa: E402
from comfyui_mlx_gen.nodes import fibo2_prompt_enhancer as node_module  # noqa: E402
from comfyui_mlx_gen.nodes.fibo2_prompt_enhancer import (  # noqa: E402
    MlxFibo2PromptEnhancerI2I,
    MlxFibo2PromptEnhancerT2I,
    _vgl_result,
)


def _valid_vgl() -> dict:
    return {
        "short_description": "A red ceramic cup on a wooden table in warm window light.",
        "objects": [
            {
                "description": "A handmade red ceramic coffee cup.",
                "location": "center foreground",
                "relationship": "Resting on the table.",
                "relative_size": "medium within frame",
                "shape_and_color": "Cylindrical form in deep red.",
                "texture": "glazed ceramic",
                "appearance_details": "A small curved handle.",
                "number_of_objects": None,
                "pose": None,
                "expression": None,
                "clothing": None,
                "action": None,
                "gender": None,
                "skin_tone_and_texture": None,
                "orientation": "upright",
            }
        ],
        "background_setting": "A quiet kitchen with a softly blurred window.",
        "lighting": {
            "conditions": "warm natural daylight",
            "direction": "side-lit from left",
            "shadows": "soft shadows",
        },
        "aesthetics": {
            "composition": "centered close-up",
            "color_scheme": "warm red and brown",
            "mood_atmosphere": "calm and inviting",
        },
        "photographic_characteristics": {
            "depth_of_field": "shallow",
            "focus": "sharp focus on the cup",
            "camera_angle": "eye-level",
            "lens_focal_length": "50mm standard lens",
        },
        "style_medium": "photograph",
        "artistic_style": "realistic",
        "context": "Commercial lifestyle product photograph.",
        "text_render": [],
        "edit_instruction": "Generate a red ceramic cup on a wooden table.",
    }


def test_t2i_payload_is_text_only_and_uses_direct_vgl_contract():
    payload = enhancer.build_request_payload(
        model="local-llm",
        mode=enhancer.MODE_T2I,
        prompt="a red cup",
    )

    assert payload["model"] == "local-llm"
    assert payload["stream"] is False
    assert payload["messages"][1]["content"] == "User generation request:\na red cup"
    system = payload["messages"][0]["content"]
    assert '"short_description"' in system
    assert '"edit_instruction"' in system
    assert "structured_prompt" in system
    assert "do not wrap" in system


def test_t2i_payload_accepts_one_optional_visual_reference():
    url = "data:image/png;base64,reference"
    payload = enhancer.build_request_payload(
        model="local-vlm",
        mode=enhancer.MODE_T2I,
        prompt="Create a new poster inspired by the palette in <image_1>",
        image_urls=[url],
    )

    content = payload["messages"][1]["content"]
    assert content == [
        {
            "type": "text",
            "text": (
                "User generation request:\n"
                "Create a new poster inspired by the palette in <image_1>\n\n"
                "Optional visual reference <image_1> follows. Use it only as visual "
                "context for the requested new image; this is not an edit request."
            ),
        },
        {"type": "text", "text": "<image_1>"},
        {"type": "image_url", "image_url": {"url": url}},
    ]
    system = payload["messages"][0]["content"]
    assert "one optional image as visual context" in system
    assert "not an image-edit operation" in system


def test_context_and_persistent_memory_are_added_before_current_request():
    current_url = "data:image/png;base64,current"
    payload = enhancer.build_request_payload(
        model="local-vlm",
        mode=enhancer.MODE_T2I,
        prompt="make the new poster warmer",
        image_urls=[current_url],
        context=[
            {
                "prompt": "use the product from <image_1>",
                "enhanced_prompt": "{\"old\": \"context\"}",
                "mode": enhancer.MODE_I2I,
                "has_image": True,
            }
        ],
        memories=[
            {
                "prompt": "prefer centered poster layouts",
                "enhanced_prompt": "{\"old\": \"memory\"}",
                "mode": enhancer.MODE_T2I,
                "has_image": False,
            }
        ],
    )

    messages = payload["messages"]
    assert "Persistent FIBO 2.0 memory explicitly saved by the user" in messages[1]["content"]
    assert "prefer centered poster layouts" in messages[1]["content"]
    assert "mode: i2i" in messages[2]["content"]
    assert "old <image_N> labels do not refer to current images" in messages[2]["content"]
    assert messages[3] == {"role": "assistant", "content": '{"old": "context"}'}
    assert messages[4]["content"][-1] == {
        "type": "image_url",
        "image_url": {"url": current_url},
    }


def test_fibo2_memory_store_is_node_scoped_bounded_and_clearable(tmp_path):
    store = Fibo2MemoryStore(
        tmp_path / "fibo2.json", max_context_turns=2, max_memory_items=2
    )
    for index in range(3):
        store.record_turn(
            "node-a",
            prompt=f"request {index}",
            enhanced_prompt=f"VGL {index}",
            mode=enhancer.MODE_T2I,
            has_image=index == 2,
            save_memory=True,
        )
    store.record_turn(
        "node-b",
        prompt="other",
        enhanced_prompt="other VGL",
        mode=enhancer.MODE_I2I,
        has_image=True,
        save_memory=False,
    )

    assert [item["prompt"] for item in store.get_context("node-a")] == [
        "request 1",
        "request 2",
    ]
    assert [item["prompt"] for item in store.get_memories("node-a")] == [
        "request 1",
        "request 2",
    ]
    assert store.get_context("node-a", 0) == []
    assert store.get_memories("node-b") == []
    assert store.get_context("node-b")[0]["has_image"] is True

    stored = json.loads((tmp_path / "fibo2.json").read_text(encoding="utf-8"))
    assert "image" not in json.dumps(stored).casefold().replace("has_image", "")

    store.clear("node-a", "context")
    assert store.get_context("node-a") == []
    assert len(store.get_memories("node-a")) == 2
    store.clear("node-a", "memory")
    assert store.get_memories("node-a") == []


def test_i2i_payload_numbers_one_to_four_images_in_order():
    urls = [f"data:image/png;base64,image-{index}" for index in range(1, 5)]
    payload = enhancer.build_request_payload(
        model="local-vlm",
        mode=enhancer.MODE_I2I,
        prompt="Use <image_1> as composition and the product from <image_4>.",
        image_urls=urls,
    )

    content = payload["messages"][1]["content"]
    assert isinstance(content, list)
    assert "<image_1>, <image_2>, <image_3>, <image_4>" in content[0]["text"]
    for index, url in enumerate(urls, 1):
        offset = 1 + (index - 1) * 2
        assert content[offset] == {"type": "text", "text": f"<image_{index}>"}
        assert content[offset + 1] == {"type": "image_url", "image_url": {"url": url}}
    assert "Change only what it requires" in payload["messages"][0]["content"]


def test_image_count_and_mode_contracts_are_enforced():
    with pytest.raises(ValueError, match="至少需要一张"):
        enhancer.build_request_payload(
            model="vlm", mode=enhancer.MODE_I2I, prompt="edit", image_urls=[]
        )
    with pytest.raises(ValueError, match="最多支持 4 张"):
        enhancer.build_request_payload(
            model="vlm",
            mode=enhancer.MODE_I2I,
            prompt="edit",
            image_urls=["data:image/png;base64,x"] * 5,
        )
    with pytest.raises(ValueError, match="文生图.*最多支持 1 张"):
        enhancer.build_request_payload(
            model="llm",
            mode=enhancer.MODE_T2I,
            prompt="generate",
            image_urls=["data:image/png;base64,x"] * 2,
        )


def test_data_url_is_a_real_png():
    data_url = enhancer.image_to_data_url(Image.new("RGB", (2, 1), (12, 34, 56)))
    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")


def test_response_cleaning_extracts_and_formats_valid_vgl():
    source = _valid_vgl()
    output = enhancer.format_vgl_response(
        "<think>private chain of thought</think>\n```json\n"
        + json.dumps(source)
        + "\n```\nThis sentence is ignored."
    )
    assert json.loads(output) == source
    assert output.startswith("{\n  \"short_description\"")


def test_validator_normalizes_omitted_nullable_object_fields():
    value = _valid_vgl()
    for field in (
        "texture",
        "appearance_details",
        "number_of_objects",
        "pose",
        "expression",
        "clothing",
        "action",
        "gender",
        "skin_tone_and_texture",
        "orientation",
    ):
        value["objects"][0].pop(field)
    normalized = enhancer.validate_vgl(value)
    assert normalized["objects"][0]["texture"] is None
    assert normalized["objects"][0]["number_of_objects"] is None
    assert list(normalized) == list(enhancer.TOP_LEVEL_FIELDS)
    assert list(normalized["objects"][0]) == list(enhancer.OBJECT_FIELDS)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.pop("lighting"), "缺少顶层字段.*lighting"),
        (lambda value: value.update(objects=[{}] * 6), "objects 最多"),
        (lambda value: value.update(text_render="none"), "text_render 必须是数组"),
        (
            lambda value: value["objects"][0].update(number_of_objects=0),
            "number_of_objects",
        ),
    ],
)
def test_validator_rejects_invalid_vgl(mutate, match):
    value = _valid_vgl()
    mutate(value)
    with pytest.raises(ValueError, match=match):
        enhancer.validate_vgl(value)


def test_invalid_completion_and_response_shapes_are_rejected():
    with pytest.raises(ValueError, match="没有 JSON 对象"):
        enhancer.parse_vgl_response("Here is your prompt, but no object")
    with pytest.raises(RuntimeError, match=r"choices\[0\]"):
        enhancer.extract_completion_text({"choices": []})
    with pytest.raises(RuntimeError, match="没有返回 assistant"):
        enhancer.extract_completion_text({"choices": [{"message": {"content": ""}}]})


def test_http_client_uses_openai_url_headers_and_body(monkeypatch):
    seen: dict = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = {key.casefold(): value for key, value in request.header_items()}
        seen["body"] = json.loads(request.data.decode())
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(enhancer, "urlopen", fake_urlopen)
    payload = {"model": "local", "messages": []}
    result = enhancer.request_completion(
        base_url="http://127.0.0.1:9000/v1/",
        payload=payload,
        timeout=8,
        api_key="secret",
    )
    assert result["choices"][0]["message"]["content"] == "{}"
    assert seen == {
        "url": "http://127.0.0.1:9000/v1/chat/completions",
        "headers": {
            "accept": "application/json",
            "authorization": "Bearer secret",
            "content-type": "application/json",
        },
        "body": payload,
        "timeout": 8.0,
    }


def test_nodes_are_registered_and_t2i_forwards_configuration(monkeypatch):
    assert NODE_CLASS_MAPPINGS["MlxFibo2PromptEnhancerT2I"] is MlxFibo2PromptEnhancerT2I
    assert NODE_CLASS_MAPPINGS["MlxFibo2PromptEnhancerI2I"] is MlxFibo2PromptEnhancerI2I
    plugin = runpy.run_path(str(ROOT / "__init__.py"))
    assert plugin["NODE_CLASS_MAPPINGS"]["MlxFibo2PromptEnhancerT2I"] is (
        MlxFibo2PromptEnhancerT2I
    )
    assert plugin["NODE_CLASS_MAPPINGS"]["MlxFibo2PromptEnhancerI2I"] is (
        MlxFibo2PromptEnhancerI2I
    )
    assert _vgl_result("{}") == {"ui": {"vgl_prompt": ["{}"]}, "result": ("{}",)}

    seen: dict = {}

    def fake_enhance_prompt(**kwargs):
        seen.update(kwargs)
        return "{\"valid\": true}"

    monkeypatch.setattr(node_module, "enhance_prompt", fake_enhance_prompt)
    result = MlxFibo2PromptEnhancerT2I().enhance(
        "a cup",
        model="my-llm",
        base_url="http://localhost:9000/v1",
        api_key="token",
        temperature=0.3,
        top_p=0.8,
        max_tokens=1234,
        timeout=7,
    )
    assert result["result"] == ("{\"valid\": true}",)
    assert "image" in MlxFibo2PromptEnhancerT2I.INPUT_TYPES()["optional"]
    assert "context_turns" in MlxFibo2PromptEnhancerT2I.INPUT_TYPES()["required"]
    assert "save_memory" in MlxFibo2PromptEnhancerT2I.INPUT_TYPES()["required"]
    assert MlxFibo2PromptEnhancerT2I.INPUT_TYPES()["hidden"]["unique_id"] == "UNIQUE_ID"
    assert MlxFibo2PromptEnhancerI2I.INPUT_TYPES()["hidden"]["unique_id"] == "UNIQUE_ID"
    assert seen == {
        "prompt": "a cup",
        "mode": enhancer.MODE_T2I,
        "images": (),
        "model": "my-llm",
        "base_url": "http://localhost:9000/v1",
        "api_key": "token",
        "temperature": 0.3,
        "top_p": 0.8,
        "max_tokens": 1234,
        "timeout": 7,
        "context": [],
        "memories": [],
    }


def test_t2i_node_converts_one_optional_image_and_rejects_batches(monkeypatch):
    seen: dict = {}

    def fake_enhance_prompt(**kwargs):
        seen.update(kwargs)
        return "{}"

    monkeypatch.setattr(node_module, "enhance_prompt", fake_enhance_prompt)
    image = np.zeros((1, 2, 3, 3), dtype=np.float32)
    result = MlxFibo2PromptEnhancerT2I().enhance("use its colors", model="vlm", image=image)
    assert result["result"] == ("{}",)
    assert seen["mode"] == enhancer.MODE_T2I
    assert len(seen["images"]) == 1
    assert seen["images"][0].size == (3, 2)

    with pytest.raises(ValueError, match="可选参考图必须是 1 张.*收到 2 张"):
        MlxFibo2PromptEnhancerT2I().enhance(
            "too many", model="vlm", image=np.zeros((2, 2, 2, 3), dtype=np.float32)
        )


def test_i2i_node_converts_batch_and_rejects_five_images(monkeypatch):
    seen: dict = {}

    def fake_enhance_prompt(**kwargs):
        seen.update(kwargs)
        return "{}"

    monkeypatch.setattr(node_module, "enhance_prompt", fake_enhance_prompt)
    images = np.zeros((4, 2, 3, 3), dtype=np.float32)
    result = MlxFibo2PromptEnhancerI2I().enhance(images, "combine them", model="vlm")
    assert result["result"] == ("{}",)
    assert seen["mode"] == enhancer.MODE_I2I
    assert len(seen["images"]) == 4
    assert all(image.size == (3, 2) for image in seen["images"])

    with pytest.raises(ValueError, match="1–4 张.*收到 5 张"):
        MlxFibo2PromptEnhancerI2I().enhance(
            np.zeros((5, 2, 2, 3), dtype=np.float32), "too many", model="vlm"
        )


def test_nodes_read_and_record_memory_after_success(monkeypatch, tmp_path):
    store = Fibo2MemoryStore(tmp_path / "node-memory.json")
    monkeypatch.setattr(node_module, "MEMORY_STORE", store)
    calls: list[dict] = []

    def fake_enhance_prompt(**kwargs):
        calls.append(kwargs)
        return f"validated VGL {len(calls)}"

    monkeypatch.setattr(node_module, "enhance_prompt", fake_enhance_prompt)
    t2i = MlxFibo2PromptEnhancerT2I()
    t2i.enhance(
        "first request",
        context_turns=2,
        save_memory=True,
        unique_id="node-42",
    )
    t2i.enhance(
        "follow-up request",
        context_turns=0,
        save_memory=False,
        unique_id="node-42",
    )
    MlxFibo2PromptEnhancerI2I().enhance(
        np.zeros((1, 2, 2, 3), dtype=np.float32),
        "edit request",
        context_turns=2,
        save_memory=False,
        unique_id="node-i2i",
    )

    assert calls[0]["context"] == []
    assert calls[0]["memories"] == []
    assert calls[1]["context"] == []
    assert calls[1]["memories"][0]["prompt"] == "first request"
    assert calls[2]["context"] == []
    assert [item["prompt"] for item in store.get_context("node-42")] == [
        "first request",
        "follow-up request",
    ]
    assert [item["prompt"] for item in store.get_memories("node-42")] == [
        "first request"
    ]
    assert store.get_context("node-i2i")[0]["mode"] == enhancer.MODE_I2I
    assert store.get_context("node-i2i")[0]["has_image"] is True


def test_node_does_not_record_failed_enhancement(monkeypatch, tmp_path):
    store = Fibo2MemoryStore(tmp_path / "node-memory.json")
    monkeypatch.setattr(node_module, "MEMORY_STORE", store)

    def failed_enhance_prompt(**kwargs):
        raise RuntimeError("service failed")

    monkeypatch.setattr(node_module, "enhance_prompt", failed_enhance_prompt)
    with pytest.raises(RuntimeError, match="service failed"):
        MlxFibo2PromptEnhancerT2I().enhance("request", unique_id="node-failed")
    assert store.get_context("node-failed") == []
    assert store.get_memories("node-failed") == []


def test_frontend_sync_targets_text_encoder_prompt_widget():
    source = (ROOT / "web" / "fibo2_prompt_sync.js").read_text(encoding="utf-8")
    assert '"MlxFibo2PromptEnhancerT2I"' in source
    assert '"MlxFibo2PromptEnhancerI2I"' in source
    assert "output?.vgl_prompt" in source
    assert "target?.comfyClass !== TEXT_ENCODER" in source
    assert 'input?.name !== "prompt"' in source
    assert 'item?.name === "text"' in source
    assert "widget.value = vglPrompt" in source
    assert 'const CLEAR_MEMORY_ENDPOINT = "/mlx_gen/fibo2_memory/clear"' in source
    assert 'clearNodeMemory(node, "context"' in source
    assert 'clearNodeMemory(node, "memory"' in source