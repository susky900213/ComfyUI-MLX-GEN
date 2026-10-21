"""MOSS prompt enhancer tests without requiring a running local LLM."""

from __future__ import annotations

import json
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen import NODE_CLASS_MAPPINGS  # noqa: E402
from comfyui_mlx_gen.moss.memory import MossMemoryStore  # noqa: E402
from comfyui_mlx_gen.moss import prompt_enhancer as enhancer  # noqa: E402
from comfyui_mlx_gen.nodes.moss_prompt_enhancer import (  # noqa: E402
    MlxMossPromptEnhancer,
    _enhanced_prompt_result,
)
from comfyui_mlx_gen.nodes import moss_prompt_enhancer as moss_node  # noqa: E402


def _valid_layout() -> dict:
    return {
        "summary": {
            "meta": ["poster"],
            "canvas": "single",
            "caption": "A minimalist travel poster with a blue mountain.",
        },
        "items": [
            {
                "label": "text",
                "caption": "Large title centered at the top.",
                "text": "EXPLORE",
                "bbox": [-416, -384, -288, 384],
            }
        ],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


def test_urls_payload_and_mode_instructions_follow_moss_contract():
    assert enhancer.completions_url("http://127.0.0.1:8001") == (
        "http://127.0.0.1:8001/v1/chat/completions"
    )
    assert enhancer.completions_url("http://127.0.0.1:8001/v1/") == (
        "http://127.0.0.1:8001/v1/chat/completions"
    )
    assert enhancer.completions_url("http://127.0.0.1:8001/v1/chat/completions") == (
        "http://127.0.0.1:8001/v1/chat/completions"
    )

    payload = enhancer.build_request_payload(
        model="local-moss-enhancer",
        mode=enhancer.MODE_JSON,
        prompt="a travel poster with EXPLORE",
        negative_prompt="bad anatomy",
        canvas_width=1280,
        canvas_height=768,
    )
    assert payload["model"] == "local-moss-enhancer"
    assert payload["stream"] is False
    assert payload["messages"][1]["content"].endswith("Target canvas: 1280x768 pixels.")
    system = payload["messages"][0]["content"]
    tag_system = enhancer.build_request_payload(
        model="local-moss-enhancer",
        mode=enhancer.MODE_TAGS,
        prompt="a travel poster with EXPLORE",
    )["messages"][0]["content"]
    assert "Quality -> Score -> Meta -> Year -> Character -> Series ->" in tag_system
    assert "bbox_coord_space" in system
    assert "1280x768" in system


def test_reference_image_is_encoded_as_openai_multimodal_content():
    image_url = enhancer.image_to_data_url(Image.new("RGB", (2, 1), (12, 34, 56)))
    assert image_url.startswith("data:image/png;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")

    payload = enhancer.build_request_payload(
        model="local-moss-enhancer",
        mode=enhancer.MODE_NATURAL_LANGUAGE,
        prompt="make the scene cinematic",
        image_url=image_url,
    )
    content = payload["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "User's image request:\nmake the scene cinematic"}
    assert content[1] == {"type": "image_url", "image_url": {"url": image_url}}


def test_context_and_persistent_memory_are_added_to_the_request():
    payload = enhancer.build_request_payload(
        model="local-moss-enhancer",
        mode=enhancer.MODE_NATURAL_LANGUAGE,
        prompt="make it warmer",
        context=[{"prompt": "a blue mountain", "enhanced_prompt": "blue mountain at dawn"}],
        memories=[{"prompt": "prefer poster layouts", "enhanced_prompt": "clean centered typography"}],
    )
    messages = payload["messages"]
    assert "Persistent memory saved by the user" in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "Earlier image request:\na blue mountain"}
    assert messages[3] == {"role": "assistant", "content": "blue mountain at dawn"}
    assert messages[4]["content"].endswith("make it warmer")


def test_moss_memory_store_is_node_scoped_and_manually_clearable(tmp_path):
    store = MossMemoryStore(tmp_path / "moss.json", max_context_turns=2, max_memory_items=2)
    store.record_turn(
        "node-a",
        prompt="first",
        enhanced_prompt="first result",
        mode=enhancer.MODE_NATURAL_LANGUAGE,
        has_image=False,
        save_memory=True,
    )
    store.record_turn(
        "node-a",
        prompt="second",
        enhanced_prompt="second result",
        mode=enhancer.MODE_NATURAL_LANGUAGE,
        has_image=True,
        save_memory=False,
    )
    store.record_turn(
        "node-b",
        prompt="other",
        enhanced_prompt="other result",
        mode=enhancer.MODE_NATURAL_LANGUAGE,
        has_image=False,
        save_memory=True,
    )

    assert [item["prompt"] for item in store.get_context("node-a")] == ["first", "second"]
    assert [item["prompt"] for item in store.get_memories("node-a")] == ["first"]
    assert [item["prompt"] for item in store.get_context("node-b")] == ["other"]

    store.clear("node-a", "context")
    assert store.get_context("node-a") == []
    assert [item["prompt"] for item in store.get_memories("node-a")] == ["first"]
    store.clear("node-a", "memory")
    assert store.get_memories("node-a") == []


def test_completion_cleaning_supports_thinking_and_all_modes():
    assert enhancer.format_completion(
        "<think>private reasoning</think>\n```text\nA detailed scene\n```",
        enhancer.MODE_NATURAL_LANGUAGE,
    ) == "A detailed scene"
    assert enhancer.format_completion(
        "<think>private reasoning</think>\n```\nbest quality, highres, 1girl\n```",
        enhancer.MODE_TAGS,
    ) == "best quality, highres, 1girl"

    result = enhancer.format_completion(
        "<think>layout reasoning</think>\n```json\n"
        + json.dumps(_valid_layout())
        + "\n```",
        enhancer.MODE_JSON,
    )
    assert json.loads(result) == _valid_layout()


def test_layout_validation_rejects_wrong_coordinate_contract():
    broken = _valid_layout()
    broken["bbox_format"] = "xyxy"
    with pytest.raises(ValueError, match="bbox_format"):
        enhancer.validate_layout_json(broken)

    broken = _valid_layout()
    broken["items"][0].pop("text")
    with pytest.raises(ValueError, match="text"):
        enhancer.validate_layout_json(broken)


def test_request_completion_uses_openai_headers_and_payload(monkeypatch):
    seen: dict = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "<think>x</think>expanded"}}]}
            ).encode()

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = {key.casefold(): value for key, value in request.header_items()}
        seen["body"] = json.loads(request.data.decode())
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(enhancer, "urlopen", fake_urlopen)
    response = enhancer.request_completion(
        base_url="http://127.0.0.1:8001",
        payload={"model": "local", "messages": []},
        timeout=9,
        api_key="secret",
    )
    assert response["choices"][0]["message"]["content"] == "<think>x</think>expanded"
    assert seen["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert seen["headers"] == {
        "accept": "application/json",
        "authorization": "Bearer secret",
        "content-type": "application/json",
    }
    assert seen["body"] == {"model": "local", "messages": []}
    assert seen["timeout"] == 9.0


def test_node_is_registered_and_returns_string_result():
    assert NODE_CLASS_MAPPINGS["MlxMossPromptEnhancer"] is MlxMossPromptEnhancer
    assert MlxMossPromptEnhancer.RETURN_TYPES == ("STRING",)
    assert _enhanced_prompt_result("expanded") == {
        "ui": {"enhanced_prompt": ["expanded"]},
        "result": ("expanded",),
    }
    web_extension = ROOT / "web" / "moss_prompt_sync.js"
    web_source = web_extension.read_text(encoding="utf-8")
    assert 'const SOURCE_NODE = "MlxMossPromptEnhancer"' in web_source
    assert 'target?.comfyClass !== TEXT_ENCODER' in web_source
    assert 'input?.name !== "prompt"' in web_source
    assert 'item?.name === "text"' in web_source
    assert "widget.value = enhancedPrompt" in web_source
    assert 'const CLEAR_MEMORY_ENDPOINT = "/mlx_gen/moss_memory/clear"' in web_source
    assert 'clearNodeMemory(node, "context"' in web_source
    assert 'clearNodeMemory(node, "memory"' in web_source
    assert set(MlxMossPromptEnhancer.INPUT_TYPES()["required"]) >= {
        "prompt",
        "mode",
        "base_url",
        "model",
    }
    assert MlxMossPromptEnhancer.INPUT_TYPES()["optional"]["image"][0] == "IMAGE"
    assert "context_turns" in MlxMossPromptEnhancer.INPUT_TYPES()["required"]
    assert "save_memory" in MlxMossPromptEnhancer.INPUT_TYPES()["required"]
    assert MlxMossPromptEnhancer.INPUT_TYPES()["hidden"]["unique_id"] == "UNIQUE_ID"


def test_node_uses_first_reference_image_for_multimodal_request(monkeypatch):
    seen: dict = {}

    def fake_enhance_prompt(**kwargs):
        seen.update(kwargs)
        return "expanded from image"

    monkeypatch.setattr(moss_node, "enhance_prompt", fake_enhance_prompt)
    images = np.zeros((2, 2, 2, 3), dtype=np.float32)
    images[0, :, :, 0] = 1.0
    images[1, :, :, 2] = 1.0

    result = MlxMossPromptEnhancer().enhance(
        prompt="keep the red subject",
        image=images,
    )

    assert result == {
        "ui": {"enhanced_prompt": ["expanded from image"]},
        "result": ("expanded from image",),
    }
    assert seen["prompt"] == "keep the red subject"
    assert seen["image_url"].startswith("data:image/png;base64,")
    encoded = base64.b64decode(seen["image_url"].split(",", 1)[1])
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
    assert Image.open(io.BytesIO(encoded)).getpixel((0, 0)) == (255, 0, 0)


def test_node_reads_context_and_records_selected_long_term_memory(monkeypatch, tmp_path):
    store = MossMemoryStore(tmp_path / "node-memory.json")
    monkeypatch.setattr(moss_node, "MEMORY_STORE", store)
    calls: list[dict] = []

    def fake_enhance_prompt(**kwargs):
        calls.append(kwargs)
        return f"expanded {len(calls)}"

    monkeypatch.setattr(moss_node, "enhance_prompt", fake_enhance_prompt)
    node = MlxMossPromptEnhancer()
    node.enhance(
        prompt="first request",
        context_turns=2,
        save_memory=True,
        unique_id="node-42",
    )
    node.enhance(
        prompt="follow-up request",
        context_turns=2,
        save_memory=False,
        unique_id="node-42",
    )

    assert calls[0]["context"] == []
    assert calls[0]["memories"] == []
    assert calls[1]["context"][0]["prompt"] == "first request"
    assert calls[1]["context"][0]["enhanced_prompt"] == "expanded 1"
    assert calls[1]["memories"][0]["prompt"] == "first request"
    assert [item["prompt"] for item in store.get_memories("node-42")] == ["first request"]


def test_extract_completion_text_supports_text_fallback_and_rejects_empty():
    assert enhancer.extract_completion_text({"choices": [{"text": "expanded"}]}) == "expanded"
    with pytest.raises(RuntimeError, match="assistant 内容"):
        enhancer.extract_completion_text({"choices": [{"message": {"content": ""}}]})