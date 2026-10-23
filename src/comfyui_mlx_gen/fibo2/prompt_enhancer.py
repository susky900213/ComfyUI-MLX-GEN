"""FIBO 2.0 VGL prompt enhancement through an OpenAI-compatible LLM.

The image model itself consumes the returned JSON string through the normal
``MlxTextEncoder`` path.  This module only asks a configurable text/VLM service
to translate a user request (and, for edits, reference images) into Bria's VGL
schema; it never calls Bria's hosted image-generation API.
"""

from __future__ import annotations

import base64
import io
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 4096
FIBO2_MAX_IMAGES = 4

MODE_T2I = "t2i"
MODE_I2I = "i2i"
MODES = (MODE_T2I, MODE_I2I)

TOP_LEVEL_FIELDS = (
    "short_description",
    "objects",
    "background_setting",
    "lighting",
    "aesthetics",
    "photographic_characteristics",
    "style_medium",
    "artistic_style",
    "context",
    "text_render",
    "edit_instruction",
)
OBJECT_FIELDS = (
    "description",
    "location",
    "relationship",
    "relative_size",
    "shape_and_color",
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
)
OBJECT_REQUIRED_STRINGS = (
    "description",
    "location",
    "relationship",
    "relative_size",
    "shape_and_color",
)
OBJECT_NULLABLE_STRINGS = (
    "texture",
    "appearance_details",
    "pose",
    "expression",
    "clothing",
    "action",
    "gender",
    "skin_tone_and_texture",
    "orientation",
)
TEXT_RENDER_FIELDS = ("text", "location", "size", "color", "font", "appearance_details")

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json|text|plaintext)?\s*|\s*```$", re.IGNORECASE)

_SCHEMA_INSTRUCTIONS = """Return exactly one JSON object with exactly these top-level fields:
{
  "short_description": "concise complete target image description (max 200 words)",
  "objects": [
    {
      "description": "detailed object description (max 100 words)",
      "location": "position in the frame",
      "relationship": "relationship to other objects",
      "relative_size": "size within the frame",
      "shape_and_color": "basic shape and dominant color",
      "texture": "surface texture or null",
      "appearance_details": "notable details or null",
      "number_of_objects": null,
      "pose": null,
      "expression": null,
      "clothing": null,
      "action": null,
      "gender": null,
      "skin_tone_and_texture": null,
      "orientation": "orientation or null"
    }
  ],
  "background_setting": "environment and background",
  "lighting": {
    "conditions": "lighting conditions",
    "direction": "lighting direction",
    "shadows": "shadow quality or null"
  },
  "aesthetics": {
    "composition": "composition and framing",
    "color_scheme": "dominant palette",
    "mood_atmosphere": "mood"
  },
  "photographic_characteristics": {
    "depth_of_field": "depth of field",
    "focus": "focus description",
    "camera_angle": "camera angle",
    "lens_focal_length": "lens/focal length"
  },
  "style_medium": "photograph, oil painting, watercolor, 3D render, etc.",
  "artistic_style": "short style description",
  "context": "image type or intended purpose",
  "text_render": [
    {
      "text": "exact user-provided visible text",
      "location": "location",
      "size": "size",
      "color": "color",
      "font": "font description",
      "appearance_details": "other appearance details or null"
    }
  ],
  "edit_instruction": "one imperative generation/edit command"
}

Schema rules:
- Include at most 5 objects. Include every object field shown above. Use null for
  irrelevant human-only fields and for other nullable fields; never use a made-up
  value merely to avoid null.
- ``number_of_objects`` is a positive integer only for an object cluster; otherwise null.
- ``photographic_characteristics`` may be null only when camera concepts truly do not apply.
- Use ``style_medium`` = "photograph" by default and ``artistic_style`` = "realistic"
  for photographs unless the request or reference clearly says otherwise.
- Keep ``text_render`` empty unless exact visible text was explicitly supplied by
  the user or is clearly present and meant to be preserved. Never invent copy or placeholders.
- Preserve exact spelling, capitalization, punctuation, and line breaks of requested visible text.
- Return valid JSON only: no Markdown fence, commentary, title, or reasoning.
"""


def completions_url(base_url: str) -> str:
    """Normalize a host, ``/v1`` URL, or full URL to chat completions."""

    value = str(base_url or "").strip().rstrip("/")
    if not value:
        raise ValueError("FIBO 2.0 提示词增强的服务地址不能为空")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def build_system_prompt(mode: str) -> str:
    """Build the VGL conversion instructions for generation or editing."""

    if mode not in MODES:
        raise ValueError(f"未知 FIBO 2.0 提示词模式：{mode!r}")
    common = """You are a professional Visual Generation Language (VGL) prompt writer
for the locally hosted Bria FIBO 2.0 image model. Translate the user's request into
one precise, self-contained VGL description of the image that should be generated.
Do not call an image API and do not wrap the result in a `structured_prompt` property.
Preserve the user's intent and exact requested visible text. Do not introduce important
subjects, brands, logos, or words that were not requested or visible in a reference.

""" + _SCHEMA_INSTRUCTIONS
    if mode == MODE_T2I:
        return common + """
Generation-mode rules:
- Describe the requested final image, resolving only minor visual ambiguity needed
  to produce a coherent scene.
- The user may supply one optional image as visual context. Inspect it carefully and
  use its relevant subjects, appearance, composition, colors, or style only as directed
  by the generation request. This remains generation mode, not an image-edit operation.
- Start `edit_instruction` with an imperative generation verb, for example
  "Generate ..." or "Create ...".
"""
    return common + """
Edit-mode rules:
- Inspect every supplied reference image. Images are ordered and named `<image_1>`
  through `<image_4>` in the user message; use those exact identifiers whenever the
  instruction assigns a role to a particular image.
- Describe the complete intended result after the edit, not just the changed fragment.
- The user's explicit edit request has priority. Change only what it requires. Preserve
  all other visible subject identities, poses, object locations and sizes, composition,
  camera/lens traits, lighting, style, materials, and fine details.
- With multiple references, infer each image's requested role from the instruction and
  preserve unambiguous `<image_N>` references in `edit_instruction` when needed.
- `edit_instruction` must be one faithful imperative command. Never refer vaguely to
  "the original image"; refer to `<image_N>` when a source must be named.
- This is standard edit mode without a mask. Do not mention masks or masked regions.
"""


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"FIBO VGL 的 {label} 必须是非空字符串")
    return value.strip()


def _nullable_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"FIBO VGL 的 {label} 必须是字符串或 null")
    return value.strip() or None


def _validate_object(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"FIBO VGL 的 objects[{index}] 必须是对象")
    result: dict[str, Any] = {}
    for field in OBJECT_REQUIRED_STRINGS:
        result[field] = _nonempty(value.get(field), f"objects[{index}].{field}")
    for field in OBJECT_NULLABLE_STRINGS:
        result[field] = _nullable_string(value.get(field), f"objects[{index}].{field}")
    count = value.get("number_of_objects")
    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 1):
        raise ValueError(
            f"FIBO VGL 的 objects[{index}].number_of_objects 必须是正整数或 null"
        )
    result["number_of_objects"] = count
    return {field: result[field] for field in OBJECT_FIELDS}


def _validate_string_object(
    value: Any,
    label: str,
    fields: Sequence[str],
    *,
    nullable: Sequence[str] = (),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"FIBO VGL 的 {label} 必须是对象")
    nullable_set = set(nullable)
    return {
        field: (
            _nullable_string(value.get(field), f"{label}.{field}")
            if field in nullable_set
            else _nonempty(value.get(field), f"{label}.{field}")
        )
        for field in fields
    }


def validate_vgl(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one object against the official VGL field contract."""

    if not isinstance(value, Mapping):
        raise ValueError("FIBO VGL 顶层必须是 JSON 对象")
    missing = [field for field in TOP_LEVEL_FIELDS if field not in value]
    if missing:
        raise ValueError(f"FIBO VGL 缺少顶层字段：{', '.join(missing)}")

    objects = value["objects"]
    if not isinstance(objects, list):
        raise ValueError("FIBO VGL 的 objects 必须是数组")
    if len(objects) > 5:
        raise ValueError("FIBO VGL 的 objects 最多包含 5 项")

    text_render = value["text_render"]
    if not isinstance(text_render, list):
        raise ValueError("FIBO VGL 的 text_render 必须是数组")

    photographic = value["photographic_characteristics"]
    if photographic is not None:
        photographic = _validate_string_object(
            photographic,
            "photographic_characteristics",
            ("depth_of_field", "focus", "camera_angle", "lens_focal_length"),
        )

    result = {
        "short_description": _nonempty(value["short_description"], "short_description"),
        "objects": [_validate_object(item, index) for index, item in enumerate(objects)],
        "background_setting": _nonempty(value["background_setting"], "background_setting"),
        "lighting": _validate_string_object(
            value["lighting"],
            "lighting",
            ("conditions", "direction", "shadows"),
            nullable=("shadows",),
        ),
        "aesthetics": _validate_string_object(
            value["aesthetics"],
            "aesthetics",
            ("composition", "color_scheme", "mood_atmosphere"),
        ),
        "photographic_characteristics": photographic,
        "style_medium": _nonempty(value["style_medium"], "style_medium"),
        "artistic_style": _nonempty(value["artistic_style"], "artistic_style"),
        "context": _nonempty(value["context"], "context"),
        "text_render": [
            _validate_string_object(
                item,
                f"text_render[{index}]",
                TEXT_RENDER_FIELDS,
                nullable=("appearance_details",),
            )
            for index, item in enumerate(text_render)
        ],
        "edit_instruction": _nonempty(value["edit_instruction"], "edit_instruction"),
    }
    return {field: result[field] for field in TOP_LEVEL_FIELDS}


def image_to_data_url(image: Any) -> str:
    """Encode one PIL-compatible reference image as a PNG data URL."""

    if image is None or not hasattr(image, "convert") or not hasattr(image, "save"):
        raise ValueError("FIBO 2.0 提示词增强的参考图必须是 PIL 图片")
    try:
        rgb = image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise ValueError("FIBO 2.0 提示词增强无法编码参考图") from exc
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_request_payload(
    *,
    model: str,
    mode: str,
    prompt: str,
    image_urls: Sequence[str] = (),
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    context: Sequence[Mapping[str, Any]] = (),
    memories: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a non-streaming OpenAI-compatible chat-completions body."""

    if mode not in MODES:
        raise ValueError(f"未知 FIBO 2.0 提示词模式：{mode!r}")
    model_name = str(model or "").strip()
    request_text = str(prompt or "").strip()
    if not model_name:
        raise ValueError("FIBO 2.0 提示词增强的 model 不能为空")
    if not request_text:
        raise ValueError("FIBO 2.0 提示词增强的 prompt 不能为空")
    temperature = float(temperature)
    top_p = float(top_p)
    max_tokens = int(max_tokens)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature 必须在 0.0 到 2.0 之间")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p 必须在 0.0（不含）到 1.0 之间")
    if max_tokens < 1:
        raise ValueError("max_tokens 必须大于 0")

    urls = [str(url or "").strip() for url in image_urls]
    if any(not url for url in urls):
        raise ValueError("FIBO 2.0 提示词增强包含空的参考图 URL")
    if mode == MODE_T2I and len(urls) > 1:
        raise ValueError("FIBO 2.0 文生图提示词增强最多支持 1 张参考图")
    if mode == MODE_I2I:
        if not urls:
            raise ValueError("FIBO 2.0 图生图提示词增强至少需要一张参考图")
        if len(urls) > FIBO2_MAX_IMAGES:
            raise ValueError(
                f"FIBO 2.0 图生图提示词增强最多支持 {FIBO2_MAX_IMAGES} 张参考图"
            )

    if mode == MODE_T2I:
        user_content: str | list[dict[str, Any]] = "User generation request:\n" + request_text
        if urls:
            user_content = [
                {
                    "type": "text",
                    "text": (
                        f"User generation request:\n{request_text}\n\n"
                        "Optional visual reference <image_1> follows. Use it only as visual "
                        "context for the requested new image; this is not an edit request."
                    ),
                },
                {"type": "text", "text": "<image_1>"},
                {"type": "image_url", "image_url": {"url": urls[0]}},
            ]
    else:
        references = ", ".join(f"<image_{index}>" for index in range(1, len(urls) + 1))
        user_content = [
            {
                "type": "text",
                "text": (
                    f"User edit request:\n{request_text}\n\n"
                    f"Ordered reference images: {references}. "
                    "Each label immediately precedes its image."
                ),
            }
        ]
        for index, url in enumerate(urls, 1):
            user_content.extend(
                [
                    {"type": "text", "text": f"<image_{index}>"},
                    {"type": "image_url", "image_url": {"url": url}},
                ]
            )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(mode)}
    ]

    memory_lines: list[str] = []
    for item in memories:
        if not isinstance(item, Mapping):
            continue
        remembered_prompt = str(item.get("prompt") or "").strip()
        remembered_result = str(item.get("enhanced_prompt") or "").strip()
        if not remembered_prompt and not remembered_result:
            continue
        remembered_mode = str(item.get("mode") or "unknown").strip()
        used_images = "yes" if item.get("has_image") else "no"
        memory_lines.append(
            f"- Mode: {remembered_mode}; used reference images: {used_images}\n"
            f"  User request: {remembered_prompt}\n"
            f"  Previous validated VGL: {remembered_result}"
        )
    if memory_lines:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Persistent FIBO 2.0 memory explicitly saved by the user. Use it only "
                    "for preferences and continuity when relevant; do not copy an old scene "
                    "unless the current request asks for it. Prior reference images are not "
                    "available in this turn, and old <image_N> labels never identify current "
                    "images:\n" + "\n".join(memory_lines)
                ),
            }
        )

    for item in context:
        if not isinstance(item, Mapping):
            continue
        previous_prompt = str(item.get("prompt") or "").strip()
        previous_result = str(item.get("enhanced_prompt") or "").strip()
        if not previous_prompt or not previous_result:
            continue
        previous_mode = str(item.get("mode") or "unknown").strip()
        image_note = (
            " Reference images were used then but are not available now; old <image_N> "
            "labels do not refer to current images."
            if item.get("has_image")
            else ""
        )
        messages.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f"Earlier FIBO 2.0 request (mode: {previous_mode}).{image_note}\n"
                        + previous_prompt
                    ),
                },
                {"role": "assistant", "content": previous_result},
            ]
        )
    messages.append({"role": "user", "content": user_content})

    return {
        "model": model_name,
        "messages": messages,
        "n": 1,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


def _error_detail(raw: bytes) -> str:
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
    except (TypeError, ValueError, UnicodeError):
        return raw.decode("utf-8", errors="replace").strip()[:500]
    if isinstance(parsed, Mapping):
        error = parsed.get("error")
        if isinstance(error, Mapping):
            detail = error.get("message") or error.get("detail")
            if detail:
                return str(detail)
        detail = parsed.get("detail") or parsed.get("message")
        if detail:
            return str(detail)
    return str(parsed)[:500]


def request_completion(
    *,
    base_url: str,
    payload: Mapping[str, Any],
    timeout: float = 120.0,
    api_key: str = "",
) -> dict[str, Any]:
    """Call an OpenAI-compatible endpoint and decode its JSON response."""

    timeout = float(timeout)
    if timeout <= 0:
        raise ValueError("FIBO 2.0 提示词增强的 timeout 必须大于 0")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    endpoint = completions_url(base_url)
    request = Request(
        endpoint,
        data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raise RuntimeError(
            f"FIBO 2.0 提示词服务返回 HTTP {exc.code}: {_error_detail(exc.read())}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 FIBO 2.0 提示词服务 {endpoint}：{exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"FIBO 2.0 提示词服务请求超时（{timeout:g} 秒）") from exc
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("FIBO 2.0 提示词服务返回的不是有效 JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("FIBO 2.0 提示词服务响应的 JSON 顶层必须是对象")
    return result


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def extract_completion_text(response: Mapping[str, Any]) -> str:
    """Extract assistant content from common OpenAI-compatible response shapes."""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise RuntimeError("FIBO 2.0 提示词服务响应缺少 choices[0]")
    choice = choices[0]
    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if content is None:
        content = choice.get("text")
    text = _content_text(content).strip()
    if not text:
        raise RuntimeError("FIBO 2.0 提示词服务没有返回 assistant 内容")
    return text


def parse_vgl_response(text: str) -> dict[str, Any]:
    """Remove common wrappers, extract one JSON object, and validate VGL."""

    value = str(text or "").strip()
    value = _THINK_BLOCK_RE.sub("", value)
    value = _OPEN_THINK_RE.sub("", value)
    if "<|im_end|>" in value:
        value = value.split("<|im_end|>", 1)[0]
    value = _FENCE_RE.sub("", value.strip()).strip()
    if not value:
        raise ValueError("FIBO 2.0 提示词服务没有返回最终 VGL JSON")
    start = value.find("{")
    if start < 0:
        raise ValueError("FIBO 2.0 提示词服务返回内容中没有 JSON 对象")
    try:
        parsed, _ = json.JSONDecoder().raw_decode(value[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("FIBO 2.0 提示词服务返回的 VGL 不是有效 JSON 对象") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("FIBO VGL 顶层必须是 JSON 对象")
    return validate_vgl(parsed)


def format_vgl_response(text: str) -> str:
    """Return a stable, directly consumable VGL JSON string."""

    return json.dumps(parse_vgl_response(text), ensure_ascii=False, indent=2)


def enhance_prompt(
    *,
    prompt: str,
    mode: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    images: Sequence[Any] = (),
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 120.0,
    api_key: str = "",
    context: Sequence[Mapping[str, Any]] = (),
    memories: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Translate one request into validated VGL using the configured LLM/VLM."""

    image_urls = [image_to_data_url(image) for image in images]
    payload = build_request_payload(
        model=model,
        mode=mode,
        prompt=prompt,
        image_urls=image_urls,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        context=context,
        memories=memories,
    )
    response = request_completion(
        base_url=base_url,
        payload=payload,
        timeout=timeout,
        api_key=api_key,
    )
    return format_vgl_response(extract_completion_text(response))


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "FIBO2_MAX_IMAGES",
    "MODE_I2I",
    "MODE_T2I",
    "build_request_payload",
    "build_system_prompt",
    "completions_url",
    "enhance_prompt",
    "extract_completion_text",
    "format_vgl_response",
    "image_to_data_url",
    "parse_vgl_response",
    "request_completion",
    "validate_vgl",
]