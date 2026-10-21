"""MOSS prompt enhancement through an OpenAI-compatible local server.

The MOSS image model accepts natural-language prompts, tag sequences, and a
layout JSON format.  This module keeps the HTTP client deliberately small and
uses only the Python standard library so the prompt node does not add another
runtime dependency to ComfyUI-MLX-GEN.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "MOSS-Image-Base"
DEFAULT_NEGATIVE_PROMPT = "worst quality, bad anatomy, watermark, jpeg artifact"
DEFAULT_CANVAS_WIDTH = 1024
DEFAULT_CANVAS_HEIGHT = 1024

MODE_NATURAL_LANGUAGE = "自然语言"
MODE_TAGS = "标签"
MODE_JSON = "JSON 布局"
MODES = (MODE_NATURAL_LANGUAGE, MODE_TAGS, MODE_JSON)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json|text|plaintext)?\s*|\s*```$", re.IGNORECASE)


def completions_url(base_url: str) -> str:
    """Normalize a server URL to an OpenAI chat-completions endpoint.

    The node accepts any of these useful forms::

        http://127.0.0.1:8001
        http://127.0.0.1:8001/v1
        http://127.0.0.1:8001/v1/chat/completions
    """

    value = str(base_url or "").strip().rstrip("/")
    if not value:
        raise ValueError("MOSS 提示词增强的服务地址不能为空")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def build_system_prompt(
    mode: str,
    *,
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    canvas_height: int = DEFAULT_CANVAS_HEIGHT,
) -> str:
    """Build the mode-specific MOSS enhancement instructions."""

    if mode not in MODES:
        raise ValueError(f"未知 MOSS 提示词模式：{mode!r}；可选：{', '.join(MODES)}")

    common = f"""You are a professional prompt enhancer for MOSS-Image-Base.
MOSS supports natural-language descriptions, comma-separated tag sequences, and
structured layout JSON. Preserve the user's intended subjects, actions, text,
and composition. Improve concrete visual details such as composition, camera,
lighting, color, materials, atmosphere, and artistic style, but do not invent
important characters, objects, logos, or readable text that the user did not
request. Keep any quoted or explicitly requested image text exactly unchanged.
Use English when it makes the visual description more precise, but preserve
Chinese or another requested language when it is part of the intended image.
When a reference image is provided, inspect it carefully and use its visible
subjects, composition, colors, lighting, style, and readable text as evidence.
Combine the image evidence with the user's written request; the user's explicit
instructions take priority, and do not invent details that are not visible or
requested.
The effective default negative tags for MOSS are:
{DEFAULT_NEGATIVE_PROMPT}
"""

    if mode == MODE_NATURAL_LANGUAGE:
        return common + """
Output rules for natural-language mode:
- Return exactly one polished image-generation prompt.
- Use one coherent paragraph; do not add a title, explanation, bullet list,
  Markdown fence, or a `negative prompt:` section.
- Describe the main subject and action first, then composition, environment,
  camera/viewpoint, lighting, color, materials, and style as appropriate.
"""
    if mode == MODE_TAGS:
        return common + """
Output rules for tag mode:
- Return only one comma-separated tag sequence, with no title, explanation,
  bullets, Markdown, or surrounding quotation marks.
- Prefer this order: Quality -> Score -> Meta -> Year -> Character -> Series ->
  Artist -> General. Put the most important subject tags near the end of that
  order and do not fabricate character, series, or artist names.
- Keep tags concise and avoid turning the result into prose sentences.
"""
    schema = """{
  "summary": {
    "meta": ["poster"],
    "canvas": "single",
    "caption": "A detailed description of the complete scene and style."
  },
  "items": [
    {
      "label": "object or text",
      "caption": "A detailed description of this item and its placement.",
      "text": "Optional exact text when label is text.",
      "bbox": [y_min, x_min, y_max, x_max]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}"""
    return common + f"""
Output rules for JSON layout mode:
- Return exactly one JSON object and nothing else. Do not use Markdown fences.
- The output schema is:
{schema}
- The canvas is {canvas_width}x{canvas_height} pixels. Coordinates are absolute
  pixels with the canvas center as origin: x increases rightward and y
  increases downward. Convert from top-left coordinates by subtracting half
  the width from x and half the height from y.
- Include text items when the user asks for visible text, put the exact text in
  `text`, and preserve line breaks when they matter. Use `bbox` order
  [y_min, x_min, y_max, x_max].
- Use a concise `summary.meta` list and `summary.canvas` equal to `single`.
"""


def build_user_prompt(
    prompt: str,
    mode: str,
    *,
    negative_prompt: str = "",
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    canvas_height: int = DEFAULT_CANVAS_HEIGHT,
) -> str:
    """Build the user message without allowing an empty enhancement request."""

    value = str(prompt or "").strip()
    if not value:
        raise ValueError("MOSS 提示词增强的 prompt 不能为空")
    if mode not in MODES:
        raise ValueError(f"未知 MOSS 提示词模式：{mode!r}；可选：{', '.join(MODES)}")

    negative = str(negative_prompt or "").strip()
    message = f"User's image request:\n{value}"
    if negative:
        message += f"\n\nNegative prompt reference (do not output it in the positive result):\n{negative}"
    if mode == MODE_JSON:
        message += f"\n\nTarget canvas: {int(canvas_width)}x{int(canvas_height)} pixels."
    return message


def build_request_payload(
    *,
    model: str,
    mode: str,
    prompt: str,
    negative_prompt: str = "",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    canvas_height: int = DEFAULT_CANVAS_HEIGHT,
    image_url: str = "",
    context: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
    memories: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
) -> dict[str, Any]:
    """Build a plain OpenAI-compatible ``/chat/completions`` request body."""

    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("MOSS 提示词增强的 model 不能为空")
    temperature = float(temperature)
    top_p = float(top_p)
    max_tokens = int(max_tokens)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature 必须在 0.0 到 2.0 之间")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p 必须在 0.0（不含）到 1.0 之间")
    if max_tokens < 1:
        raise ValueError("max_tokens 必须大于 0")
    canvas_width, canvas_height = int(canvas_width), int(canvas_height)
    if canvas_width < 16 or canvas_height < 16:
        raise ValueError("JSON 布局的画布宽高必须至少为 16")

    user_prompt = build_user_prompt(
        prompt,
        mode,
        negative_prompt=negative_prompt,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    image_url = str(image_url or "").strip()
    user_content: str | list[dict[str, Any]] = user_prompt
    if image_url:
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": build_system_prompt(
                mode,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            ),
        }
    ]

    memory_lines: list[str] = []
    for item in memories:
        if not isinstance(item, Mapping):
            continue
        remembered_prompt = str(item.get("prompt") or "").strip()
        remembered_result = str(item.get("enhanced_prompt") or "").strip()
        if not remembered_prompt and not remembered_result:
            continue
        memory_lines.append(
            f"- User request: {remembered_prompt}\n  Previous enhanced result: {remembered_result}"
        )
    if memory_lines:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Persistent memory saved by the user. Use it as a preference and continuity "
                    "reference, not as a requirement to copy an old scene:\n"
                    + "\n".join(memory_lines)
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
        messages.extend(
            [
                {
                    "role": "user",
                    "content": "Earlier image request:\n" + previous_prompt,
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


def image_to_data_url(image: Any) -> str:
    """Encode one PIL-compatible image as a PNG data URL for a VLM request."""

    if image is None or not hasattr(image, "save") or not hasattr(image, "convert"):
        raise ValueError("MOSS 提示词增强的参考图必须是 PIL 图片")
    try:
        rgb_image = image.convert("RGB")
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="PNG")
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise ValueError("MOSS 提示词增强无法编码参考图") from exc
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _error_detail(raw: bytes) -> str:
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
    except (TypeError, ValueError, UnicodeError):
        return raw.decode("utf-8", errors="replace").strip()[:500]
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("detail")
            if message:
                return str(message)
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
    """POST one non-streaming completion and return its decoded JSON object."""

    timeout = float(timeout)
    if timeout <= 0:
        raise ValueError("MOSS 提示词增强的 timeout 必须大于 0")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        completions_url(base_url),
        data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = _error_detail(exc.read())
        raise RuntimeError(f"MOSS 提示词服务返回 HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"无法连接 MOSS 提示词服务 {completions_url(base_url)}：{exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(f"MOSS 提示词服务请求超时（{timeout:g} 秒）") from exc

    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("MOSS 提示词服务返回的不是有效 JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("MOSS 提示词服务返回的 JSON 顶层必须是对象")
    return result


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def extract_completion_text(response: Mapping[str, Any]) -> str:
    """Extract assistant text from common OpenAI-compatible response shapes."""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("MOSS 提示词服务响应缺少 choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("MOSS 提示词服务响应的 choices[0] 无效")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if content is None:
        content = choice.get("text")
    value = _content_text(content).strip()
    if not value:
        raise RuntimeError("MOSS 提示词服务没有返回 assistant 内容")
    return value


def clean_text_response(text: str) -> str:
    """Remove reasoning markers and accidental Markdown fences."""

    value = str(text or "").strip()
    value = _THINK_BLOCK_RE.sub("", value)
    value = _OPEN_THINK_RE.sub("", value)
    if "<|im_end|>" in value:
        value = value.split("<|im_end|>", 1)[0]
    value = _FENCE_RE.sub("", value.strip()).strip()
    if not value:
        raise RuntimeError("MOSS 提示词服务只返回了思考内容，没有最终提示词")
    return value


def _parse_json_object(text: str) -> dict[str, Any]:
    value = clean_text_response(text)
    candidates = [value]
    start = value.find("{")
    if start >= 0:
        decoder = json.JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(value[start:])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            candidates.insert(0, parsed)

    for candidate in candidates:
        if isinstance(candidate, dict):
            parsed = candidate
        else:
            try:
                parsed = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("MOSS JSON 模式返回的内容不是有效 JSON 对象")


def validate_layout_json(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the layout contract documented by MOSS-Image-Base."""

    if not isinstance(value, Mapping):
        raise ValueError("MOSS JSON 布局顶层必须是对象")
    summary = value.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("MOSS JSON 布局必须包含 summary 对象")
    caption = summary.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("MOSS JSON 布局的 summary.caption 必须是非空字符串")
    meta = summary.get("meta")
    if not isinstance(meta, list) or not all(isinstance(item, str) for item in meta):
        raise ValueError("MOSS JSON 布局的 summary.meta 必须是字符串数组")
    if summary.get("canvas") != "single":
        raise ValueError('MOSS JSON 布局的 summary.canvas 必须为 "single"')

    items = value.get("items")
    if not isinstance(items, list):
        raise ValueError("MOSS JSON 布局必须包含 items 数组")
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"MOSS JSON 布局 items[{index}] 必须是对象")
        for key in ("label", "caption"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"MOSS JSON 布局 items[{index}].{key} 必须是非空字符串")
        bbox = item.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(number, (int, float)) and not isinstance(number, bool) for number in bbox)
        ):
            raise ValueError(f"MOSS JSON 布局 items[{index}].bbox 必须是四个数字")
        if item.get("label") == "text" and not isinstance(item.get("text"), str):
            raise ValueError(f"MOSS JSON 布局 items[{index}] 为 text 时必须包含 text 字段")

    if value.get("bbox_coord_space") != "abs_centralized":
        raise ValueError('MOSS JSON 布局的 bbox_coord_space 必须为 "abs_centralized"')
    if value.get("bbox_format") != "yxyx":
        raise ValueError('MOSS JSON 布局的 bbox_format 必须为 "yxyx"')
    return dict(value)


def format_completion(text: str, mode: str) -> str:
    """Clean one model completion and validate it for the selected mode."""

    if mode not in MODES:
        raise ValueError(f"未知 MOSS 提示词模式：{mode!r}；可选：{', '.join(MODES)}")
    if mode != MODE_JSON:
        return clean_text_response(text)
    parsed = _parse_json_object(text)
    return json.dumps(validate_layout_json(parsed), ensure_ascii=False, indent=2)


def enhance_prompt(
    *,
    prompt: str,
    mode: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    api_key: str = "",
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    canvas_height: int = DEFAULT_CANVAS_HEIGHT,
    image_url: str = "",
    context: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
    memories: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
) -> str:
    """Enhance a prompt by calling the configured local OpenAI-compatible API."""

    payload = build_request_payload(
        model=model,
        mode=mode,
        prompt=prompt,
        negative_prompt=negative_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        image_url=image_url,
        context=context,
        memories=memories,
    )
    response = request_completion(
        base_url=base_url,
        payload=payload,
        timeout=timeout,
        api_key=api_key,
    )
    return format_completion(extract_completion_text(response), mode)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_CANVAS_HEIGHT",
    "DEFAULT_CANVAS_WIDTH",
    "DEFAULT_MODEL",
    "DEFAULT_NEGATIVE_PROMPT",
    "MODE_JSON",
    "MODE_NATURAL_LANGUAGE",
    "MODE_TAGS",
    "MODES",
    "build_request_payload",
    "build_system_prompt",
    "build_user_prompt",
    "clean_text_response",
    "completions_url",
    "enhance_prompt",
    "extract_completion_text",
    "image_to_data_url",
    "format_completion",
    "request_completion",
    "validate_layout_json",
]