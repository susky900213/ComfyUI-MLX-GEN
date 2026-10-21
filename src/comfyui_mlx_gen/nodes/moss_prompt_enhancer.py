"""ComfyUI node for enhancing MOSS-Image prompts through a local LLM."""

from __future__ import annotations

from ..moss.prompt_enhancer import (
    DEFAULT_BASE_URL,
    DEFAULT_CANVAS_HEIGHT,
    DEFAULT_CANVAS_WIDTH,
    DEFAULT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    MODE_JSON,
    MODE_NATURAL_LANGUAGE,
    MODE_TAGS,
    MODES,
    enhance_prompt,
    image_to_data_url,
)
from ..moss.memory import MAX_CONTEXT_TURNS, MossMemoryStore, register_memory_routes


MEMORY_STORE = MossMemoryStore()
register_memory_routes(MEMORY_STORE)


def _enhanced_prompt_result(prompt: str) -> dict:
    """Expose the result as both a STRING socket and a UI value."""

    return {"ui": {"enhanced_prompt": [prompt]}, "result": (prompt,)}


class MlxMossPromptEnhancer:
    """Use a local OpenAI-compatible model to prepare a MOSS image prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "a minimalist travel poster with a blue mountain and the title EXPLORE",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "mode": (list(MODES), {"default": MODE_NATURAL_LANGUAGE}),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_NEGATIVE_PROMPT,
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": DEFAULT_MODEL,
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": DEFAULT_BASE_URL,
                        "multiline": False,
                        "dynamicPrompts": False,
                        "tooltip": "OpenAI 兼容服务地址；可填 host、/v1 或完整 /chat/completions 地址",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "dynamicPrompts": False,
                        "tooltip": "本地服务通常留空；如果服务要求鉴权则填写 Bearer token",
                    },
                ),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 32768}),
                "timeout": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 1800.0, "step": 1.0}),
                "canvas_width": (
                    "INT",
                    {
                        "default": DEFAULT_CANVAS_WIDTH,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                        "tooltip": "只用于 JSON 布局模式的 bbox 坐标规划",
                    },
                ),
                "canvas_height": (
                    "INT",
                    {
                        "default": DEFAULT_CANVAS_HEIGHT,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                        "tooltip": "只用于 JSON 布局模式的 bbox 坐标规划",
                    },
                ),
                "context_turns": (
                    "INT",
                    {
                        "default": 6,
                        "min": 0,
                        "max": MAX_CONTEXT_TURNS,
                        "step": 1,
                        "tooltip": "自动带入最近几轮 MOSS 请求；0 表示不发送上下文",
                    },
                ),
                "save_memory": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "保存本次结果",
                        "label_off": "不保存",
                        "tooltip": "将本次文字请求和增强结果保存到该节点的长期记忆",
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"tooltip": "可选参考图；会与文字提示词一起发送给 MOSS"},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "MLX/Prompt"

    def enhance(
        self,
        prompt: str,
        mode: str = MODE_NATURAL_LANGUAGE,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        canvas_width: int = DEFAULT_CANVAS_WIDTH,
        canvas_height: int = DEFAULT_CANVAS_HEIGHT,
        image=None,
        context_turns: int = 6,
        save_memory: bool = True,
        unique_id=None,
    ):
        image_url = ""
        if image is not None:
            from .. import image as image_util

            reference_images = image_util.to_pil_batch(image)
            if not reference_images:
                raise ValueError("MOSS 提示词增强的参考图不能为空")
            image_url = image_to_data_url(reference_images[0])
        node_id = str(unique_id or "").strip()
        context = MEMORY_STORE.get_context(node_id, context_turns) if node_id else []
        memories = MEMORY_STORE.get_memories(node_id) if node_id else []
        enhanced = enhance_prompt(
            prompt=str(prompt or ""),
            mode=mode,
            model=model,
            base_url=base_url,
            negative_prompt=negative_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            image_url=image_url,
            context=context,
            memories=memories,
        )
        if node_id:
            MEMORY_STORE.record_turn(
                node_id,
                prompt=str(prompt or ""),
                enhanced_prompt=enhanced,
                mode=mode,
                has_image=bool(image_url),
                save_memory=bool(save_memory),
            )
        return _enhanced_prompt_result(enhanced)


__all__ = ["MlxMossPromptEnhancer", "_enhanced_prompt_result"]