"""ComfyUI nodes that convert FIBO 2.0 requests into validated VGL JSON."""

from __future__ import annotations

from .. import image as image_util
from ..fibo2.memory import MAX_CONTEXT_TURNS, Fibo2MemoryStore, register_memory_routes
from ..fibo2.prompt_enhancer import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    FIBO2_MAX_IMAGES,
    MODE_I2I,
    MODE_T2I,
    enhance_prompt,
)


MEMORY_STORE = Fibo2MemoryStore()
register_memory_routes(MEMORY_STORE)


def _vgl_result(value: str) -> dict:
    """Expose VGL through both the STRING socket and ComfyUI execution UI data."""

    return {"ui": {"vgl_prompt": [value]}, "result": (value,)}


def _service_inputs() -> dict:
    return {
        "model": (
            "STRING",
            {
                "default": DEFAULT_MODEL,
                "multiline": False,
                "dynamicPrompts": False,
                "tooltip": "OpenAI-compatible 服务中的文本/视觉语言模型名称",
            },
        ),
        "base_url": (
            "STRING",
            {
                "default": DEFAULT_BASE_URL,
                "multiline": False,
                "dynamicPrompts": False,
                "tooltip": "可填 host、/v1 或完整 /chat/completions 地址",
            },
        ),
        "api_key": (
            "STRING",
            {
                "default": "",
                "multiline": False,
                "dynamicPrompts": False,
                "tooltip": "本地服务通常留空；服务要求鉴权时填写 Bearer token",
            },
        ),
        "temperature": (
            "FLOAT",
            {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05},
        ),
        "top_p": (
            "FLOAT",
            {"default": 0.95, "min": 0.01, "max": 1.0, "step": 0.01},
        ),
        "max_tokens": (
            "INT",
            {"default": DEFAULT_MAX_TOKENS, "min": 1, "max": 32768},
        ),
        "timeout": (
            "FLOAT",
            {"default": 120.0, "min": 1.0, "max": 1800.0, "step": 1.0},
        ),
        "context_turns": (
            "INT",
            {
                "default": 6,
                "min": 0,
                "max": MAX_CONTEXT_TURNS,
                "step": 1,
                "tooltip": "自动带入该节点最近几轮 FIBO 请求；0 表示不发送上下文",
            },
        ),
        "save_memory": (
            "BOOLEAN",
            {
                "default": True,
                "label_on": "保存本次结果",
                "label_off": "不保存",
                "tooltip": "将本次文字请求和 VGL 结果保存到该节点的长期记忆",
            },
        ),
    }


class MlxFibo2PromptEnhancerT2I:
    """Convert a text-to-image request into FIBO 2.0 VGL JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": (
                "STRING",
                {
                    "default": "A cinematic portrait of an astronaut in a greenhouse on Mars",
                    "multiline": True,
                    "dynamicPrompts": False,
                },
            )
        }
        required.update(_service_inputs())
        return {
            "required": required,
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "可选单张视觉参考图；仅帮助 VLM 编写文生图 VGL，"
                            "不会切换为图生图编辑"
                        )
                    },
                )
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("vgl_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "MLX/Prompt"

    def enhance(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 120.0,
        image=None,
        context_turns: int = 6,
        save_memory: bool = True,
        unique_id=None,
    ):
        reference_images = ()
        if image is not None:
            reference_images = image_util.to_pil_batch(image)
            count = len(reference_images)
            if count != 1:
                raise ValueError(
                    f"FIBO 2.0 文生图提示词增强的可选参考图必须是 1 张，收到 {count} 张"
                )
        node_id = str(unique_id or "").strip()
        context = MEMORY_STORE.get_context(node_id, context_turns) if node_id else []
        memories = MEMORY_STORE.get_memories(node_id) if node_id else []
        value = enhance_prompt(
            prompt=str(prompt or ""),
            mode=MODE_T2I,
            images=reference_images,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            context=context,
            memories=memories,
        )
        if node_id:
            MEMORY_STORE.record_turn(
                node_id,
                prompt=str(prompt or ""),
                enhanced_prompt=value,
                mode=MODE_T2I,
                has_image=bool(reference_images),
                save_memory=bool(save_memory),
            )
        return _vgl_result(value)


class MlxFibo2PromptEnhancerI2I:
    """Convert an edit request and one-to-four ordered images into FIBO VGL."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "images": (
                "IMAGE",
                {
                    "tooltip": (
                        "1–4 张有序参考图（IMAGE 批次）；在指令中可用 "
                        "<image_1>…<image_4> 引用"
                    )
                },
            ),
            "prompt": (
                "STRING",
                {
                    "default": "Change the subject's jacket to red and preserve everything else",
                    "multiline": True,
                    "dynamicPrompts": False,
                },
            ),
        }
        required.update(_service_inputs())
        return {"required": required, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("vgl_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "MLX/Prompt"

    def enhance(
        self,
        images,
        prompt: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 120.0,
        context_turns: int = 6,
        save_memory: bool = True,
        unique_id=None,
    ):
        reference_images = image_util.to_pil_batch(images)
        count = len(reference_images)
        if not 1 <= count <= FIBO2_MAX_IMAGES:
            raise ValueError(
                f"FIBO 2.0 图生图提示词增强需要 1–{FIBO2_MAX_IMAGES} 张参考图，收到 {count} 张"
            )
        node_id = str(unique_id or "").strip()
        context = MEMORY_STORE.get_context(node_id, context_turns) if node_id else []
        memories = MEMORY_STORE.get_memories(node_id) if node_id else []
        value = enhance_prompt(
            prompt=str(prompt or ""),
            mode=MODE_I2I,
            images=reference_images,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            context=context,
            memories=memories,
        )
        if node_id:
            MEMORY_STORE.record_turn(
                node_id,
                prompt=str(prompt or ""),
                enhanced_prompt=value,
                mode=MODE_I2I,
                has_image=True,
                save_memory=bool(save_memory),
            )
        return _vgl_result(value)


__all__ = ["MlxFibo2PromptEnhancerI2I", "MlxFibo2PromptEnhancerT2I", "_vgl_result"]