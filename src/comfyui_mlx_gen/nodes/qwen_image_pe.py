"""ComfyUI nodes for Qwen-Image 2.1 T2I/I2I prompt enhancement."""

from __future__ import annotations

from ..qwen_image_pe import (
    BACKEND,
    LOADER_OPTIONS,
    MLX_VLM_4BIT,
    ensure_optional_dependencies,
    model_options,
)


def _rewritten_prompt_result(prompt: str) -> dict:
    """Return the STRING result plus the value used by the canvas extension.

    ComfyUI does not expose the value of a normal STRING socket to the canvas
    after execution.  Keep the regular ``result`` tuple for downstream nodes
    and mirror the same value in ``ui`` so the PE result can be shown in the
    connected MlxTextEncoder widget.
    """
    return {"ui": {"rewritten_prompt": [prompt]}, "result": (prompt,)}


class MlxQwenImagePET2I:
    @classmethod
    def INPUT_TYPES(cls):
        options = model_options("t2i")
        return {
            "required": {
                # Scan text_encoder/ and expose the discovered PE checkpoints as a dropdown.
                "model_path": (options, {"default": options[0]}),
                "prompt": (
                    "STRING",
                    {
                        "default": "a red cube on a wooden table",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "loader_type": (list(LOADER_OPTIONS), {"default": MLX_VLM_4BIT}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("rewritten_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "MLX/Prompt"

    def enhance(self, model_path: str, prompt: str, loader_type: str = MLX_VLM_4BIT):
        if not str(prompt or "").strip():
            raise ValueError("Qwen-Image T2I PE 的 prompt 不能为空")
        ensure_optional_dependencies(loader_type)
        rewritten = BACKEND.enhance_t2i_with_loader(str(prompt), model_path, loader_type)
        return _rewritten_prompt_result(rewritten)


class MlxQwenImagePEI2I:
    @classmethod
    def INPUT_TYPES(cls):
        options = model_options("i2i")
        return {
            "required": {
                "images": ("IMAGE", {}),
                "model_path": (options, {"default": options[0]}),
                "prompt": (
                    "STRING",
                    {
                        "default": "make the sky a vivid sunset",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "loader_type": (list(LOADER_OPTIONS), {"default": MLX_VLM_4BIT}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("rewritten_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "MLX/Prompt"

    def enhance(
        self,
        images,
        model_path: str,
        prompt: str,
        loader_type: str = MLX_VLM_4BIT,
    ):
        if not str(prompt or "").strip():
            raise ValueError("Qwen-Image I2I PE 的 prompt 不能为空")
        ensure_optional_dependencies(loader_type)
        # 延迟导入 image.py；该模块包含 ComfyUI 的 torch 图像类型，不能影响
        # 纯 T2I 节点注册及缺少可选 PyTorch/Transformers 时的错误提示。
        from .. import image as image_util

        pil_images = image_util.to_pil_batch(images)
        return _rewritten_prompt_result(
            BACKEND.enhance_i2i_with_loader(
                pil_images, str(prompt), model_path, loader_type
            )
        )


__all__ = ["MlxQwenImagePEI2I", "MlxQwenImagePET2I"]