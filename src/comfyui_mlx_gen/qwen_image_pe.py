"""Qwen-Image 2.1 prompt-enhancement backend.

The official PE checkpoints are Qwen3.5-VL models.  This module supports both
the original Transformers snapshots and the ``mlx-vlm`` conversion published
by ``prithivMLmods``.  Both runtimes stay lazy so registering the other MLX
nodes does not import either optional PE runtime.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from PIL import Image

from . import paths, runtime
from .cache import CACHE

PEKind = Literal["t2i", "i2i"]
LoaderType = Literal["mlx_vlm", "transformers"]

T2I_PREFIX = "Qwen-Image-2.1-PE-T2I"
I2I_PREFIX = "Qwen-Image-2.1-PE-I2I"
NO_MODEL = "<无可用 Qwen-Image 2.1 PE 权重>"
MLX_VLM_4BIT = "MLX-VLM (4bit)"
MLX_VLM_8BIT = "MLX-VLM (8bit)"
MLX_16BIT = "MLX 16bit"
TRANSFORMERS = "Transformers"
LOADER_OPTIONS = (MLX_VLM_4BIT, MLX_VLM_8BIT, MLX_16BIT, TRANSFORMERS)


def model_prefix(kind: PEKind) -> str:
    if kind == "t2i":
        return T2I_PREFIX
    if kind == "i2i":
        return I2I_PREFIX
    raise ValueError(f"未知 Qwen-Image PE 类型: {kind}")


def normalize_loader_type(value: str | None) -> tuple[LoaderType, str | None]:
    """将节点显示值转换为运行时类型和可选的 MLX 权重变体。"""
    text = str(value or TRANSFORMERS).strip().casefold()
    if text in {"mlx", "mlx-vlm", "mlx_vlm", "mlx-vlm (4bit)", "mlx_vlm (4bit)"}:
        return "mlx_vlm", "4bit"
    if text in {"mlx-vlm (8bit)", "mlx_vlm (8bit)"}:
        return "mlx_vlm", "8bit"
    if text in {
        "mlx 16bit",
        "mlx-vlm (16bit)",
        "mlx_vlm (16bit)",
        "mlx-vlm 16bit",
        "mlx_vlm 16bit",
        "16bit",
        "bf16",
        "fp16",
    }:
        return "mlx_vlm", "16bit"
    if text in {"transformers", "pytorch", "hf"}:
        return "transformers", None
    raise ValueError(
        f"未知 Qwen-Image PE 加载类型：{value!r}；可选：{', '.join(LOADER_OPTIONS)}"
    )


def _checkpoint_kind(path: Path) -> PEKind | None:
    """从目录名、HF 缓存路径或 README 识别 T2I/I2I checkpoint。"""
    names = " ".join(candidate.name.casefold() for candidate in (path, *path.parents[:4]))
    if "pe-t2i" in names or "image-2.1-pe-t2i" in names:
        return "t2i"
    if "pe-i2i" in names or "image-2.1-pe-i2i" in names:
        return "i2i"
    readme = path / "README.md"
    if readme.is_file():
        try:
            text = readme.read_text(encoding="utf-8", errors="ignore").casefold()
        except OSError:
            text = ""
        if "pe-t2i" in text or "prompt-rewriting" in text and "text-to-image" in text:
            return "t2i"
        if "pe-i2i" in text or "image-to-image" in text or "image editing" in text:
            return "i2i"
    return None


def model_options(kind: PEKind) -> list[str]:
    """列出 text_encoder/ 直接子目录中的、能用于该任务的 PE checkpoint。"""
    found: list[str] = []
    for name in paths.list_component_dirs("text_encoder"):
        path_kind, resolved = paths.resolve("local", name, "text_encoder")
        if path_kind == "dir" and _checkpoint_kind(Path(resolved)) == kind:
            found.append(name)
    return found or [NO_MODEL]


def _choose_mlx_variant(root: Path, variant: str | None) -> Path:
    """选择 MLX snapshot 中的量化/16bit 子目录，或使用根目录权重。"""
    if root.name.casefold() in {"4bit", "8bit", "16bit", "bf16", "fp16", "float16"}:
        return root
    variant_names = {
        "4bit": ("4bit",),
        "8bit": ("8bit",),
        # 不同转换脚本对未量化 checkpoint 的目录命名并不统一。
        "16bit": ("16bit", "bf16", "fp16", "float16"),
    }
    for name in variant_names.get(variant, (variant,) if variant else ()):
        if name and (root / name).is_dir():
            return root / name
    return root


def _validate_mlx_checkpoint(root: Path) -> None:
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    missing = [name for name in required if not (root / name).is_file()]
    has_weights = bool(list(root.glob("*.safetensors")))
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            index_data = json.loads(index.read_text(encoding="utf-8"))
            shards = set((index_data.get("weight_map") or {}).values())
            has_weights = has_weights or all((root / shard).is_file() for shard in shards)
        except (OSError, json.JSONDecodeError):
            pass
    if not has_weights:
        missing.append("*.safetensors")
    if missing:
        raise FileNotFoundError(f"MLX-VLM PE 权重 {root} 缺少：{', '.join(missing)}")


def resolve_model_path(
    selection: str | Path,
    kind: PEKind,
    loader_type: str = TRANSFORMERS,
) -> Path:
    """解析模型选择、校验任务类型，并在加载前给出明确的布局错误。"""
    if str(selection) == NO_MODEL:
        raise FileNotFoundError(
            f"没有可用的 {model_prefix(kind)} 权重；请将 checkpoint 放入 "
            f"{paths.component_dir('text_encoder')}"
        )
    runtime_type, variant = normalize_loader_type(loader_type)
    path_kind, resolved = paths.resolve("local", str(selection), "text_encoder")
    if path_kind != "dir":
        raise FileNotFoundError(f"Qwen-Image PE 权重必须是模型目录，当前路径不存在：{resolved}")
    root = Path(resolved)
    actual_kind = _checkpoint_kind(root)
    if actual_kind is not None and actual_kind != kind:
        raise ValueError(
            f"Qwen-Image PE 节点类型不匹配：当前是 {model_prefix(kind)}，"
            f"但选择的权重是 {model_prefix(actual_kind)}（{root}）；请重新选择模型"
        )
    if actual_kind is None:
        raise ValueError(
            f"无法从 Qwen-Image PE 权重目录识别 T2I/I2I 类型：{root}；"
            "目录名或 README 必须包含 PE-T2I / PE-I2I"
        )
    if runtime_type == "mlx_vlm":
        root = _choose_mlx_variant(root, variant)
        _validate_mlx_checkpoint(root)
        return root
    missing = [
        name for name in ("config.json", "system_prompt.txt") if not (root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Transformers PE 权重 {root} 缺少：{', '.join(missing)}")
    return root


def build_t2i_messages(system_prompt: str, prompt: str) -> list[dict[str, Any]]:
    """构造 PE-T2I 官方 chat-template 输入。"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def build_i2i_messages(
    system_prompt: str, images: Sequence[Image.Image], prompt: str
) -> list[dict[str, Any]]:
    """构造 PE-I2I 官方多模态 chat-template 输入。"""
    content: list[dict[str, Any]] = [
        {"type": "image", "image": image.convert("RGB")} for image in images
    ]
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


def _optional_system_prompt(path: Path, kind: PEKind) -> str:
    """读取可选的官方 system prompt，不让 MLX 转换目录依赖它。

    The ``prithivMLmods`` snapshots intentionally omit this source-model file.
    If the matching original Transformers checkpoint is installed next to it in
    the same ComfyUI component directory, reuse that file automatically.
    """
    direct = path / "system_prompt.txt"
    if direct.is_file():
        return direct.read_text(encoding="utf-8").strip()
    for name in paths.list_component_dirs("text_encoder"):
        path_kind, resolved = paths.resolve("local", name, "text_encoder")
        candidate = Path(resolved)
        if path_kind == "dir" and _checkpoint_kind(candidate) == kind:
            prompt = candidate / "system_prompt.txt"
            if prompt.is_file():
                return prompt.read_text(encoding="utf-8").strip()
    return ""


def _json_from_generation(text: str) -> dict[str, Any]:
    """提取思考块后的 JSON 对象，兼容 markdown code fence。"""
    value = str(text or "").strip()
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[1].strip()
    elif "<think>" in value:
        value = value.replace("<think>", "", 1).strip()
    if "<|im_end|>" in value:
        value = value.split("<|im_end|>", 1)[0].strip()
    if value.startswith("```json"):
        value = value[len("```json") :].lstrip()
    elif value.startswith("```"):
        value = value[3:].lstrip()
    if value.endswith("```"):
        value = value[:-3].rstrip()

    candidates = [value]
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        candidates.append(value[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Qwen-Image PE 没有返回有效 JSON：{value[:400]!r}")


def clean_generation(text: str) -> str:
    """从模型输出中取出下游 Qwen-Image 应使用的 rewritten_prompt。"""
    result = _json_from_generation(text)
    prompt = result.get("rewritten_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Qwen-Image PE JSON 缺少非空 rewritten_prompt")
    return prompt.strip()


@dataclass
class _Bundle:
    kind: PEKind
    path: Path
    model: Any
    tokenizer: Any = None
    processor: Any = None
    loader_type: LoaderType = "transformers"
    config: Any = None

    def close(self) -> None:
        """释放大模型引用；Cache 随后会执行设备缓存清理。"""
        model = self.model
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.config = None
        del model
        gc.collect()


def _torch_and_transformers() -> tuple[Any, Any, Any, Any, Any]:
    """延迟导入可选依赖，并返回所需 Transformers 类。"""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Qwen-Image 2.1 PE 节点需要 PyTorch；请在 ComfyUI 使用的 Python 环境中安装 torch"
        ) from exc
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Qwen-Image 2.1 PE 节点需要 transformers>=5.4.0；"
            "请在 ComfyUI 使用的 Python 环境中运行 pip install 'transformers>=5.4.0'"
        ) from exc
    return torch, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def _mlx_vlm() -> tuple[Any, Any, Any, Any]:
    """延迟导入 mlx-vlm，避免 PE 依赖影响普通 MLX 节点注册。"""
    try:
        from mlx_vlm import apply_chat_template, generate, load
        from mlx_vlm.utils import load_config
    except ImportError as exc:
        raise RuntimeError(
            "Qwen-Image PE 的 MLX-VLM 加载类型需要 mlx-vlm；请在 ComfyUI 使用的 Python "
            "环境中运行 pip install 'mlx-vlm>=0.7.2,<0.8'"
        ) from exc
    return apply_chat_template, generate, load, load_config


def ensure_optional_dependencies(loader_type: str = TRANSFORMERS) -> None:
    """在节点开始处理输入时验证选择的 PE 运行时。"""
    runtime_type, _variant = normalize_loader_type(loader_type)
    if runtime_type == "mlx_vlm":
        _mlx_vlm()
    else:
        _torch_and_transformers()


def _device(torch: Any) -> Any:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QwenImagePEBackend:
    """共享 T2I/I2I 的懒加载、生成和路径缓存后端。"""

    @staticmethod
    def _cache_key(kind: PEKind, path: Path, loader_type: str = TRANSFORMERS) -> str:
        runtime_type, variant = normalize_loader_type(loader_type)
        return runtime.cache_key(
            {
                "kind": "qwen_image_pe",
                "pe_kind": kind,
                "loader_type": runtime_type,
                "variant": variant,
                "path": str(path),
            }
        )

    def _load_bundle(
        self, kind: PEKind, path: Path, loader_type: str = TRANSFORMERS
    ) -> _Bundle:
        runtime_type, _variant = normalize_loader_type(loader_type)
        if runtime_type == "mlx_vlm":
            try:
                _apply_chat_template, _generate, load, load_config = _mlx_vlm()
                config = load_config(str(path))
                model, processor = load(str(path), lazy=False, strict=True)
                return _Bundle(
                    kind=kind,
                    path=path,
                    model=model,
                    processor=processor,
                    loader_type=runtime_type,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"加载 MLX-VLM PE 失败（{path}）：{exc}") from exc

        torch, causal_cls, image_text_cls, processor_cls, tokenizer_cls = _torch_and_transformers()
        device = _device(torch)
        common = {
            "dtype": torch.bfloat16,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        try:
            if kind == "t2i":
                tokenizer = tokenizer_cls.from_pretrained(str(path), local_files_only=True)
                model = causal_cls.from_pretrained(str(path), **common)
                return _Bundle(
                    kind=kind,
                    path=path,
                    model=model.to(device).eval(),
                    tokenizer=tokenizer,
                    loader_type=runtime_type,
                )
            processor = processor_cls.from_pretrained(str(path), local_files_only=True)
            model = image_text_cls.from_pretrained(str(path), **common)
            return _Bundle(
                kind=kind,
                path=path,
                model=model.to(device).eval(),
                processor=processor,
                loader_type=runtime_type,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"加载 {model_prefix(kind)} 失败（{path}，设备 {device}）：{exc}"
            ) from exc

    def _bundle(self, kind: PEKind, path: Path, loader_type: str = TRANSFORMERS) -> _Bundle:
        runtime_type, _variant = normalize_loader_type(loader_type)
        key = self._cache_key(kind, path, loader_type)
        bucket = f"qwen_image_pe_{runtime_type}_{kind}_module"
        bundle, _hit = CACHE.get_or_create(
            bucket, key, lambda: self._load_bundle(kind, path, loader_type)
        )
        return bundle

    def release(
        self, kind: PEKind, model_path: str | Path, loader_type: str = TRANSFORMERS
    ) -> None:
        """释放一次 PE 推理使用的 bundle。

        ``_bundle`` 仍保留缓存语义，便于需要复用的 Python 调用方自行控制
        生命周期；ComfyUI 节点会在每次推理的 ``finally`` 中调用这里，避免
        节点输出的普通字符串反向把 PE 大模型留在进程里。
        """
        try:
            path = resolve_model_path(model_path, kind, loader_type)
            runtime_type, _variant = normalize_loader_type(loader_type)
            bucket = f"qwen_image_pe_{runtime_type}_{kind}_module"
            key = self._cache_key(kind, path, loader_type)
            if CACHE.evict(bucket, key):
                print(f"[Qwen-Image PE] 已释放 {kind} 模型（下次增强时重新懒加载）")
        except Exception as exc:  # noqa: BLE001
            # release 位于节点 finally，不能覆盖模型推理 / 路径校验的原始异常。
            print(f"[Qwen-Image PE] 释放模型失败，继续返回原始结果：{exc}")

    @staticmethod
    def _prepare_inputs(bundle: _Bundle, messages: list[dict[str, Any]]) -> Any:
        if bundle.kind == "t2i":
            text = bundle.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            return bundle.tokenizer(text, return_tensors="pt")
        return bundle.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=True,
        )

    @staticmethod
    def _move_inputs(inputs: Any, device: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _generate(self, bundle: _Bundle, messages: list[dict[str, Any]]) -> str:
        if bundle.loader_type == "mlx_vlm":
            apply_chat_template, generate, _load, _load_config = _mlx_vlm()
            prompt = apply_chat_template(
                bundle.processor,
                bundle.config,
                messages,
                add_generation_prompt=True,
                num_images=sum(
                    1
                    for message in messages
                    if message.get("role") == "user"
                    for item in (message.get("content") or [])
                    if isinstance(item, dict) and item.get("type") == "image"
                ),
                enable_thinking=True,
            )
            images = [
                item["image"]
                for message in messages
                if message.get("role") == "user"
                for item in (message.get("content") or [])
                if isinstance(item, dict) and item.get("type") == "image"
            ]
            try:
                result = generate(
                    bundle.model,
                    bundle.processor,
                    prompt,
                    image=images or None,
                    # The PE model emits a long reasoning block before its JSON.
                    # 512 (the mlx-vlm README demo value) can stop during
                    # ``<think>`` and leave no rewritten_prompt to parse.
                    max_tokens=8192 if bundle.kind == "i2i" else 4096,
                    temperature=0.2,
                    top_p=0.95,
                    top_k=20,
                    enable_thinking=True,
                    skip_special_tokens=False,
                )
                text = getattr(result, "text", result)
                return str(text)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Qwen-Image PE MLX-VLM 推理失败（{bundle.path}）：{exc}") from exc

        torch, *_ = _torch_and_transformers()
        inputs = self._prepare_inputs(bundle, messages)
        input_length = inputs["input_ids"].shape[-1]
        inputs = self._move_inputs(inputs, bundle.model.device)
        generation = {
            "max_new_tokens": 24000 if bundle.kind == "i2i" else 16256,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        }
        try:
            with torch.inference_mode():
                output = bundle.model.generate(**inputs, **generation)
            generated = output[0, input_length:].detach().to("cpu")
            tokenizer = bundle.processor.tokenizer if bundle.kind == "i2i" else bundle.tokenizer
            return tokenizer.decode(generated, skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Qwen-Image PE 推理失败（{bundle.path}）：{exc}") from exc

    def enhance_t2i(self, prompt: str, model_path: str | Path) -> str:
        return self.enhance_t2i_with_loader(prompt, model_path, TRANSFORMERS)

    def enhance_t2i_with_loader(
        self, prompt: str, model_path: str | Path, loader_type: str = TRANSFORMERS
    ) -> str:
        path = resolve_model_path(model_path, "t2i", loader_type)
        bundle = self._bundle("t2i", path, loader_type)
        runtime_type, _variant = normalize_loader_type(loader_type)
        if runtime_type == "mlx_vlm":
            system_prompt = _optional_system_prompt(path, "t2i")
            messages = build_t2i_messages(system_prompt, prompt) if system_prompt else [
                {"role": "user", "content": prompt}
            ]
            return clean_generation(self._generate(bundle, messages))
        system_prompt = (path / "system_prompt.txt").read_text(encoding="utf-8").strip()
        return clean_generation(self._generate(bundle, build_t2i_messages(system_prompt, prompt)))

    def enhance_i2i(
        self, images: Sequence[Image.Image], prompt: str, model_path: str | Path
    ) -> str:
        return self.enhance_i2i_with_loader(images, prompt, model_path, TRANSFORMERS)

    def enhance_i2i_with_loader(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        model_path: str | Path,
        loader_type: str = TRANSFORMERS,
    ) -> str:
        if not images:
            raise ValueError("Qwen-Image I2I PE 至少需要一张 IMAGE")
        path = resolve_model_path(model_path, "i2i", loader_type)
        bundle = self._bundle("i2i", path, loader_type)
        runtime_type, _variant = normalize_loader_type(loader_type)
        if runtime_type == "mlx_vlm":
            system_prompt = _optional_system_prompt(path, "i2i")
            return clean_generation(
                self._generate(bundle, build_i2i_messages(system_prompt, images, prompt))
            )
        system_prompt = (path / "system_prompt.txt").read_text(encoding="utf-8").strip()
        messages = build_i2i_messages(system_prompt, images, prompt)
        return clean_generation(self._generate(bundle, messages))


BACKEND = QwenImagePEBackend()


__all__ = [
    "BACKEND",
    "I2I_PREFIX",
    "LOADER_OPTIONS",
    "MLX_16BIT",
    "MLX_VLM_4BIT",
    "MLX_VLM_8BIT",
    "NO_MODEL",
    "QwenImagePEBackend",
    "T2I_PREFIX",
    "TRANSFORMERS",
    "build_i2i_messages",
    "build_t2i_messages",
    "clean_generation",
    "ensure_optional_dependencies",
    "model_options",
    "normalize_loader_type",
    "resolve_model_path",
]