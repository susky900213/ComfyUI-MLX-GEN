"""Isolated Torch/Diffusers runtime for Fibo 2.0.

This module intentionally has no eager Torch, Diffusers, NumPy, or Pillow
imports.  Fibo 2.0 is a separate backend from both the legacy ``fibo`` package
and the native MLX nodes; importing this module must therefore remain safe in
an MLX-only installation.
"""

from __future__ import annotations

import json
import importlib.util
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import paths
from ..types import LoraRef

FIBO2_PIPELINE_CLASS = "BriaFibo2Pipeline"
FIBO2_TRANSFORMER_CLASS = "BriaFibo2Transformer2DModel"
FIBO2_MAX_IMAGES = 5
FIBO2_TURBO_STEPS = 4
FIBO2_TURBO_GUIDANCE = 1.0
# Hugging Face Diffusers' BriaFibo2Pipeline uses 4096 for
# ``tokenizer_max_length`` and ``max_sequence_length``.  Keep this name
# implementation-specific: Bria's public Fibo-2 materials do not document
# 4096 as a model-architecture hard context limit, while the underlying
# Qwen3-VL tokenizer advertises a larger ``model_max_length``.
FIBO2_DIFFUSERS_MAX_SEQUENCE_LENGTH = 4096
# The bundled Qwen3-VL tokenizer advertises this model context size.  Expose it
# as the configurable ceiling rather than treating Diffusers' 4096 runtime
# default as a hard stop.  Values above 4096 are experimental for Fibo-2 and
# can require substantially more unified memory because attention is quadratic
# in the resulting text + visual sequence length.
FIBO2_TOKENIZER_MODEL_MAX_LENGTH = 262_144
# Fibo 2 was merged after the latest released Diffusers used by ComfyUI
# Desktop.  Pin the known-good upstream commit instead of asking users to
# guess an unreleased version number.
FIBO2_DIFFUSERS_COMMIT = "de988b7ee77c393a63865d265d19e267ef9c7bf5"
FIBO2_DIFFUSERS_SOURCE = (
    f"git+https://github.com/huggingface/diffusers.git@{FIBO2_DIFFUSERS_COMMIT}"
)

OFFICIAL_REPOSITORIES = frozenset(
    {
        "briaai/Fibo-2.0-dev",
        "briaai/Fibo-2.0-edit-turbo",
    }
)


class Fibo2ValidationError(ValueError):
    """Raised when a Fibo 2 checkpoint or request is incompatible."""


class Fibo2BackendUnavailable(RuntimeError):
    """Raised when the optional Torch/Diffusers runtime is unavailable."""


@dataclass(frozen=True)
class Fibo2ModelHandle:
    """Small immutable model selection passed between the custom nodes."""

    model_path: str
    dtype: str = "bfloat16"
    device: str = "auto"
    cache_key: str = ""
    loras: tuple[LoraRef, ...] = ()
    quantize: int = 0
    # Native MLX free-buffer cache limit selected by the unified Transformer
    # loader.  It is a runtime policy, not part of the checkpoint weights.
    compile_cache_limit: int = 0

    @property
    def model_type(self) -> str:
        """Expose the same family attribute as the native MLX model handle."""

        return "fibo2"


@dataclass(frozen=True)
class Fibo2ClipHandle:
    """Lazy CLIP/text-component configuration for the unified CLIP socket.

    Fibo 2 owns its Qwen3-VL component inside the model snapshot.  This handle
    deliberately carries configuration and LoRA references only; it is never
    an MLX text encoder and is materialized by the sampler backend.
    """

    model_path: str
    dtype: str = "bfloat16"
    device: str = "auto"
    cache_key: str = ""
    loras: tuple[LoraRef, ...] = ()
    quantize: int = 0
    max_length: int = FIBO2_DIFFUSERS_MAX_SEQUENCE_LENGTH

    @property
    def model_type(self) -> str:
        return "fibo2"

    @property
    def path(self) -> str:
        return self.model_path

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_length", normalize_sequence_length(self.max_length))


@dataclass(frozen=True)
class Fibo2VaeHandle:
    """Lazy FLUX.2 VAE configuration for the unified VAE socket."""

    model_path: str
    dtype: str = "bfloat16"
    device: str = "auto"
    cache_key: str = ""
    loras: tuple[LoraRef, ...] = ()
    quantize: int = 0

    @property
    def model_type(self) -> str:
        return "fibo2"

    @property
    def path(self) -> str:
        return self.model_path


_PIPELINE_CACHE: dict[tuple[Any, ...], Any] = {}


def normalize_sequence_length(value: int) -> int:
    """Validate a user-selected native MLX Fibo-2 sequence budget."""

    try:
        length = int(value)
    except (TypeError, ValueError) as exc:
        raise Fibo2ValidationError(f"Fibo-2 max_length 必须是整数，收到 {value!r}") from exc
    if length < 1 or length > FIBO2_TOKENIZER_MODEL_MAX_LENGTH:
        raise Fibo2ValidationError(
            "Fibo-2 max_length 必须在 1 到 "
            f"{FIBO2_TOKENIZER_MODEL_MAX_LENGTH} 之间，收到 {length}"
        )
    return length


def model_cache_key(
    model_path: str,
    dtype: str,
    device: str,
    quantize: int = 0,
    loras: Iterable[LoraRef] = (),
    *components: Fibo2ClipHandle | Fibo2VaeHandle | None,
) -> str:
    """Build a stable cache key including every component LoRA reference."""

    refs: list[dict[str, Any]] = [
        {"path": ref.path, "strength": float(ref.strength)} for ref in loras
    ]
    for component in components:
        if component is None:
            continue
        refs.extend({"path": ref.path, "strength": float(ref.strength)} for ref in component.loras)
    return json.dumps(
        {
            "family": "fibo2",
            "path": model_path,
            "dtype": normalize_dtype(dtype),
            "device": normalize_device(device),
            "quantize": int(quantize),
            "loras": refs,
        },
        sort_keys=True,
    )


def normalize_dtype(dtype: str) -> str:
    """Normalize the public dtype widget to a supported canonical name."""

    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    key = str(dtype).strip().casefold()
    try:
        return aliases[key]
    except KeyError as exc:
        raise Fibo2ValidationError(
            f"未知 Fibo 2 dtype: {dtype!r}；可选 bfloat16、float16、float32"
        ) from exc


def normalize_device(device: str) -> str:
    """Normalize a device widget without importing Torch."""

    key = str(device).strip().casefold()
    aliases = {"auto": "auto", "cpu": "cpu", "cuda": "cuda", "mps": "mps"}
    try:
        return aliases[key]
    except KeyError as exc:
        raise Fibo2ValidationError(
            f"未知 Fibo 2 device: {device!r}；可选 auto、mps、cuda、cpu"
        ) from exc


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _contains_marker(value: Any, markers: Iterable[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        text = value.casefold()
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()
    return any(marker.casefold() in text for marker in markers)


def _checkpoint_root(path: Path) -> Path:
    """Find a Diffusers root when the caller selected a component directory."""

    current = path
    for _ in range(5):
        if (current / "model_index.json").is_file():
            return current
        if current.parent == current:
            break
        current = current.parent
    return path


def _official_repository(value: str) -> bool:
    return value.casefold() in {item.casefold() for item in OFFICIAL_REPOSITORIES}


def inspect_model_path(model_path: str | Path) -> dict[str, Any]:
    """Validate a local Fibo-2 Diffusers root or an official HF repository id.

    Remote repositories are deliberately allow-listed.  A random string that
    merely contains ``fibo`` must not cause Diffusers to download an ambiguous
    or legacy checkpoint.
    """

    value = str(model_path).strip()
    if not value:
        raise Fibo2ValidationError("Fibo 2 model_path 不能为空")

    candidate = Path(value).expanduser()
    if candidate.is_dir():
        root = _checkpoint_root(candidate.resolve())
        model_index_path = root / "model_index.json"
        transformer_path = root / "transformer" / "config.json"
        model_index = _read_json(model_index_path)
        transformer = _read_json(transformer_path)
        text_encoder = _read_json(root / "text_encoder" / "config.json")
        vae = _read_json(root / "vae" / "config.json")
        configs = (model_index, transformer, text_encoder, vae)

        if any(
            _contains_marker(item, ("BriaFiboPipeline", "BriaFiboEditPipeline", "AutoencoderKLWan", "SmolLM3"))
            for item in configs
        ):
            raise Fibo2ValidationError(
                f"{root} 是 legacy FIBO checkpoint（Wan/SmolLM3），不能作为 Fibo 2 使用"
            )
        if model_index is None:
            raise Fibo2ValidationError(f"{root} 缺少有效的 model_index.json")
        if transformer is None:
            raise Fibo2ValidationError(f"{root} 缺少有效的 transformer/config.json")
        if not _contains_marker(model_index, (FIBO2_PIPELINE_CLASS,)):
            raise Fibo2ValidationError(
                f"{root} 不是 Fibo 2 checkpoint：model_index.json 缺少 {FIBO2_PIPELINE_CLASS}"
            )
        if not _contains_marker(transformer, (FIBO2_TRANSFORMER_CLASS,)):
            raise Fibo2ValidationError(
                f"{root} 不是 Fibo 2 checkpoint：transformer/config.json 缺少 "
                f"{FIBO2_TRANSFORMER_CLASS}"
            )
        if text_encoder is not None and not _contains_marker(text_encoder, ("Qwen3VL", "qwen3_vl")):
            raise Fibo2ValidationError("Fibo 2 必须使用 Qwen3-VL 文本/视觉条件编码器")
        if vae is not None and _contains_marker(vae, ("AutoencoderKLWan", "Wan")):
            raise Fibo2ValidationError("Fibo 2 必须使用 FLUX.2 VAE，不能使用 legacy Wan VAE")
        return {
            "kind": "local",
            "path": str(root),
            "family": "fibo2",
            "pipeline": True,
            "transformer": True,
            "text_encoder": text_encoder is not None,
            "vae": vae is not None,
        }

    if _official_repository(value):
        return {"kind": "repo", "path": value, "family": "fibo2"}
    raise Fibo2ValidationError(
        "Fibo 2 model_path 必须是包含 model_index.json 的本地 Fibo-2 checkpoint，"
        "或官方仓库 briaai/Fibo-2.0-dev / briaai/Fibo-2.0-edit-turbo；"
        "不会把 legacy FIBO 路径当成 Fibo 2"
    )


def resolve_model_selection(selection: str | Path, component: str = "transformer") -> str:
    """Resolve a unified-loader widget value to a Fibo checkpoint root."""

    value = str(selection).strip()
    if _official_repository(value):
        return value
    kind, resolved = paths.resolve("local", value, component)
    if kind == "missing":
        raise Fibo2ValidationError(f"未找到 Fibo 2 checkpoint: {resolved}")
    info = inspect_model_path(resolved)
    return str(info["path"])


def is_fibo2_model_path(model_path: str | Path) -> bool:
    try:
        inspect_model_path(model_path)
    except Fibo2ValidationError:
        return False
    return True


def _model_variant(model_path: str) -> str:
    return "edit-turbo" if re.search(r"edit[-_.]?turbo", model_path, re.IGNORECASE) else "dev"


def is_edit_turbo_model_path(model_path: str | Path) -> bool:
    return _model_variant(str(model_path)) == "edit-turbo"


def validate_request(
    model_path: str,
    images: Sequence[Any] | None,
    mask: Any | None,
    steps: int,
    guidance: float,
) -> None:
    """Validate image-count and edit-turbo constraints before loading weights."""

    count = len(images or ())
    if count > FIBO2_MAX_IMAGES:
        raise Fibo2ValidationError(
            f"Fibo 2 最多接收 {FIBO2_MAX_IMAGES} 张上下文图片（source + references），收到 {count} 张"
        )
    if mask is not None and count != 1:
        raise Fibo2ValidationError("mask 只支持单张 source image，不能与多参考图同时使用")
    try:
        step_count = int(steps)
        guidance_value = float(guidance)
    except (TypeError, ValueError) as exc:
        raise Fibo2ValidationError("steps 和 guidance 必须是数字") from exc
    if step_count < 1:
        raise Fibo2ValidationError("steps 必须大于 0")
    if guidance_value < 0:
        raise Fibo2ValidationError("guidance 不能小于 0")

    if is_edit_turbo_model_path(model_path):
        if count < 1:
            raise Fibo2ValidationError("Fibo-2.0-edit-turbo 是编辑模型，必须提供 source image")
        if step_count != FIBO2_TURBO_STEPS:
            raise Fibo2ValidationError("Fibo-2.0-edit-turbo 固定 4 步；文生图请换用 Fibo-2.0-dev")
        if abs(guidance_value - FIBO2_TURBO_GUIDANCE) > 1e-6:
            raise Fibo2ValidationError("Fibo-2.0-edit-turbo 是 guidance-free，guidance 必须为 1.0")


def _import_torch() -> Any:
    try:
        import torch  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - host dependent
        raise Fibo2BackendUnavailable(
            "Fibo 2 需要可选依赖 PyTorch；现有 MLX 安装不会自动启用 Torch backend"
        ) from exc
    return torch


def resolve_device(device: str) -> str:
    """Resolve ``auto`` and validate explicit Torch devices."""

    requested = normalize_device(device)
    torch = _import_torch()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise Fibo2ValidationError("请求了 cuda，但当前 PyTorch 没有可用 CUDA")
    if requested == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise Fibo2ValidationError("请求了 mps，但当前 PyTorch 没有可用 Apple MPS")
    return requested


def _torch_dtype(dtype: str) -> Any:
    torch = _import_torch()
    name = normalize_dtype(dtype)
    return getattr(torch, name)


def _fibo2_install_hint() -> str:
    """Return an install command bound to the Python running ComfyUI."""

    executable = shlex.quote(sys.executable)
    source = shlex.quote(FIBO2_DIFFUSERS_SOURCE)
    return f"{executable} -m pip install --upgrade {source}"


def _legacy_fibo_module_available(diffusers: Any) -> bool:
    """Detect old Bria Fibo modules without importing a legacy pipeline class."""

    try:
        return importlib.util.find_spec("diffusers.pipelines.bria_fibo") is not None
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        # Some Diffusers builds expose a lazy package without a normal module
        # spec.  Failure to inspect it must not change the main diagnostic.
        return False


def _pipeline_class() -> type[Any]:
    """Import only BriaFibo2Pipeline; never fall back to legacy BriaFibo classes."""

    try:
        import diffusers  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - host dependent
        raise Fibo2BackendUnavailable(
            "Fibo 2 需要包含 BriaFibo2Pipeline 的 Diffusers；当前环境没有可用 diffusers"
        ) from exc

    pipeline = getattr(diffusers, FIBO2_PIPELINE_CLASS, None)
    if pipeline is None:
        try:
            from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import (  # type: ignore
                BriaFibo2Pipeline,
            )
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - host dependent
            version = getattr(diffusers, "__version__", "unknown")
            legacy_note = (
                "当前环境检测到旧版 diffusers.pipelines.bria_fibo；它只提供 legacy "
                "BriaFibo/BriaFiboEditPipeline，不能用于 Fibo 2。"
                if _legacy_fibo_module_available(diffusers)
                else ""
            )
            raise Fibo2BackendUnavailable(
                f"当前 Diffusers {version} 未提供 BriaFibo2Pipeline；"
                f"{legacy_note}"
                "不会回退到 legacy BriaFiboEditPipeline。请在当前 ComfyUI Python 中执行：\n"
                f"{_fibo2_install_hint()}\n"
                "安装完成后重启 ComfyUI。"
            ) from exc
        pipeline = BriaFibo2Pipeline
    if getattr(pipeline, "__name__", "") != FIBO2_PIPELINE_CLASS:
        raise Fibo2BackendUnavailable(
            f"解析出的 pipeline 不是 {FIBO2_PIPELINE_CLASS}，拒绝加载以避免 FIBO 版本混用"
        )
    return pipeline


def backend_status() -> dict[str, str | bool]:
    """Return a non-throwing diagnostic for tests and future UI diagnostics."""

    try:
        pipeline = _pipeline_class()
    except Fibo2BackendUnavailable as exc:
        return {"available": False, "message": str(exc)}
    return {"available": True, "pipeline": pipeline.__name__, "message": "ok"}


def _load_pipeline(handle: Fibo2ModelHandle) -> Any:
    info = inspect_model_path(handle.model_path)
    checkpoint = info["path"] if info["kind"] == "local" else handle.model_path
    pipeline_cls = _pipeline_class()
    dtype = _torch_dtype(handle.dtype)
    device = resolve_device(handle.device)
    key = (
        str(checkpoint),
        normalize_dtype(handle.dtype),
        device,
        tuple((ref.path, float(ref.strength)) for ref in handle.loras),
    )
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    try:
        # Fibo 2's supported Diffusers revision documents ``dtype``.  Keep a
        # narrow compatibility fallback for an intermediate revision that
        # still exposed the older Diffusers keyword.
        pipe = pipeline_cls.from_pretrained(checkpoint, dtype=dtype)
    except TypeError:
        pipe = pipeline_cls.from_pretrained(checkpoint, torch_dtype=dtype)
    pipe = pipe.to(device)
    _apply_loras(pipe, handle.loras)
    _PIPELINE_CACHE[key] = pipe
    return pipe


def _resolve_lora_path(selection: str) -> tuple[str, str | None]:
    """Resolve a ComfyUI LoRA selection without importing the MLX LoRA stack."""

    selected = Path(str(selection)).expanduser()
    candidates = (
        (selected,),
        (paths.component_dir("lora") / selected, paths.MODEL_ROOT / selected),
    )
    path = next((item for group in candidates for item in group if item.is_file() or item.is_dir()), None)
    if path is None:
        raise Fibo2ValidationError(f"未找到 Fibo 2 LoRA: {selection}")
    if path.is_file():
        return str(path.parent), path.name
    return str(path), None


def _apply_loras(pipe: Any, loras: Iterable[LoraRef]) -> None:
    """Apply Diffusers LoRAs to the pipeline, preserving per-adapter strength."""

    refs = [ref for ref in loras if float(ref.strength) != 0.0]
    if not refs:
        return
    loader = getattr(pipe, "load_lora_weights", None)
    if not callable(loader):
        raise Fibo2BackendUnavailable(
            "当前 BriaFibo2Pipeline 没有 Diffusers LoRA API，无法安全应用 Fibo 2 LoRA"
        )
    names: list[str] = []
    weights: list[float] = []
    for index, ref in enumerate(refs):
        source, weight_name = _resolve_lora_path(ref.path)
        name = f"fibo2_lora_{index}"
        kwargs: dict[str, Any] = {"adapter_name": name}
        if weight_name is not None:
            kwargs["weight_name"] = weight_name
        try:
            loader(source, **kwargs)
        except TypeError:
            # Older Diffusers revisions do not expose adapter_name.  Applying
            # multiple adapters without names would silently overwrite one, so
            # only permit the unambiguous single-adapter fallback.
            if len(refs) != 1:
                raise Fibo2BackendUnavailable(
                    "当前 Diffusers LoRA API 不支持 adapter_name，不能安全叠加多个 Fibo 2 LoRA"
                )
            fallback = {key: value for key, value in kwargs.items() if key != "adapter_name"}
            loader(source, **fallback)
            names = []
            weights = [float(ref.strength)]
            break
        names.append(name)
        weights.append(float(ref.strength))
    if names:
        setter = getattr(pipe, "set_adapters", None)
        if not callable(setter):
            raise Fibo2BackendUnavailable(
                "当前 BriaFibo2Pipeline 缺少 set_adapters，无法保证 Fibo 2 LoRA strength 生效"
            )
        setter(names, adapter_weights=weights)


def _pipeline_output(result: Any) -> list[Any]:
    output = getattr(result, "images", None)
    if output is None and isinstance(result, tuple):
        output = result[0]
    if output is None:
        output = result
    if isinstance(output, (list, tuple)):
        return list(output)
    return [output]


def run_pipeline(
    handle: Fibo2ModelHandle,
    *,
    prompt: str,
    negative_prompt: str | None = None,
    images: Sequence[Any] | None,
    mask: Any | None,
    seed: int,
    steps: int,
    guidance: float,
    width: int | None = None,
    height: int | None = None,
    batch_size: int = 1,
    clip: Fibo2ClipHandle | None = None,
    vae: Fibo2VaeHandle | None = None,
) -> list[Any]:
    """Run the official Diffusers pipeline and return its PIL image list."""

    if not isinstance(handle, Fibo2ModelHandle):
        raise Fibo2ValidationError("model 必须是 Fibo2ModelHandle")
    if not isinstance(prompt, str) or not prompt.strip():
        raise Fibo2ValidationError("Fibo 2 prompt 不能为空")
    validate_request(handle.model_path, images, mask, steps, guidance)
    # Component handles are configuration-only.  Fibo 2's official pipeline
    # owns Qwen3-VL and FLUX.2 VAE; their LoRAs are merged at pipeline load.
    if clip is not None and not isinstance(clip, Fibo2ClipHandle):
        raise Fibo2ValidationError("Fibo 2 clip 必须连接统一 CLIP Loader 的 Fibo2ClipHandle")
    if vae is not None and not isinstance(vae, Fibo2VaeHandle):
        raise Fibo2ValidationError("Fibo 2 vae 必须连接统一 VAE Loader 的 Fibo2VaeHandle")
    if clip is not None and clip.model_path != handle.model_path:
        raise Fibo2ValidationError("Fibo 2 的 model、clip、vae 必须来自同一个 checkpoint")
    if vae is not None and vae.model_path != handle.model_path:
        raise Fibo2ValidationError("Fibo 2 的 model、clip、vae 必须来自同一个 checkpoint")
    if clip is not None and normalize_dtype(clip.dtype) != normalize_dtype(handle.dtype):
        raise Fibo2ValidationError("Fibo 2 的 model 与 clip 必须使用相同 dtype")
    if vae is not None and normalize_dtype(vae.dtype) != normalize_dtype(handle.dtype):
        raise Fibo2ValidationError("Fibo 2 的 model 与 vae 必须使用相同 dtype")
    combined = Fibo2ModelHandle(
        model_path=handle.model_path,
        dtype=handle.dtype,
        device=handle.device,
        cache_key=handle.cache_key,
        loras=tuple(handle.loras)
        + tuple(clip.loras if clip is not None else ())
        + tuple(vae.loras if vae is not None else ()),
    )
    pipe = _load_pipeline(combined)
    torch = _import_torch()
    device = resolve_device(handle.device)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "generator": generator,
        "output_type": "pil",
    }
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    if images:
        kwargs["image"] = list(images)
    if mask is not None:
        kwargs["mask"] = mask
    if width is not None:
        kwargs["width"] = int(width)
    if height is not None:
        kwargs["height"] = int(height)
    if int(batch_size) != 1:
        kwargs["num_images_per_prompt"] = int(batch_size)
    return _pipeline_output(pipe(**kwargs))


def clear_pipeline_cache() -> None:
    """Release only the Fibo 2 pipeline cache."""

    _PIPELINE_CACHE.clear()
    try:
        torch = _import_torch()
        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (Fibo2BackendUnavailable, AttributeError):
        pass