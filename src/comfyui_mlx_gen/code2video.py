"""SenseNova Code2Video bridge.

The official Code2Video release is an agentic code-generation model.  It does
not return diffusion latents or video frames: it writes a renderer and the
renderer produces ``out/final.mp4``.  The default local inference path in this
module is the native ``mlx-vlm`` Qwen3.5 implementation.  A Transformers path
is retained as an explicit compatibility backend for existing installations.

The original SenseNova download is a Transformers checkpoint.  It must be
converted to MLX-VLM format before the MLX backend can load it; MLX does not
read the original PyTorch tensor naming/layout by simply pointing ``load`` at
that directory.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import paths


DEFAULT_TRANSFORMERS_MODEL_PATH = os.environ.get(
    "COMFYUI_CODE2VIDEO_MODEL",
    "/Users/apple/ComfyUI-Shared/models/mlx/transformer/Code2Video",
)
DEFAULT_MODEL_PATH = os.environ.get(
    "COMFYUI_CODE2VIDEO_MLX_MODEL",
    os.environ.get(
        # Keep the old variable usable.  If it points to the original
        # Transformers directory, the MLX validator will explain how to
        # convert it instead of silently loading PyTorch.
        "COMFYUI_CODE2VIDEO_MODEL",
        str(Path(DEFAULT_TRANSFORMERS_MODEL_PATH).expanduser().with_name("Code2Video-MLX")),
    ),
)
DEFAULT_TASK_ROOT = os.environ.get(
    "COMFYUI_CODE2VIDEO_TASK_ROOT",
    str(Path(tempfile.gettempdir()) / "comfyui-code2video"),
)

NO_MODEL = "<无可用 Code2Video 模型>"
SUPPORTED_BACKENDS = ("mlx_vlm", "transformers")
DEFAULT_BACKEND = "mlx_vlm"
SUPPORTED_DEVICES = ("auto", "mps", "cuda", "cpu")
DEFAULT_DEVICE = "auto"
DEFAULT_RENDER_TIMEOUT = 1800
MAX_RENDER_TIMEOUT = 24 * 60 * 60


def _has_model_weights(path: Path) -> bool:
    """Return whether a local directory contains Code2Video weight files."""

    if list(path.glob("*.safetensors")):
        return True
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = set((index.get("weight_map") or {}).values())
    except (OSError, json.JSONDecodeError, AttributeError):
        return False
    return bool(shards) and all((path / shard).is_file() for shard in shards)


def _is_code2video_directory(path: Path) -> bool:
    """Recognize original or converted local SenseNova Code2Video snapshots."""

    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    architectures = config.get("architectures", [])
    return (
        config.get("model_type") == "qwen3_5"
        or "Qwen3_5ForConditionalGeneration" in architectures
    ) and _has_model_weights(path)


def local_models() -> tuple[str, ...]:
    """List selectable local Code2Video directories under ``transformer/``.

    Both the original Transformers snapshot and the converted MLX snapshot are
    exposed. The selected backend performs the final format validation, so an
    original snapshot can still be selected intentionally with ``transformers``.
    """

    root = paths.component_dir("transformer")
    found = [
        name
        for name in paths.list_component_dirs("transformer")
        if _is_code2video_directory(root / name)
    ]

    # Keep an explicitly configured model available even when it lives outside
    # the registered component directory (for example an external disk path).
    configured = Path(DEFAULT_MODEL_PATH).expanduser()
    if configured.is_dir() and _is_code2video_directory(configured):
        try:
            is_component_child = configured.parent.resolve() == root.resolve()
        except OSError:
            is_component_child = False
        configured_value = configured.name if is_component_child else str(configured)
        if configured_value not in found:
            found.append(configured_value)

    preferred = Path(DEFAULT_MODEL_PATH).expanduser().name
    found.sort(key=lambda name: (name != preferred, name.casefold()))
    return tuple(found or [NO_MODEL])


def model_choices() -> tuple[str, ...]:
    """Return combo values, including known absolute-path legacy aliases.

    ComfyUI validates a list-valued combo before calling ``VALIDATE_INPUTS``.
    Consequently, merely resolving an old absolute path at execution time is
    not enough: an old workflow would be rejected before the node runs.  Keep
    normal directory names first for a clean default, and append absolute
    aliases for legacy workflows that stored the previous path-valued widget.
    """

    choices = list(local_models())
    if choices == [NO_MODEL]:
        choices = []

    root = paths.component_dir("transformer")
    # ``DEFAULT_TRANSFORMERS_MODEL_PATH`` is the path used by the previous
    # string-input node.  Also include the absolute path of each scanned local
    # model: old workflows may have stored a path different from the current
    # environment variable while still referring to the same model directory.
    legacy_candidates = [Path(DEFAULT_TRANSFORMERS_MODEL_PATH).expanduser()]
    legacy_candidates.extend(
        root / name
        for name in choices
        if name != NO_MODEL and not Path(name).is_absolute()
    )
    for candidate in legacy_candidates:
        if not candidate.is_dir() or not _is_code2video_directory(candidate):
            continue
        try:
            absolute_value = str(candidate.absolute())
            resolved_value = str(candidate.resolve())
        except OSError:
            absolute_value = str(candidate)
            resolved_value = absolute_value
        for value in (absolute_value, resolved_value):
            if value not in choices:
                choices.append(value)
    return tuple(choices or [NO_MODEL])


def resolve_model_selection(selection: str | os.PathLike[str]) -> str:
    """Resolve a combo-box value while retaining absolute-path compatibility."""

    value = str(selection or "").strip()
    if not value or value == NO_MODEL:
        raise FileNotFoundError(
            f"没有可用的 Code2Video 模型；请将完整 checkpoint 放入 "
            f"{paths.component_dir('transformer')}"
        )

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    kind, resolved = paths.resolve("local", value, "transformer")
    if kind != "dir":
        raise FileNotFoundError(
            f"找不到本地 Code2Video 模型目录：{resolved}；请重新选择 transformer 模型"
        )
    return resolved

_PYTHON_BLOCK_RE = re.compile(
    r"```(?:python|py)[ \t]*\r?\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL
)
_GENERIC_BLOCK_RE = re.compile(r"```[ \t]*\r?\n(?P<code>.*?)```", re.DOTALL)


@dataclass(frozen=True)
class VideoRequest:
    """The render contract sent to SenseNova."""

    prompt: str
    width: int
    height: int
    fps: float
    duration: float

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("Code2Video 提示词不能为空")
        if self.width < 16 or self.height < 16:
            raise ValueError("Code2Video 画布宽高至少为 16")
        if self.width > 4096 or self.height > 4096:
            raise ValueError("Code2Video 画布宽高不能超过 4096")
        if self.fps <= 0 or self.fps > 120:
            raise ValueError("Code2Video fps 必须在 0 到 120 之间")
        if self.duration <= 0 or self.duration > 600:
            raise ValueError("Code2Video 时长必须在 0 到 600 秒之间")


def parse_render_timeout(value: Any) -> int:
    """Normalize the renderer timeout, including legacy empty widget values.

    Older ComfyUI workflow data can leave this widget as ``""`` after the
    node's inputs changed.  The node exposes it as STRING so ComfyUI does not
    reject the prompt before execution; an empty value intentionally means the
    documented default.
    """

    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_RENDER_TIMEOUT
    if isinstance(value, bool):
        raise ValueError("Code2Video render_timeout 必须是秒数，不能是布尔值")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Code2Video render_timeout 必须是整数秒数，例如 1800") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("Code2Video render_timeout 必须是整数秒数，例如 1800")
    timeout = int(numeric)
    if timeout < 1 or timeout > MAX_RENDER_TIMEOUT:
        raise ValueError("Code2Video render_timeout 必须在 1 秒到 24 小时之间")
    return timeout


def normalize_backend(value: Any) -> str:
    """Normalize backend widget values from current and older workflows."""

    if value is None or not str(value).strip():
        return DEFAULT_BACKEND
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "mlx": "mlx_vlm",
        "mlxvlm": "mlx_vlm",
        "local_transformers": "transformers",
        "transformers_local": "transformers",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Code2Video 不支持的 backend：{value}；"
            f"可选值：{', '.join(SUPPORTED_BACKENDS)}"
        )
    return normalized


def normalize_device(value: Any) -> str:
    """Normalize device widget values from current and older workflows."""

    if value is None or not str(value).strip():
        return DEFAULT_DEVICE
    normalized = str(value).strip().lower()
    aliases = {"metal": "mps", "gpu": "auto"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Code2Video 不支持的 device：{value}；"
            f"可选值：{', '.join(SUPPORTED_DEVICES)}"
        )
    return normalized


def normalize_node_options(
    backend: Any,
    device: Any,
    render_timeout: Any,
) -> tuple[str, str, int]:
    """Normalize the three Code2Video widgets, including one stale layout.

    When these widgets changed type in an existing ComfyUI node, older
    frontend state could keep the values under the wrong names.  The affected
    layout is recognizable as ``backend=<device>``, ``device=<timeout>`` and
    an empty ``render_timeout`` (for example ``auto, 1800, ""``).  Migrate
    only that exact shape; all other values still go through the normal strict
    validation below.
    """

    if render_timeout is None or not str(render_timeout).strip():
        try:
            shifted_device = normalize_device(backend)
            shifted_timeout = parse_render_timeout(device)
        except ValueError:
            pass
        else:
            return (
                normalize_backend(render_timeout),
                shifted_device,
                shifted_timeout,
            )

    return (
        normalize_backend(backend),
        normalize_device(device),
        parse_render_timeout(render_timeout),
    )


def build_generation_prompt(request: VideoRequest) -> str:
    """Build a direct-rendering prompt for the local MLX/Transformers path.

    The public skill normally gives the model a Claude Code tool loop.  A
    ComfyUI node has no such tool loop, so the contract must be explicit: emit
    one self-contained Python program instead of tool calls or a plan.
    """

    request.validate()
    return f"""You are SenseNova 6.8 Code2Video running inside a ComfyUI node.

Create a complete short code-driven video for this brief:
{request.prompt.strip()}

The ComfyUI adapter cannot run Claude Code, browser capture, or interactive
tools. Return exactly ONE Python code block and no prose outside the block.
The code block must contain a standalone executable Python script named
render.py. It is executed with the working directory set to the task folder.

Hard render contract:
- Generate a deterministic video at {request.width}x{request.height} pixels.
- Use {request.fps:g} fps and {request.duration:g} seconds.
- The final file MUST be written to out/final.mp4, relative to the working
  directory. Create the out directory yourself.
- Use this exact canonical path in the script: `Path("out") / "final.mp4"`.
- Prefer packages available in this environment: Python standard library,
  Pillow, NumPy, PyAV (av), imageio/imageio-ffmpeg, and ffmpeg. Do not require
  moviepy, Manim, Playwright, a browser, network access, or external assets.
- If using ffmpeg, invoke it with subprocess argument arrays, never shell=True.
- Render every frame as a deterministic function of time; do not use wall-clock
  time. Use a fixed seed for any randomness.
- When using Pillow drawing primitives, keep rectangle coordinates valid at all
  boundary values: normalize them so x0 <= x1 and y0 <= y1. Test empty/full
  progress values such as 0.0 and 1.0 instead of assuming an interior frame.
- Encode a normal playable MP4 (H.264/yuv420p is preferred). Do not merely
  save PNG frames and do not return a placeholder.
- Keep the script self-contained: do not import files outside this task folder.
- A short, finished, visually coherent film is more important than an
  ambitious incomplete one.

Return only the Python fenced code block."""


def extract_python_code(response: str) -> str:
    """Extract one executable Python block from a model response.

    Code2Video responses can contain reasoning or Markdown fences.  We accept
    a labelled Python fence first, then an unlabelled fence as a compatibility
    fallback.  Multiple blocks are rejected because concatenating unrelated
    snippets silently produces broken renderers.
    """

    text = str(response).strip()
    if not text:
        raise ValueError("Code2Video 模型返回了空响应")
    blocks = _PYTHON_BLOCK_RE.findall(text)
    if not blocks:
        blocks = _GENERIC_BLOCK_RE.findall(text)
    if len(blocks) > 1:
        raise ValueError(
            "Code2Video 模型必须返回恰好一个 Python 代码块；"
            f"实际找到 {len(blocks)} 个"
        )

    # A few Qwen3.5 checkpoints omit Markdown fences even when explicitly asked
    # for them.  Accept a response that is itself valid Python, but do not try
    # to execute arbitrary prose by guessing where code starts.
    code = (blocks[0] if blocks else text).strip()
    if not code:
        raise ValueError("Code2Video 返回了空的 Python 代码块")
    try:
        compile(code, "<code2video-renderer>", "exec")
    except SyntaxError as exc:
        raise ValueError(f"Code2Video 返回的 Python 代码无法编译：{exc}") from exc

    # Do not require one particular spelling such as ``Path('out') /
    # 'final.mp4'``.  Generated renderers commonly assign ``Path('out')`` to
    # a variable first and then join the filename later.  The actual renderer
    # still has a strict post-execution check for the task-local output file.
    try:
        tree = ast.parse(code, filename="<code2video-renderer>")
    except SyntaxError as exc:  # Keep the error consistent with compile().
        raise ValueError(f"Code2Video 返回的 Python 代码无法编译：{exc}") from exc
    string_literals = {
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    has_mp4 = any(value.endswith(".mp4") for value in string_literals)
    has_out = any(
        value == "out"
        or value.endswith("/out")
        or value.endswith("\\out")
        or value.startswith("out/")
        or value.startswith("out\\")
        for value in string_literals
    )
    if not (has_mp4 and has_out):
        raise ValueError(
            "生成代码没有包含 out 下的 MP4 输出约定；"
            "请检查任务目录中的 model_response.txt"
        )
    return code + "\n"


def resolve_device(requested: str) -> str:
    """Resolve a Transformers device selector without importing torch early."""

    value = str(requested or "auto").strip().lower()
    if value not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Code2Video 不支持的 device：{requested}")
    if value != "auto":
        return value

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_mlx_device(requested: str) -> str:
    """Resolve the node's device selector for MLX.

    MLX uses Metal rather than PyTorch's ``mps`` device name.  ``auto`` and
    ``mps`` therefore mean the MLX default Metal device; ``cpu`` is accepted
    for diagnostics.  CUDA is deliberately rejected instead of silently
    running a different backend.
    """

    value = str(requested or "auto").strip().lower()
    if value in {"auto", "mps"}:
        try:
            import mlx.core as mx

            if not mx.metal.is_available():
                raise RuntimeError("当前机器没有可用的 Apple Metal 设备")
        except ImportError as exc:
            raise RuntimeError("Code2Video MLX 后端需要安装 mlx") from exc
        return "metal"
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        raise ValueError("Code2Video 的 MLX 后端不支持 cuda；请选择 auto、mps 或 cpu")
    raise ValueError(f"Code2Video 不支持的 device：{requested}")


def _model_input_device(model: Any, fallback: str):
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return fallback


def _read_model_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not path.is_dir():
        raise ValueError(f"Code2Video 模型目录不存在：{path}")
    if not config_path.is_file():
        raise ValueError(f"Code2Video 模型目录缺少 config.json：{path}")
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 Code2Video config.json：{exc}") from exc


def _validate_qwen35_config(path: Path) -> dict[str, Any]:
    config = _read_model_config(path)
    architectures = config.get("architectures", [])
    if config.get("model_type") != "qwen3_5" and "Qwen3_5ForConditionalGeneration" not in architectures:
        raise ValueError(
            "该目录不是已验证的 SenseNova/Qwen3.5 Code2Video checkpoint；"
            f"model_type={config.get('model_type')!r}, architectures={architectures!r}"
        )
    return config


def _checkpoint_has_mlx_metadata(path: Path) -> bool:
    """Return whether at least one MLX safetensors shard has MLX metadata."""

    try:
        from safetensors import safe_open
    except ImportError:
        # mlx-vlm itself can still load a valid checkpoint without this optional
        # inspection helper.  The loader will provide the authoritative error.
        return False
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        return False
    try:
        with safe_open(str(shards[0]), framework="np", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception:
        return False
    return metadata.get("format") == "mlx"


def _validate_mlx_checkpoint(path: Path) -> dict[str, Any]:
    """Validate a local MLX-VLM Qwen3.5 directory before loading 50+ GB."""

    config = _validate_qwen35_config(path)
    if not (path / "model.safetensors.index.json").is_file() and not list(
        path.glob("*.safetensors")
    ):
        raise ValueError(
            f"MLX Code2Video 模型目录缺少 safetensors 权重：{path}"
        )

    # The SenseNova download has the same model_type as MLX-VLM but contains
    # PyTorch/Transformers shards without the MLX marker.  Refuse any unmarked
    # local checkpoint explicitly so a node execution never starts a misleading
    # multi-minute load attempt.
    if not _checkpoint_has_mlx_metadata(path):
        raise ValueError(
            "当前目录没有 MLX-VLM safetensors 标记，通常是 Transformers 原始 Code2Video "
            "checkpoint，不是可直接加载的 MLX 权重。请先执行：\n"
            f"  {sys.executable} -m mlx_vlm.convert --hf-path {path} "
            f"--mlx-path {path}-MLX --dtype bfloat16\n"
            "然后把节点的 model_path 改为生成的 *-MLX 目录。"
        )
    return config


def _load_mlx_vlm_model(model_path: str, device: str):
    """Load a converted Qwen3.5 checkpoint with native MLX-VLM."""

    path = Path(model_path).expanduser()
    _validate_mlx_checkpoint(path)
    try:
        import mlx.core as mx
        from mlx_vlm import load
    except ImportError as exc:
        raise RuntimeError(
            "Code2Video 的 MLX 后端需要 mlx>=0.32.0 和 mlx-vlm>=0.7.2；"
            "请在当前 ComfyUI Python 环境中执行 pip install -r requirements.txt。"
        ) from exc

    # ``set_default_device`` is process-global.  Explicitly select GPU here so
    # a diagnostic CPU invocation cannot poison a later auto/Metal invocation
    # in the same long-lived ComfyUI process.
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    try:
        model, processor = load(str(path), lazy=False, strict=True)
    except Exception as exc:
        raise RuntimeError(
            "Code2Video MLX-VLM 模型加载失败。请确认该目录由 mlx-vlm convert 生成、"
            "Qwen3.5 权重分片完整，并且 Apple Silicon 统一内存足够。"
        ) from exc
    return processor, model, device


def _load_transformers_model(model_path: str, device: str):
    """Load the original architecture as an explicit compatibility backend."""

    path = Path(model_path).expanduser()
    config = _read_model_config(path)
    architectures = config.get("architectures", [])
    if "Qwen3_5ForConditionalGeneration" not in architectures:
        raise ValueError(
            "该目录不是已验证的 SenseNova Code2Video Transformers checkpoint；"
            f"architectures={architectures!r}"
        )

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Code2Video 需要当前 ComfyUI Python 环境安装 transformers 和 torch。"
        ) from exc

    dtype = torch.bfloat16 if device in {"mps", "cuda"} else torch.float32
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        # Accelerate-backed loading avoids a second full copy when CUDA is used.
        # Do not pass device_map for MPS: support varies across Accelerate and
        # Transformers releases, while the regular .to("mps") path is portable.
        load_kwargs["device_map"] = {"": device}
    try:
        processor = AutoProcessor.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        try:
            model = AutoModelForImageTextToText.from_pretrained(str(path), **load_kwargs)
        except (TypeError, ValueError, RuntimeError) as first_error:
            if device not in {"cuda"}:
                raise
            load_kwargs.pop("device_map", None)
            model = AutoModelForImageTextToText.from_pretrained(str(path), **load_kwargs)
            model.to(device)
            # Preserve the original exception in the error context if the
            # fallback itself fails; the latter is generally more actionable.
            del first_error
        if device == "mps":
            # MPS does not consistently support Accelerate's device_map path.
            # The model was therefore loaded on CPU and must be moved before
            # inputs are moved to MPS in generate_code().
            model.to(device)
    except Exception as exc:
        raise RuntimeError(
            "Code2Video Transformers 模型加载失败。该模型约 54.7 GB，"
            "Apple Silicon 需要足够的统一内存；请确认 transformers 版本支持 "
            "Qwen3_5ForConditionalGeneration，并检查本地分片是否完整。"
        ) from exc
    model.eval()
    return processor, model, _model_input_device(model, device)


_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any, Any]] = {}


def _write_generation_stats(path: str | os.PathLike[str] | None, stats: dict[str, Any]) -> None:
    if path is None:
        return
    try:
        Path(path).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError:
        # Metrics must never hide a successful generation.
        pass


def _write_text_artifact(path: str | os.PathLike[str] | None, text: str) -> None:
    """Best-effort persistence for model output used by post-mortem debugging."""

    if path is None:
        return
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(text), encoding="utf-8")
    except OSError:
        # The generated result must not be masked by a diagnostics write error.
        pass


def update_generation_stats(
    path: str | os.PathLike[str] | None, **updates: Any
) -> None:
    """Append renderer/node metrics to the generation metrics artifact."""

    if path is None:
        return
    target = Path(path)
    try:
        stats: dict[str, Any] = {}
        if target.is_file():
            loaded = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                stats.update(loaded)
        stats.update(updates)
        _write_generation_stats(target, stats)
    except (OSError, json.JSONDecodeError, TypeError):
        # Metrics are diagnostic only and must never hide the renderer result.
        pass


def _generate_with_mlx(
    processor: Any,
    model: Any,
    request: VideoRequest,
    *,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    """Generate text through mlx-vlm and return response plus benchmark data."""

    try:
        from mlx_vlm import apply_chat_template, generate
    except ImportError as exc:
        raise RuntimeError("Code2Video MLX 后端需要 mlx-vlm") from exc

    raw_prompt = build_generation_prompt(request)
    try:
        prompt = apply_chat_template(
            processor,
            model.config,
            raw_prompt,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Code2Video MLX-VLM Qwen3.5 chat template 处理失败：{exc}") from exc
    started = time.perf_counter()
    generation_kwargs: dict[str, Any] = {
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": 0.95 if temperature > 0 else 1.0,
        "enable_thinking": False,
        "skip_special_tokens": True,
    }
    try:
        result = generate(
            model,
            processor,
            prompt,
            verbose=False,
            **generation_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"Code2Video MLX-VLM 代码生成失败：{exc}") from exc
    elapsed = time.perf_counter() - started
    response = str(getattr(result, "text", result) or "")
    stats = {
        "backend": "mlx_vlm",
        "elapsed_seconds": elapsed,
        "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
        "generation_tokens": int(getattr(result, "generation_tokens", 0) or 0),
        "total_tokens": int(getattr(result, "total_tokens", 0) or 0),
        "prompt_tokens_per_second": float(getattr(result, "prompt_tps", 0.0) or 0.0),
        "generation_tokens_per_second": float(
            getattr(result, "generation_tps", 0.0) or 0.0
        ),
        "peak_memory_gb": float(getattr(result, "peak_memory", 0.0) or 0.0),
        "finish_reason": getattr(result, "finish_reason", None),
    }
    return response, stats


def generate_code(
    model_path: str,
    request: VideoRequest,
    *,
    device: str = "auto",
    max_new_tokens: int = 8192,
    temperature: float = 0.2,
    backend: str = "mlx_vlm",
    metrics_path: str | os.PathLike[str] | None = None,
    response_path: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Generate a renderer and return ``(code, raw_response)``.

    ``mlx_vlm`` is the default and the only backend used by the MLX node.  The
    Transformers implementation remains available only when explicitly chosen
    for comparison or compatibility.
    """

    request.validate()
    model_path = resolve_model_selection(model_path)
    if max_new_tokens < 128 or max_new_tokens > 65536:
        raise ValueError("Code2Video max_new_tokens 必须在 128 到 65536 之间")
    if temperature < 0 or temperature > 2:
        raise ValueError("Code2Video temperature 必须在 0 到 2 之间")
    backend = normalize_backend(backend)
    device = normalize_device(device)

    resolved_device = (
        resolve_mlx_device(device) if backend == "mlx_vlm" else resolve_device(device)
    )
    key = (str(Path(model_path).expanduser().resolve()), backend, resolved_device)
    load_started = time.perf_counter()
    model_cache_hit = key in _MODEL_CACHE
    if key not in _MODEL_CACHE:
        loader = _load_mlx_vlm_model if backend == "mlx_vlm" else _load_transformers_model
        _MODEL_CACHE[key] = loader(key[0], resolved_device)
    model_load_seconds = time.perf_counter() - load_started
    processor, model, input_device = _MODEL_CACHE[key]

    if backend == "mlx_vlm":
        response, stats = _generate_with_mlx(
            processor,
            model,
            request,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        stats["model_load_seconds"] = model_load_seconds
        stats["model_cache_hit"] = model_cache_hit
        stats["device"] = resolved_device
        _write_generation_stats(metrics_path, stats)
        _write_text_artifact(response_path, response)
        return extract_python_code(response), response

    messages = [{"role": "user", "content": build_generation_prompt(request)}]
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        # Some processor builds do not expose enable_thinking in the Python
        # signature even though the bundled template supports it.
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    if hasattr(inputs, "to"):
        inputs = inputs.to(input_device)
    else:
        inputs = {
            name: value.to(input_device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }

    input_ids = inputs["input_ids"]
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": temperature > 0,
        "temperature": max(float(temperature), 1e-5),
        "top_p": 0.95,
        "use_cache": True,
    }
    if temperature <= 0:
        generation_kwargs.pop("temperature")
        generation_kwargs.pop("top_p")
    try:
        import torch

        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Code2Video 代码生成失败：{exc}") from exc

    generated_only = generated[:, input_ids.shape[1] :]
    response = processor.batch_decode(
        generated_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    _write_generation_stats(
        metrics_path,
        {
            "backend": "transformers",
            "device": resolved_device,
            "model_load_seconds": model_load_seconds,
            "model_cache_hit": model_cache_hit,
            "generation_tokens": int(generated_only.shape[-1]),
        },
    )
    _write_text_artifact(response_path, response)
    return extract_python_code(response), response


def create_task_directory(task_root: str | os.PathLike[str] | None = None) -> Path:
    """Create a unique task folder and its official ``out`` directory."""

    root = Path(task_root or DEFAULT_TASK_ROOT).expanduser()
    if not str(root).strip():
        raise ValueError("Code2Video task_root 不能为空")
    root.mkdir(parents=True, exist_ok=True)
    task = root / f"code2video-{uuid.uuid4().hex}"
    task.mkdir()
    (task / "out").mkdir()
    return task


def write_task_failure(task_dir: Path, stage: str, exc: BaseException) -> None:
    """Persist a node failure next to the task artifacts for post-mortem use."""

    task_dir = Path(task_dir)
    try:
        details = (
            "# Code2Video failure\n\n"
            f"- Stage: `{stage}`\n"
            f"- Exception: `{type(exc).__name__}: {exc}`\n\n"
            "## Traceback\n\n"
            "```text\n"
            f"{traceback.format_exc()}"
            "```\n"
        )
        (task_dir / "error.md").write_text(details, encoding="utf-8")
    except OSError:
        # Never hide the original node exception because diagnostics failed.
        pass


def write_spec(task_dir: Path, request: VideoRequest) -> None:
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    frame_count = round(request.duration * request.fps)
    text = f"""# Code2Video render specification

- Prompt: {request.prompt.strip()}
- Format: {request.width} × {request.height}
- FPS: {request.fps:g}
- Duration target: {request.duration:g} seconds
- Frame count target: {frame_count}
- Audio: renderer-defined; no audio was required by the adapter
- Renderer: generated Python script (`render.py`)
- Deliverable: `out/final.mp4`
"""
    (task_dir / "SPEC.md").write_text(text, encoding="utf-8")


def _video_metadata(path: Path) -> dict[str, Any]:
    try:
        import av

        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise ValueError("MP4 中没有视频流")
            return {
                "width": int(stream.width or 0),
                "height": int(stream.height or 0),
                "fps": float(stream.average_rate) if stream.average_rate else None,
                "duration": float(stream.duration * stream.time_base)
                if stream.duration is not None and stream.time_base
                else None,
                "codec": stream.codec_context.name if stream.codec_context else None,
            }
    except ImportError:
        return {"size_bytes": path.stat().st_size}
    except Exception as exc:
        raise RuntimeError(f"无法验证生成的 MP4：{exc}") from exc


_RENDER_COMPAT_PREAMBLE = r'''
# Code2Video adapter compatibility: Pillow rejects rectangles whose coordinates
# are reversed.  Generated animation code often computes a progress bar with a
# 0.0/1.0 boundary value, so normalize only the flat four-coordinate form while
# preserving all other ImageDraw behavior.
try:
    from PIL import ImageDraw as _code2video_image_draw

    _code2video_original_rectangle = _code2video_image_draw.ImageDraw.rectangle
    if not getattr(_code2video_original_rectangle, "_code2video_safe", False):
        def _code2video_safe_rectangle(self, xy, *args, **kwargs):
            if isinstance(xy, (list, tuple)) and len(xy) == 4:
                try:
                    x0, y0, x1, y1 = xy
                    xy = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                except (TypeError, ValueError):
                    pass
            return _code2video_original_rectangle(self, xy, *args, **kwargs)

        _code2video_safe_rectangle._code2video_safe = True
        _code2video_image_draw.ImageDraw.rectangle = _code2video_safe_rectangle
except Exception:
    pass
'''


def run_renderer(
    task_dir: Path,
    code: str,
    *,
    python_executable: str | None = None,
    timeout: int = 1800,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_fps: float | None = None,
    expected_duration: float | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write and execute the generated renderer without invoking a shell."""

    timeout = parse_render_timeout(timeout)
    task_dir = Path(task_dir).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    output_dir = task_dir / "out"
    output_dir.mkdir(exist_ok=True)
    final_path = output_dir / "final.mp4"
    if final_path.exists() or final_path.is_symlink():
        try:
            final_path.unlink()
        except OSError as exc:
            raise RuntimeError(
                "Code2Video 无法清理上一次渲染留下的 out/final.mp4："
                f"{exc}"
            ) from exc
    script_path = task_dir / "render.py"
    render_code = _RENDER_COMPAT_PREAMBLE + "\n" + str(code)
    try:
        compile(render_code, str(script_path), "exec")
    except SyntaxError as exc:
        raise RuntimeError(f"Code2Video render.py 语法检查失败：{exc}") from exc
    script_path.write_text(render_code, encoding="utf-8")

    executable = str(python_executable or sys.executable)
    if not executable or not Path(executable).is_file():
        raise ValueError("没有可用的 Python 执行器")
    log_path = task_dir / "render.log"
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONUNBUFFERED", "1")

    def _resource_limits():
        """Best-effort limits; generated code is not a security sandbox."""

        try:
            import resource

            cpu_limit = max(1, int(timeout) + 5)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 2))
            # Prevent an accidental raw-video dump from filling the disk.  The
            # task folder itself remains available for debugging after failure.
            resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))
        except (ImportError, OSError, ValueError):
            pass

    def _write_log(output: str) -> None:
        try:
            log_path.write_text(output, encoding="utf-8", errors="replace")
        except OSError:
            pass

    process = None
    try:
        process = subprocess.Popen(
            [executable, str(script_path)],
            cwd=str(task_dir),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
            preexec_fn=_resource_limits if os.name == "posix" else None,
        )
        output, _ = process.communicate(timeout=int(timeout))
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass
            output, _ = process.communicate()
        else:
            output = exc.stdout or ""
        output = str(output or "")[-12000:]
        _write_log(output)
        raise RuntimeError(f"Code2Video 渲染超时（{timeout}s）：\n{output}") from exc
    output = str(output or "")
    if process.returncode != 0:
        output_tail = output[-12000:]
        _write_log(output)
        raise RuntimeError(
            f"Code2Video render.py 退出码 {process.returncode}：\n{output_tail}"
        )
    _write_log(output)

    if not final_path.exists():
        # The checkpoint occasionally follows the instruction semantically but
        # names the only output ``video.mp4``/``result.mp4``.  Normalize that
        # unambiguous task-local result instead of discarding a valid render.
        candidates = [
            path
            for path in output_dir.iterdir()
            if path.suffix.lower() == ".mp4" and path.is_file() and not path.is_symlink()
        ]
        if len(candidates) == 1:
            try:
                candidates[0].replace(final_path)
            except OSError as exc:
                raise RuntimeError(
                    "Code2Video 无法将唯一的 MP4 输出规范化为 out/final.mp4："
                    f"{exc}"
                ) from exc
    if not final_path.exists():
        raise RuntimeError(
            "Code2Video 渲染进程结束但没有生成有效的 out/final.mp4；"
            f"任务目录：{task_dir}"
        )
    try:
        resolved_output = output_dir.resolve()
        resolved_final = final_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Code2Video 无法解析输出路径：{exc}") from exc
    if final_path.is_symlink() or resolved_output not in resolved_final.parents:
        raise RuntimeError("Code2Video 的 out/final.mp4 必须是任务目录内的普通文件")
    if not final_path.is_file() or final_path.stat().st_size == 0:
        raise RuntimeError(
            "Code2Video 渲染进程结束但没有生成有效的 out/final.mp4；"
            f"任务目录：{task_dir}"
        )
    metadata = _video_metadata(final_path)
    if expected_width is not None and metadata.get("width") not in {None, int(expected_width)}:
        raise RuntimeError(
            f"Code2Video 输出宽度不符：期望 {expected_width}，实际 {metadata.get('width')}"
        )
    if expected_height is not None and metadata.get("height") not in {None, int(expected_height)}:
        raise RuntimeError(
            f"Code2Video 输出高度不符：期望 {expected_height}，实际 {metadata.get('height')}"
        )
    if expected_fps is not None and metadata.get("fps") is not None:
        if abs(float(metadata["fps"]) - float(expected_fps)) > max(0.5, float(expected_fps) * 0.05):
            raise RuntimeError(
                f"Code2Video 输出 FPS 不符：期望 {expected_fps:g}，实际 {metadata['fps']:.4g}"
            )
    if expected_duration is not None and metadata.get("duration") is not None:
        if abs(float(metadata["duration"]) - float(expected_duration)) > max(1.0, float(expected_duration) * 0.2):
            raise RuntimeError(
                f"Code2Video 输出时长不符：期望 {expected_duration:g}s，实际 {metadata['duration']:.4g}s"
            )
    (task_dir / "VERIFY.md").write_text(
        "# Code2Video verification\n\n"
        f"- Output: `out/final.mp4`\n- Metadata: `{json.dumps(metadata, ensure_ascii=False)}`\n",
        encoding="utf-8",
    )
    return final_path, metadata


def video_from_file(path: Path):
    """Return ComfyUI's native VIDEO object without importing ComfyUI at import time."""

    try:
        from comfy_api.latest import InputImpl
    except ImportError as exc:
        raise RuntimeError(
            "当前 ComfyUI 版本没有 comfy_api.latest.InputImpl.VideoFromFile，"
            "无法把 Code2Video 文件接入原生 VIDEO 链路。"
        ) from exc
    return InputImpl.VideoFromFile(str(path))


def clear_model_cache() -> None:
    """Release the lazy model cache for tests or an explicit host cleanup."""

    _MODEL_CACHE.clear()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) is not None:
            torch.mps.empty_cache()
    except Exception:
        pass
