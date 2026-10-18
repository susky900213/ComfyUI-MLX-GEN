"""Raon Speech 的独立 PyTorch/Transformers runtime。

这个模块故意不在导入时加载 ``torch``、``torchaudio``、``transformers`` 或官方
``raon`` 包。Raon Speech 是约几十 GB 的 PyTorch checkpoint，不能让它改变现有
MLX 节点的启动和内存行为；ComfyUI 节点只传递这里定义的轻量 handle，第一次真正
执行 TTS/STT 时才创建一个 RaonPipeline 实例。

官方 pipeline 的音频接口是文件路径而不是 ComfyUI 的 AUDIO 字典，因此这里也统一
负责两种格式之间的转换。写入 WAV 使用 Python 标准库，避免为了这层适配强制安装
soundfile；官方 RaonProcessor 本身仍可能需要 soundfile，缺失时会在执行节点中给出
明确错误。
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import inspect
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np

from . import paths

RAON_MODEL = "RAON_MODEL"
DEFAULT_MODEL = "A.X-K2-Raon-Speech-21B-A3B"
NO_MODEL = "<未找到 Raon 模型目录>"
DEVICES = ("auto", "mps", "cpu")
DTYPES = ("bfloat16", "float16", "float32")
ATTENTION_IMPLEMENTATIONS = ("sdpa", "eager", "fa")


@dataclass(frozen=True)
class RaonModelHandle:
    """不含模型对象的 Raon 配置；模型在 ``use_pipeline`` 中惰性创建。"""

    model_path: str
    device: str = "auto"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"


@dataclass
class _PipelineEntry:
    pipeline: Any
    lock: threading.RLock


_PIPELINES: dict[tuple[str, str, str, str], _PipelineEntry] = {}
_CACHE_LOCK = threading.RLock()


def model_options() -> tuple[str, ...]:
    """列出 transformer/ 下的本地目录；没有模型时保留可执行的占位项。"""
    root = paths.component_dir("transformer")
    names = [
        name
        for name in paths.list_component_items("transformer")
        if (root / name).is_dir() and "raon" in name.lower()
    ]
    default = [name for name in names if name == DEFAULT_MODEL]
    names = default + [name for name in names if name != DEFAULT_MODEL]
    return tuple(names or [DEFAULT_MODEL])


def resolve_model(selection: str) -> str:
    """解析本地 Raon checkpoint，禁止因路径错误而静默联网下载。"""
    kind, resolved = paths.resolve("local", str(selection), "transformer")
    if kind != "dir":
        raise FileNotFoundError(
            f"未找到本地 Raon Speech 模型目录：{resolved}；"
            f"请把完整 checkpoint 放到 {paths.component_dir('transformer')}"
        )
    directory = Path(resolved)
    if not (directory / "config.json").is_file():
        raise ValueError(
            f"不是有效的 Raon Speech checkpoint：{directory}；目录必须包含 config.json"
        )
    return str(directory)


def make_handle(
    selection: str,
    device: str = "auto",
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
) -> RaonModelHandle:
    """校验节点参数并创建轻量 handle，不导入 PyTorch。"""
    if device not in DEVICES:
        raise ValueError(f"Raon device 必须是 {DEVICES} 之一，收到 {device!r}")
    if dtype not in DTYPES:
        raise ValueError(f"Raon dtype 必须是 {DTYPES} 之一，收到 {dtype!r}")
    if attn_implementation not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            f"Raon attention backend 必须是 {ATTENTION_IMPLEMENTATIONS} 之一，"
            f"收到 {attn_implementation!r}"
        )
    return RaonModelHandle(
        model_path=resolve_model(selection),
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )


def _dependency_error(exc: Exception) -> RuntimeError:
    missing = getattr(exc, "name", None)
    suffix = f"（缺少模块 {missing}）" if missing else ""
    return RuntimeError(
        "Raon Speech 需要独立的 PyTorch/Transformers 依赖"
        f"{suffix}。请在 ComfyUI 使用的 Python 环境执行：\n"
        "python -m pip install -r <ComfyUI-MLX-GEN>/requirements-raon.txt\n"
        "并安装官方 Raon-Speech 源码（或使用包含 custom code 的本地 checkpoint）。"
    )


def _import_pipeline_class(model_path: str) -> type:
    """导入官方 pipeline，且不因缺少可选依赖阻塞 MLX 节点注册。

    官方仓库通常作为 ``raon`` Python package 安装；某些本地 checkpoint 也会携带
    ``pipeline.py``/Transformers custom code，所以这里按这两种官方支持的布局尝试。
    所有尝试都发生在第一次推理时，而不是 plugin import 时。
    """
    errors: list[Exception] = []
    try:
        package = importlib.import_module("raon")
        pipeline_class = getattr(package, "RaonPipeline")
        return pipeline_class
    except (ImportError, AttributeError) as package_error:
        errors.append(package_error)

    # A checkout placed next to a local checkpoint may expose the official package
    # without being installed into ComfyUI's Python environment.
    model_directory = Path(model_path)
    local_import_roots = (model_directory, model_directory.parent, model_directory.parent.parent)
    previous_path = list(sys.path)
    try:
        for root in reversed(local_import_roots):
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
        for module_name in ("raon.pipeline", "raon_pipeline"):
            try:
                module = importlib.import_module(module_name)
                pipeline_class = getattr(module, "RaonPipeline")
                return pipeline_class
            except (ImportError, AttributeError) as local_error:
                errors.append(local_error)
    finally:
        sys.path[:] = previous_path

    # Some local snapshots carry the custom pipeline module directly.  Load only from
    # the checkpoint directory; unlike a normal import this also handles a standalone
    # ``pipeline.py`` without requiring a surrounding ``raon`` package.
    for filename in ("pipeline.py", "raon_pipeline.py", "modeling_raon.py"):
        module_path = model_directory / filename
        if not module_path.is_file():
            continue
        module_name = f"_comfyui_raon_custom_{abs(hash(str(module_path.resolve()))):x}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法为 {module_path} 创建 Python module spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            pipeline_class = getattr(module, "RaonPipeline")
            return pipeline_class
        except (ImportError, AttributeError, OSError, SyntaxError) as local_file_error:
            sys.modules.pop(module_name, None)
            errors.append(local_file_error)

    # Transformers custom-code snapshots use a module/class reference rather than an
    # installed package.  Keep this fallback local-only: no network access is allowed.
    try:
        dynamic = importlib.import_module("transformers.dynamic_module_utils")
        get_class = dynamic.get_class_from_dynamic_module
    except Exception as transformers_error:  # noqa: BLE001
        errors.append(transformers_error)
    else:
        for class_reference in (
            "pipeline.RaonPipeline",
            "raon_pipeline.RaonPipeline",
            "modeling_raon.RaonPipeline",
        ):
            try:
                return get_class(class_reference, model_path, local_files_only=True)
            except Exception as dynamic_error:  # noqa: BLE001
                errors.append(dynamic_error)

    detail = "; ".join(str(error) for error in errors[-3:] if str(error))
    missing = errors[0] if errors else ImportError("RaonPipeline")
    message = _dependency_error(missing)
    if detail:
        message = RuntimeError(f"{message}\n本地 Raon custom code 加载详情：{detail}")
    if errors:
        raise message from errors[-1]
    raise message


def _tts_with_reference(pipeline: Any, text: str, reference_path: str) -> Any:
    """调用不同 Raon 发布版本的参考音频参数。

    当前上游 ``RaonPipeline`` 使用 ``speaker_audio``；一些 checkpoint 搭配的
    pipeline 版本曾使用更通用的 ``ref_audio``。通过绑定方法签名选择参数名，避免
    用 ``try/except TypeError`` 重试一次推理（那可能已经触发了模型计算）。
    """
    try:
        parameters = inspect.signature(pipeline.tts).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    parameter_names = {parameter.name for parameter in parameters}
    if "ref_audio" in parameter_names:
        argument_name = "ref_audio"
    elif "speaker_audio" in parameter_names or not parameter_names:
        argument_name = "speaker_audio"
    elif any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        argument_name = "speaker_audio"
    else:
        raise TypeError(
            "RaonPipeline.tts 不接受已知的参考音频参数（预期 ref_audio 或 speaker_audio）"
        )
    return pipeline.tts(str(text), **{argument_name: reference_path})


def _resolve_device(device: str, torch_module: Any) -> str:
    if device == "auto":
        try:
            if bool(torch_module.backends.mps.is_available()):
                return "mps"
        except (AttributeError, RuntimeError):
            pass
        return "cpu"
    if device == "mps":
        try:
            if not bool(torch_module.backends.mps.is_available()):
                raise RuntimeError("当前 PyTorch 没有可用的 MPS 设备")
        except AttributeError as exc:
            raise RuntimeError("当前 PyTorch 不支持 MPS；请把 Raon device 改为 cpu") from exc
    return device


def _cache_key(handle: RaonModelHandle) -> tuple[str, str, str, str]:
    return (handle.model_path, handle.device, handle.dtype, handle.attn_implementation)


def _load_pipeline(handle: RaonModelHandle) -> Any:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001
        raise _dependency_error(exc) from exc
    device = _resolve_device(handle.device, torch)
    pipeline_class = _import_pipeline_class(handle.model_path)
    try:
        return pipeline_class(
            handle.model_path,
            device=device,
            dtype=handle.dtype,
            attn_implementation=handle.attn_implementation,
        )
    except ModuleNotFoundError as exc:
        raise _dependency_error(exc) from exc
    except RuntimeError as exc:
        if "MPS" in str(exc) or "mps" in str(exc):
            raise RuntimeError(
                f"Raon 在 {device} 上初始化失败：{exc}。"
                "Apple Silicon 可尝试 dtype=float16 或 device=cpu。"
            ) from exc
        raise


def _get_entry(handle: RaonModelHandle) -> _PipelineEntry:
    key = _cache_key(handle)
    with _CACHE_LOCK:
        entry = _PIPELINES.get(key)
        if entry is None:
            entry = _PipelineEntry(pipeline=_load_pipeline(handle), lock=threading.RLock())
            _PIPELINES[key] = entry
        return entry


@contextlib.contextmanager
def use_pipeline(handle: RaonModelHandle) -> Iterator[Any]:
    """取得缓存中的单实例 pipeline，并串行化同一实例的生成调用。"""
    if not isinstance(handle, RaonModelHandle):
        raise TypeError("Raon 节点的 model 输入必须来自 Raon Model Loader")
    entry = _get_entry(handle)
    with entry.lock:
        yield entry.pipeline


def clear_cache(handle: RaonModelHandle | None = None) -> None:
    """释放一个或全部 Raon pipeline；主要供内存紧张时或测试使用。"""
    with _CACHE_LOCK:
        if handle is None:
            entries = list(_PIPELINES.values())
            _PIPELINES.clear()
        else:
            entry = _PIPELINES.pop(_cache_key(handle), None)
            entries = [entry] if entry is not None else []
    for entry in entries:
        if entry is not None:
            # 删除大对象引用；官方模型没有统一 close() 契约。
            for name in ("model", "processor"):
                if hasattr(entry.pipeline, name):
                    setattr(entry.pipeline, name, None)
    gc.collect()
    try:
        torch = importlib.import_module("torch")
        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def normalize_audio(audio: Any) -> tuple[np.ndarray, int]:
    """把 ComfyUI AUDIO 第一批转换成 float32 单声道 NumPy 波形。"""
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Raon 音频输入必须是 ComfyUI AUDIO（含 waveform 与 sample_rate）")
    waveform = audio["waveform"]
    try:
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().float().cpu().numpy()
        array = np.asarray(waveform, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法读取 Raon AUDIO waveform：{exc}") from exc
    if array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(
            f"Raon AUDIO waveform 必须是非空 [batch, channels, samples]，收到 {array.shape}"
        )
    try:
        sample_rate = int(audio["sample_rate"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Raon AUDIO sample_rate 必须是正整数") from exc
    if sample_rate <= 0:
        raise ValueError(f"Raon AUDIO sample_rate 必须大于 0，收到 {sample_rate}")
    mono = np.mean(array[0], axis=0, dtype=np.float32)
    if not np.all(np.isfinite(mono)):
        raise ValueError("Raon AUDIO 含 NaN 或无穷值")
    return np.ascontiguousarray(np.clip(mono, -1.0, 1.0), dtype=np.float32), sample_rate


@contextlib.contextmanager
def audio_path(audio: Any) -> Iterator[str]:
    """把 ComfyUI AUDIO 写入临时 PCM16 WAV，并在调用后删除。"""
    import tempfile

    mono, sample_rate = normalize_audio(audio)
    temporary = tempfile.NamedTemporaryFile(prefix="comfyui-raon-", suffix=".wav", delete=False)
    path = Path(temporary.name)
    temporary.close()
    pcm = np.rint(mono * 32767.0).astype("<i2", copy=False)
    try:
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
        yield str(path)
    finally:
        path.unlink(missing_ok=True)


def _audio_output(waveform: Any, sample_rate: Any) -> dict[str, Any]:
    """把官方 `(waveform, sample_rate)` 结果包装为 ComfyUI AUDIO。"""
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001
        raise _dependency_error(exc) from exc
    if hasattr(waveform, "detach"):
        tensor = waveform.detach().float().cpu()
    else:
        tensor = torch.as_tensor(np.asarray(waveform), dtype=torch.float32).cpu()
    if tensor.ndim == 1:
        tensor = tensor[None, None, :]
    elif tensor.ndim == 2:
        tensor = tensor[None, :, :]
    elif tensor.ndim != 3:
        raise ValueError(f"Raon TTS waveform 必须是 1D/2D/3D，收到 shape={tuple(tensor.shape)}")
    tensor = tensor.clamp(-1.0, 1.0).contiguous()
    rate = int(sample_rate)
    if rate <= 0:
        raise ValueError(f"Raon 输出采样率必须大于 0，收到 {sample_rate!r}")
    return {"waveform": tensor, "sample_rate": rate}


def synthesize(
    handle: RaonModelHandle,
    text: str,
    speaker_audio: Any | None = None,
) -> dict[str, Any]:
    """执行官方 TTS convenience API，并转换成 ComfyUI AUDIO。"""
    if not str(text).strip():
        raise ValueError("Raon TTS 文本不能为空")
    with use_pipeline(handle) as pipeline:
        if speaker_audio is None:
            waveform, sample_rate = pipeline.tts(str(text))
        else:
            with audio_path(speaker_audio) as reference_path:
                waveform, sample_rate = _tts_with_reference(pipeline, str(text), reference_path)
    return _audio_output(waveform, sample_rate)


def transcribe(handle: RaonModelHandle, audio: Any, prompt: str = "") -> str:
    """执行官方 STT convenience API。"""
    with audio_path(audio) as input_path, use_pipeline(handle) as pipeline:
        result = pipeline.stt(input_path, prompt=str(prompt).strip() or None)
    return str(result)


def speech_chat(handle: RaonModelHandle, audio: Any, prompt: str = "") -> str:
    """执行官方 Speech-Chat convenience API，当前节点返回文本回答。"""
    result = chat(handle, audio, prompt=prompt)
    if not isinstance(result, str):
        raise TypeError("Raon Speech Chat 预期文本回答，但 pipeline 返回了音频")
    return result


def chat(
    handle: RaonModelHandle,
    audio: Any,
    prompt: str = "",
    force_audio_output: bool = False,
) -> str | dict[str, Any]:
    """调用官方 ``RaonPipeline.chat``，支持文本或语音回答。

    这个低层包装保留官方返回契约：文本回答转换成 ``str``，音频回答转换成
    ComfyUI AUDIO。ComfyUI 节点默认使用上面的 ``speech_chat`` 文本接口，避免一个
    输出端口在执行期间改变类型；需要动态选择输出的集成方可直接调用本函数。
    """
    with audio_path(audio) as input_path, use_pipeline(handle) as pipeline:
        content: list[dict[str, str]] = [{"type": "audio", "audio": input_path}]
        if str(prompt).strip():
            content.append({"type": "text", "text": str(prompt).strip()})
        result = pipeline.chat(
            [{"role": "user", "content": content}],
            force_audio_output=bool(force_audio_output),
        )
    if force_audio_output:
        waveform, sample_rate = result
        return _audio_output(waveform, sample_rate)
    return str(result)
