"""ComfyUI node for the SenseNova 6.8 Code2Video checkpoint."""

from __future__ import annotations

import sys
import time

from .. import code2video


class MlxCode2Video:
    """Generate a Python renderer with SenseNova and return its MP4 as VIDEO."""

    @classmethod
    def INPUT_TYPES(cls):
        model_paths = list(code2video.model_choices())
        return {
            "required": {
                "model_path": (
                    model_paths,
                    {
                        "default": model_paths[0],
                        "tooltip": "选择 transformer/ 下的 SenseNova Code2Video 模型；MLX 后端需选择已转换目录",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "一段简洁的抽象几何动效，蓝色圆环在深色背景中呼吸并形成优雅的循环",
                        "multiline": True,
                    },
                ),
                "task_root": (
                    "STRING",
                    {
                        "default": code2video.DEFAULT_TASK_ROOT,
                        "multiline": False,
                        "tooltip": "任务目录根路径；每次运行会创建独立的 code2video-* 子目录",
                    },
                ),
                "width": ("INT", {"default": 640, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 360, "min": 16, "max": 4096, "step": 16}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 600.0, "step": 0.1}),
                "max_new_tokens": (
                    "INT",
                    {"default": 8192, "min": 128, "max": 65536, "step": 128},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "backend": (
                    list(code2video.SUPPORTED_BACKENDS),
                    {
                        "default": code2video.DEFAULT_BACKEND,
                        "tooltip": "代码生成后端",
                    },
                ),
                "device": (
                    list(code2video.SUPPORTED_DEVICES),
                    {
                        "default": code2video.DEFAULT_DEVICE,
                        "tooltip": "模型运行设备",
                    },
                ),
                "render_timeout": (
                    "STRING",
                    {
                        "default": str(code2video.DEFAULT_RENDER_TIMEOUT),
                        "multiline": False,
                        "tooltip": "renderer 最长运行时间（秒）；留空使用 1800 秒",
                    },
                ),
            },
            "optional": {
                "python_executable": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "留空使用当前 ComfyUI Python；渲染代码将在此解释器中执行",
                    },
                )
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "generated_code", "task_directory")
    FUNCTION = "generate"
    CATEGORY = "MLX/Gen/Video"

    def generate(
        self,
        model_path,
        prompt,
        task_root,
        width,
        height,
        fps,
        duration,
        max_new_tokens,
        temperature,
        backend,
        device,
        render_timeout,
        python_executable="",
    ):
        node_started = time.perf_counter()
        request = code2video.VideoRequest(
            prompt=str(prompt),
            width=int(width),
            height=int(height),
            fps=float(fps),
            duration=float(duration),
        )
        task_dir = code2video.create_task_directory(task_root)
        code2video.write_spec(task_dir, request)
        try:
            normalized_backend, normalized_device, normalized_timeout = (
                code2video.normalize_node_options(backend, device, render_timeout)
            )
            code, response = code2video.generate_code(
                str(model_path),
                request,
                device=normalized_device,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                backend=normalized_backend,
                metrics_path=task_dir / "generation_metrics.json",
                response_path=task_dir / "model_response.txt",
            )
        except Exception as exc:
            code2video.write_task_failure(task_dir, "model_load_or_code_generation", exc)
            raise RuntimeError(
                "MlxCode2Video 模型加载或代码生成失败；"
                f"{exc}；"
                f"阶段日志见 {task_dir / 'error.md'}"
            ) from exc
        (task_dir / "model_response.txt").write_text(response, encoding="utf-8")
        renderer_started = time.perf_counter()
        try:
            final_path, metadata = code2video.run_renderer(
                task_dir,
                code,
                python_executable=str(python_executable or sys.executable),
                timeout=normalized_timeout,
                expected_width=request.width,
                expected_height=request.height,
                expected_fps=request.fps,
                expected_duration=request.duration,
            )
        except Exception as exc:
            code2video.write_task_failure(task_dir, "renderer", exc)
            raise RuntimeError(
                "MlxCode2Video renderer 执行失败；"
                f"阶段日志见 {task_dir / 'error.md'}"
            ) from exc
        code2video.update_generation_stats(
            task_dir / "generation_metrics.json",
            renderer_seconds=time.perf_counter() - renderer_started,
            total_node_seconds=time.perf_counter() - node_started,
            video_metadata=metadata,
        )
        print(
            "[MlxCode2Video] 已生成视频："
            f"{final_path} ({metadata.get('width')}x{metadata.get('height')})"
        )
        try:
            video = code2video.video_from_file(final_path)
        except Exception as exc:
            code2video.write_task_failure(task_dir, "video_wrap", exc)
            raise RuntimeError(
                "MlxCode2Video 无法将 MP4 包装为 ComfyUI VIDEO；"
                f"阶段日志见 {task_dir / 'error.md'}"
            ) from exc
        return video, code, str(task_dir)
