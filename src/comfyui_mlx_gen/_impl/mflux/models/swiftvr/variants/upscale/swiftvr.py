"""SwiftVR: one-step chunked video restoration.

SwiftVR restores a degraded video with a single DiT forward pass per chunk. There is no
sampler, no guidance and no seed-driven noise: the route is deterministic, and
``z_hq = z_lq - velocity(z_lq, t = 1000)``. Chunking is the model's own fixed
``4a + 1`` FIRST/MIDDLE/LAST protocol, not a memory-safety decision, so this route has no
streaming/quality mode axis and reports ``chunks``, never a mode.

Three pieces make up a run:

* :class:`ReAEStreamingCodec` moves between ``[0, 1]`` pixels and 48-channel latents while
  carrying convolution boundary state across chunks.
* :class:`StreamingDiT` runs the one-step restoration and accumulates the RoPE temporal
  offset so the clip shares one temporal coordinate system.
* The frozen 512-token prompt embedding stands in for the text encoder SwiftVR does not
  ship.
"""

import gc
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.swiftvr.model.swiftvr_reae.reae import ReAE
from mflux.models.swiftvr.model.swiftvr_transformer.swiftvr_transformer import SwiftVRTransformer
from mflux.models.swiftvr.streaming.chunk import ChunkSpec, ChunkType, aligned_frame_count, build_chunk_specs
from mflux.models.swiftvr.streaming.streaming_dit import StreamingDiT
from mflux.models.swiftvr.streaming.streaming_reae import ReAEStreamingCodec
from mflux.models.swiftvr.swiftvr_initializer import SwiftVRInitializer
from mflux.models.swiftvr.variants.upscale.swiftvr_util import SwiftVRUtil
from mflux.utils.generated_video import GeneratedVideo
from mflux.utils.runtime_memory import RuntimeMemory
from mflux.utils.scale_factor import ScaleFactor
from mflux.utils.video_health import VideoHealth
from mflux.utils.video_util import AudioCopyResult, VideoStreamWriter, VideoUtil

# One latent frame per decoder call is both the smallest peak and the fastest at 1080p:
# whole-chunk decode materializes a [14, 544, 960, 128] intermediate.
DEFAULT_DECODE_SLICE_LATENTS = 1
# Recorded in metadata in place of a real seed. The route has no stochastic input.
DETERMINISTIC_SEED = 0
# Abort when the run's measured peak passes this share of host memory.
PEAK_MEMORY_ABORT_FRACTION = 0.85


class SwiftVR(nn.Module):
    """Video restoration runtime backed by the SwiftVR checkpoint.

    Attributes:
        reae: The Restoration-aware Autoencoder replacing the Wan 3D VAE.
        transformer: The Wan 2.2 TI2V-5B transformer with MFSWA token routing.
        prompt_embeds: The frozen ``[1, 512, text_dim]`` prompt embedding.
    """

    reae: ReAE
    transformer: SwiftVRTransformer

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = ModelConfig.swiftvr(),
        decode_slice_latents: int = DEFAULT_DECODE_SLICE_LATENTS,
    ) -> None:
        super().__init__()
        # Checked before the initializer so a rejected request costs nothing: resolving
        # the checkpoint reads three safetensors headers and applying it moves ~18.8 GiB.
        SwiftVR._assert_quantization_supported(quantize)
        self.decode_slice_latents = decode_slice_latents
        SwiftVRInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
        )

    def restore_video_to_path(
        self,
        *,
        video_path: str | Path,
        resolution: int | ScaleFactor,
        output_path: str | Path,
        clip_len: int | None = None,
        dit_overlap: int | None = None,
        start_seconds: float = 0.0,
        max_frames: int | None = None,
        color_correction_mode: str = "wavelet",
        drop_audio: bool = False,
        export_json_metadata: bool = False,
        overwrite: bool = True,
        validate_health: bool = True,
        restore_metadata: dict | None = None,
        enforce_memory_budget: bool = True,
    ) -> Path:
        """Restore a video clip and write it to ``output_path``.

        There is deliberately no ``seed`` parameter: one forward pass at a fixed timestep
        with no noise makes the route deterministic, and accepting a seed would imply an
        influence it does not have. The CLI still records a seed in metadata for
        output-template compatibility.

        Args:
            video_path: Source clip.
            resolution: Target output size. Only 1x is supported. Upstream defaults to 4x
                and pre-upsamples the degraded source bilinearly before encoding, so the
                scaled route is the one its published numbers describe; it is withheld here
                until a bilinear-parity check and a reference comparison exist for it
                (ADR 0001). See :meth:`SwiftVRUtil.output_canvas`.
            output_path: Destination file.
            clip_len: MIDDLE chunk size in source frames; must be a multiple of 4. ``None``
                takes the catalog default (``default_clip_len``).
            dit_overlap: Latent crossfade between chunks. Only 0 is supported. ``None``
                takes the catalog default (``default_dit_overlap``).
            start_seconds: Offset into the source clip.
            max_frames: Cap on source frames; the clip is trimmed to ``4a + 1``.
            color_correction_mode: Post-restoration color transfer mode.
            drop_audio: Skip copying the source audio track.
            export_json_metadata: Write a sidecar metadata file.
            overwrite: Replace an existing output file.
            validate_health: Run the post-write video health check.
            restore_metadata: Extra key/value pairs to record in the output metadata.
            enforce_memory_budget: Abort before starting when the plan exceeds the
                measured safe envelope.

        Returns:
            The path actually written.
        """
        start_time = time.perf_counter()
        clip_len = self.runtime_settings.clip_len if clip_len is None else clip_len
        dit_overlap = self.runtime_settings.dit_overlap if dit_overlap is None else dit_overlap
        self._assert_supported_options(dit_overlap=dit_overlap, color_correction_mode=color_correction_mode)

        clip_probe = VideoUtil.read_video_clip(path=video_path, start_seconds=start_seconds, max_frames=1)
        # The same helper the CLI preflight calls: this number decides the chunk plan, and
        # a second copy of the rule could print one plan and then run another.
        requested_frames = VideoUtil.requested_clip_frame_count(clip_probe, max_frames)
        aligned_frames = aligned_frame_count(requested_frames)
        self._assert_clip_length_supported(aligned_frames)

        target_height, target_width = SwiftVRUtil.output_canvas(
            source_width=clip_probe.source_width,
            source_height=clip_probe.source_height,
            resolution=resolution,
        )
        padded_height, padded_width = SwiftVRUtil.padded_canvas(target_height, target_width)
        canvas_error = SwiftVRUtil.canvas_bound_error(padded_height, padded_width)
        if canvas_error is not None and enforce_memory_budget:
            raise ValueError(canvas_error)

        chunk_specs = build_chunk_specs(aligned_frames, clip_len)
        chunk_latent_frames = clip_len // 4
        actual_clip_start_seconds = clip_probe.clip_start_frame / clip_probe.fps
        # Read now, not at metadata time: under --low-ram the memory saver releases the
        # transformer on after_loop, so anything the metadata needs from it must already
        # be a plain value by then.
        window_hw = list(self.transformer.window_hw)
        # Same reason: the estimate is an attribute of the model, and --low-ram releases
        # the model's components before the metadata is built.
        resident_weight_bytes = int(getattr(self, "swiftvr_resident_weight_bytes", 0))

        codec = ReAEStreamingCodec(
            self.reae,
            decode_slice_latents=self.decode_slice_latents,
            clear_cache_each_slice=True,
        )
        dit = StreamingDiT(
            self.transformer,
            dit_overlap=dit_overlap,
            inference_timestep=self.runtime_settings.inference_timestep,
            clear_cache_each_block=True,
        )

        progress_ctx = self.callbacks.start(
            seed=DETERMINISTIC_SEED,
            prompt="",
            config=self._build_restore_config(
                true_height=target_height,
                true_width=target_width,
                num_inference_steps=max(1, len(chunk_specs)),
            ),
            task="video-to-video",
        )

        writer: VideoStreamWriter | None = None
        file_path: Path | None = None
        audio_copy_result: AudioCopyResult | None = None
        progress_started = False
        progress_latents = None
        previous_lq_latents = None
        frames_written = 0
        peak_bytes = 0
        try:
            chunk_clips = VideoUtil.iter_video_frame_windows(
                video_path,
                start_frame=clip_probe.clip_start_frame,
                windows=[(spec.frame_start, spec.frame_start + spec.frame_count) for spec in chunk_specs],
            )
            for chunk_index, chunk_clip in enumerate(chunk_clips):
                spec = chunk_specs[chunk_index]
                pixels = self._chunk_pixels(
                    frames=list(chunk_clip.frames),
                    spec=spec,
                    padded_height=padded_height,
                    padded_width=padded_width,
                )
                lq_latents = self._to_latent_bcfhw(codec.encode_chunk(pixels, spec))
                del pixels

                if not progress_started:
                    progress_ctx.before_loop(lq_latents)
                    progress_started = True

                if spec.ctype is ChunkType.LAST:
                    restored = dit.denoise_last_chunk(
                        lq_latents,
                        spec,
                        self.prompt_embeds,
                        previous_latents=previous_lq_latents,
                        chunk_latent_frames=chunk_latent_frames,
                    )
                else:
                    restored = dit.denoise(lq_latents, self.prompt_embeds)

                # The next LAST chunk pads its front from the PRECEDING chunk's
                # low-quality latents, not its restored ones - upstream names the buffer
                # after the DiT output but assigns it the pre-subtraction tensor.
                previous_lq_latents = self._carry_tail(lq_latents, chunk_latent_frames)
                progress_latents = previous_lq_latents
                del lq_latents

                for decoded in codec.iter_decode_chunk(self._to_latent_bthwc(restored), spec):
                    frame_arrays = self._to_uint8_frames(decoded, target_height, target_width)
                    if writer is None:
                        writer = VideoStreamWriter(
                            path=output_path,
                            fps=clip_probe.fps,
                            width=target_width,
                            height=target_height,
                            overwrite=overwrite,
                        )
                    writer.write_frame_arrays(frame_arrays)
                    frames_written += int(frame_arrays.shape[0])
                    del frame_arrays
                    del decoded

                del restored
                mx.clear_cache()
                peak_bytes = max(peak_bytes, self._peak_memory())
                self._assert_post_chunk_memory_health(
                    padded_height=padded_height,
                    padded_width=padded_width,
                    enforce_peak_budget=enforce_memory_budget,
                )
                progress_ctx.in_loop(chunk_index, progress_latents)

            if writer is None:
                raise ValueError("SwiftVR video restore did not produce any frames.")
            if frames_written != aligned_frames:
                raise ValueError(
                    f"SwiftVR wrote {frames_written} frames but the chunk plan accounts for "
                    f"{aligned_frames}. The FIRST/MIDDLE/LAST protocol is frame-exact, so a "
                    "mismatch means the decoder head trim or a chunk boundary is wrong."
                )
            file_path = writer.close()
            audio_copy_result = self._copy_audio(
                clip_probe=clip_probe,
                video_path=video_path,
                file_path=file_path,
                drop_audio=drop_audio,
                actual_clip_start_seconds=actual_clip_start_seconds,
                clip_duration_seconds=aligned_frames / clip_probe.fps,
            )
        except Exception:
            if progress_started:
                progress_ctx.failed()
            if writer is not None:
                writer.abort()
            if file_path is not None:
                SwiftVR._cleanup_video_artifacts(file_path)
            raise
        finally:
            codec.reset()
            dit.reset()
            del previous_lq_latents
            mx.clear_cache()

        try:
            if progress_started and progress_latents is not None:
                progress_ctx.after_loop(progress_latents)
            del progress_latents
            mx.clear_cache()
            metadata = GeneratedVideo.build_metadata(
                model_config=self.model_config,
                # The route is deterministic: one forward pass at a constant timestep with
                # no noise. There is no seed to record, and recording the CLI's value would
                # imply an influence it does not have - see seed_affects_output below.
                seed=DETERMINISTIC_SEED,
                prompt="",
                steps=1,
                guidance=None,
                guidance_2=None,
                flow_shift=None,
                solver=None,
                precision=ModelConfig.precision,
                quantization=self.bits,
                generation_time=time.perf_counter() - start_time,
                height=target_height,
                width=target_width,
                frame_count=aligned_frames,
                fps=clip_probe.fps,
                task="video-to-video",
                video_path=video_path,
                extra_metadata={
                    "resolution": str(resolution),
                    "restore_family": "swiftvr",
                    "seed_affects_output": False,
                    "swiftvr_clip_len": clip_len,
                    "swiftvr_dit_overlap": dit_overlap,
                    "swiftvr_chunk_count": len(chunk_specs),
                    "swiftvr_window_size": window_hw,
                    "swiftvr_inference_timestep": dit.inference_timestep,
                    "swiftvr_padded_height": padded_height,
                    "swiftvr_padded_width": padded_width,
                    "swiftvr_decode_slice_latents": self.decode_slice_latents,
                    "swiftvr_resident_weight_bytes": resident_weight_bytes or None,
                    "swiftvr_peak_memory_bytes": peak_bytes or None,
                    "swiftvr_chunk_plan": [
                        {
                            "chunk_type": spec.ctype.value,
                            "frame_start": spec.frame_start,
                            "frame_count": spec.frame_count,
                            "latent_count": spec.latent_count,
                        }
                        for spec in chunk_specs
                    ],
                    **(restore_metadata or {}),
                    "source_video_width": clip_probe.source_width,
                    "source_video_height": clip_probe.source_height,
                    "source_video_fps": round(float(clip_probe.fps), 6),
                    "source_video_frames": clip_probe.source_frame_count,
                    "source_video_duration_seconds": (
                        round(clip_probe.source_duration_seconds, 3)
                        if clip_probe.source_duration_seconds is not None
                        else None
                    ),
                    "source_clip_start_frame": clip_probe.clip_start_frame,
                    "source_clip_start_seconds": round(float(start_seconds), 3),
                    "source_clip_actual_start_seconds": round(float(actual_clip_start_seconds), 6),
                    "source_clip_frames": aligned_frames,
                    "requested_clip_frames": requested_frames,
                    "audio_present": clip_probe.audio_present,
                    "audio_copied": bool(audio_copy_result.audio_copied) if audio_copy_result else False,
                    "audio_copy_mode": audio_copy_result.copy_mode if audio_copy_result else None,
                    "audio_copy_reason": audio_copy_result.reason if audio_copy_result else "not_attempted",
                    "color_correction_mode": color_correction_mode,
                },
            )
            gc_was_enabled = gc.isenabled()
            if gc_was_enabled:
                gc.disable()
            try:
                if validate_health:
                    file_health = VideoHealth.validate_file(
                        file_path,
                        expected_width=target_width,
                        expected_height=target_height,
                        expected_frames=aligned_frames,
                        expected_fps=clip_probe.fps,
                    )
                    metadata["video_health"] = {"file": file_health.to_metadata()}
                else:
                    metadata["health_check"] = "skipped"
                if export_json_metadata:
                    GeneratedVideo.save_metadata(file_path, metadata)
            finally:
                if gc_was_enabled:
                    gc.enable()
            if progress_started:
                progress_ctx.complete()
            return file_path
        except Exception:
            if progress_started:
                progress_ctx.failed()
            if file_path is not None:
                SwiftVR._cleanup_video_artifacts(file_path)
            raise

    @staticmethod
    def _assert_quantization_supported(quantize: int | None) -> None:
        """Reject a quantization request rather than accepting one that does nothing.

        Neither level is honourable on this route. At 8 bits Wan's own q8 sensitivity
        list (``WanWeightDefinition._is_q8_sensitive_transformer_path``) spares every
        quantizable module this architecture has - the attention projections, the FFN
        linears, ``condition_embedder.*`` and ``proj_out`` - so nothing is quantized while
        ``bits`` and the metadata would both record 8. At 4 bits ``condition_embedder``
        IS quantized, and Wan's low-precision linear helper then reads ``linear.weight``
        as a packed buffer and fails inside the timestep projection.

        Raises:
            ValueError: If any quantization level was requested.
        """
        if quantize is None:
            return
        raise ValueError(
            f"SwiftVR does not support --quantize {quantize}; it runs only its bf16 source route. "
            "At 8 bits Wan's q8 sensitivity policy spares every quantizable module in this "
            "architecture, so the model would be recorded as quantized while staying bf16; at 4 "
            "bits the quantized condition embedder fails in the Wan timestep projection. A "
            "quantized SwiftVR package is gated until the bf16 route has the runtime evidence to "
            "validate one against (ADR 0001). Re-run without --quantize / -q."
        )

    @staticmethod
    def _assert_supported_options(*, dit_overlap: int, color_correction_mode: str) -> None:
        """Reject options this route cannot honour, rather than ignoring them.

        Raises:
            ValueError: If a requested option is outside the supported envelope.
        """
        if dit_overlap != 0:
            raise ValueError(
                f"SwiftVR dit_overlap={dit_overlap} is not supported. The offline route runs with no "
                "latent overlap, which is the only configuration with any evidence behind it on this "
                "backend; a positive overlap costs one extra latent frame of DiT sequence per frame and "
                "changes nothing that has been measured. Use dit_overlap=0."
            )
        if color_correction_mode != "off":
            raise ValueError(
                f"SwiftVR does not apply color correction, so --color-correction {color_correction_mode} "
                "cannot be honoured. The reference pipeline writes the decoder output unchanged, and "
                "MLX-Gen has not measured a color transfer against it for this model. Pass "
                "--color-correction off for SwiftVR, or use --model seedvr2-3b, whose restore path has a "
                "validated wavelet and LAB transfer."
            )

    def _assert_clip_length_supported(self, aligned_frames: int) -> None:
        """Fail closed when the clip runs past the rotary table.

        Raises:
            ValueError: If the clip needs more latent positions than the table holds.
        """
        rope_max_seq_len = int((self.model_config.transformer_overrides or {}).get("rope_max_seq_len", 1024))
        limit = SwiftVRUtil.max_supported_source_frames(rope_max_seq_len)
        if aligned_frames > limit:
            raise ValueError(
                f"SwiftVR can restore at most {limit} source frames in one run: the rotary table holds "
                f"{rope_max_seq_len} latent positions and ReAE emits one latent frame per four source "
                f"frames. This clip needs {aligned_frames}. Split it with --start-seconds and "
                "--max-frames, then join the outputs."
            )

    @property
    def runtime_dtype(self) -> mx.Dtype:
        """Dtype the applied weights actually carry, read from a real parameter."""
        return self.reae.encoder.layers[0].weight.dtype

    def _chunk_pixels(
        self,
        *,
        frames: list,
        spec: ChunkSpec,
        padded_height: int,
        padded_width: int,
    ) -> mx.array:
        """Decoded frames to a padded ``[1, T, H, W, 3]`` array in ``[0, 1]``.

        Padding is zeros on the bottom and right only, matching the reference, and is
        cropped away again after decoding.

        Raises:
            ValueError: If the decoder returned the wrong number of frames.
        """
        if len(frames) != spec.frame_count:
            raise ValueError(
                f"SwiftVR chunk {spec.clip_idx} ({spec.ctype.value}) expected {spec.frame_count} decoded "
                f"source frames but the reader returned {len(frames)}."
            )
        stacked = np.stack([VideoUtil._pil_rgb_to_array(frame) for frame in frames], axis=0)
        pixels = mx.array(stacked)[None].astype(self.runtime_dtype) / 255.0
        pad_bottom = padded_height - pixels.shape[2]
        pad_right = padded_width - pixels.shape[3]
        if pad_bottom < 0 or pad_right < 0:
            raise ValueError(
                f"SwiftVR chunk {spec.clip_idx} decoded frames at {pixels.shape[3]}x{pixels.shape[2]}, "
                f"larger than the planned padded canvas {padded_width}x{padded_height}."
            )
        if pad_bottom or pad_right:
            pixels = mx.pad(pixels, [(0, 0), (0, 0), (0, pad_bottom), (0, pad_right), (0, 0)])
        return pixels

    @staticmethod
    def _to_latent_bcfhw(latents: mx.array) -> mx.array:
        """ReAE ``[B, T, H, W, C]`` latents to the transformer's ``[B, C, F, H, W]``."""
        return mx.transpose(latents, (0, 4, 1, 2, 3))

    @staticmethod
    def _to_latent_bthwc(latents: mx.array) -> mx.array:
        """Transformer ``[B, C, F, H, W]`` latents back to ReAE's ``[B, T, H, W, C]``."""
        return mx.transpose(latents, (0, 2, 3, 4, 1))

    @staticmethod
    def _carry_tail(latents: mx.array, chunk_latent_frames: int) -> mx.array:
        """The trailing latents a following LAST chunk may need as its front padding."""
        keep = min(chunk_latent_frames, latents.shape[2])
        tail = latents[:, :, -keep:]
        mx.eval(tail)
        return tail

    @staticmethod
    def _to_uint8_frames(decoded: mx.array, target_height: int, target_width: int) -> np.ndarray:
        """One decoded slice to ``[T, H, W, 3]`` uint8, cropped back to the output canvas.

        Quantization rounds (``x * 255 + 0.5``), matching this repository's video writer
        (``VideoUtil``) rather than the reference's truncating ``(x * 255).to(uint8)``. The
        difference is half a least-significant bit, but it is systematic: a pixel-level
        parity test against the reference writer must allow it rather than expect equality.
        """
        cropped = decoded[0, :, :target_height, :target_width, :]
        scaled = mx.clip(cropped.astype(mx.float32) * 255.0 + 0.5, 0.0, 255.0)
        mx.eval(scaled)
        return np.array(scaled, copy=False).astype(np.uint8)

    @staticmethod
    def _copy_audio(
        *,
        clip_probe,
        video_path: str | Path,
        file_path: Path,
        drop_audio: bool,
        actual_clip_start_seconds: float,
        clip_duration_seconds: float,
    ) -> AudioCopyResult:
        """Preserve the matching source audio segment, or fail rather than drop it silently.

        Raises:
            RuntimeError: If audio was present and could not be copied safely.
        """
        if not clip_probe.audio_present:
            return AudioCopyResult(audio_present=False, audio_copied=False, copy_mode=None, reason="no_source_audio")
        if drop_audio:
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason="drop_audio_requested",
            )
        result = VideoUtil.copy_source_audio_to_video(
            source_video_path=video_path,
            restored_video_path=file_path,
            clip_start_seconds=actual_clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
        )
        if not result.audio_copied:
            raise RuntimeError(
                "Source audio was present but MLX-Gen could not preserve it safely "
                f"({result.reason}). Re-run with drop_audio=True or --drop-audio to allow a silent "
                "restored MP4 intentionally."
            )
        return result

    def _build_restore_config(self, *, true_height: int, true_width: int, num_inference_steps: int) -> Config:
        """Progress-reporting config.

        ``num_inference_steps`` is the chunk count, not the model's one step, so
        ``step / total_steps`` reads as chunk progress on both restoration families. The
        one-step fact is recorded in metadata as ``steps: 1``.
        """
        return Config(
            width=true_width,
            height=true_height,
            guidance=1.0,
            num_inference_steps=num_inference_steps,
            model_config=self.model_config,
            dimension_multiple=2,
        )

    @staticmethod
    def _peak_memory() -> int:
        """Process-wide MLX peak in bytes, as a running maximum over the whole run.

        The counter is deliberately never reset. ``mx.reset_peak_memory()`` is global:
        SwiftVR forces ``--low-ram``, so :class:`MemorySaver` reads the same counter when
        it prints its summary, and resetting it per chunk left that line reporting only
        the last (usually shortest) chunk. Nothing here needs a per-chunk figure - the
        health check asks whether the run has ever passed the host budget, which a
        running maximum answers.

        Failures are not swallowed. This is the route's only memory guard, and a guard
        that turns itself off in silence is the failure mode ADR 0002 exists to prevent.
        """
        return int(mx.get_peak_memory())

    def _assert_post_chunk_memory_health(
        self,
        *,
        padded_height: int,
        padded_width: int,
        enforce_peak_budget: bool,
    ) -> None:
        """Abort on a measured peak that leaves the host no headroom.

        This checks what the run has actually used rather than predicting what the next
        chunk will need. SeedVR2's working-set estimator solves a different attention shape
        and would produce a confident wrong number here, so there is no predictive gate.

        Raises:
            RuntimeError: If the measured peak exceeded the host budget.
        """
        if not enforce_peak_budget:
            return
        peak = self._peak_memory()
        total = RuntimeMemory.total_physical_memory_bytes()
        if not total:
            return
        if peak > total * PEAK_MEMORY_ABORT_FRACTION:
            raise RuntimeError(
                f"SwiftVR restore peaked at {peak / 1024**3:.1f} GiB on a "
                f"{total / 1024**3:.1f} GiB host, past the "
                f"{PEAK_MEMORY_ABORT_FRACTION:.0%} budget that keeps the machine responsive. "
                f"The padded canvas is {padded_width}x{padded_height}. Restore a smaller source, "
                "lower --mlx-cache-limit-gb, or pass --force-unsafe-video-memory to continue anyway."
            )

    @staticmethod
    def _cleanup_video_artifacts(file_path: Path) -> None:
        """Remove a partial output and its sidecar after a failure."""
        metadata_path = file_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            metadata_path.unlink()
        if file_path.exists():
            file_path.unlink()

    def save_model(self, path: str) -> None:
        """Write a prepared local package. Not supported yet.

        Raises:
            NotImplementedError: Always. A quantized SwiftVR package has no validated bf16
                baseline to be checked against, so it is gated behind ADR 0001 runtime
                evidence for the source route.
        """
        raise NotImplementedError(
            "SwiftVR supports only its bf16 source route. Use "
            "`mlxgen download --model H-oliday/SwiftVR`; `mlxgen prepare` and quantized "
            "SwiftVR packages are not supported yet."
        )
