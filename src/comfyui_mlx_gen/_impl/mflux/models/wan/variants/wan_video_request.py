from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mflux.utils.dimension_resolver import DimensionResolver


# eq=False: fields include an mx.array and a dict, so generated equality/hash would raise.
@dataclass(frozen=True, eq=False)
class WanVideoRequest:
    # Head bound for multi-frame context conditioning (0102): heads of 5-13
    # pixel frames cover the community handover window (SVI hands over 5).
    MAX_CONTEXT_HEAD_FRAMES = 13

    task: str
    is_image_to_video: bool
    is_video_to_video: bool
    height: int
    width: int
    # Mutable by design: generate_video merges source-video metadata into a local copy.
    spatial_metadata: dict
    num_frames: int
    guidance: float
    guidance_2: float | None
    flow_shift: float
    solver: str
    negative_prompt: str
    video_strength: float | None
    video_mask: mx.array | None
    health_check_interval: int | None
    batch_size: int
    canvas_policy: str | None = None
    resize_mode: str = "resize"

    # Executes the generate_video validation/resolution head in its original order, through the
    # model's helpers so existing instance monkeypatches keep working. Must be called at the top
    # of generate_video (validation reads live model state such as released denoisers).
    @staticmethod
    def resolve(
        model,
        *,
        guidance: float | None,
        guidance_2,
        guidance_2_unset_sentinel: object,
        height: int,
        width: int,
        num_frames: int,
        image_path: Path | str | None,
        video_path: Path | str | None,
        video_strength: float | None,
        video_mask_path: Path | str | None,
        flow_shift: float | None,
        solver: str | None,
        negative_prompt: str | None,
        tensor_health_check_interval: int | None,
        canvas_policy: str | None = None,
        resize_mode: str = "resize",
        last_image_path: Path | str | None = None,
        context_image_paths: list[Path | str] | None = None,
        context_noise: float | None = None,
        svi_anchor_image_path: Path | str | None = None,
        svi_motion_latent_path: Path | str | None = None,
        svi_motion_latent_count: int = 1,
        svi_motion_latent_export_path: Path | str | None = None,
    ) -> "WanVideoRequest":
        health_check_interval = model._validate_tensor_health_check_interval(tensor_health_check_interval)
        if (
            guidance_2 is not guidance_2_unset_sentinel
            and guidance_2 is not None
            and model._wan_config("boundary_ratio", None) is None
        ):
            raise ValueError("guidance_2 is only supported for Wan models with two-transformer boundary routing.")
        svi_mode = svi_anchor_image_path is not None
        WanVideoRequest._validate_svi_request(
            model,
            svi_mode=svi_mode,
            svi_motion_latent_path=svi_motion_latent_path,
            svi_motion_latent_count=svi_motion_latent_count,
            svi_motion_latent_export_path=svi_motion_latent_export_path,
            image_path=image_path,
            last_image_path=last_image_path,
            context_image_paths=context_image_paths,
            video_path=video_path,
        )
        # SVI conditioning IS image-to-video conditioning (the same 36-channel
        # concat route); the anchor takes the image slot end to end.
        is_image_to_video = image_path is not None or svi_mode
        is_video_to_video = video_path is not None
        if is_image_to_video and is_video_to_video:
            raise ValueError("Wan accepts either image_path or video_path, not both.")
        task = "video-to-video" if is_video_to_video else ("image-to-video" if is_image_to_video else "text-to-video")
        if is_image_to_video and not model._supports_image_to_video():
            raise ValueError(f"{model.model_config.model_name} does not support image-to-video input.")
        if video_path is not None and not model._supports_video_to_video():
            raise ValueError(f"{model.model_config.model_name} does not support video-to-video input.")
        # last_image bracket conditioning (0097) exists only on the A14B i2v
        # 36-channel concat path; every other route fails loudly (ADR 0002).
        if last_image_path is not None:
            if not is_image_to_video:
                raise ValueError(
                    "last_image_path requires image_path: Wan first+last bracket conditioning "
                    "is an image-to-video feature."
                )
            if model._uses_expanded_timesteps():
                raise ValueError(
                    f"{model.model_config.model_name} does not support last_image_path: its first-frame "
                    "conditioning (expand_timesteps) has no last-frame slot. Use a Wan A14B "
                    "image-to-video model."
                )
        # Multi-frame context conditioning (0102) shares the same gate class:
        # it extends the A14B i2v conditioned head, so every other route fails
        # loudly (ADR 0002). The head is [image_path, *context_image_paths].
        if context_image_paths:
            if not is_image_to_video:
                raise ValueError(
                    "context_image_paths requires image_path: multi-frame context conditioning extends "
                    "the image-to-video head, and image_path is the first frame of that head."
                )
            if model._uses_expanded_timesteps():
                raise ValueError(
                    f"{model.model_config.model_name} does not support context_image_paths: its first-frame "
                    "conditioning (expand_timesteps) has no multi-frame head slot. Use a Wan A14B "
                    "image-to-video model."
                )
            temporal_scale = int(model.vae.temporal_scale)
            head_frame_count = 1 + len(context_image_paths)
            if len(context_image_paths) % temporal_scale != 0:
                raise ValueError(
                    f"context_image_paths must contain a multiple of {temporal_scale} frames (got "
                    f"{len(context_image_paths)}): the conditioned head [image_path, *context] must fill whole "
                    f"latent groups of the VAE's {temporal_scale}x temporal packing, so the head count must be "
                    f"{temporal_scale}n + 1 (context counts {temporal_scale}, {temporal_scale * 2}, "
                    f"{temporal_scale * 3} = heads 5, 9, 13)."
                )
            if head_frame_count > WanVideoRequest.MAX_CONTEXT_HEAD_FRAMES:
                raise ValueError(
                    f"context_image_paths conditions {head_frame_count} head frames; the supported maximum is "
                    f"{WanVideoRequest.MAX_CONTEXT_HEAD_FRAMES} (community multi-frame handover uses 5-13; a "
                    "longer head leaves too little of the clip free to generate)."
                )
        if context_noise is not None:
            if not context_image_paths:
                raise ValueError(
                    "context_noise requires context_image_paths: it perturbs the multi-frame conditioned head."
                )
            if not (0.0 <= float(context_noise) <= 1000.0):
                raise ValueError(
                    f"context_noise must be within [0, 1000] (a timestep-like scale; ~20 is the community "
                    f"default), got {context_noise}."
                )
        # Validate the mapping mode up front, before any model work.
        resize_mode = DimensionResolver.normalize_resize_mode(resize_mode)
        # In SVI mode the anchor image is the geometry source: it fills the
        # image slot for canvas resolution exactly like an i2v first frame.
        source_image_path = image_path if image_path is not None else svi_anchor_image_path
        resolved_canvas_policy = (
            DimensionResolver.normalize_canvas_policy(canvas_policy) if (source_image_path or video_path) else None
        )
        model._validate_denoisers_available()
        height, width, spatial_metadata = model._resolve_video_spatial_size(
            height=height,
            width=width,
            image_path=source_image_path,
            video_path=video_path,
            canvas_policy=canvas_policy,
        )
        num_frames = model._validated_frame_count(num_frames)
        if svi_mode:
            WanVideoRequest._validate_svi_frames(
                model,
                num_frames=num_frames,
                svi_motion_latent_path=svi_motion_latent_path,
                svi_motion_latent_count=svi_motion_latent_count,
            )
        if last_image_path is not None and num_frames < 2:
            raise ValueError(
                "last_image_path requires at least 2 frames: the bracket needs distinct first and last frames."
            )
        if context_image_paths:
            # At least one whole latent group must remain free beyond the
            # conditioned head (plus the final frame when it is bracketed);
            # a fully conditioned clip would generate nothing.
            head_frame_count = 1 + len(context_image_paths)
            min_frames = head_frame_count + int(model.vae.temporal_scale) + (1 if last_image_path is not None else 0)
            if num_frames < min_frames:
                raise ValueError(
                    f"context_image_paths with a {head_frame_count}-frame conditioned head requires at least "
                    f"{min_frames} frames (head + one free latent group"
                    f"{' + the last-image anchor' if last_image_path is not None else ''}); got {num_frames}."
                )
        if video_strength is not None and not is_video_to_video:
            raise ValueError("video_strength requires video_path.")
        if video_mask_path is not None and not is_video_to_video:
            raise ValueError("video_mask_path requires video_path.")
        video_strength = model._resolve_video_strength(video_strength) if is_video_to_video else None
        video_mask = (
            model._prepare_video_mask(video_mask_path, height=height, width=width, resize_mode=resize_mode)
            if video_mask_path is not None
            else None
        )
        guidance, guidance_2 = model._resolve_guidance_pair(guidance=guidance, guidance_2=guidance_2)
        model._validate_guidance_values(guidance=guidance, guidance_2=guidance_2)
        flow_shift = model._resolve_flow_shift(flow_shift)
        solver = model._resolve_solver(solver)
        model._validate_video_to_video_solver(is_video_to_video=is_video_to_video, solver=solver)
        negative_prompt = model._resolve_negative_prompt(negative_prompt)
        model._validate_runtime_contract(is_image_to_video=is_image_to_video)
        return WanVideoRequest(
            task=task,
            is_image_to_video=is_image_to_video,
            is_video_to_video=is_video_to_video,
            height=height,
            width=width,
            spatial_metadata=spatial_metadata,
            num_frames=num_frames,
            guidance=guidance,
            guidance_2=guidance_2,
            flow_shift=flow_shift,
            solver=solver,
            negative_prompt=negative_prompt,
            video_strength=video_strength,
            video_mask=video_mask,
            health_check_interval=health_check_interval,
            batch_size=1,
            canvas_policy=resolved_canvas_policy,
            resize_mode=resize_mode,
        )

    @staticmethod
    def _validate_svi_request(
        model,
        *,
        svi_mode: bool,
        svi_motion_latent_path,
        svi_motion_latent_count: int,
        svi_motion_latent_export_path,
        image_path,
        last_image_path,
        context_image_paths,
        video_path,
    ) -> None:
        # SVI 2.0 Pro (0103) is one mechanism: the conditioning layout AND the
        # error-recycling LoRA pair belong together. Half-configured requests
        # fail loudly in BOTH directions (ADR 0002): the layouts are mutually
        # unintelligible to the wrong weights (upstream: stock workflow + SVI
        # LoRAs produces garbage, and vice versa).
        svi_pack_loaded = bool(getattr(model, "svi_lora_reports", ()) or ())
        if not svi_mode:
            for name, value in (
                ("svi_motion_latent_path", svi_motion_latent_path),
                ("svi_motion_latent_export_path", svi_motion_latent_export_path),
            ):
                if value is not None:
                    raise ValueError(f"{name} requires svi_anchor_image_path: SVI conditioning is anchored.")
            if svi_motion_latent_count != 1:
                raise ValueError("svi_motion_latent_count requires svi_anchor_image_path and svi_motion_latent_path.")
            if svi_pack_loaded:
                raise ValueError(
                    "This model was constructed with the SVI LoRA pair (svi_lora_high_path/svi_lora_low_path), "
                    "which retrains the conditioning convention: non-SVI generation on these weights would be "
                    "corrupted. Pass svi_anchor_image_path, or construct the model without the SVI pair."
                )
            return
        if model._uses_expanded_timesteps():
            raise ValueError(
                f"{model.model_config.model_name} does not support SVI conditioning: its first-frame "
                "conditioning (expand_timesteps) has no 20-channel conditioning stream. Use a Wan A14B "
                "image-to-video model."
            )
        if not svi_pack_loaded:
            raise ValueError(
                "svi_anchor_image_path requires the SVI LoRA pair: construct the model with "
                "svi_lora_high_path and svi_lora_low_path (running the SVI conditioning layout on stock "
                "weights produces garbage per the upstream SVI warning)."
            )
        for name, value in (
            ("image_path", image_path),
            ("last_image_path", last_image_path),
            ("video_path", video_path),
        ):
            if value is not None:
                raise ValueError(
                    f"svi_anchor_image_path conflicts with {name}: SVI mode derives the whole conditioning "
                    "from the anchor and the optional motion latent. The anchor takes the image slot; "
                    "brackets and video sources have no place in the SVI layout."
                )
        if context_image_paths:
            raise ValueError(
                "svi_anchor_image_path conflicts with context_image_paths: the SVI motion latent replaces the "
                "pixel-frame context head as the momentum carrier (pass svi_motion_latent_path instead)."
            )
        if svi_motion_latent_path is None and svi_motion_latent_count != 1:
            raise ValueError("svi_motion_latent_count requires svi_motion_latent_path (it slices that file).")
        if svi_motion_latent_count < 1:
            raise ValueError(f"svi_motion_latent_count must be at least 1, got {svi_motion_latent_count}.")

    @staticmethod
    def _validate_svi_frames(
        model,
        *,
        num_frames: int,
        svi_motion_latent_path,
        svi_motion_latent_count: int,
    ) -> None:
        from mflux.models.wan.variants.wan_svi import CONTINUE_FRAME_ADVISORY

        temporal_scale = int(model.vae.temporal_scale)
        total_latents = (num_frames - 1) // temporal_scale + 1
        conditioned_latents = 1 + (svi_motion_latent_count if svi_motion_latent_path is not None else 0)
        # At least one whole latent group must remain free beyond the anchor
        # and the motion handover; a fully conditioned clip generates nothing.
        if total_latents - conditioned_latents < 1:
            min_frames = conditioned_latents * temporal_scale + 1
            raise ValueError(
                f"SVI conditioning with {conditioned_latents} conditioned latent slot(s) requires at least "
                f"{min_frames} frames (one free latent group beyond the anchor/motion slots); got {num_frames}."
            )
        if svi_motion_latent_path is not None and num_frames > CONTINUE_FRAME_ADVISORY:
            print(
                f"⚠️  SVI continuation segments beyond {CONTINUE_FRAME_ADVISORY} frames showed per-window "
                f"color shifts in community runs (trained-length effect); requested {num_frames}. "
                "Consider chaining shorter segments."
            )
