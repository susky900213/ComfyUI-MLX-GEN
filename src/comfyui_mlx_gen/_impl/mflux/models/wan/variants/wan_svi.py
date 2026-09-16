"""SVI 2.0 Pro conditioning support for the Wan 2.2 A14B image-to-video route (0103).

Stable Video Infinity 2.0 Pro (ICLR'26 Oral, vita-epfl/Stable-Video-Infinity,
`svi_wan22` branch) conditions every clip of a chain on

    y = concat([mask, anchor_latent, motion_latent, zero_latents])

along the temporal latent axis, where:

- `anchor_latent` is the VAE encode of ONE persistent anchor image (the
  user-given first frame), re-injected into EVERY clip of the chain - the
  identity mechanism;
- `motion_latent` is the last temporal entr(y/ies) of the PREVIOUS clip's
  final denoised latent tensor (never re-encoded from decoded pixels) - the
  momentum mechanism; absent on the first clip;
- the padding is genuinely zero-valued LATENTS. Stock Wan i2v instead
  VAE-encodes a zero-padded pixel video, which yields non-zero padding
  latents: the two conventions are NOT interchangeable, and the SVI LoRA
  pair (error-recycling fine-tune) is what teaches the model this layout.
  Running one convention with the other's weights produces garbage (pinned
  upstream warning), which is why SVI mode and the SVI LoRA pair are gated
  together, loudly, in both directions.

The mask is the standard first-frame i2v mask (only the frame-0 group is
marked conditioned); the motion latent rides at temporal position 1 with
mask=0 - the fine-tuned model reads it positionally. Decoded continuation
clips therefore start with 1 anchor-restoration frame plus temporal_scale x
motion_latent_count frames that re-render the predecessor's tail: assembly
must drop `1 + temporal_scale * count` frames from every continuation clip
(the authors stitch with the first five frames removed for count=1).

Reference implementation: diffsynth/pipelines/wan_video_svi_pro.py at
vita-epfl/Stable-Video-Infinity@7dac0f9 (WanVideoUnit_ImageEmbedderVAE).
"""

from pathlib import Path

import mlx.core as mx

MOTION_LATENT_KEY = "latents"
# Continue segments beyond 65 frames showed per-window color shifts in
# community SVI runs (trained-length effect); warn, do not block.
CONTINUE_FRAME_ADVISORY = 65


class WanSvi:
    @staticmethod
    def build_condition(
        model,
        *,
        anchor_image_path: Path | str,
        motion_latent_path: Path | str | None,
        motion_latent_count: int,
        height: int,
        width: int,
        num_frames: int,
        batch_size: int,
        resize_mode: str,
    ) -> mx.array:
        # The anchor is encoded ALONE (one latent). The Wan VAE is causal, so
        # this equals the first temporal latent of any padded encode of the
        # same frame; encoding it alone matches the reference exactly.
        anchor_latent = model._load_first_frame_condition(
            image_path=anchor_image_path,
            height=height,
            width=width,
            resize_mode=resize_mode,
        )
        temporal_scale = int(model.vae.temporal_scale)
        total_latents = (num_frames - 1) // temporal_scale + 1
        latent_height = anchor_latent.shape[3]
        latent_width = anchor_latent.shape[4]
        sections = [anchor_latent.astype(mx.float32)]
        conditioned_latents = 1
        if motion_latent_path is not None:
            motion_latents = WanSvi.load_motion_latents(
                motion_latent_path,
                count=motion_latent_count,
                z_dim=int(model.vae.z_dim),
                latent_height=latent_height,
                latent_width=latent_width,
            )
            sections.append(motion_latents)
            conditioned_latents += motion_latent_count
        padding_latents = total_latents - conditioned_latents
        # Zero LATENTS by design (see module docstring); this is the layout
        # the SVI LoRA was trained on, not an approximation of stock i2v.
        sections.append(
            mx.zeros(
                (batch_size, anchor_latent.shape[1], padding_latents, latent_height, latent_width),
                dtype=mx.float32,
            )
        )
        latent_condition = mx.concatenate(sections, axis=2)
        # Standard first-frame mask: SVI marks ONLY the frame-0 group as
        # conditioned; the motion latent stays mask=0 (positional, learned).
        mask = model._packed_condition_mask(
            batch_size=batch_size,
            num_frames=num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            head_frame_count=1,
            last_frame_conditioned=False,
        )
        condition = mx.concatenate([mask[:, :, :total_latents], latent_condition], axis=1)
        mx.eval(condition)
        return condition.astype(mx.float32)

    @staticmethod
    def load_motion_latents(
        path: Path | str,
        *,
        count: int,
        z_dim: int,
        latent_height: int,
        latent_width: int,
    ) -> mx.array:
        resolved = Path(path)
        if not resolved.exists():
            raise ValueError(f"svi_motion_latent_path does not exist: {resolved}")
        arrays, _metadata = mx.load(str(resolved), return_metadata=True)
        if MOTION_LATENT_KEY not in arrays:
            raise ValueError(
                f"svi_motion_latent_path {resolved} has no '{MOTION_LATENT_KEY}' tensor; expected a file "
                "exported by a previous SVI run (svi_motion_latent_export)."
            )
        latents = arrays[MOTION_LATENT_KEY]
        if latents.ndim != 4 or latents.shape[0] != z_dim:
            raise ValueError(
                f"svi_motion_latent_path {resolved} holds a tensor of shape {tuple(latents.shape)}; expected "
                f"[{z_dim}, latent_frames, latent_height, latent_width] as exported by a previous SVI run."
            )
        if (latents.shape[2], latents.shape[3]) != (latent_height, latent_width):
            raise ValueError(
                f"svi_motion_latent_path {resolved} was exported for a {latents.shape[3] * 8}x"
                f"{latents.shape[2] * 8} canvas, but this run resolves {latent_width * 8}x{latent_height * 8}. "
                "SVI chains must keep one canvas end to end; re-run with the matching width/height."
            )
        if latents.shape[1] < count:
            raise ValueError(
                f"svi_motion_latent_count={count} exceeds the {latents.shape[1]} temporal entries stored in {resolved}."
            )
        # The LAST entries carry the predecessor's ending motion (reference:
        # prev_last_latent[:, -num_motion_latent:]).
        return latents[:, -count:].astype(mx.float32)[None, ...]

    @staticmethod
    def export_motion_latents(
        path: Path | str,
        *,
        latents: mx.array,
        width: int,
        height: int,
        num_frames: int,
        model_name: str,
    ) -> Path:
        resolved = Path(path)
        if resolved.suffix != ".safetensors":
            raise ValueError(f"svi_motion_latent_export_path must end with .safetensors, got: {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        # The FULL final denoised latent tensor is exported (a few MB); the
        # trailing-slice count stays a load-time decision of the NEXT clip.
        # float32 preserves the exact scheduler state (pre decode cast).
        tensor = latents[0].astype(mx.float32)
        mx.eval(tensor)
        mx.save_safetensors(
            str(resolved),
            {MOTION_LATENT_KEY: tensor},
            metadata={
                "svi_export": "wan-svi-2.0-pro",
                "model": model_name,
                "width": str(width),
                "height": str(height),
                "num_frames": str(num_frames),
            },
        )
        return resolved

    @staticmethod
    def assembly_trim_frames(*, temporal_scale: int, motion_latent_count: int, is_continuation: bool) -> int:
        # Continuation clips re-render the anchor-restoration frame (1) plus
        # the motion latent's pixel frames (temporal_scale x count); assembly
        # drops them to avoid duplicating the predecessor's tail. First clips
        # keep every frame.
        if not is_continuation:
            return 0
        return 1 + temporal_scale * motion_latent_count
