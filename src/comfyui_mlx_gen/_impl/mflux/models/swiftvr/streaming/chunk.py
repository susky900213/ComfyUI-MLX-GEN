"""Fixed-size causal chunk protocol for SwiftVR restoration.

A clip of ``total_frames = 4a + 1`` frames is split into one FIRST chunk, zero or
more MIDDLE chunks and one LAST chunk so that the number of decoded output frames
equals the number of source frames. ``clip_len`` is the MIDDLE chunk size and must
be a multiple of 4, because ReAE compresses 4 pixel frames into 1 latent frame.

Frame accounting (with ``n_lat = clip_len // 4``):

* FIRST  - ``clip_len + 4`` frames  -> ``n_lat + 1`` latents -> ``clip_len + 1``
  output frames, after the decoder's ``frames_to_trim`` head frames are dropped.
* MIDDLE - ``clip_len`` frames      -> ``n_lat`` latents     -> ``clip_len`` output frames.
* LAST   - ``4b + 1`` frames        -> ``b + 1`` latents     -> ``4b + 4`` output frames,
  because the codec replicates the final source frame three times before encoding.

Ported from ``swiftvr/streaming/chunk.py`` of the upstream SwiftVR reference. The
upstream ``assert`` on ``clip_len`` is raised as a ``ValueError`` here (ADR 0002:
fail closed with an actionable message, and stay effective under ``python -O``),
and the ``t = 4a + 1`` precondition that upstream enforces in its caller is
checked here as well.
"""

from dataclasses import dataclass
from enum import Enum

LATENT_TEMPORAL_DOWNSCALE = 4


class ChunkType(Enum):
    """Position of a chunk inside the clip. Determines codec framing, not the DiT."""

    FIRST = "first"
    MIDDLE = "middle"
    LAST = "last"


@dataclass(frozen=True)
class ChunkSpec:
    """One unit of work for the streaming codec and the one-step DiT.

    Attributes:
        ctype: Position of the chunk inside the clip.
        frame_start: Index of the chunk's first source frame within the clip.
        frame_count: Number of source frames consumed by the chunk.
        b: LAST chunks only - the chunk holds ``4b + 1`` source frames and produces
            ``b + 1`` latent frames. Zero and meaningless for FIRST and MIDDLE.
        clip_idx: Zero-based position of this chunk in the emitted sequence.
        is_first_decode: Whether the decoder must drop its causal-padding head
            frames after decoding this chunk. True for the first chunk only.
    """

    ctype: ChunkType
    frame_start: int
    frame_count: int
    b: int
    clip_idx: int
    is_first_decode: bool

    @property
    def latent_count(self) -> int:
        """Latent frames the encoder produces for this chunk."""
        if self.ctype == ChunkType.LAST:
            # The codec replicates the final frame three times, so 4b + 1 source
            # frames are encoded as 4b + 4 and yield b + 1 latents.
            return self.b + 1
        return self.frame_count // LATENT_TEMPORAL_DOWNSCALE


def aligned_frame_count(raw_total: int) -> int:
    """Largest ``4a + 1`` frame count not exceeding ``raw_total``.

    Raises:
        ValueError: If ``raw_total`` is smaller than one frame.
    """
    if raw_total < 1:
        raise ValueError(f"SwiftVR requires at least one source frame, got {raw_total}.")
    return LATENT_TEMPORAL_DOWNSCALE * ((raw_total - 1) // LATENT_TEMPORAL_DOWNSCALE) + 1


def build_chunk_specs(total_frames: int, clip_len: int) -> list[ChunkSpec]:
    """Split a ``4a + 1`` frame clip into FIRST / MIDDLE / LAST chunks.

    Args:
        total_frames: Source frame count of the clip. Must satisfy ``t % 4 == 1``;
            use :func:`aligned_frame_count` to trim an arbitrary count first.
        clip_len: MIDDLE chunk size in source frames. Must be a multiple of 4.

    Returns:
        The chunk specs in playback order. The sum of ``frame_count`` equals
        ``total_frames``.

    Raises:
        ValueError: If ``clip_len`` is not a positive multiple of 4, or if
            ``total_frames`` is not a positive count of the form ``4a + 1``.
    """
    if clip_len <= 0 or clip_len % LATENT_TEMPORAL_DOWNSCALE != 0:
        raise ValueError(
            f"SwiftVR clip_len must be a positive multiple of 4, got {clip_len}. "
            "ReAE compresses 4 pixel frames into 1 latent frame, so a MIDDLE chunk "
            "that is not a multiple of 4 cannot form whole temporal groups."
        )
    # Checked before the 4a + 1 test, which a negative count can satisfy: Python's modulo
    # follows the divisor's sign, so -3 % 4 == 1 and the plan would come back describing
    # a chunk of -3 frames. Unreachable through aligned_frame_count, which is exactly why
    # a caller reaching it directly deserves an error rather than a nonsense plan.
    if total_frames < 1:
        raise ValueError(f"SwiftVR requires at least one source frame, got {total_frames}.")
    if total_frames % LATENT_TEMPORAL_DOWNSCALE != 1:
        raise ValueError(
            f"SwiftVR requires a clip length of the form 4a + 1, got {total_frames}. "
            f"Trim to {aligned_frame_count(total_frames)} frames with aligned_frame_count()."
        )

    if total_frames <= clip_len + 4:
        return [
            ChunkSpec(
                ctype=ChunkType.LAST,
                frame_start=0,
                frame_count=total_frames,
                b=(total_frames - 1) // LATENT_TEMPORAL_DOWNSCALE,
                clip_idx=0,
                is_first_decode=True,
            )
        ]

    specs = [
        ChunkSpec(
            ctype=ChunkType.FIRST,
            frame_start=0,
            frame_count=clip_len + 4,
            b=0,
            clip_idx=0,
            is_first_decode=True,
        )
    ]

    remaining = total_frames - (clip_len + 4)
    position = clip_len + 4
    clip_index = 1
    while remaining > 0:
        if remaining <= clip_len:
            specs.append(
                ChunkSpec(
                    ctype=ChunkType.LAST,
                    frame_start=position,
                    frame_count=remaining,
                    b=(remaining - 1) // LATENT_TEMPORAL_DOWNSCALE,
                    clip_idx=clip_index,
                    is_first_decode=False,
                )
            )
            break
        specs.append(
            ChunkSpec(
                ctype=ChunkType.MIDDLE,
                frame_start=position,
                frame_count=clip_len,
                b=0,
                clip_idx=clip_index,
                is_first_decode=False,
            )
        )
        remaining -= clip_len
        position += clip_len
        clip_index += 1

    return specs
