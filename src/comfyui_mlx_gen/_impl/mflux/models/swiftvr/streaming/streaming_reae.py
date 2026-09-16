"""Causal streaming wrapper around ReAE.

The encoder and decoder are run chunk by chunk while the MemBlock and TPool boundary
buffers are carried across calls, so the result is identical to encoding or decoding the
whole clip at once. Dropping that carry does not raise: with the same weights and the
same trim flags it simply produces seams (measured at max 0.66 on ``[0, 1]`` pixels at a
chunk boundary), which is why :mod:`tests.swiftvr` keeps a negative control for it.

Two deliberate divergences from ``swiftvr/streaming/tae.py``:

1. State is mutated in place per layer slot instead of being rebuilt into a fresh dict
   per call. Upstream replaces the entire state object, so an early return from an
   unsatisfied TPool discards the MemBlock carries of every layer after it - unreachable
   upstream because both entry points force multiples of 4, but a real latent bug. For
   every input length upstream supports the two are identical, because each stateful slot
   is written exactly once per call.
2. Encoding and decoding are sliced internally. This is not an optimisation: at 1920x1088
   a whole-chunk decode of 7 latents peaks at 15.95 GiB while one latent frame at a time
   peaks at 3.84 GiB and is also faster. The decoder is exactly sliceable to 1 latent
   frame (1.3e-06) and the encoder to 4 pixel frames (bit-exact). This mirrors
   ``WanVae2_2.iter_decode_slices``, the in-repo precedent for a generator carrying
   convolution boundary state.

Pixel values are ``[0, 1]`` throughout, and ReAE takes no latent normalization - Wan's
``latents * std + mean`` must NOT be applied here.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import mlx.core as mx
from mlx import nn

from mflux.models.swiftvr.model.swiftvr_reae.reae import ReAE
from mflux.models.swiftvr.model.swiftvr_reae.reae_blocks import (
    MemBlock,
    TPool,
    pixel_shuffle_nhwc,
    pixel_unshuffle_nhwc,
)
from mflux.models.swiftvr.streaming.chunk import ChunkSpec, ChunkType

MIN_ENCODE_SLICE_FRAMES = 4


@dataclass
class ReAEStackState:
    """Per-layer causal carry for one ReAE stack.

    Slots are indexed by the layer's position in the stack, which is the same integer as
    the checkpoint key index, so a state dump reads against the checkpoint directly.

    Attributes:
        previous_frame: For each MemBlock slot, that block's own last *input* frame.
        pending_frames: For each TPool slot, frames that did not complete a temporal group.
    """

    previous_frame: list[mx.array | None] = field(default_factory=list)
    pending_frames: list[mx.array | None] = field(default_factory=list)

    @classmethod
    def for_stack(cls, layers: Sequence[nn.Module]) -> "ReAEStackState":
        """Allocate one empty slot per layer, stateful or not."""
        return cls(previous_frame=[None] * len(layers), pending_frames=[None] * len(layers))

    def reset(self) -> None:
        """Clear every carry, returning the stack to its start-of-clip behaviour."""
        self.previous_frame = [None] * len(self.previous_frame)
        self.pending_frames = [None] * len(self.pending_frames)

    def live_arrays(self) -> list[mx.array]:
        """Every materialized carry, for evaluation and memory accounting."""
        return [array for array in (*self.previous_frame, *self.pending_frames) if array is not None]


def run_reae_stack(
    layers: Sequence[nn.Module],
    x: mx.array,
    state: ReAEStackState,
    *,
    batch: int = 1,
) -> mx.array | None:
    """Run one ReAE stack over ``x``, threading and updating the causal state in place.

    Args:
        layers: The encoder or decoder layer list.
        x: ``[B, T, H, W, C]`` input frames or latents.
        state: Boundary state for this stack, mutated in place.
        batch: ``B``. Kept explicit so the frame axis can be recovered after layers that
            change the temporal length.

    Returns:
        ``[B, T', H', W', C']``, or ``None`` when a TPool could not complete a temporal
        group and the input was buffered instead.

    Raises:
        ValueError: If the input is not rank 5, if the batch disagrees, or if the state
            was allocated for a stack of a different length.
    """
    if x.ndim != 5:
        raise ValueError(f"ReAE stacks take [B, T, H, W, C] input, got shape {x.shape}.")
    if x.shape[0] != batch:
        raise ValueError(f"ReAE stack batch mismatch: input carries {x.shape[0]}, caller declared {batch}.")
    if len(state.previous_frame) != len(layers):
        raise ValueError(
            f"ReAE state was allocated for {len(state.previous_frame)} layers but the stack has {len(layers)}."
        )

    hidden = x.reshape(batch * x.shape[1], *x.shape[2:])
    for index, layer in enumerate(layers):
        if isinstance(layer, MemBlock):
            folded, height, width, channels = hidden.shape
            frames = folded // batch
            framed = hidden.reshape(batch, frames, height, width, channels)
            carry = state.previous_frame[index]
            head = carry if carry is not None else mx.zeros_like(framed[:, :1])
            past = mx.concatenate([head, framed[:, :-1]], axis=1)
            # The carry is captured from this block's INPUT, before the block runs.
            state.previous_frame[index] = framed[:, -1:]
            hidden = layer(hidden, past.reshape(folded, height, width, channels))
        elif isinstance(layer, TPool):
            folded, height, width, channels = hidden.shape
            frames = folded // batch
            framed = hidden.reshape(batch, frames, height, width, channels)
            pending = state.pending_frames[index]
            if pending is not None:
                framed = mx.concatenate([pending, framed], axis=1)
                frames = framed.shape[1]
            full = (frames // layer.stride) * layer.stride
            state.pending_frames[index] = framed[:, full:] if frames > full else None
            if full == 0:
                mx.eval(*state.live_arrays())
                return None
            hidden = layer(framed[:, :full].reshape(batch * full, height, width, channels))
        else:
            hidden = layer(hidden)

    output = hidden.reshape(batch, hidden.shape[0] // batch, *hidden.shape[1:])
    # Materialize the carries with the output so the boundary state does not pin the
    # whole chunk's lazy graph until the next call.
    mx.eval(output, *state.live_arrays())
    return output


class ReAEStreamingCodec:
    """Public streaming surface over a :class:`ReAE` graph.

    Owns the two stack states, the incremental encoder's leftover buffer and the one-shot
    decoder head trim. One instance restores one clip; call :meth:`reset` before the next.
    """

    def __init__(
        self,
        model: ReAE,
        *,
        encode_slice_frames: int = MIN_ENCODE_SLICE_FRAMES,
        decode_slice_latents: int = 1,
        clear_cache_each_slice: bool = False,
    ) -> None:
        """Wrap a :class:`ReAE` graph for causal chunk-by-chunk restoration.

        Args:
            model: The ReAE graph. Weights must already be applied.
            encode_slice_frames: Pixel frames per encoder call. Must be a positive
                multiple of 4, the smallest slice that keeps both stride-2 TPools fed.
            decode_slice_latents: Latent frames per decoder call. One is both the
                smallest peak and the fastest at 1080p.
            clear_cache_each_slice: Release the MLX buffer cache after every evaluated
                slice. Follows ``WanVae2_2.iter_decode_slices``: the per-slice ``mx.eval``
                already stops the lazy graph accumulating, and this trades allocator reuse
                for a lower floor when memory is the binding constraint.

        Raises:
            ValueError: If either slice size is out of range.
        """
        if encode_slice_frames < MIN_ENCODE_SLICE_FRAMES or encode_slice_frames % MIN_ENCODE_SLICE_FRAMES:
            raise ValueError(
                f"encode_slice_frames must be a multiple of {MIN_ENCODE_SLICE_FRAMES} and at least "
                f"{MIN_ENCODE_SLICE_FRAMES}, got {encode_slice_frames}; a shorter slice leaves the "
                "second TPool without a full temporal group."
            )
        if decode_slice_latents < 1:
            raise ValueError(f"decode_slice_latents must be positive, got {decode_slice_latents}.")
        self.model = model
        self.encode_slice_frames = encode_slice_frames
        self.decode_slice_latents = decode_slice_latents
        self.clear_cache_each_slice = clear_cache_each_slice
        self._encoder_state = ReAEStackState.for_stack(model.encoder.layers)
        self._decoder_state = ReAEStackState.for_stack(model.decoder.layers)
        self._encoder_leftover: mx.array | None = None
        self._decode_started = False
        self._pending_head_trim = model.frames_to_trim

    @property
    def is_first_decode(self) -> bool:
        """Whether the next decode is the clip's first, and must drop the causal head frames."""
        return not self._decode_started

    def reset(self) -> None:
        """Clear all causal state and re-arm the decoder head trim."""
        self._encoder_state.reset()
        self._decoder_state.reset()
        self._encoder_leftover = None
        self._decode_started = False
        self._pending_head_trim = self.model.frames_to_trim

    def encode_chunk(self, frames: mx.array, spec: ChunkSpec) -> mx.array:
        """Encode one fixed-protocol chunk.

        LAST chunks hold ``4b + 1`` frames; the final frame is replicated three times so
        the encoder sees ``4b + 4`` and emits ``b + 1`` latents.

        Args:
            frames: ``[B, T, H, W, 3]`` in ``[0, 1]``, spatial dims a multiple of
                ``model.spatial_scale``.
            spec: The chunk this call corresponds to.

        Returns:
            ``[B, T_lat, H // spatial_scale, W // spatial_scale, latent_channels]``.

        Raises:
            ValueError: If the frame count disagrees with ``spec``, if the layout is
                wrong, or if the encoder buffered everything (which the fixed protocol
                makes impossible).
        """
        self._validate_frames(frames)
        if frames.shape[1] != spec.frame_count:
            raise ValueError(
                f"Chunk {spec.clip_idx} ({spec.ctype.value}) expects {spec.frame_count} frames, got {frames.shape[1]}."
            )
        if spec.ctype is ChunkType.LAST:
            batch, _, height, width, channels = frames.shape
            tail = mx.broadcast_to(frames[:, -1:], (batch, 3, height, width, channels))
            frames = mx.concatenate([frames, tail], axis=1)

        latents = self._encode_frames(frames)
        if latents is None:
            raise ValueError(
                f"Chunk {spec.clip_idx} ({spec.ctype.value}) produced no latents; "
                f"{frames.shape[1]} frames did not complete a temporal group."
            )
        expected = spec.latent_count
        if latents.shape[1] != expected:
            raise ValueError(
                f"Chunk {spec.clip_idx} ({spec.ctype.value}) produced {latents.shape[1]} latents, expected {expected}."
            )
        return latents

    def encode_chunk_incremental(self, frames: mx.array) -> mx.array | None:
        """Encode an arbitrary number of frames, buffering the ``T % 4`` remainder.

        Returns:
            The latents for the frames that completed temporal groups, or ``None`` when
            everything was buffered.
        """
        self._validate_frames(frames)
        if self._encoder_leftover is not None:
            frames = mx.concatenate([self._encoder_leftover, frames], axis=1)
            self._encoder_leftover = None
        remainder = frames.shape[1] % MIN_ENCODE_SLICE_FRAMES
        if remainder:
            keep = frames.shape[1] - remainder
            self._encoder_leftover = frames[:, keep:]
            mx.eval(self._encoder_leftover)
            if keep == 0:
                return None
            frames = frames[:, :keep]
        return self._encode_frames(frames)

    def flush_encoder(self) -> mx.array | None:
        """Encode any buffered tail, padding it to a multiple of 4 by frame replication."""
        if self._encoder_leftover is None:
            return None
        frames = self._encoder_leftover
        self._encoder_leftover = None
        remainder = frames.shape[1] % MIN_ENCODE_SLICE_FRAMES
        if remainder:
            batch, _, height, width, channels = frames.shape
            pad = MIN_ENCODE_SLICE_FRAMES - remainder
            tail = mx.broadcast_to(frames[:, -1:], (batch, pad, height, width, channels))
            frames = mx.concatenate([frames, tail], axis=1)
        return self._encode_frames(frames)

    def iter_decode_chunk(self, latents: mx.array, spec: ChunkSpec | None = None) -> Iterator[mx.array]:
        """Decode latents in slices, yielding ``[B, T_out, H * scale, W * scale, 3]`` in ``[0, 1]``.

        The first ``model.frames_to_trim`` frames of the clip are the decoder's causal
        padding head and are dropped once, spread across as many slices as needed.

        Args:
            latents: ``[B, T_lat, H, W, latent_channels]``.
            spec: Optional chunk this call corresponds to. When given, its
                ``is_first_decode`` must agree with the codec's own state.

        Raises:
            ValueError: On a layout mismatch, or when ``spec`` disagrees about the head trim.
        """
        self._validate_latents(latents)
        if spec is not None and spec.is_first_decode != self.is_first_decode:
            raise ValueError(
                f"Chunk {spec.clip_idx} declares is_first_decode={spec.is_first_decode} but the codec "
                f"reports {self.is_first_decode}; the decoder head trim has two disagreeing sources."
            )
        # Validate and consume the first-decode flag now rather than on the first `next`,
        # so a caller that builds the iterator before draining it cannot observe stale
        # state or a deferred error.
        self._decode_started = True
        return self._iter_decode_slices(latents)

    def _iter_decode_slices(self, latents: mx.array) -> Iterator[mx.array]:
        batch = latents.shape[0]
        for start in range(0, latents.shape[1], self.decode_slice_latents):
            piece = latents[:, start : start + self.decode_slice_latents]
            decoded = run_reae_stack(self.model.decoder.layers, piece, self._decoder_state, batch=batch)
            if decoded is None:
                continue
            frames = self._postprocess(decoded)
            if self._pending_head_trim:
                dropped = min(self._pending_head_trim, frames.shape[1])
                self._pending_head_trim -= dropped
                frames = frames[:, dropped:]
                if frames.shape[1] == 0:
                    continue
            mx.eval(frames)
            yield frames
            del frames
            del decoded
            if self.clear_cache_each_slice:
                mx.clear_cache()

    def decode_chunk(self, latents: mx.array, spec: ChunkSpec | None = None) -> mx.array | None:
        """Concatenation of :meth:`iter_decode_chunk`. Prefer the iterator at high resolution."""
        pieces = list(self.iter_decode_chunk(latents, spec))
        if not pieces:
            return None
        return mx.concatenate(pieces, axis=1)

    def _encode_frames(self, frames: mx.array) -> mx.array | None:
        batch = frames.shape[0]
        pieces: list[mx.array] = []
        for start in range(0, frames.shape[1], self.encode_slice_frames):
            piece = frames[:, start : start + self.encode_slice_frames]
            patched = self._unshuffle(piece)
            latents = run_reae_stack(self.model.encoder.layers, patched, self._encoder_state, batch=batch)
            if latents is not None:
                pieces.append(latents)
            if self.clear_cache_each_slice:
                mx.clear_cache()
        if not pieces:
            return None
        return mx.concatenate(pieces, axis=1)

    def _unshuffle(self, frames: mx.array) -> mx.array:
        batch, count, height, width, channels = frames.shape
        folded = pixel_unshuffle_nhwc(frames.reshape(batch * count, height, width, channels), self.model.patch_size)
        return folded.reshape(batch, count, *folded.shape[1:])

    def _postprocess(self, decoded: mx.array) -> mx.array:
        batch, count, height, width, channels = decoded.shape
        clipped = mx.clip(decoded, 0.0, 1.0)
        folded = pixel_shuffle_nhwc(clipped.reshape(batch * count, height, width, channels), self.model.patch_size)
        return folded.reshape(batch, count, *folded.shape[1:])

    def _validate_frames(self, frames: mx.array) -> None:
        if frames.ndim != 5:
            raise ValueError(f"ReAE expects [B, T, H, W, C] pixel frames, got shape {frames.shape}.")
        if frames.shape[-1] != self.model.image_channels:
            raise ValueError(
                f"ReAE expects {self.model.image_channels} channels last, got {frames.shape[-1]}; "
                "convert from [B, T, C, H, W] before calling."
            )
        scale = self.model.spatial_scale
        if frames.shape[2] % scale or frames.shape[3] % scale:
            raise ValueError(
                f"ReAE requires pixel dimensions divisible by {scale}, got {frames.shape[2]}x{frames.shape[3]}."
            )

    def _validate_latents(self, latents: mx.array) -> None:
        if latents.ndim != 5:
            raise ValueError(f"ReAE expects [B, T, H, W, C] latents, got shape {latents.shape}.")
        if latents.shape[-1] != self.model.latent_channels:
            raise ValueError(
                f"ReAE expects {self.model.latent_channels} latent channels last, got {latents.shape[-1]}."
            )
