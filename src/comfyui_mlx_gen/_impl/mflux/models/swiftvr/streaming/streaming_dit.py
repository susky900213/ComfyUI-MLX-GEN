"""One-step chunked DiT driver for SwiftVR.

SwiftVR collapses iterative diffusion to a single forward pass taken at the fully
degraded endpoint of the flow, so restoration is exactly

    ``z_hq = z_lq - velocity(z_lq, t = INFERENCE_TIMESTEP)``

There is no sampler, no sigma schedule, no classifier-free guidance and no negative
prompt; there is also no text encoder, because SwiftVR ships a frozen 512-token prompt
embedding that cross-attention runs against. Nothing here is causal at the attention
level - every attention call is non-causal with no mask and no KV cache. Causality
across chunks comes from exactly two places: the RoPE temporal offset accumulated here,
and ReAE's convolution boundary state.

Ported from ``swiftvr/streaming/dit.py``. Three deliberate divergences, each recorded at
its call site:

1. Upstream's crossfade writes a linear ramp into ``den_ext[:, :, :ol]`` and then
   returns ``den_ext[:, :, ol:]`` (``dit.py``, ``StreamingDiT.denoise``). Those two lines
   are disjoint, so the ramp never reaches the returned value and never reaches the
   ``_prev_out`` carry either, which is taken from ``den_out``. Overlap here therefore
   does what it measurably does upstream - extend the chunk backwards for temporal
   context and shift the rotary offset to match - and does not carry a restored-latent
   buffer that nothing can read.
2. :meth:`denoise_last_chunk` clears the overlap carry. Upstream leaves it, which would
   feed a stale two-chunks-old prefix to a subsequent :meth:`denoise` at the wrong
   rotary offset. Unreachable in the supported flow, where LAST ends the clip.
3. Conditioning is cached as the broadcast prompt embedding and the constant timestep,
   not as the condition embedder's output. Caching ``temb`` / ``timestep_proj`` /
   ``encoder_hidden_states`` the way upstream's ``_precompute_cond`` does needs an
   injection point in ``WanTransformer.__call__``, which builds them internally; see the
   note on :meth:`_conditioning`.
"""

import math
from dataclasses import dataclass

import mlx.core as mx

from mflux.models.swiftvr.model.swiftvr_transformer.swiftvr_transformer import SwiftVRTransformer
from mflux.models.swiftvr.streaming.chunk import ChunkSpec, ChunkType
from mflux.models.wan.model.wan_transformer.wan_transformer import WanBlockHealthContext

# The conditioning timestep is a constant: one step, taken at the flow endpoint. This is
# the module default; the catalog carries the same value as swiftvr_inference_timestep and
# the SwiftVR runtime passes that one through, so a run reports what it actually used.
INFERENCE_TIMESTEP = 1000.0


@dataclass(frozen=True)
class _Conditioning:
    """Chunk-invariant DiT inputs, built once and reused for every chunk of a clip.

    Attributes:
        timestep: ``[B]`` constant endpoint timestep.
        prompt_embeds: ``[B, S, text_dim]`` frozen prompt embedding, broadcast to the batch.
    """

    timestep: mx.array
    prompt_embeds: mx.array


class StreamingDiT:
    """Drives :class:`SwiftVRTransformer` across a clip's chunks.

    State is the global temporal offset ``latent_offset`` - the number of latent frames
    already consumed, which becomes the next chunk's rotary offset - plus, when
    ``dit_overlap`` is positive, the tail of the previous chunk's low-quality latents.
    """

    def __init__(
        self,
        transformer: SwiftVRTransformer,
        *,
        dit_overlap: int = 0,
        inference_timestep: float = INFERENCE_TIMESTEP,
        clear_cache_each_block: bool = True,
    ) -> None:
        """Create the driver.

        Args:
            transformer: The SwiftVR DiT wrapper, with MFSWA already installed.
            dit_overlap: Latent frames of the previous chunk prepended to each chunk for
                temporal context. The prefix is dropped from the result, so the emitted
                frame count is unchanged; the cost is one extra latent frame of sequence
                per overlap frame. The offline route runs 0; upstream's streaming session
                runs 1. This path has no runtime evidence on this backend yet.
            inference_timestep: The constant conditioning timestep. Defaults to
                :data:`INFERENCE_TIMESTEP`; the SwiftVR runtime passes the catalog's
                ``swiftvr_inference_timestep`` so the value a run used is the value its
                metadata reports.
            clear_cache_each_block: Evaluate and release the MLX cache after each block.
                Effectively mandatory at 1080p, where a shifted layer materializes
                hundreds of megabytes of gathered query/key/value.

        Raises:
            ValueError: If ``dit_overlap`` is negative or ``inference_timestep`` is not
                a finite positive value.
        """
        if dit_overlap < 0:
            raise ValueError(f"dit_overlap must be zero or positive, got {dit_overlap}.")
        if not math.isfinite(inference_timestep) or inference_timestep <= 0:
            raise ValueError(f"inference_timestep must be finite and positive, got {inference_timestep}.")
        self.transformer = transformer
        self.dit_overlap = int(dit_overlap)
        self.inference_timestep = float(inference_timestep)
        self.clear_cache_each_block = clear_cache_each_block
        self.latent_offset = 0
        # Underscore-prefixed: per-clip carry and caches, never parameters.
        self._previous_lq: mx.array | None = None
        self._conditioning_cache: _Conditioning | None = None
        self._conditioning_key: tuple | None = None

    def reset(self) -> None:
        """Return to the start of a clip.

        The conditioning cache is keyed and clip-independent, so it survives.
        """
        self.latent_offset = 0
        self._previous_lq = None
        self.transformer.reset_chunk_state()

    def denoise(
        self,
        latents: mx.array,
        prompt_embeds: mx.array,
        *,
        block_health_context: WanBlockHealthContext | None = None,
    ) -> mx.array:
        """Restore one FIRST or MIDDLE chunk of latents and advance the temporal offset.

        Args:
            latents: ``[B, C, F, H, W]`` low-quality latents.
            prompt_embeds: ``[B, 512, text_dim]`` or ``[512, text_dim]`` frozen prompt
                embedding.
            block_health_context: Forwarded to the Wan block-health probes.

        Returns:
            ``[B, C, F, H, W]`` restored latents for this chunk's frames only.

        Raises:
            ValueError: If ``latents`` is not rank 5.
        """
        self._validate_latents(latents, name="latents")
        frame_count = latents.shape[2]
        conditioning = self._conditioning(latents.shape[0], prompt_embeds, latents.dtype)

        overlap = 0
        if self.dit_overlap > 0 and self._previous_lq is not None:
            overlap = self._previous_lq.shape[2]
            extended = mx.concatenate([self._previous_lq, latents], axis=2)
            rotary_offset = self.latent_offset - overlap
        else:
            extended = latents
            rotary_offset = self.latent_offset

        restored = self._restore(
            extended,
            conditioning=conditioning,
            t_offset=rotary_offset,
            block_health_context=block_health_context,
        )
        if overlap:
            # Upstream ramps the prefix here and then slices it away; see the module
            # docstring. Only the slice survives, so only the slice is reproduced.
            restored = restored[:, :, overlap:]

        self._previous_lq = self._overlap_carry(latents)
        self.latent_offset += frame_count
        self._settle(restored)
        return restored

    def denoise_last_chunk(
        self,
        latents: mx.array,
        spec: ChunkSpec,
        prompt_embeds: mx.array,
        *,
        previous_latents: mx.array | None,
        chunk_latent_frames: int,
        block_health_context: WanBlockHealthContext | None = None,
    ) -> mx.array:
        """Restore a short LAST chunk, front-padded so the DiT sequence length stays fixed.

        A LAST chunk carries ``b + 1`` latents where a MIDDLE chunk carries ``n_lat``. The
        chunk is padded at the front up to ``n_lat + 1`` latents - with the previous
        chunk's low-quality tail when one exists, zeros otherwise - purely so the rotary
        offset and the sequence length stay correct, and only the trailing ``b + 1``
        restored latents are kept. Holding the sequence length constant is also what keeps
        the window index tables cacheable across chunks.

        Args:
            latents: ``[B, C, b + 1, H, W]`` low-quality latents for the LAST chunk.
            spec: The LAST chunk spec.
            prompt_embeds: ``[B, 512, text_dim]`` or ``[512, text_dim]`` frozen prompt
                embedding.
            previous_latents: ``[B, C, F_prev, H, W]`` *low-quality* latents of the
                preceding chunk, or ``None`` when this is the clip's only chunk. Upstream
                names this ``prev_dit_out_cpu`` but assigns it from the pre-restoration
                tensor (``runner.py``: ``prev_dit_out_cpu = z_bcfhw[:, :, -n_lat:]``), and
                the padded prefix is discarded, so low-quality latents are correct here.
            chunk_latent_frames: ``n_lat``, the MIDDLE chunk latent count for this run,
                i.e. ``clip_len // 4``.
            block_health_context: Forwarded to the Wan block-health probes.

        Returns:
            ``[B, C, b + 1, H, W]`` restored latents for the new frames only.

        Raises:
            ValueError: If ``latents`` is not rank 5, if ``spec`` is not a LAST chunk, if
                the latent count disagrees with ``spec``, if ``chunk_latent_frames`` is too
                small to pad up to, or if ``previous_latents`` cannot supply the prefix.
        """
        self._validate_latents(latents, name="latents")
        if spec.ctype is not ChunkType.LAST:
            raise ValueError(
                f"denoise_last_chunk is only valid for LAST chunks, got {spec.ctype.value} "
                f"at clip index {spec.clip_idx}. Use denoise() for FIRST and MIDDLE chunks."
            )
        latent_count = spec.b + 1
        if latents.shape[2] != latent_count:
            raise ValueError(
                f"LAST chunk {spec.clip_idx} describes {spec.frame_count} source frames "
                f"(b={spec.b}, so {latent_count} latents) but received {latents.shape[2]}."
            )
        pad_count = (chunk_latent_frames + 1) - latent_count
        if pad_count < 0:
            raise ValueError(
                f"LAST chunk {spec.clip_idx} holds {latent_count} latents, more than the "
                f"{chunk_latent_frames + 1} a padded chunk allows. chunk_latent_frames must be "
                "clip_len // 4 for the clip_len that produced these specs."
            )

        conditioning = self._conditioning(latents.shape[0], prompt_embeds, latents.dtype)
        padded = latents
        if pad_count:
            padded = mx.concatenate([self._front_pad(latents, previous_latents, pad_count), latents], axis=2)

        restored = self._restore(
            padded,
            conditioning=conditioning,
            t_offset=max(0, self.latent_offset - pad_count),
            block_health_context=block_health_context,
        )
        restored = restored[:, :, -latent_count:]

        # LAST ends the clip. Dropping the carry keeps a stray follow-up denoise() from
        # prepending a two-chunks-old prefix at a rotary offset that no longer matches.
        self._previous_lq = None
        self.latent_offset += latent_count
        self._settle(restored)
        return restored

    def _restore(
        self,
        latents: mx.array,
        *,
        conditioning: _Conditioning,
        t_offset: int,
        block_health_context: WanBlockHealthContext | None,
    ) -> mx.array:
        """One forward pass and the one-step subtraction ``z_hq = z_lq - velocity``."""
        velocity = self.transformer.predict_velocity(
            latents,
            timestep=conditioning.timestep,
            prompt_embeds=conditioning.prompt_embeds,
            t_offset=t_offset,
            clear_cache_each_block=self.clear_cache_each_block,
            block_health_context=block_health_context,
        )
        return latents - velocity

    def _front_pad(self, latents: mx.array, previous_latents: mx.array | None, pad_count: int) -> mx.array:
        """The ``pad_count`` filler latents prepended to a LAST chunk."""
        batch, channels, _, height, width = latents.shape
        if previous_latents is None:
            return mx.zeros((batch, channels, pad_count, height, width), dtype=latents.dtype)
        self._validate_latents(previous_latents, name="previous_latents")
        expected = (batch, channels, height, width)
        actual = (
            previous_latents.shape[0],
            previous_latents.shape[1],
            previous_latents.shape[3],
            previous_latents.shape[4],
        )
        if actual != expected:
            raise ValueError(
                f"previous_latents has (batch, channels, height, width) {actual}, but the LAST chunk needs {expected}."
            )
        if previous_latents.shape[2] < pad_count:
            raise ValueError(
                f"LAST chunk padding needs {pad_count} latent frames but previous_latents holds "
                f"only {previous_latents.shape[2]}. Carry the preceding chunk's trailing "
                "clip_len // 4 latents."
            )
        return previous_latents[:, :, -pad_count:].astype(latents.dtype)

    def _overlap_carry(self, latents: mx.array) -> mx.array | None:
        """The tail of this chunk's low-quality latents kept for the next chunk's prefix."""
        keep = min(self.dit_overlap, latents.shape[2])
        if keep <= 0:
            return None
        carry = latents[:, :, -keep:]
        mx.eval(carry)
        return carry

    def _conditioning(self, batch: int, prompt_embeds: mx.array, dtype: mx.Dtype) -> _Conditioning:
        """Broadcast the frozen prompt embedding to the batch, memoized across chunks.

        Only the chunk-invariant *inputs* are cached. Upstream additionally caches the
        condition embedder's ``temb`` / ``timestep_proj`` / ``encoder_hidden_states``
        (``dit.py``, ``_precompute_cond``), which is worth roughly 22.6 GFLOP per chunk;
        ``WanTransformer.__call__`` builds those internally and exposes no injection
        point, so recomputing them is the only option that does not fork the Wan forward.
        The recomputation is deterministic and numerically identical.

        Raises:
            ValueError: If the prompt embedding is not rank 2 or rank 3, or if a rank-3
                embedding carries a batch that is neither 1 nor ``batch``.
        """
        key = (batch, prompt_embeds.shape, prompt_embeds.dtype, dtype, self.inference_timestep)
        if self._conditioning_key == key and self._conditioning_cache is not None:
            return self._conditioning_cache

        if prompt_embeds.ndim == 2:
            broadcast = mx.broadcast_to(prompt_embeds[None], (batch, *prompt_embeds.shape))
        elif prompt_embeds.ndim == 3:
            if prompt_embeds.shape[0] not in (1, batch):
                raise ValueError(
                    f"Prompt embedding carries batch {prompt_embeds.shape[0]}, which is neither 1 "
                    f"nor the latent batch {batch}."
                )
            broadcast = mx.broadcast_to(prompt_embeds, (batch, *prompt_embeds.shape[1:]))
        else:
            raise ValueError(
                f"SwiftVR expects a [512, text_dim] or [B, 512, text_dim] prompt embedding, got "
                f"shape {prompt_embeds.shape}."
            )

        conditioning = _Conditioning(
            timestep=mx.full((batch,), self.inference_timestep, dtype=mx.float32),
            prompt_embeds=broadcast.astype(dtype),
        )
        mx.eval(conditioning.timestep, conditioning.prompt_embeds)
        self._conditioning_cache = conditioning
        self._conditioning_key = key
        return conditioning

    def _settle(self, restored: mx.array) -> None:
        """Materialize a chunk's result and release the transient graph.

        Without this the next chunk's work is queued on top of this one's, so peak memory
        grows with the number of chunks instead of staying flat. Same discipline as the
        Wan and SeedVR2 chunked routes.
        """
        carries = [] if self._previous_lq is None else [self._previous_lq]
        mx.eval(restored, *carries)
        mx.clear_cache()

    @staticmethod
    def _validate_latents(latents: mx.array, *, name: str) -> None:
        """Raise unless ``latents`` is a ``[B, C, F, H, W]`` array."""
        if latents.ndim != 5:
            raise ValueError(f"SwiftVR expects [B, C, F, H, W] {name}, got shape {latents.shape}.")
