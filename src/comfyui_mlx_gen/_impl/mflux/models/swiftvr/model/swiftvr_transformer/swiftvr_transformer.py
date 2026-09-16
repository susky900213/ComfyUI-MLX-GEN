"""SwiftVR's DiT: the stock Wan 2.2 TI2V-5B transformer with MFSWA token routing.

SwiftVR's ``transformer/config.json`` is byte-identical to
``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` and all 825 checkpoint tensors match the stock
model in name, shape and dtype - it is a fine-tune, not a new architecture. There is
therefore no SwiftVR DiT: this module owns an unmodified :class:`WanTransformer` and
changes only two things around it.

1. Self-attention token routing. Every block's ``attn1`` is given an MFSWA strategy
   (see :mod:`mfswa_attention`), which reuses the same weights and the same SDPA kernel
   but attends inside shifted 2D spatial windows with a full temporal view.
2. Rotary embeddings. Restoration runs chunk by chunk over one continuous clip, so the
   rotary tables are built at the chunk's global latent-frame offset (see
   :mod:`rope_offset`) rather than always from position zero. The same tables go to the
   shared MFSWA runtime, which is what the strategies read, and to
   ``WanTransformer.__call__``, whose ``rotary_emb`` argument exists so it does not build
   an un-offset set of its own that every block would then discard.

Cross-attention is untouched and still runs globally against SwiftVR's 512 frozen
prompt-embedding tokens.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.swiftvr.model.swiftvr_transformer.mfswa_attention import (
    ShiftedWindowRuntime,
    install_shifted_window_self_attention,
    uninstall_shifted_window_self_attention,
)
from mflux.models.swiftvr.model.swiftvr_transformer.rope_offset import RopeTemporalOffset
from mflux.models.swiftvr.model.swiftvr_transformer.window_meta import DEFAULT_WINDOW_HW
from mflux.models.wan.model.wan_transformer.wan_transformer import (
    WanBlockHealthContext,
    WanTransformer,
)


class SwiftVRTransformer(nn.Module):
    """Wan transformer plus SwiftVR's windowed attention and offset rotary tables.

    The wrapped :class:`WanTransformer` is exposed as :attr:`transformer` so weight
    loading, coverage assertions and quantization can address it with the stock Wan
    parameter paths.
    """

    def __init__(
        self,
        *,
        window_hw: tuple[int, int] = DEFAULT_WINDOW_HW,
        shift_alternate_layers: bool = True,
        **transformer_kwargs,
    ) -> None:
        """Build the transformer. MFSWA is installed separately, after weight loading.

        Args:
            window_hw: Spatial window size in post-patch tokens.
            shift_alternate_layers: Apply the half-window shift on odd-indexed blocks.
            **transformer_kwargs: Forwarded verbatim to :class:`WanTransformer`.
        """
        super().__init__()
        self.transformer = WanTransformer(**transformer_kwargs)
        self.shift_alternate_layers = bool(shift_alternate_layers)
        # Underscore-prefixed: runtime state and caches, never parameters.
        self._window_runtime = ShiftedWindowRuntime(window_hw=(int(window_hw[0]), int(window_hw[1])))
        self._rope_offset = RopeTemporalOffset(self.transformer.rope)
        self._mfswa_installed = False

    @property
    def window_hw(self) -> tuple[int, int]:
        """Configured MFSWA window size. Held once, on the runtime the strategies read."""
        return self._window_runtime.window_hw

    @property
    def window_runtime(self) -> ShiftedWindowRuntime:
        """Per-chunk MFSWA state shared by every block."""
        return self._window_runtime

    @property
    def rope_offset(self) -> RopeTemporalOffset:
        """Builder for the chunk's temporally-offset rotary tables."""
        return self._rope_offset

    @property
    def mfswa_installed(self) -> bool:
        """Whether the windowed self-attention strategies are in place."""
        return self._mfswa_installed

    def install_mfswa(self) -> None:
        """Install windowed self-attention on every block.

        Call once, after weights have been applied - the strategies carry no weights,
        only index buffers.

        Raises:
            ValueError: If ``WanAttention`` does not expose the strategy hook, since
                assigning it on such a build would leave global attention in place.
        """
        install_shifted_window_self_attention(
            self.transformer,
            runtime=self._window_runtime,
            shift_alternate_layers=self.shift_alternate_layers,
        )
        self._mfswa_installed = True

    def uninstall_mfswa(self) -> None:
        """Restore Wan's global self-attention on every block."""
        uninstall_shifted_window_self_attention(self.transformer)
        self._mfswa_installed = False

    def token_grid(self, latents: mx.array) -> tuple[int, int, int]:
        """Post-patch ``(T, H, W)`` grid for a ``[B, C, F, H, W]`` latent chunk.

        Raises:
            ValueError: If any latent axis is not divisible by the patch size.
        """
        _, _, num_frames, height, width = latents.shape
        patch_t, patch_h, patch_w = self.transformer.patch_size
        if num_frames % patch_t or height % patch_h or width % patch_w:
            raise ValueError(
                f"Latent grid {(num_frames, height, width)} is not divisible by the transformer "
                f"patch size {(patch_t, patch_h, patch_w)}."
            )
        return num_frames // patch_t, height // patch_h, width // patch_w

    def predict_velocity(
        self,
        latents: mx.array,
        *,
        timestep: mx.array,
        prompt_embeds: mx.array,
        t_offset: int = 0,
        clear_cache_each_block: bool = False,
        block_health_context: WanBlockHealthContext | None = None,
    ) -> mx.array:
        """Run one DiT forward and return the predicted degradation velocity.

        SwiftVR's restoration step is ``z_hq = z_lq - velocity``; this method returns the
        velocity only, leaving the subtraction to the streaming runtime.

        Args:
            latents: ``[B, C, F, H, W]`` low-quality latents for one chunk.
            timestep: ``[B]`` conditioning timestep. SwiftVR always uses the constant
                endpoint value, see ``streaming.streaming_dit.INFERENCE_TIMESTEP``.
            prompt_embeds: ``[B, 512, text_dim]`` frozen prompt embedding.
            t_offset: Global latent-frame offset of this chunk's first frame.
            clear_cache_each_block: Evaluate and release the MLX cache after each block.
                Effectively mandatory at 1080p: a shifted layer materializes hundreds of
                megabytes of gathered query/key/value on top of the ungathered tensors.
            block_health_context: Forwarded to the Wan block-health probes.

        Returns:
            ``[B, C_out, F, H, W]`` velocity, matching the latent layout.

        Raises:
            RuntimeError: If :meth:`install_mfswa` has not been called - running the
                stock global attention here would silently produce a different model.
        """
        if not self._mfswa_installed:
            raise RuntimeError(
                "SwiftVRTransformer.predict_velocity requires MFSWA to be installed; "
                "call install_mfswa() after applying weights."
            )
        grid = self.token_grid(latents)
        rotary_emb = self._rope_offset(token_grid=grid, t_offset=t_offset)
        self._window_runtime.token_grid = grid
        self._window_runtime.rotary_emb = rotary_emb
        # Handed to WanTransformer as well, not only to the window runtime. Left to build
        # its own it would produce a second, un-offset table per chunk - about 29 MB at
        # 1080p, retained by its cache - that every MFSWA block validates and discards.
        return self.transformer(
            latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            clear_cache_each_block=clear_cache_each_block,
            block_health_context=block_health_context,
            rotary_emb=rotary_emb,
        )

    def reset_chunk_state(self) -> None:
        """Clear per-chunk routing state. Caches of window and rotary tables are kept."""
        self._window_runtime.reset()
