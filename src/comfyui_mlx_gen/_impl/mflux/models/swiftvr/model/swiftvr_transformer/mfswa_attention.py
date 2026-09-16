"""MFSWA: mask-free shifted-window self-attention for the SwiftVR DiT.

SwiftVR reuses the stock Wan 2.2 TI2V-5B transformer weights unchanged and swaps only
the *token routing* of self-attention: instead of one global attention over all
``T * H * W`` post-patch tokens, tokens are gathered into dense ``wh x ww`` spatial
windows that keep a full temporal view, attended with one plain SDPA call per window,
and scattered back. Projections, QK norms, RoPE and the output projection are exactly
Wan's - this module reuses :class:`WanAttention`'s own helpers rather than
reimplementing them.

Even-indexed blocks use the grid-aligned partition, odd-indexed blocks a half-window
shift, so consecutive layers disagree about which window owns an overlap region and
information crosses window seams. Cross-attention is untouched: only ``attn1`` ever
receives a strategy.

Reference: ``WanShiftWindow2DInferProcessor`` and
``enable_shifted_window_self_attention`` in ``swiftvr/models/transformer.py``.

Two upstream behaviours are deliberately not ported. ``fuse_projections`` concatenates
``to_q``/``to_k``/``to_v`` into a single linear, which would break the 825-tensor
identity that lets us reuse ``WanWeightMapping`` verbatim, for no numerical gain.
``_release_input_storage`` is a CUDA storage hack with no MLX analogue.
"""

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from mflux.models.swiftvr.model.swiftvr_transformer.window_meta import (
    DEFAULT_WINDOW_HW,
    WindowGrid,
    WindowGridCache,
)
from mflux.models.wan.model.wan_transformer.wan_attention import WanAttention

# Name of the optional hook on WanAttention that this module drives. WanAttention must
# initialize the attribute and dispatch to it at the top of __call__; assigning it on a
# build that lacks the hook would silently run global attention instead of MFSWA, which
# is exactly the class of failure ADR 0002 forbids, so install() checks for it.
SELF_ATTENTION_STRATEGY_ATTR = "self_attention_strategy"


@dataclass
class ShiftedWindowRuntime:
    """Per-chunk state shared by every MFSWA strategy in one transformer.

    The SwiftVR runtime sets :attr:`token_grid` and :attr:`rotary_emb` once before each
    DiT forward; both must be set, because neither can be recovered safely from inside
    an attention module. Upstream guesses the 3D factorization from the token count
    (``_infer_local_thw``) for sequence-parallel shards; we always know the grid, and a
    guess would be a silent fallback.

    Attributes:
        token_grid: Post-patch ``(T, H, W)`` for the chunk currently being denoised.
        rotary_emb: The temporally-offset ``(cos, sin)`` tables for this chunk, each
            ``[1, T * H * W, 1, head_dim]``. Authoritative, and read here rather than
            from the block argument: only the runtime can guarantee the tables carry
            this chunk's offset, whatever a caller passed to ``WanTransformer``.
        window_hw: Configured window size before clamping to the grid.
        grid_cache: Cache of the gather/scatter tables, shared across blocks.
    """

    token_grid: tuple[int, int, int] | None = None
    rotary_emb: tuple[mx.array, mx.array] | None = None
    window_hw: tuple[int, int] = DEFAULT_WINDOW_HW
    grid_cache: WindowGridCache = field(default_factory=WindowGridCache)

    def resolve_grid(self, *, do_shift: bool, num_tokens: int) -> WindowGrid:
        """Return the window tables for this chunk, validating the token count.

        Raises:
            ValueError: If :attr:`token_grid` is unset or disagrees with ``num_tokens``.
        """
        if self.token_grid is None:
            raise ValueError(
                "ShiftedWindowRuntime.token_grid must be set to the post-patch (T, H, W) "
                "grid before the SwiftVR DiT forward."
            )
        expected = self.token_grid[0] * self.token_grid[1] * self.token_grid[2]
        if expected != num_tokens:
            raise ValueError(
                f"MFSWA token count mismatch: hidden states carry {num_tokens} tokens but "
                f"token_grid {self.token_grid} implies {expected}."
            )
        return self.grid_cache.get(self.token_grid, self.window_hw, do_shift=do_shift)

    def resolve_rotary_emb(self, num_tokens: int) -> tuple[mx.array, mx.array]:
        """Return the chunk's rotary tables, validating their token count.

        Raises:
            ValueError: If :attr:`rotary_emb` is unset or covers a different token count.
        """
        if self.rotary_emb is None:
            raise ValueError(
                "ShiftedWindowRuntime.rotary_emb must be set to the temporally-offset "
                "rotary tables (t_offset may be 0) before the SwiftVR DiT forward."
            )
        freqs_cos, freqs_sin = self.rotary_emb
        if freqs_cos.shape[1] != num_tokens or freqs_sin.shape[1] != num_tokens:
            raise ValueError(
                f"MFSWA rotary table mismatch: tables cover {freqs_cos.shape[1]} tokens but "
                f"the chunk carries {num_tokens}."
            )
        return freqs_cos, freqs_sin

    def reset(self) -> None:
        """Clear the per-chunk grid and rotary tables, keeping the caches warm."""
        self.token_grid = None
        self.rotary_emb = None


class ShiftedWindowSelfAttention:
    """Self-attention token router installed on one ``WanAttention`` (``attn1`` only).

    Holds no weights: every projection and norm is read off the :class:`WanAttention`
    instance handed to :meth:`__call__`. The shift parity is baked in at construction
    rather than derived from a layer index inside the call, matching upstream's reason
    for the same choice (per-layer integer attributes become static guards under graph
    capture).
    """

    def __init__(self, runtime: ShiftedWindowRuntime, *, do_shift: bool) -> None:
        self.runtime = runtime
        self.do_shift = bool(do_shift)

    def __call__(
        self,
        attention: WanAttention,
        hidden_states: mx.array,
        rotary_emb: tuple[mx.array, mx.array] | None,
        attention_name: str,
        block_health_context: Any | None,
    ) -> mx.array:
        """Run windowed self-attention over ``hidden_states``.

        Args:
            attention: The self-attention module supplying weights and hyperparameters.
            hidden_states: ``[B, K, inner_dim]`` post-patch tokens.
            rotary_emb: The block's own rotary tables. Not read - the strategy takes the
                temporally-offset tables from the runtime, which is the authoritative
                copy - but validated against the token count so an unexpected grid raises
                instead of drifting. The SwiftVR runtime hands ``WanTransformer`` the same
                tables, so in a real run these are that same pair.
            attention_name: Dotted name used by the block-health probes.
            block_health_context: Opaque context forwarded to the probes.

        Returns:
            ``[B, K, inner_dim]``, the same shape and contract as global self-attention.

        Raises:
            ValueError: If the module is a cross-attention module, if the runtime has not
                been primed for this chunk, or if a supplied table covers a different
                token count.
        """
        if attention.cross_attention_dim_head is not None or attention.added_kv_proj_dim is not None:
            raise ValueError(
                f"MFSWA is a self-attention router but {attention_name} is configured for "
                "cross-attention; install it on attn1 only."
            )

        batch, num_tokens, _ = hidden_states.shape
        if rotary_emb is not None and rotary_emb[0].shape[1] != num_tokens:
            raise ValueError(
                f"MFSWA received rotary tables covering {rotary_emb[0].shape[1]} tokens for a "
                f"chunk of {num_tokens} tokens."
            )
        grid = self.runtime.resolve_grid(do_shift=self.do_shift, num_tokens=num_tokens)
        freqs_cos, freqs_sin = self.runtime.resolve_rotary_emb(num_tokens)

        health_enabled = WanAttention._block_health_enabled()
        runtime_dtype = hidden_states.dtype
        heads = attention.heads
        head_dim = attention.dim_head
        num_windows = grid.num_windows
        window_tokens = grid.window_tokens

        query = attention.to_q(hidden_states)
        key = attention.to_k(hidden_states)
        value = attention.to_v(hidden_states)
        for name, tensor in ((".to_q", query), (".to_k", key), (".to_v", value)):
            WanAttention._check_tensor_health(
                enabled=health_enabled,
                name=f"{attention_name}{name}",
                tensor=tensor,
                context=block_health_context,
            )

        query = WanAttention._apply_diffusers_rms_norm(query, attention.norm_q)
        key = WanAttention._apply_diffusers_rms_norm(key, attention.norm_k)
        query = query.reshape(batch, num_tokens, heads, head_dim)
        key = key.reshape(batch, num_tokens, heads, head_dim)
        value = value.reshape(batch, num_tokens, heads, head_dim)
        for name, tensor in ((".norm_q", query), (".norm_k", key)):
            WanAttention._check_tensor_health(
                enabled=health_enabled,
                name=f"{attention_name}{name}",
                tensor=tensor,
                context=block_health_context,
            )

        # Value is gathered before RoPE only to shorten the peak-memory window; the
        # ordering is numerically irrelevant but matches the reference step for step so a
        # parity harness can compare intermediates.
        value = mx.take(value, grid.lin_flat, axis=1).reshape(batch * num_windows, window_tokens, heads, head_dim)

        # RoPE is applied globally, before partitioning, exactly as in plain Wan.
        query = WanAttention._apply_rotary_emb(query, freqs_cos, freqs_sin)
        key = WanAttention._apply_rotary_emb(key, freqs_cos, freqs_sin)
        for name, tensor in ((".rotary_q", query), (".rotary_k", key)):
            WanAttention._check_tensor_health(
                enabled=health_enabled,
                name=f"{attention_name}{name}",
                tensor=tensor,
                context=block_health_context,
            )

        query = mx.take(query, grid.lin_flat, axis=1).reshape(batch * num_windows, window_tokens, heads, head_dim)
        key = mx.take(key, grid.lin_flat, axis=1).reshape(batch * num_windows, window_tokens, heads, head_dim)

        # One dense, mask-free SDPA per window. Boundary clamping guarantees every window
        # is exactly wh x ww and fully inside the grid, so there is nothing to mask.
        windowed = mx.fast.scaled_dot_product_attention(
            mx.transpose(query, (0, 2, 1, 3)),
            mx.transpose(key, (0, 2, 1, 3)),
            mx.transpose(value, (0, 2, 1, 3)),
            scale=attention.scale,
        ).astype(runtime_dtype)
        WanAttention._check_tensor_health(
            enabled=health_enabled,
            name=f"{attention_name}.window_sdpa",
            tensor=windowed,
            context=block_health_context,
        )

        windowed = mx.transpose(windowed, (0, 2, 1, 3)).reshape(batch, num_windows * window_tokens, heads, head_dim)
        # Priority-coherent scatter: a token covered by several windows keeps exactly one
        # window's output, chosen by the parity of this layer.
        output = mx.take(windowed, grid.owner_pos, axis=1).reshape(batch, num_tokens, heads * head_dim)
        output = attention.to_out[0](output)
        WanAttention._check_tensor_health(
            enabled=health_enabled,
            name=f"{attention_name}.to_out",
            tensor=output,
            context=block_health_context,
        )
        return output


def install_shifted_window_self_attention(
    transformer,
    *,
    runtime: ShiftedWindowRuntime,
    shift_alternate_layers: bool = True,
) -> list[ShiftedWindowSelfAttention]:
    """Install MFSWA on every self-attention module of a :class:`WanTransformer`.

    Block ``i`` gets the shifted partition when ``shift_alternate_layers`` and ``i`` is
    odd, matching ``enable_shifted_window_self_attention`` (transformer.py:679). Call
    after construction and after weights have been applied; the strategies hold no
    weights, only index buffers on a plain Python attribute that MLX's parameter
    traversal ignores.

    Args:
        transformer: The :class:`WanTransformer` to modify in place.
        runtime: Shared per-chunk state handed to every strategy.
        shift_alternate_layers: Alternate the half-window shift across blocks.

    Returns:
        The installed strategies, indexed by block.

    Raises:
        ValueError: If the transformer exposes no blocks, or if ``WanAttention`` does not
            carry the self-attention strategy hook - assigning it on such a build would
            leave global attention silently in place.
    """
    blocks = getattr(transformer, "blocks", None)
    if not blocks:
        raise ValueError("Cannot install MFSWA: the transformer exposes no blocks.")

    first_attention = getattr(blocks[0], "attn1", None)
    if first_attention is None:
        raise ValueError("Cannot install MFSWA: transformer blocks expose no attn1 self-attention module.")
    if not hasattr(first_attention, SELF_ATTENTION_STRATEGY_ATTR):
        raise ValueError(
            "Cannot install MFSWA: WanAttention does not expose the "
            f"'{SELF_ATTENTION_STRATEGY_ATTR}' hook. WanAttention.__init__ must initialize "
            f"self.{SELF_ATTENTION_STRATEGY_ATTR} = None and WanAttention.__call__ must "
            "dispatch to it before projecting, otherwise SwiftVR would silently run Wan's "
            "global self-attention."
        )

    strategies: list[ShiftedWindowSelfAttention] = []
    for block_index, block in enumerate(blocks):
        attention = getattr(block, "attn1", None)
        if attention is None:
            raise ValueError(f"Cannot install MFSWA: block {block_index} exposes no attn1 self-attention module.")
        do_shift = bool(shift_alternate_layers and block_index % 2 == 1)
        strategy = ShiftedWindowSelfAttention(runtime, do_shift=do_shift)
        setattr(attention, SELF_ATTENTION_STRATEGY_ATTR, strategy)
        strategies.append(strategy)
    return strategies


def uninstall_shifted_window_self_attention(transformer) -> None:
    """Restore global self-attention on every block of ``transformer``."""
    for block in getattr(transformer, "blocks", []) or []:
        attention = getattr(block, "attn1", None)
        if attention is not None and hasattr(attention, SELF_ATTENTION_STRATEGY_ATTR):
            setattr(attention, SELF_ATTENTION_STRATEGY_ATTR, None)
