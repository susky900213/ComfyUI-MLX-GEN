"""Temporally-offset rotary embeddings for SwiftVR's chunked DiT.

SwiftVR restores a clip chunk by chunk while keeping one continuous temporal
coordinate system, so chunk ``n`` must see RoPE positions starting at the global
latent-frame offset reached so far rather than at zero. The only difference from
Wan's :class:`WanRotaryPosEmbed` is the temporal slice start: ``cos_t[t_offset :
t_offset + ppf]`` instead of ``cos_t[:ppf]``. Height and width are always sliced from
zero (upstream ``_rope_with_offset`` accepts ``h_off``/``w_off`` but every caller
passes 0).

Two deliberate divergences from ``swiftvr/streaming/dit.py``:

1. The registered ``freqs_cos`` / ``freqs_sin`` parameters are never resized. They are
   real entries in ``rope.parameters()`` and ``ModelSaver`` persists
   ``dict(tree_flatten(model.parameters()))``, so growing them would silently change
   the shape of two keys in every saved checkpoint. Extended tables live in this
   object instead.
2. Tables past ``rope_max_seq_len`` are rebuilt analytically with Wan's own
   ``_get_1d_rotary_pos_embed`` rather than extrapolated from the first two rows with
   ``atan2``. The upstream trick derives each column's angular frequency from rows that
   are already rounded, and recovers accuracy by doing that arithmetic in float64, which
   MLX does not have. Measured against a float64 build at length 4096, upstream's
   extension errs by 1.66e-04 and a float32 rebuild by 2.83e-04, so the rebuild is the
   less accurate of the two by about 1.7x - it is chosen because reproducing upstream's
   accuracy needs the float64 step, and the residual is 13x smaller than the bfloat16
   rounding applied to the same tensor immediately afterwards (1.2e-03 against 1.5e-02
   on a rotated query at offset 4000).

Precision note: the float32 table already differs from a float64 build by 7.3e-05 at
length 1024, growing roughly linearly to 5.7e-04 at 8192. All of that is far below
bfloat16 unit roundoff, and because the drift is a smooth function of position shared
by query and key it largely cancels in the relative phase attention depends on. A
torch-parity test must use tolerances, never exact equality.
"""

import mlx.core as mx

from mflux.models.wan.model.wan_transformer.wan_embedding import WanRotaryPosEmbed

# Extra positions appended when the table has to grow, so consecutive chunks do not
# re-extend on every forward. Matches ROPE_EXTEND_MARGIN in swiftvr/streaming/dit.py.
ROPE_EXTEND_MARGIN = 256


class RopeTemporalOffset:
    """Builds temporally-offset rotary tables from an existing :class:`WanRotaryPosEmbed`.

    The wrapped module is read, never mutated.

    There is deliberately no per-result cache, unlike the ``(ppf, pph, ppw)`` cache Wan
    keeps for the unoffset case. Wan's key is constant within a run; this one would
    include ``t_offset``, which advances every chunk by construction, so a cache could
    never hit - it would only retain tables nothing will ask for again (about 29 MB per
    1080p entry). The extended axis tables below ARE cached, because those are keyed by
    length and are reused across chunks.
    """

    def __init__(self, rope: WanRotaryPosEmbed) -> None:
        self.rope = rope
        # Underscore-prefixed so MLX parameter traversal never sees these; they are a
        # derived cache, not weights.
        self._extended_tables: tuple[mx.array, mx.array] | None = None

    def __call__(self, *, token_grid: tuple[int, int, int], t_offset: int = 0) -> tuple[mx.array, mx.array]:
        """Return ``(freqs_cos, freqs_sin)`` of shape ``[1, T * H * W, 1, head_dim]``.

        Args:
            token_grid: Post-patch ``(T, H, W)`` grid for this chunk.
            t_offset: Global latent-frame offset of the chunk's first frame.

        Raises:
            ValueError: If the grid is not positive in every axis or ``t_offset`` is negative.
        """
        post_patch_frames, post_patch_height, post_patch_width = token_grid
        if min(post_patch_frames, post_patch_height, post_patch_width) <= 0:
            raise ValueError(f"Post-patch token grid must be positive in every axis, got {token_grid}.")
        if t_offset < 0:
            raise ValueError(f"RoPE temporal offset must be non-negative, got {t_offset}.")

        rope = self.rope
        required_length = max(t_offset + post_patch_frames, post_patch_height, post_patch_width)
        table_cos, table_sin = self._axis_tables(required_length)
        split_points = [rope.t_dim, rope.t_dim + rope.h_dim]
        cos_t, cos_h, cos_w = mx.split(table_cos, split_points, axis=1)
        sin_t, sin_h, sin_w = mx.split(table_sin, split_points, axis=1)

        grid_shape = (post_patch_frames, post_patch_height, post_patch_width)
        temporal_slice = slice(t_offset, t_offset + post_patch_frames)
        parts_cos = [
            mx.broadcast_to(cos_t[temporal_slice].reshape(post_patch_frames, 1, 1, -1), (*grid_shape, rope.t_dim)),
            mx.broadcast_to(cos_h[:post_patch_height].reshape(1, post_patch_height, 1, -1), (*grid_shape, rope.h_dim)),
            mx.broadcast_to(cos_w[:post_patch_width].reshape(1, 1, post_patch_width, -1), (*grid_shape, rope.w_dim)),
        ]
        parts_sin = [
            mx.broadcast_to(sin_t[temporal_slice].reshape(post_patch_frames, 1, 1, -1), (*grid_shape, rope.t_dim)),
            mx.broadcast_to(sin_h[:post_patch_height].reshape(1, post_patch_height, 1, -1), (*grid_shape, rope.h_dim)),
            mx.broadcast_to(sin_w[:post_patch_width].reshape(1, 1, post_patch_width, -1), (*grid_shape, rope.w_dim)),
        ]
        num_tokens = post_patch_frames * post_patch_height * post_patch_width
        freqs_cos = mx.concatenate(parts_cos, axis=-1).reshape(1, num_tokens, 1, -1)
        freqs_sin = mx.concatenate(parts_sin, axis=-1).reshape(1, num_tokens, 1, -1)
        mx.eval(freqs_cos, freqs_sin)
        return freqs_cos, freqs_sin

    def clear_cache(self) -> None:
        """Drop any extended tables, returning to the module's registered ones."""
        self._extended_tables = None

    def _axis_tables(self, required_length: int) -> tuple[mx.array, mx.array]:
        """Concatenated ``(cos, sin)`` tables covering at least ``required_length`` positions.

        Returns the registered Wan parameters untouched whenever the request fits inside
        ``rope_max_seq_len``, so the plain Wan route never reaches the extension path. At
        1080p only the temporal axis can ever exceed 1024 positions, which happens after
        1024 latent frames - 4096 source frames, about 2 min 51 s at 24 fps.
        """
        rope = self.rope
        if required_length <= rope.max_seq_len:
            return rope.freqs_cos, rope.freqs_sin

        extended = self._extended_tables
        if extended is not None and extended[0].shape[0] >= required_length:
            return extended

        target_length = required_length + ROPE_EXTEND_MARGIN
        tables = self._build_tables(target_length)
        mx.eval(*tables)
        self._extended_tables = tables
        return tables

    def _build_tables(self, length: int) -> tuple[mx.array, mx.array]:
        """Rebuild the concatenated per-axis tables at ``length`` positions.

        Reuses Wan's own generator so the extension is bit-comparable with the
        registered table over their shared prefix; there is no second implementation of
        the frequency schedule to drift.
        """
        rope = self.rope
        cosines = []
        sines = []
        for dim in (rope.t_dim, rope.h_dim, rope.w_dim):
            cos, sin = WanRotaryPosEmbed._get_1d_rotary_pos_embed(dim=dim, max_seq_len=length, theta=rope.theta)
            cosines.append(cos)
            sines.append(sin)
        return mx.concatenate(cosines, axis=1), mx.concatenate(sines, axis=1)
