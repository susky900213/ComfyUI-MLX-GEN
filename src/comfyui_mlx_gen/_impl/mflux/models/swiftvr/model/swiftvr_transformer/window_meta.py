"""Index bookkeeping for SwiftVR's mask-free shifted-window partitioning (MFSWA).

SwiftVR replaces Wan's global 3D self-attention with 2D spatial windows that keep a
full temporal view: every window spans all ``T`` post-patch frames and one
``wh x ww`` spatial tile. Windows are boundary-clamped rather than padded or
cyclically rolled, which is what makes the partition *mask-free* - each window is
exactly ``wh x ww`` wide and fully inside the grid, at the cost of overlap near the
edges.

Two index tables drive the attention:

``lin_flat``
    ``int32[Nw * Lw]``. Gathers tokens into dense windows, so attention is one plain
    SDPA call per window. Because clamping duplicates edge tokens, ``Nw * Lw`` is
    larger than the token count and ``lin_flat`` is NOT a permutation.

``owner_pos``
    ``int32[K]``. Selects, for each token, which of its (up to four) window outputs
    survives the scatter back. Unshifted layers award ownership to the lowest-index
    covering window, shifted layers to the highest-index one; that systematic
    disagreement is what lets consecutive layers exchange information across window
    seams. ``lin_flat[owner_pos] == arange(K)`` holds exactly.

Ported from ``_make_hw_starts`` / ``_build_hw_lin_indices`` /
``_WindowRuntimeMetaCache`` in ``swiftvr/models/transformer.py``. The upstream
``_infer_local_thw`` guess is deliberately not ported: the SwiftVR runtime always
knows the post-patch grid, and inferring it would be the kind of silent fallback
ADR 0002 forbids.
"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

# Window size is a SwiftVR code default (transformer.py:722), not checkpoint
# metadata: the published transformer/config.json carries no enable_swa or
# self_attn_window_hw key. Pin it here so the value is auditable and surfaced in
# run metadata rather than buried in a call site.
DEFAULT_WINDOW_HW: tuple[int, int] = (16, 16)


@dataclass(frozen=True)
class WindowGrid:
    """Precomputed gather/scatter tables for one (grid, window, parity) triple.

    Attributes:
        lin_flat: ``int32[num_windows * window_tokens]`` gather index into the token axis.
        owner_pos: ``int32[num_tokens]`` gather index into the flattened window-output axis.
        num_windows: ``Nw``, the number of windows.
        window_tokens: ``Lw = T * wh * ww``, tokens per window.
        num_tokens: ``K = T * H * W``, tokens in the post-patch grid.
        token_grid: The post-patch ``(T, H, W)`` grid this table was built for.
        window_hw: ``(wh, ww)`` after clamping to the grid.
        do_shift: Whether the half-window shift was applied.
    """

    lin_flat: mx.array
    owner_pos: mx.array
    num_windows: int
    window_tokens: int
    num_tokens: int
    token_grid: tuple[int, int, int]
    window_hw: tuple[int, int]
    do_shift: bool

    @property
    def prefer_front(self) -> bool:
        """Ownership priority. Derived from ``do_shift``; see :func:`build_owner_positions`."""
        return not self.do_shift


def window_axis_starts(size: int, window: int, *, do_shift: bool) -> list[int]:
    """Boundary-clamped window start offsets along one axis.

    Candidates are the regular stride-``window`` grid, offset by half a window when
    ``do_shift`` is set, clamped into ``[0, size - window]`` and deduplicated. Interior
    starts that add no coverage are then pruned: start ``i`` is dropped when
    ``starts[i + 1] <= starts[i - 1] + window``. The prune is evaluated simultaneously
    against the unpruned array, matching the upstream vectorized form exactly.

    Args:
        size: Axis length in post-patch tokens.
        window: Window extent along this axis.
        do_shift: Apply the half-window shift used by odd-indexed blocks.

    Returns:
        Sorted, deduplicated start offsets. A single ``[0]`` when the axis fits in
        one window.

    Raises:
        ValueError: If ``size`` or ``window`` is not positive.
    """
    if size <= 0 or window <= 0:
        raise ValueError(f"window_axis_starts requires positive size and window, got size={size}, window={window}.")
    if size <= window:
        return [0]

    shift = window // 2 if do_shift else 0
    max_start = size - window
    candidate_count = (size + window - 1) // window + 2
    starts = sorted({min(max(index * window - shift, 0), max_start) for index in range(candidate_count)})
    if len(starts) <= 2:
        return starts

    keep = [True] * len(starts)
    for index in range(1, len(starts) - 1):
        keep[index] = not (starts[index + 1] <= starts[index - 1] + window)
    return [start for start, keeps in zip(starts, keep) if keeps]


def build_window_token_indices(
    token_grid: tuple[int, int, int],
    height_starts: Sequence[int],
    width_starts: Sequence[int],
    window_hw: tuple[int, int],
) -> np.ndarray:
    """Dense per-window token index table.

    Window index is ``wi = i * len(width_starts) + j`` for height start ``i`` and width
    start ``j`` (row-major, height outer). The local slot within a window is
    ``l = t * wh * ww + a * ww + b`` for frame ``t`` and in-window position ``(a, b)``.

    Args:
        token_grid: Post-patch ``(T, H, W)``.
        height_starts: Window start rows from :func:`window_axis_starts`.
        width_starts: Window start columns from :func:`window_axis_starts`.
        window_hw: ``(wh, ww)``, already clamped to the grid.

    Returns:
        ``int64[Nw, Lw]`` flat token indices into a ``T * H * W`` token axis.
    """
    num_frames, height, width = token_grid
    window_height, window_width = window_hw
    rows = np.asarray(height_starts, dtype=np.int64)[:, None] + np.arange(window_height, dtype=np.int64)[None, :]
    columns = np.asarray(width_starts, dtype=np.int64)[:, None] + np.arange(window_width, dtype=np.int64)[None, :]
    spatial = (rows[:, None, :, None] * width + columns[None, :, None, :]).reshape(-1, window_height * window_width)
    frames = np.arange(num_frames, dtype=np.int64)[None, :, None] * (height * width)
    return (frames + spatial[:, None, :]).reshape(spatial.shape[0], num_frames * window_height * window_width)


def build_owner_positions(
    window_token_indices: np.ndarray,
    *,
    prefer_front: bool,
    num_tokens: int,
) -> np.ndarray:
    """Priority-coherent inverse of :func:`build_window_token_indices`.

    Windows are written in an order chosen so that the last write wins for the desired
    priority: ``prefer_front`` iterates windows in reverse so the lowest-index covering
    window ends up owning each token, otherwise the highest-index one does. Upstream
    derives ``prefer_front = not do_shift`` at the call site (transformer.py:474) while
    also keying its cache on both; the derivation lives in :func:`build_window_grid`
    here and this function stays explicit.

    Args:
        window_token_indices: ``int64[Nw, Lw]`` table from :func:`build_window_token_indices`.
        prefer_front: Award ownership to the lowest-index covering window.
        num_tokens: ``K``, the number of tokens in the grid.

    Returns:
        ``int64[K]`` indices into the flattened ``Nw * Lw`` window-output axis.

    Raises:
        ValueError: If the windows do not cover every token. Boundary clamping makes
            this unreachable for every grid swept during the port, but the simultaneous
            interior prune is not provably hole-free, so the check stays.
    """
    num_windows, window_tokens = window_token_indices.shape
    owner = np.full(num_tokens, -1, dtype=np.int64)
    local = np.arange(window_tokens, dtype=np.int64)
    order = range(num_windows - 1, -1, -1) if prefer_front else range(num_windows)
    for window_index in order:
        owner[window_token_indices[window_index]] = window_index * window_tokens + local
    uncovered = int(np.count_nonzero(owner < 0))
    if uncovered:
        raise ValueError(
            f"Shifted-window partition left {uncovered} of {num_tokens} tokens uncovered; "
            "the window start prune produced a coverage hole."
        )
    return owner


def build_window_grid(
    token_grid: tuple[int, int, int],
    window_hw: tuple[int, int] = DEFAULT_WINDOW_HW,
    *,
    do_shift: bool,
) -> WindowGrid:
    """Build the gather/scatter tables for one post-patch grid and shift parity.

    The configured window is clamped to the grid, so a portrait or small canvas with
    ``H < wh`` degenerates to a single full-axis window rather than failing.

    Args:
        token_grid: Post-patch ``(T, H, W)``.
        window_hw: Configured ``(wh, ww)`` before clamping.
        do_shift: Half-window shift, set on odd-indexed transformer blocks.

    Returns:
        The populated :class:`WindowGrid`.

    Raises:
        ValueError: If the grid or the window has a non-positive extent.
    """
    num_frames, height, width = token_grid
    if min(num_frames, height, width) <= 0:
        raise ValueError(f"Post-patch token grid must be positive in every axis, got {token_grid}.")
    if min(window_hw) <= 0:
        raise ValueError(f"Window size must be positive in both axes, got {window_hw}.")

    clamped_hw = (min(window_hw[0], height), min(window_hw[1], width))
    height_starts = window_axis_starts(height, clamped_hw[0], do_shift=do_shift)
    width_starts = window_axis_starts(width, clamped_hw[1], do_shift=do_shift)
    indices = build_window_token_indices(token_grid, height_starts, width_starts, clamped_hw)
    num_tokens = num_frames * height * width
    # prefer_front is a pure function of the parity: unshifted layers hand overlaps to
    # the grid-aligned window, shifted layers to the later one (transformer.py:474).
    owner = build_owner_positions(indices, prefer_front=not do_shift, num_tokens=num_tokens)

    return WindowGrid(
        lin_flat=mx.array(indices.reshape(-1), dtype=mx.int32),
        owner_pos=mx.array(owner, dtype=mx.int32),
        num_windows=int(indices.shape[0]),
        window_tokens=int(indices.shape[1]),
        num_tokens=num_tokens,
        token_grid=(num_frames, height, width),
        window_hw=clamped_hw,
        do_shift=do_shift,
    )


class WindowGridCache:
    """FIFO cache of :class:`WindowGrid` keyed by ``(T, H, W, wh, ww, do_shift)``.

    The device is not part of the key: MLX is unified memory. Four entries cover the
    realistic case of two shift parities across at most two distinct chunk lengths
    (FIRST/LAST produce ``n_lat + 1`` latents where MIDDLE produces ``n_lat``). Each
    1080p entry costs about 130 KiB, so the bound is about idle retention rather than
    memory pressure.
    """

    def __init__(self, max_entries: int = 4) -> None:
        if max_entries < 1:
            raise ValueError(f"WindowGridCache requires at least one entry, got {max_entries}.")
        self.max_entries = max_entries
        self._store: OrderedDict[tuple, WindowGrid] = OrderedDict()

    def get(
        self,
        token_grid: tuple[int, int, int],
        window_hw: tuple[int, int] = DEFAULT_WINDOW_HW,
        *,
        do_shift: bool,
    ) -> WindowGrid:
        """Return the cached grid, building and materializing it on first use."""
        key = (*token_grid, *window_hw, bool(do_shift))
        cached = self._store.get(key)
        if cached is not None:
            return cached
        grid = build_window_grid(token_grid, window_hw, do_shift=do_shift)
        mx.eval(grid.lin_flat, grid.owner_pos)
        self._store[key] = grid
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
        return grid

    def clear(self) -> None:
        """Drop every cached grid."""
        self._store.clear()
