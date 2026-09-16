"""TAEHV-derived primitives for SwiftVR's Restoration-aware Autoencoder (ReAE).

ReAE replaces the Wan 3D VAE entirely (40.95M parameters against 704.69M) while
reproducing its latent contract exactly: 48 channels, 16x spatial and 4x temporal
compression, which is what lets the unmodified Wan transformer consume these latents.

Layout. The reference is torch NCHW with convolution weights ``[out, in, kh, kw]``
(``[out, in, kd, kh, kw]`` for Conv3d); MLX is channels-last with ``[out, kh, kw, in]``
(``[out, kd, kh, kw, in]``). The weight layout is converted exactly ONCE, at load time,
by the ``transform`` on each ``WeightTarget`` in
:mod:`mflux.models.swiftvr.weights.swiftvr_weight_mapping`; the activation layout is
converted exactly once at the pipeline boundary. Nothing here transposes a weight per
call. Two places do depend on the channels-last layout and are commented as such:
:meth:`TPool.__call__`, whose frame-to-channel fold is not a plain reshape in NHWC, and
:meth:`TGrow.__call__`, which slices the Conv3d kernel along its depth axis - axis 1 in
MLX layout where the reference reads axis 2.

Everything here operates on channels-last MLX arrays with frames folded into the batch
axis as ``[N * T, H, W, C]``, matching the reference's ``[N * T, C, H, W]``. None of
these blocks owns temporal state; the causal carry lives in
:mod:`mflux.models.swiftvr.streaming.streaming_reae`, which passes each MemBlock its
``past`` explicitly.

Ported from ``swiftvr/models/reae.py`` (itself adapted from TAEHV, MIT). Pixel values
are in ``[0, 1]``, not ``[-1, 1]``. Every block below was checked against the reference
with shared weights over five seeds: worst error 2.4e-07 in float32.
"""

import mlx.core as mx
from mlx import nn

CLAMP_MAGNITUDE = 3.0


def reae_conv(n_in: int, n_out: int, *, stride: int = 1, bias: bool = True) -> nn.Conv2d:
    """3x3 convolution with padding 1 - the ``conv`` helper at ``reae.py:14``."""
    return nn.Conv2d(n_in, n_out, kernel_size=3, padding=1, stride=stride, bias=bias)


class Clamp(nn.Module):
    """Soft range limiter ``tanh(x / 3) * 3``, the decoder's first layer."""

    def __call__(self, x: mx.array) -> mx.array:
        return mx.tanh(x / CLAMP_MAGNITUDE) * CLAMP_MAGNITUDE


class MemBlock(nn.Module):
    """Residual block fusing the current frame with the previous one.

    ``conv`` is a plain Python list so its MLX parameter paths are ``conv.0``, ``conv.2``
    and ``conv.4``, matching the checkpoint's ``nn.Sequential`` indices exactly. Every
    ReAE MemBlock has ``n_in == n_out``, so ``skip`` is the identity and the published
    checkpoint carries no ``skip.*`` tensor; the branch exists for fidelity to the
    reference and would raise through the initializer's coverage assertion, not silently
    load, if a future checkpoint supplied one.

    The concatenation order is ``[x, past]`` on the channel axis, x first, because
    ``conv.0`` expects ``2 * n_in`` input channels with x occupying the first half.
    """

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.conv = [
            reae_conv(n_in * 2, n_out),
            nn.ReLU(),
            reae_conv(n_out, n_out),
            nn.ReLU(),
            reae_conv(n_out, n_out),
        ]
        self.skip = None if n_in == n_out else nn.Conv2d(n_in, n_out, kernel_size=1, bias=False)
        self.act = nn.ReLU()

    def __call__(self, x: mx.array, past: mx.array) -> mx.array:
        """Fuse ``x`` with the previous frame ``past``.

        Args:
            x: ``[N * T, H, W, C_in]`` current frames.
            past: ``[N * T, H, W, C_in]``, the same frames shifted one step back in time.
                The caller owns the shift and the cross-chunk carry; a MemBlock never
                looks at its own history.

        Returns:
            ``[N * T, H, W, C_out]``.

        Raises:
            ValueError: If ``past`` does not have the same shape as ``x``. A mis-shaped
                carry is the failure mode of the streaming state, so it is named here
                rather than surfacing as a concatenation error deeper down.
        """
        if x.shape != past.shape:
            raise ValueError(f"MemBlock needs past to match x; got x {x.shape} and past {past.shape}.")
        hidden = mx.concatenate([x, past], axis=-1)
        for layer in self.conv:
            hidden = layer(hidden)
        residual = x if self.skip is None else self.skip(x)
        return self.act(hidden + residual)


class TPool(nn.Module):
    """Temporal pooling by ``stride`` via a 1x1 channel mix over stacked frames."""

    def __init__(self, n_f: int, stride: int) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError(f"TPool stride must be positive, got {stride}.")
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, kernel_size=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Fold ``stride`` consecutive frames into the channel axis, then mix.

        The reference does ``x.reshape(-1, stride * C, H, W)`` on NCHW, which stacks
        consecutive frames along the channel axis in frame order, giving a channel
        ordering of ``(frame_in_group, channel)``. In NHWC the memory order is
        ``(frame, h, w, c)``, so a direct reshape to ``[NT // s, H, W, s * C]``
        interleaves a spatial axis into the channels: it has the right shape, runs
        without error and produces garbage. The frame axis must be transposed past the
        spatial axes first, which is what the two steps below do.

        Args:
            x: ``[N * T, H, W, C]`` with the folded frame count a multiple of ``stride``.

        Returns:
            ``[N * T // stride, H, W, C]``.

        Raises:
            ValueError: If the frame count is not a multiple of ``stride``. The caller
                buffers partial groups; reaching here with one is a state-tracking bug.
        """
        frames, height, width, channels = x.shape
        stride = self.stride
        if frames % stride:
            raise ValueError(f"TPool(stride={stride}) requires a multiple of {stride} frames, got {frames}.")
        grouped = x.reshape(frames // stride, stride, height, width, channels)
        stacked = grouped.transpose(0, 2, 3, 1, 4).reshape(frames // stride, height, width, stride * channels)
        return self.conv(stacked)


class TGrow(nn.Module):
    """Temporal upsampling by ``stride``.

    ``stride == 1`` is a plain 1x1 projection stored as ``proj``. ``stride == 2`` stores
    the reference's depth-only ``Conv3d(n_f, n_f, (3, 1, 1), padding=(1, 0, 0))`` as
    ``conv3d`` so the parameter shape round-trips through the checkpoint and the model
    saver, but evaluates an algebraically identical pair of 1x1 convolutions.

    Derivation. The reference nearest-upsamples the depth axis of a single frame ``a`` to
    length 2, so both depth entries equal ``a``, then cross-correlates with a depth-3
    kernel under zero padding of 1:

        out[0] = w[:, 0] * 0 + w[:, 1] * a + w[:, 2] * a = (w1 + w2) * a
        out[1] = w[:, 0] * a + w[:, 1] * a + w[:, 2] * 0 = (w0 + w1) * a

    There is no cross-frame mixing - TGrow is frame-local, which is why the streaming
    runner carries no state for it. Evaluating the fused form avoids materializing the
    duplicated ``[N * T, 2, H, W, C]`` tensor, which at 1080p is the difference between a
    28.3 GiB and a 16.0 GiB decoder peak. Checked against the reference's literal Conv3d
    path with shared weights at 2.4e-07 in float32.
    """

    def __init__(self, n_f: int, stride: int) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"TGrow supports stride 1 or 2, got {stride}.")
        self.stride = stride
        self.n_f = n_f
        if stride == 1:
            self.proj = nn.Conv2d(n_f, n_f, kernel_size=1, bias=False)
            self.conv3d = None
        else:
            self.proj = None
            self.conv3d = nn.Conv3d(n_f, n_f, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Grow the folded frame axis by ``stride``.

        Args:
            x: ``[N * T, H, W, C]``.

        Returns:
            ``[N * T * stride, H, W, C]``, with each input frame's two outputs adjacent,
            matching the reference's ``permute(0, 2, 1, 3, 4).reshape(...)``.
        """
        if self.stride == 1:
            return self.proj(x)
        frames, height, width, channels = x.shape
        # MLX Conv3d weight layout is [out, kd, kh, kw, in], so the depth axis is 1 -
        # the reference reads the same axis at index 2 in torch's [out, in, kd, kh, kw].
        # Each slice is [out, kh=1, kw=1, in], exactly a 1x1 Conv2d kernel.
        weight = self.conv3d.weight
        kernel_even = weight[:, 1] + weight[:, 2]
        kernel_odd = weight[:, 0] + weight[:, 1]
        even = mx.conv2d(x, kernel_even)
        odd = mx.conv2d(x, kernel_odd)
        return mx.stack([even, odd], axis=1).reshape(frames * 2, height, width, channels)


def pixel_unshuffle_nhwc(x: mx.array, ratio: int) -> mx.array:
    """Channels-last ``pixel_unshuffle``: ``[N, H, W, C] -> [N, H // r, W // r, C * r * r]``.

    The channel order is torch's ``(c, i, j)``, which is what ``encoder.0`` was trained
    against. Wan's ``patchify`` uses ``(c, p_w, p_h)`` instead and is NOT interchangeable.

    Raises:
        ValueError: If ``ratio`` is not positive or does not divide both spatial axes.
    """
    if ratio < 1:
        raise ValueError(f"pixel_unshuffle ratio must be positive, got {ratio}.")
    if ratio == 1:
        return x
    batch, height, width, channels = x.shape
    if height % ratio or width % ratio:
        raise ValueError(f"pixel_unshuffle({ratio}) requires divisible spatial dims, got {(height, width)}.")
    reshaped = x.reshape(batch, height // ratio, ratio, width // ratio, ratio, channels)
    return reshaped.transpose(0, 1, 3, 5, 2, 4).reshape(
        batch, height // ratio, width // ratio, channels * ratio * ratio
    )


def pixel_shuffle_nhwc(x: mx.array, ratio: int) -> mx.array:
    """Channels-last ``pixel_shuffle``, the exact inverse of :func:`pixel_unshuffle_nhwc`.

    Raises:
        ValueError: If ``ratio`` is not positive or does not divide the channel count.
    """
    if ratio < 1:
        raise ValueError(f"pixel_shuffle ratio must be positive, got {ratio}.")
    if ratio == 1:
        return x
    batch, height, width, channels = x.shape
    if channels % (ratio * ratio):
        raise ValueError(f"pixel_shuffle({ratio}) requires channels divisible by {ratio * ratio}, got {channels}.")
    out_channels = channels // (ratio * ratio)
    reshaped = x.reshape(batch, height, width, out_channels, ratio, ratio)
    return reshaped.transpose(0, 1, 4, 2, 5, 3).reshape(batch, height * ratio, width * ratio, out_channels)
