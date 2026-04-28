"""Bidirectional Butterworth low-pass filter — matches V3D's filtfilt spec."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def lowpass_filtfilt(
    signal: np.ndarray,
    *,
    cutoff_hz: float,
    sample_rate_hz: float,
    order: int = 4,
    axis: int = 0,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass via forward-backward filtering.

    `order` here is the *combined* order after filtfilt — internally we use
    `order // 2` for the underlying Butterworth so the round-trip is `order`
    (matches V3D's "1-pass bidirectional 4th-order" terminology).

    Args:
        signal: array of shape (N,) or (N, ...). Filter applied along `axis`.
        cutoff_hz: -3 dB cutoff.
        sample_rate_hz: sample rate.
        order: combined zero-phase order (default 4).
        axis: axis along which to filter (default 0 = time).

    Returns:
        Same shape as input.
    """
    if order % 2 != 0:
        raise ValueError(f"combined order {order} must be even (filtfilt doubles)")
    half_order = order // 2
    nyq = sample_rate_hz / 2.0
    if cutoff_hz >= nyq:
        raise ValueError(f"cutoff {cutoff_hz}Hz must be < Nyquist {nyq}Hz")
    b, a = butter(half_order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, signal, axis=axis)
