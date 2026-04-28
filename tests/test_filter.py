"""Test the bidirectional Butterworth filter."""
from __future__ import annotations

import numpy as np
import pytest

from theia_osim.kinematics_postprocess.filter import lowpass_filtfilt


def test_filter_dc_passes_through():
    """DC signal should be unchanged by lowpass."""
    signal = np.full(1000, 5.0)
    filtered = lowpass_filtfilt(signal, cutoff_hz=20.0, sample_rate_hz=300.0)
    np.testing.assert_allclose(filtered, signal, atol=1e-10)


def test_filter_attenuates_high_frequency():
    """Signal above cutoff should be heavily attenuated."""
    fs = 300.0
    n = 1000
    t = np.arange(n) / fs
    high_freq = np.sin(2 * np.pi * 100 * t)  # 100 Hz, well above 20 Hz cutoff
    filtered = lowpass_filtfilt(high_freq, cutoff_hz=20.0, sample_rate_hz=fs)
    assert np.abs(filtered).max() < 0.1  # at least 20 dB attenuation


def test_filter_passes_low_frequency():
    """Signal below cutoff should pass with minimal change."""
    fs = 300.0
    n = 1000
    t = np.arange(n) / fs
    low_freq = np.sin(2 * np.pi * 5 * t)  # 5 Hz, well below 20 Hz cutoff
    filtered = lowpass_filtfilt(low_freq, cutoff_hz=20.0, sample_rate_hz=fs)
    # Skip filter edges
    np.testing.assert_allclose(filtered[100:-100], low_freq[100:-100], atol=0.05)


def test_filter_along_axis_0_for_2d():
    fs = 300.0
    n = 500
    signal = np.random.RandomState(0).randn(n, 6)
    filtered = lowpass_filtfilt(signal, cutoff_hz=20.0, sample_rate_hz=fs, axis=0)
    assert filtered.shape == signal.shape


def test_filter_rejects_odd_order():
    with pytest.raises(ValueError, match="must be even"):
        lowpass_filtfilt(np.zeros(100), cutoff_hz=20.0, sample_rate_hz=300.0, order=3)


def test_filter_rejects_cutoff_above_nyquist():
    with pytest.raises(ValueError, match="< Nyquist"):
        lowpass_filtfilt(np.zeros(100), cutoff_hz=200.0, sample_rate_hz=300.0)
