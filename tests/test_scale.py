"""Test the scaling module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from theia_osim.analysis.scale import (
    LAI_DEFAULT_LENGTHS_M,
    THEIA_TO_OSIM_FOR_SCALING,
    compute_scale_factors,
)
from theia_osim.c3d_io.reader import read_theia_c3d
from theia_osim.constants import DEFAULT_VLB_4X4

C3D = Path(__file__).resolve().parents[1] / "pose_filt_0.c3d"


@pytest.mark.skipif(not C3D.exists(), reason="sample c3d not present")
def test_compute_scale_factors_uniform():
    t = read_theia_c3d(C3D, apply_vlb=DEFAULT_VLB_4X4)
    factors, lengths = compute_scale_factors(t, uniform_per_body=True)
    # All 6 limb bodies should have factors
    expected = {"femur_r", "femur_l", "tibia_r", "tibia_l", "humerus_r", "humerus_l"}
    assert set(factors.keys()) == expected
    # Uniform mode: sx == sy == sz per body
    for body, (sx, sy, sz) in factors.items():
        assert sx == sy == sz
        # Sanity: ratio in (0.5, 2.0) for any plausible adult athlete
        assert 0.5 < sx < 2.0


@pytest.mark.skipif(not C3D.exists(), reason="sample c3d not present")
def test_compute_scale_factors_long_axis_only():
    t = read_theia_c3d(C3D, apply_vlb=DEFAULT_VLB_4X4)
    factors, _ = compute_scale_factors(t, uniform_per_body=False)
    for _, (sx, sy, sz) in factors.items():
        # Long axis is OpenSim Y; X and Z stay at 1.0
        assert sx == 1.0
        assert sz == 1.0
        assert sy != 1.0


@pytest.mark.skipif(not C3D.exists(), reason="sample c3d not present")
def test_scale_factor_matches_length_ratio():
    t = read_theia_c3d(C3D, apply_vlb=DEFAULT_VLB_4X4)
    factors, lengths = compute_scale_factors(t)
    for theia_seg, osim_body in THEIA_TO_OSIM_FOR_SCALING.items():
        if osim_body not in factors:
            continue
        anthro = t.meta.segments_anthro[theia_seg]
        expected = anthro.length_m / LAI_DEFAULT_LENGTHS_M[osim_body]
        assert abs(factors[osim_body][0] - expected) < 1e-6
        assert abs(lengths[osim_body] - anthro.length_m) < 1e-6


def test_lai_defaults_are_anatomically_reasonable():
    """Stock LaiUhlrich segment lengths should fall in typical adult ranges."""
    assert 0.35 < LAI_DEFAULT_LENGTHS_M["femur_r"] < 0.50
    assert 0.35 < LAI_DEFAULT_LENGTHS_M["tibia_r"] < 0.50
    assert 0.20 < LAI_DEFAULT_LENGTHS_M["humerus_r"] < 0.40
    assert LAI_DEFAULT_LENGTHS_M["femur_r"] == LAI_DEFAULT_LENGTHS_M["femur_l"]
