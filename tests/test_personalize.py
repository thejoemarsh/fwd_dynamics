"""Test the personalize module + de Leva anthropometric tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from theia_osim.c3d_io.mdh_parser import parse_mdh
from theia_osim.c3d_io.reader import read_theia_c3d
from theia_osim.constants import DEFAULT_VLB_4X4
from theia_osim.model_build.anthropometrics import (
    DE_LEVA_MALE,
    V3D_TO_OSIM_BODY,
    V3D_TO_THEIA_SEG,
)
from theia_osim.model_build.personalize import (
    LAI_DEFAULT_LENGTHS_M,
    compute_scale_factors_full,
    compute_segment_lengths,
)

C3D = Path(__file__).resolve().parents[1] / "pose_filt_0.c3d"
MDH = Path(__file__).resolve().parents[1] / "theia_model.mdh"


def test_de_leva_pelvis_matches_v3d_mdh():
    """Verify pelvis (RPV) anthros match V3D's theia_model.mdh exactly."""
    rpv = DE_LEVA_MALE["RPV"]
    assert abs(rpv.mass_fraction - 0.1117) < 1e-6
    assert abs(rpv.k_xx - 0.615) < 1e-6
    assert abs(rpv.k_yy - 0.551) < 1e-6
    assert abs(rpv.k_zz - 0.587) < 1e-6


def test_de_leva_thigh_matches_v3d_mdh():
    rth = DE_LEVA_MALE["RTH"]
    assert abs(rth.mass_fraction - 0.1416) < 1e-6
    assert abs(rth.k_xx - 0.329) < 1e-6
    assert abs(rth.k_zz - 0.149) < 1e-6


def test_left_right_segments_have_same_anthros():
    pairs = [("RTH", "LTH"), ("RSK", "LSK"), ("RAR", "LAR"), ("RFA", "LFA"),
             ("RHA", "LHA"), ("RFT", "LFT"), ("RTO", "LTO")]
    for r, l in pairs:
        assert DE_LEVA_MALE[r] == DE_LEVA_MALE[l]


def test_v3d_mappings_cover_all_de_leva_segments():
    for v3d_code in DE_LEVA_MALE.keys():
        assert v3d_code in V3D_TO_THEIA_SEG
        assert v3d_code in V3D_TO_OSIM_BODY


def test_de_leva_fractions_sum_close_to_one():
    """Total body mass fractions should sum to ~0.97 (rest is misc small bodies)."""
    total = sum(a.mass_fraction for a in DE_LEVA_MALE.values())
    # Each side's symmetric segments are counted once in DE_LEVA_MALE's keys
    # (e.g. RTH and LTH each at 0.1416). So sum = head + pelvis + thorax +
    # 2×(thigh+shank+foot+toes+arm+forearm+hand). Should be ~0.97-0.98.
    assert 0.95 < total < 1.05


@pytest.mark.skipif(not C3D.exists() or not MDH.exists(), reason="fixtures not present")
def test_compute_segment_lengths_uses_c3d_distances():
    trial = read_theia_c3d(C3D, apply_vlb=DEFAULT_VLB_4X4)
    lengths = compute_segment_lengths(trial, mdh=parse_mdh(MDH))
    # Both limb pairs and MDH-direct lengths should be present
    expected_bodies = {
        "femur_r", "femur_l", "tibia_r", "tibia_l",
        "humerus_r", "humerus_l", "ulna_r", "ulna_l",
        "pelvis", "torso",
        "hand_r", "hand_l", "calcn_r", "calcn_l", "toes_r", "toes_l",
    }
    assert expected_bodies.issubset(lengths.keys())
    for body, L in lengths.items():
        assert 0.03 < L < 1.0, f"{body} length {L} out of plausible range"


@pytest.mark.skipif(not C3D.exists() or not MDH.exists(), reason="fixtures not present")
def test_scale_factors_in_plausible_range():
    trial = read_theia_c3d(C3D, apply_vlb=DEFAULT_VLB_4X4)
    lengths = compute_segment_lengths(trial, mdh=parse_mdh(MDH))
    factors = compute_scale_factors_full(lengths)
    for body, (sx, sy, sz) in factors.items():
        # All limbs should be within (0.5, 2.0) for any plausible adult athlete.
        assert 0.5 < sx < 2.0, f"{body} scale {sx} out of range"
        assert sx == sy == sz  # uniform default


def test_lai_defaults_for_all_bodies_we_personalize():
    """Every OpenSim body referenced by de Leva should have a LaiUhlrich length."""
    bodies = {b for b in V3D_TO_OSIM_BODY.values() if b is not None}
    for body in bodies:
        assert body in LAI_DEFAULT_LENGTHS_M, f"{body} missing from LAI_DEFAULT_LENGTHS_M"
