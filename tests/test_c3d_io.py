"""Test the c3d_io modules against the in-repo sample trial."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from theia_osim.c3d_io.reader import list_segments, read_theia_c3d
from theia_osim.constants import DEFAULT_VLB_4X4

C3D = Path(__file__).resolve().parents[1] / "pose_filt_0.c3d"


@pytest.mark.skipif(not C3D.exists(), reason="sample c3d not present")
def test_list_segments_includes_all_19():
    segs = list_segments(C3D)
    assert len(segs) == 19
    expected = {
        "worldbody", "head", "torso", "l_uarm", "l_larm", "l_hand",
        "r_uarm", "r_larm", "r_hand", "pelvis", "l_thigh", "l_shank",
        "l_foot", "l_toes", "r_thigh", "r_shank", "r_foot", "r_toes",
        "pelvis_shifted",
    }
    assert set(segs) == expected


@pytest.mark.skipif(not C3D.exists(), reason="sample c3d not present")
def test_read_theia_c3d_drops_ignored_and_loads_661_frames():
    t = read_theia_c3d(C3D)
    assert t.n_frames == 661
    assert t.sample_rate_hz == 300.0
    # 19 raw - 2 ignored (worldbody, pelvis_shifted) = 17
    assert len(t.transforms) == 17
    assert "worldbody" not in t.transforms
    assert "pelvis_shifted" not in t.transforms
    assert "pelvis" in t.transforms
    assert t.transforms["pelvis"].shape == (661, 4, 4)


@pytest.mark.skipif(not C3D.exists(), reason="sample c3d not present")
def test_no_nan_in_loaded_transforms():
    t = read_theia_c3d(C3D)
    for name, T in t.transforms.items():
        assert np.all(np.isfinite(T)), f"{name} has NaN/Inf"


@pytest.mark.skipif(not C3D.exists(), reason="sample c3d not present")
def test_meta_has_theia_version_and_filt_freq():
    t = read_theia_c3d(C3D)
    assert t.meta.theia_version == (2025, 2, 0)
    assert t.meta.filt_freq_hz == 30.0
    assert t.meta.filtered is True
    # 14 inertia segments expected (head, torso, both uarm/larm/hand, thigh/shank/foot)
    assert len(t.meta.segments_anthro) == 14


@pytest.mark.skipif(not C3D.exists(), reason="sample c3d not present")
def test_slope_application_changes_transforms():
    raw = read_theia_c3d(C3D)
    rotated = read_theia_c3d(C3D, apply_vlb=DEFAULT_VLB_4X4)
    assert raw.slope_applied is False
    assert rotated.slope_applied is True
    # Pelvis world position should differ after VLB rotation
    diff = np.linalg.norm(
        raw.transforms["pelvis"][330, :3, 3] - rotated.transforms["pelvis"][330, :3, 3]
    )
    assert diff > 0.01  # meters
