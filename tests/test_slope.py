"""Test slope application is just left-multiplication."""
from __future__ import annotations

import numpy as np
import pytest

from theia_osim.c3d_io.slope import apply_slope
from theia_osim.constants import DEFAULT_VLB_4X4


def test_apply_slope_with_identity_is_noop():
    T = np.tile(np.eye(4), (5, 1, 1))  # 5 frames of identity
    out = apply_slope({"foo": T}, np.eye(4))
    assert np.allclose(out["foo"], T)


def test_apply_slope_left_multiplies_transpose():
    """apply_slope uses vlb_4x4.T (V3D row-major convention)."""
    T = np.tile(np.eye(4), (3, 1, 1))
    T[:, 0:3, 3] = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    out = apply_slope({"foo": T}, DEFAULT_VLB_4X4)
    # We apply M = vlb_4x4.T, so M @ [x, 0, 0, 1] = vlb_4x4.T @ [x, 0, 0, 1]
    for i, x in enumerate([1.0, 2.0, 3.0]):
        expected = DEFAULT_VLB_4X4.T @ np.array([x, 0, 0, 1.0])
        assert np.allclose(out["foo"][i, :, 3], expected)


def test_apply_slope_pitcher_moves_in_plus_y():
    """In VLB, pitching forward direction must be +Y.

    LAB +X = pitching direction (per ref doc §1). After slope, this should
    map to VLB +Y, so a +X translation in LAB shows up as +Y in VLB.
    """
    T = np.tile(np.eye(4), (1, 1, 1))
    T[:, 0:3, 3] = np.array([[1.0, 0.0, 0.0]])  # pitcher 1m forward in LAB
    out = apply_slope({"foo": T}, DEFAULT_VLB_4X4)
    pos_vlb = out["foo"][0, :3, 3]
    # The pitcher's forward motion should land in +Y, not -Y.
    assert pos_vlb[1] > 0.5, f"pitcher should be at VLB +Y, got {pos_vlb}"


def test_apply_slope_double_application_equals_squared_transpose():
    T = np.tile(np.eye(4), (2, 1, 1))
    T[:, 0:3, 3] = np.random.RandomState(0).randn(2, 3)
    once = apply_slope({"foo": T}, DEFAULT_VLB_4X4)
    twice = apply_slope(once, DEFAULT_VLB_4X4)
    # apply_slope internally uses M.T, so applying twice = (M.T @ M.T)
    M_squared = DEFAULT_VLB_4X4.T @ DEFAULT_VLB_4X4.T
    expected = apply_slope({"foo": T}, M_squared.T)
    assert np.allclose(twice["foo"], expected["foo"])


def test_apply_slope_rejects_bad_shape():
    with pytest.raises(ValueError):
        apply_slope({"foo": np.zeros((5, 4, 4))}, np.zeros((3, 3)))


def test_apply_slope_rejects_non_3d_transform():
    with pytest.raises(ValueError):
        apply_slope({"foo": np.eye(4)}, DEFAULT_VLB_4X4)
