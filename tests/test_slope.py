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


def test_apply_slope_left_multiplies():
    # Pose: translation only.
    T = np.tile(np.eye(4), (3, 1, 1))
    T[:, 0:3, 3] = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    out = apply_slope({"foo": T}, DEFAULT_VLB_4X4)
    # vlb @ [t; 1] for t=(x,0,0) → first column of vlb scaled by x
    for i, x in enumerate([1.0, 2.0, 3.0]):
        expected = DEFAULT_VLB_4X4 @ np.array([x, 0, 0, 1.0])
        assert np.allclose(out["foo"][i, :, 3], expected)


def test_apply_slope_double_application_equals_squared():
    T = np.tile(np.eye(4), (2, 1, 1))
    T[:, 0:3, 3] = np.random.RandomState(0).randn(2, 3)
    once = apply_slope({"foo": T}, DEFAULT_VLB_4X4)
    twice = apply_slope(once, DEFAULT_VLB_4X4)
    expected = apply_slope({"foo": T}, DEFAULT_VLB_4X4 @ DEFAULT_VLB_4X4)
    assert np.allclose(twice["foo"], expected["foo"])


def test_apply_slope_rejects_bad_shape():
    with pytest.raises(ValueError):
        apply_slope({"foo": np.zeros((5, 4, 4))}, np.zeros((3, 3)))


def test_apply_slope_rejects_non_3d_transform():
    with pytest.raises(ValueError):
        apply_slope({"foo": np.eye(4)}, DEFAULT_VLB_4X4)
