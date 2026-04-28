"""Test the Recipe C 4×4 → quaternion conversion."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from theia_osim.import_pipeline.recipe_c_sto import transforms_to_quaternions


def test_quaternion_roundtrip_preserves_rotation():
    """Convert random rotations to quaternions and back; verify identity."""
    rng = np.random.RandomState(0)
    n = 50
    R = Rotation.random(n, random_state=rng).as_matrix()
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = R

    times, quats = transforms_to_quaternions(
        {"pelvis": T},
        segment_to_body={"pelvis": "pelvis"},
        osim_axis_swap=False,  # keep raw to test round-trip purely
    )
    assert "pelvis" in quats
    q_wxyz = quats["pelvis"]  # (n, 4) in (w, x, y, z)
    # Reorder to scipy's (x, y, z, w) and round-trip back to matrix.
    q_xyzw = np.column_stack([q_wxyz[:, 1:], q_wxyz[:, 0]])
    R_back = Rotation.from_quat(q_xyzw).as_matrix()
    np.testing.assert_allclose(R_back, R, atol=1e-10)


def test_quaternions_have_unit_norm():
    rng = np.random.RandomState(1)
    n = 20
    R = Rotation.random(n, random_state=rng).as_matrix()
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = R
    _, quats = transforms_to_quaternions(
        {"pelvis": T},
        segment_to_body={"pelvis": "pelvis"},
        osim_axis_swap=False,
    )
    norms = np.linalg.norm(quats["pelvis"], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_axis_swap_applies_rx_minus_90():
    """Identity transform, with axis swap, should produce Rx(-90°) quaternion."""
    n = 1
    T = np.tile(np.eye(4), (n, 1, 1))
    _, quats = transforms_to_quaternions(
        {"pelvis": T},
        segment_to_body={"pelvis": "pelvis"},
        osim_axis_swap=True,
    )
    q_wxyz = quats["pelvis"][0]
    expected_xyzw = Rotation.from_euler("x", -90, degrees=True).as_quat()
    expected_wxyz = np.array([expected_xyzw[3], *expected_xyzw[:3]])
    # Quats can flip sign and represent same rotation - normalize sign.
    if np.dot(q_wxyz, expected_wxyz) < 0:
        expected_wxyz = -expected_wxyz
    np.testing.assert_allclose(q_wxyz, expected_wxyz, atol=1e-12)


def test_skips_duplicate_target_body():
    """Both head and torso map to LaiUhlrich's torso — only one output expected."""
    n = 3
    T = np.tile(np.eye(4), (n, 1, 1))
    _, quats = transforms_to_quaternions(
        {"head": T, "torso": T},
        segment_to_body={"head": "torso", "torso": "torso"},
        osim_axis_swap=False,
    )
    assert list(quats.keys()) == ["torso"]
