"""Test landmark transforms."""
from __future__ import annotations

import numpy as np
import pytest

from theia_osim.import_pipeline.landmarks import (
    Landmark,
    SegmentMarkers,
    synthesize_all_markers,
    transform_landmark_to_world,
)


def test_origin_landmark_returns_translation_column():
    """Local (0,0,0) should equal the segment 4×4 translation column."""
    rng = np.random.RandomState(42)
    n = 10
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0:3, 3] = rng.randn(n, 3)  # random translations
    world = transform_landmark_to_world(T, np.zeros(3))
    assert world.shape == (n, 3)
    np.testing.assert_allclose(world, T[:, 0:3, 3])


def test_offset_landmark_uses_rotation():
    n = 4
    T = np.tile(np.eye(4), (n, 1, 1))
    # 90° rotation about Z for half the frames.
    T[2:, 0, 0] = 0
    T[2:, 0, 1] = -1
    T[2:, 1, 0] = 1
    T[2:, 1, 1] = 0
    local = np.array([1.0, 0.0, 0.0])
    world = transform_landmark_to_world(T, local)
    # Identity frames: world == local == (1, 0, 0)
    np.testing.assert_allclose(world[0], [1, 0, 0])
    np.testing.assert_allclose(world[1], [1, 0, 0])
    # Rotated frames: (1, 0, 0) → (0, 1, 0) under Rz(90°)
    np.testing.assert_allclose(world[2], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(world[3], [0, 1, 0], atol=1e-12)


def test_synthesize_all_markers_iterates_catalog():
    n = 5
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0:3, 3] = np.arange(n)[:, None] * 0.1  # diagonal walk
    catalog = {
        "pelvis": SegmentMarkers(
            segment="pelvis",
            body="pelvis",
            landmarks=(
                Landmark(name="origin", local_xyz=np.zeros(3)),
                Landmark(name="lat", local_xyz=np.array([0.0, 0.1, 0.0])),
            ),
        )
    }
    out = synthesize_all_markers({"pelvis": T}, catalog)
    assert set(out.keys()) == {"origin", "lat"}
    assert out["origin"].shape == (n, 3)


def test_synthesize_raises_on_missing_segment():
    catalog = {
        "pelvis": SegmentMarkers(
            segment="pelvis", body="pelvis",
            landmarks=(Landmark(name="o", local_xyz=np.zeros(3)),),
        )
    }
    with pytest.raises(ValueError, match="catalog references segment"):
        synthesize_all_markers({}, catalog)


def test_transform_landmark_rejects_bad_shapes():
    with pytest.raises(ValueError):
        transform_landmark_to_world(np.eye(4), np.zeros(3))
    with pytest.raises(ValueError):
        transform_landmark_to_world(np.tile(np.eye(4), (5, 1, 1)), np.zeros(4))
