"""Mound-slope rotation (Lab → VLB).

Theia's lab frame has +Z up but is parallel to the floor, so the mound surface
is tilted by ~4.76° from vertical. Driveline's V3D pipeline removes this tilt
+ rotates the lab so that VLB +Y points in the pitching direction (forward).

The `vlb_4x4` constant matches V3D's stored matrix (Rz(-90°) × slope-tilt).
Empirically, V3D applies it such that VLB +Y = forward and the pitcher moves
in +Y over the trial. Our pipeline must match: we left-multiply by the
TRANSPOSE of `vlb_4x4` so that the resulting world frame has +Y = forward.

Once everything is in VLB, the OpenSim ground frame *is* VLB and there's no
special-cased "express in VLB" calls anywhere downstream.
"""
from __future__ import annotations

import numpy as np


def apply_slope(
    transforms: dict[str, np.ndarray], vlb_4x4: np.ndarray
) -> dict[str, np.ndarray]:
    """Apply the V3D slope/orientation correction to every segment 4×4.

    The conventional V3D `vlb_4x4` matrix is stored as a row-major rotation
    such that to express a LAB-frame transform in the VLB frame, you
    left-multiply by `vlb_4x4.T` (V3D's internal convention puts the new
    basis vectors as rows; reading them as columns flips the rotation).
    Verified empirically against pose_filt_0.c3d: with `.T`, pitcher moves
    +1.44 m in VLB +Y as expected.

    Args:
        transforms: dict mapping segment name → (T, 4, 4) array of per-frame poses.
        vlb_4x4: (4, 4) V3D-stored Lab→VLB matrix (transposed internally).

    Returns:
        New dict with same keys; values are slope-corrected (T, 4, 4) arrays.
    """
    if vlb_4x4.shape != (4, 4):
        raise ValueError(f"vlb_4x4 must be 4×4, got {vlb_4x4.shape}")
    M = vlb_4x4.T  # V3D row-major convention → use transpose for change of basis
    out: dict[str, np.ndarray] = {}
    for name, T in transforms.items():
        if T.ndim != 3 or T.shape[1:] != (4, 4):
            raise ValueError(f"transform {name!r} must be (T, 4, 4); got {T.shape}")
        # Per-frame: T_in_vlb = M @ T_in_lab
        out[name] = np.einsum("ij,njk->nik", M, T)
    return out
