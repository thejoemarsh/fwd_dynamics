"""Mound-slope rotation (Lab → VLB).

Theia's lab frame has +Z up but is parallel to the floor, so the mound surface
is tilted by ~4.76° from vertical. Driveline's V3D pipeline removes this tilt
by left-multiplying every segment 4×4 by `vlb_4x4`. We do the same here at
c3d-load time — once everything is in VLB, the OpenSim ground frame *is* VLB
and there's no special-cased "express in VLB" calls anywhere downstream.
"""
from __future__ import annotations

import numpy as np


def apply_slope(
    transforms: dict[str, np.ndarray], vlb_4x4: np.ndarray
) -> dict[str, np.ndarray]:
    """Left-multiply every segment 4×4 by `vlb_4x4`.

    Args:
        transforms: dict mapping segment name → (T, 4, 4) array of per-frame poses.
        vlb_4x4: (4, 4) Lab → VLB transform.

    Returns:
        New dict with same keys; values are slope-corrected (T, 4, 4) arrays.
    """
    if vlb_4x4.shape != (4, 4):
        raise ValueError(f"vlb_4x4 must be 4×4, got {vlb_4x4.shape}")
    out: dict[str, np.ndarray] = {}
    for name, T in transforms.items():
        if T.ndim != 3 or T.shape[1:] != (4, 4):
            raise ValueError(f"transform {name!r} must be (T, 4, 4); got {T.shape}")
        # Per-frame: T_in_vlb = vlb_4x4 @ T_in_lab
        out[name] = np.einsum("ij,njk->nik", vlb_4x4, T)
    return out
