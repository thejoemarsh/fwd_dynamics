"""Per-body free-vector frame corrections (OpenSim body frame → V3D body frame).

OpenSim's `BodyKinematics(express_in_local_frame=true)` writes ω in the
OpenSim body frame. V3D's YABIN signals are in V3D's segment frame. For a
given body, the two frames may differ by a constant rotation `R_OS_to_V3D`,
making any free-vector quantity (ω, F, M) need:

    v_v3d = R_OS_to_V3D @ v_opensim

The same correction applies to JointReaction force/moment outputs (also
free vectors expressed in a body frame).

Pelvis: empirically a 180° rotation about Z (the long/up axis). Verified by
applying `diag(-1, -1, +1)` to Recipe D's pelvis ω drops per-axis RMSE vs
V3D from `[178, 185, 12]` to `[13, 9, 12]` deg/s on `pose_filt_0.c3d`.

Other bodies will be populated as their kinetic signals come online (M2/M3).
"""
from __future__ import annotations

import numpy as np

# Body name → 3×3 rotation that converts an OpenSim-frame vector into the
# V3D segment-frame equivalent: v_v3d = R @ v_opensim.
BODY_FRAME_TO_V3D: dict[str, np.ndarray] = {
    "pelvis": np.diag([-1.0, -1.0, 1.0]),
}


def convert_body_frame_signal(
    arr: np.ndarray, body_name: str
) -> np.ndarray:
    """Apply the OpenSim → V3D body-frame rotation to a (T, 3) vector array.

    If `body_name` has no entry in `BODY_FRAME_TO_V3D`, the input is returned
    unchanged (i.e. assume frames align).

    Args:
        arr: shape (T, 3) — per-frame free-vector quantity in OpenSim body
             frame (ω, F, M).
        body_name: OpenSim body name (e.g. "pelvis", "torso", "humerus_r").

    Returns:
        (T, 3) array in V3D segment frame.
    """
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected (T, 3), got {arr.shape}")
    R = BODY_FRAME_TO_V3D.get(body_name)
    if R is None:
        return arr.copy()
    return arr @ R.T  # (T, 3) @ (3, 3) = (T, 3); same as einsum('ij,nj->ni', R, arr)
