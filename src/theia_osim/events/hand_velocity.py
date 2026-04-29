"""Hand distal-end (fingertip) world position + velocity for V3D-style events.

V3D detects ball release (BR) from peak forward speed of the hand's distal end
(fingertip). The fingertip isn't tracked directly by Theia — it's computed
from the hand segment's 4×4 pose plus the segment's long axis × hand length.

Theia segment frame convention (verified empirically on `pose_filt_0.c3d`):
the hand long axis is **−Z** (origin is the wrist; the fingertip is in the
−Z direction). So:

    fingertip_world(t) = hand_origin_world(t) + R_hand_world(t) @ [0, 0, -length]

Velocity is `np.gradient` then 20 Hz Butterworth filtfilt (matches V3D's
1-pass bidirectional spec).

Hand length cascade: MDH `Length_RHA`/`Length_LHA` → Theia c3d INERTIA_*[0]
length field → 8 cm fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..c3d_io.mdh_parser import MDHMetrics
from ..c3d_io.reader import TrialData
from ..kinematics_postprocess.filter import lowpass_filtfilt

DEFAULT_HAND_LENGTH_M = 0.08
HAND_DISTAL_LOCAL = np.array([0.0, 0.0, -1.0])  # -Z = proximal-to-distal


@dataclass(frozen=True)
class HandVelocity:
    """Per-frame fingertip world position + velocity for both hands."""

    r_hand_pos: np.ndarray  # (T, 3) world position of right fingertip
    l_hand_pos: np.ndarray  # (T, 3)
    r_hand_vel: np.ndarray  # (T, 3) world velocity, 20 Hz lowpass
    l_hand_vel: np.ndarray  # (T, 3)
    sample_rate_hz: float
    r_hand_length_m: float
    l_hand_length_m: float
    length_source: dict[str, str]  # 'r' / 'l' → 'mdh' | 'theia' | 'default'


def _hand_length(
    side: str, trial: TrialData, mdh: MDHMetrics | None
) -> tuple[float, str]:
    """Resolve hand length using cascade. Returns (length_m, source_label)."""
    side_lower = side.lower()
    mdh_key = f"{side_lower}ha"
    theia_seg = f"{side_lower}_hand"
    if mdh is not None and mdh_key in mdh.segment_lengths_m:
        v = float(mdh.segment_lengths_m[mdh_key])
        if v > 0:
            return v, "mdh"
    anthro = trial.meta.segments_anthro.get(theia_seg)
    if anthro is not None and anthro.length_m > 0:
        return float(anthro.length_m), "theia"
    return DEFAULT_HAND_LENGTH_M, "default"


def _distal_end_world(hand_4x4: np.ndarray, length_m: float) -> np.ndarray:
    """World position of fingertip per frame (hand_4x4 origin + R @ -Z * length)."""
    R = hand_4x4[:, :3, :3]
    origin = hand_4x4[:, :3, 3]
    distal_local = HAND_DISTAL_LOCAL * length_m
    return origin + np.einsum("nij,j->ni", R, distal_local)


def compute_hand_velocity(
    trial: TrialData,
    *,
    mdh: MDHMetrics | None = None,
    lowpass_hz: float = 20.0,
    butter_order: int = 4,
) -> HandVelocity:
    """Fingertip world position + filtered velocity for both hands.

    The trial's transforms are expected to be VLB-corrected lab frame (forward
    = +Y after `apply_slope`). No Theia → OpenSim Y-up swap is applied here —
    BR detection works in the V3D-equivalent VLB frame.
    """
    r_len, r_src = _hand_length("r", trial, mdh)
    l_len, l_src = _hand_length("l", trial, mdh)

    r_pose = trial.transforms.get("r_hand")
    l_pose = trial.transforms.get("l_hand")
    if r_pose is None or l_pose is None:
        raise ValueError("trial missing r_hand or l_hand transforms")

    r_pos = _distal_end_world(r_pose, r_len)
    l_pos = _distal_end_world(l_pose, l_len)

    dt = 1.0 / trial.sample_rate_hz
    r_vel_raw = np.gradient(r_pos, dt, axis=0)
    l_vel_raw = np.gradient(l_pos, dt, axis=0)

    r_vel = lowpass_filtfilt(
        r_vel_raw, cutoff_hz=lowpass_hz,
        sample_rate_hz=trial.sample_rate_hz, order=butter_order,
    )
    l_vel = lowpass_filtfilt(
        l_vel_raw, cutoff_hz=lowpass_hz,
        sample_rate_hz=trial.sample_rate_hz, order=butter_order,
    )

    return HandVelocity(
        r_hand_pos=r_pos,
        l_hand_pos=l_pos,
        r_hand_vel=r_vel,
        l_hand_vel=l_vel,
        sample_rate_hz=trial.sample_rate_hz,
        r_hand_length_m=r_len,
        l_hand_length_m=l_len,
        length_source={"r": r_src, "l": l_src},
    )
