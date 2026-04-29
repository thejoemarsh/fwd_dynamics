"""Pitch-event detection (BR, MER) — V3D parity.

Ports the algorithms in `v3d_scripts/05_determine_side.v3s`:

BR (Release):
    1. Compute hand fingertip forward speeds (already done by hand_velocity).
    2. Build "release signal" = sum of (|fwd_speed| > 12.5 m/s gating) for both
       hands. The throwing hand naturally dominates because that's where
       speeds are highest.
    3. Find peak of release signal in [start + 100 frames, end].
    4. BR = peak frame + 3-frame offset (mechanical lag between peak hand
       speed and ball release).

MER (MAX_EXT, max external rotation):
    1. Compute throwing-arm shoulder relative rotation (torso → humerus).
    2. Decompose in **Y-X-Z** Cardan order (different from our default Z-X-Y!)
       with NEGATEY=TRUE, NEGATEZ=TRUE to match V3D's sign convention.
    3. Take the Z component (humerus long-axis rotation = external rotation).
    4. np.unwrap to handle 360° jumps.
    5. MER = argmax of Z over window [BR - 50, BR].

Forward direction in our pipeline (after `apply_slope`) = +Y in VLB.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..c3d_io.mdh_parser import MDHMetrics
from ..c3d_io.reader import TrialData
from .hand_velocity import HandVelocity, compute_hand_velocity
from .side_detect import Side, detect_throwing_side

BR_GATE_M_S = 12.5
BR_OFFSET_FRAMES = 3
BR_MIN_START_FRAMES = 100  # V3D's `START_OF_RECORDING + 100`
MER_LOOKBACK_FRAMES = 50


@dataclass(frozen=True)
class Events:
    br_frame: int
    br_time: float
    mer_frame: int
    mer_time: float
    throwing_side: Side


def _release_signal(hand_vel: HandVelocity) -> np.ndarray:
    """V3D's gated release signal: (T,) array, sum of forward speeds > 12.5 m/s.

    "Forward" = +Y (VLB lab frame).
    """
    r_fwd = np.abs(hand_vel.r_hand_vel[:, 1])
    l_fwd = np.abs(hand_vel.l_hand_vel[:, 1])
    r_g = (r_fwd > BR_GATE_M_S).astype(np.float64) * r_fwd
    l_g = (l_fwd > BR_GATE_M_S).astype(np.float64) * l_fwd
    return r_g + l_g


def detect_release(hand_vel: HandVelocity) -> int:
    """Detect ball-release frame from gated forward-speed peak + 3-frame offset."""
    sig = _release_signal(hand_vel)
    n = sig.shape[0]
    lo = min(BR_MIN_START_FRAMES, n - 1)
    search = sig[lo:]
    if not np.any(search > 0):
        raise RuntimeError(
            "No frames exceed 12.5 m/s gate — is this actually a pitching trial?"
        )
    peak = lo + int(np.argmax(search))
    return min(peak + BR_OFFSET_FRAMES, n - 1)


def _shoulder_relative_rotation(
    trial: TrialData, side: Side
) -> np.ndarray:
    """Per-frame R_torso_to_humerus = R_torso.T @ R_humerus, shape (T, 3, 3)."""
    torso = trial.transforms.get("torso")
    uarm = trial.transforms.get(f"{side.lower()}_uarm")
    if torso is None or uarm is None:
        raise ValueError(f"trial missing torso or {side.lower()}_uarm transforms")
    R_torso = torso[:, :3, :3]
    R_uarm = uarm[:, :3, :3]
    return np.einsum("nji,njk->nik", R_torso, R_uarm)


def _shoulder_yxz_negyz(R_rel: np.ndarray) -> np.ndarray:
    """Decompose (T, 3, 3) into Y-X-Z Cardan with NEGATEY=NEGATEZ=TRUE.

    Returns (T, 3) deg in the order (Y, X, Z) — same column order as
    `Rotation.as_euler('YXZ', ...)`. Z is what V3D pulls for external
    rotation.
    """
    eul = Rotation.from_matrix(R_rel).as_euler("YXZ", degrees=True)
    eul[:, 0] *= -1.0  # NEGATEY
    eul[:, 2] *= -1.0  # NEGATEZ
    return eul


def detect_max_external_rotation(
    trial: TrialData, br_frame: int, throwing_side: Side
) -> int:
    """MER frame = argmax of unwrapped shoulder Z (Y-X-Z Cardan) in [BR-50, BR]."""
    R_rel = _shoulder_relative_rotation(trial, throwing_side)
    eul = _shoulder_yxz_negyz(R_rel)
    z = np.unwrap(np.deg2rad(eul[:, 2]))
    z = np.rad2deg(z)
    n = z.shape[0]
    lo = max(0, br_frame - MER_LOOKBACK_FRAMES)
    hi = min(n, br_frame + 1)
    if hi <= lo:
        raise RuntimeError(f"MER window [{lo},{hi}) empty (br_frame={br_frame})")
    return lo + int(np.argmax(z[lo:hi]))


def detect_events(trial: TrialData, mdh: MDHMetrics | None = None) -> Events:
    """Run all event detection. Returns BR + MER frames/times + throwing side."""
    hand_vel = compute_hand_velocity(trial, mdh=mdh)
    br_frame = detect_release(hand_vel)
    side = detect_throwing_side(hand_vel, br_frame)
    mer_frame = detect_max_external_rotation(trial, br_frame, side)
    dt = 1.0 / trial.sample_rate_hz
    return Events(
        br_frame=br_frame,
        br_time=br_frame * dt,
        mer_frame=mer_frame,
        mer_time=mer_frame * dt,
        throwing_side=side,
    )
