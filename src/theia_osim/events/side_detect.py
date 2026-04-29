"""Throwing-side detection (R vs L) — V3D parity.

V3D determines throwing side by comparing each hand's forward speed at the
BR (Release) frame: whichever is faster is the throwing side. This is more
robust than "max anywhere in trial" because the windup motion can transiently
spike the glove hand's forward speed.

The corresponding V3D output is `INFO::HAND` in procdb.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .hand_velocity import HandVelocity

Side = Literal["R", "L"]


def detect_throwing_side(hand_vel: HandVelocity, br_frame: int) -> Side:
    """Compare each hand's forward speed at BR; return 'R' or 'L'.

    "Forward" in our pipeline = +Y (VLB-corrected lab frame).

    Args:
        hand_vel: per-frame fingertip velocities for both hands.
        br_frame: ball-release frame index.

    Returns:
        'R' if right hand was faster at BR, else 'L'.
    """
    n = hand_vel.r_hand_vel.shape[0]
    if not 0 <= br_frame < n:
        raise ValueError(f"br_frame {br_frame} outside [0, {n})")
    r_fwd = float(np.abs(hand_vel.r_hand_vel[br_frame, 1]))
    l_fwd = float(np.abs(hand_vel.l_hand_vel[br_frame, 1]))
    return "R" if r_fwd >= l_fwd else "L"
