"""Constants for the Theia → OpenSim pipeline.

Single source of truth for:
- Theia segment names (as they appear in .c3d ROTATION labels).
- Theia ↔ OpenSim body name mapping for the LaiUhlrich2022 model.
- Default mound-slope rotation matrix (Driveline VLB).
- Sample-rate-dependent constants from the V3D pipeline.
"""
from __future__ import annotations

import numpy as np

# -- Theia segment names (without the "_4X4" suffix, normalized) ----------------
# Order matters for some downstream consumers; keep stable.
THEIA_SEGMENTS: tuple[str, ...] = (
    "head",
    "torso",
    "l_uarm",
    "l_larm",
    "l_hand",
    "r_uarm",
    "r_larm",
    "r_hand",
    "pelvis",
    "l_thigh",
    "l_shank",
    "l_foot",
    "l_toes",
    "r_thigh",
    "r_shank",
    "r_foot",
    "r_toes",
)

# Segments to drop on load — `worldbody` is identity, `pelvis_shifted` duplicates
# `pelvis` in Theia v2025.x.
IGNORED_SEGMENTS: tuple[str, ...] = ("worldbody", "pelvis_shifted")

# -- Theia → OpenSim (LaiUhlrich2022) body name map ----------------------------
THEIA_TO_OSIM_BODY: dict[str, str] = {
    "pelvis": "pelvis",
    "torso": "torso",
    "head": "torso",  # LaiUhlrich has no separate head body; head pose folds into torso
    "r_thigh": "femur_r",
    "l_thigh": "femur_l",
    "r_shank": "tibia_r",
    "l_shank": "tibia_l",
    "r_foot": "calcn_r",
    "l_foot": "calcn_l",
    "r_toes": "toes_r",
    "l_toes": "toes_l",
    "r_uarm": "humerus_r",
    "l_uarm": "humerus_l",
    "r_larm": "ulna_r",  # forearm = ulna in LaiUhlrich (radius_hand is welded)
    "l_larm": "ulna_l",
    "r_hand": "hand_r",
    "l_hand": "hand_l",
}

# -- Default mound-slope rotation (Lab → VLB) ----------------------------------
# From Driveline V3D pipeline (`01_v6_model_build.v3s`), ~4.76° tilt.
DEFAULT_VLB_4X4: np.ndarray = np.array(
    [
        [0.0, 0.9965, -0.0831, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0831, 0.9965, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# -- V3D foot-plant frame offsets per sample rate ------------------------------
# Used by events.pitch_events.detect_foot_plant in M3.
FP_FRAME_OFFSET: dict[int, int] = {
    360: -4,
    300: -3,
    250: -3,
}

SUPPORTED_SAMPLE_RATES: tuple[int, ...] = (250, 300, 360)
