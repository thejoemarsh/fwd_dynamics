"""de Leva 1996 anthropometric tables — male values used by Visual3D.

Reference: de Leva, P. (1996). "Adjustments to Zatsiorsky-Seluyanov's segment
inertia parameters." Journal of Biomechanics, 29(9), 1223-1230.

Visual3D's Theia model file (theia_model.mdh) uses these exact values:
  Pelvis     mass_frac=0.1117  k_xx/yy/zz=0.615/0.551/0.587  com_axial=0.6115
  Thigh      mass_frac=0.1416  k_xx/yy/zz=0.329/0.329/0.149  com_axial=0.4095
  ...

We use the same so that downstream OpenSim ID/JointReaction outputs are
quantitatively comparable to V3D's.

Notes:
- Mass fractions are the segment mass / total body mass.
- COM axial is the proximal-to-COM distance / segment length.
  V3D stores PROX_TO_CG_AXIAL = com_axial * SEG_LENGTH; we use the same.
- Inertia radii (k_xx, k_yy, k_zz) are the radii of gyration about each
  axis at the COM, expressed as a fraction of segment length.
  Per-axis inertia = mass × (k × length)².
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentAnthropometry:
    """de Leva male anthropometric parameters for one segment."""
    mass_fraction: float       # frac of total body mass
    com_axial_frac: float      # proximal-to-COM / segment length (V3D PROX_TO_CG_AXIAL)
    k_xx: float                # radius of gyration about X (medio-lateral) / length
    k_yy: float                # about Y (anterior-posterior) / length
    k_zz: float                # about Z (longitudinal) / length


# de Leva 1996 male values used by V3D's theia_model.mdh.
# Keys match V3D's segment codes: RPV/RTH/LTH/RSK/LSK/RFT/LFT/RTO/LTO/RTX/RTA/RHE/RAR/LAR/RFA/LFA/RHA/LHA
DE_LEVA_MALE: dict[str, SegmentAnthropometry] = {
    # Lower trunk (pelvis)
    "RPV": SegmentAnthropometry(0.1117, 0.6115, 0.615, 0.551, 0.587),
    # Thigh (each)
    "RTH": SegmentAnthropometry(0.1416, 0.4095, 0.329, 0.329, 0.149),
    "LTH": SegmentAnthropometry(0.1416, 0.4095, 0.329, 0.329, 0.149),
    # Shank (each)
    "RSK": SegmentAnthropometry(0.0433, 0.4459, 0.255, 0.249, 0.103),
    "LSK": SegmentAnthropometry(0.0433, 0.4459, 0.255, 0.249, 0.103),
    # Foot (each)
    "RFT": SegmentAnthropometry(0.0137, 0.4415, 0.257, 0.245, 0.124),
    "LFT": SegmentAnthropometry(0.0137, 0.4415, 0.257, 0.245, 0.124),
    # Toes (each) — small, treated like a sub-foot. Fractions drawn from foot.
    "RTO": SegmentAnthropometry(0.0010, 0.50, 0.30, 0.30, 0.30),
    "LTO": SegmentAnthropometry(0.0010, 0.50, 0.30, 0.30, 0.30),
    # Upper arm (each)
    "RAR": SegmentAnthropometry(0.0271, 0.5772, 0.285, 0.269, 0.158),
    "LAR": SegmentAnthropometry(0.0271, 0.5772, 0.285, 0.269, 0.158),
    # Forearm (each)
    "RFA": SegmentAnthropometry(0.0162, 0.4574, 0.276, 0.265, 0.121),
    "LFA": SegmentAnthropometry(0.0162, 0.4574, 0.276, 0.265, 0.121),
    # Hand (each)
    "RHA": SegmentAnthropometry(0.0061, 0.7900, 0.628, 0.513, 0.401),
    "LHA": SegmentAnthropometry(0.0061, 0.7900, 0.628, 0.513, 0.401),
    # Thorax (upper trunk + abdomen lumped per V3D's RTX = 0.3 fraction)
    "RTX": SegmentAnthropometry(0.3000, 0.5000, 0.450, 0.450, 0.275),
    # Head + neck
    "RHE": SegmentAnthropometry(0.0694, 0.4998, 0.303, 0.315, 0.261),
}


# V3D segment code → Theia c3d segment label (lowercase) → OpenSim body name.
# This mapping lets us look up de Leva fractions by V3D code, then apply mass
# overrides to the matching OpenSim body.
V3D_TO_THEIA_SEG: dict[str, str] = {
    "RPV": "pelvis",
    "RTH": "r_thigh",
    "LTH": "l_thigh",
    "RSK": "r_shank",
    "LSK": "l_shank",
    "RFT": "r_foot",
    "LFT": "l_foot",
    "RTO": "r_toes",
    "LTO": "l_toes",
    "RAR": "r_uarm",
    "LAR": "l_uarm",
    "RFA": "r_larm",
    "LFA": "l_larm",
    "RHA": "r_hand",
    "LHA": "l_hand",
    "RTX": "torso",
    "RHE": "head",
}


V3D_TO_OSIM_BODY: dict[str, str] = {
    "RPV": "pelvis",
    "RTH": "femur_r",
    "LTH": "femur_l",
    "RSK": "tibia_r",
    "LSK": "tibia_l",
    "RFT": "calcn_r",
    "LFT": "calcn_l",
    "RTO": "toes_r",
    "LTO": "toes_l",
    "RAR": "humerus_r",
    "LAR": "humerus_l",
    "RFA": "ulna_r",
    "LFA": "ulna_l",
    "RHA": "hand_r",
    "LHA": "hand_l",
    "RTX": "torso",
    "RHE": None,  # LaiUhlrich2022 has no separate head body
}
