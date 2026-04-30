"""Path A — V3D-style analytical kinematics: compute OpenSim joint coordinates
directly from Theia segment 4×4s, skipping IK entirely.

V3D's Theia workflow (verified against C-Motion docs) tracks each segment
INDEPENDENTLY from its 4×4 rotation matrix — no kinematic-tree IK fit.
Joint angles are computed AFTER tracking as relative orientations between
adjacent segments. This module replicates that approach for OpenSim.

For every joint in the model:
  R_relative = (R_parent_world)^T @ R_child_world      # body-frame relative
  decompose R_relative into the joint's coordinate convention:
    - CustomJoint (3-DOF, Cardan Z-X-Y): scipy as_euler('ZXY')
    - PinJoint (1-DOF): axis-angle projection onto the joint axis

Output: an OpenSim .mot containing all coordinate trajectories — drop-in
replacement for IK output.
"""
from __future__ import annotations

import os as _os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import opensim as osim
import pandas as pd
from scipy.spatial.transform import Rotation

from .filter import lowpass_filtfilt


# Targeted lowpass on the throwing-arm Cardan coords. At gimbal lock
# (arm_add_r ≈ ±90°), neither ZXY branch is smooth: arm_flex_r and arm_rot_r
# jump ±50° in one frame even after smart-unwrap. ID's GCV spline turns that
# into a ~10-frame q̈ pulse, which inflates body-frame ω 1.3-2.8× on the arm
# and downstream kinetics 6-13×. A pre-ID lowpass on just these coords
# crushes the spike before it propagates. Set THROWING_ARM_LOWPASS_HZ=0 to
# disable. See scripts/fix_a_smooth_arm_coords.py for the diagnostic.
THROWING_ARM_LOWPASS_HZ: float = float(_os.environ.get("THROWING_ARM_LOWPASS_HZ", "20"))
THROWING_ARM_LOWPASS_COORDS: tuple[str, ...] = (
    "arm_flex_r", "arm_add_r", "arm_rot_r",
    "arm_flex_l", "arm_add_l", "arm_rot_l",
)


# Acromial joint shoulder Y-axis offset (degrees). Set by the experimental
# `LaiUhlrich2022_full_body_yroll60.osim` variant: both PhysicalOffsetFrames
# of acromial_r and acromial_l carry an Ry(+60°) orientation, which moves the
# Cardan ZXY middle-axis singularity off the throwing trajectory (peak |X|
# drops 82.8° → 55.7° on pose_filt_0.c3d). When >0, we apply the matching
# similarity transform to R_relative before ZXY decomposition so the .mot
# coordinates target the rotated joint frames. Toggle via env var
# `ACROMIAL_Y_OFFSET_DEG` (default 0 = stock model).
ACROMIAL_Y_OFFSET_DEG: float = float(_os.environ.get("ACROMIAL_Y_OFFSET_DEG", "0"))

# Shoulder parameterization (acromial_r/l rotation axis order). Default "ZXY"
# matches the stock LaiUhlrich2022 model. Override via env var
# `SHOULDER_PARAM` to test a different Cardan order — pair with the matching
# .osim variant (built by scripts/build_cardan_variant.py).
#
# Note: OpenSim CustomJoint hard-rejects collinear axes (e.g. ZYZ Euler
# raises `CustomJoint 'acromial_r' has collinear axes and are not well-defined`
# at model load). So Euler sequences (1st axis == 3rd axis) like V3D's ZYZ
# are NOT available within CustomJoint. Only the 6 Cardan permutations of
# {X,Y,Z} are valid here.
_VALID_CARDAN = {"ZXY", "ZYX", "XYZ", "XZY", "YXZ", "YZX"}
SHOULDER_PARAM: str = _os.environ.get("SHOULDER_PARAM", "ZXY").upper()
if SHOULDER_PARAM not in _VALID_CARDAN:
    raise ValueError(
        f"SHOULDER_PARAM={SHOULDER_PARAM!r} not in {sorted(_VALID_CARDAN)}. "
        f"Euler sequences (e.g. ZYZ) are not supported by OpenSim CustomJoint."
    )


# Theia segment → OpenSim body name (skipping ones with no 4×4 source).
THEIA_TO_BODY: dict[str, str] = {
    "pelvis": "pelvis",
    "torso": "torso",
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
    "r_larm": "ulna_r",
    "l_larm": "ulna_l",
    "r_hand": "hand_r",
    "l_hand": "hand_l",
}

BODY_TO_THEIA: dict[str, str] = {v: k for k, v in THEIA_TO_BODY.items()}


@dataclass(frozen=True)
class JointDef:
    name: str           # OpenSim joint name
    parent_body: str    # OpenSim body name (or "ground")
    child_body: str
    coords: tuple[str, ...]  # coordinate names in order
    kind: str           # "free", "custom_zxy", "pin_x", "pin_y", "pin_z", "weld", "skip"


# Joint definitions for LaiUhlrich2022. Verified by inspecting model:
# CustomJoints with rot axes (Z, X, Y) → intrinsic Cardan ZXY decomposition.
JOINT_DEFS: dict[str, JointDef] = {
    # 6-DOF root: 3 rotations (ZXY) + 3 translations
    "ground_pelvis": JointDef(
        "ground_pelvis", "ground", "pelvis",
        ("pelvis_tilt", "pelvis_list", "pelvis_rotation",
         "pelvis_tx", "pelvis_ty", "pelvis_tz"),
        "free",
    ),
    # 3-DOF spherical-equivalent
    "hip_r": JointDef(
        "hip_r", "pelvis", "femur_r",
        ("hip_flexion_r", "hip_adduction_r", "hip_rotation_r"), "custom_zxy"),
    "hip_l": JointDef(
        "hip_l", "pelvis", "femur_l",
        ("hip_flexion_l", "hip_adduction_l", "hip_rotation_l"), "custom_zxy"),
    "back": JointDef(
        "back", "pelvis", "torso",
        ("lumbar_extension", "lumbar_bending", "lumbar_rotation"), "custom_zxy"),
    "acromial_r": JointDef(
        "acromial_r", "torso", "humerus_r",
        # Coord names track SHOULDER_PARAM. ZXY = stock LaiUhlrich names.
        # XZY = anatomical-pitching labels (rotation1=X=abduction,
        # rotation2=Z=horizontal abduction, rotation3=Y=internal rotation).
        # Must match the .osim variant (built with `build_cardan_variant.py
        # --rename-coords`).
        ("shoulder_abd_r", "shoulder_hzn_r", "shoulder_int_rot_r")
        if SHOULDER_PARAM == "XZY"
        else ("arm_flex_r", "arm_add_r", "arm_rot_r"),
        "custom_cardan" if SHOULDER_PARAM != "ZXY" else "custom_zxy"),
    "acromial_l": JointDef(
        "acromial_l", "torso", "humerus_l",
        # L arm has sign-flipped X and Y axes when SHOULDER_PARAM=ZXY; for
        # other orders the sign-flip semantics need re-derivation, so we
        # treat them uniformly with R as a starting point.
        ("shoulder_abd_l", "shoulder_hzn_l", "shoulder_int_rot_l")
        if SHOULDER_PARAM == "XZY"
        else ("arm_flex_l", "arm_add_l", "arm_rot_l"),
        "custom_cardan" if SHOULDER_PARAM != "ZXY" else "custom_zxy_l"),

    # 1-DOF pin / coupled — use primary axis only
    # walker_knee is coupled; we treat as a single rotation about model-default
    "walker_knee_r": JointDef(
        "walker_knee_r", "femur_r", "tibia_r",
        ("knee_angle_r",), "pin_x"),
    "walker_knee_l": JointDef(
        "walker_knee_l", "femur_l", "tibia_l",
        ("knee_angle_l",), "pin_x"),
    "ankle_r": JointDef(
        "ankle_r", "tibia_r", "calcn_r", ("ankle_angle_r",), "pin_z"),
    "ankle_l": JointDef(
        "ankle_l", "tibia_l", "calcn_l", ("ankle_angle_l",), "pin_z"),
    "subtalar_r": JointDef(
        "subtalar_r", "calcn_r", "calcn_r",  # same body — see note below
        ("subtalar_angle_r",), "skip"),
    "subtalar_l": JointDef(
        "subtalar_l", "calcn_l", "calcn_l", ("subtalar_angle_l",), "skip"),
    "mtp_r": JointDef(
        "mtp_r", "calcn_r", "toes_r", ("mtp_angle_r",), "pin_z"),
    "mtp_l": JointDef(
        "mtp_l", "calcn_l", "toes_l", ("mtp_angle_l",), "pin_z"),

    "elbow_r": JointDef(
        "elbow_r", "humerus_r", "ulna_r", ("elbow_flex_r",), "pin_z"),
    "elbow_l": JointDef(
        "elbow_l", "humerus_l", "ulna_l", ("elbow_flex_l",), "pin_z"),
    "radioulnar_r": JointDef(
        "radioulnar_r", "ulna_r", "ulna_r",  # forearm rotation, no separate Theia segment
        ("pro_sup_r",), "skip"),
    "radioulnar_l": JointDef(
        "radioulnar_l", "ulna_l", "ulna_l", ("pro_sup_l",), "skip"),

    # Coupled patellofemoral and welded radius_hand have no independent kinematics
    "patellofemoral_r": JointDef(
        "patellofemoral_r", "femur_r", "femur_r", ("knee_angle_r_beta",), "skip"),
    "patellofemoral_l": JointDef(
        "patellofemoral_l", "femur_l", "femur_l", ("knee_angle_l_beta",), "skip"),
}


def _relative_rotation(R_parent: np.ndarray, R_child: np.ndarray) -> np.ndarray:
    """Per-frame R_rel = R_parent.T @ R_child. Inputs (T, 3, 3); output (T, 3, 3)."""
    return np.einsum("nji,njk->nik", R_parent, R_child)


def _theia_to_opensim_world_R(R: np.ndarray) -> np.ndarray:
    """Convert Theia world rotation submatrix to OpenSim-world orientation.

    Theia/VLB world is Z-up; OpenSim default is Y-up. We left-multiply by
    Rx(-90°) to swap frames (same convention used by recipe_a_trc and
    recipe_c_sto).
    """
    Rx_neg90 = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    return np.einsum("ij,njk->nik", Rx_neg90, R)


def _unwrap_deg(x: np.ndarray) -> np.ndarray:
    """Unwrap an angular time series (in degrees) at 360° discontinuities."""
    return np.degrees(np.unwrap(np.radians(x)))


def _smart_unwrap_cardan_zxy(angles_T_x_3: np.ndarray) -> np.ndarray:
    """Resolve gimbal-lock branch swaps in 3-DOF Cardan ZXY tracking.

    Cardan ZXY has a 2-fold ambiguity:
        (Z, X, Y)  ≡  (Z+180°, 180°−X, Y+180°)
    represent the same rotation matrix. Near X = ±90° (gimbal lock) tracking
    can flip branches between consecutive frames, producing simultaneous
    ~180° jumps in Z and Y while X reflects around 90°. Per-component
    np.unwrap can't detect this — both Z and Y appear to wrap legitimately.

    Algorithm: at each frame i, evaluate both candidate branches against the
    previously-unwrapped frame i−1; pick the branch with smaller squared
    distance after handling per-component ±360° wraps.

    Critical for the throwing-arm shoulder during late cocking → release
    where arm_add_r reaches ~80° and excites the singularity.
    """
    n = angles_T_x_3.shape[0]
    out = np.zeros_like(angles_T_x_3)
    out[0] = angles_T_x_3[0]
    for i in range(1, n):
        std = angles_T_x_3[i]
        alt = np.array([std[0] + 180.0, 180.0 - std[1], std[2] + 180.0])
        # Wrap each to (-180, 180].
        std_w = ((std + 180.0) % 360.0) - 180.0
        alt_w = ((alt + 180.0) % 360.0) - 180.0
        # For each candidate, pick the ±360°-shifted representative closest
        # to out[i-1] component-by-component.
        std_close = std_w + np.round((out[i - 1] - std_w) / 360.0) * 360.0
        alt_close = alt_w + np.round((out[i - 1] - alt_w) / 360.0) * 360.0
        d_std = float(np.sum((std_close - out[i - 1]) ** 2))
        d_alt = float(np.sum((alt_close - out[i - 1]) ** 2))
        out[i] = std_close if d_std <= d_alt else alt_close
    return out


def compute_coordinates(
    transforms_by_segment: dict[str, np.ndarray],
    sample_rate_hz: float,
    *,
    osim_axis_swap: bool = True,
    unwrap: bool = True,
    arm_lowpass_hz: float | None = None,
) -> pd.DataFrame:
    """Compute every OpenSim joint coordinate for every frame of the trial.

    Args:
        transforms_by_segment: TrialData.transforms (slope-corrected).
        sample_rate_hz: trial sample rate.
        osim_axis_swap: apply Rx(-90°) to map Theia Z-up → OpenSim Y-up.

    Returns:
        DataFrame with columns: time + every coordinate name in JOINT_DEFS.
    """
    if not transforms_by_segment:
        raise ValueError("no transforms")

    n_frames = next(iter(transforms_by_segment.values())).shape[0]
    times = np.arange(n_frames, dtype=np.float64) / sample_rate_hz

    # Pull per-body world rotations + positions (with optional axis swap).
    Rs: dict[str, np.ndarray] = {}
    ps: dict[str, np.ndarray] = {}
    for body, theia_seg in BODY_TO_THEIA.items():
        if theia_seg not in transforms_by_segment:
            continue
        T = transforms_by_segment[theia_seg]
        R = T[:, :3, :3]
        p = T[:, :3, 3]
        if osim_axis_swap:
            R = _theia_to_opensim_world_R(R)
            # Apply same Rx(-90°) to position vector
            Rx = Rotation.from_euler("x", -90, degrees=True).as_matrix()
            p = np.einsum("ij,nj->ni", Rx, p)
        Rs[body] = R
        ps[body] = p

    out: dict[str, np.ndarray] = {"time": times}

    for joint_name, jdef in JOINT_DEFS.items():
        if jdef.kind == "skip":
            for c in jdef.coords:
                out[c] = np.zeros(n_frames)
            continue

        if jdef.kind == "free":
            R_rel = Rs["pelvis"]
            ang = Rotation.from_matrix(R_rel).as_euler("ZXY", degrees=True)
            if unwrap:
                ang = _smart_unwrap_cardan_zxy(ang)
            out[jdef.coords[0]] = ang[:, 0]
            out[jdef.coords[1]] = ang[:, 1]
            out[jdef.coords[2]] = ang[:, 2]
            out[jdef.coords[3]] = ps["pelvis"][:, 0]
            out[jdef.coords[4]] = ps["pelvis"][:, 1]
            out[jdef.coords[5]] = ps["pelvis"][:, 2]
            continue

        if jdef.parent_body not in Rs or jdef.child_body not in Rs:
            for c in jdef.coords:
                out[c] = np.zeros(n_frames)
            continue

        R_rel = _relative_rotation(Rs[jdef.parent_body], Rs[jdef.child_body])

        # Acromial-only Ry(+α) offset to move gimbal lock off the throwing
        # trajectory. Both parent_offset and child_offset of acromial_r/l in
        # the matching .osim variant carry the same Ry(α) rotation; the joint's
        # observed relative rotation in the new frames is the similarity
        # transform R_off^T · R_rel · R_off.
        if ACROMIAL_Y_OFFSET_DEG != 0.0 and joint_name.startswith("acromial_"):
            R_off = Rotation.from_euler("y", ACROMIAL_Y_OFFSET_DEG, degrees=True).as_matrix()
            R_rel = np.einsum("ji,njk,kl->nil", R_off, R_rel, R_off)

        if jdef.kind == "custom_zxy":
            ang = Rotation.from_matrix(R_rel).as_euler("ZXY", degrees=True)
            if unwrap:
                ang = _smart_unwrap_cardan_zxy(ang)
            for i, c in enumerate(jdef.coords):
                out[c] = ang[:, i]
        elif jdef.kind == "custom_zxy_l":
            # acromial_l: axis2=-X, axis3=-Y → negate angles 1 and 2 (after unwrap)
            ang = Rotation.from_matrix(R_rel).as_euler("ZXY", degrees=True)
            if unwrap:
                ang = _smart_unwrap_cardan_zxy(ang)
            out[jdef.coords[0]] = ang[:, 0]
            out[jdef.coords[1]] = -ang[:, 1]
            out[jdef.coords[2]] = -ang[:, 2]
        elif jdef.kind == "custom_cardan":
            # Generic Cardan sequence selected by SHOULDER_PARAM env var.
            # All 6 Cardan permutations of {X,Y,Z} have a singularity at
            # middle = ±90°, but the geometric value of β depends on the
            # sequence. For the throwing trial: ZXY peaks β=82.8° (current,
            # 7° from lock); XZY peaks 49° (41° margin); YXZ 82° (similar
            # to ZXY); YZX 76°. See scripts/cardan_sweep.py for the table.
            ang = Rotation.from_matrix(R_rel).as_euler(SHOULDER_PARAM, degrees=True)
            if unwrap:
                ang = np.column_stack([_unwrap_deg(ang[:, i]) for i in range(3)])
            for i, c in enumerate(jdef.coords):
                out[c] = ang[:, i]
        elif jdef.kind in ("pin_x", "pin_y", "pin_z"):
            axis_idx = {"pin_x": 0, "pin_y": 1, "pin_z": 2}[jdef.kind]
            ang = Rotation.from_matrix(R_rel).as_euler("XYZ", degrees=True)
            arr = ang[:, axis_idx]
            if unwrap:
                arr = _unwrap_deg(arr)
            out[jdef.coords[0]] = arr

    cutoff = arm_lowpass_hz if arm_lowpass_hz is not None else THROWING_ARM_LOWPASS_HZ
    if cutoff > 0:
        for coord in THROWING_ARM_LOWPASS_COORDS:
            if coord in out and out[coord].size >= 16:
                out[coord] = lowpass_filtfilt(
                    out[coord][:, None], cutoff_hz=cutoff,
                    sample_rate_hz=sample_rate_hz, order=4,
                ).ravel()

    return pd.DataFrame(out)


def write_mot(coords: pd.DataFrame, path: Path | str) -> Path:
    """Write a DataFrame of OpenSim joint coordinates to a .mot file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(coords)
    n_cols = len(coords.columns)

    lines: list[str] = []
    lines.append(f"Coordinates")
    lines.append(f"version=1")
    lines.append(f"nRows={n_rows}")
    lines.append(f"nColumns={n_cols}")
    lines.append(f"inDegrees=yes")
    lines.append(f"endheader")
    lines.append("\t".join(coords.columns))
    for _, row in coords.iterrows():
        lines.append("\t".join(f"{v:.6f}" for v in row.values))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_recipe_d_mot(
    transforms_by_segment: dict[str, np.ndarray],
    out_mot: Path | str,
    sample_rate_hz: float,
    *,
    osim_axis_swap: bool = True,
    arm_lowpass_hz: float | None = None,
) -> Path:
    """End-to-end Path A: compute coords from Theia 4×4s + write OpenSim .mot."""
    coords = compute_coordinates(
        transforms_by_segment, sample_rate_hz,
        osim_axis_swap=osim_axis_swap, arm_lowpass_hz=arm_lowpass_hz,
    )
    return write_mot(coords, out_mot)
