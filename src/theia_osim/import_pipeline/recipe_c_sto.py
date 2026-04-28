"""Recipe C: convert segment 4×4 rotation matrices to OpenSense .sto orientations.

Each Theia segment ships a per-frame 3×3 rotation submatrix (already orthonormal,
det=+1, no NaN). We convert to quaternions and write an OpenSense-format .sto
that `IMUInverseKinematicsTool` consumes directly. This is mathematically the
most direct fit to Theia's data model — segment orientations in, joint angles out.

OpenSense expects a TimeSeriesTable_Quaternion with columns named after the
target OpenSim body names (e.g. "pelvis", "torso", "femur_r"). Quaternions are
formatted as the literal string "<w>,<x>,<y>,<z>" per cell.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..constants import THEIA_TO_OSIM_BODY


def transforms_to_quaternions(
    transforms_by_segment: dict[str, np.ndarray],
    segment_to_body: dict[str, str] | None = None,
    *,
    osim_axis_swap: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert segment 4×4s → per-body quaternions (w, x, y, z).

    Args:
        transforms_by_segment: TrialData.transforms (slope-corrected).
        segment_to_body: Theia segment name → OpenSim body name.
            Defaults to constants.THEIA_TO_OSIM_BODY.
        osim_axis_swap: rotate Theia Z-up → OpenSim Y-up (Rx(-90°) about world).

    Returns:
        (n_frames,) quaternions for each body in a dict.
    """
    if segment_to_body is None:
        segment_to_body = THEIA_TO_OSIM_BODY
    if not transforms_by_segment:
        raise ValueError("no transforms")

    n_frames = next(iter(transforms_by_segment.values())).shape[0]

    # Rx(-90°) world frame transform if we're going Z-up → Y-up.
    R_world: np.ndarray | None = None
    if osim_axis_swap:
        R_world = Rotation.from_euler("x", -90.0, degrees=True).as_matrix()  # (3, 3)

    out: dict[str, np.ndarray] = {}
    for seg, T in transforms_by_segment.items():
        body = segment_to_body.get(seg)
        if body is None:
            continue
        if body in out:
            # Skip duplicate target body (e.g. head and torso both map to torso).
            continue
        if T.shape != (n_frames, 4, 4):
            raise ValueError(f"{seg!r} has shape {T.shape}, expected ({n_frames}, 4, 4)")
        R_seg = T[:, :3, :3]  # (n_frames, 3, 3)
        if R_world is not None:
            # World rotation: R_seg_in_osim = R_world @ R_seg
            R_seg = np.einsum("ij,njk->nik", R_world, R_seg)
        quats_xyzw = Rotation.from_matrix(R_seg).as_quat()  # scipy returns (x,y,z,w)
        # Reorder to (w, x, y, z) which OpenSense .sto expects.
        out[body] = np.column_stack([quats_xyzw[:, 3], quats_xyzw[:, :3]])

    times = np.arange(n_frames, dtype=np.float64)
    return times, out


def write_orientations_sto(
    times_or_rate: np.ndarray | float,
    quats_by_body: dict[str, np.ndarray],
    path: Path | str,
    sample_rate_hz: float | None = None,
) -> Path:
    """Write an OpenSense-format Quaternion .sto.

    Args:
        times_or_rate: either a (n_frames,) time vector (seconds) or a frame-index
            array — if int-typed, we'll divide by `sample_rate_hz` to get seconds.
        quats_by_body: body name → (n_frames, 4) quaternions in (w, x, y, z) order.
        path: output .sto path.
        sample_rate_hz: required if `times_or_rate` is frame indices.

    Returns:
        Resolved path.
    """
    if not quats_by_body:
        raise ValueError("no quaternions to write")
    bodies = list(quats_by_body.keys())
    n_frames = next(iter(quats_by_body.values())).shape[0]
    for b in bodies:
        if quats_by_body[b].shape != (n_frames, 4):
            raise ValueError(f"body {b!r} shape {quats_by_body[b].shape} != ({n_frames}, 4)")

    times = np.asarray(times_or_rate, dtype=np.float64)
    if times.ndim == 0 or (np.issubdtype(times.dtype, np.integer)):
        if sample_rate_hz is None:
            raise ValueError("sample_rate_hz required when times_or_rate is frame indices")
        times = np.arange(n_frames, dtype=np.float64) / sample_rate_hz
    elif times.size == n_frames and times.max() > n_frames * 0.9:
        # Heuristic: looks like frame indices (max ≈ n_frames-1) → convert.
        if sample_rate_hz is None:
            raise ValueError("sample_rate_hz required to convert frame indices to time")
        times = times / sample_rate_hz

    if times.shape != (n_frames,):
        raise ValueError(f"times shape {times.shape} != ({n_frames},)")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("DataRate=" + (f"{sample_rate_hz}" if sample_rate_hz else "auto"))
    lines.append("DataType=Quaternion")
    lines.append("version=3")
    lines.append("OpenSimVersion=4.6")
    lines.append("endheader")
    header_row = ["time"] + bodies
    lines.append("\t".join(header_row))
    for f in range(n_frames):
        row = [f"{times[f]:.6f}"]
        for b in bodies:
            w, x, y, z = quats_by_body[b][f]
            row.append(f"{w:.8f},{x:.8f},{y:.8f},{z:.8f}")
        lines.append("\t".join(row))

    path.write_text("\n".join(lines) + "\n")
    return path


def write_recipe_c_sto(
    transforms_by_segment: dict[str, np.ndarray],
    out_path: Path | str,
    sample_rate_hz: float,
    *,
    osim_axis_swap: bool = True,
) -> Path:
    """End-to-end Recipe C: convert + write."""
    times, quats = transforms_to_quaternions(
        transforms_by_segment, osim_axis_swap=osim_axis_swap
    )
    return write_orientations_sto(times, quats, out_path, sample_rate_hz=sample_rate_hz)
