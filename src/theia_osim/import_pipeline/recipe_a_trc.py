"""Recipe A: synthesize virtual markers from segment 4×4s, write TRC for IK.

Theia's segment-frame origin is the proximal joint center; we use that as one
free marker per segment, then add 2+ more constructed-from-segment-frame markers
to constrain rotation. Output TRC is in OpenSim's expected format with units=m.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .landmarks import SegmentMarkers, synthesize_all_markers


def write_trc(
    markers: dict[str, np.ndarray],
    path: Path | str,
    sample_rate_hz: float,
    units: str = "m",
) -> Path:
    """Write a TRC file consumable by OpenSim's IK tool.

    Args:
        markers: dict mapping marker name → (T, 3) world positions.
        path: output .trc path.
        sample_rate_hz: e.g. 300.
        units: "m" or "mm". OpenSim is happy with either; we use m to match
            the rest of the pipeline.

    Returns:
        Resolved output path.
    """
    if not markers:
        raise ValueError("no markers to write")
    names = list(markers.keys())
    n_frames = next(iter(markers.values())).shape[0]
    for n in names:
        if markers[n].shape != (n_frames, 3):
            raise ValueError(
                f"marker {n!r} shape {markers[n].shape} != ({n_frames}, 3)"
            )
    n_markers = len(names)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    times = np.arange(n_frames, dtype=np.float64) / sample_rate_hz

    lines: list[str] = []
    # OpenSim TRC header (tab-separated).
    lines.append(f"PathFileType\t4\t(X/Y/Z)\t{path.name}")
    lines.append(
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\t"
        "OrigDataStartFrame\tOrigNumFrames"
    )
    lines.append(
        f"{sample_rate_hz}\t{sample_rate_hz}\t{n_frames}\t{n_markers}\t{units}\t"
        f"{sample_rate_hz}\t1\t{n_frames}"
    )
    # Marker name row: Frame#, Time, then each name (followed by 2 blanks for Y, Z columns).
    name_row = ["Frame#", "Time"]
    for n in names:
        name_row.extend([n, "", ""])
    lines.append("\t".join(name_row))
    # Subheader: blank, blank, then X1 Y1 Z1 X2 Y2 Z2 ...
    sub = ["", ""]
    for i in range(1, n_markers + 1):
        sub.extend([f"X{i}", f"Y{i}", f"Z{i}"])
    lines.append("\t".join(sub))
    lines.append("")  # blank line before data

    for f in range(n_frames):
        row: list[str] = [str(f + 1), f"{times[f]:.6f}"]
        for n in names:
            x, y, z = markers[n][f]
            row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        lines.append("\t".join(row))

    path.write_text("\n".join(lines) + "\n")
    return path


def write_recipe_a_trc(
    transforms_by_segment: dict[str, np.ndarray],
    catalog: dict[str, SegmentMarkers],
    out_path: Path | str,
    sample_rate_hz: float,
    *,
    osim_axis_swap: bool = True,
) -> Path:
    """End-to-end Recipe A: synthesize markers + write TRC.

    Args:
        transforms_by_segment: from TrialData.transforms (already slope-corrected).
        catalog: from landmarks.load_marker_catalog().
        out_path: output .trc path.
        sample_rate_hz: trial sample rate.
        osim_axis_swap: if True, rotate Theia (+X pitching, +Y throwing side, +Z up)
            into OpenSim's default Y-up convention via Rx(-90°). Default True.

    Returns:
        Path to the written .trc.
    """
    markers = synthesize_all_markers(transforms_by_segment, catalog)
    if osim_axis_swap:
        # Rx(-90°): (x, y, z) → (x, z, -y)  (Theia Z-up → OpenSim Y-up)
        for name, p in markers.items():
            x, y, z = p[:, 0], p[:, 1], p[:, 2]
            markers[name] = np.column_stack([x, z, -y])
    return write_trc(markers, out_path, sample_rate_hz, units="m")
