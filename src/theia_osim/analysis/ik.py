"""Inverse kinematics drivers — marker (Recipe A) and orientation (Recipe C)."""
from __future__ import annotations

from pathlib import Path

import opensim as osim


def run_marker_ik(
    model_path: Path | str,
    trc_path: Path | str,
    out_mot: Path | str,
    *,
    marker_weight: float = 1.0,
    accuracy: float = 1e-5,
) -> Path:
    """Run OpenSim's InverseKinematicsTool against a marker TRC.

    Args:
        model_path: path to .osim with matching <Marker> tags.
        trc_path: path to TRC produced by recipe_a_trc.
        out_mot: output coordinates .mot.
        marker_weight: per-marker tracking weight (uniform).
        accuracy: solver accuracy.

    Returns:
        Resolved out_mot path.
    """
    model_path = Path(model_path)
    trc_path = Path(trc_path)
    out_mot = Path(out_mot)
    out_mot.parent.mkdir(parents=True, exist_ok=True)

    model = osim.Model(str(model_path))
    model.initSystem()

    ik = osim.InverseKinematicsTool()
    ik.setName(out_mot.stem)
    ik.set_model_file(str(model_path))
    ik.setMarkerDataFileName(str(trc_path))
    ik.setOutputMotionFileName(str(out_mot))
    ik.set_accuracy(accuracy)

    # Build IKTaskSet weighting every marker on the model uniformly.
    task_set = osim.IKTaskSet()
    for i in range(model.getMarkerSet().getSize()):
        marker_name = model.getMarkerSet().get(i).getName()
        task = osim.IKMarkerTask()
        task.setName(marker_name)
        task.setApply(True)
        task.setWeight(marker_weight)
        task_set.cloneAndAppend(task)
    ik.set_IKTaskSet(task_set)

    # Time range from TRC. Read first/last time from the file header.
    times = _read_trc_time_range(trc_path)
    ik.setStartTime(times[0])
    ik.setEndTime(times[1])

    ok = ik.run()
    if not ok:
        raise RuntimeError(f"IK failed for {trc_path}")
    return out_mot


def run_imu_ik(
    model_path: Path | str,
    sto_path: Path | str,
    out_mot: Path | str,
    *,
    sensor_to_opensim_rotations: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Path:
    """Run OpenSense IMU IK against a quaternion .sto.

    Args:
        model_path: .osim with bodies named to match the .sto column headers.
        sto_path: orientation .sto from recipe_c_sto.
        out_mot: output coordinates .mot.
        sensor_to_opensim_rotations: degrees, applied as a rotation from sensor frame
            to OpenSim model frame. We already did Rx(-90°) on the data side, so this
            should be (0, 0, 0).

    Returns:
        Resolved out_mot path.
    """
    model_path = Path(model_path)
    sto_path = Path(sto_path)
    out_mot = Path(out_mot)
    out_mot.parent.mkdir(parents=True, exist_ok=True)

    tool = osim.IMUInverseKinematicsTool()
    tool.set_model_file(str(model_path))
    tool.set_orientations_file(str(sto_path))
    rot = osim.Vec3(*sensor_to_opensim_rotations)
    tool.set_sensor_to_opensim_rotations(rot)
    tool.set_results_directory(str(out_mot.parent))

    # IMUInverseKinematicsTool writes its own .sto; we won't get a .mot
    # directly, but we can read its output and convert if needed in M2.
    # For now, just run and return the expected output name.
    ok = tool.run()
    if not ok:
        raise RuntimeError(f"IMU IK failed for {sto_path}")
    # Tool writes <results_dir>/ik_<basename>.mot
    expected = out_mot.parent / f"ik_{sto_path.stem}.mot"
    return expected if expected.exists() else out_mot


def _read_trc_time_range(trc_path: Path) -> tuple[float, float]:
    """Pull (first_time, last_time) from a TRC.

    TRC layout: 5 header rows (PathFileType, DataRate header, DataRate data,
    marker-name row, subheader row), 1 blank line, then data rows.
    """
    first_time: float | None = None
    last_time: float | None = None
    with open(trc_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) < 2:
                continue
            # Skip header rows: a data row's first column is an integer frame number.
            try:
                int(parts[0])
            except ValueError:
                continue
            t = float(parts[1])
            if first_time is None:
                first_time = t
            last_time = t
    if first_time is None or last_time is None:
        raise ValueError(f"no data rows found in {trc_path}")
    return first_time, last_time
