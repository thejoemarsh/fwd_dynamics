"""BodyKinematics analysis — segment angular velocity in segment frame.

Wraps OpenSim's AnalyzeTool + BodyKinematics to compute per-frame body
velocities expressed in the local body frame (which is what V3D computes
for `*_ANGULAR_VELOCITY` resolved-in-segment signals).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import opensim as osim
import pandas as pd


def run_body_kinematics(
    model_path: Path | str,
    coords_mot: Path | str,
    out_dir: Path | str,
    *,
    bodies: tuple[str, ...] | None = None,
    name: str = "body_kinematics",
) -> dict[str, Path]:
    """Run BodyKinematics analysis with `express_in_local_frame=true`.

    Args:
        model_path: .osim used to compute kinematics.
        coords_mot: .mot from IK.
        out_dir: directory where AnalyzeTool writes its outputs.
        bodies: subset of body names to analyze (None = all).
        name: AnalyzeTool name (used as output file prefix).

    Returns:
        Dict of output kind → file path. Keys: 'pos', 'vel', 'acc'.
    """
    model_path = Path(model_path).resolve()
    coords_mot = Path(coords_mot).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = osim.Model(str(model_path))
    model.initSystem()

    # Time range from .mot.
    storage = osim.Storage(str(coords_mot))
    t_start = storage.getFirstTime()
    t_end = storage.getLastTime()

    bk = osim.BodyKinematics()
    bk.setName("BodyKinematics")
    bk.setStartTime(t_start)
    bk.setEndTime(t_end)
    bk.setOn(True)
    bk.setExpressResultsInLocalFrame(True)
    if bodies is not None:
        body_set = osim.ArrayStr()
        for b in bodies:
            body_set.append(b)
        bk.setBodiesToRecord(body_set)

    tool = osim.AnalyzeTool()
    tool.setName(name)
    tool.setModel(model)
    tool.setModelFilename(str(model_path))
    tool.setCoordinatesFileName(str(coords_mot))
    tool.setLowpassCutoffFrequency(-1.0)  # we own filtering downstream
    tool.setStartTime(t_start)
    tool.setFinalTime(t_end)
    tool.setResultsDir(str(out_dir))
    tool.getAnalysisSet().cloneAndAppend(bk)
    setup_xml = out_dir / f"{name}_setup.xml"
    tool.printToXML(str(setup_xml))

    # Re-load via setup XML to ensure consistent state for run().
    rerun_tool = osim.AnalyzeTool(str(setup_xml))
    if not rerun_tool.run():
        raise RuntimeError(f"AnalyzeTool/BodyKinematics failed; setup at {setup_xml}")

    # When express_in_local_frame=true, vel/acc get '_bodyLocal' suffix; pos stays global.
    return {
        "pos": out_dir / f"{name}_BodyKinematics_pos_global.sto",
        "vel": out_dir / f"{name}_BodyKinematics_vel_bodyLocal.sto",
        "acc": out_dir / f"{name}_BodyKinematics_acc_bodyLocal.sto",
    }


def read_body_velocities(sto_path: Path | str, body: str) -> pd.DataFrame:
    """Read a BodyKinematics velocity .sto and pick out columns for `body`.

    The .sto columns are like `<body>_X`, `<body>_Y`, `<body>_Z` for translational
    velocity then `<body>_Ox`, `<body>_Oy`, `<body>_Oz` for angular velocity in the
    body's local frame (when express_in_local_frame=true).

    Returns:
        DataFrame with columns: time, vx, vy, vz, omega_x, omega_y, omega_z.
    """
    sto_path = Path(sto_path)
    df = _read_sto(sto_path)
    needed = [
        f"{body}_X",
        f"{body}_Y",
        f"{body}_Z",
        f"{body}_Ox",
        f"{body}_Oy",
        f"{body}_Oz",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"sto {sto_path} missing columns {missing}. Available: {list(df.columns)[:30]}..."
        )
    return df[["time"] + needed].rename(
        columns={
            f"{body}_X": "vx",
            f"{body}_Y": "vy",
            f"{body}_Z": "vz",
            f"{body}_Ox": "omega_x",
            f"{body}_Oy": "omega_y",
            f"{body}_Oz": "omega_z",
        }
    )


def _read_sto(path: Path) -> pd.DataFrame:
    """Read an OpenSim .sto into a DataFrame.

    Handles both tab-separated and whitespace-separated formats.
    """
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            data_start = i + 1
            break
    else:
        raise ValueError(f"{path} has no 'endheader' line")
    header = lines[data_start].split()
    data = []
    for line in lines[data_start + 1 :]:
        if not line.strip():
            continue
        data.append([float(x) for x in line.split()])
    arr = np.asarray(data, dtype=np.float64)
    return pd.DataFrame(arr, columns=header)
