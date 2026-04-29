"""Build a residual CoordinateActuator ForceSet for ID/JR pipelines.

For V3D-parity inverse dynamics + joint reactions we want the model's
muscles out of the picture: V3D doesn't use them, and OpenSim's JR
can't reconstruct 80 redundant muscle forces from a 35-coordinate ID
output. The standard fix is to replace the model's ForceSet with one
CoordinateActuator per coordinate. After this:

  * `InverseDynamicsTool` still computes Newton-Euler GenForces per
    coordinate (one scalar each).
  * `JointReaction` reads those scalars and applies them as per-joint
    coordinate-axis torques during the analysis, producing the full
    3D reaction force + moment in the requested frame.

Each actuator is named **literally `<coord>_moment` (rotational) or
`<coord>_force` (translational)** so its name matches the column
header that `InverseDynamicsTool` writes — no .sto column renaming
needed downstream.

Coupled coordinates (motion_type=3, e.g. walker_knee `_beta`) are
skipped because ID doesn't write a GenForce column for them.
"""
from __future__ import annotations

from pathlib import Path

import opensim as osim


def build_coord_actuator_force_set(
    model_path: Path | str,
    out_xml: Path | str,
    *,
    optimal_force: float = 1.0,
) -> Path:
    """Write a ForceSet XML with one CoordinateActuator per ID-emitted coordinate.

    Args:
        model_path: source .osim. We read its CoordinateSet only.
        out_xml: destination ForceSet XML path.
        optimal_force: per-actuator optimal force. 1.0 makes control == force.

    Returns:
        Path to written XML.
    """
    out_xml = Path(out_xml).resolve()
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    model = osim.Model(str(Path(model_path).resolve()))

    # Motion-type enum integer values in OpenSim: 1=Rotational, 2=Translational,
    # 3=Coupled. The Python binding doesn't expose the enum names directly.
    ROT = 1
    TRA = 2
    fs = osim.ForceSet()
    cs = model.getCoordinateSet()
    for i in range(cs.getSize()):
        c = cs.get(i)
        mt = int(c.getMotionType())
        if mt == ROT:
            suffix = "_moment"
        elif mt == TRA:
            suffix = "_force"
        else:
            continue  # Coupled — ID skips it
        a = osim.CoordinateActuator(c.getName())
        a.setName(c.getName() + suffix)
        a.setOptimalForce(float(optimal_force))
        a.setMinControl(-1.0e9)
        a.setMaxControl(1.0e9)
        fs.cloneAndAppend(a)

    fs.printToXML(str(out_xml))
    return out_xml
