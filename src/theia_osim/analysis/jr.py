"""JointReaction driver — wraps OpenSim's JointReaction analysis.

Computes inter-segment reaction force + moment at each joint, expressed
in a chosen frame. JR runs as an Analysis inside `AnalyzeTool`, so it
needs the same .mot of coordinates that ID consumed.

V3D shoulder kinetics resolve in two frames per side:
- `SHOULDER_AR_*`  — reaction on humerus expressed in **humerus** frame
- `SHOULDER_RTA_*` — reaction on humerus expressed in **torso** frame
- `ELBOW_*`        — reaction on ulna expressed in **ulna** (forearm) frame

Per side: 3 entries (2 shoulder × 1 elbow). Glove-side mirrors are added by
specifying a list of joint specs covering both `_r` and `_l`.

OpenSim's JR analysis uses parallel arrays internally:
- joint_names:  ["acromial_r", "acromial_r", "elbow_r", ...]
- on_bodies:    ["child",      "child",      "child",   ...]
- in_frames:    ["child",      "parent",     "child",   ...]

Columns in the output .sto are suffixed with the resolving frame name, so
running the same joint twice with different `in_frame` produces distinct
columns (`..._on_humerus_r_in_humerus_r_fx` vs `..._on_humerus_r_in_torso_fx`).

Pass `actuator_force_set_xml` (built by `analysis.residual_actuators`) to
replace the model's muscle ForceSet with a residual CoordinateActuator set.
JR then reads ID's per-coordinate generalized forces directly from the
provided `.sto`, producing the full 3D reaction force + moment vector
(rather than only the constraint-axis components).

KNOWN LIMITATION — shoulder Cardan singularity:
LaiUhlrich2022's `acromial_*` is a 3-DOF CustomJoint with intrinsic Cardan
Z-X-Y axes. During the throwing arm's late cocking → release window,
arm_add_r passes through the gimbal-lock region (X ≈ ±90°). Even with
smart Cardan unwrap, the joint Jacobian is ill-conditioned there, which
inflates both InverseDynamics generalized forces and JointReaction outputs
for `acromial_r` and downstream `elbow_r`. Pelvis / hip / lumbar / leg
kinetics are unaffected. A follow-up using a different shoulder joint
representation (e.g. quaternion-based or alternate Cardan order) is
needed to validate `SHOULDER_*` and `ELBOW_*` against V3D within ±10%.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import opensim as osim


@dataclass(frozen=True)
class JRSpec:
    """One JointReaction probe: joint × resolved-on-body × resolved-in-frame."""

    joint: str
    on_body: str  # 'parent' | 'child'
    in_frame: str  # 'parent' | 'child' | 'ground'

    def label(self) -> str:
        return f"{self.joint}_on_{self.on_body}_in_{self.in_frame}"


# Default per-side specs covering V3D's marquee kinetic signals.
# Shoulder: two entries (humerus-frame + torso-frame).
# Elbow:    one entry (forearm-frame).
def default_specs(sides: tuple[str, ...] = ("r", "l")) -> tuple[JRSpec, ...]:
    out: list[JRSpec] = []
    for s in sides:
        out.extend(
            [
                JRSpec(f"acromial_{s}", "child", "child"),  # SHOULDER_AR
                JRSpec(f"acromial_{s}", "child", "parent"),  # SHOULDER_RTA
                JRSpec(f"elbow_{s}", "child", "child"),  # ELBOW
            ]
        )
    return tuple(out)


def run_joint_reaction(
    model_path: Path | str,
    coords_mot: Path | str,
    id_sto: Path | str,
    out_dir: Path | str,
    *,
    specs: tuple[JRSpec, ...] = default_specs(),
    name: str = "joint_reaction",
    actuator_force_set_xml: Path | str | None = None,
) -> Path:
    """Run JointReaction via AnalyzeTool. Returns the .sto path written.

    Args:
        model_path: model used (must be the same one used for ID).
        coords_mot: .mot of joint coordinates.
        id_sto: .sto generalized forces from InverseDynamicsTool — fed into
            JR via the model's Forces directly. JR uses the model's internal
            forces during forward integration; passing this enables ID
            consistency. We pass the path so it appears in the setup XML
            (informational only — JR reads forces from the model).
        out_dir: directory for the JR setup XML and result .sto.
        specs: JR probes to run.
        name: tool name (output filename prefix).
    """
    model_path = Path(model_path).resolve()
    coords_mot = Path(coords_mot).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    storage = osim.Storage(str(coords_mot))
    t_start = storage.getFirstTime()
    t_end = storage.getLastTime()

    model = osim.Model(str(model_path))
    model.initSystem()

    jr = osim.JointReaction()
    jr.setName("JointReaction")
    jr.setStartTime(t_start)
    jr.setEndTime(t_end)
    jr.setOn(True)

    joints = osim.ArrayStr()
    bodies = osim.ArrayStr()
    frames = osim.ArrayStr()
    for sp in specs:
        joints.append(sp.joint)
        bodies.append(sp.on_body)
        frames.append(sp.in_frame)
    jr.setJointNames(joints)
    jr.setOnBody(bodies)
    jr.setInFrame(frames)

    if id_sto is not None:
        jr.setForcesFileName(str(Path(id_sto).resolve()))

    tool = osim.AnalyzeTool()
    tool.setName(name)
    tool.setModel(model)
    tool.setModelFilename(str(model_path))
    tool.setCoordinatesFileName(str(coords_mot))
    tool.setLowpassCutoffFrequency(-1.0)
    tool.setStartTime(t_start)
    tool.setFinalTime(t_end)
    tool.setResultsDir(str(out_dir))
    tool.getAnalysisSet().cloneAndAppend(jr)
    if actuator_force_set_xml is not None:
        tool.setReplaceForceSet(True)
        fsf = osim.ArrayStr()
        fsf.append(str(Path(actuator_force_set_xml).resolve()))
        tool.setForceSetFiles(fsf)
    setup_xml = out_dir / f"{name}_setup.xml"
    tool.printToXML(str(setup_xml))

    rerun = osim.AnalyzeTool(str(setup_xml))
    if not rerun.run():
        raise RuntimeError(f"AnalyzeTool/JointReaction failed; setup at {setup_xml}")

    out_sto = out_dir / f"{name}_JointReaction_ReactionLoads.sto"
    if not out_sto.exists():
        raise RuntimeError(f"JointReaction ran but {out_sto} missing")
    return out_sto
