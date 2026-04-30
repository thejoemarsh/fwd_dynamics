"""Audit B: test whether translational CoordinateActuators (pelvis_tx/ty/tz)
are the source of the actuator-feedback F amplification in JR.

H3 showed that with feedback OFF, elbow F is V3D-clean (0.97×) but with
feedback ON, elbow F is 5-7× V3D. The actuator force-set replaces muscles
with one CoordinateActuator per coord — including 3 TRANSLATIONAL actuators
on the pelvis FreeJoint. Those apply real F (not torque), and propagate up
the kinematic chain to elbow/shoulder, inflating F at every joint.

Experiment: build a force-set XML with rotational coords ONLY, rerun JR,
compare elbow/shoulder F+M peaks against full-feedback and no-feedback.

Outputs:
  out/audit_b_rot_only/joint_reaction_*.sto
  out/audit_b_summary.txt
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import opensim as osim

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from theia_osim.analysis.jr import run_joint_reaction  # noqa: E402

REPRO = Path(os.environ.get(
    "AUDIT_B_REPRO_DIR",
    str(REPO / "out/repro_step1c_16hz"),
))
PERSONAL_OSIM = REPRO / "all_recipes/theia_pitching_personalized.osim"
ANALYTICAL_MOT = REPRO / "all_recipes/recipe_d/analytical.mot"
ID_STO = REPRO / "kinetics/inverse_dynamics.sto"

OUT_DIR = REPO / "out/audit_b_rot_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_rotational_only_force_set(model_path: Path, out_xml: Path) -> Path:
    """Like residual_actuators.build_coord_actuator_force_set but skips
    translational coords. Keeps rotational coords with `_moment` suffix."""
    model = osim.Model(str(model_path))
    ROT = 1
    fs = osim.ForceSet()
    cs = model.getCoordinateSet()
    skipped = []
    kept = []
    for i in range(cs.getSize()):
        c = cs.get(i)
        mt = int(c.getMotionType())
        if mt != ROT:
            skipped.append(c.getName())
            continue
        a = osim.CoordinateActuator(c.getName())
        a.setName(c.getName() + "_moment")
        a.setOptimalForce(1.0)
        a.setMinControl(-1.0e9)
        a.setMaxControl(1.0e9)
        fs.cloneAndAppend(a)
        kept.append(c.getName())
    fs.printToXML(str(out_xml))
    print(f"  rotational-only XML: {out_xml}")
    print(f"  kept ({len(kept)} rot): {kept}")
    print(f"  skipped ({len(skipped)} non-rot): {skipped}")
    return out_xml


def read_sto(path):
    with open(path) as f:
        for line in f:
            if line.strip().lower() == "endheader":
                break
        hdr = next(f).strip().split("\t")
        rows = [list(map(float, line.split())) for line in f if line.strip()]
    return hdr, np.array(rows)


def vec_peak(arr, hdr, prefix, kind, t_lo, t_hi):
    keys = [f"{prefix}_{kind}{a}" for a in "xyz"]
    if not all(k in hdr for k in keys):
        return float("nan")
    t = arr[:, 0]
    mask = (t >= t_lo) & (t <= t_hi)
    comps = []
    for k in keys:
        v = arr[mask, hdr.index(k)]
        if v.size == 0:
            return float("nan")
        comps.append(float(np.abs(v).max()))
    if not all(np.isfinite(c) and abs(c) < 1e6 for c in comps):
        return float("nan")
    return float(np.sqrt(sum(c ** 2 for c in comps)))


# Build the rotational-only XML
rot_xml = build_rotational_only_force_set(PERSONAL_OSIM, OUT_DIR / "rot_actuators.xml")

# Run JR with rotational-only feedback
print("\n=> Running JR with rotational-only actuator feedback")
jr_sto = run_joint_reaction(
    PERSONAL_OSIM, ANALYTICAL_MOT,
    id_sto=ID_STO,
    out_dir=OUT_DIR,
    actuator_force_set_xml=rot_xml,
)
hdr, arr = read_sto(jr_sto)
br_t = 1.593
t_lo, t_hi = br_t - 50/300.0, br_t + 30/300.0

results = {
    "shoulder_F": vec_peak(arr, hdr, "acromial_r_on_humerus_r_in_humerus_r", "f", t_lo, t_hi),
    "shoulder_M": vec_peak(arr, hdr, "acromial_r_on_humerus_r_in_humerus_r", "m", t_lo, t_hi),
    "elbow_F":    vec_peak(arr, hdr, "elbow_r_on_ulna_r_in_ulna_r", "f", t_lo, t_hi),
    "elbow_M":    vec_peak(arr, hdr, "elbow_r_on_ulna_r_in_ulna_r", "m", t_lo, t_hi),
}

# 16Hz full-feedback (control). Re-derive from out/repro_step1c_16hz/peaks.json if present.
import json
ctrl = {}
peaks_p = REPRO / "peaks.json"
if peaks_p.exists():
    p = json.loads(peaks_p.read_text())
    m = p.get("magnitudes", {})
    ctrl = {
        "shoulder_F": m.get("shoulder_humerus_F_N", float("nan")),
        "shoulder_M": m.get("shoulder_humerus_M_Nm", float("nan")),
        "elbow_F":    m.get("elbow_F_N", float("nan")),
        "elbow_M":    m.get("elbow_M_Nm", float("nan")),
    }
V3D = {"shoulder_F": 1090., "shoulder_M": 151., "elbow_F": 1142., "elbow_M": 140.}

lines = ["", "JR rotational-actuators-only vs full-feedback vs V3D:", ""]
lines.append(f"  {'metric':<14}{'rot_only':>10}{'full_fb':>10}{'v3d':>10}  {'rot/v3d':>10}  {'full/v3d':>10}")
for k in ("shoulder_F", "shoulder_M", "elbow_F", "elbow_M"):
    r = results[k]; c = ctrl.get(k, float("nan")); g = V3D[k]
    lines.append(f"  {k:<14}{r:>10.1f}{c:>10.1f}{g:>10.1f}  {r/g:>9.2f}x  {c/g:>9.2f}x")
out = "\n".join(lines)
print(out)
(REPO / "out/audit_b_summary.txt").write_text(out + "\n")
