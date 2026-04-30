"""Audit B2: which subset of CoordinateActuators is responsible for the JR
F amplification? Test three force-set configurations and compare against
no-feedback (H3) and full-feedback baselines.

  config A: arm-only      (arm_flex_r, arm_add_r, arm_rot_r, elbow_flex_r, pro_sup_r)
  config B: not-arm       (everything except the right throwing arm coords)
  config C: rot-arm-only  (right throwing arm + lumbar — minimum chain to torso)

If config A reproduces the amplification, the arm's own actuator feedback is
the cause (kinematic Jacobian issue at gimbal lock). If config B does, the
amplification comes from elsewhere in the body feeding through to the arm.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import opensim as osim

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from theia_osim.analysis.jr import run_joint_reaction  # noqa: E402

REPRO = Path(REPO / "out/repro_step1c_16hz")
PERSONAL_OSIM = REPRO / "all_recipes/theia_pitching_personalized.osim"
ANALYTICAL_MOT = REPRO / "all_recipes/recipe_d/analytical.mot"
ID_STO = REPRO / "kinetics/inverse_dynamics.sto"

ARM_R = {"arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r"}


def build_force_set(model_path: Path, out_xml: Path, include) -> Path:
    model = osim.Model(str(model_path))
    fs = osim.ForceSet()
    cs = model.getCoordinateSet()
    kept = []
    for i in range(cs.getSize()):
        c = cs.get(i)
        mt = int(c.getMotionType())
        if mt == 1: suffix = "_moment"
        elif mt == 2: suffix = "_force"
        else: continue
        if not include(c.getName(), mt):
            continue
        a = osim.CoordinateActuator(c.getName())
        a.setName(c.getName() + suffix)
        a.setOptimalForce(1.0)
        a.setMinControl(-1e9); a.setMaxControl(1e9)
        fs.cloneAndAppend(a)
        kept.append(c.getName())
    fs.printToXML(str(out_xml))
    return out_xml, kept


def read_sto(path):
    with open(path) as f:
        for line in f:
            if line.strip().lower() == "endheader": break
        hdr = next(f).strip().split("\t")
        rows = [list(map(float, l.split())) for l in f if l.strip()]
    return hdr, np.array(rows)


def vec_peak(arr, hdr, prefix, kind, t_lo, t_hi):
    keys = [f"{prefix}_{kind}{a}" for a in "xyz"]
    if not all(k in hdr for k in keys): return float("nan")
    t = arr[:, 0]; mask = (t >= t_lo) & (t <= t_hi)
    cs = []
    for k in keys:
        v = arr[mask, hdr.index(k)]
        if v.size == 0: return float("nan")
        cs.append(float(np.abs(v).max()))
    if not all(np.isfinite(c) and abs(c) < 1e6 for c in cs): return float("nan")
    return float(np.sqrt(sum(c**2 for c in cs)))


configs = {
    "A_arm_only":     lambda n, mt: n in ARM_R,
    "B_not_arm":      lambda n, mt: n not in ARM_R,
    "C_arm_plus_lumbar": lambda n, mt: n in ARM_R or n.startswith("lumbar"),
}

V3D = {"shoulder_F": 1090., "shoulder_M": 151., "elbow_F": 1142., "elbow_M": 140.}

rows = []
for cfg_name, pred in configs.items():
    out_dir = REPO / f"out/audit_b2_{cfg_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    xml, kept = build_force_set(PERSONAL_OSIM, out_dir / "actuators.xml", pred)
    print(f"\n=== {cfg_name}: {len(kept)} actuators kept ===")
    print(f"  {kept}")
    jr_sto = run_joint_reaction(
        PERSONAL_OSIM, ANALYTICAL_MOT, id_sto=ID_STO,
        out_dir=out_dir, actuator_force_set_xml=xml,
    )
    hdr, arr = read_sto(jr_sto)
    br_t = 1.593; t_lo, t_hi = br_t - 50/300., br_t + 30/300.
    res = {
        "shoulder_F": vec_peak(arr, hdr, "acromial_r_on_humerus_r_in_humerus_r", "f", t_lo, t_hi),
        "shoulder_M": vec_peak(arr, hdr, "acromial_r_on_humerus_r_in_humerus_r", "m", t_lo, t_hi),
        "elbow_F":    vec_peak(arr, hdr, "elbow_r_on_ulna_r_in_ulna_r", "f", t_lo, t_hi),
        "elbow_M":    vec_peak(arr, hdr, "elbow_r_on_ulna_r_in_ulna_r", "m", t_lo, t_hi),
    }
    rows.append((cfg_name, res))

# Baselines from prior audits
rows.insert(0, ("no_feedback", {"shoulder_F": 2360., "shoulder_M": 0.0, "elbow_F": 1111., "elbow_M": 29.}))
rows.append(("full_feedback", {"shoulder_F": 7636., "shoulder_M": 1630., "elbow_F": 5622., "elbow_M": 830.}))

print(f"\n{'config':<22}{'sF':>10}{'sM':>10}{'eF':>10}{'eM':>10}{'eF/v3d':>10}")
print("-" * 80)
for name, r in rows:
    line = f"{name:<22}{r['shoulder_F']:>10.0f}{r['shoulder_M']:>10.0f}{r['elbow_F']:>10.0f}{r['elbow_M']:>10.0f}{r['elbow_F']/V3D['elbow_F']:>9.2f}x"
    print(line)
print(f"{'V3D':<22}{V3D['shoulder_F']:>10.0f}{V3D['shoulder_M']:>10.0f}{V3D['elbow_F']:>10.0f}{V3D['elbow_M']:>10.0f}     1.00x")
