"""H3 audit: rerun JR with actuator_force_set_xml=None to see whether the
residual-actuator feedback loop is amplifying the reaction magnitudes.

Per docs/m2_kinetics_handoff.md H3:
  Reactions will be physically incomplete (don't include the active joint
  torques) but the magnitudes will be different. If they collapse toward
  V3D-typical values, the cancellation pipeline is the amplifier.

Outputs:
  out/audit_h3/joint_reaction_*.sto       (the no-actuator JR result)
  out/audit_h3_summary.txt                (peaks vs control + V3D)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from theia_osim.analysis.jr import run_joint_reaction  # noqa: E402

REPRO = Path(os.environ.get(
    "AUDIT_H3_REPRO_DIR",
    str(REPO / "../fwd_dynamics-laiuhlrich/out/ralph_laiuhlrich/iter_01"),
))
PERSONAL_OSIM = REPRO / "all_recipes/theia_pitching_personalized.osim"
ANALYTICAL_MOT = REPRO / "all_recipes/recipe_d/analytical.mot"
ID_STO = REPRO / "kinetics/inverse_dynamics.sto"

OUT_DIR = REPO / "out/audit_h3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# V3D ground truth (from handoff doc, pose_filt_0)
V3D = {
    "shoulder_F": 1090.0,
    "shoulder_M":  151.0,
    "elbow_F":    1142.0,
    "elbow_M":     140.0,
}

# Control (with actuator feedback) from latest laiuhlrich Ralph run.
CONTROL = {
    "shoulder_F": 8080.5,
    "shoulder_M": 1793.3,
    "elbow_F":    6326.2,
    "elbow_M":     892.8,
}


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


print(f"Running JR with actuator_force_set_xml=None")
print(f"  personal: {PERSONAL_OSIM}")
print(f"  mot:      {ANALYTICAL_MOT}")

jr_sto = run_joint_reaction(
    PERSONAL_OSIM, ANALYTICAL_MOT,
    id_sto=ID_STO,
    out_dir=OUT_DIR,
    actuator_force_set_xml=None,
)
print(f"\nJR (no feedback) sto: {jr_sto}")

hdr, arr = read_sto(jr_sto)
# BR window from handoff
br_t = 1.593
t_lo, t_hi = br_t - 50/300.0, br_t + 30/300.0

results = {
    "shoulder_F": vec_peak(arr, hdr, "acromial_r_on_humerus_r_in_humerus_r", "f", t_lo, t_hi),
    "shoulder_M": vec_peak(arr, hdr, "acromial_r_on_humerus_r_in_humerus_r", "m", t_lo, t_hi),
    "elbow_F":    vec_peak(arr, hdr, "elbow_r_on_ulna_r_in_ulna_r", "f", t_lo, t_hi),
    "elbow_M":    vec_peak(arr, hdr, "elbow_r_on_ulna_r_in_ulna_r", "m", t_lo, t_hi),
}

lines = ["", "JR no-actuator-feedback peaks vs control vs V3D:", ""]
lines.append(f"  {'metric':<14}{'no_fb':>10}{'control':>10}{'v3d':>10}  {'no_fb/v3d':>10}  {'ctrl/v3d':>10}")
for k, v in results.items():
    c = CONTROL[k]
    g = V3D[k]
    lines.append(f"  {k:<14}{v:>10.1f}{c:>10.1f}{g:>10.1f}  {v/g:>9.2f}x  {c/g:>9.2f}x")
out = "\n".join(lines)
print(out)
(REPO / "out/audit_h3_summary.txt").write_text(out + "\n")
