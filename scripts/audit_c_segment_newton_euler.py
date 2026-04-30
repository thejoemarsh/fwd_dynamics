"""Audit C: V3D-style segment Newton-Euler reactions, bypassing OpenSim's
JointReaction analysis entirely.

See docs/m2_kinetics_jr_amplifier.md for why JR's actuator-feedback path
amplifies the throwing-arm reactions 5-7× via the constraint solver
under an ill-conditioned 3-DOF acromial Jacobian. This audit replaces
that pipeline with direct segment Newton-Euler:

    F_proximal_on_segment = m·(a_COM - g) + F_distal
    M_proximal_on_segment = I·α + ω×(I·ω) + r×F terms

For the throwing arm chain (right side):
    forearm = ulna + hand combined (wrist welded)
    elbow JR     = forearm Newton-Euler with proximal_jc = elbow_r
    shoulder JR  = humerus Newton-Euler with proximal_jc = acromial_r,
                   distal_jc = elbow_r, distal_reaction = elbow JR

Output frames (V3D-comparable):
    ELBOW         in ulna_r body frame
    SHOULDER_AR   in humerus_r body frame
    SHOULDER_RTA  in torso body frame
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from theia_osim.analysis.segment_reactions import compute_throwing_arm_reactions  # noqa: E402

REPRO = Path(os.environ.get(
    "AUDIT_C_REPRO_DIR",
    str(REPO / "out/repro_step1c_16hz"),
))
PERSONAL_OSIM = REPRO / "all_recipes/theia_pitching_personalized.osim"
ANALYTICAL_MOT = REPRO / "all_recipes/recipe_d/analytical.mot"

OUT_DIR = REPO / "out/audit_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V3D = {
    "shoulder_F": 1090.0,    # V3D SHOULDER_AR_FORCE peak
    "shoulder_M":  151.0,    # V3D SHOULDER_AR_MMT peak
    "elbow_F":    1142.0,
    "elbow_M":     140.0,
}
# Reference: full-feedback JR (current pipeline) on the same input.
JR_FULL = {
    "shoulder_F": 7636.0,
    "shoulder_M": 1630.0,
    "elbow_F":    5622.0,
    "elbow_M":     830.0,
}


def peak_mag(arr3, t, t_lo, t_hi):
    """Peak |3D vector| within window — same convention as repro_shoulder_kinetics."""
    mask = (t >= t_lo) & (t <= t_hi)
    if not np.any(mask):
        return float("nan"), float("nan")
    sub = arr3[mask]
    mags = np.linalg.norm(sub, axis=1)
    if not np.all(np.isfinite(mags)):
        return float("nan"), float("nan")
    idx = int(np.argmax(mags))
    times_sub = t[mask]
    return float(mags[idx]), float(times_sub[idx])


print(f"Computing segment Newton-Euler reactions")
print(f"  model: {PERSONAL_OSIM}")
print(f"  mot:   {ANALYTICAL_MOT}")

result = compute_throwing_arm_reactions(PERSONAL_OSIM, ANALYTICAL_MOT, side="r")
times = result["times"]

# Same throw window as the rest of the audits.
br_t = 1.593
t_lo, t_hi = br_t - 50/300.0, br_t + 30/300.0

peaks = {}
for key in ("elbow_F_ulna_frame", "elbow_M_ulna_frame",
            "shoulder_F_humerus", "shoulder_M_humerus",
            "shoulder_F_torso", "shoulder_M_torso"):
    mag, when = peak_mag(result[key], times, t_lo, t_hi)
    peaks[key] = {"peak": mag, "t_s": when}

# Save the time series.
np.savez(OUT_DIR / "segment_reactions.npz",
         times=times, **{k: v for k, v in result.items() if k != "times"})

lines = ["", "Segment Newton-Euler vs JR (full feedback) vs V3D:", ""]
lines.append(f"  {'metric':<14}{'segNE':>10}{'JR_full':>10}{'V3D':>10}  {'segNE/V3D':>11}  {'JR/V3D':>10}")
rows = [
    ("shoulder_F", peaks["shoulder_F_humerus"]["peak"], JR_FULL["shoulder_F"], V3D["shoulder_F"]),
    ("shoulder_M", peaks["shoulder_M_humerus"]["peak"], JR_FULL["shoulder_M"], V3D["shoulder_M"]),
    ("elbow_F",    peaks["elbow_F_ulna_frame"]["peak"], JR_FULL["elbow_F"],    V3D["elbow_F"]),
    ("elbow_M",    peaks["elbow_M_ulna_frame"]["peak"], JR_FULL["elbow_M"],    V3D["elbow_M"]),
]
for name, ne, jr, v3d in rows:
    lines.append(f"  {name:<14}{ne:>10.1f}{jr:>10.1f}{v3d:>10.1f}  {ne/v3d:>10.2f}x  {jr/v3d:>9.2f}x")

# Also report shoulder_RTA (torso frame) for completeness.
lines.append("")
lines.append(f"  V3D-style SHOULDER_RTA (torso frame): "
             f"F_peak = {peaks['shoulder_F_torso']['peak']:.1f} N "
             f"at t={peaks['shoulder_F_torso']['t_s']:.3f}s   (V3D 1178 N at BR)")
lines.append(f"  V3D-style SHOULDER_RTA moment        : "
             f"M_peak = {peaks['shoulder_M_torso']['peak']:.1f} N·m "
             f"at t={peaks['shoulder_M_torso']['t_s']:.3f}s   (V3D 137 N·m at BR)")
out = "\n".join(lines)
print(out)

(REPO / "out/audit_c_summary.txt").write_text(out + "\n")
(OUT_DIR / "peaks.json").write_text(json.dumps(peaks, indent=2))
print(f"\nWrote {OUT_DIR}/peaks.json and {REPO}/out/audit_c_summary.txt")
