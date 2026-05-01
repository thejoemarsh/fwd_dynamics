"""Audit I: smoothing-method sweep. Tests whether our 16Hz/order-4
Butterworth filter on (v_origin, ω) before differentiating is the
right pre-derivative smoothing for kinetics extraction.

Sweeps cutoff (8, 12, 16, 20 Hz) × order (2, 4) plus a "no smoothing"
baseline, all on top of Dempster COMs (audit G best baseline). Reports
peak shoulder F, elbow F vs V3D.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path("/home/yabin/code/fwd_dynamics")
sys.path.insert(0, str(REPO / "src"))
from theia_osim.analysis.segment_reactions import (  # noqa: E402
    compute_throwing_arm_reactions_from_c3d,
)
import opensim as osim  # noqa: E402

C3D = REPO / "pose_filt_0.c3d"
PERSONAL = REPO / "out/audit_e/laiuhlrich_welded/all_recipes/theia_pitching_personalized.osim"

DE_LEVA = {"humerus_r": 0.5772, "ulna_r": 0.4574, "hand_r": 0.7900}
DEMPSTER = {"humerus_r": 0.436, "ulna_r": 0.430, "hand_r": 0.506}
V3D = {"shoulder_F": 1090., "shoulder_M": 151., "elbow_F": 1142., "elbow_M": 140.}


def derive_lengths(model_path):
    m = osim.Model(str(model_path)); m.initSystem()
    bs = m.getBodySet()
    return {b: float(np.linalg.norm(np.array([
        bs.get(b).getMassCenter().get(i) for i in range(3)
    ]))) / DE_LEVA[b] for b in DE_LEVA}


def dempster_coms(L):
    return {b: np.array([0.0, -DEMPSTER[b] * L[b], 0.0]) for b in DEMPSTER}


def peak_mag(arr3, t, t_lo, t_hi):
    mask = (t >= t_lo) & (t <= t_hi)
    sub = arr3[mask]
    if sub.size == 0: return float("nan")
    mags = np.linalg.norm(sub, axis=1)
    if not np.all(np.isfinite(mags)): return float("nan")
    return float(mags[np.argmax(mags)])


def run(cutoff_hz, com_o):
    res = compute_throwing_arm_reactions_from_c3d(
        C3D, PERSONAL, side="r", wrist_mode="welded",
        com_overrides=com_o, smoothing_hz=cutoff_hz,
    )
    t = res["times"]
    br_t = 1.593; t_lo, t_hi = br_t - 50/300., br_t + 30/300.
    return {
        "shoulder_F": peak_mag(res["shoulder_F_humerus"], t, t_lo, t_hi),
        "shoulder_M": peak_mag(res["shoulder_M_humerus"], t, t_lo, t_hi),
        "elbow_F":    peak_mag(res["elbow_F_ulna_frame"], t, t_lo, t_hi),
        "elbow_M":    peak_mag(res["elbow_M_ulna_frame"], t, t_lo, t_hi),
    }


def main():
    L = derive_lengths(PERSONAL)
    com_d = dempster_coms(L)

    print(f"Audit I — smoothing-cutoff sweep (Dempster COMs throughout)")
    print(f"{'cutoff_Hz':<14}{'sF':>10}{'sM':>10}{'eF':>10}{'eM':>10}{'sF/V3D':>10}{'eF/V3D':>10}")
    print("-" * 76)
    rows = []
    cutoffs = [None, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 30.0]
    for hz in cutoffs:
        p = run(hz, com_d)
        sf, sm = p["shoulder_F"], p["shoulder_M"]
        ef, em = p["elbow_F"], p["elbow_M"]
        label = "none" if hz is None else f"{hz:.0f}"
        print(f"{label:<14}{sf:>10.0f}{sm:>10.0f}{ef:>10.0f}{em:>10.0f}"
              f"{sf/V3D['shoulder_F']:>9.2f}x{ef/V3D['elbow_F']:>9.2f}x")
        rows.append((label, p))
    print(f"{'V3D':<14}"
          f"{V3D['shoulder_F']:>10.0f}{V3D['shoulder_M']:>10.0f}"
          f"{V3D['elbow_F']:>10.0f}{V3D['elbow_M']:>10.0f}     1.00x     1.00x")

    # Find the best cutoff for sF and eF.
    best_sF = min(rows, key=lambda r: abs(r[1]["shoulder_F"] - V3D["shoulder_F"]))
    best_eF = min(rows, key=lambda r: abs(r[1]["elbow_F"] - V3D["elbow_F"]))
    print(f"\nBest match to V3D shoulder_F: cutoff={best_sF[0]} Hz "
          f"(F={best_sF[1]['shoulder_F']:.0f}, ratio={best_sF[1]['shoulder_F']/V3D['shoulder_F']:.2f}x)")
    print(f"Best match to V3D elbow_F:    cutoff={best_eF[0]} Hz "
          f"(F={best_eF[1]['elbow_F']:.0f}, ratio={best_eF[1]['elbow_F']/V3D['elbow_F']:.2f}x)")


if __name__ == "__main__":
    main()
