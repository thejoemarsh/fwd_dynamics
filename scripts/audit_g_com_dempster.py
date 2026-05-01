"""Audit G: replace de Leva COM offsets with Dempster/Hanavan COM offsets
on humerus_r, ulna_r, hand_r and re-run the c3d-driven Newton-Euler.

Tests whether the residual ~2× F gap from audit E (mass-matched) is
explained by COM-offset disagreement between de Leva (used by our
personalize.py) and Dempster (V3D's likely default for HYBRID_SEGMENT
when MASS-fraction is set but PROX_TO_CG_* is blank).

de Leva 1996 (males) — fraction of segment length from PROXIMAL end:
  Upper arm: 0.5772
  Forearm:   0.4574
  Hand:      0.7900

Dempster 1955 / Winter biomech — fraction from PROXIMAL end:
  Upper arm: 0.436
  Forearm:   0.430
  Hand:      0.506

Segment length is back-derived from our model's existing COM offset:
  L_segment = |existing_com| / de_Leva_fraction

Body-frame convention: Y axis along proximal→distal, COM at -Y of body origin.
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
PERSONAL = (REPO / "out/audit_e/laiuhlrich_welded/all_recipes"
            / "theia_pitching_personalized.osim")

DE_LEVA = {"humerus_r": 0.5772, "ulna_r": 0.4574, "hand_r": 0.7900}
DEMPSTER = {"humerus_r": 0.436,  "ulna_r": 0.430,  "hand_r": 0.506}

V3D = {"shoulder_F": 1090., "shoulder_M": 151., "elbow_F": 1142., "elbow_M": 140.}


def derive_segment_lengths(model_path: Path) -> dict:
    """L_seg = |existing_com| / de_leva_fraction. Existing COM is along
    body -Y per LaiUhlrich2022 convention."""
    m = osim.Model(str(model_path)); m.initSystem()
    bs = m.getBodySet()
    out = {}
    for body, frac in DE_LEVA.items():
        com = bs.get(body).getMassCenter()
        com_np = np.array([com.get(0), com.get(1), com.get(2)])
        L = float(np.linalg.norm(com_np)) / frac
        out[body] = L
    return out


def dempster_com_offsets(lengths: dict) -> dict:
    """Same direction as the existing -Y COM offset, magnitude = Dempster · L."""
    return {body: np.array([0.0, -DEMPSTER[body] * L, 0.0])
            for body, L in lengths.items()}


def peak_mag(arr3, t, t_lo, t_hi):
    mask = (t >= t_lo) & (t <= t_hi)
    sub = arr3[mask]
    if sub.size == 0: return float("nan"), float("nan")
    mags = np.linalg.norm(sub, axis=1)
    if not np.all(np.isfinite(mags)): return float("nan"), float("nan")
    idx = int(np.argmax(mags))
    return float(mags[idx]), float(t[mask][idx])


def run_one(label: str, com_overrides: dict | None) -> dict:
    print(f"\n=== {label} ===")
    if com_overrides is not None:
        print("COM overrides (body frame):")
        for k, v in com_overrides.items():
            print(f"  {k}: |com|={np.linalg.norm(v):.4f} m")
    res = compute_throwing_arm_reactions_from_c3d(
        C3D, PERSONAL, side="r", wrist_mode="welded",
        com_overrides=com_overrides,
    )
    times = res["times"]
    br_t = 1.593; t_lo, t_hi = br_t - 50/300., br_t + 30/300.
    peaks = {}
    for key, tag in [
        ("shoulder_F_humerus", "shoulder_F"),
        ("shoulder_M_humerus", "shoulder_M"),
        ("elbow_F_ulna_frame", "elbow_F"),
        ("elbow_M_ulna_frame", "elbow_M"),
    ]:
        mag, when = peak_mag(res[key], times, t_lo, t_hi)
        peaks[tag] = {"peak": mag, "t_s": when}
    return peaks


def main():
    L = derive_segment_lengths(PERSONAL)
    print("Derived segment lengths from existing de Leva COM offsets:")
    for k, v in L.items():
        print(f"  {k}: L = {v:.4f} m  (de_Leva COM = {DE_LEVA[k]:.4f}·L; "
              f"Dempster would be {DEMPSTER[k]:.4f}·L = {DEMPSTER[k]*v:.4f} m)")

    dempster_overrides = dempster_com_offsets(L)

    base = run_one("Baseline (de Leva COMs, audit E reference)", None)
    demp = run_one("Dempster COMs", dempster_overrides)

    rows = [
        ("baseline_de_leva", base),
        ("dempster",         demp),
    ]
    print(f"\nCOM-offset bake-off (c3d-driven Newton-Euler, welded wrist):")
    print(f"  {'config':<22}{'sF':>10}{'sM':>10}{'eF':>10}{'eM':>10}"
          f"{'sF/V3D':>10}{'eF/V3D':>10}")
    print("-" * 82)
    for name, p in rows:
        sf = p["shoulder_F"]["peak"]; sm = p["shoulder_M"]["peak"]
        ef = p["elbow_F"]["peak"];    em = p["elbow_M"]["peak"]
        print(f"  {name:<22}{sf:>10.0f}{sm:>10.0f}{ef:>10.0f}{em:>10.0f}"
              f"{sf/V3D['shoulder_F']:>9.2f}x{ef/V3D['elbow_F']:>9.2f}x")
    print(f"  {'V3D':<22}{V3D['shoulder_F']:>10.0f}{V3D['shoulder_M']:>10.0f}"
          f"{V3D['elbow_F']:>10.0f}{V3D['elbow_M']:>10.0f}     1.00x     1.00x")


if __name__ == "__main__":
    main()
