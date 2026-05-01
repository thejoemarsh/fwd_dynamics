"""Audit H: override segment inertias with Hanavan/Yeadon-style values
on top of audit-G's Dempster COM correction. Tests whether inertia
mismatch is the lever for the remaining moment gap.

NOTE: F does not depend on I in Newton-Euler (F = m·(a_COM - g) + F_child),
so this audit can only close M. Including for completeness and to validate
the architectural prediction.

Hanavan 1964 / Winter biomech radii of gyration as fractions of segment
length (males, around COM):
  Upper arm: K_axial=0.158, K_transverse=0.322
  Forearm:   K_axial=0.121, K_transverse=0.303
  Hand:      K_axial=0.223, K_transverse=0.297
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
HANAVAN_K = {
    "humerus_r": (0.158, 0.322, 0.322),  # axial(Y), transverse, transverse
    "ulna_r":    (0.121, 0.303, 0.303),
    "hand_r":    (0.223, 0.297, 0.297),
}
V3D = {"shoulder_F": 1090., "shoulder_M": 151., "elbow_F": 1142., "elbow_M": 140.}


def derive_lengths(model_path):
    m = osim.Model(str(model_path)); m.initSystem()
    bs = m.getBodySet()
    out = {}
    for body, frac in DE_LEVA.items():
        com = bs.get(body).getMassCenter()
        com_np = np.array([com.get(0), com.get(1), com.get(2)])
        out[body] = float(np.linalg.norm(com_np)) / frac
    return out


def hanavan_inertia(model_path, lengths):
    """I_diag (about COM) = m·K². Body Y = long axis → I_yy uses K_axial,
    I_xx and I_zz use K_transverse."""
    m = osim.Model(str(model_path)); m.initSystem()
    bs = m.getBodySet()
    out = {}
    for body, ks in HANAVAN_K.items():
        mass = bs.get(body).getMass()
        L = lengths[body]
        K_ax, K_t1, K_t2 = ks
        Ixx = mass * (K_t1 * L) ** 2
        Iyy = mass * (K_ax * L) ** 2
        Izz = mass * (K_t2 * L) ** 2
        out[body] = np.diag([Ixx, Iyy, Izz])
    return out


def dempster_coms(lengths):
    return {b: np.array([0.0, -DEMPSTER[b] * L, 0.0]) for b, L in lengths.items()}


def peak_mag(arr3, t, t_lo, t_hi):
    mask = (t >= t_lo) & (t <= t_hi)
    sub = arr3[mask]
    if sub.size == 0: return float("nan")
    mags = np.linalg.norm(sub, axis=1)
    if not np.all(np.isfinite(mags)): return float("nan")
    return float(mags[np.argmax(mags)])


def run(label, com_o, I_o):
    print(f"\n=== {label} ===")
    res = compute_throwing_arm_reactions_from_c3d(
        C3D, PERSONAL, side="r", wrist_mode="welded",
        com_overrides=com_o, inertia_overrides=I_o,
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
    print("Segment lengths from de Leva back-derivation:")
    for k, v in L.items():
        print(f"  {k}: L = {v:.4f} m")

    com_dempster = dempster_coms(L)
    I_hanavan = hanavan_inertia(PERSONAL, L)

    # Print what we're putting in.
    m = osim.Model(str(PERSONAL)); m.initSystem(); bs = m.getBodySet()
    print("\nInertia comparison (about COM, body frame, kg·m²):")
    print(f"  {'body':<12}{'OUR_Ixx':>10}{'HAN_Ixx':>10}"
          f"{'OUR_Iyy':>10}{'HAN_Iyy':>10}{'OUR_Izz':>10}{'HAN_Izz':>10}")
    for body in HANAVAN_K:
        I_ours = np.zeros((3, 3))
        mom = bs.get(body).getInertia().getMoments()
        I_ours[0, 0] = mom.get(0); I_ours[1, 1] = mom.get(1); I_ours[2, 2] = mom.get(2)
        I_han = I_hanavan[body]
        print(f"  {body:<12}"
              f"{I_ours[0,0]:>10.5f}{I_han[0,0]:>10.5f}"
              f"{I_ours[1,1]:>10.5f}{I_han[1,1]:>10.5f}"
              f"{I_ours[2,2]:>10.5f}{I_han[2,2]:>10.5f}")

    rows = [
        ("baseline_de_leva",    None, None),
        ("dempster_COM",        com_dempster, None),
        ("dempster + hanavan_I", com_dempster, I_hanavan),
    ]
    results = []
    for label, co, io in rows:
        results.append((label, run(label, co, io)))

    print(f"\n{'config':<26}{'sF':>10}{'sM':>10}{'eF':>10}{'eM':>10}"
          f"{'sF/V3D':>10}{'sM/V3D':>10}{'eF/V3D':>10}{'eM/V3D':>10}")
    print("-" * 116)
    for name, p in results:
        sf, sm, ef, em = p["shoulder_F"], p["shoulder_M"], p["elbow_F"], p["elbow_M"]
        print(f"  {name:<24}{sf:>10.0f}{sm:>10.0f}{ef:>10.0f}{em:>10.0f}"
              f"{sf/V3D['shoulder_F']:>9.2f}x{sm/V3D['shoulder_M']:>9.2f}x"
              f"{ef/V3D['elbow_F']:>9.2f}x{em/V3D['elbow_M']:>9.2f}x")
    print(f"  {'V3D':<24}{V3D['shoulder_F']:>10.0f}{V3D['shoulder_M']:>10.0f}"
          f"{V3D['elbow_F']:>10.0f}{V3D['elbow_M']:>10.0f}     1.00x     1.00x     1.00x     1.00x")


if __name__ == "__main__":
    main()
