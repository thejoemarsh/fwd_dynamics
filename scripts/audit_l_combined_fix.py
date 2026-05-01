"""Audit L: combined-fix verdict.

Stack the two levers identified in audits I and J:
  - I: smoothing cutoff 10 Hz (was 16 Hz)
  - J: COM aligned along Theia segment Y axis instead of OpenSim body -Y

Then compare against V3D for the final V3D-match number.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import yaml

REPO = Path("/home/yabin/code/fwd_dynamics")
sys.path.insert(0, str(REPO / "src"))
from theia_osim.c3d_io.reader import read_theia_c3d  # noqa: E402
from theia_osim.analysis.body_kin import _read_sto  # noqa: E402
from theia_osim.analysis.segment_reactions import (  # noqa: E402
    compute_throwing_arm_reactions_from_c3d,
)
import opensim as osim  # noqa: E402

C3D = REPO / "pose_filt_0.c3d"
PERSONAL = REPO / "out/audit_e/laiuhlrich_welded/all_recipes/theia_pitching_personalized.osim"
BK_POS = REPO / "out/audit_h2_body_kin/audit_h2_BodyKinematics_pos_global.sto"

DE_LEVA = {"humerus_r": 0.5772, "ulna_r": 0.4574, "hand_r": 0.7900}
DEMPSTER = {"humerus_r": 0.436, "ulna_r": 0.430, "hand_r": 0.506}
THEIA_TO_OSIM = {"r_uarm": "humerus_r", "r_larm": "ulna_r", "r_hand": "hand_r"}
V3D = {"shoulder_F": 1090., "shoulder_M": 151., "elbow_F": 1142., "elbow_M": 140.}


def _euler_xyz_to_R(eul_rad):
    cx, cy, cz = np.cos(eul_rad); sx, sy, sz = np.sin(eul_rad)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx


def derive_lengths(model_path):
    m = osim.Model(str(model_path)); m.initSystem()
    bs = m.getBodySet()
    return {b: float(np.linalg.norm(np.array([
        bs.get(b).getMassCenter().get(i) for i in range(3)
    ]))) / DE_LEVA[b] for b in DE_LEVA}


def theia_aligned_coms(model_path):
    """Theia-aligned Dempster COMs: rotate the COM offset to point along
    the actual c3d segment long axis at frame 0, expressed in OpenSim's
    body frame at frame 0."""
    cfg = yaml.safe_load(open(REPO / "configs/default.yaml"))
    vlb = np.array(cfg["slope"]["vlb_4x4"], dtype=np.float64)
    trial = read_theia_c3d(C3D, apply_vlb=vlb)
    df = _read_sto(BK_POS)
    L = derive_lengths(model_path)
    out = {}
    for theia_seg, osim_body in THEIA_TO_OSIM.items():
        eul0 = np.deg2rad(df[[f"{osim_body}_Ox", f"{osim_body}_Oy",
                              f"{osim_body}_Oz"]].iloc[0].to_numpy())
        R0 = _euler_xyz_to_R(eul0)
        theia_y_w_0 = trial.transforms[theia_seg][0, :3, 1]
        theia_y_in_osim_body = R0.T @ theia_y_w_0
        com_body = theia_y_in_osim_body * (DEMPSTER[osim_body] * L[osim_body])
        # Sign: COM should point distal. Original COM is along -Y; compare.
        m = osim.Model(str(model_path)); m.initSystem()
        b = m.getBodySet().get(osim_body)
        orig = np.array([b.getMassCenter().get(0), b.getMassCenter().get(1),
                         b.getMassCenter().get(2)])
        if np.dot(com_body, orig) < 0:
            com_body = -com_body
        out[osim_body] = com_body
    return out


def peak_mag(arr3, t, t_lo, t_hi):
    mask = (t >= t_lo) & (t <= t_hi)
    sub = arr3[mask]
    if sub.size == 0: return float("nan"), float("nan")
    mags = np.linalg.norm(sub, axis=1)
    if not np.all(np.isfinite(mags)): return float("nan"), float("nan")
    idx = int(np.argmax(mags))
    return float(mags[idx]), float(t[mask][idx])


def run(co, hz, label):
    res = compute_throwing_arm_reactions_from_c3d(
        C3D, PERSONAL, side="r", wrist_mode="welded",
        com_overrides=co, smoothing_hz=hz,
    )
    t = res["times"]
    br_t = 1.593; t_lo, t_hi = br_t - 50/300., br_t + 30/300.
    out = {}
    for key, tag in [
        ("shoulder_F_humerus", "shoulder_F"),
        ("shoulder_M_humerus", "shoulder_M"),
        ("elbow_F_ulna_frame", "elbow_F"),
        ("elbow_M_ulna_frame", "elbow_M"),
    ]:
        mag, when = peak_mag(res[key], t, t_lo, t_hi)
        out[tag] = mag
    return out


def main():
    L = derive_lengths(PERSONAL)
    com_dempster_Y    = {b: np.array([0.0, -DEMPSTER[b] * L[b], 0.0]) for b in DEMPSTER}
    com_theia_aligned = theia_aligned_coms(PERSONAL)
    print("\nTheia-aligned Dempster COM (in OpenSim body frame):")
    for k, v in com_theia_aligned.items():
        print(f"  {k}: {v}  |c|={np.linalg.norm(v):.4f}")

    configs = [
        ("baseline (16Hz, dempster -Y)",            com_dempster_Y,    16.0),
        ("10Hz smoothing only",                     com_dempster_Y,    10.0),
        ("theia-aligned COM only",                  com_theia_aligned, 16.0),
        ("10Hz + theia-aligned (combined)",         com_theia_aligned, 10.0),
        ("12Hz + theia-aligned",                    com_theia_aligned, 12.0),
        ("8Hz + theia-aligned",                     com_theia_aligned,  8.0),
    ]
    print(f"\n{'config':<38}{'sF':>10}{'sM':>10}{'eF':>10}{'eM':>10}{'sF/V3D':>10}{'eF/V3D':>10}")
    print("-" * 96)
    for label, co, hz in configs:
        p = run(co, hz, label)
        sf, sm = p["shoulder_F"], p["shoulder_M"]
        ef, em = p["elbow_F"], p["elbow_M"]
        print(f"  {label:<36}{sf:>10.0f}{sm:>10.0f}{ef:>10.0f}{em:>10.0f}"
              f"{sf/V3D['shoulder_F']:>9.2f}x{ef/V3D['elbow_F']:>9.2f}x")
    print(f"  {'V3D':<36}{V3D['shoulder_F']:>10.0f}{V3D['shoulder_M']:>10.0f}"
          f"{V3D['elbow_F']:>10.0f}{V3D['elbow_M']:>10.0f}     1.00x     1.00x")


if __name__ == "__main__":
    main()
