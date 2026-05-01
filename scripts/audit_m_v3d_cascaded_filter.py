"""Audit M: V3D-style cascaded filtering.

V3D's standard pipeline filters at two stages:
  1. Kinematics filter (on positions / orientations / velocities pre-derivative)
  2. Kinetics filter (on F/M output time series, post-Newton-Euler)

Both typically at 20 Hz Butterworth. The cascade gives a sharper effective
rolloff than either stage alone but doesn't dramatically lower the cutoff.

Tests several configurations:
  - 20 Hz kine + 20 Hz kinet (user's proposal — V3D's default)
  - 20 Hz kine + no kinet         (single-stage 20Hz)
  - 16 Hz kine + no kinet         (our current default)
  - 10 Hz kine + no kinet         (audit I winner)
  - 20 Hz kine + 12 Hz kinet
  - 20 Hz kine + 10 Hz kinet

All on Dempster -Y COMs (audit G best baseline).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt

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


def lowpass(arr, dt, cutoff_hz, order=4):
    if cutoff_hz is None:
        return arr.copy()
    fs = 1.0 / dt
    b, a = butter(order, cutoff_hz / (fs / 2.0), btype="lowpass")
    return filtfilt(b, a, arr, axis=0)


def peak_mag(arr3, t, t_lo, t_hi):
    mask = (t >= t_lo) & (t <= t_hi)
    sub = arr3[mask]
    if sub.size == 0: return float("nan")
    mags = np.linalg.norm(sub, axis=1)
    if not np.all(np.isfinite(mags)): return float("nan")
    return float(mags[np.argmax(mags)])


def run(kine_hz, kinet_hz, com_o, label):
    res = compute_throwing_arm_reactions_from_c3d(
        C3D, PERSONAL, side="r", wrist_mode="welded",
        com_overrides=com_o, smoothing_hz=kine_hz,
    )
    t = res["times"]
    dt = float(np.median(np.diff(t)))
    # Apply post-NE kinetic smoothing on F/M time series.
    keys = ["shoulder_F_humerus", "shoulder_M_humerus",
            "elbow_F_ulna_frame", "elbow_M_ulna_frame",
            "shoulder_F_torso", "shoulder_M_torso"]
    if kinet_hz is not None:
        for k in keys:
            res[k] = lowpass(res[k], dt, kinet_hz, order=4)

    br_t = 1.593; t_lo, t_hi = br_t - 50/300., br_t + 30/300.
    return {
        "shoulder_F": peak_mag(res["shoulder_F_humerus"], t, t_lo, t_hi),
        "shoulder_M": peak_mag(res["shoulder_M_humerus"], t, t_lo, t_hi),
        "elbow_F":    peak_mag(res["elbow_F_ulna_frame"], t, t_lo, t_hi),
        "elbow_M":    peak_mag(res["elbow_M_ulna_frame"], t, t_lo, t_hi),
    }


def main():
    L = derive_lengths(PERSONAL)
    com_d = {b: np.array([0.0, -DEMPSTER[b] * L[b], 0.0]) for b in DEMPSTER}

    configs = [
        ("16Hz kine + no kinet (current default)", 16.0, None),
        ("10Hz kine + no kinet (audit I winner)",  10.0, None),
        ("20Hz kine + no kinet",                   20.0, None),
        ("20Hz kine + 20Hz kinet (V3D-style)",     20.0, 20.0),
        ("20Hz kine + 12Hz kinet",                 20.0, 12.0),
        ("20Hz kine + 10Hz kinet",                 20.0, 10.0),
        ("20Hz kine +  8Hz kinet",                 20.0,  8.0),
        ("16Hz kine + 12Hz kinet",                 16.0, 12.0),
    ]

    print(f"\n{'config':<48}{'sF':>10}{'sM':>10}{'eF':>10}{'eM':>10}{'sF/V3D':>10}{'eF/V3D':>10}")
    print("-" * 106)
    for label, kine_hz, kinet_hz in configs:
        p = run(kine_hz, kinet_hz, com_d, label)
        sf, sm = p["shoulder_F"], p["shoulder_M"]
        ef, em = p["elbow_F"], p["elbow_M"]
        print(f"  {label:<46}{sf:>10.0f}{sm:>10.0f}{ef:>10.0f}{em:>10.0f}"
              f"{sf/V3D['shoulder_F']:>9.2f}x{ef/V3D['elbow_F']:>9.2f}x")
    print(f"  {'V3D':<46}{V3D['shoulder_F']:>10.0f}{V3D['shoulder_M']:>10.0f}"
          f"{V3D['elbow_F']:>10.0f}{V3D['elbow_M']:>10.0f}     1.00x     1.00x")


if __name__ == "__main__":
    main()
