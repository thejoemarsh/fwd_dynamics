"""Audit J: body-frame axis alignment.

Our COM offsets are stored along body -Y in OpenSim. We assume body Y
points proximal-to-distal, so COM at -Y means COM is distal of body
origin. This audit verifies that assumption empirically:

  1. From the personalized model, take humerus_r/ulna_r/hand_r body Y
     direction in world frame at frame 0.
  2. From Theia c3d 4×4s for r_uarm/r_larm/r_hand, take the segment's
     own Y axis in world (column 1 of the 4×4 rotation block) at frame 0.
  3. Compute the angle between OpenSim body Y and Theia segment Y.

If the angle is large (>10°), our COM offset is pointing in the wrong
direction relative to the actual physical long axis — and ω×(ω×r)
projections come out wrong, causing F errors.

If misalignment is found, also runs a second pass that overrides COMs
to point along the actual Theia segment long axis, and reports F deltas.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import yaml
import opensim as osim

REPO = Path("/home/yabin/code/fwd_dynamics")
sys.path.insert(0, str(REPO / "src"))
from theia_osim.c3d_io.reader import read_theia_c3d  # noqa: E402
from theia_osim.analysis.segment_reactions import (  # noqa: E402
    compute_throwing_arm_reactions_from_c3d,
)

C3D = REPO / "pose_filt_0.c3d"
PERSONAL = REPO / "out/audit_e/laiuhlrich_welded/all_recipes/theia_pitching_personalized.osim"

DE_LEVA = {"humerus_r": 0.5772, "ulna_r": 0.4574, "hand_r": 0.7900}
DEMPSTER = {"humerus_r": 0.436, "ulna_r": 0.430, "hand_r": 0.506}
THEIA_TO_OSIM = {"r_uarm": "humerus_r", "r_larm": "ulna_r", "r_hand": "hand_r"}
V3D = {"shoulder_F": 1090., "shoulder_M": 151., "elbow_F": 1142., "elbow_M": 140.}


def derive_lengths(model_path):
    m = osim.Model(str(model_path)); m.initSystem()
    bs = m.getBodySet()
    return {b: float(np.linalg.norm(np.array([
        bs.get(b).getMassCenter().get(i) for i in range(3)
    ]))) / DE_LEVA[b] for b in DE_LEVA}


def opensim_body_axes_world(model_path, frame_idx=0):
    """Get body→world rotation at the model's default state (no .mot loaded).
    Approximation — it gives the rotation when all coords = default. Suitable
    for relative-axis comparison since we only care about *direction*, not
    pose-dependent specifics. Actually we want pose at frame 0 — load .mot.
    """
    return None  # filled by run_full_compare


def angle_between(u, v):
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1))))


def peak_mag(arr3, t, t_lo, t_hi):
    mask = (t >= t_lo) & (t <= t_hi)
    sub = arr3[mask]
    if sub.size == 0: return float("nan")
    mags = np.linalg.norm(sub, axis=1)
    if not np.all(np.isfinite(mags)): return float("nan")
    return float(mags[np.argmax(mags)])


def main():
    cfg = yaml.safe_load(open(REPO / "configs/default.yaml"))
    vlb = np.array(cfg["slope"]["vlb_4x4"], dtype=np.float64)
    trial = read_theia_c3d(C3D, apply_vlb=vlb)

    # Get OpenSim body Y in world via BodyKinematics-style approach: at every
    # frame, R_b2g · [0,1,0] gives the body Y axis in world. We use the .mot
    # to drive coords, then we extract R_humerus_to_ground at frame 0 and BR.
    #
    # Easier: use the audit_h2 BodyKinematics output we already have, which
    # exports XYZ Euler angles per frame. Reconstruct R from those.
    bk_pos = REPO / "out/audit_h2_body_kin/audit_h2_BodyKinematics_pos_global.sto"
    if not bk_pos.exists():
        raise SystemExit(f"Missing {bk_pos}; run audit H2 first.")

    from theia_osim.analysis.body_kin import _read_sto
    df = _read_sto(bk_pos)
    times = df["time"].to_numpy()

    # Build R for each body of interest at each frame from XYZ-body Euler.
    def euler_xyz_to_R(eul_rad):
        cx, cy, cz = np.cos(eul_rad); sx, sy, sz = np.sin(eul_rad)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        return Rz @ Ry @ Rx

    osim_y_world = {}
    for theia_seg, osim_body in THEIA_TO_OSIM.items():
        cols = [f"{osim_body}_Ox", f"{osim_body}_Oy", f"{osim_body}_Oz"]
        eul = np.deg2rad(df[cols].to_numpy())
        N = len(eul)
        ys = np.zeros((N, 3))
        for f in range(N):
            R = euler_xyz_to_R(eul[f])
            ys[f] = R @ np.array([0., 1., 0.])
        osim_y_world[osim_body] = ys

    # Theia segment long-axis = r_*_4×4 column 1 (its body Y in world).
    theia_y_world = {}
    for theia_seg, osim_body in THEIA_TO_OSIM.items():
        T = trial.transforms[theia_seg]
        theia_y_world[osim_body] = T[:, :3, 1]

    # Compute angle at frame 0, mid, and BR.
    br_idx = int(np.argmin(np.abs(times - 1.593)))
    print("Body Y-axis vs Theia segment Y-axis (in world frame):")
    print(f"{'body':<12}{'frame':<8}{'OpenSim Y_world':<32}{'Theia Y_world':<32}{'angle':>8}")
    for osim_body in THEIA_TO_OSIM.values():
        for label, idx in [("0", 0), ("mid", len(times)//2), ("BR", br_idx)]:
            uo = osim_y_world[osim_body][idx]
            ut = theia_y_world[osim_body][idx]
            ang = angle_between(uo, ut)
            print(f"{osim_body:<12}{label:<8}"
                  f"{np.array2string(uo, precision=3):<32}"
                  f"{np.array2string(ut, precision=3):<32}"
                  f"{ang:>7.1f}°")

    # If the misalignment is large, attempt a corrective override: place each
    # body's COM along its TRUE long axis (Theia segment Y at frame 0, expressed
    # in OpenSim's body frame at frame 0).
    print("\nAttempting corrective COM override along Theia long-axis ...")
    L_seg = derive_lengths(PERSONAL)
    com_overrides = {}
    for theia_seg, osim_body in THEIA_TO_OSIM.items():
        # Compute the Theia long-axis direction in OpenSim's body frame at
        # frame 0.  R_osim_b2w · v_body = v_world  →  v_body = R^T · v_world
        eul0 = np.deg2rad(df[[f"{osim_body}_Ox", f"{osim_body}_Oy",
                              f"{osim_body}_Oz"]].iloc[0].to_numpy())
        R0 = euler_xyz_to_R(eul0)
        theia_y_w_0 = trial.transforms[theia_seg][0, :3, 1]
        theia_y_in_osim_body = R0.T @ theia_y_w_0
        # Place Dempster COM along that axis (in body frame). Convention sign:
        # Theia Y likely points proximal→distal; COM should be at +Theia_Y * D.
        D = DEMPSTER[osim_body] * L_seg[osim_body]
        com_body = theia_y_in_osim_body * D
        # Check sign by comparing with original (which is -Y_osim)
        if np.dot(com_body, np.array([0., -1., 0.])) < 0:
            com_body = -com_body
        com_overrides[osim_body] = com_body
        print(f"  {osim_body}: theia_long_axis_in_body={theia_y_in_osim_body}")
        print(f"    COM override (Dempster·L along corrected axis): {com_body}  |c|={np.linalg.norm(com_body):.4f}")

    # Run NE with the corrected COM overrides; compare against pure Dempster (-Y).
    base_dempster_coms = {b: np.array([0.0, -DEMPSTER[b] * L_seg[b], 0.0]) for b in DEMPSTER}

    def run(co):
        res = compute_throwing_arm_reactions_from_c3d(
            C3D, PERSONAL, side="r", wrist_mode="welded", com_overrides=co)
        t = res["times"]
        br_t = 1.593; t_lo, t_hi = br_t - 50/300., br_t + 30/300.
        return {
            "shoulder_F": peak_mag(res["shoulder_F_humerus"], t, t_lo, t_hi),
            "shoulder_M": peak_mag(res["shoulder_M_humerus"], t, t_lo, t_hi),
            "elbow_F":    peak_mag(res["elbow_F_ulna_frame"], t, t_lo, t_hi),
            "elbow_M":    peak_mag(res["elbow_M_ulna_frame"], t, t_lo, t_hi),
        }

    base = run(base_dempster_coms)
    aligned = run(com_overrides)
    print(f"\nF/M comparison (Dempster -Y vs theia-aligned Dempster):")
    print(f"  {'config':<28}{'sF':>10}{'sM':>10}{'eF':>10}{'eM':>10}{'sF/V3D':>10}{'eF/V3D':>10}")
    for name, p in [("dempster_-Y_baseline", base), ("dempster_theia_aligned", aligned)]:
        sf = p["shoulder_F"]; sm = p["shoulder_M"]; ef = p["elbow_F"]; em = p["elbow_M"]
        print(f"  {name:<28}{sf:>10.0f}{sm:>10.0f}{ef:>10.0f}{em:>10.0f}"
              f"{sf/V3D['shoulder_F']:>9.2f}x{ef/V3D['elbow_F']:>9.2f}x")


if __name__ == "__main__":
    main()
