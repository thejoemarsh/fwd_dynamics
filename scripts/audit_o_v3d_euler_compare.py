"""Audit O: V3D-comparable Euler decomposition for shoulder kinematics.

Recipe D currently decomposes shoulder relative orientation via ZXY Cardan
(matching LaiUhlrich2022's acromial joint axes) — that's correct for
driving OpenSim, but not directly comparable to V3D's reported shoulder
angles. V3D uses ZYZ Euler (AXIS1=Z, AXIS3=Z per RT_SHOULDER_ANGLE in
v3d_scripts/02_CMO_v6_v1.v3s line 242-260).

This audit:
  1. Pulls torso (r_uarm relative to torso) segment 4×4s from the c3d.
  2. Computes R_relative = R_torso^T @ R_uarm at every frame.
  3. Decomposes via:
       - intrinsic ZYZ (V3D's primary convention)
       - intrinsic YXZ (V3D's alternate, used in some scripts)
       - intrinsic ZXY (our current Recipe D output, for sanity)
  4. Smart-unwraps the periodic angles (np.unwrap on z1, z2 for ZYZ).
  5. Compares to V3D's procdb SHOULDER_ANGLE per component.
  6. Writes:
       out/audit_o_zyz_overlay.png  — overlay of all three decompositions
                                      vs V3D's three reported components
       out/audit_o_summary.txt      — peak / RMS comparison

If ZYZ matches V3D within a few degrees, that confirms our kinematic
processing is correct and the giant 150° RMS in audit N was purely a
convention mismatch.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

REPO = Path("/home/yabin/code/fwd_dynamics")
sys.path.insert(0, str(REPO / "src"))
from theia_osim.c3d_io.reader import read_theia_c3d  # noqa: E402

C3D = REPO / "pose_filt_0.c3d"
PROCDB = REPO / "pose_filt_0_procdb.json"
BR_TIME = 1.593
THROW_WINDOW = (BR_TIME - 50/300., BR_TIME + 30/300.)


def load_v3d(name):
    p = json.loads(PROCDB.read_text())
    for it in p["Visual3D"]:
        if it.get("name") == name:
            comps = it["signal"]
            data = np.array([
                [float("nan") if v is None else float(v) for v in c["data"]]
                for c in comps
            ], dtype=np.float64).T
            t = np.arange(data.shape[0]) / 300.0
            return t, data
    raise KeyError(name)


def smart_unwrap_columns(arr_deg, axes_to_unwrap=(0, 2)):
    """np.unwrap on selected axes. arr is (T, 3) in degrees."""
    out = arr_deg.copy()
    for ax in axes_to_unwrap:
        out[:, ax] = np.degrees(np.unwrap(np.radians(arr_deg[:, ax])))
    return out


def decompose(R_traj, sequence: str, intrinsic: bool = True):
    """Decompose (T, 3, 3) rotation trajectory into Euler angles in degrees.
    `sequence` is a 3-char string. If intrinsic=True, scipy uses lowercase."""
    seq = sequence.lower() if intrinsic else sequence.upper()
    angles = Rotation.from_matrix(R_traj).as_euler(seq, degrees=True)
    return angles


def main():
    cfg = yaml.safe_load(open(REPO / "configs/default.yaml"))
    vlb = np.array(cfg["slope"]["vlb_4x4"], dtype=np.float64)
    trial = read_theia_c3d(C3D, apply_vlb=vlb)
    fs = trial.sample_rate_hz
    n = trial.n_frames
    t = np.arange(n) / fs

    # R_torso, R_uarm (right side)
    R_torso = trial.transforms["torso"][:, :3, :3]
    R_uarm  = trial.transforms["r_uarm"][:, :3, :3]
    # Relative rotation: humerus expressed in torso frame.
    R_rel = np.einsum("tji,tjk->tik", R_torso, R_uarm)  # R_torso^T @ R_uarm

    # Decompose three ways.
    zyz = decompose(R_rel, "ZYZ", intrinsic=True)
    yxz = decompose(R_rel, "YXZ", intrinsic=True)
    zxy = decompose(R_rel, "ZXY", intrinsic=True)

    # Smart-unwrap the proper-Euler sequence (ZYZ): unwrap the two Z angles.
    zyz_unw = smart_unwrap_columns(zyz, axes_to_unwrap=(0, 2))
    yxz_unw = smart_unwrap_columns(yxz, axes_to_unwrap=(0, 1, 2))
    zxy_unw = smart_unwrap_columns(zxy, axes_to_unwrap=(0, 1, 2))

    # No anchoring — display the raw ZYZ-with-np.unwrap signal so the
    # SHAPE comparison against V3D is uncluttered by branch corrections.
    zyz_disp = zyz_unw

    # Apply V3D's anchor unwrap (period=360°). Our np.unwrap is already
    # equivalent in absence of an explicit anchor frame.
    try:
        t_v3d, v3d = load_v3d("SHOULDER_ANGLE")
    except KeyError:
        print("V3D SHOULDER_ANGLE missing"); return
    # V3D plotted as-is. Two corrections applied to OUR side:
    #   1. Multiply our z1 and z2 by -1 (sign convention).
    #   2. Swap z1 ↔ z2 columns. V3D's component "X" reports the third Z
    #      rotation (z2) and component "Z" reports the first (z1) — opposite
    #      of scipy's `as_euler("zyz")` ordering.
    # User-confirmed 2026-05-01.
    for arr in (zyz_unw, yxz_unw, zxy_unw):
        arr[:, 0] *= -1.0
        arr[:, 2] *= -1.0
        arr[:, [0, 2]] = arr[:, [2, 0]]
    # ZYZ first/third Z angles wrap onto a different branch than V3D's
    # anchored unwrap. Subtract the median offset against V3D per axis,
    # snapped to the nearest 180° multiple, so the residual reflects only
    # genuine kinematic disagreement.
    for ax in (0, 2):
        v_at_us = np.interp(t, t_v3d, v3d[:, ax])
        valid = np.isfinite(v_at_us) & np.isfinite(zyz_unw[:, ax])
        if not np.any(valid):
            continue
        median_off = float(np.median(zyz_unw[valid, ax] - v_at_us[valid]))
        # Snap to nearest 180° to avoid biasing toward unwrapped branches
        snap = 180.0 * round(median_off / 180.0)
        zyz_unw[:, ax] -= snap

    # Single clean ZYZ overlay vs V3D, with the anchored unwrap applied so
    # the curves sit in the same 360°-branch as V3D.
    component_labels = ["X = z1 (axial-1)", "Y = y (elevation)", "Z = z2 (axial-2)"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for row, lbl in enumerate(component_labels):
        ax = axes[row]
        ax.plot(t_v3d, v3d[:, row], color="#000", lw=1.8, alpha=0.85,
                label=f"V3D component {lbl.split('=')[0].strip()}")
        ax.plot(t, zyz_disp[:, row], color="#c33", lw=1.2, ls="--",
                label=f"ours ZYZ Euler axis {row+1}")
        ax.axvline(BR_TIME, color="k", ls=":", alpha=0.4, lw=0.7,
                   label="BR" if row == 0 else None)
        ax.set_ylabel(f"{lbl} (deg)")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Shoulder ZYZ Euler — ours vs V3D SHOULDER_ANGLE (anchor-unwrapped to V3D branch)")
    fig.tight_layout()
    fig.savefig(REPO / "out/audit_o_zyz_overlay.png", dpi=120)
    plt.close(fig)
    print(">>> wrote out/audit_o_zyz_overlay.png")

    # Summary: per-axis RMS in throw window for each decomposition.
    mask = (t >= THROW_WINDOW[0]) & (t <= THROW_WINDOW[1])
    lines = ["", "Audit O — shoulder Euler decomposition vs V3D SHOULDER_ANGLE",
             "  RMS (deg) computed in throw window [BR-50, BR+30] frames", ""]
    lines.append(f"  {'decomp':<22}{'axis-1 RMS':>14}{'axis-2 RMS':>14}{'axis-3 RMS':>14}{'best-match-axis':>22}")
    lines.append("-" * 88)
    decomps = [
        ("ZYZ (anchored)", zyz_disp, "#c33"),
        ("YXZ (V3D alternate)", yxz_unw, "#3a3"),
        ("ZXY (our Recipe D)", zxy_unw, "#39c"),
    ]
    for label, decomp, _ in decomps:
        rmses = []
        for ax_idx in range(3):
            v3d_resamp = np.interp(t[mask], t_v3d, v3d[:, ax_idx])
            valid = np.isfinite(v3d_resamp)
            r = float(np.sqrt(np.mean((decomp[mask, ax_idx][valid] -
                                       v3d_resamp[valid]) ** 2)))
            rmses.append(r)
        # best matching: try every permutation of (decomp axes) to (v3d axes)
        # to handle conventions where V3D's component order differs from ours.
        from itertools import permutations
        best_rms_sum = float("inf"); best_perm = None
        for perm in permutations(range(3)):
            total = 0.0
            for v_ax, d_ax in enumerate(perm):
                v3d_r = np.interp(t[mask], t_v3d, v3d[:, v_ax])
                valid = np.isfinite(v3d_r)
                total += np.sqrt(np.mean((decomp[mask, d_ax][valid] - v3d_r[valid]) ** 2))
            if total < best_rms_sum:
                best_rms_sum = total; best_perm = perm
        perm_str = "→".join(f"d{p+1}↔V{i+1}" for i, p in enumerate(best_perm))
        lines.append(f"  {label:<22}{rmses[0]:>13.1f}°{rmses[1]:>13.1f}°{rmses[2]:>13.1f}°"
                     f"{perm_str:>22}")
    print("\n".join(lines))
    (REPO / "out/audit_o_summary.txt").write_text("\n".join(lines) + "\n")

    # Best-matching permutation, full rerun with that permutation
    # for the ZYZ decomp specifically (the V3D primary).
    lines2 = ["", "ZYZ decomposition with best-matching axis permutation:"]
    decomp = zyz_unw
    from itertools import permutations
    best_rms_sum = float("inf"); best_perm = None; best_signs = None
    for perm in permutations(range(3)):
        for signs in [(s1, s2, s3) for s1 in (1, -1) for s2 in (1, -1) for s3 in (1, -1)]:
            total = 0.0
            for v_ax, d_ax in enumerate(perm):
                v3d_r = np.interp(t[mask], t_v3d, v3d[:, v_ax])
                valid = np.isfinite(v3d_r)
                total += np.sqrt(np.mean((signs[v_ax]*decomp[mask, d_ax][valid] - v3d_r[valid]) ** 2))
            if total < best_rms_sum:
                best_rms_sum = total; best_perm = perm; best_signs = signs
    lines2.append(f"  best perm:  decomp axes {best_perm} mapped to V3D 1/2/3")
    lines2.append(f"  best signs: {best_signs}")
    lines2.append(f"  total RMS:  {best_rms_sum:.1f}° (sum across 3 axes)")
    print("\n".join(lines2))
    (REPO / "out/audit_o_summary.txt").write_text(
        "\n".join(lines + lines2) + "\n")


if __name__ == "__main__":
    main()
