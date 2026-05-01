"""Audit K: per-component overlay of our F vs V3D's F.

Reads SHOULDER_AR_FORCE, ELBOW_FORCE time series from procdb (3D vector
in resolved frame), pads to our timing, and overlays component-by-
component on a single panel. Diagnoses whether the residual gap is:

  - uniform scale (our_F = k · V3D_F for all components, all times)
  - per-axis bias (one component matches, others off)
  - peak-timing offset

Output:
  out/audit_k_v3d_components.png   side-by-side X/Y/Z overlay
  out/audit_k_summary.txt          per-component peak comparison
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path("/home/yabin/code/fwd_dynamics")
sys.path.insert(0, str(REPO / "src"))
from theia_osim.analysis.segment_reactions import (  # noqa: E402
    compute_throwing_arm_reactions_from_c3d,
)
import opensim as osim  # noqa: E402

C3D = REPO / "pose_filt_0.c3d"
PERSONAL = REPO / "out/audit_e/laiuhlrich_welded/all_recipes/theia_pitching_personalized.osim"
PROCDB = REPO / "pose_filt_0_procdb.json"

DE_LEVA = {"humerus_r": 0.5772, "ulna_r": 0.4574, "hand_r": 0.7900}
DEMPSTER = {"humerus_r": 0.436, "ulna_r": 0.430, "hand_r": 0.506}


def derive_lengths(model_path):
    m = osim.Model(str(model_path)); m.initSystem()
    bs = m.getBodySet()
    return {b: float(np.linalg.norm(np.array([
        bs.get(b).getMassCenter().get(i) for i in range(3)
    ]))) / DE_LEVA[b] for b in DE_LEVA}


def load_v3d_signal(name):
    """Return (t, vec) for a procdb signal."""
    p = json.loads(PROCDB.read_text())
    for it in p["Visual3D"]:
        if it.get("name") == name:
            sig = np.asarray(it["signal"])
            frames = np.asarray(it["frames"])
            return frames, sig
    raise KeyError(name)


def main():
    L = derive_lengths(PERSONAL)
    com_d = {b: np.array([0.0, -DEMPSTER[b] * L[b], 0.0]) for b in DEMPSTER}
    res = compute_throwing_arm_reactions_from_c3d(
        C3D, PERSONAL, side="r", wrist_mode="welded",
        com_overrides=com_d, smoothing_hz=16.0,
    )
    t_us = res["times"]
    sF_us = res["shoulder_F_humerus"]
    eF_us = res["elbow_F_ulna_frame"]

    f_v3d_sF, sig_v3d_sF = load_v3d_signal("SHOULDER_AR_FORCE")
    f_v3d_eF, sig_v3d_eF = load_v3d_signal("ELBOW_FORCE")
    # V3D 'frames' is sample index; convert to seconds. sample rate = 300 Hz.
    t_v3d_sF = f_v3d_sF / 300.0
    t_v3d_eF = f_v3d_eF / 300.0

    # Overlay.
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    BR = 1.593

    def plot_row(ax_row, t_us, ours, t_v3d, v3d, title_prefix):
        for i, comp in enumerate("XYZ"):
            ax = ax_row[i]
            ax.plot(t_v3d, v3d[:, i], color="#39c", lw=1.6, label="V3D")
            ax.plot(t_us, ours[:, i], color="#c33", lw=1.0, label="ours")
            ax.set_title(f"{title_prefix} {comp}")
            ax.axvline(BR, color="k", ls=":", lw=0.7, alpha=0.5)
            ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")
        ax = ax_row[3]
        m_v3d = np.linalg.norm(v3d, axis=1)
        m_us  = np.linalg.norm(ours, axis=1)
        ax.plot(t_v3d, m_v3d, color="#39c", lw=1.6, label=f"V3D |F| (max {m_v3d.max():.0f})")
        ax.plot(t_us, m_us, color="#c33", lw=1.0, label=f"ours |F| (max {m_us.max():.0f})")
        ax.axvline(BR, color="k", ls=":", lw=0.7)
        ax.set_title(f"{title_prefix} |F|")
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")

    plot_row(axes[0], t_us, sF_us, t_v3d_sF, sig_v3d_sF, "SHOULDER_AR_F")
    plot_row(axes[1], t_us, eF_us, t_v3d_eF, sig_v3d_eF, "ELBOW_F")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"Audit K — per-component F overlay (Dempster COM, 16Hz smoothing)")
    fig.tight_layout()
    out_png = REPO / "out/audit_k_v3d_components.png"
    fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f">>> wrote {out_png}")

    # Per-component peak summary in throw window.
    t_lo, t_hi = BR - 50/300.0, BR + 30/300.0
    mask_us = (t_us >= t_lo) & (t_us <= t_hi)
    mask_sF_v = (t_v3d_sF >= t_lo) & (t_v3d_sF <= t_hi)
    mask_eF_v = (t_v3d_eF >= t_lo) & (t_v3d_eF <= t_hi)
    lines = []
    lines.append("Per-component peak |value| in throw window [BR-50, BR+30] frames:\n")
    for label, our, our_mask, v3d, v3d_mask in [
        ("SHOULDER_AR_F", sF_us, mask_us, sig_v3d_sF, mask_sF_v),
        ("ELBOW_F",       eF_us, mask_us, sig_v3d_eF, mask_eF_v),
    ]:
        lines.append(f"  {label}:")
        for i, comp in enumerate("XYZ"):
            our_peak = float(np.max(np.abs(our[our_mask, i])))
            v3d_peak = float(np.max(np.abs(v3d[v3d_mask, i])))
            ratio = our_peak / v3d_peak if v3d_peak else float("nan")
            lines.append(f"    {comp}:   ours={our_peak:>8.1f}   V3D={v3d_peak:>8.1f}   ratio={ratio:>5.2f}x")
    out_txt = REPO / "out/audit_k_summary.txt"
    out_txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n>>> wrote {out_txt}")


if __name__ == "__main__":
    main()
