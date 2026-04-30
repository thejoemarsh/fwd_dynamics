"""H2 follow-up: plot q, q̇, q̈ for the throwing arm coordinates from the
recipe_d analytical.mot. We compute q̇/q̈ via numerical gradient (matches what
OpenSim's spline derivative sees, modulo the GCV smoothing). Goal: localize
which coordinate produces the spike that inflates body-frame ω at BR.

Coordinates of interest:
  arm_flex_r, arm_add_r, arm_rot_r, elbow_flex_r, pro_sup_r

Outputs (in out/):
  audit_h2_qdot_<coord>.png  — q, q̇, q̈ time series with BR/MER markers
  audit_h2_qdot_overview.png — peak |q̇| and |q̈| bar chart
  audit_h2_qdot_summary.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "out"
MOT = REPO / "out/repro_baseline/all_recipes/recipe_d/analytical.mot"
sys.path.insert(0, str(REPO / "src"))
from theia_osim.analysis.body_kin import _read_sto  # noqa: E402

COORDS = ["arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r"]
T_BR = 1.593
T_MER = 1.563


def main() -> None:
    df = _read_sto(MOT)
    t = df["time"].to_numpy()
    dt = float(np.median(np.diff(t)))

    rows = [f"{'coord':14s} {'peak|q|':>9s} {'peak|qdot|':>11s} {'peak|qddot|':>12s} {'t(qdot)':>9s} {'t(qddot)':>9s}"]
    rows.append("-" * 72)

    peak_qd = []
    peak_qdd = []

    for coord in COORDS:
        col = coord if coord in df.columns else None
        if col is None:
            print(f"missing column: {coord}")
            continue
        q = df[col].to_numpy()  # degrees in OpenSim .mot for rotational coords
        qd = np.gradient(q, dt)
        qdd = np.gradient(qd, dt)

        peak_q = float(np.max(np.abs(q)))
        peak_qd_v = float(np.max(np.abs(qd)))
        peak_qdd_v = float(np.max(np.abs(qdd)))
        t_qd = float(t[np.argmax(np.abs(qd))])
        t_qdd = float(t[np.argmax(np.abs(qdd))])
        peak_qd.append(peak_qd_v)
        peak_qdd.append(peak_qdd_v)
        rows.append(f"{coord:14s} {peak_q:9.1f} {peak_qd_v:11.1f} {peak_qdd_v:12.1f} {t_qd:9.3f} {t_qdd:9.3f}")

        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        for ax in axes:
            ax.axvline(T_BR, color="k", ls=":", lw=0.8, alpha=0.6, label="BR")
            ax.axvline(T_MER, color="m", ls=":", lw=0.8, alpha=0.6, label="MER")
            ax.grid(alpha=0.3)
        axes[0].plot(t, q, color="#333"); axes[0].set_ylabel(f"{coord}\n(deg)")
        axes[1].plot(t, qd, color="#c33"); axes[1].set_ylabel("q̇ (deg/s)")
        axes[2].plot(t, qdd, color="#39c"); axes[2].set_ylabel("q̈ (deg/s²)")
        axes[2].set_xlabel("time (s)")
        axes[0].legend(loc="upper left", fontsize=8)
        fig.suptitle(f"{coord}: peak|q|={peak_q:.1f}°  peak|q̇|={peak_qd_v:.0f}°/s  peak|q̈|={peak_qdd_v:.0f}°/s²")
        fig.tight_layout()
        out_png = OUT / f"audit_h2_qdot_{coord}.png"
        fig.savefig(out_png, dpi=120); plt.close(fig)
        print(f">>> wrote {out_png}")

    # overview
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(COORDS))
    axes[0].bar(x, peak_qd, color="#c33"); axes[0].set_xticks(x); axes[0].set_xticklabels(COORDS, rotation=20)
    axes[0].set_ylabel("peak |q̇| (deg/s)"); axes[0].set_title("Throwing-arm q̇ peaks")
    for i, v in enumerate(peak_qd): axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom")
    axes[1].bar(x, peak_qdd, color="#39c"); axes[1].set_xticks(x); axes[1].set_xticklabels(COORDS, rotation=20)
    axes[1].set_ylabel("peak |q̈| (deg/s²)"); axes[1].set_title("Throwing-arm q̈ peaks")
    for i, v in enumerate(peak_qdd): axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "audit_h2_qdot_overview.png", dpi=120); plt.close(fig)
    print(f">>> wrote {OUT / 'audit_h2_qdot_overview.png'}")

    summary = "\n".join(rows)
    print("\n" + summary)
    (OUT / "audit_h2_qdot_summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
