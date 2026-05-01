"""H2 audit: compare OpenSim BodyKinematics body-frame ω vs V3D YABIN angular
velocity for pelvis / torso / humerus_r / ulna_r / hand_r.

If our ω matches V3D within ~5%, kinematics input is fine and the kinetics
inflation lives in mass/inertia/COM (already audited in Step 1) or in JR's
actuator-feedback pipeline (Step 3 / H3).

If our ω is inflated 1.5-3×, the bug is upstream in Recipe D's segment-4×4 →
coordinate decomposition or in OpenSim's spline derivative of the .mot.

Inputs:
  out/repro_baseline/all_recipes/theia_pitching_personalized.osim
  out/repro_baseline/recipe_d/analytical.mot
  pose_filt_0_procdb.json  (V3D ground truth)

Outputs:
  out/audit_h2_omega_<body>.png  — one panel per body, X/Y/Z + magnitude overlays
  out/audit_h2_omega_overview.png  — magnitude peaks summary across bodies
  out/audit_h2_omega_summary.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from theia_osim.analysis.body_kin import (  # noqa: E402
    read_body_velocities,
    run_body_kinematics,
)

import os
_REPRO_DIR = Path(os.environ.get(
    "AUDIT_H2_REPRO_DIR",
    str(REPO / "out/repro_baseline"),
))
PERSONAL_OSIM = _REPRO_DIR / "all_recipes/theia_pitching_personalized.osim"
ANALYTICAL_MOT = _REPRO_DIR / "all_recipes/recipe_d/analytical.mot"
PROCDB = REPO / "pose_filt_0_procdb.json"
OUT = REPO / "out"
BK_OUT = OUT / "audit_h2_body_kin"

# OpenSim body name -> V3D YABIN signal name
BODY_TO_V3D = {
    "pelvis":    "PELVIS_ANGULAR_VELOCITY",
    "torso":     "TORSO_ANGULAR_VELOCITY",
    "humerus_r": "SHOULDER_ANGULAR_VELOCITY",
    "ulna_r":    "ELBOW_ANGULAR_VELOCITY",
    "hand_r":    "WRIST_ANGULAR_VELOCITY",
}
BODIES = tuple(BODY_TO_V3D.keys())

# V3D procdb says 600 frames covering t in [0.003, 2.0] (handoff line 80)
V3D_T_START = 0.003
V3D_T_END = 2.0
V3D_N = 600


def load_v3d_angvel(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (time_s [N], xyz_deg_per_s [N,3])."""
    d = json.load(open(PROCDB))
    for it in d["Visual3D"]:
        if it.get("name") == name and it.get("folder") == "YABIN":
            sigs = {s["component"]: np.asarray(s["data"], dtype=float)
                    for s in it["signal"]}
            arr = np.column_stack([sigs["X"], sigs["Y"], sigs["Z"]])
            t = np.linspace(V3D_T_START, V3D_T_END, arr.shape[0])
            return t, arr
    raise KeyError(name)


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)
    BK_OUT.mkdir(exist_ok=True, parents=True)

    # --- run BodyKinematics for all bodies of interest
    print(f"Running BodyKinematics on {len(BODIES)} bodies...")
    bk = run_body_kinematics(
        PERSONAL_OSIM, ANALYTICAL_MOT, BK_OUT,
        bodies=BODIES, name="audit_h2",
    )
    print(f"  vel sto: {bk['vel']}")

    # --- collect ours per body
    ours: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body in BODIES:
        df = read_body_velocities(bk["vel"], body)
        omega = np.column_stack([df["omega_x"], df["omega_y"], df["omega_z"]])
        # OpenSim BodyKinematics emits angular velocity in deg/s already.
        # Pelvis sign convention: V3D and our pelvis frame disagree on the
        # X and Y axis direction (verified visually 2026-05-01). Flip to
        # match V3D so the overlay plots are meaningful.
        if body == "pelvis":
            omega[:, 0] *= -1.0
            omega[:, 1] *= -1.0
        elif body == "torso":
            omega[:, 1] *= -1.0
        ours[body] = (df["time"].to_numpy(), omega)

    # --- load V3D
    v3d: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body, name in BODY_TO_V3D.items():
        try:
            v3d[body] = load_v3d_angvel(name)
        except KeyError:
            print(f"  V3D signal missing: {name}")

    # --- per-body 4-panel plots
    summary_rows: list[str] = []
    summary_rows.append(f"{'body':12s} {'our_peak':>10s} {'v3d_peak':>10s} {'ratio':>7s} {'our_t':>7s} {'v3d_t':>7s}")
    summary_rows.append("-" * 60)

    for body in BODIES:
        t_us, om_us = ours[body]
        if body not in v3d:
            continue
        t_v, om_v = v3d[body]

        mag_us = np.linalg.norm(om_us, axis=1)
        mag_v = np.linalg.norm(om_v, axis=1)
        peak_us = float(np.max(mag_us))
        peak_v = float(np.max(mag_v))
        t_peak_us = float(t_us[np.argmax(mag_us)])
        t_peak_v = float(t_v[np.argmax(mag_v)])
        ratio = peak_us / peak_v if peak_v else float("nan")
        summary_rows.append(
            f"{body:12s} {peak_us:10.1f} {peak_v:10.1f} {ratio:7.2f} {t_peak_us:7.3f} {t_peak_v:7.3f}"
        )

        fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
        for i, comp in enumerate(["X", "Y", "Z"]):
            ax = axes.flat[i]
            ax.plot(t_v, om_v[:, i], color="#39c", lw=1.6, label=f"V3D {comp}")
            ax.plot(t_us, om_us[:, i], color="#c33", lw=1.2, label=f"ours {comp}")
            ax.set_ylabel(f"ω_{comp} (deg/s)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="upper left")
            ax.axvline(1.593, color="k", ls=":", lw=0.7, alpha=0.5)  # BR
        ax = axes.flat[3]
        ax.plot(t_v, mag_v, color="#39c", lw=1.6, label=f"V3D |ω| (peak {peak_v:.0f})")
        ax.plot(t_us, mag_us, color="#c33", lw=1.2, label=f"ours |ω| (peak {peak_us:.0f})")
        ax.axvline(1.593, color="k", ls=":", lw=0.7, alpha=0.5)
        ax.set_ylabel("|ω| (deg/s)")
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        fig.suptitle(f"H2 ω audit: {body}  (peak ratio ours/V3D = {ratio:.2f}×)")
        fig.tight_layout()
        out_png = OUT / f"audit_h2_omega_{body}.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f">>> wrote {out_png}")

    # --- overview bar chart of peak |ω| ratios
    fig, ax = plt.subplots(figsize=(9, 5))
    bodies_plot = [b for b in BODIES if b in v3d]
    peaks_us = [float(np.max(np.linalg.norm(ours[b][1], axis=1))) for b in bodies_plot]
    peaks_v = [float(np.max(np.linalg.norm(v3d[b][1], axis=1))) for b in bodies_plot]
    ratios = [u/v for u, v in zip(peaks_us, peaks_v)]
    x = np.arange(len(bodies_plot))
    bars = ax.bar(x, ratios, color="#c33")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.axhspan(0.9, 1.1, color="#9f9", alpha=0.25, label="±10% band")
    ax.set_xticks(x); ax.set_xticklabels(bodies_plot)
    ax.set_ylabel("peak |ω| ratio  ours / V3D")
    ax.set_title("H2 — peak body-frame ω, ours vs V3D YABIN")
    for i, r in enumerate(ratios):
        ax.text(i, r, f"{r:.2f}×", ha="center", va="bottom")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "audit_h2_omega_overview.png", dpi=120)
    plt.close(fig)
    print(f">>> wrote {OUT / 'audit_h2_omega_overview.png'}")

    summary_text = "\n".join(summary_rows)
    print("\n" + summary_text)
    (OUT / "audit_h2_omega_summary.txt").write_text(summary_text + "\n")
    print(f">>> wrote {OUT / 'audit_h2_omega_summary.txt'}")


if __name__ == "__main__":
    main()
