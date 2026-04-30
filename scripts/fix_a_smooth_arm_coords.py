"""Fix A: post-process recipe_d analytical.mot to crush the gimbal-lock q̈
spike on the throwing arm before ID/JR sees it.

At t≈1.603s, arm_add_r passes through +90° (X-axis Cardan singularity). Both
Cardan branches are non-smooth there: arm_flex_r and arm_rot_r jump ±53° in
one frame even after the existing smart-unwrap branch selection. ID's GCV
spline turns that into a ~10-frame q̈ pulse → inflated body-frame ω → 6-13×
kinetics inflation.

This script:
  1. Loads out/repro_baseline/all_recipes/recipe_d/analytical.mot
  2. Applies a 12 Hz Butterworth lowpass (4th order, filtfilt) to
     arm_flex_r, arm_add_r, arm_rot_r only — leaves all other coords untouched
  3. Writes out/repro_fixA/recipe_d/analytical.mot
  4. Re-runs the q̇/q̈ overview plots side-by-side with baseline
  5. Re-runs BodyKinematics → ω comparison vs V3D

The 12 Hz cutoff was chosen to crush a 1-frame spike at 300 Hz sample rate
while preserving the legitimate motion bandwidth (pitching is < 8 Hz on q).

Outputs in out/:
  audit_fixA_qdot_overview.png       — peak q̇/q̈ before vs after Fix A
  audit_fixA_qdot_<coord>.png         — q, q̇, q̈ for each arm coord
  audit_fixA_omega_overview.png       — body-frame ω peaks before vs after
  audit_fixA_summary.txt
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "out"
sys.path.insert(0, str(REPO / "src"))
from theia_osim.analysis.body_kin import (  # noqa: E402
    _read_sto, read_body_velocities, run_body_kinematics,
)
from theia_osim.kinematics_postprocess.filter import lowpass_filtfilt  # noqa: E402

BASELINE_MOT = REPO / "out/repro_baseline/all_recipes/recipe_d/analytical.mot"
PERSONAL_OSIM = REPO / "out/repro_baseline/all_recipes/theia_pitching_personalized.osim"
FIXA_DIR = OUT / "repro_fixA" / "recipe_d"
FIXA_MOT = FIXA_DIR / "analytical.mot"
FIXA_BK = OUT / "repro_fixA" / "body_kin"

ARM_COORDS = ["arm_flex_r", "arm_add_r", "arm_rot_r"]
ALL_ARM_DIAG_COORDS = ARM_COORDS + ["elbow_flex_r"]
SAMPLE_RATE = 300.0
CUTOFF_HZ = 12.0
BUTTER_ORDER = 4

T_BR = 1.593
T_MER = 1.563

BODY_TO_V3D = {
    "pelvis":    "PELVIS_ANGULAR_VELOCITY",
    "torso":     "TORSO_ANGULAR_VELOCITY",
    "humerus_r": "SHOULDER_ANGULAR_VELOCITY",
    "ulna_r":    "ELBOW_ANGULAR_VELOCITY",
    "hand_r":    "WRIST_ANGULAR_VELOCITY",
}


def load_v3d_angvel_peak(name: str) -> float:
    import json
    d = json.load(open(REPO / "pose_filt_0_procdb.json"))
    for it in d["Visual3D"]:
        if it.get("name") == name and it.get("folder") == "YABIN":
            sigs = {s["component"]: np.asarray(s["data"], dtype=float)
                    for s in it["signal"]}
            arr = np.column_stack([sigs["X"], sigs["Y"], sigs["Z"]])
            return float(np.max(np.linalg.norm(arr, axis=1)))
    return float("nan")


def write_mot_inplace(src_mot: Path, dst_mot: Path, replacements: dict[str, np.ndarray]) -> None:
    """Copy src_mot to dst_mot but replace the named columns with new arrays."""
    dst_mot.parent.mkdir(parents=True, exist_ok=True)
    text = src_mot.read_text()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            data_start = i + 1
            break
    else:
        raise ValueError("no endheader")
    header = lines[data_start].split()
    col_idx = {h: header.index(h) for h in replacements}
    out_data_lines = []
    n_data = len(lines) - data_start - 1
    for k, frame_idx in zip(replacements, [None]*len(replacements)):
        assert replacements[k].shape == (n_data,)
    for j in range(n_data):
        row = lines[data_start + 1 + j].split()
        for h, arr in replacements.items():
            row[col_idx[h]] = f"{arr[j]:.10g}"
        out_data_lines.append("\t".join(row))
    out = lines[:data_start + 1] + out_data_lines
    dst_mot.write_text("\n".join(out) + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)

    # 1. Load baseline mot, filter the three arm coords
    df = _read_sto(BASELINE_MOT)
    t = df["time"].to_numpy()
    replacements: dict[str, np.ndarray] = {}
    baseline_q: dict[str, np.ndarray] = {}
    fixed_q: dict[str, np.ndarray] = {}
    for c in ARM_COORDS:
        q = df[c].to_numpy().astype(float)
        baseline_q[c] = q.copy()
        # filtfilt expects (N, K) or (N,) — use 1-D path
        q_filt = lowpass_filtfilt(
            q[:, None], cutoff_hz=CUTOFF_HZ,
            sample_rate_hz=SAMPLE_RATE, order=BUTTER_ORDER,
        ).ravel()
        fixed_q[c] = q_filt
        replacements[c] = q_filt

    write_mot_inplace(BASELINE_MOT, FIXA_MOT, replacements)
    print(f"wrote {FIXA_MOT}")

    # 2. q̇/q̈ comparison plots
    dt = float(np.median(np.diff(t)))
    summary_rows = [f"{'coord':14s} {'pk|qdot|':>10s} {'pk|qddot|':>12s}    after->{'pk|qdot|':>10s} {'pk|qddot|':>12s}   q̈ ratio"]
    summary_rows.append("-" * 95)
    for c in ARM_COORDS:
        qb = baseline_q[c]; qf = fixed_q[c]
        qdb = np.gradient(qb, dt); qddb = np.gradient(qdb, dt)
        qdf = np.gradient(qf, dt); qddf = np.gradient(qdf, dt)
        pk_qdb = float(np.max(np.abs(qdb))); pk_qddb = float(np.max(np.abs(qddb)))
        pk_qdf = float(np.max(np.abs(qdf))); pk_qddf = float(np.max(np.abs(qddf)))
        ratio = pk_qddf / pk_qddb if pk_qddb else float("nan")
        summary_rows.append(
            f"{c:14s} {pk_qdb:10.0f} {pk_qddb:12.0f}            {pk_qdf:10.0f} {pk_qddf:12.0f}    {ratio:6.3f}"
        )

        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        for ax in axes:
            ax.axvline(T_BR, color="k", ls=":", lw=0.8, alpha=0.6)
            ax.axvline(T_MER, color="m", ls=":", lw=0.8, alpha=0.6)
            ax.grid(alpha=0.3)
        axes[0].plot(t, qb, color="#888", label="baseline"); axes[0].plot(t, qf, color="#c33", label="Fix A 12Hz")
        axes[0].set_ylabel(f"{c}  q (deg)"); axes[0].legend(fontsize=8)
        axes[1].plot(t, qdb, color="#888", label="baseline"); axes[1].plot(t, qdf, color="#c33", label="Fix A")
        axes[1].set_ylabel("q̇ (deg/s)"); axes[1].legend(fontsize=8)
        axes[2].plot(t, qddb, color="#888", label="baseline"); axes[2].plot(t, qddf, color="#c33", label="Fix A")
        axes[2].set_ylabel("q̈ (deg/s²)"); axes[2].legend(fontsize=8)
        axes[2].set_xlabel("time (s)")
        fig.suptitle(f"Fix A: {c} — peak q̈ {pk_qddb:.0f} → {pk_qddf:.0f} °/s²  ({ratio:.2f}×)")
        fig.tight_layout()
        fig.savefig(OUT / f"audit_fixA_qdot_{c}.png", dpi=120); plt.close(fig)
        print(f">>> wrote {OUT / f'audit_fixA_qdot_{c}.png'}")

    # bar overview q̈ before/after
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ARM_COORDS)); w = 0.38
    qddb_peaks = [float(np.max(np.abs(np.gradient(np.gradient(baseline_q[c], dt), dt)))) for c in ARM_COORDS]
    qddf_peaks = [float(np.max(np.abs(np.gradient(np.gradient(fixed_q[c], dt), dt)))) for c in ARM_COORDS]
    ax.bar(x - w/2, qddb_peaks, w, color="#888", label="baseline 20 Hz")
    ax.bar(x + w/2, qddf_peaks, w, color="#c33", label=f"Fix A {CUTOFF_HZ:.0f} Hz")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(ARM_COORDS)
    ax.set_ylabel("peak |q̈| (deg/s²)  — log scale")
    ax.set_title("Fix A — peak q̈ on throwing-arm coords (lower is better)")
    for i, (b, f) in enumerate(zip(qddb_peaks, qddf_peaks)):
        ax.text(i - w/2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, f, f"{f:.0f}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "audit_fixA_qdot_overview.png", dpi=120); plt.close(fig)
    print(f">>> wrote {OUT / 'audit_fixA_qdot_overview.png'}")

    # 3. BodyKinematics on Fix A mot, vs baseline + V3D
    print("Running BodyKinematics on Fix A mot...")
    bk = run_body_kinematics(
        PERSONAL_OSIM, FIXA_MOT, FIXA_BK,
        bodies=tuple(BODY_TO_V3D), name="audit_fixA",
    )
    base_bk_vel = OUT / "audit_h2_body_kin/audit_h2_BodyKinematics_vel_bodyLocal.sto"

    fig, ax = plt.subplots(figsize=(11, 5))
    bodies = list(BODY_TO_V3D)
    base_peaks, fixa_peaks, v3d_peaks = [], [], []
    for body in bodies:
        df_b = read_body_velocities(base_bk_vel, body)
        df_f = read_body_velocities(bk["vel"], body)
        om_b = np.column_stack([df_b["omega_x"], df_b["omega_y"], df_b["omega_z"]])
        om_f = np.column_stack([df_f["omega_x"], df_f["omega_y"], df_f["omega_z"]])
        base_peaks.append(float(np.max(np.linalg.norm(om_b, axis=1))))
        fixa_peaks.append(float(np.max(np.linalg.norm(om_f, axis=1))))
        v3d_peaks.append(load_v3d_angvel_peak(BODY_TO_V3D[body]))

    x = np.arange(len(bodies)); w = 0.27
    ax.bar(x - w, base_peaks, w, color="#888", label="baseline")
    ax.bar(x,     fixa_peaks, w, color="#c33", label="Fix A 12 Hz")
    ax.bar(x + w, v3d_peaks,  w, color="#39c", label="V3D ground truth")
    ax.set_xticks(x); ax.set_xticklabels(bodies)
    ax.set_ylabel("peak |ω| (deg/s)")
    ax.set_title("Fix A effect on body-frame ω peaks")
    for i, vs in enumerate(zip(base_peaks, fixa_peaks, v3d_peaks)):
        for j, v in enumerate(vs):
            ax.text(x[i] + (j-1)*w, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "audit_fixA_omega_overview.png", dpi=120); plt.close(fig)
    print(f">>> wrote {OUT / 'audit_fixA_omega_overview.png'}")

    # ratios summary
    summary_rows.append("")
    summary_rows.append(f"{'body':12s} {'baseline':>10s} {'fixA':>10s} {'V3D':>10s}  {'base/V3D':>10s} {'fixA/V3D':>10s}")
    summary_rows.append("-" * 72)
    for body, b, f, v in zip(bodies, base_peaks, fixa_peaks, v3d_peaks):
        summary_rows.append(
            f"{body:12s} {b:10.0f} {f:10.0f} {v:10.0f}  {b/v:10.2f} {f/v:10.2f}"
        )
    text = "\n".join(summary_rows)
    print("\n" + text)
    (OUT / "audit_fixA_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
