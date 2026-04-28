"""One-trial CLI: read Theia .c3d → IK → BodyKinematics → plot pelvis ω.

Usage:
    theia-osim-trial --c3d pose_filt_0.c3d --config configs/default.yaml \\
                     --out out/m1 --recipes a,c
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.body_kin import read_body_velocities, run_body_kinematics
from ..analysis.ik import run_imu_ik, run_marker_ik
from ..c3d_io.reader import read_theia_c3d
from ..config import Config, load_config
from ..import_pipeline.landmarks import load_marker_catalog
from ..import_pipeline.recipe_a_trc import write_recipe_a_trc
from ..import_pipeline.recipe_c_sto import write_recipe_c_sto
from ..kinematics_postprocess.filter import lowpass_filtfilt
from ..model_build.add_markers import add_virtual_markers
from ..validation.compare import compare_pelvis_omega
from ..validation.load_v3d_json import load_v3d_procdb


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c3d", type=Path, required=True, help="Path to Theia .c3d file")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config (default: configs/default.yaml)",
    )
    p.add_argument(
        "--markers",
        type=Path,
        default=Path("configs/markers.yaml"),
        help="Path to marker catalog (default: configs/markers.yaml)",
    )
    p.add_argument(
        "--src-model",
        type=Path,
        default=Path("data/models/recipes/LaiUhlrich2022_full_body.osim"),
        help="Source OpenSim model (default: LaiUhlrich2022_full_body.osim)",
    )
    p.add_argument("--out", type=Path, default=Path("out/m1"), help="Output directory")
    p.add_argument(
        "--recipes",
        type=str,
        default="a,c",
        help="Comma-separated list of recipes to run: a, c (default: a,c)",
    )
    p.add_argument("--side", type=str, choices=["auto", "R", "L"], default=None)
    p.add_argument(
        "--v3d-procdb",
        type=Path,
        default=None,
        help="Optional V3D procdb JSON for ground-truth comparison",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config) if args.config.exists() else Config()
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    recipes = tuple(r.strip().lower() for r in args.recipes.split(",") if r.strip())

    print(f"=> trial: {args.c3d}")
    print(f"   config: {args.config if args.config.exists() else '(defaults)'}")
    print(f"   out: {out_root}")
    print(f"   recipes: {recipes}")

    # 1. Load + slope correct
    trial = read_theia_c3d(args.c3d, apply_vlb=cfg.slope.vlb_4x4)
    print(
        f"   loaded {trial.n_frames} frames @ {trial.sample_rate_hz}Hz "
        f"(Theia v{trial.meta.theia_version}, filt {trial.meta.filt_freq_hz}Hz)"
    )

    # 2. Build markered model
    markered_osim = out_root / "theia_pitching.osim"
    add_virtual_markers(args.src_model, args.markers, markered_osim)
    print(f"   markered model: {markered_osim.relative_to(out_root.parent.parent)}")

    summary: dict = {
        "trial": str(args.c3d),
        "n_frames": trial.n_frames,
        "sample_rate_hz": trial.sample_rate_hz,
        "theia_version": list(trial.meta.theia_version),
        "filt_freq_hz": trial.meta.filt_freq_hz,
        "ik_lowpass_hz": cfg.filters.ik_lowpass_hz,
        "recipes": {},
    }

    pelvis_omega: dict[str, np.ndarray] = {}
    pelvis_time: dict[str, np.ndarray] = {}

    # 3. Recipe A — marker IK
    if "a" in recipes:
        print(f"\n=> Recipe A (marker IK)")
        cat = load_marker_catalog(args.markers)
        trc_path = out_root / "recipe_a" / "markers.trc"
        write_recipe_a_trc(trial.transforms, cat, trc_path, trial.sample_rate_hz)
        mot_path = out_root / "recipe_a" / "ik_solution.mot"
        run_marker_ik(markered_osim, trc_path, mot_path)
        bk = run_body_kinematics(
            markered_osim, mot_path, out_root / "recipe_a" / "body_kin",
            bodies=("pelvis",),
        )
        df = read_body_velocities(bk["vel"], "pelvis")
        omega = np.column_stack([df["omega_x"], df["omega_y"], df["omega_z"]])
        # Apply 20 Hz bidirectional Butterworth (matches V3D)
        omega_filt = lowpass_filtfilt(
            omega,
            cutoff_hz=cfg.filters.ik_lowpass_hz,
            sample_rate_hz=trial.sample_rate_hz,
            order=cfg.filters.butterworth_order,
        )
        pelvis_omega["a"] = omega_filt
        pelvis_time["a"] = df["time"].to_numpy()
        peak_y = float(np.abs(omega_filt[:, 1]).max())
        peak_y_idx = int(np.abs(omega_filt[:, 1]).argmax())
        peak_y_t = float(df["time"].iloc[peak_y_idx])
        print(f"   peak |pelvis ω_y| = {peak_y:.1f} deg/s at t={peak_y_t:.3f}s")
        summary["recipes"]["a"] = {
            "trc": str(trc_path),
            "ik_mot": str(mot_path),
            "body_kin_vel": str(bk["vel"]),
            "peak_pelvis_omega_y_dps": peak_y,
            "peak_pelvis_omega_y_time_s": peak_y_t,
        }

    # 4. Recipe C — IMU IK
    if "c" in recipes:
        print(f"\n=> Recipe C (IMU IK)")
        sto_path = out_root / "recipe_c" / "orientations.sto"
        write_recipe_c_sto(trial.transforms, sto_path, trial.sample_rate_hz)
        mot_path = out_root / "recipe_c" / "imu_ik_solution.mot"
        run_imu_ik(markered_osim, sto_path, mot_path)
        # IMU IK output filename is `ik_<sto_stem>.mot`
        actual_mot = mot_path.parent / f"ik_{sto_path.stem}.mot"
        if not actual_mot.exists():
            actual_mot = mot_path
        bk = run_body_kinematics(
            markered_osim, actual_mot, out_root / "recipe_c" / "body_kin",
            bodies=("pelvis",),
        )
        df = read_body_velocities(bk["vel"], "pelvis")
        omega = np.column_stack([df["omega_x"], df["omega_y"], df["omega_z"]])
        omega_filt = lowpass_filtfilt(
            omega,
            cutoff_hz=cfg.filters.ik_lowpass_hz,
            sample_rate_hz=trial.sample_rate_hz,
            order=cfg.filters.butterworth_order,
        )
        pelvis_omega["c"] = omega_filt
        pelvis_time["c"] = df["time"].to_numpy()
        peak_y = float(np.abs(omega_filt[:, 1]).max())
        peak_y_idx = int(np.abs(omega_filt[:, 1]).argmax())
        peak_y_t = float(df["time"].iloc[peak_y_idx])
        print(f"   peak |pelvis ω_y| = {peak_y:.1f} deg/s at t={peak_y_t:.3f}s")
        summary["recipes"]["c"] = {
            "orientations_sto": str(sto_path),
            "ik_mot": str(actual_mot),
            "body_kin_vel": str(bk["vel"]),
            "peak_pelvis_omega_y_dps": peak_y,
            "peak_pelvis_omega_y_time_s": peak_y_t,
        }

    # 5. Cross-validate A vs C if both ran
    if "a" in pelvis_omega and "c" in pelvis_omega:
        # align lengths (IMU IK and marker IK should produce identical n_frames if input is the same)
        n = min(pelvis_omega["a"].shape[0], pelvis_omega["c"].shape[0])
        diff = pelvis_omega["a"][:n] - pelvis_omega["c"][:n]
        rmse = float(np.sqrt(np.mean(diff**2, axis=0)).mean())
        print(f"\n=> Recipe A vs C cross-validation")
        print(f"   pelvis ω RMSE (deg/s, mean over xyz): {rmse:.2f}")
        summary["a_vs_c_rmse_dps"] = rmse

    # 6. V3D comparison if reference available
    if args.v3d_procdb is not None and args.v3d_procdb.exists():
        print(f"\n=> V3D comparison ({args.v3d_procdb.name})")
        v3d = load_v3d_procdb(args.v3d_procdb)
        print(
            f"   V3D info: HAND={v3d.info.get('HAND')}, "
            f"PITCH_VELO={v3d.info.get('PITCH_VELO')}, QA={v3d.info.get('QA')}"
        )
        v3d_peaks = v3d.metrics.get("PELVIS_ANGULAR_VELOCITY_MAX", {})
        if v3d_peaks:
            print(f"   V3D PELVIS_ANGULAR_VELOCITY_MAX: {v3d_peaks}")
        summary["v3d"] = {
            "info": v3d.info,
            "events": v3d.events,
            "pelvis_angular_velocity_max": v3d_peaks,
            "comparisons": {},
        }
        for r in pelvis_omega:
            sto = (
                out_root / f"recipe_{r}" / "body_kin"
                / "body_kinematics_BodyKinematics_vel_bodyLocal.sto"
            )
            if not sto.exists():
                continue
            rep = compare_pelvis_omega(v3d, sto, out_dir=out_root, label=f"recipe_{r}")
            print(f"   recipe {r.upper()} vs V3D — peaks XYZ:")
            print(f"     V3D    : {rep.v3d_peak_xyz}")
            print(f"     OpenSim: {rep.osim_peak_xyz}")
            print(f"     RMSE   : x={rep.rmse_xyz[0]:.1f}  y={rep.rmse_xyz[1]:.1f}  z={rep.rmse_xyz[2]:.1f} deg/s")
            print(f"     overlay: {rep.overlay_png.name}")
            summary["v3d"]["comparisons"][r] = {
                "v3d_peak_xyz": rep.v3d_peak_xyz,
                "osim_peak_xyz": rep.osim_peak_xyz,
                "rmse_xyz": rep.rmse_xyz,
                "mae_xyz": rep.mae_xyz,
                "n_aligned_frames": rep.n_aligned_frames,
                "overlay_png": str(rep.overlay_png),
                "raw_csv": str(rep.raw_csv),
            }

    # 7. Plot
    plot_path = out_root / "pelvis_angular_velocity.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axis_labels = ["ω_x (sagittal-ish)", "ω_y (long axis = pelvis rotation)", "ω_z (lateral-ish)"]
    for i, lbl in enumerate(axis_labels):
        for r in pelvis_omega:
            axes[i].plot(
                pelvis_time[r], pelvis_omega[r][:, i],
                label=f"Recipe {r.upper()}", alpha=0.85,
            )
        axes[i].set_ylabel(lbl)
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"Pelvis angular velocity in body frame  ({Path(args.c3d).name}, "
        f"{trial.sample_rate_hz}Hz, {cfg.filters.ik_lowpass_hz}Hz lowpass)"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"\n=> plot saved: {plot_path}")

    # 8. Write summary
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"=> summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
