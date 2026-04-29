"""Compare our OpenSim BodyKinematics output to V3D's YABIN time-series export.

V3D's procdb time series are windowed (PrePKH → PostBR per ref doc §14) at the
trial sample rate. Our BodyKinematics output spans the full trial. We align by
matching V3D's TIME array (which is t=0 at PrePKH ≈ start of windowed export)
to our IK output time, find the common interval, and compare frame-by-frame.

For pelvis specifically:
- V3D YABIN.PELVIS_ANGULAR_VELOCITY is resolved in the RPV (Theia pelvis) frame.
- Our OpenSim ω is resolved in the LaiUhlrich pelvis body frame after Rx(-90°)
  axis swap. The two body frames may differ by a constant rotation — that's
  what we're diagnosing here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.body_kin import _read_sto
from ..kinematics_postprocess.body_frame_corrections import convert_body_frame_signal
from .load_v3d_json import V3DTrial, get_time_array, get_yabin_xyz


@dataclass(frozen=True)
class ComparisonReport:
    signal: str
    v3d_peak_xyz: tuple[float, float, float]
    osim_peak_xyz: tuple[float, float, float]
    rmse_xyz: tuple[float, float, float]
    mae_xyz: tuple[float, float, float]
    n_aligned_frames: int
    overlay_png: Path
    raw_csv: Path


def _read_pelvis_omega_local(sto_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, omega_xyz) from a BodyKinematics _vel_bodyLocal sto."""
    df = _read_sto(Path(sto_path))
    t = df["time"].to_numpy()
    omega = np.column_stack([df["pelvis_Ox"], df["pelvis_Oy"], df["pelvis_Oz"]])
    return t, omega


def _align_to_common_time(
    t1: np.ndarray, x1: np.ndarray, t2: np.ndarray, x2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate both onto a common time grid (the denser of the two).

    Returns (t_common, x1_resampled, x2_resampled). The grid is restricted to
    `[max(t1[0], t2[0]), min(t1[-1], t2[-1])]`.
    """
    t_lo = max(t1[0], t2[0])
    t_hi = min(t1[-1], t2[-1])
    # Use the higher-rate grid in the common interval.
    dt1 = float(np.median(np.diff(t1)))
    dt2 = float(np.median(np.diff(t2)))
    dt = min(dt1, dt2)
    n = int((t_hi - t_lo) / dt) + 1
    t = np.linspace(t_lo, t_lo + (n - 1) * dt, n)

    def _interp_columns(t_src: np.ndarray, x_src: np.ndarray) -> np.ndarray:
        if x_src.ndim == 1:
            return np.interp(t, t_src, x_src)
        return np.column_stack([np.interp(t, t_src, x_src[:, i]) for i in range(x_src.shape[1])])

    return t, _interp_columns(t1, x1), _interp_columns(t2, x2)


def compare_pelvis_omega(
    v3d_trial: V3DTrial,
    osim_sto: Path | str,
    *,
    out_dir: Path | str,
    label: str = "recipe_a",
    apply_frame_correction: bool = True,
) -> ComparisonReport:
    """Plot V3D vs our OpenSim pelvis angular velocity overlay; report errors.

    With `apply_frame_correction=True` (default), rotates OpenSim ω from the
    OpenSim pelvis body frame into the V3D RPV frame before comparison
    (180° about Z).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # V3D side
    v3d_x, v3d_y, v3d_z = get_yabin_xyz(v3d_trial, "PELVIS_ANGULAR_VELOCITY")
    v3d_t = get_time_array(v3d_trial)
    n = min(len(v3d_t), len(v3d_x), len(v3d_y), len(v3d_z))
    v3d_t = v3d_t[:n]
    v3d_xyz = np.column_stack([v3d_x[:n], v3d_y[:n], v3d_z[:n]])

    # Our side
    osim_t, osim_xyz = _read_pelvis_omega_local(osim_sto)
    if apply_frame_correction:
        osim_xyz = convert_body_frame_signal(osim_xyz, "pelvis")

    # Align onto a common time grid
    t_c, v3d_r, osim_r = _align_to_common_time(v3d_t, v3d_xyz, osim_t, osim_xyz)

    # Per-component error metrics
    diff = osim_r - v3d_r
    rmse = np.sqrt(np.mean(diff**2, axis=0))
    mae = np.mean(np.abs(diff), axis=0)

    v3d_peak = tuple(float(np.max(np.abs(v3d_r[:, i]))) for i in range(3))
    osim_peak = tuple(float(np.max(np.abs(osim_r[:, i]))) for i in range(3))

    # Plot
    overlay_png = out_dir / f"v3d_vs_{label}_pelvis_omega.png"
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    comp_labels = ["X (anterior)", "Y (lateral)", "Z (long axis — pelvis rotation)"]
    for i, comp in enumerate(comp_labels):
        axes[i].plot(t_c, v3d_r[:, i], label="V3D", color="tab:orange", linewidth=2)
        axes[i].plot(
            t_c, osim_r[:, i],
            label=f"OpenSim ({label})", color="tab:blue", alpha=0.85, linewidth=1.5,
        )
        axes[i].set_ylabel(f"ω_{comp} (deg/s)")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(0, color="black", linewidth=0.5)

    # Mark events on the bottom axis
    for ev in ["PKH_time", "FP_time", "MER_time", "BR_time"]:
        if ev in v3d_trial.events:
            t_ev = v3d_trial.events[ev]
            for ax in axes:
                ax.axvline(t_ev, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            axes[-1].text(
                t_ev, axes[-1].get_ylim()[0], ev.replace("_time", ""),
                rotation=90, fontsize=8, color="gray", verticalalignment="bottom",
            )

    axes[-1].set_xlabel("time (s) — V3D-windowed (PrePKH → PostBR)")
    fig.suptitle(
        f"Pelvis angular velocity: V3D YABIN vs OpenSim BodyKinematics ({label})  "
        f"[{v3d_trial.path.name}]"
    )
    fig.tight_layout()
    fig.savefig(overlay_png, dpi=120)
    plt.close(fig)

    # Raw CSV for further analysis
    raw_csv = out_dir / f"v3d_vs_{label}_pelvis_omega.csv"
    with open(raw_csv, "w") as f:
        f.write("time,v3d_x,v3d_y,v3d_z,osim_x,osim_y,osim_z\n")
        for k in range(len(t_c)):
            f.write(
                f"{t_c[k]:.6f},"
                f"{v3d_r[k, 0]:.6f},{v3d_r[k, 1]:.6f},{v3d_r[k, 2]:.6f},"
                f"{osim_r[k, 0]:.6f},{osim_r[k, 1]:.6f},{osim_r[k, 2]:.6f}\n"
            )

    return ComparisonReport(
        signal="PELVIS_ANGULAR_VELOCITY",
        v3d_peak_xyz=v3d_peak,
        osim_peak_xyz=osim_peak,
        rmse_xyz=tuple(float(x) for x in rmse),
        mae_xyz=tuple(float(x) for x in mae),
        n_aligned_frames=len(t_c),
        overlay_png=overlay_png,
        raw_csv=raw_csv,
    )


def search_axis_permutation(
    v3d_trial: V3DTrial, osim_sto: Path | str
) -> dict[str, dict[str, float]]:
    """Brute-force best (permutation × sign) mapping from OpenSim → V3D axes.

    Useful when initial RMSE is high and we suspect a frame-orientation mismatch
    (e.g. OpenSim pelvis body frame differs from Theia's RPV by a constant
    rotation, so axes might be swapped or sign-flipped). For each of the 48
    signed permutations of (x, y, z), reports per-component RMSE.

    Returns dict mapping permutation label → {rmse_x, rmse_y, rmse_z, total}.
    """
    import itertools

    v3d_x, v3d_y, v3d_z = get_yabin_xyz(v3d_trial, "PELVIS_ANGULAR_VELOCITY")
    v3d_t = get_time_array(v3d_trial)
    n = min(len(v3d_t), len(v3d_x), len(v3d_y), len(v3d_z))
    v3d_t = v3d_t[:n]
    v3d_xyz = np.column_stack([v3d_x[:n], v3d_y[:n], v3d_z[:n]])

    osim_t, osim_xyz = _read_pelvis_omega_local(osim_sto)
    t_c, v3d_r, osim_r = _align_to_common_time(v3d_t, v3d_xyz, osim_t, osim_xyz)

    results: dict[str, dict[str, float]] = {}
    axes = ["x", "y", "z"]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            mapped = np.column_stack(
                [signs[i] * osim_r[:, perm[i]] for i in range(3)]
            )
            diff = mapped - v3d_r
            rmse = np.sqrt(np.mean(diff**2, axis=0))
            label = "".join(
                f"{'+' if signs[i] > 0 else '-'}{axes[perm[i]]}" for i in range(3)
            )
            results[label] = {
                "rmse_x": float(rmse[0]),
                "rmse_y": float(rmse[1]),
                "rmse_z": float(rmse[2]),
                "total": float(np.sqrt(np.sum(rmse**2))),
            }
    return results
