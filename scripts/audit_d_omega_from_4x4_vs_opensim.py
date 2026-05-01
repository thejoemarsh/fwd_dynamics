"""Audit D1: compute body-frame ω directly from segment 4×4 trajectories
(Theia c3d) and compare to OpenSim's BodyKinematics ω for the same trial.

If the two match, the residual ulna/hand inflation in H2 isn't from
OpenSim's coord-chain propagating gimbal-lock artifacts — it's
definitional (V3D's RHA is independent of RFA, ours has welded wrist).

Mapping (Theia c3d segment ↔ OpenSim body):
   r_uarm  ↔ humerus_r
   r_larm  ↔ ulna_r
   r_hand  ↔ hand_r
   torso   ↔ torso
   pelvis  ↔ pelvis
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import yaml

REPO = Path("/home/yabin/code/fwd_dynamics")
sys.path.insert(0, str(REPO / "src"))
from theia_osim.c3d_io.reader import read_theia_c3d  # noqa: E402
from theia_osim.analysis.body_kin import read_body_velocities  # noqa: E402

C3D = REPO / "pose_filt_0.c3d"
BK_VEL = REPO / "out/audit_h2_body_kin/audit_h2_BodyKinematics_vel_bodyLocal.sto"

THEIA_TO_OSIM = {
    "r_uarm": "humerus_r",
    "r_larm": "ulna_r",
    "r_hand": "hand_r",
    "torso":  "torso",
    "pelvis": "pelvis",
}


def omega_body_from_R_traj(R_traj: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular velocity from a (T, 3, 3) rotation-matrix trajectory.

    Uses the central-difference formula:
        Ṙ ≈ (R(t+dt) - R(t-dt)) / (2dt)
        Ω_body = R^T · Ṙ          (skew-symmetric, body frame)
        ω_body = vec3 of Ω_body
    """
    T = R_traj.shape[0]
    R_dot = np.zeros_like(R_traj)
    R_dot[1:-1] = (R_traj[2:] - R_traj[:-2]) / (2 * dt)
    R_dot[0]    = (R_traj[1]  - R_traj[0])   / dt
    R_dot[-1]   = (R_traj[-1] - R_traj[-2])  / dt
    Omega_body = np.einsum("tji,tjk->tik", R_traj, R_dot)  # R^T · Ṙ
    omega = np.stack([
        Omega_body[:, 2, 1], Omega_body[:, 0, 2], Omega_body[:, 1, 0],
    ], axis=1)
    return np.degrees(omega)  # deg/s for direct comparison with BK output


def main() -> None:
    cfg = yaml.safe_load(open(REPO / "configs/default.yaml"))
    vlb = np.array(cfg["slope"]["vlb_4x4"], dtype=np.float64)
    trial = read_theia_c3d(C3D, apply_vlb=vlb)
    dt = 1.0 / trial.sample_rate_hz

    print(f"\n=== D1: body-frame ω peak comparison ===")
    print(f"{'segment':<12}{'4x4_peak':>11}{'osim_peak':>11}{'V3D_peak':>10}{'4x4/osim':>11}{'4x4/V3D':>10}")
    print("-" * 70)

    # V3D peaks for reference (from H2 audit summary).
    V3D_PEAKS = {
        "humerus_r": 4166.1,
        "ulna_r":    2859.2,
        "hand_r":    2141.3,
        "torso":     1063.3,
        "pelvis":     615.5,
    }

    rows = []
    for theia_seg, osim_body in THEIA_TO_OSIM.items():
        if theia_seg not in trial.transforms:
            print(f"  [skip] {theia_seg} not in c3d")
            continue
        # Take rotation part only (3x3) of each 4x4 frame.
        T_arr = trial.transforms[theia_seg]    # (n_frames, 4, 4)
        R_arr = T_arr[:, :3, :3]
        omega_4x4 = omega_body_from_R_traj(R_arr, dt)
        peak_4x4 = float(np.linalg.norm(omega_4x4, axis=1).max())

        # Pull OpenSim BodyKinematics ω for same body
        try:
            df = read_body_velocities(BK_VEL, osim_body)
            om_osim = np.column_stack([df["omega_x"], df["omega_y"], df["omega_z"]])
            peak_osim = float(np.linalg.norm(om_osim, axis=1).max())
        except Exception as e:
            peak_osim = float("nan")

        v3d = V3D_PEAKS.get(osim_body, float("nan"))
        rows.append((osim_body, peak_4x4, peak_osim, v3d))
        ratio_4x4_osim = peak_4x4 / peak_osim if peak_osim else float("nan")
        ratio_4x4_v3d = peak_4x4 / v3d if v3d else float("nan")
        print(f"{osim_body:<12}{peak_4x4:>11.1f}{peak_osim:>11.1f}{v3d:>10.1f}{ratio_4x4_osim:>10.2f}x{ratio_4x4_v3d:>9.2f}x")

    print("\n=== D1 interpretation guide ===")
    print("4x4/osim ratio close to 1.00x  → coord-chain isn't introducing artifacts;")
    print("                                  remaining V3D mismatch is definitional.")
    print("4x4/osim significantly < 1     → OpenSim chain inflates ω; Fix 1 (bypass)")
    print("                                  is justified.")


if __name__ == "__main__":
    main()
