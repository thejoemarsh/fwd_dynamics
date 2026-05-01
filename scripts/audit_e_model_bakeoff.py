"""Audit E: model-topology bake-off using c3d-driven Newton-Euler.

Tests whether different .osim model topologies produce different joint
reactions on the same trial when fed through the same Newton-Euler pipeline.

Pipeline per candidate:
  1. Personalize the model on pose_filt_0.c3d (Theia INERTIA_* anthropometrics).
  2. Run compute_throwing_arm_reactions_from_c3d (no Recipe D, no OpenSim
     coord chain — kinematics come straight from c3d 4×4s, only the model's
     mass/COM/inertia and joint topology are consulted).
  3. Extract peak F, M in throw window, compare against V3D.

Candidates:
  laiuhlrich_welded   — current baseline; radius_hand_r is WeldJoint.
  rajagopal_movable   — RajagopalLaiUhlrich2023 with UniversalJoint wrist.
  laiuhlrich_movable  — Lai-Uhlrich, but force movable-wrist mode in NE
                        (treat hand and ulna as separate segments even though
                        the model welds them — uses same inertial properties
                        as baseline, only the recursion changes).

Outputs:
  out/audit_e_summary.txt   — comparison table.
  out/audit_e/<name>/peaks.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path("/home/yabin/code/fwd_dynamics")
sys.path.insert(0, str(REPO / "src"))
from theia_osim.analysis.segment_reactions import (  # noqa: E402
    compute_throwing_arm_reactions_from_c3d,
)
sys.path.insert(0, str(REPO / "scripts"))
import repro_shoulder_kinetics as repro  # noqa: E402

C3D = REPO / "pose_filt_0.c3d"

V3D = {
    "shoulder_F": 1090.0,
    "shoulder_M":  151.0,
    "elbow_F":    1142.0,
    "elbow_M":     140.0,
}

CANDIDATES = [
    {
        "name": "laiuhlrich_welded",
        "src_model": REPO / "data/models/recipes/LaiUhlrich2022_full_body.osim",
        "wrist_mode": "auto",   # will autodetect WeldJoint → welded
    },
    {
        "name": "rajagopal_movable",
        "src_model": REPO / "data/models/recipes/RajagopalLaiUhlrich2023.osim",
        "wrist_mode": "auto",   # will autodetect UniversalJoint → movable
    },
    {
        "name": "laiuhlrich_force_movable",
        "src_model": REPO / "data/models/recipes/LaiUhlrich2022_full_body.osim",
        "wrist_mode": "movable",  # override the welded autodetect; isolates
                                  # the topology change from the model change.
    },
]


def peak_mag(arr3, t, t_lo, t_hi):
    mask = (t >= t_lo) & (t <= t_hi)
    sub = arr3[mask]
    if sub.size == 0:
        return float("nan"), float("nan")
    mags = np.linalg.norm(sub, axis=1)
    if not np.all(np.isfinite(mags)):
        return float("nan"), float("nan")
    idx = int(np.argmax(mags))
    return float(mags[idx]), float(t[mask][idx])


def get_personalized(name: str, src_model: Path) -> Path:
    """Run a Recipe D trial just to get the personalized .osim. Cache by name."""
    out_root = REPO / f"out/audit_e/{name}"
    personalized = out_root / "all_recipes/theia_pitching_personalized.osim"
    if personalized.exists():
        print(f"  [skip personalize] cached at {personalized}")
        return personalized

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"  [personalize] {src_model.name} → {personalized}")
    # Reuse the existing repro driver to do personalization (Recipe D run).
    args = repro.parse_args([
        "--src-model", str(src_model),
        "--out-dir", str(out_root),
    ])
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_recipes_dir = args.out_dir / "all_recipes"
    repro.run_trial(args, all_recipes_dir)
    if not personalized.exists():
        raise RuntimeError(f"personalize failed; expected {personalized}")
    return personalized


def main() -> None:
    br_t = 1.593
    t_lo, t_hi = br_t - 50/300.0, br_t + 30/300.0

    rows = []
    for cand in CANDIDATES:
        print(f"\n{'='*72}\nCandidate: {cand['name']}\n{'='*72}")
        personalized = get_personalized(cand["name"], cand["src_model"])
        result = compute_throwing_arm_reactions_from_c3d(
            C3D, personalized, side="r", wrist_mode=cand["wrist_mode"],
        )
        times = result["times"]

        peaks = {}
        for key, lbl in [
            ("shoulder_F_humerus",  "shoulder_F"),
            ("shoulder_M_humerus",  "shoulder_M"),
            ("elbow_F_ulna_frame",  "elbow_F"),
            ("elbow_M_ulna_frame",  "elbow_M"),
        ]:
            mag, when = peak_mag(result[key], times, t_lo, t_hi)
            peaks[lbl] = {"peak": mag, "t_s": when}
        peaks["wrist_mode_used"] = result["wrist_mode_used"]

        out_dir = REPO / f"out/audit_e/{cand['name']}"
        (out_dir / "peaks.json").write_text(json.dumps(peaks, indent=2))
        np.savez(out_dir / "reactions.npz", times=times,
                 **{k: v for k, v in result.items() if k not in ("times", "wrist_mode_used")})
        print(f"  wrist_mode = {peaks['wrist_mode_used']}")
        print(f"  shoulder_F = {peaks['shoulder_F']['peak']:.1f} N "
              f"@ t={peaks['shoulder_F']['t_s']:.3f}s")
        print(f"  shoulder_M = {peaks['shoulder_M']['peak']:.1f} N·m "
              f"@ t={peaks['shoulder_M']['t_s']:.3f}s")
        print(f"  elbow_F    = {peaks['elbow_F']['peak']:.1f} N "
              f"@ t={peaks['elbow_F']['t_s']:.3f}s")
        print(f"  elbow_M    = {peaks['elbow_M']['peak']:.1f} N·m "
              f"@ t={peaks['elbow_M']['t_s']:.3f}s")
        rows.append((cand["name"], peaks))

    # Comparison table.
    lines = ["", "Model bake-off (c3d-driven Newton-Euler) vs V3D:", ""]
    header = (f"  {'candidate':<28}{'wrist':<10}"
              f"{'sF':>10}{'sM':>10}{'eF':>10}{'eM':>10}"
              f"{'sF/V3D':>10}{'eF/V3D':>10}")
    lines.append(header)
    lines.append("-" * len(header))
    for name, p in rows:
        sf = p["shoulder_F"]["peak"]; sm = p["shoulder_M"]["peak"]
        ef = p["elbow_F"]["peak"];    em = p["elbow_M"]["peak"]
        lines.append(
            f"  {name:<28}{p['wrist_mode_used']:<10}"
            f"{sf:>10.0f}{sm:>10.0f}{ef:>10.0f}{em:>10.0f}"
            f"{sf/V3D['shoulder_F']:>9.2f}x{ef/V3D['elbow_F']:>9.2f}x"
        )
    lines.append(f"  {'V3D':<28}{'-':<10}"
                 f"{V3D['shoulder_F']:>10.0f}{V3D['shoulder_M']:>10.0f}"
                 f"{V3D['elbow_F']:>10.0f}{V3D['elbow_M']:>10.0f}"
                 f"{1.00:>9.2f}x{1.00:>9.2f}x")
    out = "\n".join(lines)
    print(out)
    (REPO / "out/audit_e_summary.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
