"""Reproducer for the M2 shoulder-kinetics blocker on `pose_filt_0.c3d`.

Tight-loop tool for the within-OpenSim shoulder fix experiments (axis-rotation,
BallJoint swap). Each experiment will iterate model edits and pipeline reruns
many times; this script puts the full Recipe D → ID → JR validation behind a
single command and emits a structured `peaks.json` so we can grep for
regressions instead of eyeballing `.sto` files.

Usage:
    uv run python scripts/repro_shoulder_kinetics.py \\
        --src-model data/models/recipes/LaiUhlrich2022_full_body.osim \\
        --out-dir out/repro_baseline

What it does, in order:
    1. Run `theia-osim-trial` (Recipe D only) on `pose_filt_0.c3d`.
    2. Print the personalized model's full CoordinateSet (name + motion type).
       Used as a smoke test that joint-type swaps preserve user-authored
       coordinate names through ScaleTool's XML round-trip.
    3. Run InverseDynamicsTool on Recipe D's `analytical.mot`.
    4. Build the residual CoordinateActuator XML from the personalized model.
    5. Run JointReaction with the default V3D-equivalent specs.
    6. Print the JR `.sto` header (column names) for visual inspection.
    7. Extract throwing-arm peaks (acromial_r + elbow_r reactions, plus
       throwing-arm ID GenForces) within [BR-50, BR+30] frames.
    8. Compare each peak to V3D-typical ranges from `docs/m2_kinetics_plan.md`
       and tag PASS / WEAK / FAIL.
    9. Emit `peaks.json` and `peaks.txt` summary.

Acceptance gates (per `final-plan.md` step 4):
    - PASS:  shoulder force in [700, 1200] N AND moment in [70, 110] N·m
    - WEAK:  shoulder force in [200, 2500] N (the experiment didn't make
             things worse, but didn't fully fix either)
    - FAIL:  outside the WEAK band — the experiment either broke something
             or did nothing
    Lower-body invariance check on hip/knee/ankle/lumbar ID peaks is in a
    separate section: warn if any moves >5% from the v3 baseline.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import opensim as osim

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_C3D = REPO_ROOT / "pose_filt_0.c3d"
DEFAULT_PROCDB = REPO_ROOT / "pose_filt_0_procdb.json"
DEFAULT_MDH = REPO_ROOT / "theia_model.mdh"

# V3D-typical pitching ranges from docs/m2_kinetics_plan.md:240
V3D_RANGES = {
    "shoulder_force_N":  (700.0, 1200.0),
    "shoulder_moment_Nm": (70.0, 110.0),
    "elbow_force_N":     (700.0, 1100.0),
    "elbow_moment_Nm":   (60.0, 100.0),
}
# "Experiment didn't make it worse" weak gate
WEAK_FORCE_MAX = 2500.0
WEAK_MOMENT_MAX = 250.0

# Lower-body baseline ID peaks within the throw window [BR-50, BR+30],
# captured from the pre-fix run at out/m2_phase5_check_v3 (smoothness-only
# smart-unwrap, current main-branch state). Used as an invariance check:
# model edits that target only the throwing arm should not perturb these
# by >5%. NOTE: some of these absolute values are themselves implausibly
# high (lumbar ~4000 N·m), suggesting lower-body ID may have its own
# conditioning issues. We're not investigating those here — only checking
# that experimental shoulder fixes don't disturb whatever the current
# lower-body computation produces.
LB_BASELINE_NM = {
    "hip_flexion_r_moment": 851.5,
    "hip_adduction_r_moment": 467.4,
    "hip_rotation_r_moment": 201.7,
    "knee_angle_r_moment": 275.4,
    "ankle_angle_r_moment": 12.4,
    "lumbar_extension_moment": 3575.4,
    "lumbar_bending_moment": 4262.5,
    "lumbar_rotation_moment": 783.2,
}
LB_TOLERANCE_FRAC = 0.05


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-model", type=Path,
                   default=REPO_ROOT / "data/models/recipes/LaiUhlrich2022_full_body.osim",
                   help="Source .osim model. Default: stock LaiUhlrich2022. Override "
                        "to test axis-rotation or BallJoint variants.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory. Will contain Recipe D outputs, ID/JR .sto, "
                        "peaks.json, peaks.txt.")
    p.add_argument("--c3d", type=Path, default=DEFAULT_C3D)
    p.add_argument("--procdb", type=Path, default=DEFAULT_PROCDB)
    p.add_argument("--mdh", type=Path, default=DEFAULT_MDH)
    p.add_argument("--skip-trial", action="store_true",
                   help="Skip the theia-osim-trial step (assume out-dir already has "
                        "a personalized model + recipe_d/analytical.mot).")
    return p.parse_args(argv)


def run_trial(args: argparse.Namespace, all_recipes_dir: Path) -> Path:
    """Invoke `theia-osim-trial` (Recipe D only) and return the personalized .osim."""
    cmd = [
        "uv", "run", "theia-osim-trial",
        "--c3d", str(args.c3d),
        "--out", str(all_recipes_dir),
        "--recipes", "d",
        "--src-model", str(args.src_model),
        "--v3d-procdb", str(args.procdb),
        "--mdh", str(args.mdh),
    ]
    print(f"=> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"theia-osim-trial failed (exit {result.returncode})")
    personalized = all_recipes_dir / "theia_pitching_personalized.osim"
    if not personalized.exists():
        raise SystemExit(f"Expected {personalized} after trial; not found.")
    return personalized


def dump_coordinate_set(model_path: Path) -> list[dict]:
    """Smoke test: print every coordinate's name + motion type. Returns the list."""
    model = osim.Model(str(model_path))
    cs = model.getCoordinateSet()
    rows = []
    print(f"\n=> CoordinateSet of {model_path.name} ({cs.getSize()} coords):")
    print(f"   {'idx':>3s}  {'name':30s}  motion_type")
    for i in range(cs.getSize()):
        c = cs.get(i)
        mt = int(c.getMotionType())
        mt_str = {1: "ROT", 2: "TRA", 3: "COUPLED"}.get(mt, f"?({mt})")
        rows.append({"idx": i, "name": c.getName(), "motion_type": mt_str})
        print(f"   {i:3d}  {c.getName():30s}  {mt_str}")
    # Quick joint enumeration too — useful for verifying joint-type swaps.
    js = model.getJointSet()
    print(f"\n=> JointSet ({js.getSize()} joints):")
    for i in range(js.getSize()):
        j = js.get(i)
        print(f"   {j.getName():18s}  {j.getConcreteClassName()}")
    return rows


def read_sto(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path) as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        if ln.strip() == "endheader":
            hdr = lines[i + 1].rstrip("\n").split("\t")
            data = lines[i + 2:]
            break
    else:
        raise RuntimeError(f"No endheader in {path}")
    arr = np.array([[float(x) for x in ln.split()] for ln in data if ln.strip()])
    return hdr, arr


def find_event_frame(procdb_path: Path, event: str) -> tuple[int, float]:
    """Look up V3D-recorded frame index + time for an event (BR or MER).

    Uses the repo's V3D loader so we get the same parsing behavior as the
    rest of the pipeline (handles the wrapped Visual3D/EVENT structure).
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from theia_osim.validation.load_v3d_json import load_v3d_procdb
    v3d = load_v3d_procdb(procdb_path)
    fkey = f"i_{event}"
    tkey = f"{event}_time"
    if fkey not in v3d.events or tkey not in v3d.events:
        raise RuntimeError(f"Event {event} not in {procdb_path}")
    return int(round(v3d.events[fkey])), float(v3d.events[tkey])


def peak_in_window(arr: np.ndarray, hdr: list[str], col_substr: str,
                   t_lo: float, t_hi: float) -> dict:
    """Find peaks of |v| for any column whose name contains col_substr,
    within [t_lo, t_hi]. Returns max across xyz components for forces and moments
    separately based on suffix."""
    t = arr[:, 0]
    mask = (t >= t_lo) & (t <= t_hi)
    out = {}
    for i, h in enumerate(hdr):
        if col_substr in h and (h.endswith("_fx") or h.endswith("_fy") or h.endswith("_fz")
                                or h.endswith("_mx") or h.endswith("_my") or h.endswith("_mz")):
            v = arr[mask, i]
            if v.size == 0:
                continue
            idx = int(np.abs(v).argmax())
            out[h] = {
                "peak_abs": float(abs(v[idx])),
                "peak_signed": float(v[idx]),
                "peak_t_s": float(t[mask][idx]),
            }
    return out


def vector_magnitude_peak(comp_dict: dict, prefix: str, kind: str) -> float | None:
    """Given the JR comp_dict, return the peak |F| (or |M|) magnitude.
    `kind` is "f" or "m"."""
    keys = [f"{prefix}_{kind}{a}" for a in "xyz"]
    if not all(k in comp_dict for k in keys):
        return None
    # Component-wise peaks may not be co-temporal; take Euclidean norm of the peaks.
    # This is a conservative upper bound — the actual instantaneous |vec| could
    # be lower. Sufficient for pass/fail tagging.
    return float(np.sqrt(sum(comp_dict[k]["peak_abs"] ** 2 for k in keys)))


def classify(value: float, target_lo: float, target_hi: float,
             weak_max: float) -> str:
    if target_lo <= value <= target_hi:
        return "PASS"
    if value <= weak_max:
        return "WEAK"
    return "FAIL"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_recipes_dir = args.out_dir / "all_recipes"

    # 1. Recipe D + personalization
    if not args.skip_trial:
        personalized_osim = run_trial(args, all_recipes_dir)
    else:
        personalized_osim = all_recipes_dir / "theia_pitching_personalized.osim"
        if not personalized_osim.exists():
            raise SystemExit(f"--skip-trial set but {personalized_osim} missing")
    analytical_mot = all_recipes_dir / "recipe_d" / "analytical.mot"
    if not analytical_mot.exists():
        raise SystemExit(f"Missing {analytical_mot}")

    # 2. CoordinateSet smoke test
    coord_rows = dump_coordinate_set(personalized_osim)

    # 3-5. ID + JR (import here so `--help` works without OpenSim)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from theia_osim.analysis.id import run_inverse_dynamics
    from theia_osim.analysis.jr import run_joint_reaction
    from theia_osim.analysis.residual_actuators import build_coord_actuator_force_set

    kinetics_dir = args.out_dir / "kinetics"
    kinetics_dir.mkdir(exist_ok=True)
    print(f"\n=> Building residual actuator XML")
    actuator_xml = build_coord_actuator_force_set(
        personalized_osim, kinetics_dir / "residual_actuators.xml")

    print(f"\n=> Running InverseDynamicsTool")
    id_sto = run_inverse_dynamics(
        personalized_osim, analytical_mot, kinetics_dir / "inverse_dynamics.sto")

    print(f"\n=> Running JointReaction")
    jr_sto = run_joint_reaction(
        personalized_osim, analytical_mot, id_sto=id_sto, out_dir=kinetics_dir,
        actuator_force_set_xml=actuator_xml)

    # 6. JR header smoke test
    jr_hdr, jr_arr = read_sto(jr_sto)
    print(f"\n=> JR .sto header ({len(jr_hdr)} columns):")
    for h in jr_hdr[:8]:
        print(f"   {h}")
    if len(jr_hdr) > 8:
        print(f"   ... and {len(jr_hdr) - 8} more")

    # 7. Peak extraction in throwing window [BR-50, BR+30] frames
    br_frame, br_time = find_event_frame(args.procdb, "BR")
    sample_rate = 300.0  # confirmed by trial metadata; could read from .mot if needed
    t_lo = (br_frame - 50) / sample_rate
    t_hi = (br_frame + 30) / sample_rate
    print(f"\n=> BR={br_frame} (t={br_time:.3f}s); peak window [{t_lo:.3f}, {t_hi:.3f}]s")

    shoulder_humerus = peak_in_window(jr_arr, jr_hdr,
                                      "acromial_r_on_humerus_r_in_humerus_r", t_lo, t_hi)
    shoulder_torso = peak_in_window(jr_arr, jr_hdr,
                                    "acromial_r_on_humerus_r_in_torso", t_lo, t_hi)
    elbow = peak_in_window(jr_arr, jr_hdr,
                           "elbow_r_on_ulna_r_in_ulna_r", t_lo, t_hi)

    # ID GenForces in the same window. Probe for either stock LaiUhlrich
    # coord names (arm_flex_r etc.) or the XZY-anatomical variant
    # (shoulder_abd_r etc.) so this works for both shoulder param choices.
    id_hdr, id_arr = read_sto(id_sto)
    id_peaks = {}
    mask = (id_arr[:, 0] >= t_lo) & (id_arr[:, 0] <= t_hi)
    candidate_cols = (
        "arm_flex_r_moment", "arm_add_r_moment", "arm_rot_r_moment",
        "shoulder_abd_r_moment", "shoulder_hzn_r_moment", "shoulder_int_rot_r_moment",
        "elbow_flex_r_moment",
    )
    for c in candidate_cols:
        if c in id_hdr:
            v = id_arr[mask, id_hdr.index(c)]
            id_peaks[c] = float(np.abs(v).max())

    # Lower-body invariance — restricted to throw window for parity with how
    # the baseline was captured.
    lb_check = {}
    lb_mask = mask
    for c, baseline in LB_BASELINE_NM.items():
        if c in id_hdr:
            v = id_arr[lb_mask, id_hdr.index(c)]
            peak = float(np.abs(v).max())
            delta_frac = (peak - baseline) / baseline if baseline else 0.0
            lb_check[c] = {
                "baseline": baseline, "peak": peak, "delta_frac": delta_frac,
                "ok": abs(delta_frac) <= LB_TOLERANCE_FRAC,
            }

    # 8. Classify
    sh_humerus_F = vector_magnitude_peak(shoulder_humerus,
                                         "acromial_r_on_humerus_r_in_humerus_r", "f")
    sh_humerus_M = vector_magnitude_peak(shoulder_humerus,
                                         "acromial_r_on_humerus_r_in_humerus_r", "m")
    elbow_F = vector_magnitude_peak(elbow, "elbow_r_on_ulna_r_in_ulna_r", "f")
    elbow_M = vector_magnitude_peak(elbow, "elbow_r_on_ulna_r_in_ulna_r", "m")

    gates = {
        "shoulder_force": (sh_humerus_F, *V3D_RANGES["shoulder_force_N"], WEAK_FORCE_MAX),
        "shoulder_moment": (sh_humerus_M, *V3D_RANGES["shoulder_moment_Nm"], WEAK_MOMENT_MAX),
        "elbow_force": (elbow_F, *V3D_RANGES["elbow_force_N"], WEAK_FORCE_MAX),
        "elbow_moment": (elbow_M, *V3D_RANGES["elbow_moment_Nm"], WEAK_MOMENT_MAX),
    }
    verdicts = {}
    for k, (val, lo, hi, weak) in gates.items():
        if val is None:
            verdicts[k] = "MISSING"
        else:
            verdicts[k] = classify(val, lo, hi, weak)

    # 9. Write outputs
    peaks = {
        "src_model": str(args.src_model),
        "out_dir": str(args.out_dir),
        "br_frame": br_frame, "br_time_s": br_time,
        "throw_window_s": [t_lo, t_hi],
        "joint_set": [],  # filled below for diff-friendly output
        "coord_set": coord_rows,
        "jr_columns": jr_hdr,
        "shoulder_in_humerus_frame": shoulder_humerus,
        "shoulder_in_torso_frame": shoulder_torso,
        "elbow_in_ulna_frame": elbow,
        "id_throwing_arm_peaks_Nm": id_peaks,
        "lower_body_invariance": lb_check,
        "magnitudes": {
            "shoulder_humerus_F_N": sh_humerus_F,
            "shoulder_humerus_M_Nm": sh_humerus_M,
            "elbow_F_N": elbow_F,
            "elbow_M_Nm": elbow_M,
        },
        "verdicts": verdicts,
        "v3d_target_ranges": V3D_RANGES,
    }
    # Re-load model to grab joint info
    model = osim.Model(str(personalized_osim))
    js = model.getJointSet()
    peaks["joint_set"] = [
        {"name": js.get(i).getName(), "type": js.get(i).getConcreteClassName()}
        for i in range(js.getSize())
    ]

    (args.out_dir / "peaks.json").write_text(json.dumps(peaks, indent=2))

    # Human-readable summary
    summary_lines = [
        f"# Reproducer summary",
        f"src_model: {args.src_model}",
        f"out_dir:   {args.out_dir}",
        f"BR frame:  {br_frame}  t={br_time:.3f}s",
        f"window:    [{t_lo:.3f}, {t_hi:.3f}]s",
        "",
        f"## Magnitudes (peak vector norm, conservative)",
        f"  shoulder F (humerus frame): {sh_humerus_F:7.1f} N    target {V3D_RANGES['shoulder_force_N']}  -> {verdicts['shoulder_force']}",
        f"  shoulder M (humerus frame): {sh_humerus_M:7.1f} N·m  target {V3D_RANGES['shoulder_moment_Nm']}  -> {verdicts['shoulder_moment']}",
        f"  elbow F    (ulna frame):    {elbow_F:7.1f} N    target {V3D_RANGES['elbow_force_N']}  -> {verdicts['elbow_force']}",
        f"  elbow M    (ulna frame):    {elbow_M:7.1f} N·m  target {V3D_RANGES['elbow_moment_Nm']}  -> {verdicts['elbow_moment']}",
        "",
        f"## ID throwing-arm peaks (N·m, in throw window)",
    ]
    for c, v in id_peaks.items():
        summary_lines.append(f"  {c:25s}  {v:8.1f}")
    summary_lines += ["", "## Lower-body invariance (vs out/m2_phase5_check_v3 baseline)"]
    for c, info in lb_check.items():
        flag = "OK" if info["ok"] else "DRIFT"
        summary_lines.append(
            f"  {c:25s}  base={info['baseline']:7.1f}  now={info['peak']:7.1f}  "
            f"Δ={info['delta_frac']*100:+5.1f}%  [{flag}]")
    overall = "PASS" if all(v == "PASS" for v in verdicts.values()) else (
        "WEAK" if all(v in ("PASS", "WEAK") for v in verdicts.values()) else "FAIL")
    summary_lines += ["", f"## OVERALL: {overall}"]

    summary = "\n".join(summary_lines)
    (args.out_dir / "peaks.txt").write_text(summary + "\n")
    print(f"\n{summary}")
    print(f"\n=> wrote: {args.out_dir / 'peaks.json'}")
    print(f"=> wrote: {args.out_dir / 'peaks.txt'}")
    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
