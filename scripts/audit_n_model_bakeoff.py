"""Audit N: model topology bake-off — kinematics AND kinetics.

For each candidate .osim model:
  1. Personalize on pose_filt_0.c3d via repro.run_trial (cached in
     out/audit_n/<name>/all_recipes/).
  2. Score kinematics: per-axis RMS error of throw-window joint angles
     vs V3D's procdb (SHOULDER_ANGLE, ELBOW_ANGLE, RT_WRIST_ANGLE).
  3. Score kinetics: peak shoulder F/M, elbow F/M from
     compute_throwing_arm_reactions_from_c3d with the production
     18 Hz + 10 Hz cascaded filter and Dempster COMs.

Outputs:
  out/audit_n_summary.txt      single comparison table
  out/audit_n_kinematics.png   joint-angle overlays per model vs V3D
  out/audit_n_kinetics.png     F/M peak comparison
  out/audit_n/<name>/peaks.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import opensim as osim

REPO = Path("/home/yabin/code/fwd_dynamics")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from theia_osim.analysis.segment_reactions import (  # noqa: E402
    compute_throwing_arm_reactions_from_c3d,
)
import repro_shoulder_kinetics as repro  # noqa: E402

C3D = REPO / "pose_filt_0.c3d"
PROCDB = REPO / "pose_filt_0_procdb.json"

CANDIDATES = [
    {
        "name": "laiuhlrich",
        "src_model": REPO / "data/models/recipes/LaiUhlrich2022_full_body.osim",
        "wrist_class": "welded",
    },
    {
        "name": "rajagopal_lai_2023",
        "src_model": REPO / "data/models/recipes/RajagopalLaiUhlrich2023.osim",
        "wrist_class": "movable",
    },
    {
        "name": "rajagopal_2016",
        "src_model": REPO / "data/models/recipes/Rajagopal2016.osim",
        "wrist_class": "unknown",
    },
]

DE_LEVA = {"humerus_r": 0.5772, "ulna_r": 0.4574, "hand_r": 0.7900}
DEMPSTER = {"humerus_r": 0.436,  "ulna_r": 0.430,  "hand_r": 0.506}
V3D = {"shoulder_F": 1090., "shoulder_M": 151., "elbow_F": 1142., "elbow_M": 140.}

BR_TIME = 1.593
THROW_WINDOW = (BR_TIME - 50/300., BR_TIME + 30/300.)


# ---------------------------------------------------------------------- helpers
def derive_lengths(model_path):
    m = osim.Model(str(model_path)); m.initSystem()
    bs = m.getBodySet()
    return {b: float(np.linalg.norm(np.array([
        bs.get(b).getMassCenter().get(i) for i in range(3)
    ]))) / DE_LEVA[b] for b in DE_LEVA if bs.contains(b)}


def dempster_coms(L):
    return {b: np.array([0.0, -DEMPSTER[b] * L[b], 0.0]) for b in L}


def get_personalized(name: str, src_model: Path) -> Path:
    out_root = REPO / f"out/audit_n/{name}"
    personalized = out_root / "all_recipes/theia_pitching_personalized.osim"
    if personalized.exists():
        print(f"  [skip personalize] cached at {personalized}")
        return personalized
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"  [personalize] {src_model.name} → {personalized}")
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


def read_mot(mot_path: Path):
    with open(mot_path) as f:
        for line in f:
            if line.strip().lower() == "endheader":
                break
        hdr = next(f).strip().split()
        rows = [list(map(float, ln.split())) for ln in f if ln.strip()]
    arr = np.array(rows)
    return hdr, arr


def load_v3d(name: str):
    """Read a V3D procdb signal of the form
    `[{"component": "X", "data": [...]}, {"component": "Y", "data": [...]},
      {"component": "Z", "data": [...]}]`. Returns (t, vec) at 300 Hz."""
    p = json.loads(PROCDB.read_text())
    for it in p["Visual3D"]:
        if it.get("name") == name:
            comps = it["signal"]
            data = np.array([
                [None_to_nan(v) for v in c["data"]]
                for c in comps
            ], dtype=np.float64).T  # (T, 3)
            t = np.arange(data.shape[0]) / 300.0
            return t, data
    raise KeyError(name)


def None_to_nan(x):
    return float("nan") if x is None else float(x)


def build_t_axis(n_frames, sample_rate=300.0):
    return np.arange(n_frames) / sample_rate


def kinematics_score(mot_path: Path, model_path: Path) -> dict:
    """Pull shoulder/elbow/wrist coord trajectories from .mot and compare
    each available coord against V3D angle components. Reports per-coord
    peak |angle| and per-coord RMS difference vs the closest V3D
    component (since each model's coord ordering may differ)."""
    hdr, arr = read_mot(mot_path)
    times = arr[:, 0]
    coord_idx = {h: i for i, h in enumerate(hdr)}

    # Resolve coord-name candidates per joint
    shoulder_candidates = [
        "arm_flex_r", "arm_add_r", "arm_rot_r",     # standard Cardan
        "shoulder_elv", "elv_angle", "shoulder_rot", # other conventions
    ]
    elbow_candidates = ["elbow_flex_r"]
    wrist_candidates = ["wrist_flex_r", "wrist_dev_r", "pro_sup_r"]

    def gather(names):
        out = {}
        for n in names:
            if n in coord_idx:
                out[n] = arr[:, coord_idx[n]]
        return out

    sh = gather(shoulder_candidates)
    el = gather(elbow_candidates)
    wr = gather(wrist_candidates)

    # Throw-window mask
    mask = (times >= THROW_WINDOW[0]) & (times <= THROW_WINDOW[1])

    # V3D references (in degrees)
    try:
        t_v3d_sh, v3d_sh = load_v3d("SHOULDER_ANGLE")
    except KeyError:
        v3d_sh = None
    try:
        t_v3d_el, v3d_el = load_v3d("ELBOW_ANGLE")
    except KeyError:
        v3d_el = None
    try:
        t_v3d_wr, v3d_wr = load_v3d("RT_WRIST_ANGLE")
    except KeyError:
        v3d_wr = None

    def coord_peak_and_best_match(coord_arr, t_us, v3d_data, t_v3d):
        """Peak |angle| + RMS vs the closest V3D component."""
        if v3d_data is None:
            return None
        c_us = coord_arr[mask]
        peak = float(np.nanmax(np.abs(c_us)))
        # Resample V3D to our timing
        best_rms = float("inf"); best_axis = None
        for ax_idx, ax_name in enumerate("XYZ"):
            v3d_resamp = np.interp(t_us[mask], t_v3d, v3d_data[:, ax_idx])
            valid = np.isfinite(v3d_resamp)
            if not np.any(valid):
                continue
            rms = float(np.sqrt(np.mean((c_us[valid] - v3d_resamp[valid]) ** 2)))
            if rms < best_rms:
                best_rms = rms; best_axis = ax_name
        return {"peak": peak, "rms_vs_v3d": best_rms, "best_v3d_axis": best_axis}

    out = {"shoulder_coords": {}, "elbow_coords": {}, "wrist_coords": {},
           "available_coords": list(sorted(set(list(sh) + list(el) + list(wr))))}
    for n, a in sh.items():
        out["shoulder_coords"][n] = coord_peak_and_best_match(a, times, v3d_sh,
                                                              t_v3d_sh if v3d_sh is not None else None)
    for n, a in el.items():
        out["elbow_coords"][n] = coord_peak_and_best_match(a, times, v3d_el,
                                                            t_v3d_el if v3d_el is not None else None)
    for n, a in wr.items():
        out["wrist_coords"][n] = coord_peak_and_best_match(a, times, v3d_wr,
                                                           t_v3d_wr if v3d_wr is not None else None)

    return out


def peak_mag(arr3, t, t_lo, t_hi):
    mask = (t >= t_lo) & (t <= t_hi)
    sub = arr3[mask]
    if sub.size == 0: return float("nan")
    mags = np.linalg.norm(sub, axis=1)
    return float(mags[np.argmax(mags)]) if np.all(np.isfinite(mags)) else float("nan")


def kinetics_score(personalized: Path, wrist_mode: str = "auto") -> dict:
    L = derive_lengths(personalized)
    com_d = dempster_coms(L) if all(b in L for b in DEMPSTER) else None
    res = compute_throwing_arm_reactions_from_c3d(
        C3D, personalized, side="r", wrist_mode=wrist_mode,
        com_overrides=com_d,
        # Production cascade defaults — 18 Hz kine + 10 Hz kinet — pulled
        # automatically from KINEMATIC_LOWPASS_HZ / KINETIC_LOWPASS_HZ env vars.
    )
    t = res["times"]
    return {
        "wrist_mode_used": res["wrist_mode_used"],
        "kinematic_lowpass_hz": res["kinematic_lowpass_hz"],
        "kinetic_lowpass_hz": res["kinetic_lowpass_hz"],
        "shoulder_F": peak_mag(res["shoulder_F_humerus"], t, *THROW_WINDOW),
        "shoulder_M": peak_mag(res["shoulder_M_humerus"], t, *THROW_WINDOW),
        "elbow_F":    peak_mag(res["elbow_F_ulna_frame"], t, *THROW_WINDOW),
        "elbow_M":    peak_mag(res["elbow_M_ulna_frame"], t, *THROW_WINDOW),
        "_time_series": {k: res[k].copy() for k in (
            "shoulder_F_humerus", "shoulder_M_humerus",
            "elbow_F_ulna_frame", "elbow_M_ulna_frame", "times",
        )},
    }


def write_kinetic_plot(results: list[tuple[str, dict, dict]]):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    metrics = ["shoulder_F", "shoulder_M", "elbow_F", "elbow_M"]
    units = ["N", "N·m", "N", "N·m"]
    for ax, m, u in zip(axes, metrics, units):
        names = [r[0] for r in results]
        vals = [r[2][m] for r in results]
        ax.bar(range(len(names)), vals, color="#c33", alpha=0.85)
        ax.axhline(V3D[m], color="#39c", lw=2, ls="--", label=f"V3D={V3D[m]:.0f}")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, fontsize=8)
        ax.set_title(f"{m} ({u})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        for i, v in enumerate(vals):
            if v == v:  # not nan
                ax.text(i, v, f"{v:.0f}\n({v/V3D[m]:.2f}x)", ha="center",
                        va="bottom", fontsize=7)
    fig.suptitle("Audit N — kinetics peaks per model vs V3D")
    fig.tight_layout()
    out = REPO / "out/audit_n_kinetics.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f">>> wrote {out}")


def write_kinematics_plot(results: list[tuple[str, dict, dict]]):
    """For each model, overlay the three shoulder coord traces against V3D's
    SHOULDER_ANGLE components (X/Y/Z) over the throw window."""
    try:
        t_v3d, v3d_sh = load_v3d("SHOULDER_ANGLE")
    except KeyError:
        print("V3D SHOULDER_ANGLE not in procdb; skipping kinematics plot.")
        return
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 3.0 * len(results)),
                             sharex=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (name, kine, _) in zip(axes, results):
        # X/Y/Z V3D components
        for ax_idx, comp_name, color in [(0, "X", "#2a6"), (1, "Y", "#a26"),
                                          (2, "Z", "#26a")]:
            ax.plot(t_v3d, v3d_sh[:, ax_idx], color=color, lw=1.4,
                    label=f"V3D {comp_name}", alpha=0.7)
        # Plot whichever shoulder coords are available (max 3)
        coord_colors = ["#000", "#444", "#888"]
        for ci, (cname, info) in enumerate(kine["shoulder_coords"].items()):
            if info is None or ci >= 3:
                continue
            mot = REPO / f"out/audit_n/{name}/all_recipes/recipe_d/analytical.mot"
            hdr, arr = read_mot(mot)
            t_us = arr[:, 0]
            data = arr[:, hdr.index(cname)]
            ax.plot(t_us, data, color=coord_colors[ci], lw=1.0,
                    label=f"{cname} (RMS={info['rms_vs_v3d']:.1f}° vs V3D-{info['best_v3d_axis']})",
                    ls="--")
        ax.axvline(BR_TIME, color="k", ls=":", alpha=0.4, lw=0.7)
        ax.set_title(f"{name} — shoulder coords vs V3D SHOULDER_ANGLE")
        ax.set_ylabel("angle (deg)")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    out = REPO / "out/audit_n_kinematics.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f">>> wrote {out}")


# ----------------------------------------------------------------------- main
def main():
    results = []
    for cand in CANDIDATES:
        print(f"\n{'='*70}\nCandidate: {cand['name']}\n{'='*70}")
        try:
            personalized = get_personalized(cand["name"], cand["src_model"])
        except Exception as e:
            print(f"  [SKIP] personalize failed: {e}")
            continue

        mot = REPO / f"out/audit_n/{cand['name']}/all_recipes/recipe_d/analytical.mot"
        if not mot.exists():
            print(f"  [SKIP] missing {mot}")
            continue

        kine = kinematics_score(mot, personalized)
        kinet = kinetics_score(personalized, wrist_mode="auto")
        print(f"  wrist_mode = {kinet['wrist_mode_used']}")
        print(f"  available coords (throwing arm + wrist): "
              f"{kine['available_coords']}")
        print(f"  kinetics: sF={kinet['shoulder_F']:.0f} N "
              f"({kinet['shoulder_F']/V3D['shoulder_F']:.2f}x V3D), "
              f"eF={kinet['elbow_F']:.0f} N ({kinet['elbow_F']/V3D['elbow_F']:.2f}x)")
        results.append((cand["name"], kine, kinet))

        out_dir = REPO / f"out/audit_n/{cand['name']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save peaks (drop time-series for compactness)
        compact_kinet = {k: v for k, v in kinet.items() if k != "_time_series"}
        (out_dir / "peaks.json").write_text(json.dumps(
            {"kinematics": kine, "kinetics": compact_kinet}, indent=2))

    # Summary table
    lines = [""]
    lines.append("Audit N — model bake-off summary (production 18Hz+10Hz cascade, Dempster COMs):")
    lines.append("")
    header = (f"  {'model':<22}{'wrist':<10}{'sF':>8}{'sM':>8}{'eF':>8}{'eM':>8}"
              f"{'sF/V3D':>9}{'eF/V3D':>9}  shoulder_coords (RMS° vs V3D best-axis)")
    lines.append(header)
    lines.append("-" * 130)
    for name, kine, kinet in results:
        coords_str = " ".join(
            f"{n}={info['rms_vs_v3d']:.1f}°({info['best_v3d_axis']})"
            for n, info in kine["shoulder_coords"].items()
            if info is not None
        )
        sf = kinet["shoulder_F"]; sm = kinet["shoulder_M"]
        ef = kinet["elbow_F"];    em = kinet["elbow_M"]
        lines.append(f"  {name:<22}{kinet['wrist_mode_used']:<10}"
                     f"{sf:>8.0f}{sm:>8.0f}{ef:>8.0f}{em:>8.0f}"
                     f"{sf/V3D['shoulder_F']:>8.2f}x{ef/V3D['elbow_F']:>8.2f}x"
                     f"  {coords_str}")
    lines.append(f"  {'V3D':<22}{'-':<10}"
                 f"{V3D['shoulder_F']:>8.0f}{V3D['shoulder_M']:>8.0f}"
                 f"{V3D['elbow_F']:>8.0f}{V3D['elbow_M']:>8.0f}"
                 f"{1.0:>8.2f}x{1.0:>8.2f}x")
    lines.append("")
    # Wrist availability
    lines.append("Wrist DOF availability:")
    for name, kine, _ in results:
        wcoords = list(kine["wrist_coords"])
        if wcoords:
            wrms = " ".join(f"{n}: RMS={kine['wrist_coords'][n]['rms_vs_v3d']:.1f}°"
                            for n in wcoords if kine['wrist_coords'][n] is not None)
            lines.append(f"  {name}: {wcoords}    {wrms}")
        else:
            lines.append(f"  {name}: (no wrist coords in .mot — welded)")

    summary = "\n".join(lines)
    print(summary)
    (REPO / "out/audit_n_summary.txt").write_text(summary + "\n")

    # Plots
    if results:
        write_kinetic_plot(results)
        write_kinematics_plot(results)


if __name__ == "__main__":
    main()
