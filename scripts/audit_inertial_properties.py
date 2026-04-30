"""Step 1 of M2 kinetics handoff: audit personalized-model inertial properties.

Dumps mass, COM, and inertia for humerus_r/ulna_r/hand_r/torso from
- the stock LaiUhlrich2022 model
- the personalized model produced by Recipe D
- de Leva 1996 expected values (mass = frac*subject_mass, COM = com_axial_frac*L,
  inertia diag = mass*(k*L)^2)

Saves a summary table to out/audit_inertial_summary.txt and bar-chart plots to
out/audit_inertial_<body>.png and out/audit_inertial_overview.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import opensim as osim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from theia_osim.model_build.anthropometrics import (  # noqa: E402
    DE_LEVA_MALE,
    V3D_TO_OSIM_BODY,
)
from theia_osim.c3d_io.reader import read_theia_c3d  # noqa: E402
from theia_osim.c3d_io.mdh_parser import parse_mdh  # noqa: E402
from theia_osim.model_build.personalize import (  # noqa: E402
    compute_segment_lengths,
    LAI_DEFAULT_LENGTHS_M,
)

REPO = Path(__file__).resolve().parent.parent
STOCK = REPO / "data/models/recipes/LaiUhlrich2022_full_body.osim"
PERSONAL = REPO / "out/repro_baseline/all_recipes/theia_pitching_personalized.osim"
OUT = REPO / "out"

SUBJECT_MASS = 89.81
SUBJECT_HEIGHT = 1.88

# subject segment lengths (from personalize report we replicate by re-reading)
# Pull from the personalized model's geometry by measuring joint origins.
BODIES = ["humerus_r", "ulna_r", "hand_r", "torso"]
BODY_TO_V3D = {v: k for k, v in V3D_TO_OSIM_BODY.items() if v}


def dump_body(model: osim.Model, body_name: str) -> dict:
    model.initSystem()
    bs = model.getBodySet()
    b = bs.get(body_name)
    com = b.getMassCenter()
    inertia = b.getInertia()
    moments = inertia.getMoments()
    products = inertia.getProducts()
    return {
        "mass": b.getMass(),
        "com": (com.get(0), com.get(1), com.get(2)),
        "I": (moments.get(0), moments.get(1), moments.get(2)),
        "Iprod": (products.get(0), products.get(1), products.get(2)),
    }


def get_subject_lengths() -> dict[str, float]:
    c3d = REPO / "pose_filt_0.c3d"
    mdh = REPO / "theia_model.mdh"
    trial = read_theia_c3d(c3d)
    mdh_obj = parse_mdh(mdh) if mdh.exists() else None
    return compute_segment_lengths(trial, mdh=mdh_obj)


def deleva_expected(body_name: str, length_m: float, subject_mass: float) -> dict:
    v3d = BODY_TO_V3D.get(body_name)
    if v3d is None or v3d not in DE_LEVA_MALE:
        return {}
    a = DE_LEVA_MALE[v3d]
    mass = a.mass_fraction * subject_mass
    if body_name == "torso":
        head = DE_LEVA_MALE.get("RHE")
        if head:
            mass += head.mass_fraction * subject_mass
    return {
        "mass": mass,
        "com_axial": a.com_axial_frac * length_m,
        "I": (mass * (a.k_xx * length_m) ** 2,
              mass * (a.k_yy * length_m) ** 2,
              mass * (a.k_zz * length_m) ** 2),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)
    stock = osim.Model(str(STOCK))
    personal = osim.Model(str(PERSONAL))
    stock.initSystem(); personal.initSystem()

    subj_lengths = get_subject_lengths()
    rows = []
    for body in BODIES:
        L_personal = subj_lengths.get(body, LAI_DEFAULT_LENGTHS_M.get(body, float("nan")))
        L_stock = LAI_DEFAULT_LENGTHS_M.get(body, float("nan"))
        s = dump_body(stock, body)
        p = dump_body(personal, body)
        e = deleva_expected(body, L_personal, SUBJECT_MASS)
        rows.append((body, L_stock, L_personal, s, p, e))

    # ---------- text summary ----------
    txt_lines = []
    txt_lines.append(f"Subject: mass={SUBJECT_MASS} kg, height={SUBJECT_HEIGHT} m")
    txt_lines.append("=" * 100)
    for body, L_stock, L_pers, s, p, e in rows:
        com_axial_pers = max(abs(c) for c in p["com"])  # dominant component
        txt_lines.append(f"\n[{body}]  L_stock={L_stock:.4f}m  L_personalized={L_pers:.4f}m")
        txt_lines.append(f"  mass:   stock={s['mass']:.4f}   personalized={p['mass']:.4f}   deLeva_exp={e.get('mass', float('nan')):.4f}  kg")
        txt_lines.append(f"  COM:    stock={tuple(round(x,4) for x in s['com'])}")
        txt_lines.append(f"          personalized={tuple(round(x,4) for x in p['com'])}")
        if e:
            txt_lines.append(f"          deLeva_exp_axial_offset={e['com_axial']:.4f} m  (signed axis depends on body frame)")
            txt_lines.append(f"          |dominant personalized COM offset|={com_axial_pers:.4f} m")
            ratio = com_axial_pers / e["com_axial"] if e["com_axial"] else float("nan")
            txt_lines.append(f"          personalized/deLeva ratio = {ratio:.3f}")
        txt_lines.append(f"  I diag: stock={tuple(round(x,5) for x in s['I'])}")
        txt_lines.append(f"          personalized={tuple(round(x,5) for x in p['I'])}")
        if e:
            txt_lines.append(f"          deLeva_exp={tuple(round(x,5) for x in e['I'])}")
            r = tuple(p['I'][k]/e['I'][k] if e['I'][k] else float('nan') for k in range(3))
            txt_lines.append(f"          personalized/deLeva I ratio = {tuple(round(x,3) for x in r)}")

    summary_path = OUT / "audit_inertial_summary.txt"
    summary_path.write_text("\n".join(txt_lines))
    print("\n".join(txt_lines))
    print(f"\n>>> wrote {summary_path}")

    # ---------- per-body bar charts (mass, COM mag, I diag) ----------
    for body, L_stock, L_pers, s, p, e in rows:
        if not e:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        # mass
        labels = ["stock", "personalized", "deLeva exp"]
        masses = [s["mass"], p["mass"], e["mass"]]
        axes[0].bar(labels, masses, color=["#888", "#c33", "#39c"])
        axes[0].set_ylabel("mass (kg)")
        axes[0].set_title(f"{body}: mass")
        for i, v in enumerate(masses):
            axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom")

        # COM offset magnitude (dominant axis only — segment frames vary)
        com_mag_stock = max(abs(c) for c in s["com"])
        com_mag_pers = max(abs(c) for c in p["com"])
        com_mag_exp = e["com_axial"]
        coms = [com_mag_stock, com_mag_pers, com_mag_exp]
        axes[1].bar(labels, coms, color=["#888", "#c33", "#39c"])
        axes[1].set_ylabel("|dominant COM offset| (m)")
        axes[1].set_title(f"{body}: COM offset (dominant axis)")
        for i, v in enumerate(coms):
            axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom")

        # inertia diag
        x = np.arange(3)
        w = 0.27
        axes[2].bar(x - w, s["I"], w, label="stock", color="#888")
        axes[2].bar(x,     p["I"], w, label="personalized", color="#c33")
        axes[2].bar(x + w, e["I"], w, label="deLeva exp", color="#39c")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(["Ixx", "Iyy", "Izz"])
        axes[2].set_ylabel("I (kg·m²)")
        axes[2].set_title(f"{body}: inertia diag (about COM)")
        axes[2].legend()

        fig.suptitle(f"Inertial audit — {body}  (subject L={L_pers:.3f} m)")
        fig.tight_layout()
        out_png = OUT / f"audit_inertial_{body}.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f">>> wrote {out_png}")

    # ---------- overview: ratio plot for upper-extremity bodies ----------
    fig, ax = plt.subplots(figsize=(11, 5))
    bodies_plot = [b for b, *_ in rows if BODY_TO_V3D.get(b) in DE_LEVA_MALE]
    metrics = ["mass", "|COM|", "Ixx", "Iyy", "Izz"]
    width = 0.16
    x = np.arange(len(metrics))
    for i, (body, L_stock, L_pers, s, p, e) in enumerate(rows):
        if not e:
            continue
        com_mag_p = max(abs(c) for c in p["com"])
        ratios = [
            p["mass"] / e["mass"],
            com_mag_p / e["com_axial"] if e["com_axial"] else 0,
            p["I"][0] / e["I"][0] if e["I"][0] else 0,
            p["I"][1] / e["I"][1] if e["I"][1] else 0,
            p["I"][2] / e["I"][2] if e["I"][2] else 0,
        ]
        ax.bar(x + (i - 1.5) * width, ratios, width, label=body)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("personalized / deLeva-expected")
    ax.set_title("Inertial audit overview — ratio of personalized model to de Leva expected")
    ax.legend()
    fig.tight_layout()
    out_png = OUT / "audit_inertial_overview.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f">>> wrote {out_png}")


if __name__ == "__main__":
    main()
