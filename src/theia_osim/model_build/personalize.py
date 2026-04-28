"""V3D-style per-athlete personalization for the OpenSim model.

Mirrors the recipe in `theia_model.mdh`:

1. Compute per-segment scale factors from Theia c3d (segment-pair distances
   for limbs, MDH-direct lengths for hands/feet/torso, fallback to default).
2. Run OpenSim ScaleTool to resize body geometry.
3. Override each body's mass with de Leva fraction × subject_mass.
4. Override each body's inertia diagonal with `mass × (k × length)²` using
   de Leva radii of gyration.

After this the model is dynamically equivalent to V3D's hybrid segment model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import opensim as osim

from ..c3d_io.mdh_parser import MDHMetrics
from ..c3d_io.reader import TrialData
from .anthropometrics import (
    DE_LEVA_MALE,
    V3D_TO_OSIM_BODY,
    V3D_TO_THEIA_SEG,
    SegmentAnthropometry,
)


# LaiUhlrich2022 default segment lengths (m), measured once from joint offsets.
LAI_DEFAULT_LENGTHS_M: dict[str, float] = {
    "femur_r": 0.4096,
    "femur_l": 0.4096,
    "tibia_r": 0.4001,
    "tibia_l": 0.4001,
    "humerus_r": 0.2867,
    "humerus_l": 0.2867,
    # Below are LaiUhlrich-specific reference distances we use for proxy scaling.
    "pelvis": 0.154,    # 2× hip half-width (0.077 each side)
    "torso": 0.409,     # pelvis-to-acromion distance (note: V3D's RTX is thorax-only)
    "calcn_r": 0.179,   # ankle-to-MTP
    "calcn_l": 0.179,
    "toes_r": 0.05,     # nominal
    "toes_l": 0.05,
    "ulna_r": 0.25,     # elbow-to-wrist distance (sum of ulna + radius/hand frames)
    "ulna_l": 0.25,
    "hand_r": 0.10,     # wrist-to-fingertip nominal
    "hand_l": 0.10,
}


@dataclass(frozen=True)
class PersonalizeReport:
    scaled_model_path: Path
    scale_factors: dict[str, tuple[float, float, float]]
    subject_lengths_m: dict[str, float]
    subject_mass_kg: float
    subject_height_m: float
    body_masses_kg: dict[str, float]
    total_mass_kg: float


def compute_segment_lengths(
    trial: TrialData,
    mdh: MDHMetrics | None = None,
) -> dict[str, float]:
    """Compute per-OpenSim-body segment length (m) from Theia c3d data.

    Order of preference:
      1. MDH `Length_<SEG>` direct values (V3D's hand-measured lengths).
      2. Distance between Theia segment 4×4 origins for joint pairs.
      3. LaiUhlrich default for any body we can't measure.

    Returns dict: OpenSim body name → length (m).
    """
    out: dict[str, float] = {}

    # 1. Limb pairs from segment-pair distances (computed at frame 0)
    pair_distance: dict[str, tuple[str, str]] = {
        "femur_r": ("r_thigh", "r_shank"),
        "femur_l": ("l_thigh", "l_shank"),
        "tibia_r": ("r_shank", "r_foot"),
        "tibia_l": ("l_shank", "l_foot"),
        "humerus_r": ("r_uarm", "r_larm"),
        "humerus_l": ("l_uarm", "l_larm"),
        "ulna_r": ("r_larm", "r_hand"),
        "ulna_l": ("l_larm", "l_hand"),
    }
    for body, (seg_a, seg_b) in pair_distance.items():
        if seg_a in trial.transforms and seg_b in trial.transforms:
            pa = trial.transforms[seg_a][0, :3, 3]
            pb = trial.transforms[seg_b][0, :3, 3]
            out[body] = float(np.linalg.norm(pa - pb))

    # 2. Pelvis width (hip-to-hip distance)
    if "r_thigh" in trial.transforms and "l_thigh" in trial.transforms:
        pa = trial.transforms["r_thigh"][0, :3, 3]
        pb = trial.transforms["l_thigh"][0, :3, 3]
        out["pelvis"] = float(np.linalg.norm(pa - pb))

    # 3. Torso length: pelvis to torso 4×4 distance (covers neck-to-pelvis)
    if "pelvis" in trial.transforms and "torso" in trial.transforms:
        pa = trial.transforms["pelvis"][0, :3, 3]
        pb = trial.transforms["torso"][0, :3, 3]
        out["torso"] = float(np.linalg.norm(pa - pb))

    # 4. MDH-direct lengths for hands, feet, toes
    if mdh is not None:
        if "rha" in mdh.segment_lengths_m:
            out["hand_r"] = mdh.segment_lengths_m["rha"]
        if "lha" in mdh.segment_lengths_m:
            out["hand_l"] = mdh.segment_lengths_m["lha"]
        if "rft" in mdh.segment_lengths_m:
            out["calcn_r"] = mdh.segment_lengths_m["rft"]
        if "lft" in mdh.segment_lengths_m:
            out["calcn_l"] = mdh.segment_lengths_m["lft"]
        if "rto" in mdh.segment_lengths_m:
            out["toes_r"] = mdh.segment_lengths_m["rto"]
        if "lto" in mdh.segment_lengths_m:
            out["toes_l"] = mdh.segment_lengths_m["lto"]
    return out


def compute_scale_factors_full(
    subject_lengths_m: dict[str, float],
    *,
    uniform_per_body: bool = True,
) -> dict[str, tuple[float, float, float]]:
    """Per-body scale factors using all available segment lengths."""
    factors: dict[str, tuple[float, float, float]] = {}
    for body, subj_len in subject_lengths_m.items():
        ref_len = LAI_DEFAULT_LENGTHS_M.get(body)
        if ref_len is None or ref_len <= 0:
            continue
        s = subj_len / ref_len
        if uniform_per_body:
            factors[body] = (s, s, s)
        else:
            factors[body] = (1.0, s, 1.0)
    return factors


def apply_de_leva_mass_and_inertia(
    model: osim.Model,
    subject_mass_kg: float,
    subject_lengths_m: dict[str, float],
) -> dict[str, float]:
    """Override each body's mass + inertia using de Leva fractions.

    Returns dict mapping OpenSim body name → mass (kg) actually set.
    Bodies not covered by de Leva (e.g. patella, talus, radius) are
    left at their post-scale defaults — they're small contributors.
    """
    body_masses: dict[str, float] = {}

    # Head mass folds into torso since LaiUhlrich has no separate head body.
    head_anthro = DE_LEVA_MALE.get("RHE")
    head_mass_to_fold = head_anthro.mass_fraction * subject_mass_kg if head_anthro else 0.0

    for v3d_code, anthro in DE_LEVA_MALE.items():
        osim_body_name = V3D_TO_OSIM_BODY.get(v3d_code)
        if osim_body_name is None:
            continue
        body = model.getBodySet().get(osim_body_name)

        mass = anthro.mass_fraction * subject_mass_kg
        # Fold head into torso.
        if v3d_code == "RTX" and head_mass_to_fold > 0:
            mass += head_mass_to_fold
        body.setMass(mass)
        body_masses[osim_body_name] = mass

        # Inertia diagonals about COM, in body frame.
        # length = subject's segment length in meters; fallback to LaiUhlrich
        # default if we don't have a measured length for this body.
        length = subject_lengths_m.get(
            osim_body_name, LAI_DEFAULT_LENGTHS_M.get(osim_body_name, 0.3)
        )
        ixx = mass * (anthro.k_xx * length) ** 2
        iyy = mass * (anthro.k_yy * length) ** 2
        izz = mass * (anthro.k_zz * length) ** 2
        # OpenSim Inertia: (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
        body.setInertia(osim.Inertia(ixx, iyy, izz, 0.0, 0.0, 0.0))

    return body_masses


def personalize_model(
    src_model: Path | str,
    trial: TrialData,
    static_trc: Path | str,
    out_model: Path | str,
    *,
    subject_mass_kg: float,
    subject_height_m: float,
    mdh: MDHMetrics | None = None,
) -> PersonalizeReport:
    """V3D-style personalization end-to-end.

    Args:
        src_model: input markered .osim (output of model_build/add_markers).
        trial: TrialData from c3d_io.reader (slope-corrected).
        static_trc: short TRC required by ScaleTool.
        out_model: output personalized .osim path.
        subject_mass_kg: total subject mass.
        subject_height_m: total subject height.
        mdh: parsed MDHMetrics (optional, supplies hand/foot/toe lengths).

    Returns:
        PersonalizeReport with applied factors, lengths, and per-body masses.
    """
    src_model = Path(src_model).resolve()
    static_trc = Path(static_trc).resolve()
    out_model = Path(out_model).resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)

    # 1. Compute segment lengths + scale factors
    lengths = compute_segment_lengths(trial, mdh=mdh)
    factors = compute_scale_factors_full(lengths, uniform_per_body=True)

    # 2. Run ScaleTool for geometric scaling
    intermediate = out_model.parent / f"{out_model.stem}_geom.osim"
    tool = osim.ScaleTool()
    tool.setName("theia_personalize")
    tool.setSubjectMass(float(subject_mass_kg))
    tool.setSubjectHeight(float(subject_height_m * 1000.0))
    tool.setSubjectAge(-1.0)
    tool.getGenericModelMaker().setModelFileName(str(src_model))

    ms = tool.getModelScaler()
    ms.setApply(True)
    ms.setPreserveMassDist(False)
    scale_set = ms.getScaleSet()
    for body_name, (sx, sy, sz) in factors.items():
        s = osim.Scale()
        s.setSegmentName(body_name)
        s.setScaleFactors(osim.Vec3(sx, sy, sz))
        s.setApply(True)
        scale_set.cloneAndAppend(s)
    ms.setMarkerFileName(str(static_trc))
    ms.setOutputModelFileName(str(intermediate))
    tool.getMarkerPlacer().setApply(False)
    times = osim.ArrayDouble()
    times.append(0.0)
    times.append(0.5)
    ms.setTimeRange(times)
    tool.getMarkerPlacer().setTimeRange(times)
    setup_xml = out_model.parent / f"{out_model.stem}_scale_setup.xml"
    tool.printToXML(str(setup_xml))
    osim.ScaleTool(str(setup_xml)).run()
    if not intermediate.exists():
        raise RuntimeError(f"ScaleTool produced no output; setup at {setup_xml}")

    # 3. Open scaled model, override mass + inertia using de Leva
    model = osim.Model(str(intermediate))
    model.initSystem()
    body_masses = apply_de_leva_mass_and_inertia(model, subject_mass_kg, lengths)

    # 4. Save final personalized model
    model.printToXML(str(out_model))
    total = sum(
        model.getBodySet().get(i).getMass() for i in range(model.getBodySet().getSize())
    )

    return PersonalizeReport(
        scaled_model_path=out_model,
        scale_factors=factors,
        subject_lengths_m=lengths,
        subject_mass_kg=subject_mass_kg,
        subject_height_m=subject_height_m,
        body_masses_kg=body_masses,
        total_mass_kg=total,
    )
