"""Scale the LaiUhlrich2022 model to a subject using Theia segment lengths.

Theia ships per-segment lengths (in meters) inside the THEIA3D parameter group.
We compare each subject limb length to LaiUhlrich's stock geometry and apply
per-body scale factors via OpenSim's `ScaleTool`.

Scope (M1.5):
- Major limbs only: femur, tibia, humerus.
- Calcn (foot), ulna/radius (forearm), torso, pelvis, hand, toes — all left
  at 1.0 (geometry definitions don't map cleanly between Theia and LaiUhlrich
  for these). Backfill in M2 once we have static-pose anthropometric data.

Why we don't use ScaleTool's MeasurementSet mode:
- That mode wants marker-pair distance ratios computed from a TRC. Our markers
  are synthesized from segment frames, so the relevant "subject" length info
  is already in the c3d THEIA3D metadata — direct ratio is cleaner.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import opensim as osim

from ..c3d_io.reader import TrialData
from ..c3d_io.theia_meta import SegmentAnthro


# LaiUhlrich2022 default segment "lengths" (joint-frame offset magnitudes,
# computed once via PhysicalOffsetFrame.get_translation()).
LAI_DEFAULT_LENGTHS_M: dict[str, float] = {
    "femur_r": 0.4096,
    "femur_l": 0.4096,
    "tibia_r": 0.4001,
    "tibia_l": 0.4001,
    "humerus_r": 0.2867,
    "humerus_l": 0.2867,
    # The following are intentionally not scaled in M1.5:
    # "calcn_r/l": 0.1788  (foot length — Theia's r_foot length is computed
    #                       differently, not directly comparable)
    # "ulna_r/l": 0.0299   (radioulnar joint offset, NOT the forearm length)
    # "torso": 0.4086      (pelvis-to-acromion, not pure thorax)
}

# Theia segment → OpenSim body. Only entries we actually scale.
THEIA_TO_OSIM_FOR_SCALING: dict[str, str] = {
    "r_thigh": "femur_r",
    "l_thigh": "femur_l",
    "r_shank": "tibia_r",
    "l_shank": "tibia_l",
    "r_uarm": "humerus_r",
    "l_uarm": "humerus_l",
}


@dataclass(frozen=True)
class ScaleReport:
    scaled_model_path: Path
    scale_factors: dict[str, tuple[float, float, float]]
    subject_lengths_m: dict[str, float]
    subject_mass_kg: float
    subject_height_m: float


def compute_scale_factors(
    trial: TrialData,
    *,
    uniform_per_body: bool = True,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, float]]:
    """Per-body scale factors derived from Theia segment lengths.

    Args:
        trial: TrialData with `meta.segments_anthro` populated.
        uniform_per_body: if True, apply the same scalar to (sx, sy, sz). If
            False, only scale the long axis (Y in OpenSim's femur/tibia/humerus
            convention) and leave the other axes at 1.0.

    Returns:
        (scale_factors, subject_lengths_m) where:
            - scale_factors maps OpenSim body name → (sx, sy, sz)
            - subject_lengths_m maps OpenSim body name → Theia length (m)
    """
    factors: dict[str, tuple[float, float, float]] = {}
    lengths: dict[str, float] = {}
    for theia_seg, osim_body in THEIA_TO_OSIM_FOR_SCALING.items():
        anthro: SegmentAnthro | None = trial.meta.segments_anthro.get(theia_seg)
        if anthro is None:
            continue
        subj_len = anthro.length_m
        ref_len = LAI_DEFAULT_LENGTHS_M[osim_body]
        s = subj_len / ref_len
        lengths[osim_body] = subj_len
        if uniform_per_body:
            factors[osim_body] = (s, s, s)
        else:
            # OpenSim femur/tibia/humerus long axis is Y (down in default pose).
            factors[osim_body] = (1.0, s, 1.0)
    return factors, lengths


def write_static_trc_from_first_frame(
    trial: TrialData,
    catalog_path: Path | str,
    out_trc: Path | str,
    *,
    n_frames: int = 5,
    osim_axis_swap: bool = True,
) -> Path:
    """Take the first N frames of synthesized markers and write a static TRC.

    OpenSim's ScaleTool requires a marker file even when MarkerPlacer is off.
    We use a short window from the start of the trial (subject's setup pose).
    """
    from ..import_pipeline.landmarks import load_marker_catalog
    from ..import_pipeline.recipe_a_trc import write_recipe_a_trc

    cat = load_marker_catalog(catalog_path)
    sub_transforms = {
        seg: T[:n_frames] for seg, T in trial.transforms.items()
    }
    return write_recipe_a_trc(
        sub_transforms, cat, out_trc, trial.sample_rate_hz,
        osim_axis_swap=osim_axis_swap,
    )


def run_scale(
    src_model: Path | str,
    trial: TrialData,
    static_trc: Path | str,
    out_model: Path | str,
    *,
    subject_mass_kg: float,
    subject_height_m: float,
    uniform_per_body: bool = True,
) -> ScaleReport:
    """Drive OpenSim ScaleTool with Theia-derived per-body scale factors.

    Args:
        src_model: input markered .osim (output of model_build/add_markers).
        trial: TrialData (we read meta.segments_anthro for lengths).
        static_trc: short TRC for ScaleTool (we use first 5 frames of trial).
        out_model: output scaled .osim path.
        subject_mass_kg: subject mass for ScaleTool's mass distribution.
        subject_height_m: subject height (informational; ScaleTool stores it).
        uniform_per_body: passed through to compute_scale_factors.

    Returns:
        ScaleReport with paths and applied factors.
    """
    src_model = Path(src_model).resolve()
    static_trc = Path(static_trc).resolve()
    out_model = Path(out_model).resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)

    factors, lengths = compute_scale_factors(trial, uniform_per_body=uniform_per_body)

    tool = osim.ScaleTool()
    tool.setName("theia_scale")
    tool.setSubjectMass(float(subject_mass_kg))
    tool.setSubjectHeight(float(subject_height_m * 1000.0))  # mm per OpenSim convention
    tool.setSubjectAge(-1.0)

    # GenericModelMaker — points at the source .osim
    gmm = tool.getGenericModelMaker()
    gmm.setModelFileName(str(src_model))

    # ModelScaler — apply our per-body factors
    ms = tool.getModelScaler()
    ms.setApply(True)
    ms.setPreserveMassDist(False)  # let ScaleTool redistribute mass
    scale_set = ms.getScaleSet()
    for body_name, (sx, sy, sz) in factors.items():
        s = osim.Scale()
        s.setSegmentName(body_name)
        s.setScaleFactors(osim.Vec3(sx, sy, sz))
        s.setApply(True)
        scale_set.cloneAndAppend(s)
    ms.setMarkerFileName(str(static_trc))
    ms.setOutputModelFileName(str(out_model))

    # MarkerPlacer — disabled, our markers are synthesized exactly at segment frames
    mp = tool.getMarkerPlacer()
    mp.setApply(False)

    # ScaleTool wants time range for the static trial (we use the whole thing)
    times = osim.ArrayDouble()
    times.append(0.0)
    times.append(0.5)
    ms.setTimeRange(times)
    mp.setTimeRange(times)

    # Need a setup XML so ScaleTool resolves paths consistently
    setup_xml = out_model.parent / f"{out_model.stem}_scale_setup.xml"
    tool.printToXML(str(setup_xml))
    rerun = osim.ScaleTool(str(setup_xml))
    rerun.run()
    # ScaleTool.run() returns False if MarkerPlacer is skipped — but the
    # ModelScaler step still writes the output. Check the file directly.
    if not out_model.exists():
        raise RuntimeError(f"ScaleTool produced no output; setup at {setup_xml}")

    return ScaleReport(
        scaled_model_path=out_model,
        scale_factors=factors,
        subject_lengths_m=lengths,
        subject_mass_kg=subject_mass_kg,
        subject_height_m=subject_height_m,
    )
