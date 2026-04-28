"""Parse the THEIA3D parameter group of a Theia .c3d.

THEIA3D ships per-segment anthropometric data (mass, COM, inertia tensors) plus
filter info, model version, and various model-build flags. The 15-element
INERTIA_* layout is inferred — flagged for Theia support confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SegmentAnthro:
    """Anthropometric data for one segment (inferred from Theia INERTIA_* vectors).

    The 15-element layout we infer is:
        [length, m1, comx1, comy1, comz1, Ixx1, Iyy1, Izz1,
                 m2, comx2, comy2, comz2, Ixx2, Iyy2, Izz2]
    where m1/m2 are likely two anthropometric variants (e.g. male/female).
    We use the first variant (indices 1..7) by default.
    """

    length_m: float
    mass_kg: float
    com_local_m: np.ndarray  # (3,) in segment frame
    inertia_diag: np.ndarray  # (3,) Ixx, Iyy, Izz
    raw_15: np.ndarray  # full vector for debugging


@dataclass(frozen=True)
class TheiaMeta:
    theia_version: tuple[int, int, int]  # (year, major, minor)
    model_version: tuple[int, int, int]
    filtered: bool
    filt_freq_hz: float
    segments_anthro: dict[str, SegmentAnthro]


# Mapping from Theia INERTIA_<KEY> parameter to our normalized segment name.
INERTIA_KEY_TO_SEGMENT: dict[str, str] = {
    "INERTIA_HEAD": "head",
    "INERTIA_THORAX": "torso",
    "INERTIA_L_UARM": "l_uarm",
    "INERTIA_L_LARM": "l_larm",
    "INERTIA_L_HAND": "l_hand",
    "INERTIA_R_UARM": "r_uarm",
    "INERTIA_R_LARM": "r_larm",
    "INERTIA_R_HAND": "r_hand",
    "INERTIA_L_THIGH": "l_thigh",
    "INERTIA_L_SHANK": "l_shank",
    "INERTIA_L_FOOT": "l_foot",
    "INERTIA_R_THIGH": "r_thigh",
    "INERTIA_R_SHANK": "r_shank",
    "INERTIA_R_FOOT": "r_foot",
}


def _scalar(theia_group: dict[str, Any], key: str) -> Any:
    if key not in theia_group:
        return None
    val = theia_group[key]["value"]
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return val[0]
    if isinstance(val, np.ndarray) and val.size > 0:
        return val.flat[0]
    return val


def _parse_inertia_15vec(vec: np.ndarray) -> SegmentAnthro:
    flat = np.asarray(vec, dtype=np.float64).flatten()
    if flat.size != 15:
        raise ValueError(f"expected 15-element INERTIA vector, got {flat.size}")
    length = float(flat[0])
    mass = float(flat[1])
    com = flat[2:5].copy()
    inertia = flat[5:8].copy()
    return SegmentAnthro(
        length_m=length,
        mass_kg=mass,
        com_local_m=com,
        inertia_diag=inertia,
        raw_15=flat,
    )


def parse_theia3d_group(c3d_params: dict[str, Any]) -> TheiaMeta:
    """Pull the THEIA3D group out of a parsed ezc3d c3d.parameters dict."""
    if "THEIA3D" not in c3d_params:
        raise ValueError("c3d has no THEIA3D parameter group — not a Theia file?")
    g = c3d_params["THEIA3D"]

    ver = g.get("THEIA3D_VERSION", {}).get("value", [0, 0, 0])
    mver = g.get("MODEL_VERSION", {}).get("value", [0, 0, 0])
    theia_version = tuple(int(x) for x in np.asarray(ver).flatten()[:3])
    model_version = tuple(int(x) for x in np.asarray(mver).flatten()[:3])

    filtered_flag = _scalar(g, "FILTERED")
    filt_freq = _scalar(g, "FILT_FREQ")

    if filtered_flag is None or float(filtered_flag) != 1.0:
        raise ValueError(
            "Theia file is not flagged as FILTERED=1 — pipeline assumes "
            "Theia's pre-applied 30 Hz lowpass."
        )

    segs: dict[str, SegmentAnthro] = {}
    for key, seg_name in INERTIA_KEY_TO_SEGMENT.items():
        if key in g:
            try:
                segs[seg_name] = _parse_inertia_15vec(g[key]["value"])
            except ValueError:
                pass

    return TheiaMeta(
        theia_version=theia_version,  # type: ignore[arg-type]
        model_version=model_version,  # type: ignore[arg-type]
        filtered=bool(filtered_flag),
        filt_freq_hz=float(filt_freq) if filt_freq is not None else 0.0,
        segments_anthro=segs,
    )
