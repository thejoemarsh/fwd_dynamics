"""Virtual-marker catalog + segment-local → world transform.

Single source of truth for both Recipe A (TRC writer) and the model-side
`<Marker>` tags added by `model_build/add_markers.py`. The catalog is loaded
from `configs/markers.yaml`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Landmark:
    name: str
    local_xyz: np.ndarray  # (3,) in segment-local frame, meters


@dataclass(frozen=True)
class SegmentMarkers:
    segment: str  # Theia segment name (e.g. "pelvis", "r_thigh")
    body: str  # OpenSim body name (e.g. "pelvis", "femur_r")
    landmarks: tuple[Landmark, ...]


def load_marker_catalog(path: Path | str) -> dict[str, SegmentMarkers]:
    """Parse configs/markers.yaml → {segment_name: SegmentMarkers}."""
    with open(path) as f:
        data = yaml.safe_load(f)
    out: dict[str, SegmentMarkers] = {}
    for seg, spec in (data.get("markers") or {}).items():
        landmarks = tuple(
            Landmark(name=lm["name"], local_xyz=np.asarray(lm["local_xyz"], dtype=np.float64))
            for lm in spec["landmarks"]
        )
        out[seg] = SegmentMarkers(segment=seg, body=spec["body"], landmarks=landmarks)
    return out


def transform_landmark_to_world(
    transforms: np.ndarray, local_xyz: np.ndarray
) -> np.ndarray:
    """Per-frame: world_xyz = T_world_seg @ [local_xyz; 1].

    Args:
        transforms: (T, 4, 4) per-frame segment poses in world frame.
        local_xyz: (3,) point in segment-local coords.

    Returns:
        (T, 3) array of world-space positions.
    """
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"transforms must be (T, 4, 4); got {transforms.shape}")
    if local_xyz.shape != (3,):
        raise ValueError(f"local_xyz must be (3,); got {local_xyz.shape}")
    homogeneous = np.append(local_xyz, 1.0)  # (4,)
    # (T, 4, 4) @ (4,) → (T, 4); take first 3 components.
    world_h = np.einsum("nij,j->ni", transforms, homogeneous)
    return world_h[:, :3]


def synthesize_all_markers(
    transforms_by_segment: dict[str, np.ndarray],
    catalog: dict[str, SegmentMarkers],
) -> dict[str, np.ndarray]:
    """Compute world-space positions for every marker in the catalog.

    Args:
        transforms_by_segment: from TrialData.transforms.
        catalog: from load_marker_catalog().

    Returns:
        dict mapping marker name → (T, 3) world positions.
    """
    out: dict[str, np.ndarray] = {}
    for seg_name, seg_markers in catalog.items():
        if seg_name not in transforms_by_segment:
            raise ValueError(
                f"catalog references segment {seg_name!r} but transforms only has "
                f"{sorted(transforms_by_segment)}"
            )
        T = transforms_by_segment[seg_name]
        for lm in seg_markers.landmarks:
            out[lm.name] = transform_landmark_to_world(T, lm.local_xyz)
    return out
