"""Read a Theia .c3d into our internal TrialData representation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ezc3d
import numpy as np

from .slope import apply_slope
from .theia_meta import TheiaMeta, parse_theia3d_group


def _normalize_segment_name(label: str) -> str:
    """Strip Theia's '_4X4' suffix and lowercase."""
    return label.replace("_4X4", "").replace("_4x4", "").lower()


@dataclass(frozen=True)
class TrialData:
    """All spatial data we need from one Theia trial.

    `transforms[segment]` is a (T, 4, 4) array of per-frame world poses.
    Segments listed in `ignored_segments` (worldbody, pelvis_shifted) are dropped.
    """

    path: Path
    transforms: dict[str, np.ndarray]  # segment → (T, 4, 4)
    sample_rate_hz: float
    n_frames: int
    meta: TheiaMeta
    slope_applied: bool


def read_theia_c3d(
    path: Path | str,
    *,
    apply_vlb: np.ndarray | None = None,
    ignored_segments: tuple[str, ...] = ("worldbody", "pelvis_shifted"),
) -> TrialData:
    """Load a Theia .c3d.

    Args:
        path: filesystem path to the .c3d.
        apply_vlb: if given, left-multiply every segment 4×4 by this (Lab → VLB).
            Shape (4, 4). Pass None to leave transforms in raw Theia lab frame.
        ignored_segments: segment names (post-normalization) to drop.

    Returns:
        TrialData with transforms keyed by normalized segment name.

    Raises:
        ValueError: c3d has no ROTATION block, no THEIA3D group, or
            FILTERED!=1 (we require Theia's pre-applied 30 Hz lowpass).
    """
    path = Path(path)
    c = ezc3d.c3d(str(path))

    # Sample rate from POINT.RATE (ROTATION rate is the same per ezc3d c3d spec).
    rate = float(np.asarray(c["parameters"]["POINT"]["RATE"]["value"]).flatten()[0])

    # ROTATION block: shape (4, 4, n_seg, n_frames).
    rotations = c["data"].get("rotations")
    if rotations is None or rotations.size == 0:
        raise ValueError(f"{path} has no ROTATION data — not a Theia .c3d?")
    n_frames = rotations.shape[3]

    rot_labels = c["parameters"]["ROTATION"]["LABELS"]["value"]
    ignored = {s.lower() for s in ignored_segments}

    transforms: dict[str, np.ndarray] = {}
    for i, raw_label in enumerate(rot_labels):
        seg = _normalize_segment_name(raw_label)
        if seg in ignored:
            continue
        # Take (4, 4, n_frames) for segment i, transpose to (n_frames, 4, 4).
        per_frame = np.transpose(rotations[:, :, i, :], (2, 0, 1)).astype(np.float64)
        transforms[seg] = per_frame

    if not np.all(np.isfinite(np.concatenate([T.flatten() for T in transforms.values()]))):
        raise ValueError(f"{path} contains NaN/Inf in segment transforms")

    meta = parse_theia3d_group(c["parameters"])

    slope_applied = False
    if apply_vlb is not None:
        transforms = apply_slope(transforms, apply_vlb)
        slope_applied = True

    return TrialData(
        path=path,
        transforms=transforms,
        sample_rate_hz=rate,
        n_frames=n_frames,
        meta=meta,
        slope_applied=slope_applied,
    )


def list_segments(path: Path | str) -> list[str]:
    """Return normalized segment names present in a Theia c3d (no filtering)."""
    c = ezc3d.c3d(str(path))
    rot_labels = c["parameters"]["ROTATION"]["LABELS"]["value"]
    return [_normalize_segment_name(lab) for lab in rot_labels]
