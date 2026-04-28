"""Parser for Visual3D `.mdh` model files.

The MDH is V3D's pipeline-generated model definition with:
- Subject mass, height
- Per-segment lengths (where directly known from Theia)
- Per-segment radii (anatomical/geometric)
- Segment definitions with mass fractions + de Leva inertia ratios

We only pull the values that drive personalization — segment definitions
themselves stay in OpenSim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MDHMetrics:
    mass_kg: float | None
    height_m: float | None
    default_theia_height_m: float | None
    # Direct length values from MDH `Length_<SEG>` metrics (those V3D measures
    # directly because Theia gives them — RTX (thorax), RHA/LHA (hands),
    # RTO/LTO (toes), RFT/LFT (feet)).
    segment_lengths_m: dict[str, float] = field(default_factory=dict)
    raw_metrics: dict[str, str] = field(default_factory=dict)


# `Set_Model_Metric ... /METRIC_NAME=NAME /METRIC_VALUE=VAL` blocks
_METRIC_BLOCK = re.compile(
    r"Set_Model_Metric\s*"
    r"(?:!\s*/CALIBRATION_FILE=[^\n]*\n)?"
    r"\s*/METRIC_NAME=(?P<name>[^\s/]+)\s*\n"
    r"(?:\s*/METRIC_VALUE=(?P<value>[^\n;]*)\n)?",
    re.IGNORECASE,
)


def parse_mdh(path: Path | str) -> MDHMetrics:
    """Read an MDH file and return the metrics that drive personalization."""
    text = Path(path).read_text()

    raw: dict[str, str] = {}
    for m in _METRIC_BLOCK.finditer(text):
        name = m.group("name").strip()
        value = (m.group("value") or "").strip()
        if value:
            raw[name] = value

    def _to_float(s: str | None) -> float | None:
        if s is None:
            return None
        try:
            return float(s)
        except ValueError:
            return None  # symbolic expression like "0.9*Distance(...)"

    mass = _to_float(raw.get("Mass"))
    height = _to_float(raw.get("Height"))
    default_h = _to_float(raw.get("DefaultTheia3DHeight"))

    # Direct lengths: Length_<SEG> for the segments V3D measures directly.
    segment_lengths: dict[str, float] = {}
    for k, v in raw.items():
        if k.lower().startswith("length_"):
            seg = k[len("length_"):].lower()
            f = _to_float(v)
            if f is not None and f > 0:
                segment_lengths[seg] = f

    return MDHMetrics(
        mass_kg=mass,
        height_m=height,
        default_theia_height_m=default_h,
        segment_lengths_m=segment_lengths,
        raw_metrics=raw,
    )
