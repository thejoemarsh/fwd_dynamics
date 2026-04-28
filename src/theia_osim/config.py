"""Pipeline configuration loader.

Reads YAML, returns frozen dataclasses. Defaults defined inline so a config
file is optional for unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from .constants import DEFAULT_VLB_4X4, SUPPORTED_SAMPLE_RATES


@dataclass(frozen=True)
class FilterConfig:
    theia_lowpass_hz: float = 30.0
    ik_lowpass_hz: float = 20.0
    butterworth_order: int = 4


@dataclass(frozen=True)
class SlopeConfig:
    vlb_4x4: np.ndarray = field(default_factory=lambda: DEFAULT_VLB_4X4.copy())
    apply_at: Literal["c3d_load", "post_ik", "model_ground"] = "c3d_load"


@dataclass(frozen=True)
class SampleRateConfig:
    source: Literal["auto", "override"] = "auto"
    override_hz: float | None = None
    supported: tuple[int, ...] = SUPPORTED_SAMPLE_RATES


@dataclass(frozen=True)
class AnthropometricsConfig:
    source: Literal["theia", "de_leva"] = "theia"
    override_mass_kg: float | None = None
    override_height_m: float | None = None


@dataclass(frozen=True)
class PathsConfig:
    models_dir: Path = field(default_factory=lambda: Path("data/models"))
    output_root: Path = field(default_factory=lambda: Path("out"))


@dataclass(frozen=True)
class RecipesConfig:
    enabled: tuple[str, ...] = ("a", "c")
    primary: Literal["a", "c"] = "a"


@dataclass(frozen=True)
class Config:
    filters: FilterConfig = field(default_factory=FilterConfig)
    slope: SlopeConfig = field(default_factory=SlopeConfig)
    sample_rate: SampleRateConfig = field(default_factory=SampleRateConfig)
    anthropometrics: AnthropometricsConfig = field(default_factory=AnthropometricsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    recipes: RecipesConfig = field(default_factory=RecipesConfig)
    throwing_side: Literal["auto", "R", "L"] = "auto"
    ignored_segments: tuple[str, ...] = ("pelvis_shifted", "worldbody")


def load_config(path: Path | str | None) -> Config:
    """Load config from YAML file. If path is None, return defaults."""
    if path is None:
        return Config()
    path = Path(path)
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    return _from_dict(data)


def _from_dict(data: dict[str, Any]) -> Config:
    f = data.get("filters", {}) or {}
    s = data.get("slope", {}) or {}
    sr = data.get("sample_rate", {}) or {}
    a = data.get("anthropometrics", {}) or {}
    p = data.get("paths", {}) or {}
    r = data.get("recipes", {}) or {}

    vlb = s.get("vlb_4x4")
    vlb_arr = np.asarray(vlb, dtype=np.float64) if vlb is not None else DEFAULT_VLB_4X4.copy()
    if vlb_arr.shape != (4, 4):
        raise ValueError(f"slope.vlb_4x4 must be 4×4, got shape {vlb_arr.shape}")

    return Config(
        filters=FilterConfig(
            theia_lowpass_hz=float(f.get("theia_lowpass_hz", 30.0)),
            ik_lowpass_hz=float(f.get("ik_lowpass_hz", 20.0)),
            butterworth_order=int(f.get("butterworth_order", 4)),
        ),
        slope=SlopeConfig(
            vlb_4x4=vlb_arr,
            apply_at=s.get("apply_at", "c3d_load"),
        ),
        sample_rate=SampleRateConfig(
            source=sr.get("source", "auto"),
            override_hz=sr.get("override_hz"),
            supported=tuple(sr.get("supported", SUPPORTED_SAMPLE_RATES)),
        ),
        anthropometrics=AnthropometricsConfig(
            source=a.get("source", "theia"),
            override_mass_kg=a.get("override_mass_kg"),
            override_height_m=a.get("override_height_m"),
        ),
        paths=PathsConfig(
            models_dir=Path(p.get("models_dir", "data/models")),
            output_root=Path(p.get("output_root", "out")),
        ),
        recipes=RecipesConfig(
            enabled=tuple(r.get("enabled", ("a", "c"))),
            primary=r.get("primary", "a"),
        ),
        throwing_side=data.get("throwing_side", "auto"),
        ignored_segments=tuple(data.get("ignored_segments", ("pelvis_shifted", "worldbody"))),
    )
