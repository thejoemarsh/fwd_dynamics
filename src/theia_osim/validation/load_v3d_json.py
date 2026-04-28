"""Parse Visual3D procdb JSON exports into structured access.

The procdb JSON format is a flat list of records under a single `Visual3D`
key. Each record has `folder` (INFO/EXPORT/EVENTS/LIMB_LENGTHS/YABIN/ORIGINAL),
`name`, `type`, `frames`, and `signal` (a list of {component, data} dicts —
one per axis for vector signals, one per metric for scalars).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class V3DTrial:
    path: Path
    info: dict[str, str]  # SUBJECT_NAME, HAND, PITCH_VELO, QA, etc.
    events: dict[str, float]  # event name → time in seconds (and i_<EVENT> → frame index)
    metrics: dict[str, dict[str, float]]  # metric name → {component → value}
    yabin: dict[str, dict[str, np.ndarray]]  # signal name → {component → (n_frames,)}
    original: dict[str, dict[str, np.ndarray]]  # TIME, FRAMES
    limb_lengths: dict[str, float]


def load_v3d_procdb(path: Path | str) -> V3DTrial:
    """Load a `*_procdb.json` and return structured access."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    items = data.get("Visual3D", [])

    info: dict[str, str] = {}
    events: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = defaultdict(dict)
    yabin: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    original: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    limb_lengths: dict[str, float] = {}

    for it in items:
        folder = it.get("folder", "")
        name = it.get("name", "")
        signals = it.get("signal", [])

        if folder == "INFO":
            if signals and signals[0].get("data"):
                info[name] = signals[0]["data"][0]
        elif folder == "EVENTS":
            if signals and signals[0].get("data"):
                events[name] = float(signals[0]["data"][0])
        elif folder == "EXPORT":
            for s in signals:
                comp = s.get("component", "?")
                d = s.get("data", [])
                if d:
                    metrics[name][comp] = float(d[0])
        elif folder == "LIMB_LENGTHS":
            if signals and signals[0].get("data"):
                limb_lengths[name] = float(signals[0]["data"][0])
        elif folder == "YABIN":
            for s in signals:
                comp = s.get("component", "?")
                d = s.get("data", [])
                yabin[name][comp] = np.asarray(d, dtype=np.float64)
        elif folder == "ORIGINAL":
            for s in signals:
                comp = s.get("component", "?")
                d = s.get("data", [])
                original[name][comp] = np.asarray(d, dtype=np.float64)

    return V3DTrial(
        path=path,
        info=dict(info),
        events=dict(events),
        metrics={k: dict(v) for k, v in metrics.items()},
        yabin={k: dict(v) for k, v in yabin.items()},
        original={k: dict(v) for k, v in original.items()},
        limb_lengths=dict(limb_lengths),
    )


def get_yabin_xyz(trial: V3DTrial, signal_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a YABIN signal's X, Y, Z components as numpy arrays."""
    if signal_name not in trial.yabin:
        raise KeyError(f"YABIN signal {signal_name!r} not found. Available: {sorted(trial.yabin)}")
    sig = trial.yabin[signal_name]
    return sig.get("X", np.array([])), sig.get("Y", np.array([])), sig.get("Z", np.array([]))


def get_time_array(trial: V3DTrial) -> np.ndarray:
    """Return the V3D ORIGINAL.TIME array (seconds, windowed PrePKH→PostBR)."""
    if "TIME" not in trial.original:
        raise KeyError("ORIGINAL.TIME not found in trial")
    return trial.original["TIME"]["X"]
