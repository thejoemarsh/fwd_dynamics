"""Test V3D procdb JSON loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from theia_osim.validation.load_v3d_json import (
    get_time_array,
    get_yabin_xyz,
    load_v3d_procdb,
)

PROCDB = Path(__file__).resolve().parents[1] / "pose_filt_0_procdb.json"


@pytest.mark.skipif(not PROCDB.exists(), reason="V3D procdb fixture not present")
def test_load_basic_fields():
    t = load_v3d_procdb(PROCDB)
    assert t.info["HAND"] == "R"
    assert t.info["QA"] == "GOOD"
    assert t.info["PITCH_VELO"] == "937"


@pytest.mark.skipif(not PROCDB.exists(), reason="V3D procdb fixture not present")
def test_events_present_with_expected_ordering():
    t = load_v3d_procdb(PROCDB)
    pkh = t.events["PKH_time"]
    fp = t.events["FP_time"]
    mer = t.events["MER_time"]
    br = t.events["BR_time"]
    # Expected pitching event order: PKH < FP < MER ≈ BR
    assert pkh < fp < mer
    assert abs(mer - br) < 0.1


@pytest.mark.skipif(not PROCDB.exists(), reason="V3D procdb fixture not present")
def test_pelvis_angular_velocity_xyz_shapes():
    t = load_v3d_procdb(PROCDB)
    x, y, z = get_yabin_xyz(t, "PELVIS_ANGULAR_VELOCITY")
    assert len(x) == len(y) == len(z) == 600
    # Z component (long-axis rotation) peaks much higher than X/Y for a pitcher
    assert np.abs(z).max() > np.abs(x).max()
    assert np.abs(z).max() > np.abs(y).max()


@pytest.mark.skipif(not PROCDB.exists(), reason="V3D procdb fixture not present")
def test_export_metric_pelvis_max():
    t = load_v3d_procdb(PROCDB)
    peaks = t.metrics.get("PELVIS_ANGULAR_VELOCITY_MAX", {})
    assert "Z" in peaks
    # Peak Z must equal max(|z|) of the time series (within float precision)
    _, _, z = get_yabin_xyz(t, "PELVIS_ANGULAR_VELOCITY")
    assert abs(peaks["Z"] - float(np.abs(z).max())) < 0.01


@pytest.mark.skipif(not PROCDB.exists(), reason="V3D procdb fixture not present")
def test_time_array_is_300hz():
    t = load_v3d_procdb(PROCDB)
    times = get_time_array(t)
    dt = float(np.median(np.diff(times)))
    assert abs(dt - 1.0 / 300.0) < 1e-4


@pytest.mark.skipif(not PROCDB.exists(), reason="V3D procdb fixture not present")
def test_unknown_signal_raises():
    t = load_v3d_procdb(PROCDB)
    with pytest.raises(KeyError, match="not found"):
        get_yabin_xyz(t, "DOES_NOT_EXIST")
