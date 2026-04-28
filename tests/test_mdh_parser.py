"""Test MDH parser."""
from __future__ import annotations

from pathlib import Path

import pytest

from theia_osim.c3d_io.mdh_parser import parse_mdh

MDH = Path(__file__).resolve().parents[1] / "theia_model.mdh"


@pytest.mark.skipif(not MDH.exists(), reason="theia_model.mdh fixture not present")
def test_parses_mass_height_default():
    m = parse_mdh(MDH)
    assert m.mass_kg is not None
    assert 50 < m.mass_kg < 200  # plausible adult range
    assert m.height_m is not None
    assert 1.0 < m.height_m < 2.5
    assert m.default_theia_height_m is not None
    assert 1.5 < m.default_theia_height_m < 2.0


@pytest.mark.skipif(not MDH.exists(), reason="theia_model.mdh fixture not present")
def test_extracts_direct_segment_lengths():
    m = parse_mdh(MDH)
    # V3D directly measures hand, foot, toe, thorax lengths
    assert "rha" in m.segment_lengths_m
    assert "lha" in m.segment_lengths_m
    assert "rft" in m.segment_lengths_m
    assert "lft" in m.segment_lengths_m
    # All should be positive, plausibly small
    for k, v in m.segment_lengths_m.items():
        assert 0.01 < v < 0.5, f"{k}={v} out of range"


@pytest.mark.skipif(not MDH.exists(), reason="theia_model.mdh fixture not present")
def test_specific_known_values_for_jansonjunk():
    """Sanity-check exact values against the in-repo fixture."""
    m = parse_mdh(MDH)
    assert abs(m.mass_kg - 89.8128) < 0.001
    assert abs(m.height_m - 1.8796) < 0.001
    assert abs(m.default_theia_height_m - 1.709) < 0.001
    assert abs(m.segment_lengths_m["rha"] - 0.082341) < 1e-6


@pytest.mark.skipif(not MDH.exists(), reason="theia_model.mdh fixture not present")
def test_skips_symbolic_expressions():
    """Length_RHE has no METRIC_VALUE; symbolic Distance() values are skipped."""
    m = parse_mdh(MDH)
    # rhe is left empty in MDH; should not appear
    assert "rhe" not in m.segment_lengths_m


def test_returns_none_when_file_missing(tmp_path):
    """Parser raises on missing file, doesn't return None silently."""
    with pytest.raises(FileNotFoundError):
        parse_mdh(tmp_path / "does_not_exist.mdh")
