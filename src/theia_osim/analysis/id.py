"""Inverse Dynamics driver — wraps `osim.InverseDynamicsTool`.

ID computes generalized forces (joint moments + ground residuals) from a
motion `.mot` of joint coordinates. We feed it Recipe D's analytical .mot;
no external loads (V3D's pitching pipeline is inertial-only — no force
plates).

Filter: 20 Hz Butterworth on coordinates, matching V3D's spec. We use
OpenSim's built-in `lowpass_cutoff_frequency_for_coordinates` rather than
pre-filtering the .mot, so the filter is bound to the tool config and
visible in the printed setup XML.
"""
from __future__ import annotations

from pathlib import Path

import opensim as osim


def run_inverse_dynamics(
    model_path: Path | str,
    coords_mot: Path | str,
    out_sto: Path | str,
    *,
    lowpass_hz: float = 20.0,
    name: str = "inverse_dynamics",
    exclude_muscles: bool = True,
) -> Path:
    """Run InverseDynamicsTool and write generalized-force .sto.

    Args:
        model_path: .osim used to compute kinetics.
        coords_mot: .mot of joint coordinates (e.g. Recipe D's analytical.mot).
        out_sto: where to write the generalized-force .sto.
        lowpass_hz: coordinate lowpass cutoff (default 20 Hz, matches V3D).
        name: tool name (used as setup-XML stem).

    Returns:
        Path to the written .sto.
    """
    model_path = Path(model_path).resolve()
    coords_mot = Path(coords_mot).resolve()
    out_sto = Path(out_sto).resolve()
    out_dir = out_sto.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    storage = osim.Storage(str(coords_mot))
    t_start = storage.getFirstTime()
    t_end = storage.getLastTime()

    tool = osim.InverseDynamicsTool()
    tool.setName(name)
    tool.setModelFileName(str(model_path))
    tool.setCoordinatesFileName(str(coords_mot))
    tool.setLowpassCutoffFrequency(float(lowpass_hz))
    tool.setStartTime(t_start)
    tool.setEndTime(t_end)
    tool.setResultsDir(str(out_dir))
    tool.setOutputGenForceFileName(out_sto.name)
    if exclude_muscles:
        excluded = osim.ArrayStr()
        excluded.append("Muscles")
        tool.setExcludedForces(excluded)
    setup_xml = out_dir / f"{name}_setup.xml"
    tool.printToXML(str(setup_xml))

    # Re-load via setup XML so all paths are consistent for run().
    rerun = osim.InverseDynamicsTool(str(setup_xml))
    if not rerun.run():
        raise RuntimeError(f"InverseDynamicsTool failed; setup at {setup_xml}")
    if not out_sto.exists():
        # Some OpenSim versions write into results dir using only the basename
        # passed in setOutputGenForceFileName; fall back to that.
        candidate = out_dir / out_sto.name
        if candidate.exists():
            return candidate
        raise RuntimeError(f"InverseDynamicsTool ran but {out_sto} missing")
    return out_sto
