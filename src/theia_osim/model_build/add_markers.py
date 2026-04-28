"""Add virtual markers to an OpenSim model from a YAML catalog.

Reads `configs/markers.yaml`, opens the source `.osim`, and inserts a
`<Marker>` for every landmark. The marker's `socket_parent_frame` points at
the body it's attached to (translated through THEIA_TO_OSIM_BODY).
"""
from __future__ import annotations

from pathlib import Path

import opensim as osim

from ..import_pipeline.landmarks import load_marker_catalog


def add_virtual_markers(
    src_model: Path | str,
    catalog_path: Path | str,
    dst_model: Path | str,
    *,
    osim_axis_swap: bool = True,
) -> Path:
    """Open `src_model`, append markers from `catalog_path`, write `dst_model`.

    Args:
        src_model: input .osim (e.g. data/models/recipes/LaiUhlrich2022_full_body.osim).
        catalog_path: configs/markers.yaml.
        dst_model: output path (typically data/models/theia_pitching.osim).
        osim_axis_swap: if True, marker local_xyz from the catalog is interpreted
            in Theia segment frame and rotated to OpenSim segment frame via
            Rx(-90°) (Theia Z-up → OpenSim Y-up). Default True.

    Returns:
        Resolved dst_model path.
    """
    src_model = Path(src_model)
    dst_model = Path(dst_model)
    dst_model.parent.mkdir(parents=True, exist_ok=True)

    model = osim.Model(str(src_model))
    catalog = load_marker_catalog(catalog_path)

    body_names = {model.getBodySet().get(i).getName() for i in range(model.getBodySet().getSize())}

    for seg_name, seg_markers in catalog.items():
        if seg_markers.body not in body_names:
            raise ValueError(
                f"catalog references body {seg_markers.body!r} for segment {seg_name!r}, "
                f"but model {src_model.name} has no such body. Known: {sorted(body_names)}"
            )
        body = model.getBodySet().get(seg_markers.body)
        for lm in seg_markers.landmarks:
            if osim_axis_swap:
                # Theia (+X, +Y, +Z) → OpenSim (+X, +Z, -Y) per Rx(-90°).
                x, y, z = lm.local_xyz
                loc = osim.Vec3(float(x), float(z), float(-y))
            else:
                loc = osim.Vec3(*[float(c) for c in lm.local_xyz])
            marker = osim.Marker(lm.name, body, loc)
            model.addMarker(marker)

    model.finalizeConnections()
    model.printToXML(str(dst_model))
    return dst_model
