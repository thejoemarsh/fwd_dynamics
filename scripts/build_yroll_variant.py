"""Build a LaiUhlrich2022 variant with Ry(α) offsets on the acromial joints.

Used by the M2 within-OpenSim Experiment A (axis-rotation). Both PhysicalOffsetFrames
on acromial_r and acromial_l receive the same orientation, which preserves the
resting humerus pose (similarity transform on a zero rotation = zero) but rotates
the joint coordinate axes by Ry(α) in body frame. The intent is to move the
Cardan ZXY middle-axis (arm_add_r) singularity off the throwing trajectory.

Empirical sweep on pose_filt_0.c3d (see scripts/repro_shoulder_kinetics.py for
the test harness): Ry(+60°) drops peak |arm_add_r| from 82.8° → 55.7°, giving
a 34° margin to the X=±90° singularity.

Usage:
    uv run python scripts/build_yroll_variant.py --angle-deg 60 \\
        --src data/models/recipes/LaiUhlrich2022_full_body.osim \\
        --dst data/models/recipes/LaiUhlrich2022_full_body_yroll60.osim
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


def edit_acromial_offsets(text: str, joint_name: str, new_orientation: str) -> str:
    """Replace the two <orientation>0 0 0</orientation> elements within the
    acromial_* CustomJoint block (which are the orientations of torso_offset
    and humerus_*_offset)."""
    start = re.search(rf'<CustomJoint name="{re.escape(joint_name)}">', text)
    if start is None:
        raise SystemExit(f"{joint_name} not found in source model")
    rest = text[start.start():]
    end_match = re.search(r'</CustomJoint>', rest)
    if end_match is None:
        raise SystemExit(f"{joint_name} closing tag not found")
    block_start = start.start()
    block_end = block_start + end_match.end()
    block = text[block_start:block_end]
    edited, n = re.subn(
        r'<orientation>0 0 0</orientation>',
        new_orientation, block, count=2,
    )
    if n != 2:
        raise SystemExit(f"Expected 2 orientation replacements in {joint_name}, got {n}")
    return text[:block_start] + edited + text[block_end:]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--angle-deg", type=float, default=60.0,
                   help="Y-axis offset angle in degrees (default 60).")
    p.add_argument("--src", type=Path,
                   default=Path("data/models/recipes/LaiUhlrich2022_full_body.osim"))
    p.add_argument("--dst", type=Path, required=True)
    args = p.parse_args(argv)

    if not args.src.exists():
        raise SystemExit(f"Source model not found: {args.src}")

    src_text = args.src.read_text()
    rad = math.radians(args.angle_deg)
    new_orientation = f"<orientation>0 {rad:.16f} 0</orientation>"

    modified = src_text
    modified = edit_acromial_offsets(modified, "acromial_r", new_orientation)
    modified = edit_acromial_offsets(modified, "acromial_l", new_orientation)

    n_added = modified.count(new_orientation)
    if n_added != 4:
        raise SystemExit(f"Expected 4 inserted orientations, got {n_added}")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(modified)
    print(f"=> wrote {args.dst} ({args.dst.stat().st_size} bytes)")
    print(f"   Ry(+{args.angle_deg}°) applied to acromial_r and acromial_l offsets.")
    print(f"   To use with the reproducer:")
    print(f"     ACROMIAL_Y_OFFSET_DEG={args.angle_deg} uv run python "
          f"scripts/repro_shoulder_kinetics.py --src-model {args.dst} --out-dir <dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
