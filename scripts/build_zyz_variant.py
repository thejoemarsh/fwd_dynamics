"""Build a LaiUhlrich2022 variant where acromial_r/l use a Z-Y-Z Euler
parameterization instead of the stock Z-X-Y Cardan.

Why: V3D's marquee shoulder kinematic (per v3d_scripts/02_CMO_v6_v1.v3s line
255-257) is `RT_SHOULDER_ANGLE` with AXIS1=Z, AXIS2=Y, AXIS3=Z — Z-Y-Z
intrinsic Euler. For pitching, the throwing humerus stays heavily elevated
(β = middle = elevation ~100° throughout the throw) so the Euler singularity
at β = 0° / ±180° is far away. Empirically on pose_filt_0.c3d:
- ZXY Cardan (stock): peak |middle β| = 82.8°, dist to ±90° lock = 7.2°
- ZYZ Euler: peak |middle β| = 106.5°, dist to 0°/180° lock = 73.5°

Plus ZYZ has max framewise jump of 14.7° vs Cardan ZXY's 127.6° — no
chart-cut crossings, so smart-unwrap is unneeded.

The XML edit is small: in each acromial_r/l CustomJoint's SpatialTransform,
change rotation2 axis from `1 0 0` (X) to `0 1 0` (Y), and rotation3 axis
from `0 1 0` (Y) to `0 0 1` (Z). Coordinate names stay the same (they're
just labels in our pipeline). PhysicalOffsetFrames are unchanged.

Usage:
    uv run python scripts/build_zyz_variant.py \\
        --src data/models/recipes/LaiUhlrich2022_full_body.osim \\
        --dst data/models/recipes/LaiUhlrich2022_full_body_zyz.osim

Then run the reproducer with:
    SHOULDER_PARAM=ZYZ uv run python scripts/repro_shoulder_kinetics.py \\
        --src-model data/models/recipes/LaiUhlrich2022_full_body_zyz.osim \\
        --out-dir out/repro_expC_zyz
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def edit_acromial_zyz(text: str, joint_name: str) -> str:
    """Within the acromial_* CustomJoint, change rotation2.axis from X (1 0 0)
    to Y (0 1 0), and rotation3.axis from Y (0 1 0) to Z (0 0 1).

    Done by string-targeted replacement scoped to the joint's block, so other
    joints with similar TransformAxis structures are not affected.
    """
    start = re.search(rf'<CustomJoint name="{re.escape(joint_name)}">', text)
    if start is None:
        raise SystemExit(f"{joint_name} not found")
    rest = text[start.start():]
    end = re.search(r'</CustomJoint>', rest)
    if end is None:
        raise SystemExit(f"{joint_name} closing tag not found")
    block_start = start.start()
    block_end = block_start + end.end()
    block = text[block_start:block_end]

    # Replace rotation2: <coordinates>arm_add_*</coordinates>...<axis>1 0 0</axis>
    # then rotation3:   <coordinates>arm_rot_*</coordinates>...<axis>0 1 0</axis>
    # Use anchored patterns to avoid touching translation axes.
    rot2_pat = re.compile(
        r'(<TransformAxis name="rotation2">.*?<axis>)1 0 0(</axis>)',
        re.DOTALL)
    rot3_pat = re.compile(
        r'(<TransformAxis name="rotation3">.*?<axis>)0 1 0(</axis>)',
        re.DOTALL)
    block, n2 = rot2_pat.subn(r'\g<1>0 1 0\g<2>', block, count=1)
    if n2 != 1:
        raise SystemExit(f"{joint_name}: rotation2 axis edit failed (matches: {n2})")
    block, n3 = rot3_pat.subn(r'\g<1>0 0 1\g<2>', block, count=1)
    if n3 != 1:
        raise SystemExit(f"{joint_name}: rotation3 axis edit failed (matches: {n3})")
    return text[:block_start] + block + text[block_end:]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path,
                   default=Path("data/models/recipes/LaiUhlrich2022_full_body.osim"))
    p.add_argument("--dst", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.src.exists():
        raise SystemExit(f"Source not found: {args.src}")
    text = args.src.read_text()
    text = edit_acromial_zyz(text, "acromial_r")
    text = edit_acromial_zyz(text, "acromial_l")
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(text)
    print(f"=> wrote {args.dst} ({args.dst.stat().st_size} bytes)")
    print(f"   acromial_r/l now use Z-Y-Z Euler axes.")
    print(f"\nVerify: rotation2 axis should be '0 1 0' (Y), rotation3 '0 0 1' (Z) "
          f"within each acromial joint block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
