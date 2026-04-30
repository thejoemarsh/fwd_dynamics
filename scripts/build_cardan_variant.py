"""Build a LaiUhlrich2022 variant where acromial_r/l use a chosen Cardan
sequence instead of the stock Z-X-Y.

Note: OpenSim CustomJoint hard-rejects collinear axes (Euler sequences with
1st axis == 3rd axis, e.g. Z-Y-Z). Only the 6 Cardan permutations of {X,Y,Z}
are loadable. ZYZ Euler — V3D's natural shoulder representation — is
unavailable in CustomJoint.

Usage:
    uv run python scripts/build_cardan_variant.py --seq XZY \\
        --src data/models/recipes/LaiUhlrich2022_full_body.osim \\
        --dst data/models/recipes/LaiUhlrich2022_full_body_xzy.osim
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

AXIS_VEC = {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}
VALID_CARDAN = {"ZXY", "ZYX", "XYZ", "XZY", "YXZ", "YZX"}

# Anatomical coordinate names per Cardan sequence. These reflect what each
# rotation actually means physically in the OpenSim torso frame (X=anterior,
# Y=superior, Z=lateral) for the throwing shoulder. Default ZXY uses the
# stock LaiUhlrich names (rotation1=Z=flexion, rotation2=X=adduction,
# rotation3=Y=axial). XZY puts abduction first, which is the canonical
# pitching parameterization. Other sequences would need their own anatomical
# analysis; we only define ZXY (stock) and XZY (anatomical-pitching) here.
ANATOMICAL_COORDS = {
    "ZXY": {  # stock
        "r": ("arm_flex_r", "arm_add_r", "arm_rot_r"),
        "l": ("arm_flex_l", "arm_add_l", "arm_rot_l"),
    },
    "XZY": {  # rotation1=X=abduction, rotation2=Z=horizontal abd, rotation3=Y=internal rot
        "r": ("shoulder_abd_r", "shoulder_hzn_r", "shoulder_int_rot_r"),
        "l": ("shoulder_abd_l", "shoulder_hzn_l", "shoulder_int_rot_l"),
    },
}


def edit_acromial_axes(text: str, joint_name: str, seq: str) -> str:
    """Set rotation1/2/3 axis vectors of the named acromial CustomJoint to
    match the given Cardan sequence."""
    start = re.search(rf'<CustomJoint name="{re.escape(joint_name)}">', text)
    if start is None:
        raise SystemExit(f"{joint_name} not found")
    rest = text[start.start():]
    end = re.search(r'</CustomJoint>', rest)
    if end is None:
        raise SystemExit(f"{joint_name} closing tag not found")
    block_start, block_end = start.start(), start.start() + end.end()
    block = text[block_start:block_end]

    for i, axis_letter in enumerate(seq, start=1):
        target = AXIS_VEC[axis_letter]
        # Replace the FIRST <axis>...</axis> following <TransformAxis name="rotationN">.
        pat = re.compile(
            rf'(<TransformAxis name="rotation{i}">.*?<axis>)[^<]+(</axis>)',
            re.DOTALL)
        block, n = pat.subn(rf'\g<1>{target}\g<2>', block, count=1)
        if n != 1:
            raise SystemExit(
                f"{joint_name}: rotation{i} axis edit failed (matches: {n})")
    return text[:block_start] + block + text[block_end:]


def rename_acromial_coords(text: str, side: str, seq: str) -> str:
    """Rename arm_flex_{side}/arm_add_{side}/arm_rot_{side} to the anatomical
    names for the chosen sequence, globally throughout the model file.

    Renames hit three structural locations:
    - `<Coordinate name="OLD">` declarations inside acromial_{side}'s joint block.
    - `<coordinates>OLD</coordinates>` references inside each TransformAxis.
    - `<coordinate>OLD</coordinate>` references inside CoordinateActuators in
      the model's top-level ForceSet (the `shoulder_flex_r/add_r/rot_r`
      actuators bind to these coords).

    The coord names are unique in the model file (grep confirms 9 occurrences
    per right-side coord, all in those three categories), so a global
    string-anchored regex replace is safe.
    """
    new_names = ANATOMICAL_COORDS[seq][side]
    old_names = ANATOMICAL_COORDS["ZXY"][side]
    for old, new in zip(old_names, new_names):
        # Anchored on `>OLD<` so we only hit the XML element-content occurrences,
        # not e.g. a substring of a longer name.
        text = re.sub(rf'>{re.escape(old)}<', f'>{new}<', text)
        # Also rename the <Coordinate name="OLD"> declaration.
        text = re.sub(rf'name="{re.escape(old)}"', f'name="{new}"', text)
    return text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True, help=f"Cardan sequence, one of {sorted(VALID_CARDAN)}")
    p.add_argument("--src", type=Path,
                   default=Path("data/models/recipes/LaiUhlrich2022_full_body.osim"))
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--rename-coords", action="store_true",
                   help="Also rename the acromial coordinates to match the anatomical "
                        "meaning of the chosen sequence (e.g. shoulder_abd_r/_hzn_r/_int_rot_r "
                        "for XZY). Only defined for the sequences in ANATOMICAL_COORDS.")
    args = p.parse_args(argv)
    seq = args.seq.upper()
    if seq not in VALID_CARDAN:
        raise SystemExit(f"--seq must be one of {sorted(VALID_CARDAN)} (Euler "
                         f"sequences with collinear axes are rejected by OpenSim)")
    if args.rename_coords and seq not in ANATOMICAL_COORDS:
        raise SystemExit(f"--rename-coords not supported for {seq}; only "
                         f"{sorted(ANATOMICAL_COORDS)} have anatomical labels defined.")
    if not args.src.exists():
        raise SystemExit(f"Source not found: {args.src}")
    text = args.src.read_text()
    text = edit_acromial_axes(text, "acromial_r", seq)
    text = edit_acromial_axes(text, "acromial_l", seq)
    if args.rename_coords:
        text = rename_acromial_coords(text, "r", seq)
        text = rename_acromial_coords(text, "l", seq)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(text)
    print(f"=> wrote {args.dst} ({args.dst.stat().st_size} bytes)")
    print(f"   acromial_r/l rotation axes: {seq[0]}-{seq[1]}-{seq[2]}")
    if args.rename_coords:
        names = ANATOMICAL_COORDS[seq]["r"]
        print(f"   acromial_r coords renamed to: {names[0]}, {names[1]}, {names[2]}")
    print(f"\nRun:  SHOULDER_PARAM={seq} uv run python "
          f"scripts/repro_shoulder_kinetics.py --src-model {args.dst} --out-dir <dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
