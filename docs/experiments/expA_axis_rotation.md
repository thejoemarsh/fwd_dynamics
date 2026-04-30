# Experiment A — Acromial axis-rotation (Ry+60°). Result: FAILED.

**Date**: 2026-04-29 / 30
**Branch**: `m2-kinetics`
**Commits**: see `scripts/build_yroll_variant.py`, `scripts/repro_shoulder_kinetics.py`,
and the env-toggle `ACROMIAL_Y_OFFSET_DEG` added to `cardan_from_4x4.py`.

## Hypothesis

The 5–8× JR shoulder/elbow force inflation in Recipe D (vs V3D-typical 700–1200 N)
is caused by the Cardan ZXY parameterization of `acromial_r/l` reaching gimbal-lock
conditioning during the throw (`arm_add_r` peaks at 82.8°, 7.2° from singularity).
**If** that hypothesis holds, applying a symmetric Ry(α) orientation to both
PhysicalOffsetFrames of the acromial joints — which preserves resting pose but
rotates the joint coordinate axes — should move the singularity off the
throwing trajectory and bring forces into the V3D-typical range.

The "Ry" axis was picked by sweeping {Rx, Ry, Rz} × {±15°…±90°} in pure
simulation (no model edits), reporting peak |arm_add_r| in each case.
Ry(+60°) was the best-conditioned symmetric offset for the throwing arm:

| axis     | offset | peak \|arm_add_r\| | margin to 90° |
|----------|-------:|-------------------:|--------------:|
| Z (gemini's recommendation) | −45° | 81.5° | +8.5° |
| Z        | −60°   | 78.1° |  +11.9°       |
| X        | +30°   | 72.1° |  +17.9°       |
| **Y**    | **+60°** | **55.7°** | **+34.3°**    |
| Y        | +75°   | 43.9° |  +46.1°       |

## Setup

Variant model: `LaiUhlrich2022_full_body_yroll60.osim` (built by
`scripts/build_yroll_variant.py`). Both `torso_offset` and `humerus_r_offset`
inside `<CustomJoint name="acromial_r">` carry `<orientation>0 1.047... 0</orientation>`
(60° in radians); same for `acromial_l`.

In `cardan_from_4x4.py`, the matching similarity transform is applied when
the env var `ACROMIAL_Y_OFFSET_DEG` is set: `R_rel_new = R_off^T · R_rel · R_off`
where `R_off = Ry(α)`. This ensures the `.mot` ZXY decomposition targets the
rotated joint frames consistently.

Reproduced via:

```bash
ACROMIAL_Y_OFFSET_DEG=60 uv run python scripts/repro_shoulder_kinetics.py \
    --src-model data/models/recipes/LaiUhlrich2022_full_body_yroll60.osim \
    --out-dir out/repro_expA_yroll60
```

## Result

| Metric                       | Baseline | Exp A (Ry+60°) | Δ      | V3D target |
|------------------------------|---------:|---------------:|-------:|-----------:|
| Shoulder F (humerus frame)   | 9532 N   | 8679 N         | **−9%**  | 700–1200   |
| Shoulder M (humerus frame)   | 1940 N·m | 2098 N·m       | +8%    | 70–110     |
| Elbow F    (ulna frame)      | 7191 N   | 6655 N         | −7%    | 700–1100   |
| Elbow M    (ulna frame)      | 994 N·m  | 916 N·m        | −8%    | 60–100     |
| arm_flex_r_moment (ID GenF)  | 848 N·m  | 1933 N·m       | +128%  | —          |
| arm_add_r_moment (ID GenF)   | 1559 N·m | 594 N·m        | −62%   | —          |
| arm_rot_r_moment (ID GenF)   | 481 N·m  | 380 N·m        | −21%   | —          |
| Cardan peak \|arm_add_r\|    | 82.8°    | 55.7°          | −34°   | (away from 90°) |

The singularity moved 34° away from the throwing trajectory as designed. **JR
force/moment magnitudes are essentially unchanged** (7–9% noise drift). The
ID GenForce redistribution between `arm_flex_r_moment` and `arm_add_r_moment`
is just the rotated coordinate axes re-projecting the same body-frame moment;
total magnitude is approximately preserved.

Lower-body invariance was largely OK (all <5% drift) except `lumbar_extension_moment`
which moved +16% — likely a personalization/ScaleTool artifact with the new
offset frames.

## What this falsifies

JR analysis reports forces and moments **in body frames**, which are
coordinate-system-invariant. Rotating the offset frames by the same R on both
parent and child changes how forces project onto coordinate axes (visible in
the ID GenForce redistribution) but does NOT change body-frame physics.
Frame rotation cannot fix a body-frame magnitude problem.

This means the 5–8× inflation is **not from coordinate-system / Jacobian
conditioning at the singularity**. The marquee disagreement in the MoA
session — codex+sonnet (BallJoint will fix it via quaternion mobilizer) vs
gemini (BallJoint won't, axis-rotation will) — was framed on a shared root
cause assumption that this experiment now contradicts.

## What this implies for next steps

BallJoint (Experiment B) would also produce body-frame JR magnitudes, and
since the inflation is in body-frame magnitudes, BallJoint is unlikely to
fix it on its own. **We should pause Experiment B and re-investigate where
the 5–8× body-frame inflation actually comes from.**

Candidate alternative root causes to investigate before re-pivoting:

1. **`setForcesFileName(id_sto)` + `setReplaceForceSet(True)` JR pipeline**:
   ID's coordinate GenForces are fed back to JR as actuator inputs; the
   "constraint forces required to maintain prescribed motion minus applied
   actuator forces" cancellation is numerically delicate. The 9532 N
   shoulder force may be an artifact of this cancellation, not a real
   reaction.
2. **Mass / inertia properties** of personalized humerus, forearm, hand —
   verified gross magnitudes are sensible (2.4 kg humerus, 1.5 kg ulna,
   0.5 kg hand for an 89.8 kg subject) but the inertia tensors and COM
   offsets weren't independently verified against V3D.
3. **20 Hz coordinate lowpass interaction** with the splined q_ddot — could
   smear actual peaks differently than V3D's filtering.
4. **Body-frame convention mismatch** between OpenSim humerus/ulna/torso
   and V3D's RTA/AR/forearm frames — V3D-typical 700-1200 N is in V3D's
   frame; we may already be reporting in a different frame.

## Reverting the experiment

Stock model is unchanged (the variant is a separate file). To revert the
code change:

```bash
unset ACROMIAL_Y_OFFSET_DEG  # default is 0 = no transform applied
```

The cardan_from_4x4.py toggle is a no-op when the env var is unset, so
leaving the code in place doesn't affect normal operation. We may keep it
for future experiments or remove if we settle on a different fix.
