# Experiment C — Cardan/Euler shoulder parameterization. Result: FAILED + DECISIVE.

**Date**: 2026-04-29 / 30
**Branch**: `m2-kinetics`
**Trigger**: User asked whether V3D-style Cardan/Euler sequences (ZYZ, YXZ)
would solve the kinetics inflation that Experiment A (frame rotation) failed
to fix.

## What V3D actually uses (from v3d_scripts/)

- `02_CMO_v6_v1.v3s` line 255-257: `RT_SHOULDER_ANGLE` uses
  `AXIS1=Z, AXIS2=Y, AXIS3=Z` — **Z-Y-Z intrinsic Euler** (1st axis = 3rd axis).
  This is V3D's canonical shoulder kinematic representation.
- `02_CMO_v6_v1.v3s` line 1184-1224: `RT_SHOULDER_RAR_FORCE` is the joint
  force on RAR (humerus) resolved in RAR (humerus) frame — **frame-matched
  to our `acromial_r_on_humerus_r_in_humerus_r`**, so the 8000 vs 700-1200 N
  gap is a real physics gap, not an apples-to-oranges frame mismatch.
- `05_determine_side.v3s` line 539-557: `RShoulder_Angle_YXZ` uses
  `AXIS1=Y, AXIS2=X, AXIS3=Z` Y-X-Z Cardan with NEGATEY=NEGATEZ=TRUE.
  But this is only used for MER timing detection, not shoulder kinematics.

## Sequence sweep — all 12 sequences scipy supports

Pure simulation on `pose_filt_0.c3d` (no model edits): apply each sequence's
`as_euler` to the segment-derived R_relative for the throwing shoulder.

| seq   | type   | window peak \|β\| | dist to lock | max framewise jump |
|-------|--------|-----------------:|-------------:|-------------------:|
| **ZYZ Euler** (V3D)  | Euler  | 106.5° | **+73.5°** | **14.7°** |
| ZXZ Euler            | Euler  | 106.5° | +73.5°    | 14.7° |
| XYX Euler            | Euler  | 134.8° | +14.1°    | 14.9° |
| **XZY Cardan**       | Cardan | 49.3°  | **+40.7°** | **14.4°** |
| YZX Cardan           | Cardan | 76.5°  | +13.5°    | 61.1° |
| YXZ Cardan           | Cardan | 82.3°  | +7.7°     | 14.6° |
| **ZXY Cardan (current)** | Cardan | 82.8° | +7.2° | 127.6° |
| ZYX Cardan           | Cardan | 87.7°  | +2.3°     | 57.3° |
| XYZ Cardan           | Cardan | 87.9°  | +2.1°     | 64.0° |

**ZYZ Euler is the ideal kinematic representation by every metric** —
73.5° margin to singularity, 14.7° max framewise jump (no chart cuts), and
matches V3D's choice. It works in scipy decomposition.

## Hard limitation: OpenSim CustomJoint rejects Euler sequences

When attempting to load a model variant with `acromial_r` SpatialTransform
axes Z-Y-Z, `osim.Model()` raises:

> `RuntimeError: ... CustomJoint 'acromial_r' has collinear axes and are not well-defined. Please fix and retry loading.`

**OpenSim CustomJoint requires three non-collinear (geometrically distinct)
rotation axes.** Any 3-axis Euler sequence (1st axis == 3rd axis) is
structurally impossible within `CustomJoint`. ZYZ, ZXZ, XYX, XZX, YXY, YZY
all unavailable.

So V3D's natural shoulder representation cannot be expressed in OpenSim's
`CustomJoint` mobilizer. This is a hard, documented constraint.

## Best Cardan alternative tested: XZY

XZY is the best **Cardan** sequence by margin to singularity (40.7°).
Built `LaiUhlrich2022_full_body_xzy.osim` via
`scripts/build_cardan_variant.py --seq XZY`, ran with `SHOULDER_PARAM=XZY`.

| Metric                       | Baseline (ZXY) | Exp A (Ry+60°) | Exp C (XZY) | V3D target |
|------------------------------|---------------:|---------------:|------------:|-----------:|
| Shoulder F (humerus frame)   | 9532 N         | 8679 N (-9%)   | 8805 N (-8%) | 700–1200   |
| Shoulder M (humerus frame)   | 1940 N·m       | 2098 (+8%)     | 1840 (-5%)  | 70–110     |
| Elbow F    (ulna frame)      | 7191 N         | 6655 (-7%)     | 6731 (-6%)  | 700–1100   |
| Elbow M    (ulna frame)      | 994 N·m        | 916 (-8%)      | 901 (-9%)   | 60–100     |

XZY moved the singularity 41° away from the throwing trajectory (vs Exp A's
34° with frame rotation). Yet the body-frame F/M magnitudes drifted only
~5-9% — the same noise band as Exp A.

## What this proves decisively

Three independent within-OpenSim tests all fail to reduce body-frame F/M:

1. **Frame rotation** (Experiment A): rotates joint coordinate axes via
   symmetric `Ry(+60°)` on offset frames. Singularity moves 34° away.
   Result: ~9% kinetic drift.
2. **Sequence change** (Experiment C XZY): different Cardan sequence,
   different geometric meaning of middle β, singularity moves 41° away.
   Result: ~8% kinetic drift.
3. **Sequence sweep** (theoretical): ZYZ Euler would give 73.5° margin
   but is structurally rejected by OpenSim CustomJoint.

These all change the **coordinate-system parameterization**. None of them
change the underlying body-frame physics, and **JR analysis reports
body-frame quantities, which are coordinate-invariant**. The 7-8× inflation
must be in the body-frame computation, not in the coordinate parameterization.

## What this rules out

- **Cardan/Euler sequence change can NOT fix the inflation within OpenSim CustomJoint.**
  Both proposers in the MoA session (codex+sonnet for BallJoint, gemini for
  axis rotation) framed their plans on a singularity-conditioning hypothesis
  that is now experimentally falsified.
- **Smart-unwrap algorithm tweaks can NOT fix it** (already shown by the
  ROM-penalty experiment that produced 226000 N forces).
- **BallJoint (Experiment B from the original plan) is unlikely to fix it**
  by the same body-frame invariance argument: BallJoint also reports
  body-frame F/M, so changing the mobilizer doesn't change the magnitudes
  it produces. (Caveat: BallJoint changes how ω is derived — `u = ω` rather
  than `u = q_dot` — so if the inflation has any residual numerical-conditioning
  component coming through the spline-derived q_dot, BallJoint *might*
  partially help. But based on Exp A and Exp C, there's little evidence
  this conditioning component is large.)

## Where the inflation actually lives — candidates to investigate next

Body-frame F/M is computed by Simbody as `F = m·a_COM`, `M = I·α + ω×Iω`
on each segment, recursively distal → proximal. The inputs are:

1. **Mass `m`, inertia `I`, COM offset** — from the personalized model.
2. **Body-frame angular velocity `ω`, acceleration `α`, COM linear acceleration `a_COM`**
   — derived by Simbody from the kinematic state (q, q_dot, q_ddot).

The 7-8× inflation could come from any of:

- **Spline q_ddot inflation**: even though body-frame physics is invariant,
  the NUMERICAL computation of α from spline-derived q_ddot may be
  poorly conditioned in ways that don't change with Cardan offset/sequence.
  (Worth verifying by inspecting q_dot/q_ddot trajectories around BR.)
- **The `setForcesFileName(id_sto) + setReplaceForceSet(True)` JR pipeline**:
  feeding ID's GenForces back as actuator inputs creates a constraint-vs-applied
  cancellation that's numerically delicate. Worth running JR without the
  ID actuator set to see if reactions drop materially.
- **Mass / inertia / COM properties of the personalized model**: gross
  values look plausible (humerus 2.4 kg, ulna 1.5 kg, hand 0.55 kg for an
  89.8 kg subject) but the inertia tensors and COM offsets weren't
  independently verified against V3D's RBI/anthropometric values.
- **Filter interaction**: the 20 Hz Butterworth filter on the .mot before
  ID interacts non-trivially with the splined derivatives. V3D's filter
  spec is supposedly the same but the implementations differ.
- **Frame-convention sign conventions**: even though V3D's RAR force is
  "in RAR (humerus) frame" and ours is "in humerus frame", the actual axis
  conventions of OpenSim's humerus_r body frame vs V3D's RAR segment frame
  may differ by a constant rotation. We'd then be comparing 3D vector
  magnitudes that are equal modulo a frame rotation — but the magnitudes
  should still match. So this only matters if V3D reports a SCALAR (e.g.
  a specific axis component) labeled as "FORCE_MAX", which we'd be
  computing incorrectly. Worth verifying what `SHOULDER_FORCE_MAX 700-1200`
  actually means in V3D's procdb.

## Reverting

Stock model is unchanged; variants are separate files (gitignored under
`recipes/`). Code toggle `SHOULDER_PARAM` defaults to `ZXY` (no-op). Code
left in for future experimentation.
