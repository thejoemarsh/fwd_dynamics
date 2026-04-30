# Joint reaction inflation — root cause and fix options

**Date**: 2026-04-30
**Branch**: `m2-kinetics`
**Context**: Follow-on investigation to `m2_kinetics_handoff.md`. After the
H1/H2/H3 audits and Fix A (`fc5d62e` + `be7f7f2`), kinetics on the throwing
arm are still inflated 5-7× over V3D. This doc records the diagnosis and
the three viable fix options, ranked.

## TL;DR

The throwing-arm CoordinateActuator feedback loop in OpenSim's
`AnalyzeTool` + `JointReaction` produces ~80% of the F amplification at
both elbow and shoulder. The pipeline-as-built is fundamentally
incompatible with high-velocity multi-DOF motion through gimbal lock —
not a bug we can patch, an architecture mismatch.

V3D produces sane numbers because it bypasses the constraint solver
entirely and computes joint reactions via direct segment Newton-Euler.
That's the long-term fix.

## Audit chain results (post-Fix A, post-Fix B)

```
Lowpass + elbow_flex_r at 16 Hz default:
  pelvis ω        1.02× V3D  ✓
  torso ω         1.08× V3D  ✓
  humerus_r ω     0.99× V3D  ✓
  ulna_r ω        2.06× V3D    (residual chart-cut)
  hand_r ω        2.74× V3D    (welded to ulna)

JR with no actuator feedback (H3):
  elbow F         0.97× V3D  ✓  ← physically correct
  shoulder F      2.16× V3D     ← still inflated even constraint-only
  elbow M         0.21× V3D    (no active torque available)
  shoulder M      0.00× V3D    (3-DOF acromial — all M is active)

JR with throwing-arm actuators excluded (B2):
  elbow F         1.29× V3D  ≈ matches no-feedback
  shoulder F      2.93× V3D  ≈ matches no-feedback

JR with full feedback (current pipeline):
  elbow F         4.92× V3D  ← +4.5× from arm actuators alone
  shoulder F      7.01× V3D  ← +4.8× from arm actuators alone
```

**The ~80% of amplification comes from the 5 right-arm
CoordinateActuators** (`arm_flex_r`, `arm_add_r`, `arm_rot_r`,
`elbow_flex_r`, `pro_sup_r`) being fed back through `setForcesFileName`
into `AnalyzeTool`. Excluding them collapses the kinetics back to
constraint-only levels.

## Root cause

`AnalyzeTool.setCoordinatesFileName` does not strictly prescribe
kinematics — it tracks them. With prescribed coords + active actuators,
the constraint solver computes reactions that include both:

1. The inertial reaction needed to maintain prescribed motion
2. The reaction needed to counter applied actuator torques

For a 3-DOF acromial CustomJoint with intrinsic Cardan ZXY axes during
late cocking → release, the joint Jacobian is ill-conditioned (X axis
passes through ±90°). Small kinematic-tracking errors are amplified
into huge constraint reactions when ID-computed actuator torques
(~1000 N·m each) are applied. Outside the throwing arm the same
mechanism produces small reactions — non-singular Jacobian, smaller
ID-computed torques, no amplification.

V3D's segment-by-segment Newton-Euler doesn't have a constraint
solver, doesn't apply ID-computed torques as actuator forces, and
doesn't go through the joint Jacobian at all. It computes
`F_proximal_on_segment = m·(a_COM - g) + F_distal` directly from
segment inertia and observed kinematics. No amplification mechanism
exists in that formulation.

## Fix options

### Option 1: Direct segment Newton-Euler reaction (V3D-style) — RECOMMENDED

Bypass `JointReaction` entirely. For each joint of interest, compute
reaction recursively from distal segments:

```
F_jr_i = m_i·(a_COM_i - g) + F_jr_(i+1)
M_jr_i = I_i·α_i + ω_i × (I_i · ω_i)
        + r_(COM_i → distal_jc) × F_jr_(i+1)
        + M_jr_(i+1)
        - r_(COM_i → proximal_jc) × F_jr_i
```

Inputs from OpenSim per-frame:
- COM position / velocity / acceleration in ground (per body)
- Angular velocity / acceleration in ground (per body)
- Joint center positions in ground
- Body-frame inertia tensors → rotated to ground

For our case: combined ulna+hand for the elbow reaction (welded wrist),
humerus for the shoulder reaction with `F_elbow` propagated.

**Pros**:
- Matches V3D's formulation exactly — directly comparable
- No constraint solver, no actuator feedback, no Jacobian
- Self-contained, well-understood biomechanics formulation
- ~100 lines of code in a new analysis module

**Cons**:
- Replaces, doesn't extend, the existing JR pipeline
- Requires manual segment-frame transformations for V3D-style F/M
  reporting (`SHOULDER_AR` in humerus frame, `SHOULDER_RTA` in torso
  frame, `ELBOW` in ulna frame)
- Welded-wrist requires combining ulna+hand mass/inertia/COM correctly

### Option 2: Splice no-feedback JR forces with ID GenForces for moments

Use `actuator_force_set_xml=None` JR for the full F vector (already
V3D-clean at the elbow, 2.16× V3D at shoulder). For M, take ID's
per-coord GenForces directly and combine with constraint M on locked
axes.

**Pros**:
- Smaller change (~20 lines) than Option 1
- Reuses existing `run_joint_reaction` and `run_inverse_dynamics`
- Solves the F side immediately

**Cons**:
- M side stays inflated 5-6× because ID GenForces are themselves
  inflated through the gimbal-lock Jacobian (the q̈ feeding ID is
  already amplified)
- Hybrid F-from-JR + M-from-ID is non-standard and obscures the
  underlying physics
- Doesn't match V3D formulation; we'd be inventing a new convention

### Option 3: Switch shoulder to BallJoint / quaternion representation

Replace the 3-DOF acromial CustomJoint with a `BallJoint` (quaternion-
based, no Jacobian singularity).

**Pros**:
- Eliminates the gimbal-lock root cause at the model level
- Should make the existing actuator-feedback pipeline well-conditioned
- Matches the Ralph swarm's `balljoint` candidate

**Cons**:
- Indirect evidence (Cardan permutation experiments A and C) suggests
  body-frame magnitudes don't actually depend on the axis choice — the
  inflation may persist even with a quaternion shoulder
- Loses interpretable joint angles (`shoulder_elv`, `shoulder_rot`);
  must remap from quaternion post-hoc for V3D comparison
- Big model surgery: parent_offset/child_offset, residual actuator XML,
  marker mappings, downstream coord-name dependencies
- The Ralph balljoint branch produced lower-body drift of 30-370%
  on iter 1, suggesting the swap is non-trivial to get right
- Doesn't address the deeper architectural mismatch between
  `AnalyzeTool` + actuator feedback and high-velocity prescribed motion

### Why Option 1 is the recommendation

The data from the audits paints a consistent picture: the existing
pipeline is amplifying through a mechanism (constraint solve under
ill-conditioned Jacobian + active actuator torques) that simply doesn't
exist in the V3D formulation. Patching around that mechanism (Option 2)
or trying to make it well-conditioned (Option 3) both leave the
fundamental architectural gap in place. Option 1 closes the gap by
adopting V3D's formulation, which is also what the published pitching
biomechanics literature uses.

## Implementation pointers (Option 1)

- New module: `src/theia_osim/analysis/segment_reactions.py`
- New audit: `scripts/audit_c_segment_newton_euler.py`
- Reuse `run_body_kinematics` for ω, ω̇ in ground frame
- Use `osim.Body.findStationLocationInGround(state, body.getMassCenter())`
  for COM in ground; `findStationAccelerationInGround` for a_COM in ground
- Use `osim.Joint.getChildFrame().getPositionInGround(state)` for
  joint centers in ground
- Inertia in ground: `R_body_to_ground · I_body · R_body_to_ground^T`
- Welded wrist: combine ulna + hand into a single segment with
  parallel-axis-corrected inertia about the combined COM

## What this doesn't fix

The remaining 2.16× shoulder F under no-feedback JR (and any
equivalent residual under Option 1) is still anomalous if it persists.
Given clean humerus ω (0.99× V3D), the residual likely comes from
either:

1. Inertia tensor or COM offset for humerus_r being subtly off
   despite the H1 audit (worth a partial recheck targeting humerus
   specifically with the combined ulna+hand mass propagated up)
2. The welded-wrist hand+ulna lump being treated as a single segment
   with a long lever arm that V3D may handle differently (V3D may
   compute hand and forearm separately even though our model welds
   them — segment-frame mismatch)

Both are second-order concerns to be revisited if Option 1's elbow
numbers come back V3D-clean but shoulder numbers stay at ~2× V3D.
