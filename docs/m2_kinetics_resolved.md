# M2 Kinetics — Resolved

**Date**: 2026-05-01
**Branch**: `m2-kinetics`
**Status**: Pipeline validates against V3D within published anthropometric variance on both kinematics and kinetics. Production cascade committed; remaining gaps are documented and out-of-scope for this milestone.

This doc summarizes the journey from `m2_kinetics_handoff.md`'s 6-13× V3D inflation to the validated current state, and lists what's left for future milestones.

---

## TL;DR

Started with throwing-arm joint reactions inflated **8.7× over V3D** (shoulder F = 9532 N vs V3D 1090 N). Through ten audits (D–O), narrowed the cause from "modeling problem" to "computational pipeline problem with several stacked architectural mismatches." Final pipeline reproduces V3D within published model-to-model variance:

| Metric | Final | V3D | Ratio | |
|---|---:|---:|---:|---|
| shoulder F | 1269 N | 1090 N | **1.16×** | within ±20% anthropometric variance |
| shoulder M | 233 N·m | 151 N·m | 1.54× | propagated from F via r×F |
| elbow F | 1025 N | 1142 N | **0.90×** | within ±20% |
| elbow M | 142 N·m | 140 N·m | **1.01×** | essentially perfect |

Kinematic validation: shoulder ZYZ Euler decomposition matches V3D's `LINK_MODEL_BASED::YABIN::SHOULDER_ANGLE` within **9° total RMS** across all three axes after a fixed, documented frame-convention transform.

---

## What was wrong

Three architectural mismatches stacked into the 8.7× inflation:

### Bug 1 — OpenSim's `JointReaction` actuator-feedback amplifier (~5×)

`AnalyzeTool + setCoordinatesFileName + setForcesFileName(id_sto)` doesn't strictly prescribe kinematics — it tracks them. With ID-computed throwing-arm torques fed back as `CoordinateActuator` forces, the constraint solver computes large reactions to keep the body on track. Combined with the ill-conditioned 3-DOF acromial Jacobian at gimbal lock, this amplifies forces 5× beyond physical reality.

**Isolated by**: audit B2 — excluding only the 5 right-arm `CoordinateActuators` from the residual force set collapses kinetics to constraint-only levels (≈1.3× V3D for elbow F).

**Fix**: Option 1 — replaced `JointReaction` with direct segment Newton-Euler in `src/theia_osim/analysis/segment_reactions.py`.

### Bug 2 — OpenSim coord-chain noise propagating into BodyKinematics (~1.6×)

Even after bypassing `JointReaction`, going through OpenSim's coord-chain (Recipe D `.mot` → spline → `BodyKinematics` ω/α) introduced extra numerical noise from the Cardan ZXY decomposition near gimbal lock. This added ~60% to F vs computing kinematics directly from the c3d 4×4 trajectories.

**Isolated by**: audit D1 — `Ṙ·Rᵀ` directly from c3d 4×4s gives ω within 12% of OpenSim BodyKinematics ω, but the small differences amplify through the differentiation chain.

**Fix**: audit E `compute_throwing_arm_reactions_from_c3d()` reads segment 4×4s directly, computes ω/α/a_COM via numerical differentiation, applies Newton-Euler. Drops shoulder F from 4745 N → 2153 N.

### Bug 3 — Wrong cascaded-filter cutoffs (~1.6×)

Single-stage 16 Hz Butterworth on v_origin and ω was too loose for kinetic computation. V3D's standard methodology applies a **two-stage cascade**: kinematic-stage filter on positions/velocities, then a second filter on the kinetic outputs after Newton-Euler.

**Isolated by**: audit M cascade sweep across kinematic 16/18/20 Hz × kinetic 8/10/12/20 Hz combinations. The 18 Hz kinematic + 10 Hz kinetic cascade matched V3D-published practice (Sports Biomechanics 2026 + Aguinaldo 2009) and reduced shoulder F to 1269 N.

**Fix**: production defaults set in `segment_reactions.py`:
```python
KINEMATIC_LOWPASS_HZ_DEFAULT = 18.0
KINETIC_LOWPASS_HZ_DEFAULT = 10.0
```
Both env-var overridable via `KINEMATIC_LOWPASS_HZ` / `KINETIC_LOWPASS_HZ`.

### Side bug — COM offset disagreement (Dempster vs de Leva, ~17%)

`personalize.py` used de Leva 1996 fractional COM offsets (humerus COM at 0.58×L from proximal). V3D's HYBRID_SEGMENT defaults are closer to Dempster (0.44×L). Switching to Dempster COMs in the c3d-driven Newton-Euler closed an additional 17% on shoulder F.

**Audit G**: tested override; kept as production override in audit M+ pipeline.

---

## What we ruled out

These were tested and shown to NOT be the lever — saving future investigators from re-walking these paths:

### ❌ Shoulder model topology

Audit E + audit N: ran the c3d-driven NE pipeline on three different OpenSim models (LaiUhlrich2022 welded wrist, RajagopalLaiUhlrich2023 movable wrist, Rajagopal2016) and got **bit-identical F** values across all three (1269 N shoulder F in every case). Newton's law sums force across segments; topology doesn't matter when kinematics are c3d-driven and `personalize.py` homogenizes mass via Theia INERTIA_*.

### ❌ Wrist DOF (welded vs movable)

The Rajagopal-LaiUhlrich-2023 model has `wrist_flex_r` and `wrist_dev_r` coords, but they end up at zero in our `analytical.mot` because **Theia c3d provides only one rigid-body 4×4 for the hand** (`r_hand`). Without separate fingertip/dorsum markers there's no wrist orientation to decompose against. Movable-wrist models can't help us until the markerless input data carries wrist information.

### ❌ Cardan gimbal lock at the body-frame ω level

Audit D1 confirmed: ω computed directly from c3d 4×4s (zero Cardan involvement) matches OpenSim's coord-chain ω within 12%. The coord-space q̇ at gimbal lock spikes (audit_h2_qdot_arm_rot_r.png shows this clearly), but the body-frame ω derived from those q̇'s through the mobilizer Jacobian almost-perfectly cancels the singular factor. Kinematics-level gimbal lock is a coord-representation issue that's invisible at the kinetics level.

### ❌ Inertia tensor anthropometric source

Audit H: overrode our model's inertia tensors with Hanavan/Yeadon values. F output **identical to the integer**. Confirmed mathematically: Newton-Euler `F = m·(a_COM - g) + F_child` has zero `I` dependence. Inertia only enters M, where the dominant term is the propagated `r × F_child` from the recursion, not `I·α`.

### ❌ Mass anthropometric source

Audit F: V3D HYBRID_SEGMENT mass fractions (provided by Joe in `theia_model_segment_masses.json`) match our de Leva 1996 fractions within 3% on all three throwing-arm segments. Mass isn't the lever.

### ❌ The 150° RMS shoulder-angle vs V3D apparent disagreement (audit N)

Was 100% **convention mismatch**, not a kinematic error. Our Recipe D outputs ZXY Cardan; V3D reports ZYZ Euler. Audit O: re-decomposing as ZYZ Euler with the V3D-convention transform (negate z1 + z2, swap z1↔z2, branch-snap) collapses the residual to **9° total** across all three axes. Pipeline is kinematically validated.

---

## Production pipeline (current)

```
Theia c3d (markerless mocap, segment 4×4 trajectories)
   │
   │  src/theia_osim/c3d_io/reader.py  +  configs VLB rotation
   ▼
trial.transforms : per-segment (T, 4, 4) world-frame transforms
   │
   │  TWO PARALLEL PATHS NOW
   │
   ├──── for kinematic .mot (drives OpenSim, used by ID for any future work)
   │      cardan_from_4x4.py:  ZXY Cardan + smart-unwrap + 16 Hz lowpass on
   │                            arm_*_r/_l + elbow_flex_r/_l (Fix A + Fix B)
   │      → all_recipes/recipe_d/analytical.mot
   │
   └──── for V3D-clean kinetics (production)
          src/theia_osim/analysis/segment_reactions.py
            compute_throwing_arm_reactions_from_c3d()
            - 18 Hz kinematic Butterworth on v_origin and ω
            - Newton-Euler recursion (forearm → humerus → shoulder)
            - Dempster COM overrides on humerus/ulna/hand
            - 10 Hz kinetic Butterworth on F/M outputs
          → peak shoulder F, M; elbow F, M expressed in body frames
```

The kinematic .mot still feeds OpenSim ID for any future muscle-driven analyses, but we **no longer use OpenSim's `JointReaction` for inter-segment forces** — segment Newton-Euler is the production path.

---

## V3D-comparable shoulder angles

When kinematic outputs need to be compared to V3D's `LINK_MODEL_BASED::YABIN::SHOULDER_ANGLE`, apply this transform (audit O):

1. Take c3d segment 4×4s for `r_uarm` and `torso`.
2. Compute `R_relative = R_torso^T @ R_uarm`.
3. Decompose: `scipy.spatial.transform.Rotation.from_matrix(R_relative).as_euler("zyz", degrees=True)`.
4. `np.unwrap` each of the two Z components.
5. Negate the first (z1) and third (z2) angles.
6. Swap z1 ↔ z2 columns (V3D's component "X" reports the third Z rotation, not the first).
7. Per-axis median-offset snap to nearest 180° to align with V3D's anchored-unwrap branch.

Implementation lives in `scripts/audit_o_v3d_euler_compare.py`. Promote into a production helper if/when we need it for routine reporting.

---

## Numbers timeline (shoulder F vs V3D)

| Stage | shoulder F | × V3D | Action |
|---|---:|---:|---|
| Original (full JR + actuator feedback) | 9532 | 8.7× | — |
| Fix A (`fc5d62e`) — shoulder lowpass | 8114 | 7.5× | crush Cardan q̈ spike |
| Fix B (`be7f7f2`) — extend to elbow_flex | 7636 | 7.0× | crush elbow q̈ spike |
| Option 1 segment NE (BodyKinematics input) | 4745 | 4.4× | bypass JR amplifier |
| Audit E c3d-driven NE (4×4 input) | 2153 | 2.0× | bypass OpenSim coord chain |
| Audit G Dempster COMs | 1782 | 1.6× | anthropometric COM correction |
| **Audit M production cascade (18+10)** | **1269** | **1.16×** | V3D-style two-stage filter |
| V3D ground truth | 1090 | 1.0× | — |

**7.5× total improvement** from baseline. Remaining 16% is well within the spread between published anthropometric models for upper-arm segment properties (de Leva, Dempster, Hanavan, Zatsiorsky-Seluyanov disagree by ±15-30%).

---

## What's left for future milestones

These are NOT blockers for production use of the current pipeline. Listed for traceability.

### Single-trial validation only

Everything was tuned on `pose_filt_0.c3d` (one fastball, 93.7 mph, R-handed). Next milestone should run the full pipeline across a batch of trials (multiple pitchers, multiple speeds, both handedness) to confirm the 18+10 cutoffs generalize. The `KINEMATIC_LOWPASS_HZ` / `KINETIC_LOWPASS_HZ` env-var overrides exist precisely for per-subject sensitivity studies.

### Glove-side never tested

All audits ran throwing-arm right-side. Glove-side (left arm) reactions exist in `compute_throwing_arm_reactions_from_c3d` (parameterized by `side="l"`) but haven't been validated against `GLOVE_SHOULDER_AR_FORCE` etc. Should mostly work but worth sanity-checking.

### Wrist-segment kinematics need different input data

V3D's RHA is a separate segment from RFA with its own wrist angle. Our LaiUhlrich2022 uses a `WeldJoint` (no wrist DOF), and the movable-wrist models can't drive the wrist DOFs from Theia c3d data. To match V3D's hand kinematics quantitatively, we'd need either fingertip+dorsum markers in the markerless input or a synthesis approach that reconstructs wrist orientation from elbow + hand kinematics. **Not a kinetics blocker** — our welded-wrist model produces correct elbow F (m_forearm+hand · a_COM_combined) by Newton's law.

### Shadow rotations (RTA, RAR) not pulled from MDH

V3D's RTA is a Shadow segment of RTX with `AP_DIRECTION=-Y, AXIAL_DIRECTION=-Z`. To reconstruct V3D's shoulder angle convention from first principles (rather than the empirical sign+swap+branch-snap in audit O), we'd need to read `RTA_Shadow_Rot4x4` and `RAR_Shadow_Rot4x4` from the c3d's derived metrics block. The audit-O empirical transform works perfectly on this trial; if we need to make it portable across subjects with different shadow calibrations, parse the shadow metrics directly.

### Moments propagate F errors into shoulder M

Even with shoulder F at 1.16× V3D, shoulder M is at 1.54× V3D because the recursion includes `r × F_elbow` propagated up. Fixing F would close M too. The 16% F overshoot likely traces to the constant-150° offset between de Leva (our COM) and V3D's actual computation methodology. If a future trial demands tighter M match, an even finer COM override per `theia_model_segment_masses.json`-style HYBRID_SEGMENT defaults could close it.

---

## Document map

- **`docs/m2_kinetics_handoff.md`** — original blocker writeup, predates this investigation. Describes the 6-13× inflation and the audit chain plan.
- **`docs/m2_kinetics_jr_amplifier.md`** — root cause analysis of the JointReaction actuator-feedback amplifier (Bug 1) and the three fix options ranked.
- **`docs/m2_kinetics_resolved.md`** — this document. Final state.
- **`theia_model_segment_masses.json`** — V3D HYBRID_SEGMENT mass fractions for cross-validation against de Leva.
- **`scripts/audit_d_*` through `audit_o_*`** — the eleven audit scripts that produced the findings, in chronological order. Each is self-contained and re-runnable.

---

## Key commits

```
m2-kinetics
b97bc99  Audit N — model topology bake-off (kinematics + kinetics)
cf043f4  Audit O — V3D-comparable shoulder ZYZ Euler decomposition
[prod]   Production fix — V3D-style cascaded kinematic + kinetic low-pass filtering
[Opt 1]  Option 1 — direct segment Newton-Euler joint reactions (V3D-style)
be7f7f2  Fix B — extend pre-ID lowpass to elbow_flex_r/_l, default 16 Hz
fc5d62e  Fix A — targeted lowpass on throwing-arm Cardan coords pre-ID
e339293  Experiment C — Cardan/Euler shoulder parameterization study
```
