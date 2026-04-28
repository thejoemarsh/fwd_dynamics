# Kinematics approach — why we bypass OpenSim IK

**Status**: locked-in decision as of 2026-04-28. Path A (Recipe D) is the
canonical kinematics solver. Recipes A (marker IK) and C (IMU IK) are wired
in as cross-validators only.

## The problem we faced

Driveline's existing Visual3D pipeline produces a peak `PELVIS_ANGULAR_VELOCITY::Z`
of **491.7 deg/s** on `pose_filt_0.c3d`. Our initial OpenSim port (Recipe A —
marker IK against synthesized virtual markers) produced **846 deg/s** for the
same trial — 1.7× too high.

We chased filtering, scaling (`analysis/scale.py`), and full V3D-style anthropometric
personalization (`model_build/personalize.py`). None moved the pelvis ω peak.

## Why nothing was working

V3D and OpenSim solve the kinematics problem **fundamentally differently**.
This wasn't obvious until we read the C-Motion docs.

### OpenSim's approach (constrained IK)

- Markers + a kinematic tree (`pelvis ↔ femur ↔ tibia`, rigidly connected at
  joint centers)
- `InverseKinematicsTool` solves a least-squares fit of the tree to the markers
- Joint angles fall out of that fit
- The kinematic chain is **enforced** — the pelvis-to-femur connection is
  always exactly at the hip joint center, no exceptions

### Visual3D + Theia approach (independent segment tracking)

Confirmed via direct quotes from C-Motion's docs:

> "The Visual3D model is a 6DOF model, meaning that there are no joint
> constraints enforced within Visual3D and each segment is tracked with
> six degrees of freedom."
>
> "The Visual3D model does not utilize a static posture in its definition.
> All joint angles are calculated as the full angle values between the
> adjacent segments, with no baseline values subtracted."

The MDH segment definitions show the mechanism:

```
/USE_CAL_TARGETS_FOR_TRACKING=FALSE   ← NOT using marker fitting
/TRACKING_TYPES=ROTATION              ← track via rotation matrix directly
/TRACKING_NAMES=pelvis_4x4            ← straight from Theia's 4×4 signal
/DO_NOT_USE_LOCAL_TRANSFORMATION=TRUE
```

Each segment's pose at frame N **is** the corresponding Theia 4×4 at frame N.
Joint angles are computed *after* tracking as relative orientations between
adjacent segment pairs. Theia's own internal IK already glued the segments
upstream — V3D just trusts the result.

### Why our Recipe A inflated by 1.7×

OpenSim's IK has to satisfy the kinematic tree constraint AND minimize marker
residuals on every segment simultaneously. With our 3 pelvis markers (origin
+ two markers <10 cm offset), IK has poor leverage on pelvis orientation but
must keep the femur attached at the hip — so the IK trades pelvis rotation for
femur fit. The pelvis "spins" further than it physically did so other segments
can match their markers.

V3D never has this conflict because it doesn't fit anything — `pelvis_4x4` is
the pelvis pose, period.

## The decision

**Path A (Recipe D)**: replicate V3D's logic in OpenSim — bypass IK entirely.

For each frame:

1. Read every Theia segment 4×4 (slope-corrected, axis-swapped to OpenSim Y-up)
2. For each joint, compute relative rotation between adjacent body frames
3. Decompose into the joint's coordinate convention:
   - `CustomJoint` (3-DOF: hip, back, acromial, ground_pelvis): intrinsic Cardan **Z-X-Y**
   - `PinJoint` (1-DOF: ankle, mtp, elbow, radioulnar): project onto joint axis
   - `acromial_l`: axis 2 = -X, axis 3 = -Y → negate components 1 and 2 after decomposition
   - `walker_knee`: 1-DOF coupled, project onto X axis
   - `patellofemoral`, `subtalar`, `radius_hand`, `radioulnar`: skip (no independent
     Theia segment, leave at 0)
4. **Apply `np.unwrap`** to every rotational coordinate (V3D does the same via
   `Resolve_Discontinuity` per ref doc §4) — without this, the angle wrap at
   ±180° causes BodyKinematics' state-derived ω to spike to >50,000 deg/s
5. Write a `.mot` file directly. **No `InverseKinematicsTool` call.**
6. Feed the `.mot` to `BodyKinematics`, `InverseDynamics`, `JointReaction` as usual

### Validation

Run on `pose_filt_0.c3d` against V3D's `pose_filt_0_procdb.json`:

| Component | V3D peak (deg/s) | Path A peak | Difference |
|---|---|---|---|
| Pelvis ω_x | 352 | 370 | +5% |
| Pelvis ω_y | 323 | 339 | +5% |
| **Pelvis ω_z (long axis = pitcher rotation)** | **491.7** | **500.1** | **+1.7%** |
| Pelvis ω_z RMSE over the whole trial | — | — | **11.8 deg/s** |

Z component is essentially identical (within noise). X/Y residuals are
attributable to OpenSim pelvis body frame ≠ V3D RPV body frame on the X/Y
axes (Z axes happen to coincide because both point "up"). Per-segment
frame-mapping rotation can fix this in M2 if needed.

## Why this is OK long-term

We researched OpenSim forum + docs to verify Path A doesn't foreclose
predictive biomechanics later:

- **`ForwardTool`, `CMCTool`, `MocoTrack`, `MocoStudy` are provenance-agnostic.**
  They consume a `.mot` of joint coordinates without checking how it was
  derived. `ForwardTool.cpp` in opensim-core literally just reads initial
  states and integrates.
- **Markerless OpenSim communities routinely do FD/CMC from non-IK kinematics.**
  - **OpenSense** (IMU IK) is the OpenSim-blessed pipeline for kinematics from
    segment orientations — same data shape as ours
  - **OpenCap** (PLOS Comp Bio 2023) runs muscle-driven forward simulations from
    pose-estimation kinematics
  - **Pose2Sim** users routinely run ID and Moco
  - **`theia3d_to_osim`** on SimTK exists specifically for this use case
- **Driveline themselves have published CMC pitching analysis** in 2017
  (drivelinebaseball.com/2017/03/computed-muscle-control-analysis-pitching-mechanics)
- **The real gating concern for FD/CMC isn't IK provenance — it's dynamic
  consistency with ground reaction forces.** This bites any kinematics source.
  Standard fix is `RRA` (Residual Reduction Algorithm) or `MocoTrack` with low
  coordinate-tracking weight. We'd run RRA before CMC regardless of which
  kinematics path we use.

## Known downsides of Path A

For descriptive biomechanics (joint angles, ω, ID, JointReaction), none of
these are blockers. They matter mostly if/when we move to forward dynamics
or muscle-driven simulation.

1. **No kinematic-chain constraint enforcement.** Theia's IK already glued the
   segments, so in practice this rarely matters. But if Theia ever produces a
   frame where r_thigh's hip end and pelvis's r-hip end disagree by 5cm, we
   silently report a "joint that translates" — same as V3D does. OpenSim's IK
   would have caught it and forced a compromise.

2. **Tied to LaiUhlrich2022's joint axis conventions.** `JOINT_DEFS` in
   `cardan_from_4x4.py` is hand-built to match LaiUhlrich's Z-X-Y Cardan order,
   sign-flips for the L arm, etc. Different OpenSim model = regenerate the
   table.

3. **Pelvis ω X and Y components don't match V3D yet.** Z (the metric Driveline
   cares about most) is dead-on. X and Y differ by ~180-200 deg/s RMSE because
   the OpenSim pelvis body frame's X/Y axes are rotated relative to V3D's RPV.
   Fixable with a per-body frame-mapping rotation; deferred.

4. **`np.unwrap` is mandatory.** Without it, BodyKinematics ω spikes >50,000
   deg/s at Cardan wraps. V3D does the same.

5. **OpenSim's downstream tools see a model state that's not IK-validated.**
   Inverse Dynamics and JointReaction work fine (they read the `.mot` and
   compute kinetics from F=ma). But the state isn't constrained-tree-consistent
   in OpenSim's strict sense — for FD/CMC, plan to run RRA first.

## When to revisit (move to Path B / static-pose calibration for IK)

Specific triggers:

- **Forward Dynamics or CMC fails** with kinematic inconsistency errors that
  RRA can't absorb. Path B's calibrated IK output would give a constrained-tree-
  consistent state.
- **Closed-loop constraint** is added to the model (e.g. swap to a model with
  a coupled patellofemoral or radioulnar that we can't skip). Analytical coords
  won't satisfy the closed loop; IK does.
- **Different OpenSim model with non-orthogonal joint axes** that can't be
  cleanly Cardan-decomposed.
- **Forensic/clinical use case** demanding the kinematic tree be enforced as
  a sanity check on Theia's output.

## Pointers to the implementation

| File | What it does |
|---|---|
| `src/theia_osim/kinematics_postprocess/cardan_from_4x4.py` | Path A core — `JOINT_DEFS`, `compute_coordinates()`, `write_recipe_d_mot()` |
| `src/theia_osim/c3d_io/reader.py` | Reads Theia `.c3d` → `TrialData` (4×4s + meta) |
| `src/theia_osim/c3d_io/slope.py` | Applies `vlb_4x4.T` mound-slope correction |
| `src/theia_osim/c3d_io/mdh_parser.py` | Parses V3D `.mdh` for subject mass/height/lengths |
| `src/theia_osim/model_build/anthropometrics.py` | de Leva 1996 male tables (mass fractions, k_xx/yy/zz) |
| `src/theia_osim/model_build/personalize.py` | Geometry scaling + de Leva mass/inertia override |
| `src/theia_osim/analysis/body_kin.py` | Wraps `AnalyzeTool + BodyKinematics` |
| `src/theia_osim/validation/compare.py` | V3D-vs-OpenSim overlay plots + RMSE |
| `src/theia_osim/drivers/run_trial.py` | CLI: `theia-osim-trial --c3d ... --recipes a,c,d` |

To run all three recipes against V3D ground truth:

```bash
uv run theia-osim-trial \
    --c3d pose_filt_0.c3d \
    --out out/all_recipes \
    --recipes a,c,d \
    --v3d-procdb pose_filt_0_procdb.json \
    --mdh theia_model.mdh
```

Outputs land in:
- `out/all_recipes/recipe_d/analytical.mot` — the canonical joint-angle file
- `out/all_recipes/recipe_d/body_kin/*_vel_bodyLocal.sto` — pelvis ω
- `out/all_recipes/v3d_vs_recipe_d_pelvis_omega.png` — overlay plot
- `out/all_recipes/summary.json` — peaks + RMSE per recipe

## Reference reading

- Visual3D / Theia integration: <https://www.theiamarkerless.com/blog/working-with-theia-markerless-data-in-visual3d>
- C-Motion ROTATION data type: <https://wiki.has-motion.com/doku.php?id=visual3d:documentation:c3d_signal_types:rotation_data_type>
- C-Motion IK docs: <https://wiki.has-motion.com/doku.php?id=visual3d:documentation:kinematics_and_kinetics:inverse_kinematics>
- OpenSim CMC: <https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089706/How+CMC+Works>
- OpenSense (IMU IK): <https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53084203/OpenSense+-+Kinematics+with+IMU+Data>
- OpenCap (PLOS Comp Bio 2023): <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462>
- de Leva 1996 anthropometric tables: J. Biomech 29(9), 1223-1230
- Theia C3D file format: <https://docs.theiamarkerless.com/theia3d-documentation/data-formats/c3d-files>
- This repo's V3D pipeline reference: `theia_pitching_signals_reference.md` (one level up)
