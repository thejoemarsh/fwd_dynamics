# M2 Kinetics — work-in-progress status

**As of 2026-04-29.** Working on branch `m2-kinetics` (off `main`). Phases 1-5
of the [M2 plan](m2_kinetics_plan.md) are committed; Phase 5 surfaced a
real blocker that needs a decision before continuing.

This doc is written so a future contributor (LLM or human) can pick up
without re-running the investigation.

## Quick orientation

- **Goal of M2**: produce V3D-equivalent joint forces/moments
  (`ELBOW_MMT`, `SHOULDER_AR_FORCE`, etc.) by feeding Recipe D's
  analytical `.mot` through OpenSim's `InverseDynamicsTool` +
  `JointReaction`.
- **Status**: framework code is built and BR/MER events validate to
  0-frame error vs V3D. **Joint kinetics for the throwing arm don't
  validate** — see "The blocker" below.
- **Plan to read**: [`docs/m2_kinetics_plan.md`](m2_kinetics_plan.md)
  has the original 8-phase plan. Validation targets, V3D event-detection
  algorithms, and pointers all live there.
- **Background context**: [`docs/kinematics_approach.md`](kinematics_approach.md)
  documents why we use Recipe D (V3D-style analytical, no IK) instead of
  OpenSim IK. Critical for understanding the M1 baseline.

## Branch state

```
m2-kinetics  (6 commits ahead of main)

c6d689a Add JointReaction driver + smart Cardan unwrap
7ae9524 Add InverseDynamicsTool driver
c9e27f9 Add per-body frame corrections + fix pelvis ω X/Y mismatch
0481bc2 Add BR + MER event detection (V3D-style)
dac1932 Add throwing-side autodetection
c34ee39 Add hand-velocity computation for event detection
```

All 6 commits passed smoke checks against `pose_filt_0.c3d` +
`pose_filt_0_procdb.json` (V3D ground truth).

## What's been implemented (commits 1-5 of the 10-commit plan)

### Commit 1 — `events/hand_velocity.py` ✓

Computes per-frame fingertip world position + 20 Hz lowpass velocity for
both hands. Distal-end direction is **−Z** in the Theia hand segment
frame (verified empirically: r_hand z-axis · forearm-direction = −0.97
across the trial on `pose_filt_0.c3d`). Hand length cascade: MDH
`Length_RHA/LHA` → Theia c3d INERTIA_*[0] → 8 cm fallback.

Smoke check on `pose_filt_0.c3d`:
- R fingertip peak forward speed: 24.59 m/s
- L fingertip peak forward speed: 5.05 m/s
- At V3D-reported BR (frame 478), R fingertip 22.4 m/s

### Commit 2 — `events/side_detect.py` ✓

`detect_throwing_side(hand_vel, br_frame) → 'R' | 'L'` — V3D parity
(higher forward speed at BR wins). Matches `INFO::HAND='R'`.

### Commit 3 — `events/pitch_events.py` ✓

`detect_release()` and `detect_max_external_rotation()` per
`v3d_scripts/05_determine_side.v3s`:

- BR: peak of `release_signal = sum_over_hands((|fwd_speed| > 12.5) * |fwd_speed|)`
  in `[start+100, end]`, plus 3-frame mechanical-lag offset.
- MER: shoulder relative rotation (torso → throwing humerus) decomposed
  in **Y-X-Z Cardan** with NEGATEY=NEGATEZ=TRUE, Z component is humeral
  external rotation, np.unwrap'd, argmax in `[BR-50, BR]`.

Result on `pose_filt_0.c3d`:
```
BR  frame=478  t=1.5933s  (V3D: 1.5933s)  → 0-frame error
MER frame=469  t=1.5633s  (V3D: 1.5633s)  → 0-frame error
side: R                                     (V3D: R)
```

### Commit 4 — `kinematics_postprocess/body_frame_corrections.py` ✓

Per-body free-vector frame conversion (OpenSim body frame → V3D segment
frame). The same correction will apply to JR force/moment outputs in M2.

```python
BODY_FRAME_TO_V3D = {
    "pelvis": np.diag([-1.0, -1.0, 1.0]),  # 180° about Z
}
convert_body_frame_signal(arr_T_x_3, body_name)  # returns rotated copy
```

User confirmed pelvis correction by tracing recipe_d overlay plot:
"X needs to be made negative on the opensim side, Y needs to be made
negative on the opensim side, Z is good." Applied that diagonal sign
flip; pelvis ω RMSE vs V3D dropped per axis:
- X: 178 → 12.7 deg/s
- Y: 185 → 9.2 deg/s
- Z: 12 → 12 (already aligned)

Wired into `validation/compare.py` so all overlay plots show
frame-corrected data by default. `compare_pelvis_omega(...,
apply_frame_correction=True)` is the new default.

### Commit 5 — `analysis/id.py` ✓

Wraps `osim.InverseDynamicsTool`. No external loads (V3D pipeline is
inertial-only, no force plates). 20 Hz coordinate lowpass via OpenSim's
built-in filter. New arg `exclude_muscles: bool = True` excludes the
80-muscle ForceSet — standard practice for residual-only ID.

Smoke: 661 frames × 36 generalized-force columns, no NaN.

### Commit 6 — `analysis/jr.py` + `analysis/residual_actuators.py` + smart Cardan unwrap ✓ (with caveat)

Three pieces:

1. **`analysis/jr.py`** — `run_joint_reaction(model, mot, id_sto, out_dir,
   specs, actuator_force_set_xml)`. `JRSpec` is `(joint, on_body,
   in_frame)`. `default_specs(sides=("r","l"))` returns the V3D-equivalent
   probe set:
   - `acromial_*` ON child IN child  → SHOULDER_AR (humerus frame)
   - `acromial_*` ON child IN parent → SHOULDER_RTA (torso frame)
   - `elbow_*`    ON child IN child  → ELBOW (forearm frame)

2. **`analysis/residual_actuators.py`** — `build_coord_actuator_force_set()`
   emits a ForceSet XML with one CoordinateActuator per ID-emitted
   coordinate. Actuators are named **literally** `<coord>_moment` /
   `<coord>_force` so column names in ID's `.sto` match what JR
   looks up. Coupled coords (motion_type=3, e.g. `knee_angle_*_beta`)
   are skipped because ID doesn't write a column for them. JR loads
   this XML via `setReplaceForceSet(True)` + `setForceSetFiles([xml])`.

3. **`kinematics_postprocess/cardan_from_4x4.py`** — `_smart_unwrap_cardan_zxy()`.
   At gimbal lock (`X ≈ ±90°`) the Cardan ZXY decomposition has a
   2-fold ambiguity:
   ```
   (Z, X, Y)  ≡  (Z+180°, 180°−X, Y+180°)
   ```
   Per-component `np.unwrap` can't pick between branches. New algorithm:
   at each frame, evaluate both candidate branches against the
   previously-unwrapped frame and pick the closer one (component-wise
   ±360° representative chosen first). Applied to `ground_pelvis`,
   `hip_*`, `back`, `acromial_*`. Dropped throwing-shoulder peak
   `|vel|` from 21,374 → 5,816 deg/s on `pose_filt_0.c3d`.

## The blocker

**Joint kinetics (Phase 5 validation against V3D) don't pass** for the
throwing-arm `acromial_r` and downstream `elbow_r`. Two recipes, two
different failure modes:

| Path | Result vs V3D-typical |
|---|---|
| **Recipe D** (smart-unwrapped) → ID → JR | `elbow_flex_r_moment` peak 839 N·m vs V3D ~70-100 (≈10× inflated). Shoulder JR resultant 1771 N·m vs V3D ~70-110. |
| **Recipe A** (marker IK) → ID → JR | ID alone is plausible (`elbow_flex_r_moment` 21 N·m, `arm_flex_r_moment` 96 N·m). **JR blows up** to ~10²⁰⁰ N at 8 scattered frames; non-NaN values are also wrong. |

### Root cause (Recipe D)

LaiUhlrich2022's `acromial_*` is a 3-DOF CustomJoint with intrinsic
Cardan Z-X-Y axes. During the throw, `arm_add_r` passes through the
gimbal-lock region (X ≈ ±90°). Smart unwrap correctly handles the
branch swap *as a representation* — the unwrapped coordinates encode
the right rotation matrices — but the **joint Jacobian itself is
ill-conditioned** in that neighborhood. ID's generalized force is
computed by projecting the full 3D inter-segment moment onto each joint
coordinate axis; near gimbal lock, two of those axes become nearly
parallel, so projections inflate.

Evidence:
- Body-frame angular velocities (from `BodyKinematics`) are physically
  correct: `humerus_r` peak 5429 deg/s, `ulna_r` peak 7880 deg/s —
  matches V3D pitcher ranges.
- Pelvis / hip / lumbar / leg ID GenForces are sane — they don't go
  through gimbal lock.
- Cardan ZXY q_dot peak `arm_flex_r` is 5816 deg/s post-smart-unwrap,
  but the implicit Jacobian inversion in ID inflates the GenForce.

### Root cause (Recipe A)

Less clear. Marker IK fits the kinematic tree, distributing marker
residuals across all segments. JR's failure mode (NaN at scattered
frames + huge magnitudes elsewhere) suggests numerical instability in
the constraint-satisfaction step at certain configurations. Not yet
investigated in depth — Recipe A is the validator, not the canonical
path.

### What works regardless

- All lower-body and torso kinetics from Recipe D look plausible
  (peaks at hip / knee / ankle / lumbar, no obvious blowup)
- Pelvis ω X/Y/Z all match V3D within ~13 deg/s after frame correction
- Event detection is exact

## The decision pending from the user

User was offered three options at end of last session; no choice locked
in yet:

**A) Continue Phase 6-8 with limitation documented.** Ship the M2
framework. Lower-body kinetics validate. Shoulder/elbow flagged for a
later milestone.

**B) Pivot to V3D-style Newton-Euler in Python.** Bypass OpenSim ID/JR
for the upper-extremity chain. Compute per-segment `F=ma`, `M=Iα + ω×Iω`
in body frame, sum recursively distal → proximal (forearm + hand →
elbow → upper arm + → shoulder). This is exactly what V3D does. Needs:

- Per-segment ω, α, a_COM derivation from segment 4×4 derivatives
- Segment masses + inertias (already in personalized model — pull via
  `osim.Model().getBodySet().get(body)`)
- Per-segment COM position in body frame (already in
  `personalize_report.body_masses_kg` and equivalent)
- Recursive Newton-Euler routine
- ~1 commit's worth of code

Recommended by previous-session-me. The hand-velocity machinery is
already most of the per-segment kinematics computation.

**C) Pause and try a different shoulder joint representation.**
Quaternion-based or alternate Cardan order. Research-level. Requires
modifying LaiUhlrich2022.osim and updating `cardan_from_4x4.py`'s
`JOINT_DEFS` to match. Highest risk, also most architecturally
ambitious.

## Reproducing the failure

Once you've checked out `m2-kinetics` and synced (`uv sync`):

```bash
# Re-run Recipe D end-to-end (writes out/all_recipes/)
uv run theia-osim-trial \
    --c3d pose_filt_0.c3d \
    --out out/all_recipes \
    --recipes a,c,d \
    --v3d-procdb pose_filt_0_procdb.json \
    --mdh theia_model.mdh
```

Then to see the kinetics blowup:

```bash
uv run python <<'PY'
from pathlib import Path
from src.theia_osim.analysis.id import run_inverse_dynamics
from src.theia_osim.analysis.jr import run_joint_reaction
from src.theia_osim.analysis.residual_actuators import build_coord_actuator_force_set

out = Path("out/m2_phase5_check_v3"); out.mkdir(exist_ok=True, parents=True)
xml = build_coord_actuator_force_set(
    "out/all_recipes/theia_pitching_personalized.osim",
    out / "residual_actuators.xml")
id_sto = run_inverse_dynamics(
    "out/all_recipes/theia_pitching_personalized.osim",
    "out/all_recipes/recipe_d/analytical.mot",
    out / "inverse_dynamics.sto")
jr_sto = run_joint_reaction(
    "out/all_recipes/theia_pitching_personalized.osim",
    "out/all_recipes/recipe_d/analytical.mot",
    id_sto=id_sto, out_dir=out, actuator_force_set_xml=xml)
print(jr_sto)
PY
```

Inspect: `elbow_r_on_ulna_r_in_ulna_r_m{x,y,z}` peaks should be ~7-100 N·m
each (V3D-typical); they'll come out ~150-1300.

## Phases not yet started (commits 7-10 of the plan)

These are gated on the kinetics decision above. Phases 6-7 only make
sense once we know what kinetics path we're on:

- **Phase 6a** — `metrics/windowed.py`: `max_in_window`,
  `min_in_window`, `value_at_event`. Compute `ELBOW_MMT_MAX`,
  `SHOULDER_*_MMT_MAX` etc. in `[BR-50, BR+30]` and add to
  `summary.json`.
- **Phase 6b** — `kinematics_postprocess/yabin.py`: BACK/FRONT/GLOVE
  re-keying based on throwing side; sign-convention shims per
  `theia_pitching_signals_reference.md` §6.
- **Phase 7** — generic `compare_signal()` in `validation/compare.py`
  for V3D-vs-our overlay plots on any time-series pair (not just
  pelvis ω).
- **Phase 8** — README + docs update.

## File map

### Implemented this milestone
- `src/theia_osim/events/hand_velocity.py` — fingertip pos/vel, length cascade
- `src/theia_osim/events/side_detect.py` — throwing-side at BR
- `src/theia_osim/events/pitch_events.py` — BR + MER orchestrator (`detect_events`)
- `src/theia_osim/kinematics_postprocess/body_frame_corrections.py` — per-body OS→V3D rotations
- `src/theia_osim/analysis/id.py` — InverseDynamicsTool wrapper
- `src/theia_osim/analysis/jr.py` — JointReaction wrapper + JRSpec + default_specs
- `src/theia_osim/analysis/residual_actuators.py` — CoordinateActuator XML builder
- `src/theia_osim/kinematics_postprocess/cardan_from_4x4.py` — adds `_smart_unwrap_cardan_zxy`

### Modified
- `src/theia_osim/validation/compare.py` — applies pelvis frame correction by default
- `src/theia_osim/kinematics_postprocess/cardan_from_4x4.py` — smart unwrap on Cardan ZXY joints

### Pre-existing dependencies (don't re-build)
- `src/theia_osim/c3d_io/{reader,slope,theia_meta,mdh_parser}.py`
- `src/theia_osim/model_build/{anthropometrics,personalize,add_markers}.py`
- `src/theia_osim/import_pipeline/{landmarks,recipe_a_trc,recipe_c_sto}.py`
- `src/theia_osim/analysis/{ik,body_kin,scale}.py`
- `src/theia_osim/kinematics_postprocess/filter.py` — 20 Hz Butterworth filtfilt
- `src/theia_osim/validation/load_v3d_json.py` — V3D procdb parser
- `src/theia_osim/drivers/run_trial.py` — top-level CLI

## Validation references

- V3D ground truth: `pose_filt_0_procdb.json` (loaded via
  `validation/load_v3d_json.py`)
- V3D events: `BR_time=1.593s`, `MER_time=1.563s`, `INFO::HAND='R'`,
  `PITCH_VELO=937` (subjective text field)
- V3D-typical pitching MLB ranges (from M2 plan validation table):
  - Pelvis ω peak: ~500 deg/s long axis
  - `ELBOW_MMT_MAX`: 60-100 N·m
  - `SHOULDER_AR_MMT_MAX`: 70-110 N·m
  - `ELBOW_FORCE_MAX`: 700-1100 N
  - `SHOULDER_*_FORCE_MAX`: 700-1200 N

## Where to start picking back up

1. Read this doc, `m2_kinetics_plan.md`, and `kinematics_approach.md`
2. Check the user's last reply for option A/B/C choice
3. If **B** (recommended): build `kinetics_postprocess/newton_euler.py`
   that takes a `TrialData` + `osim.Model` and returns per-segment
   F/M time series in body frame, then per-joint reaction by
   distal→proximal recursion. Compare directly to V3D YABIN.
4. If **A**: skip to Phase 6a; explicitly mark `acromial_*` and
   `elbow_*` outputs as "framework only — magnitudes inflated, see
   docs/m2_status.md"
5. If **C**: deep work, propose a specific replacement joint first
