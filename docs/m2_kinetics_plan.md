# M2 Kinetics — implementation plan

**Status**: planned 2026-04-29, not yet started. Builds on M1 + M1.5 work
documented in [`kinematics_approach.md`](kinematics_approach.md).

## Context for a future contributor (or future-me)

By this point we have:
- M1: end-to-end Theia `.c3d` → joint-angle `.mot` via three recipes:
  Recipe A (marker IK), Recipe C (IMU IK), **Recipe D (analytical, V3D-style,
  no IK — canonical)**.
- M1.5: V3D-style per-athlete personalization (`model_build/personalize.py`):
  per-body scale factors from Theia segment lengths + de Leva 1996 mass and
  inertia overrides. Validated to match subject mass within 0.02 kg.
- Pelvis ω validation: Recipe D matches V3D's `PELVIS_ANGULAR_VELOCITY::Z`
  within 1.7% peak, RMSE 11.8 deg/s. X and Y axes are off (RMSE ~180 deg/s)
  due to OpenSim pelvis body frame ≠ V3D RPV body frame — fix is in M2.

M2 is the **kinetics** layer: forces and moments at each joint via
`InverseDynamicsTool` and `JointReaction` analyses, fed by Recipe D's `.mot`.
The marquee Driveline metrics (`ELBOW_MMT`, `SHOULDER_AR_FORCE`, etc.) live
here.

User-locked decisions for M2:
1. **Skip wrist** — LaiUhlrich2022's wrist is a `WeldJoint` (0 DOFs). No
   `WRIST_FORCE`/`WRIST_MMT` in M2. Revisit if/when we graft Saul 2015.
2. **Detect events from c3d ourselves** — port V3D's BR + MER detection
   (just for this trial as initial validation; needed eventually for batch).
3. **Fix pelvis frame mismatch in M2 prep work**, since the same per-body
   frame correction pattern will apply to shoulder/elbow downstream.
4. **No external loads (no force plates)** — V3D is doing inertial-only ID
   per Dapena's approach for upper-extremity pitching kinetics.
5. **Pass thresholds**: ±10% peak, RMSE under 5 N·m on time series for
   moments. ±10% peak on forces.
6. **Apply YABIN re-keying** so output names match V3D's procdb columns
   directly (`BACK_HIP_*`, `FRONT_KNEE_*`, `GLOVE_SHOULDER_*`).

## V3D event detection — verified algorithms

Sourced from `v3d_scripts/05_determine_side.v3s` (in repo). Documented for
porting parity.

### BR (Release)

```
1. Get hand DistEndVel for both hands (LHA + RHA segments).
   DistEndVel is the velocity of the hand's distal end (fingertip),
   computed from hand_4x4 origin + R @ axis × hand_length.
2. Transform velocity into VLB (mound-slope-corrected lab frame).
3. Lowpass 20 Hz Butterworth (V3D's standard 1-pass bidirectional).
4. Rectify (absolute value).
5. Build "release signal" — sum of forward-velocity components, gated:
     ( |L_hand_v_forward| > 12.5 m/s ) * |L_hand_v_forward|
   + ( |R_hand_v_forward| > 12.5 m/s ) * |R_hand_v_forward|
   In our pipeline (after Rx(-90°) + VLB), "forward" = OpenSim +X.
   In V3D's VLB, "forward" = +Y.
6. Search for max of release signal in [START_OF_RECORDING + 100, END].
7. BR = (peak frame) + 3-frame offset
   Ball release ≈ 3 frames after peak hand speed.
```

Threshold of 12.5 m/s gates out non-pitching motion. The +3-frame offset
captures the small mechanical lag between peak hand speed and ball release.

### Throwing side (R/L)

At the BR frame, compare each hand's forward speed; whichever is higher =
throwing side. Stored as `INFO::HAND` in V3D's procdb. We replicate.

### MER (MAX_EXT)

```
1. Compute throwing-arm shoulder angle in Y-X-Z Cardan order
   (DIFFERENT from default ZXY!) with NEGATEY=TRUE, NEGATEZ=TRUE.
   This is V3D's "Y-X-Z Euler shoulder" decomposition specifically tuned
   for capturing humeral external rotation cleanly.
2. Apply RESOLVE_DISCONTINUITY at the RAR_MIN frame (V3D-internal anchor)
   to unwrap 360° jumps.
3. Take Z component (humerus long-axis rotation = external rotation).
4. MER = global max of Z-component in window [BR - 50 frames, BR].
```

For us, `np.unwrap` doesn't need the RAR_MIN anchor — V3D needed it because
their `RESOLVE_DISCONTINUITY` requires a reference event. We can skip it.

## Pelvis frame mismatch — root cause

Now I see why the X/Y axes don't match V3D while Z does:

| Axis convention | OpenSim pelvis body | V3D RPV (Theia segment) |
|---|---|---|
| Long axis (up) | +Y | +Z |
| Anterior | +X | +X |
| Lateral right | +Z | +Y |

These differ by a **constant Rx(+90°)** about the body's own X axis (swap Y
and Z). For the same physical motion:

```
ω_in_v3d_rpv_frame = Rx(+90°) @ ω_in_opensim_pelvis_frame
```

If `ω_opensim = (a, b, c)`, then `ω_v3d = (a, c, -b)`.

This is a **free-vector frame conversion**, pose-independent. **The same
pattern applies to forces and moments** at every joint — JointReaction
output expressed in OpenSim child/parent body frame must be rotated to V3D's
body frame for comparison. This is why we fix it now: it sets up the
generic per-body correction infrastructure that downstream signals need.

## Implementation phases

Each phase = one focused commit with passing tests + working subsystem.

### Phase 1 — Hand velocity + side detection

Files to add:
- `src/theia_osim/events/hand_velocity.py` — compute hand distal-end position
  per frame from `hand_4x4` origin + R @ axis × length. Differentiate, lowpass,
  rectify. Returns `{'r_hand': (T, 3), 'l_hand': (T, 3)}` linear-velocity arrays.
  Hand length source: prefer MDH `Length_RHA/LHA`, fall back to Theia c3d
  `RHAND_LENGTH/LHAND_LENGTH`, fall back to 8 cm hardcoded.
- `src/theia_osim/events/side_detect.py` — `detect_throwing_side(trial,
  br_frame)` returns `'R'` or `'L'` by comparing hand forward speeds at BR.

Tests: `tests/test_hand_velocity.py`, `tests/test_side_detect.py`.

### Phase 2 — BR + MER event detection

Files to add:
- `src/theia_osim/events/pitch_events.py`:
  - `Events` frozen dataclass (`br_frame`, `br_time`, `mer_frame`, `mer_time`,
    `throwing_side`)
  - `detect_release(trial, hand_vel, sample_rate)` per V3D recipe above
  - `detect_max_external_rotation(trial, br_frame, throwing_side, sample_rate)`
    via Y-X-Z Cardan + np.unwrap + window max
  - `detect_events(trial, mdh)` orchestrator returning `Events`

Validation against V3D's procdb for `pose_filt_0.c3d`:
- V3D `BR_time = 1.593s`, `MER_time = 1.563s`
- Pass threshold: ±5 frames (~17 ms at 300 Hz)

Tests: `tests/test_pitch_events.py`.

### Phase 3 — Per-body frame corrections (fixes pelvis ω X/Y)

Files to add:
- `src/theia_osim/kinematics_postprocess/body_frame_corrections.py`:
  - `BODY_FRAME_TO_V3D: dict[str, np.ndarray]` — 3×3 rotation matrices.
    Pelvis: `Rx(+90°)`. Other bodies populated as they come online (likely
    same pattern: humerus, ulna, etc., all need similar Y↔Z swap).
  - `convert_body_frame_signal(arr_T_x_3, body_name) -> np.ndarray` —
    applies the rotation to a (T, 3) vector array.

Verify empirically before committing: take Recipe D's pelvis ω at BR (we
have it from prior runs), apply Rx(+90°), check it matches V3D's pelvis ω at
BR within a few deg/s. If not, the rotation direction is wrong — try Rx(-90°).

Driver wire-up: apply correction in `validation/compare.py` so V3D-vs-OpenSim
overlay plots show frame-corrected data.

Tests: `tests/test_body_frame_corrections.py` — verifying Rx(+90°) on a
known ω vector produces the expected output.

After this lands, Recipe D's pelvis ω X/Y RMSE should drop from ~180 deg/s
to single digits, matching the Z-axis result.

### Phase 4 — Inverse Dynamics

Files to add:
- `src/theia_osim/analysis/id.py`:
  - `run_inverse_dynamics(model_path, mot_path, out_sto_path, lowpass_hz=20.0)`
  - Wraps `osim.InverseDynamicsTool`
  - No external loads (no force plates)
  - 20 Hz Butterworth on coordinates per V3D filter spec

Tests: `tests/test_id.py` — smoke test on Recipe D's `.mot`, verify output
`.sto` has expected coordinate columns and no NaN.

### Phase 5 — JointReaction

Files to add:
- `src/theia_osim/analysis/jr.py`:
  - `run_joint_reaction(model, mot, id_sto, out_dir, joint_specs)` wrapping
    `AnalyzeTool` + `JointReaction` analysis.
  - `joint_specs` = list of `(joint_name, apply_on, express_in)` triples.
  - For shoulder, run two entries to get both V3D shoulder frames:
    - `(acromial_r, child, child)` → `SHOULDER_AR_*` (humerus frame)
    - `(acromial_r, child, parent)` → `SHOULDER_RTA_*` (torso frame)
  - For elbow: `(elbow_r, child, child)` → forearm frame
  - Plus glove-side mirrors

Tests: `tests/test_jr.py` — smoke test, output columns, no NaN.

### Phase 6 — Windowed metrics + YABIN re-keying

Files to add:
- `src/theia_osim/metrics/windowed.py`:
  - `max_in_window(signal, comp, t0, t1)`
  - `min_in_window(signal, comp, t0, t1)`
  - `value_at_event(signal, comp, event_t)`
- `src/theia_osim/kinematics_postprocess/yabin.py`:
  - Apply V3D's BACK/FRONT/GLOVE convention based on throwing side
  - Apply V3D sign-convention shims per ref doc §6:
    - `GLOVE_SHOULDER_AR_MMT`, `GLOVE_SHOULDER_RTA_MMT`: ×−1 on Z
    - `BACK_HIP_ANGLE`, `FRONT_HIP_ANGLE`: ×−1 on X
    - (etc., comprehensive list in `theia_pitching_signals_reference.md` §6)

Driver: compute `ELBOW_MMT_MAX`, `SHOULDER_AR_MMT_MAX`, `SHOULDER_RTA_MMT_MAX`,
`ELBOW_FORCE_MAX` etc. in window `[BR-50, BR+30]` (per V3D ref doc §13a).
Add to `summary.json`.

### Phase 7 — Kinetics V3D-comparison overlay

Refactor existing `validation/compare.py::compare_pelvis_omega` into a
generic `compare_signal(v3d_yabin_name, our_columns, label, frame_correction_body)`
that handles any V3D-vs-OpenSim time-series pair.

New comparison plots:
- `out/<run>/v3d_vs_recipe_d_elbow_mmt.png`
- `out/<run>/v3d_vs_recipe_d_shoulder_ar_mmt.png`
- `out/<run>/v3d_vs_recipe_d_shoulder_rta_mmt.png`
- (forces likewise)

Each shows V3D YABIN line vs our line on 3 components, with BR/MER markers,
RMSE + peak in title.

### Phase 8 — Documentation update

Update `README.md` to reflect M2 capabilities. Update `docs/kinematics_approach.md`
if any of the M2 work changes the calculus around when to revisit Path B.

## Validation targets

| Metric | V3D-typical (MLB FB) | Pass threshold |
|---|---|---|
| BR detection vs V3D's BR_time | exact frame | ±5 frames (~17 ms at 300 Hz) |
| MER detection vs V3D's MER_time | exact frame | ±5 frames |
| Pelvis ω X/Y/Z RMSE (after Phase 3 frame correction) | ω_z is ~12 deg/s already | < 50 deg/s on each axis |
| `ELBOW_MMT_MAX` | 60–100 N·m | ±10% peak, RMSE < 5 N·m time series |
| `SHOULDER_AR_MMT_MAX` (ext-rot moment) | 70–110 N·m | ±10% peak |
| `ELBOW_FORCE_MAX` | 700–1100 N | ±10% peak |
| `SHOULDER_*_FORCE_MAX` magnitudes | 700–1200 N | ±10% peak |

## Open questions resolved before coding

1. **Hand distal-end length source**: prefer MDH `Length_RHA/LHA`, fall back
   to Theia c3d `RHAND_LENGTH/LHAND_LENGTH` parameter (we already parse this
   in `theia_meta.py`), fall back to 8 cm hardcoded. Document the cascade in
   `hand_velocity.py` docstring.
2. **Throwing-side detection trust**: do it the V3D way for parity (compare
   hand speeds at BR, not "max anywhere in trial").
3. **Pelvis frame correction direction**: `Rx(+90°)` per the geometric analysis
   above, but **verify empirically before committing Phase 3**. Take a known
   peak-frame ω from V3D, apply correction, check it matches our output.
4. **Y-X-Z Cardan convention**: confirm scipy's intrinsic ordering matches
   V3D's. Decompose a known rotation both ways and check.

## Why this should "just work"

For upper-extremity kinetics, the critical chain is `pelvis → torso →
humerus → ulna → hand`. Inverse Dynamics computes forces from the **distal
end inward**, propagating mass × acceleration up the chain:

- Wrist/elbow/shoulder force at frame N depends on hand + forearm + (sometimes
  upper arm) mass × acceleration at that frame
- Pelvis-level errors don't propagate distally (kinetics flows distal →
  proximal in this analysis)
- Mass + inertia are correct (de Leva personalization is locked-in)
- Joint angles are correct (Recipe D matches V3D within 2% on the validated
  Z channel)

Expected: `ELBOW_MMT` magnitudes should match V3D within 5–10% on the first
try, possibly better after Phase 3 frame correction propagates to elbow/
shoulder body frames as well.

## File-by-file commit ordering

Each commit lands a working subsystem with tests:

1. `Add hand-velocity computation for event detection` (Phase 1)
2. `Add throwing-side autodetection` (Phase 1)
3. `Add BR + MER event detection (V3D-style)` (Phase 2)
4. `Add per-body frame corrections + fix pelvis ω X/Y mismatch` (Phase 3)
5. `Add InverseDynamicsTool driver` (Phase 4)
6. `Add JointReaction driver (SHOULDER_AR + SHOULDER_RTA, ELBOW)` (Phase 5)
7. `Add windowed-metric helpers + ELBOW_MMT_MAX, SHOULDER_*_MMT_MAX in summary.json` (Phase 6)
8. `Add YABIN re-keying` (Phase 6)
9. `Add kinetics V3D-comparison overlay plots` (Phase 7)
10. `Update README + docs/ for M2 features` (Phase 8)

## Pointers to existing code

| Need | Already exists at |
|---|---|
| Read Theia c3d → segment 4×4s | `src/theia_osim/c3d_io/reader.py` |
| Theia anthropometrics (RHAND_LENGTH etc.) | `src/theia_osim/c3d_io/theia_meta.py` |
| Parse V3D MDH | `src/theia_osim/c3d_io/mdh_parser.py` |
| de Leva tables + V3D ↔ OpenSim mappings | `src/theia_osim/model_build/anthropometrics.py` |
| Personalize model | `src/theia_osim/model_build/personalize.py` |
| Recipe D analytical kinematics | `src/theia_osim/kinematics_postprocess/cardan_from_4x4.py` |
| 20 Hz Butterworth (filtfilt) | `src/theia_osim/kinematics_postprocess/filter.py` |
| BodyKinematics wrapper | `src/theia_osim/analysis/body_kin.py` |
| Load V3D procdb JSON | `src/theia_osim/validation/load_v3d_json.py` |
| V3D-vs-our overlay plot | `src/theia_osim/validation/compare.py` |
| Driver CLI | `src/theia_osim/drivers/run_trial.py` |
| Sample trial fixtures | `pose_filt_0.c3d`, `pose_filt_0_procdb.json`, `theia_model.mdh` |
| V3D pipeline reference (THE map) | `theia_pitching_signals_reference.md` |
| V3D event-detection reference script | `v3d_scripts/05_determine_side.v3s` |

Run all three recipes against V3D ground truth (current canonical command):

```bash
uv run theia-osim-trial \
    --c3d pose_filt_0.c3d \
    --out out/all_recipes \
    --recipes a,c,d \
    --v3d-procdb pose_filt_0_procdb.json \
    --mdh theia_model.mdh
```

## Out of scope for M2

- Wrist kinetics (LaiUhlrich2022 wrist is welded; revisit when Saul 2015
  is grafted)
- Foot-plant detection (no force plate; V3D's velocity-threshold method is
  for windowing only — M3)
- Energy flow per Robertson & Winter (M3)
- Stride length / arm slot / timing metrics (M3)
- `_procdb.json`-format database export (M3)
- Multi-trial batch processing (M4)
