# M2 Kinetics — Investigation Handoff

**Repo**: `/home/yabin/code/fwd_dynamics`
**Branch**: `m2-kinetics`
**Date snapshot**: 2026-05-01
**Status**: Pipeline runs end-to-end. Kinematics validate against V3D. **Throwing-arm kinetics are inflated by 6-13× vs V3D ground truth and we don't yet know why.** Multiple coordinate-parameterization fixes have been tried and ruled out. Need fresh eyes on the actual inflation source.

This document is self-contained: anyone with access to the repo + a running OpenSim 4.6 install can reproduce every number here.

---

## TL;DR for a reviewer

- We're computing baseball-pitching joint kinetics (forces + moments at shoulder and elbow) from Theia markerless mocap data via OpenSim's `InverseDynamicsTool` + `JointReaction` analyses.
- Visual3D (V3D) is the ground-truth reference. V3D's procdb output contains the canonical kinetic signals via the `LINK_MODEL_BASED::YABIN::*` records.
- **Lower body and pelvis: validates against V3D fine.** Pelvis angular velocity peak matches V3D within ~14% on Z; ID generalized forces look reasonable on hip/knee/ankle.
- **Throwing-arm shoulder + elbow: forces inflated 5.8-7.8×, moments inflated 6.5-12.9× vs V3D.**
- **Time of peaks matches V3D exactly. Signal shape matches. Only the magnitude is wrong.**
- We have ruled out coordinate-parameterization issues (Cardan ZXY, frame rotation, alternate Cardan order — all produce the same body-frame magnitudes ±10%).
- The smoking-gun clue is **moments are inflated MORE than forces** (12-13× vs 7-8×), which suggests an extra torque-arm or inertia error on top of any acceleration/mass error. Pure scale issues would inflate forces and moments by the same factor.

---

## Pipeline architecture

The Recipe D path that produces the kinetic outputs:

```
pose_filt_0.c3d  (Theia 300 Hz markerless mocap, 661 frames, 2.20 s)
   │
   │  src/theia_osim/c3d_io/reader.py — read_theia_c3d()
   │  Apply VLB slope correction (mound tilt removed)
   ▼
trial.transforms : dict[segment_name -> (T, 4, 4)]   per-segment 4×4 transforms in Z-up Theia world
   │
   │  src/theia_osim/kinematics_postprocess/cardan_from_4x4.py — write_recipe_d_mot()
   │  Per joint: R_relative = R_parent.T @ R_child
   │  Decompose intrinsic Cardan ZXY (LaiUhlrich2022's acromial axes) → Euler angles
   │  smart-unwrap branch handling for gimbal lock (helpful for arm_flex_r)
   ▼
recipe_d/analytical.mot   661 frames × 36 generalized coordinates
   │
   │  Personalize the model:
   │  src/theia_osim/model_build/personalize.py
   │  Apply de Leva 1996 mass / inertia per body, scale segments to subject limb lengths.
   ▼
theia_pitching_personalized.osim   89.8 kg subject, 1.88 m height; humerus_r 2.43 kg, ulna_r 1.45 kg, hand_r 0.55 kg
   │
   │  src/theia_osim/analysis/id.py — run_inverse_dynamics()
   │  osim.InverseDynamicsTool, 20 Hz Butterworth lowpass on coordinates
   │  exclude muscles (residual-only ID)
   ▼
inverse_dynamics.sto   per-coordinate generalized forces (N·m for rotational, N for translational)
   │
   │  src/theia_osim/analysis/residual_actuators.py — build_coord_actuator_force_set()
   │  Generate ForceSet XML with one CoordinateActuator per ID-emitted coord
   │  Names match ID column headers literally (`<coord>_moment` / `<coord>_force`)
   │
   │  src/theia_osim/analysis/jr.py — run_joint_reaction()
   │  osim.AnalyzeTool with JointReaction analysis
   │  setReplaceForceSet(True) + setForceSetFiles([actuator_xml])
   │  setForcesFileName(id_sto) feeds ID GenForces back as actuator inputs
   │  Default specs: SHOULDER_AR (humerus frame), SHOULDER_RTA (torso frame), ELBOW (forearm frame)
   ▼
joint_reaction_JointReaction_ReactionLoads.sto   3D F/M time series per joint per frame
```

**Reproducer**: `scripts/repro_shoulder_kinetics.py`. One command runs the full pipeline + emits `peaks.json` + `peaks.txt`:

```bash
uv run python scripts/repro_shoulder_kinetics.py \
    --src-model data/models/recipes/LaiUhlrich2022_full_body.osim \
    --out-dir out/repro_baseline
```

---

## V3D ground truth (from `pose_filt_0_procdb.json`)

V3D's YABIN folder contains time-series kinetic signals computed via V3D's own `Compute_Model_Based_Data /FUNCTION=JOINT_FORCE` and `JOINT_MOMENT` calls. The procdb stores 600 frames covering t=0.003 to t=2.0s of the same trial.

Peak values (3D vector magnitude in throw window):

| Signal | Frame convention | Peak \|3D\| | When | Notable |
|---|---|---:|---|---|
| `ELBOW_FORCE` | RFA (forearm) | **1142 N** | BR | Z dominant: +1099 N at BR (forearm long-axis distractive) |
| `SHOULDER_AR_FORCE` | RAR (humerus) | **1090 N** | BR | Z dominant: +1037 N at BR |
| `SHOULDER_RTA_FORCE` | RTA (torso) | **1178 N** | BR | X dominant: −1040 N at BR |
| `ELBOW_MMT` | RFA | **140 N·m** | MER | Y: +109 N·m at MER (valgus / flexion moment) |
| `SHOULDER_AR_MMT` | RAR | **151 N·m** | post-BR | X: −127 N·m |
| `SHOULDER_RTA_MMT` | RTA | **137 N·m** | BR | Y: −130 N·m |

These align with published pitching biomechanics literature. ~120% of bodyweight (881 N) for shoulder distractive force = ~1057 N — matches V3D's 1037 N at BR.

V3D's events (also in the procdb):
- BR (release) at t = 1.593 s, frame 479 in our trial
- MER (max external rotation) at t = 1.563 s, frame 470

V3D's kinematic shoulder representation (from `v3d_scripts/02_CMO_v6_v1.v3s` lines 242-282):
- `RT_SHOULDER_ANGLE`: AXIS1=Z, AXIS3=Z (intent: Z-Y-Z Euler).
- The procdb-stored `YABIN::SHOULDER_ANGLE` has X range [−108°, +43°], Y range [+25°, +106°], Z range [−70°, +188°].
- Peak Z = +188° at MER. Stays positive through BR (+126°) — V3D applies `RESOLVE_DISCONTINUITY(period=360°, anchor=AR_MIN_event)`.

---

## What our pipeline produces

Reproduced from `out/repro_baseline/kinetics/joint_reaction_JointReaction_ReactionLoads.sto`:

| Signal | Our peak | When | V3D peak | **Inflation** |
|---|---:|---|---:|---:|
| ELBOW F (ulna frame) | 6650 N | t=1.593s | 1142 N | **5.82×** |
| SHOULDER_AR F (humerus frame) | 8525 N | t=1.597s | 1090 N | **7.82×** |
| SHOULDER_RTA F (torso frame) | 8525 N | t=1.597s | 1178 N | **7.24×** |
| ELBOW M (ulna frame) | 901 N·m | t=1.593s | 140 N·m | **6.45×** |
| SHOULDER_AR M (humerus frame) | 1771 N·m | t=1.570s | 151 N·m | **11.71×** |
| SHOULDER_RTA M (torso frame) | 1771 N·m | t=1.570s | 137 N·m | **12.93×** |

**The peak time is correct (BR ± 1 frame). The signal SHAPE is correct (single sharp peak around BR with smooth pre-/post- decay). Only the MAGNITUDE is wrong.**

See `out/diagnostics_kinetics_v3d_vs_ours.png` for time-series overlays.

### Lower-body kinetics (for context — these are OK)

V3D pelvis Z angular velocity peak: 491 deg/s. Ours: 500 deg/s after frame correction (3% over).
V3D pelvis X/Y RMSE after frame correction: 12.7 / 9.2 deg/s.
ID generalized forces on hip/knee/ankle (throw window): plausible magnitudes.

---

## Notable structural finding: moments inflated more than forces

This is the most diagnostically important number in the document.

| | Force inflation | Moment inflation |
|---|---:|---:|
| Elbow | 5.8× | 6.5× |
| Shoulder AR | 7.8× | 11.7× |
| Shoulder RTA | 7.2× | 12.9× |

If the inflation were purely a "kinematics is too fast" issue (ω, α inflated), forces and moments would scale by the SAME factor (both linear in α and ω²). They don't. Moments scale by ~1.5-1.7× MORE than forces.

That extra factor on moments suggests a torque-arm or inertia-tensor magnitude error layered on top of whatever else is going on:

- **M = I·α + ω×(I·ω) + r×F.** If r (the COM offset from joint center, or the moment arm) is ~1.7× too large, the M contribution scales up by ~1.7× while F is unaffected.
- **I scales as m·r².** If r is wrong, I scales quadratically with the same error.

So **a torque-arm / COM-offset error on humerus_r and ulna_r in the personalized model is the leading hypothesis**.

---

## Experiments performed (and what each ruled out)

### Experiment: ROM-penalty smart unwrap (FALSIFIED)

**Hypothesis**: smart-unwrap drift to non-canonical Cardan branches (e.g. `arm_flex_r` reaching −334° during the throw) inflates spline-derived q̈, breaking ID.

**Setup**: 54-candidate per-frame search over Cardan branch identity + ±360° wraps; cost = smoothness + λ · max(0, |q| − ROM)².

**Result**: With ROM penalty, the algorithm forces canonical-interval values, which introduces ±360° **chart-cut discontinuities** at ±180° crossings (frames 446, 480 in this trial). The 20 Hz lowpass smears these into ~15-frame q̈ pulses, **inflating shoulder F from ~8000 N to ~226000 N (28× worse)**.

**Conclusion**: Multi-rev Cardan drift in q is NOT the cause. ID processes coordinates by smoothness, not interpretability.

**Reverted**. Code: see commit `eaee527` history.

### Experiment A: Acromial Ry(+60°) frame rotation (FALSIFIED)

**Hypothesis**: Cardan ZXY hits gimbal lock (`arm_add_r` peaks at 82.8°, 7° from the X = ±90° singularity). A symmetric Ry(+60°) rotation on parent_offset and child_offset of acromial_r/l moves the singularity 34° off the throwing trajectory while preserving resting pose.

**Setup**: Variant `LaiUhlrich2022_full_body_yroll60.osim` built via `scripts/build_yroll_variant.py`. Matching similarity transform applied in `cardan_from_4x4.py` via env var `ACROMIAL_Y_OFFSET_DEG=60`. Singularity successfully moved: peak |arm_add_r| dropped 82.8° → 55.7° (34° margin).

**Result**:
| Metric | Baseline | Exp A | Δ |
|---|---:|---:|---:|
| Shoulder F | 9532 N | 8679 N | **-9%** |
| Shoulder M | 1940 N·m | 2098 N·m | +8% |

JR magnitudes essentially unchanged. ID GenForces redistributed between `arm_flex_r_moment` (848→1933) and `arm_add_r_moment` (1559→594) but total moment magnitude preserved — that's coordinate axes rotating, same body-frame physics.

**Conclusion**: JR reports body-frame quantities, which are coordinate-system-invariant. Frame rotation cannot change body-frame magnitudes. **Singularity-conditioning is not the dominant cause.**

Commit `675a5bd`. Writeup at `docs/experiments/expA_axis_rotation.md`.

### Experiment C: Cardan/Euler sequence change (FALSIFIED + structural finding)

**Hypothesis**: A different Cardan/Euler sequence with its singularity outside the throwing trajectory would resolve the inflation. V3D uses Z-Y-Z Euler for shoulder kinematics (Euler with 1st=3rd axis); peak |middle β| = 106° = 73° from the 0°/180° singularity.

**Setup attempted**: build a ZYZ Euler variant of the model.

**Hard structural finding**: OpenSim's `CustomJoint` rejects collinear axes at model load:

```
RuntimeError: ... CustomJoint 'acromial_r' has collinear axes and are not well-defined. Please fix and retry loading.
```

So **any 3-axis Euler sequence (1st axis = 3rd axis) is structurally unavailable in OpenSim CustomJoint**. ZYZ, ZXZ, XYX, XZX, YXY, YZY all fail to load. We're limited to the 6 Cardan permutations.

**Best Cardan available**: XZY. Peak |middle β| = 49°, 41° margin to ±90°. Anatomically maps to (rotation1=X=abduction, rotation2=Z=horizontal abduction, rotation3=Y=internal rotation), which is the user's preferred shoulder coordinate naming.

Built `LaiUhlrich2022_full_body_xzy_anatomical.osim` with renamed coords (`shoulder_abd_r`, `shoulder_hzn_r`, `shoulder_int_rot_r`).

**Result**: Shoulder F dropped from 9532 → 8805 N (−8%). Same noise band as Exp A.

**Conclusion**: All 6 Cardan permutations within OpenSim CustomJoint produce body-frame magnitudes within ~10% of each other. Coordinate parameterization is **decisively** not the lever.

Commit `e339293`. Writeup at `docs/experiments/expC_cardan_sequence.md`.

### Subsidiary finding: gimbal-lock 52° spike at BR

`out/diagnostics_shoulder_er_signal.png` shows the throwing-arm humeral axial rotation in three decompositions:

| Decomposition | max framewise |Δ| through throw |
|---|---:|
| ZXY arm_rot_r (current default) | **52.4°** at t=1.603s |
| XZY shoulder_int_rot_r (anatomical variant) | 11.8° |
| ZYZ axial (V3D-style, can't load in OpenSim) | 14.7° |

The 52° single-frame jump in the default ZXY pipeline at t=1.603s (3 frames after BR) is a Cardan chart-cut artifact. The smart-unwrap is doing its best — the underlying parameterization is non-smooth at this configuration. ID's GCV-spline turns the spike into a ~10-frame q̈ pulse around BR.

This is real but not the dominant cause: switching to XZY (4.4× smoother kinematics) didn't fix the kinetics inflation.

### Subsidiary finding: V3D's `RESOLVE_DISCONTINUITY`

V3D applies `RESOLVE_DISCONTINUITY(signal, period=360°, anchor=AR_MIN_event)` to shoulder Y-X-Z humeral axial rotation for MER detection (script `05_determine_side.v3s` line 589). This is anchored np.unwrap.

For the shoulder kinematic representation in `02_CMO_v6_v1.v3s`, the same anchored unwrap appears to be applied implicitly (V3D's reported Z range [−70°, +188°] sits in a single 360° window despite a 258° physical sweep, which means an anchor was set somewhere).

It would not have helped Experiment A or Exp C — those were body-frame magnitude problems, not 360°-wrap problems.

---

## What we have ruled OUT

1. **Time alignment / event detection bug** — V3D and our peaks fall on the same frames.
2. **Signal-shape pipeline bug** — qualitative shapes overlay V3D's nearly perfectly through the throw.
3. **Lower-body kinematic bug** — pelvis ω validates within 13 deg/s on Z.
4. **Coordinate parameterization** — Cardan ZXY (stock), XZY (best Cardan), Ry(+60°) frame rotation all produce within 10% of the same body-frame magnitudes. Three independent tests.
5. **Cardan unwrap drift** — ROM penalty experiment proved this conclusively. The 226000 N result rules out the entire class of "smarter unwrap" fixes.
6. **OpenSim joint type misalignment** — JR's `default_specs()` keys joints by name (`acromial_r`, `elbow_r`) and frame role (child/parent) — verified to be joint-type-agnostic.

---

## What we have NOT yet ruled out (priority-ordered hypotheses)

### H1: Personalized-model COM offsets / inertia for humerus_r and ulna_r are wrong (HIGH)

**Why**: Moments inflated 1.5-1.7× more than forces — points squarely at a torque-arm / inertia-magnitude error.

**What to check**: After running Recipe D, load `theia_pitching_personalized.osim` in a Python session. For humerus_r and ulna_r, dump:
- mass, COM position relative to body origin, inertia tensor (in body frame)
- compare against de Leva 1996 fractions of segment mass and `ρ_COM/L` ratios for these segments
- compare against what V3D's MDH file (`theia_model.mdh`) says for these segments

The personalize code is at `src/theia_osim/model_build/personalize.py`. de Leva tables at `src/theia_osim/model_build/anthropometrics.py`. Segment lengths come from Theia c3d's `INERTIA_*` parameters via `c3d_io/theia_meta.py`.

**Specific suspect**: scaling the inertia tensor when scaling the segment. Inertia about COM scales as m·r², so a length scale factor of `s` should multiply the inertia diagonal by `s²` (about COM, with mass scaled separately) — easy to mis-apply as `s³` or `s` instead.

### H2: OpenSim humerus / ulna body-frame ω is inflated (MEDIUM)

**Why**: We've validated PELVIS_ANGULAR_VELOCITY against V3D within 14% on Z. We have NOT validated SHOULDER_ANGULAR_VELOCITY or ELBOW_ANGULAR_VELOCITY. V3D's procdb has both as `LINK_MODEL_BASED::YABIN::SHOULDER_ANGULAR_VELOCITY` (peak ~4500 deg/s) and `ELBOW_ANGULAR_VELOCITY` (peak ~3000 deg/s).

**What to check**: Edit `src/theia_osim/drivers/run_trial.py` to add `humerus_r, ulna_r, hand_r, torso` to the `bodies=` tuple of `run_body_kinematics()`, re-run, and compare body-frame ω to V3D's YABIN signals.

If our ω matches V3D within ~5%, kinematic input is fine and the issue is in mass/inertia/COM (H1).

If our ω is 1.5-3× higher than V3D, the issue is upstream — in Recipe D's segment 4×4 → coord computation, or in OpenSim's ω derivation from the .mot.

### H3: JR's `setForcesFileName + setReplaceForceSet` cancellation pipeline (MEDIUM)

**Why**: JR works by feeding ID's GenForces back as CoordinateActuator inputs and having OpenSim solve for constraint forces to maintain the prescribed motion. The constraint-force ↔ applied-actuator-force cancellation can be numerically unstable when ID is itself stressed near gimbal lock.

**What to check**: Run JR with `actuator_force_set_xml=None` (don't replace the muscle ForceSet, or replace it with an empty one). Reactions will be physically incomplete (don't include the active joint torques) but the magnitudes will be different. If they collapse toward V3D-typical values, the cancellation pipeline is the amplifier.

Code: `src/theia_osim/analysis/jr.py:130-147`.

### H4: V3D vs OpenSim segment frame convention difference (LOW)

**Why**: The 3D vector magnitude is frame-rotation-invariant, so a frame difference cannot explain a 7-13× magnitude gap. But it COULD explain why our component breakdown looks different from V3D's:

- V3D `ELBOW_FORCE` Z=+1099 N (axial along forearm), Y=−164 N (small)
- Ours `elbow_r_on_ulna_r_in_ulna_r`: y=+3596 N appears axial-ish, z=+4225 N

Frame conventions don't fix the inflation, but auditing them is necessary for any point-by-point comparison.

OpenSim's ulna body Y axis is along the forearm long axis (proximal to distal). V3D's RFA segment Y is conventionally the same. But there could be a 90° rotation between them (e.g. axial along Z in V3D vs Y in OpenSim).

### H5: Inflated q̈ at BR from the 52° gimbal-lock spike (LOW)

The chart-cut spike at t=1.603s pollutes ID's q̈ for ~10 frames around BR. This is real but the XZY anatomical variant (4.4× smoother) didn't reduce kinetics, so it's not the dominant lever.

---

## Repository pointers for the reviewer

### Pipeline code (the kinetics chain)

- `src/theia_osim/c3d_io/reader.py` — Theia c3d → segment 4×4s
- `src/theia_osim/kinematics_postprocess/cardan_from_4x4.py` — segment 4×4s → joint coordinate `.mot` (Recipe D)
- `src/theia_osim/model_build/personalize.py` — apply de Leva mass/inertia + segment scaling (**H1 suspect**)
- `src/theia_osim/model_build/anthropometrics.py` — de Leva 1996 lookup tables
- `src/theia_osim/analysis/id.py` — `run_inverse_dynamics()` wrapper around `osim.InverseDynamicsTool`
- `src/theia_osim/analysis/residual_actuators.py` — builds CoordinateActuator ForceSet XML for JR
- `src/theia_osim/analysis/jr.py` — `run_joint_reaction()` wrapper (**H3 suspect**)
- `src/theia_osim/drivers/run_trial.py` — top-level CLI orchestrator

### Validation / reproducer

- `scripts/repro_shoulder_kinetics.py` — one-command end-to-end runner; emits `peaks.json` + `peaks.txt` + smoke-test of CoordinateSet/JointSet
- `scripts/build_cardan_variant.py` — generates Cardan variants of the .osim
- `scripts/build_yroll_variant.py` — generates Ry-rotation variants
- `scripts/build_zyz_variant.py` — kept as documentation of OpenSim's ZYZ rejection

### Trial fixtures

- `pose_filt_0.c3d` — sample pitching trial (300 Hz, 661 frames)
- `pose_filt_0_procdb.json` — V3D ground truth (events, kinematic + kinetic time series, metric peaks)
- `theia_model.mdh` — V3D MDH file with subject-specific mass/height/segment lengths

### V3D reference

- `v3d_scripts/02_CMO_v6_v1.v3s` — V3D pipeline that computes the YABIN signals (joint angles + kinetics)
- `v3d_scripts/05_determine_side.v3s` — V3D's BR/MER event detection + RESOLVE_DISCONTINUITY usage

### Documentation

- `docs/m2_status.md` — original M2 blocker writeup (predates this investigation)
- `docs/m2_kinetics_plan.md` — original 8-phase M2 plan
- `docs/kinematics_approach.md` — Recipe D rationale
- `docs/experiments/expA_axis_rotation.md` — Ry(+60°) frame-rotation experiment writeup
- `docs/experiments/expC_cardan_sequence.md` — Cardan-sequence sweep + ZYZ rejection writeup
- **`docs/m2_kinetics_handoff.md`** — this document

### Diagnostic plots (in `out/`)

- `out/diagnostics_kinetics_v3d_vs_ours.png` — **the key plot for the inflation question.** V3D YABIN forces/moments vs our JR outputs, plus angular velocities. 6 panels.
- `out/diagnostics_kinetics_comparison.png` — JR/ID across baseline + Exp A + Exp C (shows all three are ~equivalent inflation)
- `out/diagnostics_shoulder_er_signal.png` — shoulder humeral axial rotation 3-way decomposition + framewise discontinuity (the 52° spike at BR)
- `out/diagnostics_shoulder_yxz_match_v3d.png` — V3D YABIN::SHOULDER_ANGLE vs our YXZ Cardan (kinematic, not kinetic)

### Output directories from each experiment

- `out/repro_baseline/` — stock ZXY pipeline, peaks.json + .txt + JR/ID .sto
- `out/repro_expA_yroll60/` — Ry(+60°) frame rotation
- `out/repro_expC_xzy/` — XZY Cardan sequence (pre-rename)
- `out/repro_xzy_anatomical/` — XZY with anatomical coord names

### Git commits relevant to this investigation

```
e339293 Experiment C — Cardan/Euler shoulder parameterization study
675a5bd Experiment A — acromial axis-rotation (Ry+60°). FAILED to fix kinetics
07b9876 Add scripts/repro_shoulder_kinetics.py + ignore .moa/
eaee527 Add M2 work-in-progress status doc
c6d689a Add JointReaction driver + smart Cardan unwrap
7ae9524 Add InverseDynamicsTool driver
```

---

## Specific data tables for the reviewer to verify

### Subject

- mass: 89.81 kg
- height: 1.88 m
- handedness: R
- pitch velocity: 93.7 mph (per V3D `INFO::PITCH_VELO`)

### Personalized body masses (from `out/repro_baseline/all_recipes/summary.json`)

```
pelvis: 10.03 kg     femur_r: 12.72   femur_l: 12.72
tibia_r: 3.89        tibia_l: 3.89
calcn_r: 1.23        calcn_l: 1.23
toes_r:  0.09        toes_l:  0.09
torso:  33.18 kg
humerus_r: 2.43      humerus_l: 2.43
ulna_r:    1.45      ulna_l:    1.45
hand_r:    0.55      hand_l:    0.55
total: 89.83 kg  (vs 89.81 reported — 0.02 kg over, fine)
```

These are de Leva 1996 fractions of total mass × subject mass. Total mass matches subject mass to within rounding. **What is NOT verified is whether the COM offsets and the inertia tensors of humerus_r / ulna_r match V3D's computation.**

### Reproduction commands

```bash
# Capture baseline
uv run python scripts/repro_shoulder_kinetics.py \
    --src-model data/models/recipes/LaiUhlrich2022_full_body.osim \
    --out-dir out/repro_baseline

# Run Exp A
uv run python scripts/build_yroll_variant.py --angle-deg 60 \
    --dst data/models/recipes/LaiUhlrich2022_full_body_yroll60.osim
ACROMIAL_Y_OFFSET_DEG=60 uv run python scripts/repro_shoulder_kinetics.py \
    --src-model data/models/recipes/LaiUhlrich2022_full_body_yroll60.osim \
    --out-dir out/repro_expA_yroll60

# Run Exp C
uv run python scripts/build_cardan_variant.py --seq XZY --rename-coords \
    --dst data/models/recipes/LaiUhlrich2022_full_body_xzy_anatomical.osim
SHOULDER_PARAM=XZY uv run python scripts/repro_shoulder_kinetics.py \
    --src-model data/models/recipes/LaiUhlrich2022_full_body_xzy_anatomical.osim \
    --out-dir out/repro_xzy_anatomical
```

---

## Concrete next-step plan (priority-ordered)

The reviewer should focus on **H1 (mass/inertia/COM audit)** as the highest-information-per-time experiment. Three specific checks:

### Step 1: Audit personalized model inertial properties (~30 min)

```python
import opensim as osim
m = osim.Model("out/repro_baseline/theia_pitching_personalized.osim")
m.initSystem()
for body_name in ("humerus_r", "ulna_r", "hand_r", "torso"):
    b = m.getBodySet().get(body_name)
    print(f"{body_name}:")
    print(f"  mass: {b.getMass():.4f} kg")
    print(f"  COM position (body frame): {b.getMassCenter()}")
    inertia = b.getInertia()
    moments = inertia.getMoments()
    products = inertia.getProducts()
    print(f"  inertia moments (kg·m²): {moments}")
    print(f"  inertia products: {products}")
```

Then cross-check against:
- de Leva 1996 published values for `ρ_COM/segment_length` and `r_g/segment_length`
- V3D's MDH-derived inertia (open `theia_model.mdh` in `c3d_io/mdh_parser.py` test)
- Published cadaveric humerus inertia (~0.011-0.015 kg·m² about COM lateral axis for an adult male)

If our humerus_r I_lateral is dramatically off from ~0.013 kg·m², or COM is dramatically not at ~0.45·segment_length from proximal end, that's the bug.

### Step 2: Validate humerus_r body-frame ω against V3D (~15 min)

Edit `src/theia_osim/drivers/run_trial.py` line ~205 to add upper-extremity bodies to BodyKinematics:

```python
bk = run_body_kinematics(
    markered_osim, mot_path, out_root / "recipe_d" / "body_kin",
    bodies=("pelvis", "torso", "humerus_r", "ulna_r", "hand_r"),
)
```

Then plot `humerus_r` body-frame ω against V3D's `YABIN::SHOULDER_ANGULAR_VELOCITY` (also a body-frame quantity per V3D's convention). If ω matches within 5%, kinematic input is fine and we lock onto Step 1. If ω is inflated 2-3×, the issue is in the segment-4×4 → ω derivation, and Recipe D needs investigation.

### Step 3: JR-without-actuator-feedback spike test (~5 min)

```python
from theia_osim.analysis.jr import run_joint_reaction
run_joint_reaction(
    model, mot, id_sto=id_sto, out_dir=out_dir,
    actuator_force_set_xml=None,  # disable the residual-actuator feedback loop
)
```

Inspect the resulting JR `.sto`. Reactions will be physically incomplete (active joint torques missing) but if the magnitudes drop toward V3D-typical values, the actuator-feedback cancellation pipeline is the amplifier and we redesign the JR setup.

### Step 4 (if Steps 1-3 don't isolate it): segment-frame audit

Audit OpenSim humerus_r and ulna_r body-frame conventions vs V3D's RAR and RFA segment frames. Dump R_world→OpenSim_humerus and R_world→V3D_RAR at a known-quiet frame (e.g. t=0, standing pose) and compare. Should be identity (or one fixed rotation) if aligned. Any time-varying difference is a bug.

---

## Deliverables expected from the reviewer

1. Identification of the dominant cause of the 6-13× kinetic inflation (one of H1-H5, or something we haven't considered).
2. Specific test that falsifies or confirms the hypothesis on this trial.
3. Code-level recommendation for the fix (which file, which function, what change).
4. Estimate of how much of the 6-13× the fix is expected to absorb.

---

## Plots

- `out/diagnostics_kinetics_v3d_vs_ours.png` — V3D ground-truth vs our pipeline kinetics, 6 panels
- `out/diagnostics_kinetics_comparison.png` — baseline vs Exp A vs Exp C (all 3 produce equivalent inflation)
- `out/diagnostics_shoulder_er_signal.png` — gimbal-lock 52° spike at BR in default ZXY
- `out/diagnostics_shoulder_yxz_match_v3d.png` — kinematic representation match (separate concern from kinetics)
- `out/diagnostics_shoulder_angle_vs_v3d.png` — earlier kinematic comparison
- `out/diagnostics_shoulder_yxz_anchored.png` — earlier anchored-unwrap attempt (didn't help)
- `out/diagnostics_shoulder_yxz_vs_v3d.png` — earlier YXZ comparison
