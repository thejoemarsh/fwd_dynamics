# theia-osim

Port of Driveline's Visual3D pitching pipeline to OpenSim. Input: a Theia3D
markerless mocap `.c3d`. Output: an OpenSim model + IK solution + body
kinematics that should match V3D's signals so we can later add things V3D
can't easily do (inverse dynamics, joint reactions, forward sims).

The reference V3D pipeline this is replicating is documented in
[`theia_pitching_signals_reference.md`](theia_pitching_signals_reference.md);
the V3D model file we mirror is [`theia_model.mdh`](theia_model.mdh).

---

## Status

**Branch:** only `main` exists. **16 commits stacked locally, none pushed**
(`origin/main` still points at `e7a30fa` "Initial commit"). No other branches
exist, no PRs, no merge conflicts to worry about — pushing is just
`git push origin main`.

**Untracked at repo root:** `ik_solution_ik_marker_errors.sto` — OpenSim's
per-marker error sidecar that ended up in CWD because a tool was invoked
without changing dirs. Safe to delete, or add to `.gitignore`. (`opensim.log`
also lands in CWD; it's already covered by `*.log` in `.gitignore`.)

**What works (M1 + M1.5):**
- End-to-end pelvis angular velocity through both Recipe A (marker IK) and
  Recipe C (OpenSense IMU IK) on the in-repo trial `pose_filt_0.c3d`.
- VLB slope correction matches V3D (pitcher moves +Y in VLB; verified
  empirically — see `bca3915`).
- V3D-style per-athlete personalization: ScaleTool with per-body factors
  derived from Theia segment lengths, then de Leva 1996 mass + inertia
  override. Total mass after override matches subject mass to 0.02 kg.
- Bidirectional 20 Hz Butterworth on IK output (matches V3D).
- V3D ground-truth comparison harness: overlay PNGs + per-axis RMSE/MAE.
- 49 unit tests, all passing.

**What's known-broken / not done yet:**
- Recipe C (IMU IK) signal magnitudes are wrong without per-segment
  calibration rotations between Theia segment frames and LaiUhlrich body
  frames. M2 work.
- Recipe A peak pelvis ω is ~1.7× V3D's because of frame-mapping mismatch
  between OpenSim's pelvis body frame and V3D's RPV frame. The brute-force
  axis-permutation search (`validation/compare.py::search_axis_permutation`)
  brings it closer but doesn't fully fix it. M2 work — same calibration
  story.
- Inverse dynamics, joint reactions, energy flow, events (PKH/FP/MER/BR/MIR),
  discrete metrics — all M2/M3, scaffolded as empty packages
  (`kinetics_postprocess/`, `events/`, `metrics/`).

---

## Pipeline (data flow)

```
Theia .c3d  ──►  c3d_io.reader   ──►  TrialData (segment 4×4s, slope-corrected)
                                          │
                                          ├──►  model_build.add_markers ──┐
                                          │       (LaiUhlrich2022.osim +  │
                                          │        markers from YAML)     │
                                          │                               │
                                          ├──►  model_build.personalize ──┤  markered + scaled .osim
                                          │       (V3D-style ScaleTool +  │
                                          │        de Leva mass/inertia)  │
                                          │                               ▼
                                          ├──►  Recipe A: synthesize markers → .trc ──► analysis.ik.run_marker_ik ──► .mot ──┐
                                          │                                                                                 │
                                          └──►  Recipe C: 4×4 → quaternions → .sto ──► analysis.ik.run_imu_ik    ──► .mot ──┤
                                                                                                                            ▼
                                                                                          analysis.body_kin.run_body_kinematics
                                                                                              (BodyKinematics, express_in_local_frame)
                                                                                                                 │
                                                                                                                 ▼
                                                                                       kinematics_postprocess.filter (20 Hz filtfilt)
                                                                                                                 │
                                                                                                                 ▼
                                                                                  validation.compare (vs V3D procdb JSON) + plots
```

Two parallel "recipes" feed IK because Theia gives us segment poses, not
markers — Recipe A reconstructs markers from the segment frames (canonical),
Recipe C feeds the segment orientations directly to OpenSense (validator).

---

## Repo layout

```
src/theia_osim/
  constants.py          THEIA_SEGMENTS, THEIA_TO_OSIM_BODY, default VLB matrix
  config.py             frozen dataclass tree + YAML loader
  c3d_io/
    reader.py           ezc3d → TrialData (4×4 per segment per frame)
    theia_meta.py       parse THEIA3D parameter group (versions, anthropometrics)
    slope.py            apply VLB slope correction (Lab → VLB +Y forward)
    mdh_parser.py       parse Visual3D .mdh model files
  model_build/
    add_markers.py      append <Marker> tags to a source .osim from YAML
    anthropometrics.py  de Leva 1996 male tables + V3D ↔ OpenSim body maps
    personalize.py      V3D-style: ScaleTool + de Leva mass/inertia override
  import_pipeline/
    landmarks.py        load + apply marker catalog (segment-local → world)
    recipe_a_trc.py     synthesize virtual markers, write OpenSim TRC
    recipe_c_sto.py     4×4 → quaternion .sto for IMUInverseKinematicsTool
  analysis/
    ik.py               run_marker_ik, run_imu_ik
    body_kin.py         BodyKinematics with express_in_local_frame=true
    scale.py            limb-only scaling (M1.5 narrow path; superseded by personalize)
  kinematics_postprocess/
    filter.py           bidirectional Butterworth (matches V3D filtfilt)
  validation/
    load_v3d_json.py    parse V3D *_procdb.json into V3DTrial dataclass
    compare.py          overlay V3D vs OpenSim pelvis ω, axis-permutation search
  drivers/
    run_trial.py        theia-osim-trial CLI

configs/
  default.yaml          filter cutoffs, slope matrix, recipe selection
  markers.yaml          virtual-marker catalog (M1 subset: 6 segments × 3 markers)

scripts/
  fetch_models.py       one-shot: download LaiUhlrich2022 from opencap-core

tests/                  49 tests covering c3d_io, slope, landmarks, recipe C
                        quaternion round-trip, filter, MDH parser, scale,
                        personalize, V3D JSON loader

data/models/recipes/    LaiUhlrich2022_full_body.osim (gitignored, fetched)

# In-repo trial fixtures (committed):
pose_filt_0.c3d         300 Hz, 661-frame RHP pitching trial, Theia v2025.2.0
pose_filt_0.json        full V3D signal export
pose_filt_0_procdb.json V3D ground-truth procdb (used by validation harness)
theia_model.mdh         V3D model file matching that subject (89.81 kg, 1.88 m)
theia_pitching_signals_reference.md   683-line map of the V3D pipeline
```

---

## Recreate

Requires Python 3.10–3.13 (opensim 4.6 wheels) and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install deps into a project venv
uv sync

# 2. Pull LaiUhlrich2022 model into data/models/recipes/ (one shot, idempotent)
uv run python scripts/fetch_models.py

# 3. Run the in-repo trial end-to-end (M1 smoke test)
uv run theia-osim-trial \
  --c3d pose_filt_0.c3d \
  --config configs/default.yaml \
  --markers configs/markers.yaml \
  --out out/m1

# 4. Same trial with V3D-style personalization + ground-truth comparison
uv run theia-osim-trial \
  --c3d pose_filt_0.c3d \
  --mdh theia_model.mdh \
  --v3d-procdb pose_filt_0_procdb.json \
  --out out/m1_personalized

# 5. Tests (49 should pass)
uv run pytest -q
```

Outputs land under `out/<run_name>/`:

```
theia_pitching.osim                    markered model
theia_pitching_personalized.osim       + ScaleTool + de Leva (when --mdh given)
recipe_a/
  markers.trc, ik_solution.mot, body_kin/*.sto
recipe_c/
  orientations.sto, ik_orientations.mot, body_kin/*.sto
pelvis_angular_velocity.png            A vs C overlay
v3d_vs_recipe_{a,c}_pelvis_omega.png   V3D ground-truth overlay (when --v3d-procdb given)
v3d_vs_recipe_{a,c}_pelvis_omega.csv   raw aligned series
summary.json                           machine-readable run metadata
```

`out/` is gitignored. The committed reference runs (`out/m1`, `out/m1_personalized`,
etc.) are local artifacts only.

---

## CLI reference

```
theia-osim-trial
  --c3d PATH                 Theia .c3d (required)
  --config PATH              configs/default.yaml
  --markers PATH             configs/markers.yaml
  --src-model PATH           data/models/recipes/LaiUhlrich2022_full_body.osim
  --out PATH                 out/m1
  --recipes a,c              comma-separated subset
  --side {auto,R,L}          throwing side override
  --v3d-procdb PATH          enables V3D comparison overlay
  --mdh PATH                 V3D .mdh; triggers personalization
  --subject-mass-kg FLOAT    override (or supply when --mdh missing)
  --subject-height-m FLOAT   override
```

If `--mdh` is supplied, mass and height come from the MDH unless overridden.
If both `--subject-mass-kg` and `--subject-height-m` are supplied without
`--mdh`, personalization runs from those values plus c3d-derived segment
lengths.

---

## Configuration

[`configs/default.yaml`](configs/default.yaml) controls filter cutoffs, the
VLB slope matrix, recipe enablement, and which segments to drop on load.

[`configs/markers.yaml`](configs/markers.yaml) is the virtual-marker catalog
shared by both `model_build/add_markers.py` (model side) and
`import_pipeline/recipe_a_trc.py` (data side). Currently the M1 subset:
6 segments (pelvis, torso, both thighs, both shanks) × 3 markers each. Full
catalog backfills in M1.5/M2 when arms come online.

---

## Validation against V3D

`validation/compare.py` aligns our BodyKinematics output to V3D's YABIN time
series (`*_procdb.json`) on a common time grid and produces:

- per-axis RMSE / MAE
- peak magnitudes (V3D vs ours)
- overlay PNG with V3D events (`PKH`, `FP`, `MER`, `BR`) marked
- raw aligned CSV for further analysis

Current numbers on `pose_filt_0.c3d`:

| signal               | V3D peak |ω| | Recipe A peak | A vs V3D RMSE |
| -------------------- | ------------ | ------------- | ------------- |
| `PELVIS_ANGULAR_VELOCITY` | 491.7 deg/s | ~846 deg/s | ~350 deg/s |

The 1.7× peak gap is the OpenSim-pelvis-frame ↔ V3D-RPV-frame mismatch — same
intrinsic motion expressed in different body frames. `search_axis_permutation`
brute-forces 48 signed permutations to confirm; the right fix is per-segment
calibration rotations (M2), not an axis swap.

---

## Tests

Run with `uv run pytest -q`. Coverage:

- `test_c3d_io.py` — sample c3d loads, frame count, ignored segments dropped, anthropometrics parse
- `test_slope.py` — identity no-op, transpose direction (regression guard for `bca3915`), pitcher moves +Y
- `test_landmarks.py` — segment-local → world transform
- `test_recipe_c_quaternion.py` — round-trip rotation matrix → quaternion → matrix
- `test_filter.py` — DC pass-through, attenuation at 5× cutoff
- `test_mdh_parser.py` — exact subject values from `theia_model.mdh`
- `test_scale.py` — ratio correctness, plausible adult lengths
- `test_personalize.py` — de Leva ↔ MDH agreement, L/R symmetry, mapping coverage
- `test_load_v3d_json.py` — info / events / time-series shape

49 passing.

---

## Roadmap (from the milestone plan implicit in the commit history)

- **M1 ✅** Pelvis ω end-to-end through both recipes, V3D overlay.
- **M1.5 ✅** Per-athlete personalization (ScaleTool + de Leva mass/inertia).
- **M2 (next)** Per-segment calibration rotations (Theia segment frame →
  OpenSim body frame), full marker catalog (arms), inverse dynamics, joint
  reactions. Closes the ~350 deg/s pelvis RMSE gap and unlocks `kinetics_postprocess/`.
- **M3** Pitching events (PKH/FP/MER/BR/MIR) and discrete metrics — see
  the FP frame offset constants already in `constants.py`.

---

## A few things to know

- **Slope matrix is applied as `vlb_4x4.T`, not `vlb_4x4`.** V3D's row-major
  basis-vector convention means reading the rows as columns flips the
  rotation. Verified empirically: with `.T`, the pitcher moves +1.44 m in
  VLB +Y (forward) over the trial; without `.T`, the same trial reads as
  the pitcher moving backward. Don't "fix" this without re-reading
  `bca3915` and `c3d_io/slope.py`.

- **Theia → OpenSim axis swap is `Rx(-90°)`** (Theia +Z up → OpenSim +Y up).
  Applied on both the marker-catalog side (model build) and the data side
  (TRC + STO writers) so the world frame matches end-to-end. After the
  swap, V3D's `*_ANGULAR_VELOCITY::Z` maps to our `omega_y`, not `omega_z`.

- **Recipe A is canonical, Recipe C is the validator.** Both run by default;
  divergence between them is a frame-mapping diagnostic, not a sign of bugs.

- **OpenSim writes `opensim.log` and `*_ik_marker_errors.sto` to CWD.**
  `*.log` is gitignored; the `*_ik_marker_errors.sto` files are not. If
  one shows up at the repo root, it's a stale byproduct — delete or add a
  glob to `.gitignore`.
