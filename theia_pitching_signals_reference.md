# Theia/Visual3D Pitching Signals Reference

A laundry list of every kinematic and kinetic signal computed by the live Theia
Visual3D pitching pipeline, with how each one is built. Read this as a map from
**raw Theia pose `.c3d` → segments → joint angles/velocities → joint forces &
moments → energy flow → events → discrete metrics → JSON exports**.

Source files (driveline `theia_biomechanics` repo):

- `apps/main_visual3d.py` — orchestrator that watches for new pose `.c3d` trials
  and dispatches `v3d_pitching(path, facility, logger)` per trial.
- `processing/baseball/pitching/visual3d/v7_script_start_trials.v3s` — top-level
  Visual3D pipeline: opens trial, applies model, calls each `v3d_scripts/*.v3s`
  in order, then exports JSONs (full export + `*_procdb.json` for the DB).
- `processing/baseball/pitching/visual3d/v3d_scripts/` (run in order):
  1. `00_get_filename_info.v3s` — parse filename → metadata text.
  2. `01_v6_model_build.v3s` — append `vlb_model.mdh`, set MASS/HEIGHT, lowpass
     filter targets at 20 Hz, QA short trials, hardcode VLB orientation.
  3. `02_CMO_v6_v1.v3s` — **the signal calc engine** (joint angles, velocities,
     forces, moments, magnitudes, segment-energy flow). 2189 lines.
  4. `03_filter_siggies.v3s` — copy ORIGINAL link-model signals into PROCESSED
     (with optional 20 Hz Butterworth filter on derived signals).
  5. `04_v6_move_signals.v3s` — string parameters listing the joint
     angles/velos/moments/forces to export.
  6. `05_determine_side.v3s` — auto-detect throwing arm (R/L), build canonical
     events: `Release`, `MAX_EXT`, `FtStrike`, `Max_Knee_Lift`, `START`/`END`,
     `START_NORM`/`END_NORM`, `RELEASE_MINUS_50`, `RELEASE_PLUS_30`, plus QA
     tags for tracking failures.
  7. `06_yabin_ops.v3s` — pivot left/right joints into `BACK_*`/`FRONT_*` and
     `*_ANGLE`/`GLOVE_*_ANGLE` framing, sign-flip per axis to put signals into
     Driveline conventions, lowpass kinetics at 20 Hz, build pelvis/torso ankle
     positions, COG, ankle velocity, and slap a final 20 Hz pass on YABIN.
  8. `07_yabin_metrics.v3s` — pitching events (`PKH`, `PAV`, `FP`, `MER`,
     `MIR`, `BR`, `MAD`, `TRO`, `PWV`, `PTV`, `PPV`, `PrePKH`, `PostBR`,
     `PreMER`/`PostMER`), then build all the discrete metrics (max/min/value-
     at-event), STRIDE_LENGTH, STRIDE_DIRECTION, STRIDE_ANGLE, ARM_SLOT,
     PEAK_TORSO_VELO_TO_PEAK_PELVIS_VELO_TIMING, limb lengths, QA tagging.

Output: per trial Visual3D writes `*_procdb.json` (the canonical DB payload)
plus a full `*.json` of all signals/events. `processing/baseball/pitching/
visual3d/visual3d_processing_trial.py::v3d_pitching` is the Python wrapper
that calls Visual3D headlessly with `v7_script_start_trials.v3s`.

---

## 1. Coordinate system & sign conventions

- **Lab frame (`LAB`)**: standard Visual3D lab. The pitch is thrown in the +X
  direction; +Y is to the throwing side; +Z is up.
- **Virtual Lab `VLB`**: a hardcoded rotation of the lab frame from
  `01_v6_model_build.v3s`:
  `Create_Rotation /SIGNAL_NAMES=vlb_4x4
   /EXPRESSION=VECTOR(0,0.9965,-0.0831, 0, -1,0,0, 0, 0.0831,0.9965,0, 0, 0,0,0,1)`
  This corrects for mound slope (~4.76°). All "global" angles
  (e.g. `RT_ANKLE_VIRTUAL_LAB_ANGLE`) are computed against `VLB` so the slope
  is removed and "vertical" really means vertical relative to the floor.
- **Resolution coordinate systems** for each Compute_Model_Based_Data call
  (the frame the joint angle/velocity/force/moment is expressed in) are noted
  per signal below.
- **Sign convention for output (post-`06_yabin_ops`)**: the YABIN folder is the
  canonical Driveline-convention version of each signal. Sign flips are applied
  per axis per side so a "+Z" pelvis velocity always means rotation toward the
  throwing side, etc. Where you see a `Multiply_Signals_By_Constant /CONSTANT=-1`
  or `Add_Constant_To_Signals /CONSTANT=90`, that is the convention shim.
- **Cardan sequences**: most joint angles default Visual3D's Cardan sequence
  (X-Y-Z = sagittal/coronal/transverse). Pelvis, torso, shoulders use
  `AXIS1=Z`/`AXIS3=X` to get a Z-X-Y (transverse-first) sequence so the primary
  axis is rotation about the long axis (which is what matters for rotation
  metrics like pelvis/torso angle).
- **Throwing arm autodetection** (`05_determine_side.v3s`): both hand
  `DistEndVel` magnitudes are transformed to VLB, lowpassed at 20 Hz,
  rectified, and whichever side has higher Y-component speed at Release is
  tagged `R` or `L` and stored as `INFO::HAND`. Glove side = the opposite.
  Throughout the pipeline, `::ARM` and `::GLOVE` are pipeline parameters that
  get substituted into segment names — e.g. for a righty `&::ARM&AR` is `RAR`
  and `&::GLOVE&AR` is `LAR`.

---

## 2. Segments (Theia pose output)

These are the link-model segments delivered by Theia (via the pose `.c3d`):

| Code | Segment | Code | Segment |
| --- | --- | --- | --- |
| `RPV` | Pelvis (root) | `LFT` | Left foot |
| `RTA` | Torso (thorax) | `RFT` | Right foot |
| `RHE` | Head | `LTO` | Left toes |
| `RTH` | Right thigh | `RTO` | Right toes |
| `LTH` | Left thigh | `RAR` | Right upper arm (humerus) |
| `RSK` | Right shank | `LAR` | Left upper arm |
| `LSK` | Left shank | `RFA` | Right forearm |
| `RHA` | Right hand | `LFA` | Left forearm |
| `LHA` | Left hand | `VLB` | Virtual lab (mound-slope corrected) |

Per-segment kinetic-kinematic signals exported in the full JSON
(see `v7_script_start_trials.v3s` line ~459 onward):
`AngVel`, `CGPos`, `DistEndPos`, `ProxEndPos`, `ProxEndForce`, `ProxEndTorque`
for each segment in `LAR/LFA/LFT/LHA/LSK/LTH/LTO/RAR/RFA/RFT/RHA/RHE/RPV/RSK/
RTA/RTH/RTO/RTX`. Plus 4×4 rotation matrices (`*_4X4`) for every segment
including `worldbody_4X4`, `pelvis_shifted_4X4`, etc.

Landmarks exported: `L_HEEL`, `R_HEEL`, `LAB_ORIGIN`, `LAB_X/Y/Z`, joint
centers `LT_ANKLE`, `LT_ANKLE_VIRTUAL_LAB`, `LT_ELBOW`, `LT_HIP`, `LT_KNEE`,
`LT_SHOULDER`, `LT_WRIST`, `PELVIS`, `RT_*` mirrors, `TORSO`,
`TORSO_PELVIS`.

Mass/height: `MASS = SUBJECT_WEIGHT_POUNDS * 0.4536` (kg);
`HEIGHT = SUBJECT_HEIGHT_INCHES * 0.0254` (m). Stored as model metrics
(`MODEL::SEGMENT::*::LENGTH` is also accessible per segment for limb-length
metrics).

---

## 3. Filtering

| Stage | Filter | Cutoff | Source |
| --- | --- | --- | --- |
| Raw markers (TARGET) before CMO | Butterworth low-pass | 20 Hz | `01_v6_model_build.v3s` |
| Force-plate analog (commented out, not used in mocap-only pipeline) | Butterworth low-pass | 40 Hz | `01_v6_model_build.v3s` |
| Derived signals (PROCESSED → FILTERED) | Butterworth low-pass | 20 Hz | `03_filter_siggies.v3s` |
| Joint kinetics (forces/moments) before YABIN | Butterworth low-pass | 20 Hz | `06_yabin_ops.v3s` (twice — `YABIN_FORCES_PRE` then final pass on `YABIN`) |
| Original link-model moments for `ELBOW_MMT_YAB`/`SHOULDER_MMT_YAB` | Butterworth low-pass | 20 Hz | `07_yabin_metrics.v3s` |

All filters: bidirectional 1-pass Butterworth (Visual3D default), 6-frame
buffer, no extrapolation.

---

## 4. Joint angles (`02_CMO_v6_v1.v3s`)

All built via `Compute_Model_Based_Data /FUNCTION=JOINT_ANGLE`.
Output convention: `[X, Y, Z] = [flexion/extension, abduction/adduction,
internal/external rotation]` (XYZ Cardan), except where `AXIS1`/`AXIS3` are
overridden to put the rotation axis first.

| Result name | Segment | Reference seg | Negate | AXIS1/3 | Notes |
| --- | --- | --- | --- | --- | --- |
| `RT_ANKLE_ANGLE` | RFT | RSK | -Z | XYZ | dorsiflexion / inversion / IR-ER |
| `LT_ANKLE_ANGLE` | LFT | LSK | -Y, -Z | XYZ | mirror sign |
| `RT_KNEE_ANGLE` | RSK | RTH | (none cmd; default) | XYZ | flexion/abd/rot |
| `LT_KNEE_ANGLE` | LSK | LTH | (none cmd; default) | XYZ | |
| `RT_HIP_ANGLE` | RTH | RPV | (none) | XYZ | |
| `LT_HIP_ANGLE` | LTH | RPV | -Z | XYZ | (note: ref is RPV both sides) |
| `PELVIS_ANGLE` | RPV | VLB | none | Z-Y-X | Z first → primary = pelvis rotation about vertical |
| `TORSO_ANGLE` | RTA | VLB | none | Z-Y-X | "Trunk" angle |
| `TORSO_PELVIS_ANGLE` | RTA | RPV | none | XYZ default | Hip-shoulder separation |
| `RT_SHOULDER_ANGLE` | RAR | RTA | -X,-Y,-Z | Z-Y-Z | Z-Y-Z (Euler "shoulder") |
| `LT_SHOULDER_ANGLE` | LAR | RTA | none | Z-Y-Z | |
| `RT_ELBOW_ANGLE` | RFA | RAR | none | XYZ | |
| `LT_ELBOW_ANGLE` | LFA | LAR | -Z | XYZ | |
| `RT_WRIST_ANGLE` | RHA | RFA | -X | XYZ | |
| `LT_WRIST_ANGLE` | LHA | LFA | -X | XYZ | |
| `RT_ANKLE_VIRTUAL_LAB_ANGLE` | RFT | VLB | none | XYZ | "global" foot orientation |
| `LT_ANKLE_VIRTUAL_LAB_ANGLE` | LFT | VLB | none | XYZ | |

These are first written to `LINK_MODEL_BASED::ORIGINAL`. They are then copied
into `PROCESSED` (or `FILTERED` if the optional 20 Hz pass is enabled) by
`03_filter_siggies.v3s`. `06_yabin_ops.v3s` then re-keys them into the
`YABIN` namespace with the BACK/FRONT/GLOVE convention (see §6 below) and
applies sign flips and a `Resolve_Discontinuity(*, 360)` to unwrap angles
that cross 180°/360° boundaries.

---

## 5. Joint angular velocities (`02_CMO_v6_v1.v3s`)

All via `Compute_Model_Based_Data /FUNCTION=JOINT_VELOCITY`. Each is resolved
in a specific segment's coordinate system (see "ResCS" column) so peak
angular velocity is reported about a meaningful axis.

| Result name | Segment | Ref seg | ResCS | Negate |
| --- | --- | --- | --- | --- |
| `RT_KNEE_ANGULAR_VELOCITY` | RSK | RTH | RTH | -Y |
| `LT_KNEE_ANGULAR_VELOCITY` | LSK | LTH | LTH | -Z |
| `RT_HIP_ANGULAR_VELOCITY` | RTH | RPV | RTH | -Y |
| `LT_HIP_ANGULAR_VELOCITY` | LTH | RPV | LTH | -Z |
| `PELVIS_ANGULAR_VELOCITY` | RPV | VLB | RPV | none |
| `TORSO_ANGULAR_VELOCITY` | RTA | VLB | RTA | none |
| `TORSO_PELVIS_ANGULAR_VELOCITY` | RTA | RPV | RTA | -X,-Y,-Z |
| `RT_SHOULDER_ANGULAR_VELOCITY` | RAR | RTA | RAR | none |
| `LT_SHOULDER_ANGULAR_VELOCITY` | LAR | RTA | LAR | none |
| `RT_ELBOW_ANGULAR_VELOCITY` | RFA | RAR | RAR | none |
| `LT_ELBOW_ANGULAR_VELOCITY` | LFA | LAR | LAR | none |
| `RT_WRIST_ANGULAR_VELOCITY` | RHA | RFA | RFA | -X |
| `LT_WRIST_ANGULAR_VELOCITY` | LHA | LFA | LFA | -X |
| `RT_SHOULDER_GLOBAL_ANGULAR_VELOCITY` | RAR | VLB | RAR | none |
| `LT_SHOULDER_GLOBAL_ANGULAR_VELOCITY` | LAR | VLB | LAR | none |
| `RT_HAND_GLOBAL_ANGULAR_VELOCITY` | RHA | VLB | RHA | none |
| `LT_HAND_GLOBAL_ANGULAR_VELOCITY` | LHA | VLB | LHA | none |

`*_GLOBAL_*` velocities are referenced to VLB (i.e. absolute / global rotation
rate of the segment) instead of being relative to a parent segment.

Magnitudes built downstream:
- `LT_WRIST_GLOBAL_LINEAR_VELOCITY_MAG` = `|LFA::DistEndVel|`
- `RT_WRIST_GLOBAL_LINEAR_VELOCITY_MAG` = `|RFA::DistEndVel|`
- `TORSO_ANGULAR_VELOCITY_MAG`, `PELVIS_ANGULAR_VELOCITY_MAG`,
  `RT_SHOULDER_GLOBAL_ANGULAR_VELOCITY_MAG`,
  `LT_SHOULDER_GLOBAL_ANGULAR_VELOCITY_MAG` — all via `Signal_Magnitude`.

---

## 6. The `YABIN` re-keying (`06_yabin_ops.v3s`)

Up to this point everything has both an `RT_*` and `LT_*` form. `06_yabin_ops`
collapses these into the **throwing-arm-relative** naming the database expects:

For lowers (`KNEE`, `HIP`, `ANKLE`, `ANKLE_VIRTUAL_LAB`):
- `BACK_<JOINT>_ANGLE`        ← `<ARM_SIDE>T_<JOINT>_ANGLE`
- `FRONT_<JOINT>_ANGLE`       ← `<GLOVE_SIDE>T_<JOINT>_ANGLE`
- `BACK_<JOINT>_ANGULAR_VELOCITY`, `FRONT_<JOINT>_ANGULAR_VELOCITY` likewise.

For mids (`PELVIS`, `TORSO`, `TORSO_PELVIS`):
- `<MID>_ANGLE`, `<MID>_ANGULAR_VELOCITY` (no L/R — these are central segments).

For uppers (`SHOULDER`, `ELBOW`, `WRIST`):
- `<UPPER>_ANGLE`             ← throwing-arm side
- `GLOVE_<UPPER>_ANGLE`       ← non-throwing side
- `<UPPER>_ANGULAR_VELOCITY`, `GLOVE_<UPPER>_ANGULAR_VELOCITY` likewise.

For kinetics (filtered into `YABIN_FORCES_PRE` first at 20 Hz):
- Elbow/wrist: `ELBOW_FORCE`, `ELBOW_MMT`, `WRIST_FORCE`, `WRIST_MMT` ←
  throwing arm; `GLOVE_*` for opposite arm.
- Shoulder is split into two coordinate frames:
  - `SHOULDER_AR_FORCE`, `SHOULDER_AR_MMT` — resolved in the upper arm
    (humerus, AR) frame → *anatomically meaningful* shoulder force/moment
    (proximal-to-distal force, internal/external rotation moment).
  - `SHOULDER_RTA_FORCE`, `SHOULDER_RTA_MMT` — resolved in the torso (RTA)
    frame → *globally framed* shoulder force/moment.
  - Same `GLOVE_*` mirrors for non-throwing arm.

Sign-fix and unwrap operations applied in `YABIN`:

- `Resolve_Discontinuity(*, 360)` on `BACK_HIP_ANGLE`, `FRONT_HIP_ANGLE`,
  `PELVIS_ANGLE`, `TORSO_ANGLE`, `SHOULDER_ANGLE` so they don't jump 360°.
- `BACK_HIP_ANGLE`, `FRONT_HIP_ANGLE`: ×-1 on X. `BACK_HIP_ANGLE`: also ×-1 on Y, Z.
- `FRONT_KNEE_ANGLE`, `BACK_KNEE_ANGLE`: ×-1 on X (so flexion is positive).
- `PELVIS_ANGLE`: ×-1 on X, Y; +90 on Z (so 0° = facing home plate).
- `TORSO_ANGLE`: ×-1 on X, Y; +90 on Z. (Lefty branch: ×-1 on ALL components,
  then ×-1 on Y again, then +90 on Z — different sign fix-up vs righty.)
- `TORSO_PELVIS_ANGLE`: ×-1 on Z, ×-1 on Y (righty); ×-1 ALL, then ×-1 Y (lefty).
- `SHOULDER_ANGLE`: ×-1 ALL components.
- `GLOVE_ELBOW_ANGLE`: +90 on Z, ×-1 on Z. `ELBOW_ANGLE`: ×-1 on Z (after
  the L/R conditionals).
- `GLOVE_SHOULDER_ANGLE`: ×-1 (lefty branch).
- Velocities mirror the sign flips of their parent angles.
- `FRONT_ANKLE_ANGLE`, `BACK_ANKLE_ANGLE`: +90 on X (so 0° = neutral).
- `GLOVE_SHOULDER_AR_MMT`, `GLOVE_SHOULDER_RTA_MMT`: ×-1 on Z.

Then **one final 20 Hz Butterworth low-pass on every signal in `YABIN`** so
all output siggies (angles, velocities, forces, moments) come out at the same
filter spec.

Other signals built in `06_yabin_ops`:

- `LAnklePos` / `RAnklePos` = `SEG_PROXIMAL_JOINT(LFT/RFT, ref=VLB, ResCS=VLB)`
  (foot position in mound-slope-corrected lab frame).
- `LAnklePos_deriv` / `RAnklePos_deriv` = first derivative (ankle linear velocity).
- `LAnklePos_deriv_deriv` / `RAnklePos_deriv_deriv` = second derivative (ankle accel).
- `ModelCOG_VL` = `MODEL_COG` in LAB → first derivative → `COG_VELO` (in YABIN);
  Y component flipped per arm.

---

## 7. Joint forces (`02_CMO_v6_v1.v3s`, "PROCESSED KINETICS" block)

All via `Compute_Model_Based_Data /FUNCTION=JOINT_FORCE`. Reaction force is
expressed at the **proximal end of `SEGMENT`** (i.e. the joint above), in the
`RESOLUTION_COORDINATE_SYSTEM` frame.

| Result name | Segment | ResCS | Negate | Meaning |
| --- | --- | --- | --- | --- |
| `LT_SHOULDER_RTA_FORCE` | LAR | RTA | -X | L-shoulder JRF in torso frame |
| `LT_SHOULDER_LAR_FORCE` | LAR | LAR | none (cmd; -X commented) | L-shoulder JRF in upper-arm frame |
| `RT_SHOULDER_RTA_FORCE` | RAR | RTA | none | R-shoulder JRF in torso frame |
| `RT_SHOULDER_RAR_FORCE` | RAR | RAR | none | R-shoulder JRF in upper-arm frame |
| `LT_ELBOW_FORCE` | LFA | LFA | -X | L-elbow JRF in forearm frame |
| `RT_ELBOW_FORCE` | RFA | RFA | none | R-elbow JRF in forearm frame |
| `LT_WRIST_FORCE` | LHA | LFA | -X | L-wrist JRF in forearm frame |
| `RT_WRIST_FORCE` | RHA | RFA | none | R-wrist JRF in forearm frame |

All also exported per-segment as `KINETIC_KINEMATIC::<seg>::ProxEndForce`
(the inverse-dynamics output by Visual3D's segment chain, in lab frame).

---

## 8. Joint moments (`02_CMO_v6_v1.v3s`, "Joint Moments" block)

All via `/FUNCTION=JOINT_MOMENT`.

| Result name | Segment | ResCS | Cardan | Negate | Meaning |
| --- | --- | --- | --- | --- | --- |
| `LT_SHOULDER_RTA_MMT` | LAR | RTA | (default) | -X, -Z | L-shoulder NJM in torso frame |
| `LT_SHOULDER_LAR_MMT` | LAR | LAR | (default) | -X, -Z | L-shoulder NJM in upper-arm frame |
| `RT_SHOULDER_RTA_MMT` | RAR | RTA | (default) | -Y | R-shoulder NJM in torso frame |
| `RT_SHOULDER_RAR_MMT` | RAR | RAR | (default) | none | R-shoulder NJM in upper-arm frame |
| `LT_ELBOW_MMT` | LFA | LFA | Cardan=TRUE | -Y | |
| `RT_ELBOW_MMT` | RFA | RFA | Cardan=TRUE | -Z | |
| `LT_WRIST_MMT` | LHA | LFA | Cardan=TRUE | -Y | |
| `RT_WRIST_MMT` | RHA | RFA | Cardan=TRUE | -Y | |

Per-segment `KINETIC_KINEMATIC::<seg>::ProxEndTorque` is the Visual3D
inverse-dynamics torque applied at the proximal end of each segment, in lab
frame — used by the energy-flow block below.

There is also a parallel set of `ElbowMoment` and `ShoulderMoment` computed
in `07_yabin_metrics.v3s` (with `USE_CARDAN_SEQUENCE=TRUE`, ResCS = humerus
for elbow / torso for shoulder) used to derive the "yab" max moment metrics
`ELBOW_MMT_YAB`, `SHOULDER_MMT_YAB`.

---

## 9. Discontinuity-check sentinels (`02_CMO_v6_v1.v3s` lines 1–49)

These compare a Theia-given joint center against a Visual3D-reconstructed
proximal-end position, to detect tracking jumps:

- `DISCONT_RSHO_MAG = |FILTERED::RSJC − RAR::ProxEndPos|`
- `DISCONT_LSHO_MAG = |FILTERED::LSJC − LAR::ProxEndPos|`
- `DISCONT_RHIP_MAG = |FILTERED::RIGHT_HIP − RTH::ProxEndPos|`
- `DISCONT_LHIP_MAG = |FILTERED::LEFT_HIP − LTH::ProxEndPos|`

(Used as a QA visualization — large values mean tracking discontinuity.)

---

## 10. Energy flow (`02_CMO_v6_v1.v3s` lines 1554+, Robertson & Winter 1980)

For each instrumented segment `S` the script computes:

- **Proximal joint-force power (JFP)** — energy flow into `S` from its
  proximal neighbor via the joint reaction force at the proximal joint:

  ```
  S_PROX_JFP = dot( S::ProxEndForce, S::ProxEndVel )      # (proximal end)
  ```

- **Proximal segment-torque power (STP)** — energy flow via the joint torque:

  ```
  S_PROX_STP = dot( S::ProxEndTorque, S::AngVel )
  ```

- **Distal JFP / STP** — energy flow into `S` from its distal neighbor (note
  the −1 because the next segment's proximal force/torque is the equal-and-
  opposite of `S`'s distal force/torque):

  ```
  S_DIST_JFP = dot( -1 * D::ProxEndForce, S::DistEndVel )   # D = distal child
  S_DIST_STP = dot( -1 * D::ProxEndTorque, S::AngVel )
  ```

- **Segment power (SP)** at each end and net:

  ```
  S_PROX_SP = S_PROX_JFP + S_PROX_STP
  S_DIST_SP = S_DIST_JFP + S_DIST_STP
  S_NET_SP  = S_PROX_SP + S_DIST_SP
  ```

Computed segments and their distal neighbors:

| Segment | Distal neighbor | Names |
| --- | --- | --- |
| LAR (left humerus) | LFA | `LAR_PROX_JFP/STP`, `LAR_DIST_JFP/STP`, `LAR_PROX_SP`, `LAR_DIST_SP`, `LAR_NET_SP` |
| RAR (right humerus) | RFA | `RAR_*` |
| LFA (left forearm) | LHA | `LFA_*` |
| RFA (right forearm) | RHA | `RFA_*` |
| RTA (thorax) | LAR + RAR (two distal ends, "L" and "R") | `RTA_PROX_JFP/STP`, `RTA_DIST_L_JFP/STP`, `RTA_DIST_R_JFP/STP`, `RTA_PROX_SP`, `RTA_DIST_L_SP`, `RTA_DIST_R_SP`, `RTA_NET_SP = RTA_PROX_SP + RTA_DIST_L_SP + RTA_DIST_R_SP` |

(Lower-extremity energetics are scaffolded but not active in the live
pipeline; the Robertson & Winter implementation in this script is upper-body-
only as of the current version.)

Author of the energy block: Kyle Wasserberger (per file comment, last
updated 2021-08-12).

---

## 11. Global-frame joint forces & torques (`02_CMO_v6_v1.v3s` lines 2103+)

A loop over `SEGMENTS = LAR+LFA+LFT+LHA+LSK+LTH+RAR+RFA+RFT+RHA+RHE+RPV+RSK+
RTA+RTH` does, for each segment:

- `<seg>_ROTMAT` = `JOINT_ROTATION` of `<seg>` w.r.t. `LAB`, resolved in `LAB`
  (the segment's 4×4 in lab frame).
- `LINK_MODEL_BASED::GLOBAL_JOINT_FORCES::<seg>` = copy of
  `KINETIC_KINEMATIC::<seg>::ProxEndForce` (so it's available as a link-model
  signal that can be exported, etc.).
- `LINK_MODEL_BASED::GLOBAL_JOINT_TORQUES::<seg>` = copy of
  `KINETIC_KINEMATIC::<seg>::ProxEndTorque` similarly.

These are the per-segment joint reaction force/torque in **lab frame**, used
by downstream visualization and any analysis that needs the global vector.

---

## 12. Pitching events (`07_yabin_metrics.v3s`, after `05_determine_side`)

| Event | How it's defined |
| --- | --- |
| `Release` (a.k.a. `BR`) | First peak of combined hand `DistEndVel` magnitude in VLB frame, after `START + 100 frames`, +3 frame offset (peak hand speed, then 3 frames later for ball release). Built in `05_determine_side.v3s`. |
| `MAX_EXT` (a.k.a. `MER`) | Global max of `<HAND>Shoulder_Angle_YXZ::Z` (Y-X-Z Euler shoulder angle, Z = ext rotation) between `RELEASE_MINUS_50` and `Release`. Built in `05_determine_side.v3s`. |
| `FtStrike` | `MAX_EXT − 35 frames`. (Coarse fallback for foot strike — superseded by the velocity-threshold `FP` in `07_yabin_metrics.v3s`.) |
| `Max_Knee_Lift` (renamed `PKH`) | Global max of glove-side shank `ProxEndPos::Z` (knee height) between `START` and `MAX_EXT`. |
| `START_OF_RECORDING` | frame 2 |
| `END` | EOF − 1 |
| `START` | `MAX_EXT − 1800 frames` (5 s at 360 fps) — used as cutoff for downstream events to avoid pre-windup junk. Falls back to `START_OF_RECORDING` if `MAX_EXT − 1800` is before the file. |
| `START_NORM` | `MAX_EXT − 75 frames`. |
| `END_NORM` | `RELEASE + 65 frames`. |
| `RELEASE_MINUS_50` | `RELEASE − 50 frames`. |
| `RELEASE_PLUS_30` | `RELEASE + 30 frames`. |
| `MER` | renamed copy of `MAX_EXT`. |
| `BR` | renamed copy of `Release`. |
| `PKH` | renamed copy of `Max_Knee_Lift`. |
| `PreMER` | `MER − 48*(360/FS_POINT)` frames (~48 frames at 360 fps). |
| `PreMER_60` | `MER − 60*(360/FS_POINT)` frames (used as start of FP search range). |
| `PostMER` | `MER + 36*(360/FS_POINT)` frames. |
| `PrePKH` | `PKH − 300*(360/FS_POINT)` frames; falls back to `START` if undefined. |
| `PostBR` | `BR + 102*(360/FS_POINT)` frames; falls back to `END`. |
| `PAV` (Peak Ankle Velocity) | Global max of glove-side ankle position derivative X-component, search range `START`–`MER`. |
| `FP` (Foot Plant) | First **descending** crossing of glove-side `AnklePos_deriv::Z` ≤ −0.2 in `[PreMER_60, MER]`, with a 2-frame moving window. Frame offset is −4 at 360 fps, −3 at 300 fps and 250 fps (so the event is placed slightly before threshold crossing). The pipeline branches on `FS_POINT == 360 / 300 / 250`. |
| `MAD` (Max Ankle Displacement) | Global max of `(GLOVE_FT::ProxEndPos − GLOVE_TH::ProxEndPos)::Y` between `PKH` and `MER`, threshold −0.8. |
| `TRO` | First ascending zero-crossing of `YABIN::TORSO_ANGLE::Z` between `PKH` and `MER` (torso rotation toward home). |
| `PWV` (Peak Wrist Velocity) | Max of throwing-arm `WRISTVELOCITY_MAG` between `PreMER` and `PostMER`. `WRISTVELOCITY` = first derivative of `<ARM>HA::ProxEndPos`. |
| `MIR` (Max Internal Rotation) | Global min of throwing-arm `<ARM>Shoulder_Angle_YXZ::Z` between `MER` and `END`. |
| `PTV` (Peak Torso Velocity) | Global max of `YABIN::TORSO_ANGULAR_VELOCITY::Z` in `[PKH, BR]`. |
| `PPV` (Peak Pelvis Velocity) | Global max of `YABIN::PELVIS_ANGULAR_VELOCITY::Z` in `[PKH, BR]`. |
| `BAD_END`, `BAD_START` | First crossing of `LHA_DistVel_vLab::X` or `RHA_DistVel_vLab::X` ≥ 50 m/s after / before Release. Used to flag tracking failure at end / start of trial. |

For each event in `{PKH, PWV, PAV, MER, MIR, BR, FP, MAD, TRO}` two metrics
are exported:

- `<EVENT>_time` = `EVENT_LABEL::ORIGINAL::<EVENT>` (seconds since start)
- `i_<EVENT>` = `EVENT_LABEL::ORIGINAL::<EVENT> * POINT_RATE + 1` (1-based frame).

QA tags applied as `BAD`:
- Trial shorter than `1.4 s * POINT_RATE` frames.
- `MODEL::SEGMENT::RTH::LENGTH > 0.55 m` (crossed-leg model build).
- `EVENT_LABEL::ORIGINAL::END < 0.1` (EOF too early).
- Hand vel exceeds 50 m/s before Release (`BAD_VEL_AT_START` → `BAD`).
- Required event missing: any of `RAR_MIN, LAR_MIN, Release, FtStrike, START, END, RELEASE_MINUS_50, MAX_EXT`.
- `METRIC::EXPORT::SHOULDER_ANGLE_BR::Y < 50°` (no real throw).
- `METRIC::EVENTS::i_BR::X < 200` (export ended up at ~30 fps, sample).
- `PrePKH` or `PostBR` event undefined.

`BAD_VEL_AT_END` is informational only (not auto-`BAD`) — happens when
pitcher walks out of frame after release.

---

## 13. Discrete metrics exported to the DB (`07_yabin_metrics.v3s` + `v7_script_start_trials.v3s`)

The `*_procdb.json` export writes these metrics under `METRIC::EXPORT`. Each
is one of: max in a window, min in a window, value at an event, or a derived
expression. Component column shows which axis (X/Y/Z) of the link-model
signal is being read.

### 13a. Max-in-window metrics

| Metric | Signal | Comp | Window | Notes |
| --- | --- | --- | --- | --- |
| `ELBOW_ANGLE_MAX` | `YABIN::ELBOW_ANGLE` | X | `PKH` → `RELEASE_PLUS_30` | flexion |
| `SHOULDER_ANGLE_MAX` | `YABIN::SHOULDER_ANGLE` | Z | `PKH` → `RELEASE_PLUS_30` | external rotation peak |
| `COG_VELO_MAX` | `YABIN::COG_VELO` | Z | `PKH` → `RELEASE_PLUS_30` | |
| `TORSO_PELVIS_ANGLE_MAX` | `YABIN::TORSO_PELVIS_ANGLE` | Z | `PKH` → `MER` | hip-shoulder separation |
| `PELVIS_ANGULAR_VELOCITY_MAX` | `YABIN::PELVIS_ANGULAR_VELOCITY` | Z | `PKH` → `RELEASE_PLUS_30` | |
| `TORSO_ANGULAR_VELOCITY_MAX` | `YABIN::TORSO_ANGULAR_VELOCITY` | Z | `PKH` → `RELEASE_PLUS_30` | |
| `ELBOW_ANGULAR_VELOCITY_MAX` | `YABIN::ELBOW_ANGULAR_VELOCITY` | X | `PKH` → `RELEASE_PLUS_30` | |
| `SHOULDER_ANGULAR_VELOCITY_MAX` | `YABIN::SHOULDER_ANGULAR_VELOCITY` | Z | `PKH` → `RELEASE_PLUS_30` | |
| `FRONT_KNEE_ANGULAR_VELOCITY_MAX` | `YABIN::FRONT_KNEE_ANGULAR_VELOCITY` | X | `FP` → `BR` | |
| `ELBOW_FORCE_MAX` | `YABIN::ELBOW_FORCE` | ALL | `RELEASE_MINUS_50` → `RELEASE_PLUS_30` | per-component |
| `ELBOW_MMT_MAX` | `YABIN::ELBOW_MMT` | ALL | `RELEASE_MINUS_50` → `RELEASE_PLUS_30` | |
| `SHOULDER_AR_FORCE_MAX` | `YABIN::SHOULDER_AR_FORCE` | ALL | `RELEASE_MINUS_50` → `RELEASE_PLUS_30` | |
| `SHOULDER_AR_MMT_MAX` | `YABIN::SHOULDER_AR_MMT` | ALL | `RELEASE_MINUS_50` → `RELEASE_PLUS_30` | |
| `SHOULDER_RTA_FORCE_MAX` | `YABIN::SHOULDER_RTA_FORCE` | ALL | `RELEASE_MINUS_50` → `RELEASE_PLUS_30` | |
| `SHOULDER_RTA_MMT_MAX` | `YABIN::SHOULDER_RTA_MMT` | ALL | `RELEASE_MINUS_50` → `RELEASE_PLUS_30` | |
| `ELBOW_MMT_YAB` | `MOMENTS::ElbowMoment` (filtered) | ALL | `PKH` → `RELEASE_PLUS_30` | parallel "yab" max-moment |
| `SHOULDER_MMT_YAB` | `MOMENTS::ShoulderMoment` (filtered) | ALL | `PKH` → `RELEASE_PLUS_30` | |

### 13b. Min-in-window metrics

| Metric | Signal | Comp | Window |
| --- | --- | --- | --- |
| `TORSO_ANGLE_MIN` | `YABIN::TORSO_ANGLE` | X | `START` → `MER` |
| (`SHOULDER_ANGLE_MIN`, computed but not in main DB export) | `YABIN::SHOULDER_ANGLE` | ALL | `START` → `MER` |
| `ELBOW_FORCE_MIN`, `ELBOW_MMT_MIN`, `SHOULDER_AR_FORCE_MIN`, `SHOULDER_AR_MMT_MIN`, `SHOULDER_RTA_FORCE_MIN`, `SHOULDER_RTA_MMT_MIN` | YABIN | ALL | `RELEASE_MINUS_50` → `RELEASE_PLUS_30` |

### 13c. Value-at-event metrics

`07_yabin_metrics.v3s` does a nested loop:
```
For each YABIN signal × each event in {PKH, PAV, FP, MER, BR}:
  EXPORT::<SIGNAL>_<EVENT> = signal value at that event
```

So every YABIN signal is sampled at five events. The DB export pulls
specific (signal, component, event) tuples, e.g.:

| Metric | Signal | Comp | Event |
| --- | --- | --- | --- |
| `ELBOW_ANGLE_FP` | `ELBOW_ANGLE` | X | FP |
| `ELBOW_ANGLE_MER` | `ELBOW_ANGLE` | X | MER |
| `SHOULDER_ANGLE_FP` (X/Y/Z) | `SHOULDER_ANGLE` | X, Y, Z | FP |
| `GLOVE_SHOULDER_ANGLE_FP` (X/Y/Z) | `GLOVE_SHOULDER_ANGLE` | X, Y, Z | FP |
| `GLOVE_SHOULDER_ANGLE_MER` | `GLOVE_SHOULDER_ANGLE` | Y | MER |
| `TORSO_PELVIS_ANGLE_FP` | `TORSO_PELVIS_ANGLE` | Z | FP |
| `PELVIS_ANGLE_FP` (X/Y/Z) | `PELVIS_ANGLE` | X, Y, Z | FP |
| `TORSO_ANGLE_FP` (X/Y/Z) | `TORSO_ANGLE` | X, Y, Z | FP |
| `TORSO_ANGLE_MER` (X/Y/Z) | `TORSO_ANGLE` | X, Y, Z | MER |
| `TORSO_ANGLE_BR` (X/Y/Z) | `TORSO_ANGLE` | X, Y, Z | BR |
| `FRONT_KNEE_ANGLE_FP` / `_BR` | `FRONT_KNEE_ANGLE` | X | FP, BR |
| `FRONT_KNEE_ANGULAR_VELOCITY_FP` / `_BR` | `FRONT_KNEE_ANGULAR_VELOCITY` | X | FP, BR |
| `GLOVE_ELBOW_ANGLE_BR/FP/MER/PKH` | `GLOVE_ELBOW_ANGLE` | X | per event |
| `GLOVE_ELBOW_ANGULAR_VELOCITY_FP/MER` | `GLOVE_ELBOW_ANGULAR_VELOCITY` | Y | FP, MER |
| `COG_VELO_PKH` | `COG_VELO` | X | PKH |

(Full list per metric is in the `Export_Data_To_Ascii_File` `SIGNAL_NAMES`
column of `v7_script_start_trials.v3s` lines 514–520.)

### 13d. Derived metrics

- **`STRIDE_LENGTH`** (inches):
  ```
  delta_x = GLOVE_AnklePos@MER::X − ARM_AnklePos@PKH::X     # meters
  STRIDE_LENGTH = delta_x * 39.3701
  ```
  (Glove-side ankle position at MER minus throwing-side ankle position at PKH,
  converted from m to in.)

- **`STRIDE_DIRECTION`** (inches):
  ```
  delta_y = ARM_AnklePos@PKH::Y − GLOVE_AnklePos@MER::Y     # meters
  STRIDE_DIRECTION = delta_y * 39.3701   # ×−1 for lefties
  ```

- **`STRIDE_ANGLE`** (degrees):
  ```
  STRIDE_ANGLE = atan( STRIDE_DIRECTION / STRIDE_LENGTH ) * 180 / π
  ```

- **`ARM_SLOT`** (degrees, computed at BR):
  ```
  diff = ARM_HA::ProxEndPos − ARM_FA::ProxEndPos          # wrist − elbow vector
  ARM_SLOT = atan( diff::X / sqrt( diff::Y² + diff::Z² ) ) * 180 / π
  ```
  Sign: positive = high arm slot (above shoulder).

- **`PEAK_TORSO_VELO_TO_PEAK_PELVIS_VELO_TIMING`** (seconds):
  ```
  = EVENT_LABEL::PTV − EVENT_LABEL::PPV
  ```

- **`FRONT_KNEE_ANGLE_FP_TO_BR`** (degrees):
  ```
  = FRONT_KNEE_ANGLE_FP − FRONT_KNEE_ANGLE_BR
  ```
  (Lead-knee flexion change from foot plant to ball release.)

### 13e. Limb lengths (LIMB_LENGTHS folder)

Loop over `SEG ∈ {AR, FA, SK, TH}`:

- `DOM_<SEG>` = `MODEL::SEGMENT::<ARM_SIDE><SEG>::LENGTH` (m)
- `NONDOM_<SEG>` = `MODEL::SEGMENT::<GLOVE_SIDE><SEG>::LENGTH` (m)

So you get `DOM_AR`, `DOM_FA`, `DOM_SK`, `DOM_TH` and the four `NONDOM_*`
mirrors.

### 13f. Subject / pitch info text data (INFO folder)

`SUBJECT_NAME`, `HAND` (R/L), `SUBJECT_WEIGHT_POUNDS`, `SUBJECT_HEIGHT_INCHES`,
`SUBJECT_DATE`, `SUBJECT_TIME`, `SUBJECT_ACTIVITY`, `SUBJECT_LOCATION`,
`PITCH_VELO`, `QA` (`GOOD`/`BAD`), `PITCH_TYPE`, `SUBJECT_TRAQ`. Most are
parsed from the trial filename in `00_get_filename_info.v3s`.

---

## 14. Time-series exports (windowed)

The full `*.json` export normalizes most signals between `PrePKH` and `PostBR`
(a pre-windup-to-follow-through window). Specifically the Visual3D
`Export_Data_To_Ascii_File` call writes:

- All segment kinetic-kinematic `AngVel/CGPos/DistEndPos/ProxEndPos/ProxEndForce/
  ProxEndTorque` arrays clipped to `[PrePKH, PostBR]`.
- All segment 4×4 rotation matrices (for animation/playback).
- All landmarks (LAB origin/X/Y/Z, joint centers, heels).
- `JOINT_ANGLES` link-model signals.
- `JOINT_VELOS` link-model signals.
- `Model::CenterOfMass` clipped to `[PrePKH, PostBR]`.
- `FS_POINT`, `HEIGHT`, `MASS` metrics.

The `_procdb.json` adds `LIMB_LENGTHS` metrics, the discrete metric flat
list described in §13, frame-number arrays (`TIME`, `FRAMES`), and event-time
metrics (`PKH_time`, `i_PKH`, ...).

---

## 15. Quick recipe map (signal → file → operation)

If you're hunting a specific number in the DB, follow this chain:

1. **Filename → metadata**: `00_get_filename_info.v3s` parses the underscore-
   separated trial filename and writes `INFO::SUBJECT_*`, `INFO::PITCH_*`.
2. **Pose `.c3d` (Theia) → Visual3D segments**: applied automatically when the
   `.c3d` is opened (the Theia pose `.c3d` ships with all segments built in).
3. **Mass/height + slope correction + 20 Hz target filter**:
   `01_v6_model_build.v3s`.
4. **Joint angles, velocities, forces, moments, energies**: `02_CMO_v6_v1.v3s`.
   These land in `LINK_MODEL_BASED::ORIGINAL` (angles, velocities) /
   `LINK_MODEL_BASED::PROCESSED` (forces, moments) / `DERIVED::PROCESSED`
   (energies).
5. **Discontinuity check magnitudes**: top of `02_CMO_v6_v1.v3s`.
6. **Magnitudes, `Signal_Magnitude`** for wrist linear vels, torso/pelvis/
   shoulder global angular vels: `02_CMO_v6_v1.v3s` "NEW STUFF" block.
7. **Filter pass + copy ORIGINAL → PROCESSED**: `03_filter_siggies.v3s`.
8. **(Bookkeeping only)** Joint name-list parameters: `04_v6_move_signals.v3s`.
9. **Throwing-arm detection, MER/Release/PKH events, normalization range,
   per-trial QA**: `05_determine_side.v3s`.
10. **YABIN re-keying (BACK/FRONT/GLOVE), sign flips, hip/pelvis/torso/shoulder
    angle unwrap, kinetics filter, COG velo, ankle pos derivatives, final
    20 Hz pass**: `06_yabin_ops.v3s`.
11. **Pitching events (PKH/PAV/FP/MER/MIR/BR/MAD/TRO/PWV/PTV/PPV), discrete
    metrics, stride/arm-slot/timing derivations, limb lengths, post-metric
    QA tagging, GOOD/BAD text data**: `07_yabin_metrics.v3s`.
12. **JSON export**: `v7_script_start_trials.v3s` writes
    `<trial>.json` (full signal export windowed to `PrePKH`–`PostBR`) and
    `<trial>_procdb.json` (DB-flavored metric + windowed signal payload).
13. **Database write**: `apps/main_visual3d.py::process_pitching` calls
    `update_metadata(path, 'pitching')` after a successful `v3d_pitching`
    run, which inserts the `_procdb.json` into the pitching DB.

---

## 16. Cheat sheet for axis/sign meanings (post-YABIN, throwing-arm-relative)

| Signal | X | Y | Z |
| --- | --- | --- | --- |
| `BACK_KNEE_ANGLE` / `FRONT_KNEE_ANGLE` | + flexion | abd/add | rotation |
| `BACK_HIP_ANGLE` / `FRONT_HIP_ANGLE` | + flexion | abd/add | int rot |
| `FRONT_ANKLE_ANGLE` / `BACK_ANKLE_ANGLE` | + dorsiflexion (offset +90°) | inv/ev | rot |
| `PELVIS_ANGLE` | tilt | obliquity | + rotation toward home (offset +90°) |
| `TORSO_ANGLE` | flexion | side bend | + rotation toward home (offset +90°) |
| `TORSO_PELVIS_ANGLE` | (separation in sag) | (separation in cor) | + hip-shoulder separation toward back |
| `SHOULDER_ANGLE` | abduction (Z-Y-Z Euler #2) | horizontal abd/add | + ext rotation |
| `ELBOW_ANGLE` | + flexion | carrying angle | + supination |
| `WRIST_ANGLE` | flexion | rad/uln dev | rot |
| `*_ANGULAR_VELOCITY` | matches angle's positive convention | | |
| `ELBOW_FORCE` | + along forearm long axis (proximal pull) | | |
| `ELBOW_MMT::X` | + flexion moment | | |
| `SHOULDER_AR_FORCE` | resolved in humerus frame | | |
| `SHOULDER_RTA_FORCE` | resolved in torso frame (i.e. anterior/lateral/superior in torso) | | |
| `SHOULDER_AR_MMT::Z` | + ext-rot moment | | |

---

## 17. Outstanding caveats

- The mound-slope rotation in `vlb_4x4` (8.31°) is **hardcoded** — if the rig
  ever changes (different mound, flat ground, indoor net), this matrix needs
  to be updated or signals will be slope-biased.
- `BAD` tags can compound silently: `BAD_LEG_LENGTH`, `BAD_END`, `BAD_VEL_AT_END`
  are informational (only `BAD_VEL_AT_END` doesn't auto-`BAD`); but
  `BAD_VEL_AT_START` and any missing event in `EVENT_LIST` does. Rerunning a
  trial without rebuilding the model will retain old QA tags.
- Lefties: the YABIN sign fixups are NOT a simple mirror of the righty branch.
  Specifically `TORSO_ANGLE` and `TORSO_PELVIS_ANGLE` use different
  multiply-by-constant sequences for L vs R (see `06_yabin_ops.v3s` lines
  608–714 vs 1651–1704). If you're chasing a sign discrepancy between a L and
  R pitcher, that's the first place to look.
- `STRIDE_DIRECTION` is sign-flipped for lefties (line 1034 `06_yabin_ops` →
  no, line 1029 of `07_yabin_metrics`). Confirms +Y always means "open" /
  "toward 1B-side for righty / 3B-side for lefty" after the flip.
- `ELBOW_ANGLE_FP` appears twice in the procdb signal list — it's actually
  exported as both X and Z components (FE and rotation) which is why two
  entries with the same name.
- `LT_HIP_ANGLE` uses `RPV` as reference (not a mirrored `LPV`). That's because
  Theia only emits one pelvis segment (`RPV`), used bilaterally.
- The `JFP/STP` energy block has typos (`/RESURT_TYPES=` instead of
  `/RESULT_TYPES=`) on the right-side blocks — they're commented-out lines
  so it doesn't break, but worth knowing if you fork the script.
