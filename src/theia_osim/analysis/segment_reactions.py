"""Direct segment Newton-Euler reactions (V3D-style).

Bypasses OpenSim's JointReaction analysis, which through `AnalyzeTool` +
CoordinateActuator feedback produces 5-7× inflated reactions for
high-velocity throwing arms (see docs/m2_kinetics_jr_amplifier.md).

For each segment of interest, we compute the proximal-joint reaction by
recursive Newton-Euler from the distal end:

    F_i = m_i · (a_COM_i - g)  +  F_(child)
    M_i = I_i · α_i  +  ω_i × (I_i · ω_i)
          + r_(COM_i → distal_jc) × F_(child)
          + M_(child)
          - r_(COM_i → proximal_jc) × F_i

where F_i, M_i are the force/moment the parent applies on segment i at
its proximal joint center, expressed in ground frame. F_(child) and
M_(child) are the same quantities for the next segment downstream
(force/moment the parent of the child — i.e. THIS segment — applies on
the child).

For the throwing-arm chain: hand → ulna → humerus → torso. Wrist is
welded, so ulna and hand are combined into a single rigid segment with
mass-weighted COM and parallel-axis-corrected inertia.

V3D's reporting frames:
  ELBOW          : F, M in ulna (forearm) frame
  SHOULDER_AR    : F, M in humerus frame
  SHOULDER_RTA   : F, M in torso frame
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import opensim as osim


GRAVITY_G = np.array([0.0, -9.80665, 0.0])  # OpenSim default; verify per-model


@dataclass
class SegmentSnapshot:
    """Per-frame kinematic + inertial state of a segment in GROUND frame."""

    name: str
    com_world: np.ndarray            # (3,) COM position in ground
    a_com_world: np.ndarray          # (3,) COM linear acceleration in ground
    omega_world: np.ndarray          # (3,) angular velocity in ground
    alpha_world: np.ndarray          # (3,) angular acceleration in ground
    I_world: np.ndarray              # (3,3) inertia tensor about COM in ground
    mass: float


@dataclass
class CombinedSegment:
    """Treat ulna + hand (welded) as a single rigid body. Mass-weighted COM,
    parallel-axis-corrected inertia. ω/α are identical (welded), so just take
    one. a_COM is mass-weighted of the constituents."""

    name: str
    parts: tuple[SegmentSnapshot, ...]

    def collapse(self) -> SegmentSnapshot:
        m_total = sum(p.mass for p in self.parts)
        com = sum(p.mass * p.com_world for p in self.parts) / m_total
        a_com = sum(p.mass * p.a_com_world for p in self.parts) / m_total
        # Welded → ω, α identical across parts. Average to absorb numerical noise.
        omega = np.mean([p.omega_world for p in self.parts], axis=0)
        alpha = np.mean([p.alpha_world for p in self.parts], axis=0)
        # Combined inertia about combined COM via parallel-axis theorem:
        #   I_combined = Σ ( I_part + m_part · (||r||² · I3 - r ⊗ r) )
        # where r = part.com - combined.com
        I_total = np.zeros((3, 3))
        for p in self.parts:
            r = p.com_world - com
            I_total += p.I_world + p.mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        return SegmentSnapshot(
            name=self.name,
            com_world=com,
            a_com_world=a_com,
            omega_world=omega,
            alpha_world=alpha,
            I_world=I_total,
            mass=m_total,
        )


def _vec3_to_np(v: osim.Vec3) -> np.ndarray:
    return np.array([v.get(0), v.get(1), v.get(2)])


def _inertia_to_np(inertia: osim.Inertia) -> np.ndarray:
    """SimTK Inertia → 3x3 numpy. Inertia stores [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]."""
    m = inertia.getMoments()    # Vec3: Ixx, Iyy, Izz
    p = inertia.getProducts()   # Vec3: Ixy, Ixz, Iyz
    Ixx, Iyy, Izz = m.get(0), m.get(1), m.get(2)
    Ixy, Ixz, Iyz = p.get(0), p.get(1), p.get(2)
    return np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz],
    ])


def _rotation_to_np(R: osim.Rotation) -> np.ndarray:
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            out[i, j] = R.get(i, j)
    return out


def snapshot_body(state: osim.State, body: osim.Body) -> SegmentSnapshot:
    """Pull all kinematic + inertial quantities for a body in ground frame."""
    com_body = body.getMassCenter()  # Vec3 in body frame
    com_world = _vec3_to_np(body.findStationLocationInGround(state, com_body))
    a_com_world = _vec3_to_np(body.findStationAccelerationInGround(state, com_body))
    omega_world = _vec3_to_np(body.getAngularVelocityInGround(state))
    alpha_world = _vec3_to_np(body.getAngularAccelerationInGround(state))

    # Inertia tensor: body frame → ground frame.
    I_body = _inertia_to_np(body.getInertia())
    R_body_to_ground = _rotation_to_np(
        body.getTransformInGround(state).R()
    )
    I_world = R_body_to_ground @ I_body @ R_body_to_ground.T

    return SegmentSnapshot(
        name=body.getName(),
        com_world=com_world,
        a_com_world=a_com_world,
        omega_world=omega_world,
        alpha_world=alpha_world,
        I_world=I_world,
        mass=body.getMass(),
    )


def joint_center_in_ground(state: osim.State, joint: osim.Joint) -> np.ndarray:
    """JC location = origin of the joint's child frame, in ground."""
    child_frame = joint.getChildFrame()
    return _vec3_to_np(child_frame.getPositionInGround(state))


@dataclass
class JointReactionResult:
    """Reaction the parent applies on the child segment at the joint center."""

    F_world: np.ndarray   # (3,) world frame
    M_world: np.ndarray   # (3,) world frame
    jc_world: np.ndarray  # (3,) world frame, joint center


def newton_euler_step(
    seg: SegmentSnapshot,
    proximal_jc: np.ndarray,
    distal_jc: np.ndarray | None,
    distal_reaction: JointReactionResult | None,
) -> JointReactionResult:
    """One step of the recursive Newton-Euler.

    Args:
        seg: this segment's snapshot.
        proximal_jc: position of the proximal joint center in ground.
        distal_jc: position of the distal joint center in ground (None for
            the most distal segment in our chain).
        distal_reaction: reaction at the distal joint, computed from the
            child segment's recursion (None if no child).

    Returns:
        The reaction at this segment's proximal joint (parent on this segment).
    """
    F_child = (distal_reaction.F_world if distal_reaction is not None
               else np.zeros(3))
    M_child = (distal_reaction.M_world if distal_reaction is not None
               else np.zeros(3))

    # Newton: parent provides whatever segment needs minus what distal gives.
    # By 3rd law, force from child on this segment = -F_child.
    # m·a = F_p + (-F_child) + m·g  ⇒  F_p = m·(a - g) + F_child
    F_p = seg.mass * (seg.a_com_world - GRAVITY_G) + F_child

    # Euler about COM:
    #   I·α + ω×(I·ω) = M_p + r_(COM→p)×F_p + (-M_child) + r_(COM→d)×(-F_child)
    # ⇒ M_p = I·α + ω×(I·ω) + M_child - r_(COM→d)×(-F_child) - r_(COM→p)×F_p
    #       = I·α + ω×(I·ω) + M_child + r_(COM→d)×F_child - r_(COM→p)×F_p
    Ialpha = seg.I_world @ seg.alpha_world
    omega_x_Iomega = np.cross(seg.omega_world, seg.I_world @ seg.omega_world)
    r_to_p = proximal_jc - seg.com_world
    M_p = Ialpha + omega_x_Iomega + M_child - np.cross(r_to_p, F_p)
    if distal_jc is not None and distal_reaction is not None:
        r_to_d = distal_jc - seg.com_world
        M_p += np.cross(r_to_d, F_child)

    return JointReactionResult(F_world=F_p, M_world=M_p, jc_world=proximal_jc)


def express_in_body_frame(
    F_world: np.ndarray,
    M_world: np.ndarray,
    state: osim.State,
    body: osim.Body,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (F, M) from ground frame into body frame for V3D comparison."""
    R_body_to_ground = _rotation_to_np(body.getTransformInGround(state).R())
    R_ground_to_body = R_body_to_ground.T
    return R_ground_to_body @ F_world, R_ground_to_body @ M_world


def _euler_xyz_to_R(angles_rad: np.ndarray) -> np.ndarray:
    """OpenSim BodyKinematics pos_global stores orientation as XYZ-body Euler
    angles in degrees: rotation about body X, then body Y, then body Z.
    Build R_body_to_ground from the same convention."""
    cx, cy, cz = np.cos(angles_rad)
    sx, sy, sz = np.sin(angles_rad)
    # Rx then Ry then Rz, applied as v_world = Rz·Ry·Rx · v_body
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _read_bk_sto(path: Path, body: str, kind: str) -> np.ndarray:
    """Pull the 6 columns for a body from a BodyKinematics .sto.
    Returns array of shape (T, 6): translational + rotational triplets."""
    from theia_osim.analysis.body_kin import _read_sto
    df = _read_sto(Path(path))
    cols = [f"{body}_X", f"{body}_Y", f"{body}_Z",
            f"{body}_Ox", f"{body}_Oy", f"{body}_Oz"]
    return df[["time"] + cols].to_numpy()


def compute_throwing_arm_reactions(
    model_path: Path | str,
    coords_mot: Path | str,
    side: str = "r",
    bk_dir: Path | str | None = None,
) -> dict:
    """Compute recursive Newton-Euler reactions at acromial_<side> and
    elbow_<side> across all frames in the .mot.

    Returns a dict with per-frame arrays:
      times                 : (T,)
      elbow_F_ulna_frame    : (T, 3) — V3D ELBOW_FORCE-equivalent
      elbow_M_ulna_frame    : (T, 3) — V3D ELBOW_MMT-equivalent
      shoulder_F_humerus    : (T, 3) — V3D SHOULDER_AR_FORCE
      shoulder_M_humerus    : (T, 3) — V3D SHOULDER_AR_MMT
      shoulder_F_torso      : (T, 3) — V3D SHOULDER_RTA_FORCE
      shoulder_M_torso      : (T, 3) — V3D SHOULDER_RTA_MMT
    """
    model_path = Path(model_path).resolve()
    coords_mot = Path(coords_mot).resolve()
    model = osim.Model(str(model_path))
    state = model.initSystem()

    bs = model.getBodySet()
    js = model.getJointSet()
    body_humerus = bs.get(f"humerus_{side}")
    body_ulna = bs.get(f"ulna_{side}")
    body_hand = bs.get(f"hand_{side}")
    body_torso = bs.get("torso")
    joint_acromial = js.get(f"acromial_{side}")
    joint_elbow = js.get(f"elbow_{side}")

    bodies_to_track = {
        "humerus": body_humerus, "ulna": body_ulna,
        "hand": body_hand, "torso": body_torso,
    }

    # Run BodyKinematics through AnalyzeTool — H2 verified this produces
    # clean ω despite gimbal lock. Manual per-frame model.assemble() did
    # NOT (peak ω 90× inflated by assembly tolerance jitter). Trust the
    # AnalyzeTool path.
    if bk_dir is None:
        bk_dir = coords_mot.parent.parent / "segment_reactions_bk"
    bk_dir = Path(bk_dir)
    from theia_osim.analysis.body_kin import run_body_kinematics
    bk_paths = run_body_kinematics(
        model_path, coords_mot, bk_dir,
        bodies=tuple(b.getName() for b in bodies_to_track.values()),
    )

    # Read the .sto files: each row is [time, X, Y, Z, Ox, Oy, Oz] per body.
    pos_arr = {k: _read_bk_sto(bk_paths["pos"], b.getName(), "pos")
               for k, b in bodies_to_track.items()}
    vel_arr = {k: _read_bk_sto(bk_paths["vel"], b.getName(), "vel")
               for k, b in bodies_to_track.items()}
    acc_arr = {k: _read_bk_sto(bk_paths["acc"], b.getName(), "acc")
               for k, b in bodies_to_track.items()}

    times = pos_arr["humerus"][:, 0]
    n_frames = len(times)
    dt = float(np.median(np.diff(times)))

    # Build per-body, per-frame ground-frame kinematics.
    # pos_global stores orientation as XYZ-body Euler angles (degrees).
    # vel/acc with express_in_local_frame=true store linear in body frame +
    # angular in body frame. Convert all to ground.
    R_b2g = {}
    origin_world = {}
    omega_world = {}
    alpha_world = {}
    a_origin_world = {}
    for k in bodies_to_track:
        eul_deg = pos_arr[k][:, 4:7]
        eul_rad = np.deg2rad(eul_deg)
        R = np.zeros((n_frames, 3, 3))
        for f in range(n_frames):
            R[f] = _euler_xyz_to_R(eul_rad[f])
        R_b2g[k] = R
        origin_world[k] = pos_arr[k][:, 1:4]
        omega_body = np.deg2rad(vel_arr[k][:, 4:7])    # deg/s → rad/s
        alpha_body = np.deg2rad(acc_arr[k][:, 4:7])
        a_origin_body = acc_arr[k][:, 1:4]              # m/s² body frame
        omega_world[k] = np.einsum("fij,fj->fi", R, omega_body)
        alpha_world[k] = np.einsum("fij,fj->fi", R, alpha_body)
        a_origin_world[k] = np.einsum("fij,fj->fi", R, a_origin_body)

    # Constants from the model: COM offset in body frame, joint center
    # offset in body frame.
    com_in_body = {
        k: _vec3_to_np(b.getMassCenter()) for k, b in bodies_to_track.items()
    }
    inertia_body = {
        k: _inertia_to_np(b.getInertia()) for k, b in bodies_to_track.items()
    }
    mass = {k: b.getMass() for k, b in bodies_to_track.items()}

    # Joint centers as offsets from their parent body's origin (constant).
    # acromial_<side>: child = humerus_r; the JC in humerus_r body frame
    # is the constant location of the joint's child PhysicalOffsetFrame.
    # We can read this from the joint's frames.
    def jc_offset_in_child_body(joint: osim.Joint) -> np.ndarray:
        # The child frame's location relative to its parent (the body itself
        # for a direct attachment, or via a PhysicalOffsetFrame). Easiest:
        # query at frame 0 with default state.
        cf = joint.getChildFrame()
        # PhysicalOffsetFrame has translation + orientation.
        if isinstance(cf, osim.PhysicalOffsetFrame):
            return _vec3_to_np(cf.get_translation())
        return np.zeros(3)

    elbow_jc_in_ulna   = jc_offset_in_child_body(joint_elbow)     # joint center in ulna's frame
    acromial_jc_in_humerus = jc_offset_in_child_body(joint_acromial)

    # Convert constants to ground frame per frame.
    elbow_jc_pos = origin_world["ulna"] + np.einsum(
        "fij,j->fi", R_b2g["ulna"], elbow_jc_in_ulna)
    acromial_jc_pos = origin_world["humerus"] + np.einsum(
        "fij,j->fi", R_b2g["humerus"], acromial_jc_in_humerus)

    # COM in ground.
    pos_com = {
        k: origin_world[k] + np.einsum("fij,j->fi", R_b2g[k], com_in_body[k])
        for k in bodies_to_track
    }

    # a_COM in ground via rigid-body kinematics:
    #   a_COM = a_origin + α × r + ω × (ω × r)   where r = R · com_in_body
    a_com = {}
    for k in bodies_to_track:
        r = pos_com[k] - origin_world[k]   # = R · com_in_body, per frame
        a_com[k] = (
            a_origin_world[k]
            + np.cross(alpha_world[k], r)
            + np.cross(omega_world[k], np.cross(omega_world[k], r))
        )
    alpha = alpha_world
    omega = omega_world
    R_body_to_ground = R_b2g

    # Pass 3: Newton-Euler at every frame. We have all kinematics + frame
    # rotations cached, so this is pure numpy from here on.
    inertia_body = {
        k: _inertia_to_np(b.getInertia()) for k, b in bodies_to_track.items()
    }
    mass = {k: b.getMass() for k, b in bodies_to_track.items()}

    out = {
        "times": times,
        "elbow_F_ulna_frame":  np.zeros((n_frames, 3)),
        "elbow_M_ulna_frame":  np.zeros((n_frames, 3)),
        "shoulder_F_humerus":  np.zeros((n_frames, 3)),
        "shoulder_M_humerus":  np.zeros((n_frames, 3)),
        "shoulder_F_torso":    np.zeros((n_frames, 3)),
        "shoulder_M_torso":    np.zeros((n_frames, 3)),
    }

    for f in range(n_frames):
        # Build per-segment snapshots in ground frame.
        def _snap(name):
            R = R_body_to_ground[name][f]
            I_world = R @ inertia_body[name] @ R.T
            return SegmentSnapshot(
                name=name, com_world=pos_com[name][f],
                a_com_world=a_com[name][f], omega_world=omega[name][f],
                alpha_world=alpha[name][f], I_world=I_world,
                mass=mass[name],
            )
        h = _snap("humerus"); u = _snap("ulna"); d = _snap("hand")
        forearm = CombinedSegment("forearm", parts=(u, d)).collapse()

        elbow_react = newton_euler_step(
            seg=forearm, proximal_jc=elbow_jc_pos[f],
            distal_jc=None, distal_reaction=None,
        )
        shoulder_react = newton_euler_step(
            seg=h, proximal_jc=acromial_jc_pos[f],
            distal_jc=elbow_jc_pos[f], distal_reaction=elbow_react,
        )

        # Express in V3D-style frames using cached rotations.
        R_u = R_body_to_ground["ulna"][f]
        R_h = R_body_to_ground["humerus"][f]
        R_t = R_body_to_ground["torso"][f]
        out["elbow_F_ulna_frame"][f] = R_u.T @ elbow_react.F_world
        out["elbow_M_ulna_frame"][f] = R_u.T @ elbow_react.M_world
        out["shoulder_F_humerus"][f] = R_h.T @ shoulder_react.F_world
        out["shoulder_M_humerus"][f] = R_h.T @ shoulder_react.M_world
        out["shoulder_F_torso"][f]   = R_t.T @ shoulder_react.F_world
        out["shoulder_M_torso"][f]   = R_t.T @ shoulder_react.M_world

    return out


# ============================================================================
# c3d-driven Newton-Euler (no Recipe D, no OpenSim coord chain)
# ============================================================================

# Theia c3d segment names → OpenSim body names. Right side only here; the
# bake-off operates on the throwing arm.
THEIA_TO_OSIM_R = {
    "r_uarm": "humerus_r",
    "r_larm": "ulna_r",
    "r_hand": "hand_r",
    "torso":  "torso",
}


def _omega_world_from_R_traj(R_traj: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a (T, 3, 3) rotation-matrix trajectory.
    Uses Ω_world = Ṙ · R^T (skew). Returns rad/s."""
    T = R_traj.shape[0]
    R_dot = np.zeros_like(R_traj)
    R_dot[1:-1] = (R_traj[2:] - R_traj[:-2]) / (2 * dt)
    R_dot[0]    = (R_traj[1]  - R_traj[0])   / dt
    R_dot[-1]   = (R_traj[-1] - R_traj[-2])  / dt
    Omega = np.einsum("tij,tkj->tik", R_dot, R_traj)
    return np.stack([Omega[:, 2, 1], Omega[:, 0, 2], Omega[:, 1, 0]], axis=1)


def _smooth(arr: np.ndarray, dt: float, cutoff_hz: float = 16.0,
            order: int = 4) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    fs = 1.0 / dt
    b, a = butter(order, cutoff_hz / (fs / 2.0), btype="lowpass")
    return filtfilt(b, a, arr, axis=0)


import os as _os_filt

# V3D-style two-stage filter cascade. Defaults set by audit M (2026-05-01)
# after a sweep across 8/10/12/14/16/18/20 Hz cutoff combinations:
#   - Kinematic stage at 18 Hz matches Sports Biomechanics 2026 recommendation
#     (240 Hz sampling + 18 Hz Butterworth) and stays inside the 13-18 Hz
#     range used by published baseball-pitching kinetics literature.
#   - Kinetic stage at 10 Hz applies a second low-pass to F/M outputs after
#     the Newton-Euler propagation, mirroring V3D's documented two-stage
#     methodology (kinematic filter → compute → kinetic filter). Closes the
#     residual M overshoot that single-stage filtering can't reach.
#   On pose_filt_0 (93.7 mph pitcher) this gives shoulder F = 1.16× V3D,
#   elbow F = 0.90× V3D, elbow M = 1.01× V3D. Within ±20% on all four metrics
#   — well inside the spread between published anthropometric models.
KINEMATIC_LOWPASS_HZ_DEFAULT = float(
    _os_filt.environ.get("KINEMATIC_LOWPASS_HZ", "18.0"))
KINETIC_LOWPASS_HZ_DEFAULT = float(
    _os_filt.environ.get("KINETIC_LOWPASS_HZ", "10.0"))


def compute_throwing_arm_reactions_from_c3d(
    c3d_path: Path | str,
    model_path: Path | str,
    side: str = "r",
    *,
    wrist_mode: str = "auto",       # "auto" | "welded" | "movable"
    apply_vlb: np.ndarray | None = None,
    smoothing_hz: float | None = None,         # kinematic-stage cutoff
    kinetic_smoothing_hz: float | None = None, # kinetic-stage cutoff
    br_time: float = 1.593,
    com_overrides: dict | None = None,
    inertia_overrides: dict | None = None,
) -> dict:
    """Newton-Euler joint reactions driven directly by c3d segment 4×4s.

    Pipeline (two-stage V3D-style filter cascade):
      1. Read Theia c3d (with VLB rotation matching Recipe D).
      2. For each upper-arm segment, derive ω, α, a_origin from the 4×4
         trajectory via numerical differentiation of R(t) and origin(t).
         A kinematic-stage low-pass (smoothing_hz, default 18 Hz) is
         applied to v_origin and ω BEFORE differentiating.
      3. Get mass / COM / inertia from the **personalized** model file.
         (Caller is responsible for personalization first; we just read.)
      4. Recurse Newton-Euler:
           - wrist_mode='movable': hand → forearm → humerus
           - wrist_mode='welded':  forearm_combined → humerus
           - 'auto': inspect the model's radius_hand_<side> joint type.
      5. Apply kinetic-stage low-pass (kinetic_smoothing_hz, default
         10 Hz) to F/M output time series. This second stage absorbs
         numerical noise from the differentiation chain and matches V3D's
         documented two-pass methodology.

    Args:
        c3d_path: Theia c3d.
        model_path: personalized .osim with model-specific mass/inertia.
        side: "r" or "l".
        wrist_mode: see above.
        apply_vlb: 4×4 VLB rotation (load from configs/default.yaml). If None,
                   reads from configs/default.yaml automatically.
        smoothing_hz: kinematic-stage low-pass cutoff (Hz). None = no smoothing,
                      default = `KINEMATIC_LOWPASS_HZ_DEFAULT` (18 Hz).
                      Applied to v_origin and ω before
                      differentiating to a_COM and α. None = no smoothing.
        br_time: ball-release time, used only for the throw-window peak.

    Returns:
        dict with per-frame arrays: same keys as compute_throwing_arm_reactions
        plus 'wrist_mode_used' and 'time'.
    """
    from theia_osim.c3d_io.reader import read_theia_c3d  # noqa: E402
    import yaml as _yaml

    # Resolve filter cutoffs from explicit args, falling back to env-var-driven
    # defaults (KINEMATIC_LOWPASS_HZ / KINETIC_LOWPASS_HZ).
    if smoothing_hz is None:
        smoothing_hz = KINEMATIC_LOWPASS_HZ_DEFAULT
    if kinetic_smoothing_hz is None:
        kinetic_smoothing_hz = KINETIC_LOWPASS_HZ_DEFAULT

    c3d_path = Path(c3d_path).resolve()
    model_path = Path(model_path).resolve()

    # 1. Load c3d with VLB transform.
    if apply_vlb is None:
        cfg = _yaml.safe_load(open(Path(__file__).resolve().parents[3] /
                                   "configs/default.yaml"))
        apply_vlb = np.array(cfg["slope"]["vlb_4x4"], dtype=np.float64)
    trial = read_theia_c3d(c3d_path, apply_vlb=apply_vlb)
    dt = 1.0 / trial.sample_rate_hz

    # 2. Per-segment kinematics from 4×4s.
    seg_kine: dict[str, dict] = {}
    for theia_name, osim_name in THEIA_TO_OSIM_R.items():
        if theia_name not in trial.transforms:
            raise RuntimeError(f"c3d missing segment {theia_name}")
        T_arr = trial.transforms[theia_name]    # (n_frames, 4, 4)
        origin = T_arr[:, :3, 3]                # body-origin position in world
        R = T_arr[:, :3, :3]                    # body-to-world rotation

        # ω and v_origin straight from R(t), origin(t). Smooth then diff for α
        # and a (not the values themselves — those aren't noisy enough to need it).
        omega_world = _omega_world_from_R_traj(R, dt)
        v_origin = np.gradient(origin, dt, axis=0)
        if smoothing_hz is not None:
            v_origin = _smooth(v_origin, dt, smoothing_hz)
            omega_world = _smooth(omega_world, dt, smoothing_hz)
        a_origin_world = np.gradient(v_origin, dt, axis=0)
        alpha_world = np.gradient(omega_world, dt, axis=0)

        seg_kine[osim_name] = dict(
            origin_world=origin,
            R=R,
            omega=omega_world,
            alpha=alpha_world,
            a_origin=a_origin_world,
        )

    # 3. Inertial properties from the model.
    model = osim.Model(str(model_path))
    model.initSystem()
    bs = model.getBodySet()
    js = model.getJointSet()

    def body_inertials(name: str) -> dict:
        b = bs.get(name)
        com = _vec3_to_np(b.getMassCenter())
        I_body = _inertia_to_np(b.getInertia())
        if com_overrides and name in com_overrides:
            com = np.asarray(com_overrides[name], dtype=float)
            print(f"[segNE-from-c3d] override {name} COM → {com}")
        if inertia_overrides and name in inertia_overrides:
            I_body = np.asarray(inertia_overrides[name], dtype=float)
            print(f"[segNE-from-c3d] override {name} I_body diag → "
                  f"[{I_body[0,0]:.5f}, {I_body[1,1]:.5f}, {I_body[2,2]:.5f}]")
        return dict(mass=b.getMass(), com=com, I_body=I_body)

    inert = {n: body_inertials(n) for n in (f"humerus_{side}", f"ulna_{side}",
                                            f"hand_{side}", "torso")}

    # Wrist mode autodetection.
    wrist_joint = js.get(f"radius_hand_{side}")
    wrist_type = wrist_joint.getConcreteClassName()
    if wrist_mode == "auto":
        wrist_mode = "welded" if wrist_type == "WeldJoint" else "movable"
    print(f"[segNE-from-c3d] wrist joint type: {wrist_type} → mode={wrist_mode}")

    # 4. Joint center offsets in body frame (constants, taken from the model).
    def jc_offset(joint: osim.Joint) -> np.ndarray:
        cf = joint.getChildFrame()
        if isinstance(cf, osim.PhysicalOffsetFrame):
            return _vec3_to_np(cf.get_translation())
        return np.zeros(3)

    elbow_jc_in_ulna       = jc_offset(js.get(f"elbow_{side}"))
    acromial_jc_in_humerus = jc_offset(js.get(f"acromial_{side}"))
    wrist_jc_in_hand       = jc_offset(wrist_joint)

    n_frames = trial.n_frames
    times = np.arange(n_frames) * dt

    # 5. Per-frame derived quantities.
    body_state: dict[str, dict] = {}
    for name, kine in seg_kine.items():
        i = inert[name]
        com_world = kine["origin_world"] + np.einsum("tij,j->ti", kine["R"], i["com"])
        # a_COM via rigid-body kinematics: a_COM = a_origin + α × r + ω × (ω × r)
        r_origin_to_com = com_world - kine["origin_world"]
        a_com = (
            kine["a_origin"]
            + np.cross(kine["alpha"], r_origin_to_com)
            + np.cross(kine["omega"], np.cross(kine["omega"], r_origin_to_com))
        )
        # Inertia in world per frame.
        I_world = np.einsum("tij,jk,tlk->til", kine["R"], i["I_body"], kine["R"])
        body_state[name] = dict(
            mass=i["mass"], com_world=com_world, a_com=a_com,
            omega=kine["omega"], alpha=kine["alpha"],
            R=kine["R"], origin=kine["origin_world"], I_world=I_world,
        )

    # Joint center positions in world per frame.
    elbow_jc = (body_state[f"ulna_{side}"]["origin"]
                + np.einsum("tij,j->ti", body_state[f"ulna_{side}"]["R"], elbow_jc_in_ulna))
    acromial_jc = (body_state[f"humerus_{side}"]["origin"]
                   + np.einsum("tij,j->ti", body_state[f"humerus_{side}"]["R"],
                               acromial_jc_in_humerus))
    wrist_jc = (body_state[f"hand_{side}"]["origin"]
                + np.einsum("tij,j->ti", body_state[f"hand_{side}"]["R"], wrist_jc_in_hand))

    out = {
        "times": times,
        "wrist_mode_used": wrist_mode,
        "elbow_F_ulna_frame":  np.zeros((n_frames, 3)),
        "elbow_M_ulna_frame":  np.zeros((n_frames, 3)),
        "shoulder_F_humerus":  np.zeros((n_frames, 3)),
        "shoulder_M_humerus":  np.zeros((n_frames, 3)),
        "shoulder_F_torso":    np.zeros((n_frames, 3)),
        "shoulder_M_torso":    np.zeros((n_frames, 3)),
    }

    # 6. Per-frame Newton-Euler.
    for f in range(n_frames):
        h = SegmentSnapshot(
            "humerus", body_state[f"humerus_{side}"]["com_world"][f],
            body_state[f"humerus_{side}"]["a_com"][f],
            body_state[f"humerus_{side}"]["omega"][f],
            body_state[f"humerus_{side}"]["alpha"][f],
            body_state[f"humerus_{side}"]["I_world"][f],
            body_state[f"humerus_{side}"]["mass"],
        )
        u = SegmentSnapshot(
            "ulna", body_state[f"ulna_{side}"]["com_world"][f],
            body_state[f"ulna_{side}"]["a_com"][f],
            body_state[f"ulna_{side}"]["omega"][f],
            body_state[f"ulna_{side}"]["alpha"][f],
            body_state[f"ulna_{side}"]["I_world"][f],
            body_state[f"ulna_{side}"]["mass"],
        )
        d = SegmentSnapshot(
            "hand", body_state[f"hand_{side}"]["com_world"][f],
            body_state[f"hand_{side}"]["a_com"][f],
            body_state[f"hand_{side}"]["omega"][f],
            body_state[f"hand_{side}"]["alpha"][f],
            body_state[f"hand_{side}"]["I_world"][f],
            body_state[f"hand_{side}"]["mass"],
        )

        if wrist_mode == "welded":
            forearm = CombinedSegment("forearm", parts=(u, d)).collapse()
            elbow_react = newton_euler_step(
                seg=forearm, proximal_jc=elbow_jc[f],
                distal_jc=None, distal_reaction=None,
            )
        else:
            # hand recurses first (open chain at fingertip), then forearm.
            hand_react = newton_euler_step(
                seg=d, proximal_jc=wrist_jc[f],
                distal_jc=None, distal_reaction=None,
            )
            elbow_react = newton_euler_step(
                seg=u, proximal_jc=elbow_jc[f],
                distal_jc=wrist_jc[f], distal_reaction=hand_react,
            )

        shoulder_react = newton_euler_step(
            seg=h, proximal_jc=acromial_jc[f],
            distal_jc=elbow_jc[f], distal_reaction=elbow_react,
        )

        R_u = body_state[f"ulna_{side}"]["R"][f]
        R_h = body_state[f"humerus_{side}"]["R"][f]
        R_t = body_state["torso"]["R"][f]
        out["elbow_F_ulna_frame"][f] = R_u.T @ elbow_react.F_world
        out["elbow_M_ulna_frame"][f] = R_u.T @ elbow_react.M_world
        out["shoulder_F_humerus"][f] = R_h.T @ shoulder_react.F_world
        out["shoulder_M_humerus"][f] = R_h.T @ shoulder_react.M_world
        out["shoulder_F_torso"][f]   = R_t.T @ shoulder_react.F_world
        out["shoulder_M_torso"][f]   = R_t.T @ shoulder_react.M_world

    # 7. Kinetic-stage low-pass on F/M output time series. Mirrors V3D's
    # documented two-pass methodology: filter raw kinematics → compute →
    # filter the kinetics outputs. Default 10 Hz cutoff absorbs numerical
    # noise from the differentiation chain that single-stage kinematic
    # filtering can't remove.
    if kinetic_smoothing_hz is not None:
        for k in ("elbow_F_ulna_frame", "elbow_M_ulna_frame",
                  "shoulder_F_humerus", "shoulder_M_humerus",
                  "shoulder_F_torso",   "shoulder_M_torso"):
            out[k] = _smooth(out[k], dt, kinetic_smoothing_hz)
    out["kinematic_lowpass_hz"] = float(smoothing_hz) if smoothing_hz else None
    out["kinetic_lowpass_hz"] = (
        float(kinetic_smoothing_hz) if kinetic_smoothing_hz else None)

    return out
