"""Audit F: compare segment mass/COM/inertia between our personalized
model (de Leva 1996 fractions via personalize.py) and what V3D's MDH
file would produce (truncated cone geometric model with per-subject
radii, default body density).

If these diverge meaningfully on humerus/ulna/hand, that explains the
~2× residual F gap from audit E:  F = m·(a-g) is linear in mass,
and a_COM = a_origin + α×r + ω×(ω×r) is linear in COM offset r.

Inputs:
  out/audit_e/laiuhlrich_welded/all_recipes/theia_pitching_personalized.osim
  theia_model.mdh

Output:
  out/audit_f_anthropometrics.txt   side-by-side comparison
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import opensim as osim

REPO = Path("/home/yabin/code/fwd_dynamics")

PERSONAL = (REPO / "out/audit_e/laiuhlrich_welded/all_recipes"
            / "theia_pitching_personalized.osim")

# Subject metrics from MDH (lines 17, 25, 53):
SUBJECT_MASS_KG    = 89.8128
SUBJECT_HEIGHT_M   = 1.8796
DEFAULT_THEIA_HT_M = 1.709

# V3D radii spec from MDH (lines 528-588). Constants in meters per metre of
# Default Theia3D height; multiplied by Height/DefaultTheia3DHeight.
_radius_scale = SUBJECT_HEIGHT_M / DEFAULT_THEIA_HT_M
RADII = {
    "RAR_distal":    0.030 * _radius_scale,   # upper arm distal  (= elbow side)
    "RAR_proximal":  0.030 * _radius_scale,   # upper arm proximal (= shoulder side)
    "RFA_distal":    0.025 * _radius_scale,   # forearm distal (= wrist)
    "RFA_proximal":  0.030 * _radius_scale,   # forearm proximal (= elbow)
    "RHA_distal":    0.025 * _radius_scale,
    "RHA_proximal":  0.025 * _radius_scale,
}

# Segment lengths. RHA from MDH directly. RAR/RFA: V3D infers them from
# the c3d 4×4 segment proximal/distal joint markers; we use Theia INERTIA_*
# values from the model's segment scaling — read from the personalized
# model itself (humerus/ulna/hand long-axis dimension).
LENGTH_RHA = 0.082341  # MDH line 66

V3D_BODY_DENSITY = 1056.0  # kg/m³, standard biomech value for limb segments


def truncated_cone_mass_com_iyy(R_p: float, R_d: float, h: float,
                                rho: float) -> tuple[float, float, dict]:
    """Mass, axial COM offset from proximal end, and principal inertia
    moments for a uniform-density truncated cone.

    Returns (m, z_com_from_prox, {Iaxial, I_lateral_about_COM}).
    Closed-form from biomech texts (Yeadon, Hanavan).
    """
    # Volume of frustum
    V = (np.pi * h / 3.0) * (R_p**2 + R_p*R_d + R_d**2)
    m = rho * V
    # Axial COM offset from proximal (along long axis):
    z_com = (h/4.0) * (R_p**2 + 2*R_p*R_d + 3*R_d**2) / (R_p**2 + R_p*R_d + R_d**2)
    # Inertia about COM, axial direction (along long axis):
    # I_zz = (3m/10) · (R_p^5 - R_d^5) / (R_p^3 - R_d^3)   for cone-frustum,
    # but the standard biomech approximation is a uniform truncated cone with
    # I_axial = (3/10) m (R_p^2 + R_p R_d + R_d^2)·(...)  — we just give the
    # rough magnitude useful for comparison.
    # Use solid cylinder approximation for I_axial about COM (acceptable for
    # comparison purposes — V3D itself uses Yeadon-style formulas).
    R_avg = 0.5 * (R_p + R_d)
    I_axial = 0.5 * m * R_avg**2
    # Lateral (perpendicular to long axis) about COM:
    I_lat = (m/12.0) * (3.0 * R_avg**2 + h**2)
    return m, z_com, {"I_axial": I_axial, "I_lateral": I_lat}


def get_segment_length(model: osim.Model, body_name: str,
                       parent_joint: str) -> float:
    """Crude length: distance from this body's parent-joint child frame to
    the next-distal joint's child frame, both in body frame."""
    js = model.getJointSet()
    bs = model.getBodySet()
    body = bs.get(body_name)
    # Length-of-segment proxy: just use V3D's reported metric file values
    # OR pull from the model's body Inertia eigenvalues. For now we'll use
    # the c3d-reported INERTIA_* values which personalize.py uses.
    state = model.initSystem()
    return float(np.linalg.norm(_inertia_principal_extent(body)))


def _inertia_principal_extent(body: osim.Body) -> np.ndarray:
    """Crude segment length proxy from inertia tensor: for a uniform cylinder
    of mass m and length L radius R, I_axial = mR²/2, I_lateral = m(3R²+L²)/12.
    Solve for L: L² = 12·I_lat/m - 3R². Use the smallest moment as I_axial."""
    mom = body.getInertia().getMoments()
    Ixx, Iyy, Izz = mom.get(0), mom.get(1), mom.get(2)
    moments = sorted([Ixx, Iyy, Izz])
    return np.array(moments)


def main():
    print(f"\n=== Audit F: anthropometric comparison ===")
    print(f"Subject: mass={SUBJECT_MASS_KG} kg, height={SUBJECT_HEIGHT_M} m\n")

    print("V3D-style truncated cone (per MDH radii, density=1056 kg/m³):")
    print(f"  scale factor = Height/DefaultTheia3DHeight = "
          f"{SUBJECT_HEIGHT_M:.4f}/{DEFAULT_THEIA_HT_M:.3f} = {_radius_scale:.4f}")

    # We need segment lengths to plug into the truncated cone. Use the
    # personalized model's bounding extent along the long axis as proxy.
    model = osim.Model(str(PERSONAL))
    model.initSystem()
    bs = model.getBodySet()

    # OpenSim body lengths from inertia tensor (estimate L from cylinder
    # approximation given mass and inertia tensor).
    def estimate_length(body: osim.Body) -> float:
        m = body.getMass()
        mom = body.getInertia().getMoments()
        I = sorted([mom.get(0), mom.get(1), mom.get(2)])
        I_axial, I_lat1, I_lat2 = I[0], I[1], I[2]
        # I_lat = m(3R² + L²)/12, I_axial = mR²/2 → R² = 2·I_axial/m
        if m <= 0: return float("nan")
        R2 = 2.0 * I_axial / m
        L2 = 12.0 * I_lat1 / m - 3.0 * R2
        return float(np.sqrt(L2)) if L2 > 0 else float("nan")

    print(f"\n{'segment':<14}{'OUR_mass':>12}{'V3D_mass':>12}{'Δmass':>10}"
          f"{'OUR_Iax':>12}{'V3D_Iax':>12}{'OUR_Ilat':>12}{'V3D_Ilat':>12}")
    print("-" * 100)

    SEGMENTS = [
        ("humerus_r", "RAR_proximal", "RAR_distal"),
        ("ulna_r",    "RFA_proximal", "RFA_distal"),
        ("hand_r",    "RHA_proximal", "RHA_distal"),
    ]
    for body_name, R_p_key, R_d_key in SEGMENTS:
        b = bs.get(body_name)
        m_ours = b.getMass()
        com_ours = np.array([b.getMassCenter().get(i) for i in range(3)])
        mom = b.getInertia().getMoments()
        I_ours = sorted([mom.get(0), mom.get(1), mom.get(2)])  # I_axial first
        L_estimated = estimate_length(b)

        # V3D-style truncated cone with same length, V3D's radii.
        if body_name == "hand_r":
            L_v3d = LENGTH_RHA
        else:
            L_v3d = L_estimated   # rough proxy; we don't have exact RAR/RFA
                                  # length from MDH so we use the same length
                                  # for fair comparison.
        R_p = RADII[R_p_key]; R_d = RADII[R_d_key]
        m_v3d, z_com_v3d, I_v3d = truncated_cone_mass_com_iyy(
            R_p, R_d, L_v3d, V3D_BODY_DENSITY)

        d_mass = (m_ours - m_v3d) / m_v3d * 100.0 if m_v3d else float("nan")

        print(f"{body_name:<14}{m_ours:>12.4f}{m_v3d:>12.4f}{d_mass:>9.1f}%"
              f"{I_ours[0]:>12.5f}{I_v3d['I_axial']:>12.5f}"
              f"{I_ours[1]:>12.5f}{I_v3d['I_lateral']:>12.5f}")

    print()
    print("Notes:")
    print("  - 'OUR' values are from personalize.py (de Leva fractions of subject mass).")
    print("  - 'V3D' values are truncated-cone with MDH radii, density 1056 kg/m³.")
    print("  - I_axial (along long axis) and I_lateral (perpendicular) at COM.")
    print("  - Lengths used: humerus/ulna estimated from our model's inertia eigenvalues;")
    print("    hand from MDH Length_RHA = 0.082 m.")


if __name__ == "__main__":
    main()
