"""
Skeleton normalization via affine transformations + SVD-based Procrustes rotation.

Developer 1 module.

Input:  raw skeleton S(t) ∈ R^{33×2}  (normalized coords from MediaPipe)
Output: normalized skeleton Ŝ(t) ∈ R^{33×2}  invariant to camera distance,
        position, and viewing angle.

Pipeline
--------
1. Translate:  S̃ ← S − c,   c = hip midpoint (landmarks 23, 24)
2. Scale:      S̃ ← S̃ / ‖torso‖,   torso = ‖shoulder_mid − hip_mid‖₂
3. Rotate:     Ŝ ← S̃ R*,   R* found via Procrustes alignment to a reference pose

Procrustes alignment (Section 4 of the report)
-----------------------------------------------
Given centered/scaled pose S̃ ∈ R^{33×2} and reference pose R_ref ∈ R^{33×2}:

    min_{R: RᵀR=I} ‖S̃R − R_ref‖²_F

Closed-form solution:
    M  = S̃ᵀ R_ref          (2×2 cross-covariance matrix)
    M  = U Σ Vᵀ             (SVD)
    R* = V diag(1, det(VUᵀ)) Uᵀ   (sign correction ensures det = +1)
    Ŝ  = S̃ R*
"""

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Pose, 33 landmarks)
# ---------------------------------------------------------------------------

_LEFT_SHOULDER: int = 11
_RIGHT_SHOULDER: int = 12
_LEFT_HIP: int = 23
_RIGHT_HIP: int = 24


# ---------------------------------------------------------------------------
# Canonical reference pose
# ---------------------------------------------------------------------------

def _build_reference_pose() -> NDArray[np.float64]:
    """Build a neutral upright standing reference pose in R^{33×2}.

    Joint positions are anatomically proportional and centred on the hip
    midpoint (origin).  The torso length is 1.0 by construction so the pose
    is already in the normalised coordinate system.

    This pose is used as R_ref in the Procrustes alignment step.
    """
    pose = np.zeros((33, 2), dtype=np.float64)

    # Head / face landmarks (0–10) — clustered near the top
    face_y = -1.7
    for i in range(11):
        pose[i] = [0.0, face_y]

    # Shoulders (11, 12)
    pose[_LEFT_SHOULDER] = [-0.3, -1.0]
    pose[_RIGHT_SHOULDER] = [0.3, -1.0]

    # Elbows (13, 14)
    pose[13] = [-0.45, -0.5]
    pose[14] = [0.45, -0.5]

    # Wrists (15, 16)
    pose[15] = [-0.5, 0.0]
    pose[16] = [0.5, 0.0]

    # Finger landmarks (17–22) — near wrists
    for i, side in zip([17, 19, 21], [-0.55, -0.55, -0.52]):
        pose[i] = [side, 0.05]
    for i, side in zip([18, 20, 22], [0.55, 0.55, 0.52]):
        pose[i] = [side, 0.05]

    # Hips (23, 24) — at the origin (torso anchor point)
    pose[_LEFT_HIP] = [-0.15, 0.0]
    pose[_RIGHT_HIP] = [0.15, 0.0]

    # Knees (25, 26)
    pose[25] = [-0.15, 0.55]
    pose[26] = [0.15, 0.55]

    # Ankles (27, 28)
    pose[27] = [-0.15, 1.1]
    pose[28] = [0.15, 1.1]

    # Heels (29, 30)
    pose[29] = [-0.18, 1.15]
    pose[30] = [0.18, 1.15]

    # Foot index (31, 32)
    pose[31] = [-0.12, 1.2]
    pose[32] = [0.12, 1.2]

    return pose


# Module-level constant so it is computed once.
REFERENCE_POSE: NDArray[np.float64] = _build_reference_pose()


# ---------------------------------------------------------------------------
# Step 1 — Translation
# ---------------------------------------------------------------------------

def compute_hip_center(skeleton: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the midpoint c = (p₂₃ + p₂₄) / 2 between the two hip landmarks.

    Parameters
    ----------
    skeleton:
        Array of shape (33, 2) or (33, 3).

    Returns
    -------
    center:
        2-D coordinate of shape (2,).
    """
    return (skeleton[_LEFT_HIP, :2] + skeleton[_RIGHT_HIP, :2]) / 2.0


def compute_shoulder_center(skeleton: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the midpoint between the two shoulder landmarks.

    Parameters
    ----------
    skeleton:
        Array of shape (33, 2) or (33, 3).

    Returns
    -------
    center:
        2-D coordinate of shape (2,).
    """
    return (skeleton[_LEFT_SHOULDER, :2] + skeleton[_RIGHT_SHOULDER, :2]) / 2.0


def translate(
    skeleton: NDArray[np.float64],
    center: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Centre the skeleton: p̃ᵢ = pᵢ − c.

    Parameters
    ----------
    skeleton:
        Array of shape (33, 2).
    center:
        Translation vector of shape (2,).

    Returns
    -------
    translated:
        Array of shape (33, 2).
    """
    return skeleton - center


# ---------------------------------------------------------------------------
# Step 2 — Scaling
# ---------------------------------------------------------------------------

def compute_torso_length(skeleton: NDArray[np.float64]) -> float:
    """Return d_ref = ‖shoulder_mid − hip_mid‖₂ (Euclidean torso length).

    Parameters
    ----------
    skeleton:
        Array of shape (33, 2) or (33, 3).

    Returns
    -------
    d_ref:
        Scalar torso length.  Returns 1.0 if degenerate (< 1e-6) to avoid
        division by zero.
    """
    d = float(np.linalg.norm(
        compute_shoulder_center(skeleton) - compute_hip_center(skeleton)
    ))
    return d if d >= 1e-6 else 1.0


def scale(
    skeleton: NDArray[np.float64],
    d_ref: float,
) -> NDArray[np.float64]:
    """Scale all joints uniformly: p̂ᵢ = p̃ᵢ / d_ref.

    Parameters
    ----------
    skeleton:
        Array of shape (33, 2).
    d_ref:
        Reference distance (torso length).

    Returns
    -------
    scaled:
        Array of shape (33, 2).
    """
    if d_ref < 1e-6:
        return skeleton.copy()
    return skeleton / d_ref


# ---------------------------------------------------------------------------
# Step 3 — Procrustes rotation via SVD
# ---------------------------------------------------------------------------

def procrustes_rotation(
    s_tilde: NDArray[np.float64],
    r_ref: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Find the optimal 2×2 rotation matrix via SVD-based Procrustes alignment.

    Solves:
        min_{R: RᵀR=I, det(R)=1}  ‖S̃ R − R_ref‖²_F

    Algorithm:
        M  = S̃ᵀ R_ref                           (2×2 cross-covariance)
        U, Σ, Vᵀ = svd(M)
        R* = V diag(1, det(VUᵀ)) Uᵀ             (sign correction)

    The sign correction in the diagonal matrix ensures det(R*) = +1 so R* is
    a proper rotation (not a reflection).

    Parameters
    ----------
    s_tilde:
        Centred and scaled pose of shape (33, 2).
    r_ref:
        Reference pose of shape (33, 2).  Must already be in the same
        normalised coordinate system (centred, unit torso).

    Returns
    -------
    R_star:
        Optimal 2×2 rotation matrix.  Satisfies R*ᵀ R* = I and det(R*) = +1.
    """
    # 2×2 cross-covariance matrix: M = S̃ᵀ R_ref
    M = s_tilde.T @ r_ref                  # shape (2, 2)

    U, _, Vt = np.linalg.svd(M)            # U (2,2), Vt (2,2)
    V = Vt.T

    # Sign correction: replace the last diagonal entry so det = +1
    sign_correction = np.diag([1.0, float(np.linalg.det(V @ U.T))])
    R_star: NDArray[np.float64] = V @ sign_correction @ U.T

    return R_star


def rotation_matrix(angle_rad: float) -> NDArray[np.float64]:
    """Build a standard 2×2 rotation matrix for a given angle.

    Parameters
    ----------
    angle_rad:
        Rotation angle in radians.

    Returns
    -------
    R:
        2×2 orthogonal rotation matrix with det = +1.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate(
    skeleton: NDArray[np.float64],
    R_or_angle: "NDArray[np.float64] | float",
) -> NDArray[np.float64]:
    """Apply a rotation to all joints: Ŝ = S̃ Rᵀ.

    Accepts either a pre-built 2×2 rotation matrix or a scalar angle in
    radians (converted to a matrix via ``rotation_matrix``).

    Parameters
    ----------
    skeleton:
        Array of shape (33, 2).
    R_or_angle:
        Either a 2×2 rotation matrix or a scalar angle in radians.

    Returns
    -------
    rotated:
        Array of shape (33, 2).
    """
    if isinstance(R_or_angle, (int, float, np.floating)):
        R = rotation_matrix(float(R_or_angle))
    else:
        R = np.asarray(R_or_angle, dtype=np.float64)
    return skeleton @ R.T


# ---------------------------------------------------------------------------
# Full normalisation pipeline
# ---------------------------------------------------------------------------

def normalize_skeleton(
    skeleton: NDArray[np.float64],
    reference_pose: NDArray[np.float64] | None = None,
    apply_procrustes: bool = True,
) -> NDArray[np.float64]:
    """Normalise a raw MediaPipe skeleton to a camera-invariant representation.

    Pipeline:
        q = (1/d_ref) · R* · (p − c)

    Steps:
        1. Extract xy columns (first 2) if skeleton has 3 columns (x, y, vis).
        2. Translate: centre on hip midpoint.
        3. Scale: divide by torso length.
        4. Rotate (optional): Procrustes alignment to reference_pose.

    Parameters
    ----------
    skeleton:
        Raw skeleton of shape (33, 2) or (33, 3).
    reference_pose:
        Reference pose of shape (33, 2) used for Procrustes alignment.
        Defaults to the module-level ``REFERENCE_POSE``.
    apply_procrustes:
        If False, skip the rotation step.  Useful for debugging.

    Returns
    -------
    s_hat:
        Normalised skeleton of shape (33, 2).
    """
    skeleton = np.asarray(skeleton, dtype=np.float64)
    xy: NDArray[np.float64] = skeleton[:, :2].copy()

    # Step 1: translate
    center = compute_hip_center(xy)
    xy = translate(xy, center)

    # Step 2: scale
    d_ref = compute_torso_length(skeleton)
    xy = scale(xy, d_ref)

    # Step 3: Procrustes rotation
    if apply_procrustes:
        ref = reference_pose if reference_pose is not None else REFERENCE_POSE
        R_star = procrustes_rotation(xy, ref)
        xy = rotate(xy, R_star)

    return xy


def normalize_sequence(
    skeletons: NDArray[np.float64],
    reference_pose: NDArray[np.float64] | None = None,
    apply_procrustes: bool = True,
) -> NDArray[np.float64]:
    """Normalise a sequence of T skeletons.

    Convenience wrapper so Developer 2 can pass the full array from
    ``run_analysis`` directly without a Python loop.

    Parameters
    ----------
    skeletons:
        Array of shape (T, 33, 2) or (T, 33, 3).
    reference_pose:
        Passed through to ``normalize_skeleton``.
    apply_procrustes:
        Passed through to ``normalize_skeleton``.

    Returns
    -------
    s_hat_sequence:
        Normalised skeletons of shape (T, 33, 2).
    """
    skeletons = np.asarray(skeletons, dtype=np.float64)
    if skeletons.ndim != 3 or skeletons.shape[1] != 33:
        raise ValueError(
            f"skeletons must have shape (T, 33, 2) or (T, 33, 3), "
            f"got {skeletons.shape}"
        )
    return np.stack([
        normalize_skeleton(s, reference_pose, apply_procrustes)
        for s in skeletons
    ])
