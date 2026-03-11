"""
skeleton normalization via affine transformations: translation, scaling, rotation.
"""

import numpy as np


def compute_hip_center(skeleton: np.ndarray) -> np.ndarray:
    """compute the midpoint between left hip (23) and right hip (24).

    c = (p_23 + p_24) / 2, the center of mass reference point.
    """
    return (skeleton[23] + skeleton[24]) / 2.0


def compute_shoulder_center(skeleton: np.ndarray) -> np.ndarray:
    """compute the midpoint between left shoulder (11) and right shoulder (12)."""
    return (skeleton[11] + skeleton[12]) / 2.0


def translate(skeleton: np.ndarray, center: np.ndarray) -> np.ndarray:
    """center the skeleton by subtracting a reference point.

    p_tilde_i = p_i - c (affine translation to origin)
    """
    return skeleton - center


def compute_reference_distance(skeleton: np.ndarray) -> float:
    """compute the torso length as the euclidean norm between shoulder and hip midpoints.

    d_ref = ||p_shoulders - p_hips||_2
    """
    shoulder_mid = compute_shoulder_center(skeleton)
    hip_mid = compute_hip_center(skeleton)
    d_ref = np.linalg.norm(shoulder_mid - hip_mid)
    return d_ref


def scale(skeleton: np.ndarray, d_ref: float) -> np.ndarray:
    """scale the skeleton so the torso length becomes 1.

    p_hat_i = p_tilde_i / d_ref (linear scaling transformation)
    """
    if d_ref < 1e-6:
        return skeleton
    return skeleton / d_ref


def rotation_matrix(alpha: float) -> np.ndarray:
    """build the 2d rotation matrix R(alpha).

    R = [[cos(a), -sin(a)],
         [sin(a),  cos(a)]]

    this is an orthogonal matrix with det(R) = 1 that preserves lengths and angles.
    """
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s],
                     [s,  c]])


def compute_torso_angle(skeleton: np.ndarray) -> float:
    """compute the angle between the torso vector and the vertical axis.

    used to determine how much to rotate the skeleton to align it upright.
    """
    shoulder_mid = compute_shoulder_center(skeleton)
    hip_mid = compute_hip_center(skeleton)
    torso_vec = shoulder_mid - hip_mid
    vertical = np.array([0.0, -1.0])  # y-axis points down in image coords

    norm_t = np.linalg.norm(torso_vec)
    if norm_t < 1e-6:
        return 0.0

    cos_angle = np.dot(torso_vec, vertical) / norm_t
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def rotate(skeleton: np.ndarray, alpha: float) -> np.ndarray:
    """rotate all joints by angle alpha using the rotation matrix.

    each row p_i is transformed: p_i' = R(alpha) @ p_i
    in matrix form: S' = S @ R^T (since rows are points)
    """
    R = rotation_matrix(alpha)
    return skeleton @ R.T


def normalize_skeleton(skeleton: np.ndarray, apply_rotation: bool = False) -> np.ndarray:
    """full normalization pipeline: q_i = (1/d_ref) * R(alpha) * (p_i - c).

    composes translation, optional rotation, and scaling — an affine transformation.
    operates on the xy columns only (first 2 columns).
    """
    if skeleton.shape[1] > 2:
        xy = skeleton[:, :2].copy()
    else:
        xy = skeleton.copy()

    # step 1: translate (center on hip midpoint)
    center = compute_hip_center(xy)
    xy = translate(xy, center)

    # step 2: optional rotation alignment
    if apply_rotation:
        angle = compute_torso_angle(xy)
        if abs(angle) > 0.05:  # only rotate if significant tilt
            xy = rotate(xy, -angle)

    # step 3: scale by torso length
    d_ref = np.linalg.norm(
        compute_shoulder_center(skeleton[:, :2]) - compute_hip_center(skeleton[:, :2])
    )
    xy = scale(xy, d_ref)

    return xy
