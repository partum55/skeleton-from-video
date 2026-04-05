"""
Skeleton normalization: translation, scaling, SVD-based Procrustes rotation.

Pipeline:  translate to hip center -> scale by torso length -> rotate via SVD.
"""

import numpy as np
from numpy.typing import NDArray


_LEFT_SHOULDER: int = 11
_RIGHT_SHOULDER: int = 12
_LEFT_HIP: int = 23
_RIGHT_HIP: int = 24


def _build_reference_pose() -> NDArray[np.float64]:
    """Neutral standing pose in R^{33×2}, centred on hip midpoint, torso = 1."""
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


REFERENCE_POSE: NDArray[np.float64] = _build_reference_pose()


def compute_hip_center(skeleton: NDArray[np.float64]) -> NDArray[np.float64]:
    """Midpoint of the two hip landmarks: c = (p_23 + p_24) / 2."""
    return (skeleton[_LEFT_HIP, :2] + skeleton[_RIGHT_HIP, :2]) / 2.0


def compute_shoulder_center(skeleton: NDArray[np.float64]) -> NDArray[np.float64]:
    """Midpoint of the two shoulder landmarks."""
    return (skeleton[_LEFT_SHOULDER, :2] + skeleton[_RIGHT_SHOULDER, :2]) / 2.0


def translate(
    skeleton: NDArray[np.float64],
    center: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Translate the skeleton so that the given centre lands on the origin."""
    return skeleton - center


def compute_torso_length(skeleton: NDArray[np.float64]) -> float:
    """Euclidean distance between shoulder midpoint and hip midpoint.
    Returns 1.0 if degenerate to avoid division by zero.
    """
    d = float(np.linalg.norm(
        compute_shoulder_center(skeleton) - compute_hip_center(skeleton)
    ))
    return d if d >= 1e-6 else 1.0


def scale(
    skeleton: NDArray[np.float64],
    d_ref: float,
) -> NDArray[np.float64]:
    """Divide every joint by d_ref so that the torso length becomes 1."""
    if d_ref < 1e-6:
        return skeleton.copy()
    return skeleton / d_ref


def procrustes_rotation(
    s_tilde: NDArray[np.float64],
    r_ref: NDArray[np.float64],
) -> NDArray[np.float64]:
    """SVD-based Procrustes: find R* that minimises ||S_tilde R - R_ref||^2_F.

    M = S_tilde^T R_ref  ->  SVD  ->  R* = V diag(1, det(VU^T)) U^T
    The sign correction guarantees det(R*) = +1 (proper rotation, no reflection).
    """
    M = s_tilde.T @ r_ref

    U, _, Vt = np.linalg.svd(M)
    V = Vt.T

    sign_correction = np.diag([1.0, float(np.linalg.det(V @ U.T))])
    R_star: NDArray[np.float64] = V @ sign_correction @ U.T

    return R_star


def rotation_matrix(angle_rad: float) -> NDArray[np.float64]:
    """Standard 2x2 rotation matrix for the given angle (radians)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate(
    skeleton: NDArray[np.float64],
    R_or_angle: "NDArray[np.float64] | float",
) -> NDArray[np.float64]:
    """Rotate all joints: accepts a 2x2 matrix or an angle in radians."""
    if isinstance(R_or_angle, (int, float, np.floating)):
        R = rotation_matrix(float(R_or_angle))
    else:
        R = np.asarray(R_or_angle, dtype=np.float64)
    return skeleton @ R.T


def normalize_skeleton(
    skeleton: NDArray[np.float64],
    reference_pose: NDArray[np.float64] | None = None,
    apply_procrustes: bool = True,
) -> NDArray[np.float64]:
    """Full normalization: translate -> scale -> (optional) Procrustes rotation."""
    skeleton = np.asarray(skeleton, dtype=np.float64)
    xy: NDArray[np.float64] = skeleton[:, :2].copy()

    center = compute_hip_center(xy)
    xy = translate(xy, center)

    d_ref = compute_torso_length(skeleton)
    xy = scale(xy, d_ref)

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
    """Normalise a batch of T skeletons, shape (T, 33, 2|3) -> (T, 33, 2)."""
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


def ema_smooth_skeleton(
    skeleton: NDArray[np.float64],
    prev_skeleton: NDArray[np.float64] | None = None,
    alpha: float = 0.4,
) -> NDArray[np.float64]:
    """Apply EMA smoothing to a normalized skeleton (33×2)."""
    sk = np.asarray(skeleton, dtype=np.float64)
    if sk.ndim != 2 or sk.shape != (33, 2):
        raise ValueError(f"skeleton must have shape (33, 2), got {sk.shape}")

    a = float(np.clip(alpha, 0.01, 1.0))
    if prev_skeleton is None:
        return sk.copy()

    prev = np.asarray(prev_skeleton, dtype=np.float64)
    if prev.shape != sk.shape:
        return sk.copy()
    return a * sk + (1.0 - a) * prev


class NormalizedSkeletonTemporalFilter:
    """Stateful EMA filter for normalized skeleton sequences."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self._prev: NDArray[np.float64] | None = None

    def reset(self) -> None:
        self._prev = None

    def update(self, skeleton: NDArray[np.float64]) -> NDArray[np.float64]:
        smoothed = ema_smooth_skeleton(skeleton, prev_skeleton=self._prev, alpha=self.alpha)
        self._prev = smoothed
        return smoothed
