"""
feature extraction from skeleton sequences: joint angles, velocity, distance metrics.
"""

import numpy as np


def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """compute the angle at joint b formed by vectors ba and bc.

    uses the inner product cosine formula:
    theta = arccos( (u . v) / (||u|| * ||v||) )
    where u = a - b, v = c - b.
    """
    u = a - b
    v = c - b
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-8 or norm_v < 1e-8:
        return 0.0
    cos_theta = np.dot(u, v) / (norm_u * norm_v)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def compute_key_angles(skeleton: np.ndarray) -> dict[str, float]:
    """compute key joint angles for exercise detection.

    uses landmark indices from mediapipe:
    - left_elbow: shoulder(11) -> elbow(13) -> wrist(15)
    - right_elbow: shoulder(12) -> elbow(14) -> wrist(16)
    - left_knee: hip(23) -> knee(25) -> ankle(27)
    - right_knee: hip(24) -> knee(26) -> ankle(28)
    - left_shoulder: elbow(13) -> shoulder(11) -> hip(23)
    - right_shoulder: elbow(14) -> shoulder(12) -> hip(24)
    - left_hip: shoulder(11) -> hip(23) -> knee(25)
    - right_hip: shoulder(12) -> hip(24) -> knee(26)
    """
    xy = skeleton[:, :2] if skeleton.shape[1] > 2 else skeleton

    angles = {
        "left_elbow": compute_angle(xy[11], xy[13], xy[15]),
        "right_elbow": compute_angle(xy[12], xy[14], xy[16]),
        "left_knee": compute_angle(xy[23], xy[25], xy[27]),
        "right_knee": compute_angle(xy[24], xy[26], xy[28]),
        "left_shoulder": compute_angle(xy[13], xy[11], xy[23]),
        "right_shoulder": compute_angle(xy[14], xy[12], xy[24]),
        "left_hip": compute_angle(xy[11], xy[23], xy[25]),
        "right_hip": compute_angle(xy[12], xy[24], xy[26]),
    }
    return angles


def compute_velocity(skeleton_prev: np.ndarray, skeleton_curr: np.ndarray) -> np.ndarray:
    """compute joint velocity as finite difference: v_i(t) = p_i(t) - p_i(t-1).

    this is a linear operation on the pose sequence.
    """
    return skeleton_curr - skeleton_prev


def flatten_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """flatten S(t) in R^{33x2} into a pose vector s(t) in R^{66}."""
    xy = skeleton[:, :2] if skeleton.shape[1] > 2 else skeleton
    return xy.flatten()


def euclidean_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """compute euclidean distance between two pose vectors.

    d(s1, s2) = ||s1 - s2||_2 = sqrt(sum((s1_k - s2_k)^2))
    """
    return float(np.linalg.norm(s1 - s2))


def cosine_similarity(s1: np.ndarray, s2: np.ndarray) -> float:
    """compute cosine similarity between two pose vectors.

    sim(s1, s2) = <s1, s2> / (||s1|| * ||s2||)
    """
    norm1 = np.linalg.norm(s1)
    norm2 = np.linalg.norm(s2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(s1, s2) / (norm1 * norm2))
