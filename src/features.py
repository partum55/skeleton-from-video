"""
feature extraction from skeleton sequences: joint angles, velocity, distance metrics.
"""

import numpy as np

# Landmark indices for reference
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28


class AngleTemporalSmoother:
    """EMA smoother for per-frame joint angles."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self._prev: dict[str, float] | None = None

    def reset(self) -> None:
        self._prev = None

    def update(self, angles: dict[str, float]) -> dict[str, float]:
        if self._prev is None:
            self._prev = dict(angles)
            return dict(angles)

        smoothed: dict[str, float] = {}
        for key, value in angles.items():
            prev_value = self._prev.get(key, value)
            smoothed_value = self.alpha * float(value) + (1.0 - self.alpha) * float(prev_value)
            smoothed[key] = smoothed_value
        self._prev = smoothed
        return smoothed


def get_primary_angle(angles: dict[str, float], exercise: str | None) -> float:
    """Return a representative angle for plotting/debug output."""
    if exercise == "squat":
        return (angles["left_knee"] + angles["right_knee"]) / 2.0
    if exercise == "pushup":
        return (angles["left_elbow"] + angles["right_elbow"]) / 2.0
    if exercise == "jumping_jack":
        return (angles["left_shoulder"] + angles["right_shoulder"]) / 2.0
    return (angles["left_knee"] + angles["right_knee"]) / 2.0


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


def compute_body_position_features(skeleton: np.ndarray) -> dict[str, float]:
    """Compute body position features for improved exercise detection.
    
    Features computed:
    - hip_center_y: vertical position of hip midpoint (normalized 0-1)
    - shoulder_center_y: vertical position of shoulder midpoint
    - torso_verticality: ratio indicating if torso is vertical vs horizontal
    - leg_spread: horizontal distance between ankles (normalized by torso length)
    
    These features help distinguish:
    - Jumping (hip_y changes between frames)
    - Push-ups (low torso_verticality = horizontal body)
    - Standing exercises (high hip_y, high torso_verticality)
    """
    xy = skeleton[:, :2] if skeleton.shape[1] > 2 else skeleton
    
    # Hip and shoulder centers
    hip_center = (xy[_LEFT_HIP] + xy[_RIGHT_HIP]) / 2.0
    shoulder_center = (xy[_LEFT_SHOULDER] + xy[_RIGHT_SHOULDER]) / 2.0
    
    # Torso vector and length
    torso_vec = shoulder_center - hip_center
    torso_length = float(np.linalg.norm(torso_vec))
    if torso_length < 1e-6:
        torso_length = 1.0
    
    # Torso verticality: 1.0 = perfectly vertical, 0.0 = perfectly horizontal
    # Based on ratio of vertical to total torso displacement
    torso_verticality = abs(torso_vec[1]) / torso_length
    
    # Leg spread normalized by torso length
    ankle_spread = abs(xy[_LEFT_ANKLE, 0] - xy[_RIGHT_ANKLE, 0])
    leg_spread = ankle_spread / torso_length
    
    # Body y-position (average of key joints)
    body_y = (hip_center[1] + shoulder_center[1]) / 2.0
    
    return {
        "hip_center_y": float(hip_center[1]),
        "shoulder_center_y": float(shoulder_center[1]),
        "torso_verticality": float(torso_verticality),
        "leg_spread": float(leg_spread),
        "body_y": float(body_y),
        "torso_length": float(torso_length),
    }


class BodyPositionTracker:
    """Track body position changes over time for jump detection."""
    
    def __init__(self, history_frames: int = 5):
        self._history_frames = max(2, history_frames)
        self._hip_y_history: list[float] = []
        self._body_y_history: list[float] = []
    
    def reset(self) -> None:
        self._hip_y_history.clear()
        self._body_y_history.clear()
    
    def update(self, features: dict[str, float]) -> dict[str, float]:
        """Update tracker and return velocity/displacement features."""
        hip_y = features.get("hip_center_y", 0.5)
        body_y = features.get("body_y", 0.5)
        
        self._hip_y_history.append(hip_y)
        self._body_y_history.append(body_y)
        
        # Keep limited history
        if len(self._hip_y_history) > self._history_frames:
            self._hip_y_history.pop(0)
        if len(self._body_y_history) > self._history_frames:
            self._body_y_history.pop(0)
        
        # Compute velocity (positive = moving down in image coords)
        hip_velocity = 0.0
        body_velocity = 0.0
        if len(self._hip_y_history) >= 2:
            hip_velocity = self._hip_y_history[-1] - self._hip_y_history[-2]
            body_velocity = self._body_y_history[-1] - self._body_y_history[-2]
        
        # Compute recent displacement (max - min over history)
        hip_displacement = 0.0
        if len(self._hip_y_history) >= 2:
            hip_displacement = max(self._hip_y_history) - min(self._hip_y_history)
        
        return {
            "hip_y_velocity": float(hip_velocity),
            "body_y_velocity": float(body_velocity),
            "hip_y_displacement": float(hip_displacement),
        }


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
