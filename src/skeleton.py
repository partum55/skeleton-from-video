"""
pose estimation via mediapipe and skeleton graph representation.
"""

import os
import logging

# Configure C++ logging BEFORE importing mediapipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

# Suppress Python-level logging from mediapipe
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# Initialize absl logging to suppress warnings
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.use_python_logging()

import numpy as np
import mediapipe as mp


# mediapipe pose landmark indices for reference
LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
    14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky", 19: "left_index",
    20: "right_index", 21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel",
    30: "right_heel", 31: "left_foot_index", 32: "right_foot_index",
}

NUM_LANDMARKS = 33


def filter_landmarks_by_visibility(
    landmarks: np.ndarray,
    prev_landmarks: np.ndarray | None = None,
    min_visibility: float = 0.5,
) -> np.ndarray:
    """Replace low-visibility joints using the previous frame when available.

    Parameters
    ----------
    landmarks:
        Array of shape (33, 3) with columns [x, y, visibility].
    prev_landmarks:
        Previous filtered/smoothed landmarks of shape (33, 3).
    min_visibility:
        Visibility threshold below which a joint is considered unreliable.

    Returns
    -------
    filtered:
        Landmarks with unstable joints replaced in x/y coordinates.
    """
    arr = np.asarray(landmarks, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != NUM_LANDMARKS:
        raise ValueError(f"landmarks must have shape (33, 3), got {arr.shape}")
    if arr.shape[1] < 3:
        return arr.copy()

    filtered = arr.copy()
    low_vis_mask = filtered[:, 2] < min_visibility
    if not np.any(low_vis_mask):
        return filtered

    if prev_landmarks is not None:
        prev = np.asarray(prev_landmarks, dtype=np.float64)
        if prev.shape[0] == NUM_LANDMARKS and prev.shape[1] >= 2:
            filtered[low_vis_mask, :2] = prev[low_vis_mask, :2]
            if prev.shape[1] >= 3:
                filtered[low_vis_mask, 2] = np.maximum(
                    filtered[low_vis_mask, 2],
                    prev[low_vis_mask, 2],
                )
            return filtered

    # No previous frame: keep raw coordinates to avoid collapsing the skeleton.
    return filtered


def ema_smooth_landmarks(
    landmarks: np.ndarray,
    prev_smoothed: np.ndarray | None = None,
    alpha: float = 0.35,
) -> np.ndarray:
    """Apply EMA smoothing to landmark coordinates.

    The update is: s_t = alpha * x_t + (1 - alpha) * s_{t-1}.
    """
    current = np.asarray(landmarks, dtype=np.float64)
    if current.ndim != 2 or current.shape[0] != NUM_LANDMARKS:
        raise ValueError(f"landmarks must have shape (33, C), got {current.shape}")

    a = float(np.clip(alpha, 0.01, 1.0))
    if prev_smoothed is None:
        return current.copy()

    prev = np.asarray(prev_smoothed, dtype=np.float64)
    if prev.shape != current.shape:
        return current.copy()

    smoothed = current.copy()
    smoothed[:, :2] = a * current[:, :2] + (1.0 - a) * prev[:, :2]
    if smoothed.shape[1] >= 3:
        smoothed[:, 2] = np.maximum(current[:, 2], prev[:, 2])
    return smoothed


class LandmarksTemporalFilter:
    """Visibility-aware temporal filter for MediaPipe pose landmarks."""

    def __init__(self, min_visibility: float = 0.3, ema_alpha: float = 0.35):
        self.min_visibility = float(min_visibility)
        self.ema_alpha = float(ema_alpha)
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def update(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        if landmarks is None:
            return None
        filtered = filter_landmarks_by_visibility(
            landmarks,
            prev_landmarks=self._prev,
            min_visibility=self.min_visibility,
        )
        smoothed = ema_smooth_landmarks(filtered, prev_smoothed=self._prev, alpha=self.ema_alpha)
        self._prev = smoothed
        return smoothed

# bone connections defining the skeleton graph edges
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# default model path — relative to project root
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pose_landmarker_lite.task"
)


class PoseEstimator:
    """wraps mediapipe pose landmarker (tasks api) for skeleton extraction."""

    def __init__(self, model_path: str | None = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 running_mode: str = "video"):
        model_path = model_path or _DEFAULT_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"pose model not found at '{model_path}'. "
                "download it with: curl -L -o pose_landmarker_lite.task "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            )

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        mode = VisionRunningMode.VIDEO if running_mode == "video" else VisionRunningMode.IMAGE

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mode,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self._running_mode = mode
        self._timestamp_ms = 0

    def extract_landmarks(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """extract 33 keypoints from an rgb frame.

        returns S(t) in R^{33x3} with columns [x, y, visibility],
        where x, y are normalized to [0, 1]. returns None if no pose detected.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._running_mode == mp.tasks.vision.RunningMode.VIDEO:
            self._timestamp_ms += 33  # ~30fps
            result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        pose = result.pose_landmarks[0]
        landmarks = np.array([
            [lm.x, lm.y, lm.visibility]
            for lm in pose
        ])
        return landmarks

    def extract_xy(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """extract only x, y coordinates — S(t) in R^{33x2}."""
        landmarks = self.extract_landmarks(frame_rgb)
        if landmarks is None:
            return None
        return landmarks[:, :2]

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def build_adjacency_matrix() -> np.ndarray:
    """build the adjacency matrix A in {0,1}^{33x33} for the body skeleton graph.

    A_ij = 1 if joints i and j are connected by a bone, 0 otherwise.
    A is symmetric: A = A^T.
    """
    A = np.zeros((NUM_LANDMARKS, NUM_LANDMARKS), dtype=np.int32)
    for i, j in SKELETON_CONNECTIONS:
        A[i, j] = 1
        A[j, i] = 1
    return A


def build_degree_matrix(A: np.ndarray) -> np.ndarray:
    """build the degree matrix D = diag(d_1, ..., d_33) where d_i = sum_j A_ij."""
    degrees = A.sum(axis=1)
    return np.diag(degrees)


def build_graph_laplacian(A: np.ndarray) -> np.ndarray:
    """build the graph laplacian L = D - A.

    L is symmetric positive semi-definite. its smallest eigenvalue is 0.
    """
    D = build_degree_matrix(A)
    return D - A
