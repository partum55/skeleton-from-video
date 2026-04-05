"""Regression tests for runtime stability features added to live pipeline."""

import numpy as np

from src.classify import ExerciseClassifier, count_reps_from_signal
from src.normalize import NormalizedSkeletonTemporalFilter, ema_smooth_skeleton
from src.pca import align_components_to_reference
from src.repetition import fuse_exercise_labels
from src.skeleton import (
    NUM_LANDMARKS,
    LandmarksTemporalFilter,
    ema_smooth_landmarks,
    filter_landmarks_by_visibility,
)


def _make_landmarks(value_xy: float = 0.5, visibility: float = 1.0) -> np.ndarray:
    """Build a synthetic (33, 3) landmark array."""
    arr = np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)
    arr[:, 0] = value_xy
    arr[:, 1] = value_xy
    arr[:, 2] = visibility
    return arr


class TestVisibilityAndLandmarkSmoothing:
    def test_low_visibility_uses_previous_coordinates(self) -> None:
        prev = _make_landmarks(value_xy=0.8, visibility=0.95)
        cur = _make_landmarks(value_xy=0.2, visibility=0.95)
        cur[5, 2] = 0.1

        filtered = filter_landmarks_by_visibility(cur, prev_landmarks=prev, min_visibility=0.5)

        np.testing.assert_allclose(filtered[5, :2], prev[5, :2])
        np.testing.assert_allclose(filtered[6, :2], cur[6, :2])

    def test_ema_smooth_landmarks_moves_toward_previous(self) -> None:
        prev = _make_landmarks(value_xy=0.0, visibility=0.9)
        cur = _make_landmarks(value_xy=1.0, visibility=0.9)

        smoothed = ema_smooth_landmarks(cur, prev_smoothed=prev, alpha=0.25)

        np.testing.assert_allclose(smoothed[:, 0], 0.25)
        np.testing.assert_allclose(smoothed[:, 1], 0.25)

    def test_landmarks_temporal_filter_reset(self) -> None:
        filt = LandmarksTemporalFilter(min_visibility=0.5, ema_alpha=0.3)
        first = _make_landmarks(value_xy=0.1, visibility=1.0)
        second = _make_landmarks(value_xy=0.9, visibility=1.0)

        out1 = filt.update(first)
        out2 = filt.update(second)
        assert not np.allclose(out2[:, :2], second[:, :2])

        filt.reset()
        out3 = filt.update(second)
        np.testing.assert_allclose(out3[:, :2], second[:, :2])
        np.testing.assert_allclose(out1[:, :2], first[:, :2])


class TestNormalizedSkeletonSmoothing:
    def test_ema_smooth_skeleton_weighting(self) -> None:
        prev = np.zeros((33, 2), dtype=np.float64)
        cur = np.ones((33, 2), dtype=np.float64)

        smoothed = ema_smooth_skeleton(cur, prev_skeleton=prev, alpha=0.4)
        np.testing.assert_allclose(smoothed, 0.4)

    def test_normalized_filter_reset_behavior(self) -> None:
        filt = NormalizedSkeletonTemporalFilter(alpha=0.2)
        a = np.zeros((33, 2), dtype=np.float64)
        b = np.ones((33, 2), dtype=np.float64)

        _ = filt.update(a)
        smoothed = filt.update(b)
        assert np.all(smoothed < 1.0)

        filt.reset()
        after_reset = filt.update(b)
        np.testing.assert_allclose(after_reset, b)


class TestFSMAndTimeBasedLogic:
    def _squat_frame(self, knee: float, hip: float = 110.0) -> dict[str, float]:
        return {
            "left_knee": knee,
            "right_knee": knee,
            "left_elbow": 165.0,
            "right_elbow": 165.0,
            "left_shoulder": 35.0,
            "right_shoulder": 35.0,
            "left_hip": hip,
            "right_hip": hip,
        }

    def _run_squat_cycles(self, clf: ExerciseClassifier, cycles: int, dt: float) -> tuple[str | None, int]:
        for _ in range(int(0.35 / dt)):
            clf.update(self._squat_frame(85.0, hip=100.0), dt=dt)
        for _ in range(cycles):
            for _ in range(int(0.25 / dt)):
                clf.update(self._squat_frame(90.0, hip=100.0), dt=dt)
            for _ in range(int(0.25 / dt)):
                clf.update(self._squat_frame(170.0, hip=170.0), dt=dt)
        return clf.current_exercise, clf.rep_count

    def test_classifier_uses_time_not_frame_count(self) -> None:
        clf_fast = ExerciseClassifier(default_fps=60.0)
        clf_slow = ExerciseClassifier(default_fps=15.0)

        ex_fast, reps_fast = self._run_squat_cycles(clf_fast, cycles=3, dt=1.0 / 60.0)
        ex_slow, reps_slow = self._run_squat_cycles(clf_slow, cycles=3, dt=1.0 / 15.0)

        assert ex_fast == "squat"
        assert ex_slow == "squat"
        assert reps_fast == reps_slow == 3

    def test_count_reps_from_signal_respects_sample_rate(self) -> None:
        duration_s = 8.0
        reps_true = 4

        fs_a = 20.0
        fs_b = 50.0
        t_a = np.arange(0.0, duration_s, 1.0 / fs_a)
        t_b = np.arange(0.0, duration_s, 1.0 / fs_b)

        angle_a = 130.0 + 45.0 * np.cos(2.0 * np.pi * reps_true * t_a / duration_s)
        angle_b = 130.0 + 45.0 * np.cos(2.0 * np.pi * reps_true * t_b / duration_s)

        reps_a = count_reps_from_signal(angle_a, up_threshold=160.0, down_threshold=100.0, sample_rate_hz=fs_a)
        reps_b = count_reps_from_signal(angle_b, up_threshold=160.0, down_threshold=100.0, sample_rate_hz=fs_b)

        assert reps_a == reps_true
        assert reps_b == reps_true


class TestFusionAndPCAAlignment:
    def test_fuse_label_prefers_pca_only_when_confident(self) -> None:
        assert fuse_exercise_labels("squat", "pushup", pca_confidence=0.5) == "squat"
        assert fuse_exercise_labels("squat", "pushup", pca_confidence=0.95) == "pushup"
        assert fuse_exercise_labels(None, "jumping_jack", pca_confidence=0.8) == "jumping_jack"

    def test_align_components_corrects_sign_flip(self) -> None:
        ref = np.eye(4, 2, dtype=np.float64)
        flipped = ref.copy()
        flipped[:, 1] *= -1.0

        aligned = align_components_to_reference(flipped, reference_components=ref)

        np.testing.assert_allclose(aligned, ref, atol=1e-12)

    def test_align_components_reorders_to_reference(self) -> None:
        ref = np.eye(4, 2, dtype=np.float64)
        swapped = ref[:, [1, 0]]

        aligned = align_components_to_reference(swapped, reference_components=ref)

        np.testing.assert_allclose(aligned, ref, atol=1e-12)
