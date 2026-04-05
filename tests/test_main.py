"""
tests for skeleton extraction, normalization, features, and classification.
"""

import numpy as np
import pytest

from src.skeleton import (
    build_adjacency_matrix,
    build_degree_matrix,
    build_graph_laplacian,
    NUM_LANDMARKS,
    SKELETON_CONNECTIONS,
)
from src.normalize import (
    compute_hip_center,
    compute_shoulder_center,
    translate,
    compute_torso_length,
    scale,
    rotation_matrix,
    rotate,
    normalize_skeleton,
)
from src.features import (
    compute_angle,
    compute_key_angles,
    flatten_skeleton,
    euclidean_distance,
    cosine_similarity,
)
from src.classify import ExerciseClassifier, count_reps_from_signal


# --- fixtures ---

def _make_skeleton(seed=42) -> np.ndarray:
    """create a realistic-ish skeleton in R^{33x2} for testing."""
    rng = np.random.RandomState(seed)
    skeleton = rng.rand(NUM_LANDMARKS, 2) * 0.5 + 0.25

    # place key joints in known positions for predictable tests
    skeleton[11] = [0.4, 0.3]   # left shoulder
    skeleton[12] = [0.6, 0.3]   # right shoulder
    skeleton[13] = [0.35, 0.45] # left elbow
    skeleton[14] = [0.65, 0.45] # right elbow
    skeleton[15] = [0.3, 0.6]   # left wrist
    skeleton[16] = [0.7, 0.6]   # right wrist
    skeleton[23] = [0.45, 0.6]  # left hip
    skeleton[24] = [0.55, 0.6]  # right hip
    skeleton[25] = [0.43, 0.8]  # left knee
    skeleton[26] = [0.57, 0.8]  # right knee
    skeleton[27] = [0.42, 0.95] # left ankle
    skeleton[28] = [0.58, 0.95] # right ankle

    return skeleton


def _standing_angles() -> dict[str, float]:
    """angles for a neutral standing position — should NOT trigger any exercise."""
    return {
        "left_knee": 170, "right_knee": 170,
        "left_elbow": 165, "right_elbow": 165,
        "left_shoulder": 30, "right_shoulder": 30,
        "left_hip": 170, "right_hip": 170,
    }


def _squat_down_angles() -> dict[str, float]:
    """angles for the bottom of a squat."""
    return {
        "left_knee": 75, "right_knee": 75,
        "left_elbow": 165, "right_elbow": 165,
        "left_shoulder": 35, "right_shoulder": 35,
        "left_hip": 85, "right_hip": 85,
    }


def _pushup_down_angles() -> dict[str, float]:
    """angles for the bottom of a pushup."""
    return {
        "left_knee": 170, "right_knee": 170,
        "left_elbow": 75, "right_elbow": 75,
        "left_shoulder": 50, "right_shoulder": 50,
        "left_hip": 140, "right_hip": 140,
    }


def _jj_up_angles() -> dict[str, float]:
    """angles for jumping jack with arms up."""
    return {
        "left_knee": 170, "right_knee": 170,
        "left_elbow": 165, "right_elbow": 165,
        "left_shoulder": 150, "right_shoulder": 150,
        "left_hip": 170, "right_hip": 170,
    }


# --- adjacency matrix tests ---

class TestAdjacencyMatrix:
    def test_shape(self):
        A = build_adjacency_matrix()
        assert A.shape == (NUM_LANDMARKS, NUM_LANDMARKS)

    def test_symmetric(self):
        A = build_adjacency_matrix()
        np.testing.assert_array_equal(A, A.T)

    def test_binary(self):
        A = build_adjacency_matrix()
        assert set(np.unique(A)).issubset({0, 1})

    def test_zero_diagonal(self):
        A = build_adjacency_matrix()
        np.testing.assert_array_equal(np.diag(A), np.zeros(NUM_LANDMARKS))

    def test_connections_count(self):
        A = build_adjacency_matrix()
        # each connection creates 2 entries (symmetric), so nonzeros = 2 * edges
        assert A.sum() == 2 * len(SKELETON_CONNECTIONS)


class TestDegreeMatrix:
    def test_diagonal(self):
        A = build_adjacency_matrix()
        D = build_degree_matrix(A)
        assert D.shape == (NUM_LANDMARKS, NUM_LANDMARKS)
        # should be diagonal
        off_diag = D - np.diag(np.diag(D))
        np.testing.assert_array_equal(off_diag, np.zeros_like(off_diag))

    def test_degrees_match(self):
        A = build_adjacency_matrix()
        D = build_degree_matrix(A)
        np.testing.assert_array_equal(np.diag(D), A.sum(axis=1))


class TestGraphLaplacian:
    def test_shape(self):
        A = build_adjacency_matrix()
        L = build_graph_laplacian(A)
        assert L.shape == (NUM_LANDMARKS, NUM_LANDMARKS)

    def test_symmetric(self):
        A = build_adjacency_matrix()
        L = build_graph_laplacian(A)
        np.testing.assert_array_equal(L, L.T)

    def test_positive_semidefinite(self):
        A = build_adjacency_matrix()
        L = build_graph_laplacian(A)
        eigenvalues = np.linalg.eigvalsh(L)
        assert np.all(eigenvalues >= -1e-10)

    def test_smallest_eigenvalue_zero(self):
        A = build_adjacency_matrix()
        L = build_graph_laplacian(A)
        eigenvalues = np.sort(np.linalg.eigvalsh(L))
        assert abs(eigenvalues[0]) < 1e-10

    def test_row_sums_zero(self):
        # each row of L sums to 0 (L * 1 = 0)
        A = build_adjacency_matrix()
        L = build_graph_laplacian(A)
        row_sums = L.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.zeros(NUM_LANDMARKS))


# --- normalization tests ---

class TestNormalization:
    def test_hip_center(self):
        sk = _make_skeleton()
        center = compute_hip_center(sk)
        expected = (sk[23] + sk[24]) / 2
        np.testing.assert_array_almost_equal(center, expected)

    def test_translation_centers_hips(self):
        sk = _make_skeleton()
        center = compute_hip_center(sk)
        translated = translate(sk, center)
        new_center = compute_hip_center(translated)
        np.testing.assert_array_almost_equal(new_center, [0, 0], decimal=10)

    def test_scale_normalizes_torso(self):
        sk = _make_skeleton()
        center = compute_hip_center(sk)
        translated = translate(sk, center)
        d_ref = compute_torso_length(sk)
        scaled = scale(translated, d_ref)
        # recompute torso length after scaling
        sh_mid = compute_shoulder_center(scale(sk, d_ref))
        hip_mid = compute_hip_center(scale(sk, d_ref))
        np.testing.assert_almost_equal(np.linalg.norm(sh_mid - hip_mid), 1.0, decimal=5)

    def test_rotation_matrix_orthogonal(self):
        for angle in [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi]:
            R = rotation_matrix(angle)
            # R^T R = I
            np.testing.assert_array_almost_equal(R.T @ R, np.eye(2))
            # det(R) = 1
            np.testing.assert_almost_equal(np.linalg.det(R), 1.0)

    def test_rotation_preserves_lengths(self):
        sk = _make_skeleton()
        angle = np.pi / 6
        rotated = rotate(sk, angle)
        for i in range(NUM_LANDMARKS):
            np.testing.assert_almost_equal(
                np.linalg.norm(sk[i]), np.linalg.norm(rotated[i]), decimal=10
            )

    def test_normalize_skeleton_output_shape(self):
        sk = _make_skeleton()
        result = normalize_skeleton(sk)
        assert result.shape == (NUM_LANDMARKS, 2)


# --- features tests ---

class TestAngles:
    def test_right_angle(self):
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 0.0])
        angle = compute_angle(a, b, c)
        np.testing.assert_almost_equal(angle, 90.0, decimal=5)

    def test_straight_angle(self):
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 0.0])
        c = np.array([0.0, -1.0])
        angle = compute_angle(a, b, c)
        np.testing.assert_almost_equal(angle, 180.0, decimal=5)

    def test_zero_angle(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([2.0, 0.0])
        angle = compute_angle(a, b, c)
        np.testing.assert_almost_equal(angle, 0.0, decimal=5)

    def test_45_degree_angle(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 1.0])
        angle = compute_angle(a, b, c)
        np.testing.assert_almost_equal(angle, 45.0, decimal=5)

    def test_key_angles_returns_all_keys(self):
        sk = _make_skeleton()
        angles = compute_key_angles(sk)
        expected_keys = {
            "left_elbow", "right_elbow", "left_knee", "right_knee",
            "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        }
        assert set(angles.keys()) == expected_keys

    def test_angles_range(self):
        sk = _make_skeleton()
        angles = compute_key_angles(sk)
        for name, val in angles.items():
            assert 0 <= val <= 180, f"{name} angle {val} out of [0, 180] range"


class TestDistanceMetrics:
    def test_euclidean_distance_same(self):
        s = np.random.rand(66)
        assert euclidean_distance(s, s) == 0.0

    def test_euclidean_distance_symmetric(self):
        s1, s2 = np.random.rand(66), np.random.rand(66)
        np.testing.assert_almost_equal(
            euclidean_distance(s1, s2), euclidean_distance(s2, s1)
        )

    def test_euclidean_distance_positive(self):
        s1, s2 = np.random.rand(66), np.random.rand(66) + 1
        assert euclidean_distance(s1, s2) > 0

    def test_cosine_similarity_identical(self):
        s = np.random.rand(66) + 0.1
        np.testing.assert_almost_equal(cosine_similarity(s, s), 1.0)

    def test_cosine_similarity_range(self):
        s1, s2 = np.random.rand(66), np.random.rand(66)
        sim = cosine_similarity(s1, s2)
        assert -1.0 <= sim <= 1.0

    def test_flatten_skeleton_shape(self):
        sk = _make_skeleton()
        flat = flatten_skeleton(sk)
        assert flat.shape == (66,)


# --- classifier tests ---

class TestClassifier:
    def test_initial_state(self):
        clf = ExerciseClassifier()
        assert clf.current_exercise is None
        assert clf.rep_count == 0

    def test_reset(self):
        clf = ExerciseClassifier()
        clf.rep_count = 5
        clf.current_exercise = "squat"
        clf.reset()
        assert clf.current_exercise is None
        assert clf.rep_count == 0

    def test_standing_stays_idle(self):
        """standing still should never trigger an exercise."""
        clf = ExerciseClassifier()
        for _ in range(50):
            exercise, reps = clf.update(_standing_angles())
        assert clf.current_exercise is None
        assert clf.rep_count == 0

    def test_squat_detection(self):
        """squat cycle (standing -> deep squat -> standing) should be detected."""
        clf = ExerciseClassifier()
        # feed enough squat-down frames to pass the confidence gate
        for _ in range(30):
            clf.update(_squat_down_angles())
        assert clf.current_exercise == "squat"

    def test_squat_reps(self):
        """full squat cycles should produce rep counts."""
        clf = ExerciseClassifier()
        for cycle in range(4):
            # standing phase
            for _ in range(20):
                clf.update({
                    "left_knee": 170, "right_knee": 170,
                    "left_elbow": 165, "right_elbow": 165,
                    "left_shoulder": 30, "right_shoulder": 30,
                    "left_hip": 170, "right_hip": 170,
                })
            # squat phase
            for _ in range(20):
                clf.update(_squat_down_angles())
        assert clf.current_exercise == "squat"
        # at least 1 rep should be counted from the oscillation
        assert clf.rep_count >= 1

    def test_pushup_detection(self):
        """pushup down position should be detected after confidence frames."""
        clf = ExerciseClassifier()
        for _ in range(30):
            clf.update(_pushup_down_angles())
        assert clf.current_exercise == "pushup"

    def test_jumping_jack_detection(self):
        """jumping jack arms-up should be detected after confidence frames."""
        clf = ExerciseClassifier()
        for _ in range(30):
            clf.update(_jj_up_angles())
        assert clf.current_exercise == "jumping_jack"

    def test_idle_after_sustained_standing(self):
        """exercise should drop to idle after enough standing frames."""
        clf = ExerciseClassifier()
        # first detect a squat
        for _ in range(30):
            clf.update(_squat_down_angles())
        assert clf.current_exercise == "squat"

        # now stand for a long time
        for _ in range(50):
            clf.update(_standing_angles())
        assert clf.current_exercise is None

    def test_no_false_trigger_slight_bend(self):
        """slightly bent knees (like casual standing) should not trigger squat."""
        clf = ExerciseClassifier()
        slightly_bent = _standing_angles()
        slightly_bent["left_knee"] = 155
        slightly_bent["right_knee"] = 155
        for _ in range(50):
            clf.update(slightly_bent)
        assert clf.current_exercise is None

    def test_count_reps_from_signal(self):
        # create a sinusoidal signal mimicking squat angle oscillation
        # 6 full cycles so find_peaks detects interior peaks reliably
        t = np.linspace(0, 12 * np.pi, 600)
        signal = 120 + 50 * np.cos(t)  # oscillates between 70 and 170
        reps = count_reps_from_signal(signal, up_threshold=160, min_distance=15)
        assert reps >= 1  # should detect at least 1 rep


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
