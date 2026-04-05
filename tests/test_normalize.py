"""
Unit tests for src/normalize.py  (Developer 1 — Procrustes normalisation).

Test classes
------------
TestTranslation          — translate() centres the hip midpoint at the origin
TestScaling              — scale() makes the torso length equal to 1
TestProcrustesRotation   — R* is orthogonal, det = +1, recovers a known angle
TestNormalizeSkeleton    — full pipeline: shape, hip at origin, torso = 1
TestNormalizeSequence    — batch wrapper behaves consistently
TestEdgeCases            — degenerate inputs and bad shapes
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.normalize import (
    REFERENCE_POSE,
    compute_hip_center,
    compute_shoulder_center,
    compute_torso_length,
    normalize_sequence,
    normalize_skeleton,
    procrustes_rotation,
    rotate,
    scale,
    translate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_LANDMARKS: int = 33


def _make_skeleton(seed: int = 0) -> NDArray[np.float64]:
    """Random skeleton in R^{33×2} with key joints placed realistically."""
    rng = np.random.default_rng(seed)
    sk = rng.uniform(0.2, 0.8, size=(N_LANDMARKS, 2))

    # key joints in anatomically plausible positions
    sk[11] = [0.38, 0.28]   # left shoulder
    sk[12] = [0.62, 0.28]   # right shoulder
    sk[13] = [0.32, 0.45]   # left elbow
    sk[14] = [0.68, 0.45]   # right elbow
    sk[15] = [0.28, 0.62]   # left wrist
    sk[16] = [0.72, 0.62]   # right wrist
    sk[23] = [0.44, 0.60]   # left hip
    sk[24] = [0.56, 0.60]   # right hip
    sk[25] = [0.43, 0.78]   # left knee
    sk[26] = [0.57, 0.78]   # right knee
    sk[27] = [0.42, 0.94]   # left ankle
    sk[28] = [0.58, 0.94]   # right ankle

    return sk.astype(np.float64)


def _make_skeleton_3d(seed: int = 0) -> NDArray[np.float64]:
    """Skeleton with visibility column (shape 33×3)."""
    sk = _make_skeleton(seed)
    vis = np.ones((N_LANDMARKS, 1), dtype=np.float64)
    return np.hstack([sk, vis])


def _rotation_matrix_2d(angle_rad: float) -> NDArray[np.float64]:
    """Standard 2×2 rotation matrix for a given angle."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

class TestTranslation:
    def test_hip_center_formula(self) -> None:
        sk = _make_skeleton()
        expected = (sk[23] + sk[24]) / 2.0
        np.testing.assert_array_almost_equal(compute_hip_center(sk), expected)

    def test_hip_center_at_origin_after_translate(self) -> None:
        sk = _make_skeleton()
        center = compute_hip_center(sk)
        sk_t = translate(sk, center)
        new_center = compute_hip_center(sk_t)
        np.testing.assert_array_almost_equal(new_center, [0.0, 0.0], decimal=12)

    def test_translate_preserves_shape(self) -> None:
        sk = _make_skeleton()
        assert translate(sk, compute_hip_center(sk)).shape == sk.shape

    def test_translate_is_linear(self) -> None:
        """translate(a + b, c) == translate(a, c) + b (translation is affine)."""
        sk = _make_skeleton()
        shift = np.array([0.1, 0.2])
        c = compute_hip_center(sk)
        np.testing.assert_array_almost_equal(
            translate(sk + shift, c),
            translate(sk, c) + shift,
        )

    def test_works_with_3d_skeleton(self) -> None:
        sk3 = _make_skeleton_3d()
        center = compute_hip_center(sk3)
        assert center.shape == (2,)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

class TestScaling:
    def test_torso_length_is_positive(self) -> None:
        assert compute_torso_length(_make_skeleton()) > 0.0

    def test_scale_makes_torso_unit_length(self) -> None:
        sk = _make_skeleton()
        center = compute_hip_center(sk)
        sk_t = translate(sk, center)
        d = compute_torso_length(sk)
        sk_s = scale(sk_t, d)
        new_torso = float(np.linalg.norm(
            compute_shoulder_center(sk_s) - compute_hip_center(sk_s)
        ))
        np.testing.assert_almost_equal(new_torso, 1.0, decimal=6)

    def test_scale_preserves_shape(self) -> None:
        sk = _make_skeleton()
        assert scale(sk, 2.0).shape == sk.shape

    def test_scale_zero_d_returns_copy(self) -> None:
        sk = _make_skeleton()
        result = scale(sk, 0.0)
        np.testing.assert_array_equal(result, sk)


# ---------------------------------------------------------------------------
# Procrustes rotation — the core Developer 1 contribution
# ---------------------------------------------------------------------------

class TestProcrustesRotation:
    """R* must be orthogonal, have det = +1, and recover a known rotation."""

    def _get_rotation(self, angle_deg: float = 30.0) -> NDArray[np.float64]:
        """Helper: normalise a skeleton, rotate it by angle_deg, then recover R*."""
        sk = _make_skeleton()
        center = compute_hip_center(sk)
        sk_t = translate(sk, center)
        sk_s = scale(sk_t, compute_torso_length(sk))

        # Apply a known rotation to simulate a tilted camera view
        angle_rad = np.deg2rad(angle_deg)
        R_applied = _rotation_matrix_2d(angle_rad)
        sk_rotated = sk_s @ R_applied.T   # row-vector convention

        # Procrustes should recover the inverse of R_applied to align back
        R_star = procrustes_rotation(sk_rotated, sk_s)
        return R_star

    @pytest.mark.parametrize("angle", [0, 15, 30, 45, 90, 135, 180])
    def test_orthogonality(self, angle: float) -> None:
        """R*ᵀ R* = I₂."""
        R = self._get_rotation(angle)
        np.testing.assert_allclose(R.T @ R, np.eye(2), atol=1e-10)

    @pytest.mark.parametrize("angle", [0, 15, 30, 45, 90, 135, 180])
    def test_determinant_is_plus_one(self, angle: float) -> None:
        """det(R*) = +1  (proper rotation, not a reflection)."""
        R = self._get_rotation(angle)
        np.testing.assert_almost_equal(np.linalg.det(R), 1.0, decimal=10)

    def test_recovers_known_rotation(self) -> None:
        """After rotating by α, Procrustes should return a rotation ≈ R(−α)."""
        sk = _make_skeleton()
        center = compute_hip_center(sk)
        sk_t = translate(sk, center)
        sk_s = scale(sk_t, compute_torso_length(sk))

        angle_rad = np.deg2rad(25.0)
        R_applied = _rotation_matrix_2d(angle_rad)
        sk_rotated = sk_s @ R_applied.T

        R_star = procrustes_rotation(sk_rotated, sk_s)

        # Applying R* to the rotated skeleton should approximately recover sk_s
        sk_recovered = sk_rotated @ R_star.T
        np.testing.assert_allclose(sk_recovered, sk_s, atol=1e-6)

    def test_identity_for_aligned_pose(self) -> None:
        """If S̃ already matches the reference, R* should be (close to) I₂."""
        ref = REFERENCE_POSE.copy()
        R_star = procrustes_rotation(ref, ref)
        np.testing.assert_allclose(R_star, np.eye(2), atol=1e-10)

    def test_output_shape(self) -> None:
        sk = _make_skeleton()
        R = procrustes_rotation(sk, REFERENCE_POSE)
        assert R.shape == (2, 2)

    def test_no_reflection(self) -> None:
        """Even with a symmetric pose the sign correction must prevent reflection."""
        # Build a pose that is already symmetric — SVD ambiguity is highest here
        ref = REFERENCE_POSE.copy()
        symmetric = ref.copy()
        symmetric[:, 0] = 0.0   # collapse all x-coords to zero

        R = procrustes_rotation(symmetric, ref)
        np.testing.assert_almost_equal(np.linalg.det(R), 1.0, decimal=10)


# ---------------------------------------------------------------------------
# rotate()
# ---------------------------------------------------------------------------

class TestRotate:
    def test_preserves_vector_lengths(self) -> None:
        sk = _make_skeleton()
        R = _rotation_matrix_2d(np.deg2rad(37))
        sk_rot = rotate(sk, R)
        for i in range(N_LANDMARKS):
            np.testing.assert_almost_equal(
                np.linalg.norm(sk[i]), np.linalg.norm(sk_rot[i]), decimal=10
            )

    def test_identity_rotation(self) -> None:
        sk = _make_skeleton()
        np.testing.assert_array_almost_equal(rotate(sk, np.eye(2)), sk)

    def test_output_shape(self) -> None:
        sk = _make_skeleton()
        assert rotate(sk, np.eye(2)).shape == (N_LANDMARKS, 2)


# ---------------------------------------------------------------------------
# Full pipeline — normalize_skeleton
# ---------------------------------------------------------------------------

class TestNormalizeSkeleton:
    def test_output_shape(self) -> None:
        sk = _make_skeleton()
        result = normalize_skeleton(sk)
        assert result.shape == (N_LANDMARKS, 2)

    def test_output_shape_from_3d_input(self) -> None:
        sk = _make_skeleton_3d()
        result = normalize_skeleton(sk)
        assert result.shape == (N_LANDMARKS, 2)

    def test_hip_center_at_origin(self) -> None:
        """After normalisation the hip midpoint must be (approximately) at origin."""
        sk = _make_skeleton()
        s_hat = normalize_skeleton(sk)
        hip_mid = compute_hip_center(s_hat)
        np.testing.assert_allclose(hip_mid, [0.0, 0.0], atol=1e-6)

    def test_torso_length_is_one(self) -> None:
        """After normalisation the torso length must be (approximately) 1."""
        sk = _make_skeleton()
        s_hat = normalize_skeleton(sk)
        torso = float(np.linalg.norm(
            compute_shoulder_center(s_hat) - compute_hip_center(s_hat)
        ))
        np.testing.assert_almost_equal(torso, 1.0, decimal=5)

    def test_rotation_invariant(self) -> None:
        """Normalising a skeleton and its rotated version should give similar results."""
        sk = _make_skeleton()
        angle_rad = np.deg2rad(40.0)
        R_applied = _rotation_matrix_2d(angle_rad)

        sk_rotated = sk.copy()
        sk_rotated[:, :2] = sk[:, :2] @ R_applied.T

        s_hat_orig = normalize_skeleton(sk)
        s_hat_rot = normalize_skeleton(sk_rotated)

        # After Procrustes both should be close to the same reference
        np.testing.assert_allclose(s_hat_orig, s_hat_rot, atol=0.05)

    def test_no_procrustes_still_centres(self) -> None:
        """Even without Procrustes, hip must be at origin and torso = 1."""
        sk = _make_skeleton()
        s_hat = normalize_skeleton(sk, apply_procrustes=False)
        hip_mid = compute_hip_center(s_hat)
        np.testing.assert_allclose(hip_mid, [0.0, 0.0], atol=1e-6)

    def test_custom_reference_pose(self) -> None:
        """Should accept an arbitrary reference pose without raising."""
        sk = _make_skeleton()
        custom_ref = np.zeros((33, 2), dtype=np.float64)
        custom_ref[11] = [-0.3, -1.0]
        custom_ref[12] = [0.3, -1.0]
        custom_ref[23] = [-0.15, 0.0]
        custom_ref[24] = [0.15, 0.0]
        result = normalize_skeleton(sk, reference_pose=custom_ref)
        assert result.shape == (N_LANDMARKS, 2)


# ---------------------------------------------------------------------------
# Sequence wrapper
# ---------------------------------------------------------------------------

class TestNormalizeSequence:
    def test_output_shape(self) -> None:
        T = 20
        skeletons = np.stack([_make_skeleton(seed=i) for i in range(T)])
        result = normalize_sequence(skeletons)
        assert result.shape == (T, N_LANDMARKS, 2)

    def test_each_frame_matches_single(self) -> None:
        """normalize_sequence must match normalize_skeleton frame by frame."""
        T = 10
        skeletons = np.stack([_make_skeleton(seed=i) for i in range(T)])
        batch = normalize_sequence(skeletons)
        for t in range(T):
            expected = normalize_skeleton(skeletons[t])
            np.testing.assert_allclose(batch[t], expected, atol=1e-12)

    def test_rejects_2d_input(self) -> None:
        with pytest.raises(ValueError):
            normalize_sequence(np.zeros((10, 66)))

    def test_rejects_wrong_landmark_count(self) -> None:
        with pytest.raises(ValueError):
            normalize_sequence(np.zeros((10, 30, 2)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
