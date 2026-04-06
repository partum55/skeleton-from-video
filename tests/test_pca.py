"""Tests for PCA (SVD, component selection, projection, reconstruction)."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.linalg_utils import center_matrix, flatten_poses, select_n_components
from src.pca import DEFAULT_VARIANCE_THRESHOLD, PCA, fit_pca, project_poses


RNG_SEED: int = 42
N_FRAMES: int = 300     # T — enough frames for stable SVD
N_FEATURES: int = 66    # d — matches flattened skeleton size
N_LANDMARKS: int = 33


def _make_low_rank_data(
    n_frames: int = N_FRAMES,
    n_features: int = N_FEATURES,
    true_rank: int = 5,
    noise_std: float = 0.01,
    seed: int = RNG_SEED,
) -> np.ndarray:
    """Synthetic data with *true_rank* dominant directions + small noise.

    The top-k singular values are much larger than the rest, so PCA should
    select exactly *true_rank* components at a high variance threshold.
    """
    rng = np.random.default_rng(seed)
    # Low-rank signal
    U = rng.standard_normal((n_frames, true_rank))
    V = rng.standard_normal((n_features, true_rank))
    signal = U @ V.T  # shape (n_frames, n_features)
    # Tiny noise
    noise = noise_std * rng.standard_normal((n_frames, n_features))
    return (signal + noise).astype(np.float64)


def _make_isotropic_data(
    n_frames: int = N_FRAMES,
    n_features: int = N_FEATURES,
    seed: int = RNG_SEED,
) -> np.ndarray:
    """Isotropic Gaussian: all singular values approximately equal."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_frames, n_features)).astype(np.float64)


# linalg_utils.center_matrix

class TestCenterMatrix:
    def test_output_shape(self) -> None:
        X = np.ones((10, 5))
        X_c, mu = center_matrix(X)
        assert X_c.shape == X.shape
        assert mu.shape == (5,)

    def test_column_mean_is_zero(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 8))
        X_c, _ = center_matrix(X)
        np.testing.assert_allclose(X_c.mean(axis=0), np.zeros(8), atol=1e-12)

    def test_mu_matches_original_mean(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 4)) + 5.0
        _, mu = center_matrix(X)
        np.testing.assert_allclose(mu, X.mean(axis=0), rtol=1e-12)

    def test_rejects_1d_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            center_matrix(np.ones(10))


# linalg_utils.select_n_components

class TestSelectNComponents:
    def test_perfect_single_component(self) -> None:
        # One dominant singular value — k should be 1
        s = np.array([100.0, 0.001, 0.001])
        k = select_n_components(s, variance_threshold=0.95)
        assert k == 1

    def test_all_equal_components(self) -> None:
        # Equal singular values: need all of them to reach 95 %
        s = np.ones(10)
        k = select_n_components(s, variance_threshold=0.95)
        assert k == 10

    def test_threshold_one_returns_all(self) -> None:
        s = np.array([3.0, 2.0, 1.0])
        k = select_n_components(s, variance_threshold=1.0)
        assert k == len(s)

    def test_result_in_valid_range(self) -> None:
        rng = np.random.default_rng(7)
        s = np.sort(rng.uniform(0, 10, 20))[::-1]
        k = select_n_components(s, variance_threshold=0.90)
        assert 1 <= k <= len(s)

    def test_rejects_invalid_threshold(self) -> None:
        with pytest.raises(ValueError):
            select_n_components(np.array([1.0, 2.0]), variance_threshold=0.0)

    def test_rejects_2d_input(self) -> None:
        with pytest.raises(ValueError):
            select_n_components(np.ones((3, 3)))


# linalg_utils.flatten_poses

class TestFlattenPoses:
    def test_output_shape(self) -> None:
        skeletons = np.zeros((50, N_LANDMARKS, 2))
        X = flatten_poses(skeletons)
        assert X.shape == (50, N_LANDMARKS * 2)

    def test_values_preserved(self) -> None:
        skeletons = np.arange(33 * 2, dtype=np.float64).reshape(1, 33, 2)
        X = flatten_poses(skeletons)
        np.testing.assert_array_equal(X[0], skeletons[0].ravel())

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            flatten_poses(np.zeros((10, 33, 3)))  # 3 columns, not 2

    def test_rejects_2d_input(self) -> None:
        with pytest.raises(ValueError):
            flatten_poses(np.zeros((10, 66)))


# PCA: orthonormality of V_k

class TestPCAOrthonormality:
    """V_kᵀ V_k must equal the k×k identity matrix."""

    def test_orthonormality_low_rank_data(self) -> None:
        X = _make_low_rank_data()
        pca = PCA().fit(X)
        Vk = pca.components_   # shape (d, k)
        k = pca.n_components_
        product = Vk.T @ Vk     # should be I_k
        np.testing.assert_allclose(product, np.eye(k), atol=1e-10)

    def test_orthonormality_isotropic_data(self) -> None:
        X = _make_isotropic_data()
        pca = PCA().fit(X)
        Vk = pca.components_
        k = pca.n_components_
        np.testing.assert_allclose(Vk.T @ Vk, np.eye(k), atol=1e-10)

    def test_columns_have_unit_norm(self) -> None:
        X = _make_low_rank_data()
        pca = PCA().fit(X)
        norms = np.linalg.norm(pca.components_, axis=0)  # one norm per column
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-10)


# PCA: zero-mean projections

class TestPCAZeroMeanProjection:
    """Z = X̃ V_k, and X̃ is already zero-mean, so Z must be zero-mean."""

    def test_zero_mean_fit_transform(self) -> None:
        X = _make_low_rank_data()
        Z = PCA().fit_transform(X)
        np.testing.assert_allclose(Z.mean(axis=0), np.zeros(Z.shape[1]), atol=1e-10)

    def test_zero_mean_separate_fit_transform(self) -> None:
        X = _make_low_rank_data()
        pca = PCA().fit(X)
        Z = pca.transform(X)
        np.testing.assert_allclose(Z.mean(axis=0), np.zeros(Z.shape[1]), atol=1e-10)

    def test_projection_shape(self) -> None:
        X = _make_low_rank_data()
        pca = PCA().fit(X)
        Z = pca.transform(X)
        assert Z.shape == (N_FRAMES, pca.n_components_)


# PCA: variance retention

class TestPCAVarianceRetention:
    """Cumulative variance ratio must reach the requested threshold."""

    @pytest.mark.parametrize("threshold", [0.80, 0.90, 0.95, 0.99])
    def test_threshold_reached(self, threshold: float) -> None:
        X = _make_low_rank_data()
        pca = PCA(variance_threshold=threshold).fit(X)
        retained = pca.cumulative_variance_ratio[-1]
        assert retained >= threshold - 1e-9, (
            f"Only {retained:.4f} variance retained for threshold {threshold}"
        )

    def test_variance_ratio_sums_leq_one(self) -> None:
        X = _make_isotropic_data()
        pca = PCA().fit(X)
        assert pca.variance_ratio_.sum() <= 1.0 + 1e-10

    def test_fewer_components_for_lower_threshold(self) -> None:
        X = _make_low_rank_data()
        k_90 = PCA(variance_threshold=0.90).fit(X).n_components_
        k_99 = PCA(variance_threshold=0.99).fit(X).n_components_
        assert k_90 <= k_99

    def test_singular_values_descending(self) -> None:
        X = _make_low_rank_data()
        pca = PCA().fit(X)
        s = pca.singular_values_
        assert np.all(s[:-1] >= s[1:]), "singular values must be non-increasing"


# PCA: reconstruction error decreases with k

class TestPCAReconstructionError:
    """‖X̃ − Z V_kᵀ‖_F must be non-increasing as k grows."""

    def test_error_decreases_with_k(self) -> None:
        X = _make_low_rank_data()
        X_centered, _ = center_matrix(X)

        errors: list[float] = []
        for k in range(1, 10):
            # Force exactly k components by using a threshold just above the
            # cumulative variance at k (we compute it from the full SVD).
            _, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
            Vk = Vt[:k].T
            Z_k = X_centered @ Vk
            X_reconstructed = Z_k @ Vk.T
            err = np.linalg.norm(X_centered - X_reconstructed, "fro")
            errors.append(err)

        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-9, (
                f"Reconstruction error increased from k={i+1} to k={i+2}: "
                f"{errors[i]:.6f} → {errors[i+1]:.6f}"
            )

    def test_full_rank_gives_near_zero_error(self) -> None:
        # With k = min(T, d) the reconstruction should be near-perfect.
        T, d = 50, 10
        rng = np.random.default_rng(99)
        X = rng.standard_normal((T, d))
        X_centered, _ = center_matrix(X)
        _, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        Vk = Vt.T  # all components
        Z = X_centered @ Vk
        X_hat = Z @ Vk.T
        err = np.linalg.norm(X_centered - X_hat, "fro")
        assert err < 1e-9


# PCA: interface correctness

class TestPCAInterface:
    def test_fit_returns_self(self) -> None:
        pca = PCA()
        result = pca.fit(_make_low_rank_data())
        assert result is pca

    def test_fit_transform_matches_fit_then_transform(self) -> None:
        X = _make_low_rank_data()
        Z_a = PCA().fit_transform(X)
        pca = PCA()
        Z_b = pca.fit(X).transform(X)
        np.testing.assert_allclose(Z_a, Z_b, atol=1e-12)

    def test_inverse_transform_shape(self) -> None:
        X = _make_low_rank_data()
        pca = PCA().fit(X)
        Z = pca.transform(X)
        X_hat = pca.inverse_transform(Z)
        assert X_hat.shape == X.shape

    def test_single_vector_transform(self) -> None:
        X = _make_low_rank_data()
        pca = PCA().fit(X)
        single = X[0]               # shape (66,)
        z = pca.transform(single)
        assert z.shape == (pca.n_components_,)

    def test_fit_pca_convenience(self) -> None:
        X = _make_low_rank_data()
        pca = fit_pca(X, variance_threshold=0.90)
        assert isinstance(pca, PCA)
        assert pca.n_components_ is not None

    def test_project_poses_convenience(self) -> None:
        X = _make_low_rank_data()
        pca = fit_pca(X)
        Z = project_poses(X, pca)
        assert Z.shape[0] == X.shape[0]
        assert Z.shape[1] == pca.n_components_


# PCA: edge cases and error handling

class TestPCAEdgeCases:
    def test_transform_before_fit_raises(self) -> None:
        pca = PCA()
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.transform(np.zeros((5, 66)))

    def test_invalid_variance_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            PCA(variance_threshold=0.0)

        with pytest.raises(ValueError):
            PCA(variance_threshold=1.5)

    def test_fit_rejects_1d_input(self) -> None:
        pca = PCA()
        with pytest.raises(ValueError):
            pca.fit(np.zeros(66))

    def test_attributes_set_after_fit(self) -> None:
        X = _make_low_rank_data()
        pca = PCA().fit(X)
        assert pca.mean_ is not None
        assert pca.components_ is not None
        assert pca.singular_values_ is not None
        assert pca.n_components_ is not None
        assert pca.variance_ratio_ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
