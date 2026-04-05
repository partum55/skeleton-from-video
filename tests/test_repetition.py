"""
Unit tests for src/repetition.py  (Developer 3 — repetition counting).

Test classes
------------
TestBuildDesignMatrix     — shape, sin/cos columns, constant column
TestFitFrequency          — normal equations hold, residual is non-negative
TestFindBestFrequency     — recovers known frequency from synthetic sinusoid
TestCountRepetitions      — round(ω̂ · T / 2π) formula
TestClassifyExercise      — label returned for synthetic Z matrices
TestFullPipeline          — count_reps_and_classify end-to-end
TestEdgeCases             — bad inputs raise correct exceptions
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.repetition import (
    build_design_matrix,
    build_frequency_grid,
    classify_exercise,
    count_repetitions,
    count_reps_and_classify,
    find_best_frequency,
    fit_frequency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sinusoid(
    T: int,
    true_reps: int,
    amplitude: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Synthetic sinusoidal signal with a known repetition count."""
    omega_true = 2.0 * np.pi * true_reps / T
    t = np.arange(1, T + 1, dtype=np.float64)
    signal = amplitude * np.sin(omega_true * t)
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        signal += noise_std * rng.standard_normal(T)
    return signal


def _make_Z(
    T: int,
    z1_mean: float = 0.0,
    z2_std: float = 0.1,
    amplitude: float = 0.5,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Synthetic PCA projection matrix Z of shape (T, 3)."""
    rng = np.random.default_rng(seed)
    omega = 2.0 * np.pi * 5 / T
    t = np.arange(1, T + 1, dtype=np.float64)
    z1 = amplitude * np.sin(omega * t) + z1_mean
    z2 = z2_std * rng.standard_normal(T)
    z3 = 0.01 * rng.standard_normal(T)
    return np.column_stack([z1, z2, z3]).astype(np.float64)


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

class TestBuildDesignMatrix:
    def test_shape(self) -> None:
        A = build_design_matrix(omega=0.5, T=100)
        assert A.shape == (100, 3)

    def test_constant_column(self) -> None:
        A = build_design_matrix(omega=0.3, T=50)
        np.testing.assert_array_equal(A[:, 2], np.ones(50))

    def test_sin_column_values(self) -> None:
        omega = 0.4
        T = 30
        A = build_design_matrix(omega=omega, T=T)
        t = np.arange(1, T + 1, dtype=np.float64)
        np.testing.assert_allclose(A[:, 0], np.sin(omega * t), atol=1e-12)

    def test_cos_column_values(self) -> None:
        omega = 0.4
        T = 30
        A = build_design_matrix(omega=omega, T=T)
        t = np.arange(1, T + 1, dtype=np.float64)
        np.testing.assert_allclose(A[:, 1], np.cos(omega * t), atol=1e-12)

    def test_zero_omega_gives_zero_sin(self) -> None:
        A = build_design_matrix(omega=0.0, T=20)
        np.testing.assert_allclose(A[:, 0], np.zeros(20), atol=1e-12)

    def test_rejects_zero_T(self) -> None:
        with pytest.raises(ValueError):
            build_design_matrix(omega=0.5, T=0)


# ---------------------------------------------------------------------------
# Least-squares fit for one frequency
# ---------------------------------------------------------------------------

class TestFitFrequency:
    def test_residual_is_non_negative(self) -> None:
        z1 = _make_sinusoid(T=200, true_reps=5)
        omega = 2.0 * np.pi * 5 / 200
        _, residual = fit_frequency(omega, z1)
        assert residual >= 0.0

    def test_normal_equations_hold(self) -> None:
        """AᵀA ŵ ≈ Aᵀz₁  (verify the normal equations directly)."""
        T = 150
        z1 = _make_sinusoid(T=T, true_reps=6)
        omega = 2.0 * np.pi * 6 / T
        A = build_design_matrix(omega, T)
        w_hat, _ = fit_frequency(omega, z1)
        lhs = A.T @ A @ w_hat
        rhs = A.T @ z1
        np.testing.assert_allclose(lhs, rhs, atol=1e-8)

    def test_perfect_fit_gives_near_zero_residual(self) -> None:
        """At the true frequency with no noise, the residual should be ~0."""
        T = 200
        true_reps = 7
        omega_true = 2.0 * np.pi * true_reps / T
        t = np.arange(1, T + 1, dtype=np.float64)
        z1 = 2.0 * np.sin(omega_true * t) + 0.5 * np.cos(omega_true * t) + 0.3
        _, residual = fit_frequency(omega_true, z1)
        assert residual < 1e-8

    def test_w_hat_shape(self) -> None:
        z1 = _make_sinusoid(T=100, true_reps=4)
        w_hat, _ = fit_frequency(0.3, z1)
        assert w_hat.shape == (3,)

    def test_wrong_frequency_gives_larger_residual(self) -> None:
        T = 300
        true_reps = 8
        z1 = _make_sinusoid(T=T, true_reps=true_reps)
        omega_true = 2.0 * np.pi * true_reps / T
        omega_wrong = 2.0 * np.pi * 3 / T   # clearly wrong frequency
        _, res_true = fit_frequency(omega_true, z1)
        _, res_wrong = fit_frequency(omega_wrong, z1)
        assert res_wrong > res_true


# ---------------------------------------------------------------------------
# Grid search — core Developer 3 contribution
# ---------------------------------------------------------------------------

class TestFindBestFrequency:
    """The grid search must recover the known repetition frequency."""

    @pytest.mark.parametrize("true_reps", [3, 5, 8, 10, 15])
    def test_recovers_known_reps_noiseless(self, true_reps: int) -> None:
        T = 600
        z1 = _make_sinusoid(T=T, true_reps=true_reps)
        omega_hat, _, _ = find_best_frequency(z1, min_reps=1, max_reps=20, n_steps=1000)
        recovered = round(omega_hat * T / (2.0 * np.pi))
        assert recovered == true_reps, (
            f"Expected {true_reps} reps, recovered {recovered} "
            f"(ω̂={omega_hat:.5f}, ω_true={2*np.pi*true_reps/T:.5f})"
        )

    @pytest.mark.parametrize("true_reps", [4, 7, 12])
    def test_recovers_known_reps_with_noise(self, true_reps: int) -> None:
        """With low noise the estimate should still be correct."""
        T = 600
        z1 = _make_sinusoid(T=T, true_reps=true_reps, noise_std=0.05)
        omega_hat, _, _ = find_best_frequency(z1, min_reps=1, max_reps=20, n_steps=1000)
        recovered = round(omega_hat * T / (2.0 * np.pi))
        assert recovered == true_reps

    def test_residual_at_best_is_minimum(self) -> None:
        """The selected ω must have residual ≤ all other candidates."""
        T = 400
        z1 = _make_sinusoid(T=T, true_reps=6)
        omega_hat, _, best_res = find_best_frequency(z1, n_steps=300)

        from src.repetition import build_frequency_grid
        omegas = build_frequency_grid(T, n_steps=300)
        for omega in omegas:
            _, res = fit_frequency(omega, z1)
            assert res >= best_res - 1e-10, (
                f"Found smaller residual {res:.6f} at ω={omega:.4f} "
                f"than best {best_res:.6f} at ω̂={omega_hat:.4f}"
            )

    def test_omega_hat_in_search_range(self) -> None:
        T = 300
        z1 = _make_sinusoid(T=T, true_reps=5)
        omega_hat, _, _ = find_best_frequency(z1, min_reps=1, max_reps=20)
        omega_min = 2.0 * np.pi * 1 / T
        omega_max = 2.0 * np.pi * 20 / T
        assert omega_min <= omega_hat <= omega_max

    def test_rejects_too_short_signal(self) -> None:
        with pytest.raises(ValueError, match="10 samples"):
            find_best_frequency(np.zeros(5))

    def test_rejects_2d_input(self) -> None:
        with pytest.raises(ValueError):
            find_best_frequency(np.zeros((50, 2)))


# ---------------------------------------------------------------------------
# Repetition count formula
# ---------------------------------------------------------------------------

class TestCountRepetitions:
    @pytest.mark.parametrize("true_reps,T", [
        (5, 300), (10, 600), (1, 120), (20, 1200),
    ])
    def test_round_trip(self, true_reps: int, T: int) -> None:
        """ω = 2π·r/T  →  count_repetitions(ω, T) = r."""
        omega = 2.0 * np.pi * true_reps / T
        assert count_repetitions(omega, T) == true_reps

    def test_returns_non_negative(self) -> None:
        assert count_repetitions(0.0, 100) >= 0

    def test_returns_int(self) -> None:
        omega = 2.0 * np.pi * 5 / 300
        result = count_repetitions(omega, 300)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestClassifyExercise:
    def _w_hat(self, amplitude: float = 0.5) -> NDArray[np.float64]:
        """Coefficient vector with given amplitude."""
        return np.array([amplitude, 0.0, 0.0], dtype=np.float64)

    def test_squat_detected(self) -> None:
        Z = _make_Z(T=300, z1_mean=0.5, z2_std=0.05, amplitude=0.4)
        label = classify_exercise(Z, self._w_hat(0.4))
        assert label == "squat"

    def test_jumping_jack_detected(self) -> None:
        Z = _make_Z(T=300, z1_mean=0.0, z2_std=0.3, amplitude=0.4)
        label = classify_exercise(Z, self._w_hat(0.4))
        assert label == "jumping_jack"

    def test_pushup_detected(self) -> None:
        Z = _make_Z(T=300, z1_mean=0.0, z2_std=0.05, amplitude=0.3)
        label = classify_exercise(Z, self._w_hat(0.3))
        assert label == "pushup"

    def test_none_for_zero_amplitude(self) -> None:
        Z = _make_Z(T=300, z1_mean=0.0, z2_std=0.1, amplitude=0.1)
        label = classify_exercise(Z, np.zeros(3))
        assert label is None

    def test_none_for_too_few_components(self) -> None:
        Z = np.random.default_rng(0).standard_normal((100, 1))
        label = classify_exercise(Z, self._w_hat())
        assert label is None

    def test_returns_valid_label_or_none(self) -> None:
        from src.repetition import EXERCISES
        Z = _make_Z(T=200)
        label = classify_exercise(Z, self._w_hat())
        assert label in EXERCISES or label is None


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_output_types(self) -> None:
        T = 400
        Z = _make_Z(T=T, z1_mean=0.0, z2_std=0.2, amplitude=0.5)
        r, label, omega_hat = count_reps_and_classify(Z)
        assert isinstance(r, int)
        assert isinstance(omega_hat, float)
        assert label is None or isinstance(label, str)

    def test_rep_count_reasonable(self) -> None:
        T = 600
        true_reps = 8
        z1 = _make_sinusoid(T=T, true_reps=true_reps, amplitude=1.0)
        z2 = 0.05 * np.random.default_rng(1).standard_normal(T)
        Z = np.column_stack([z1, z2])
        r, _, _ = count_reps_and_classify(Z, min_reps=1, max_reps=20, n_steps=1000)
        assert r == true_reps

    def test_rejects_1d_input(self) -> None:
        with pytest.raises(ValueError):
            count_reps_and_classify(np.zeros(100))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
