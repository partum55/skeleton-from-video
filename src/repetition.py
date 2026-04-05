"""
Repetition counting via least-squares sinusoidal fitting + exercise classification.

Fit z_1(t) ~ w1*sin(wt) + w2*cos(wt) + w3 by solving normal equations,
grid-search over candidate frequencies, pick omega with lowest residual,
r = round(omega_hat * T / 2pi).
"""

import numpy as np
from numpy.typing import NDArray


EXERCISES: tuple[str, ...] = ("squat", "pushup", "jumping_jack")
EXERCISE_NONE: None = None


def build_design_matrix(omega: float, T: int) -> NDArray[np.float64]:
    """A(omega) in R^{T x 3}: columns are [sin(wt), cos(wt), 1]."""
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    t = np.arange(1, T + 1, dtype=np.float64)
    return np.column_stack([np.sin(omega * t), np.cos(omega * t), np.ones(T)])


def fit_frequency(
    omega: float,
    z1: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Solve A^T A w = A^T z1 via lstsq for a single candidate frequency."""
    T = len(z1)
    A = build_design_matrix(omega, T)
    w_hat, _, _, _ = np.linalg.lstsq(A, z1, rcond=None)
    residual = float(np.linalg.norm(A @ w_hat - z1))
    return w_hat, residual


def build_frequency_grid(
    T: int,
    min_reps: int = 1,
    max_reps: int = 30,
    n_steps: int = 500,
) -> NDArray[np.float64]:
    """Linspace of candidate omegas from 2pi*min_reps/T to 2pi*max_reps/T."""
    omega_min = 2.0 * np.pi * min_reps / T
    omega_max = 2.0 * np.pi * max_reps / T
    return np.linspace(omega_min, omega_max, n_steps)


def find_best_frequency(
    z1: NDArray[np.float64],
    min_reps: int = 1,
    max_reps: int = 30,
    n_steps: int = 500,
) -> tuple[float, NDArray[np.float64], float]:
    """Try every candidate omega and return the one with the lowest residual."""
    z1 = np.asarray(z1, dtype=np.float64)
    if z1.ndim != 1:
        raise ValueError(f"z1 must be 1-D, got shape {z1.shape}")
    T = len(z1)
    if T < 10:
        raise ValueError(f"z1 must have at least 10 samples, got {T}")

    omegas = build_frequency_grid(T, min_reps, max_reps, n_steps)

    best_omega = omegas[0]
    best_w = np.zeros(3, dtype=np.float64)
    best_residual = np.inf

    for omega in omegas:
        w_hat, residual = fit_frequency(omega, z1)
        if residual < best_residual:
            best_residual = residual
            best_omega = omega
            best_w = w_hat

    return best_omega, best_w, best_residual


def count_repetitions(omega_hat: float, T: int) -> int:
    """r = round(omega_hat * T / 2pi), clamped to >= 0."""
    r = round(omega_hat * T / (2.0 * np.pi))
    return max(0, int(r))


def classify_exercise(
    Z: NDArray[np.float64],
    w_hat: NDArray[np.float64],
) -> str | None:
    """Classify exercise using thresholds on Z[:,0] mean, Z[:,1] std, and
    the sinusoidal amplitude sqrt(w1^2 + w2^2).
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[1] < 2:
        return None

    z1_mean = float(np.mean(Z[:, 0]))
    z2_std = float(np.std(Z[:, 1]))
    amplitude = float(np.sqrt(w_hat[0] ** 2 + w_hat[1] ** 2))

    # No meaningful motion detected
    if amplitude < 0.01:
        return None

    # Squat: large mean offset on z₁ (body is crouched relative to reference)
    if abs(z1_mean) > 0.3:
        return "squat"

    # Jumping jack: strong secondary component (arms + legs synchronised)
    if z2_std > 0.15:
        return "jumping_jack"

    # Pushup: moderate primary motion, weak secondary (body is horizontal)
    if amplitude > 0.05:
        return "pushup"

    return None


def count_reps_and_classify(
    Z: NDArray[np.float64],
    min_reps: int = 1,
    max_reps: int = 30,
    n_steps: int = 500,
) -> tuple[int, str | None, float]:
    """Full pipeline: frequency fit -> rep count -> classification."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D, got shape {Z.shape}")

    z1 = Z[:, 0]
    omega_hat, w_hat, _ = find_best_frequency(z1, min_reps, max_reps, n_steps)
    r = count_repetitions(omega_hat, len(z1))
    label = classify_exercise(Z, w_hat) if Z.shape[1] >= 2 else None

    return r, label, omega_hat


def estimate_pca_confidence(
    z1: NDArray[np.float64],
    w_hat: NDArray[np.float64],
    residual: float,
) -> float:
    """Estimate confidence of PCA-based periodic motion fit in [0, 1]."""
    signal = np.asarray(z1, dtype=np.float64)
    amplitude = float(np.sqrt(w_hat[0] ** 2 + w_hat[1] ** 2))
    signal_std = float(np.std(signal))
    if signal_std < 1e-8:
        return 0.0

    quality = 1.0 - float(residual) / (signal_std * np.sqrt(len(signal)) + 1e-8)
    amp_ratio = amplitude / (signal_std + 1e-8)
    conf = 0.6 * np.clip(quality, 0.0, 1.0) + 0.4 * np.clip(amp_ratio / 2.0, 0.0, 1.0)
    return float(np.clip(conf, 0.0, 1.0))


def count_reps_and_classify_with_confidence(
    Z: NDArray[np.float64],
    min_reps: int = 1,
    max_reps: int = 30,
    n_steps: int = 500,
) -> tuple[int, str | None, float, float]:
    """Extended PCA pipeline returning (reps, label, omega, confidence)."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D, got shape {Z.shape}")

    z1 = Z[:, 0]
    omega_hat, w_hat, residual = find_best_frequency(z1, min_reps, max_reps, n_steps)
    reps = count_repetitions(omega_hat, len(z1))
    label = classify_exercise(Z, w_hat) if Z.shape[1] >= 2 else None
    confidence = estimate_pca_confidence(z1, w_hat, residual)
    return reps, label, omega_hat, confidence


def fuse_exercise_labels(
    angle_label: str | None,
    pca_label: str | None,
    pca_confidence: float,
    min_pca_confidence: float = 0.65,
) -> str | None:
    """Fuse angle- and PCA-based labels with confidence gating."""
    if angle_label is None and pca_confidence >= min_pca_confidence:
        return pca_label
    if pca_label is None:
        return angle_label
    if angle_label == pca_label:
        return angle_label
    if pca_confidence >= 0.9:
        return pca_label
    return angle_label
