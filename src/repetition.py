"""
Repetition counting via least-squares sinusoidal fitting + exercise classification.

Developer 3 module.

Input:  z_1(t) ∈ R^T  — first PCA component time-series (from Developer 2)
        Z ∈ R^{T×k}   — full PCA projection (for classification)
Output: r  — repetition count
        ĉ  — exercise label  ('squat' | 'pushup' | 'jumping_jack' | None)

Algorithm
---------
For each candidate frequency ω:

1. Build design matrix A(ω) ∈ R^{T×3}:

       A(ω) = [ sin(ω·1)  cos(ω·1)  1 ]
               [    ⋮          ⋮      ⋮ ]
               [ sin(ω·T)  cos(ω·T)  1 ]

2. Solve the normal equations (least-squares):

       AᵀA ŵ = Aᵀz₁   →   ŵ = (AᵀA)⁻¹ Aᵀz₁

   via numpy.linalg.lstsq  (numerically stable, handles rank-deficiency).

3. Record residual ‖A(ω)ŵ − z₁‖₂.

4. Grid search: ω̂ = argmin_ω  residual(ω).

5. Repetition count: r = round(ω̂ · T / (2π)).

Classification
--------------
Uses the first two PCA components Z[:,0] and Z[:,1]:
- The dominant component z₁(t) captures the primary motion amplitude.
- The second component z₂(t) captures secondary body-part movement.
Simple linear thresholds distinguish exercise types.
"""

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Exercise labels
# ---------------------------------------------------------------------------

EXERCISES: tuple[str, ...] = ("squat", "pushup", "jumping_jack")
EXERCISE_NONE: None = None


# ---------------------------------------------------------------------------
# Step 1 — Design matrix
# ---------------------------------------------------------------------------

def build_design_matrix(omega: float, T: int) -> NDArray[np.float64]:
    """Build the sinusoidal design matrix A(ω) ∈ R^{T×3}.

    Each row t corresponds to one frame:

        A[t] = [ sin(ω · t),  cos(ω · t),  1 ]

    The constant column allows the model to fit a non-zero mean.
    The model is then:  z₁(t) ≈ w₁ sin(ωt) + w₂ cos(ωt) + w₃.

    Parameters
    ----------
    omega:
        Candidate angular frequency in radians per frame.
    T:
        Number of frames (rows of the design matrix).

    Returns
    -------
    A:
        Design matrix of shape (T, 3).

    Raises
    ------
    ValueError
        If T < 1.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    t = np.arange(1, T + 1, dtype=np.float64)
    return np.column_stack([np.sin(omega * t), np.cos(omega * t), np.ones(T)])


# ---------------------------------------------------------------------------
# Step 2 — Least-squares fit for one frequency
# ---------------------------------------------------------------------------

def fit_frequency(
    omega: float,
    z1: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Fit a sinusoidal model at frequency ω to signal z₁ via least-squares.

    Solves the normal equations:
        AᵀA ŵ = Aᵀz₁

    using numpy.linalg.lstsq for numerical stability.

    Parameters
    ----------
    omega:
        Candidate angular frequency in radians per frame.
    z1:
        Time-series vector of shape (T,).

    Returns
    -------
    w_hat:
        Coefficient vector of shape (3,): [w₁, w₂, w₃].
    residual:
        ‖A(ω) ŵ − z₁‖₂  (scalar, lower = better fit).
    """
    T = len(z1)
    A = build_design_matrix(omega, T)
    w_hat, _, _, _ = np.linalg.lstsq(A, z1, rcond=None)
    residual = float(np.linalg.norm(A @ w_hat - z1))
    return w_hat, residual


# ---------------------------------------------------------------------------
# Step 3-4 — Grid search over candidate frequencies
# ---------------------------------------------------------------------------

def build_frequency_grid(
    T: int,
    min_reps: int = 1,
    max_reps: int = 30,
    n_steps: int = 500,
) -> NDArray[np.float64]:
    """Build a grid of candidate angular frequencies.

    Frequencies correspond to repetition counts in [min_reps, max_reps]
    over a sequence of T frames:

        ω = 2π · r / T   for r ∈ {min_reps, …, max_reps}

    A finer grid (n_steps) is used for sub-integer resolution.

    Parameters
    ----------
    T:
        Number of frames.
    min_reps:
        Minimum expected repetition count.
    max_reps:
        Maximum expected repetition count.
    n_steps:
        Number of frequency candidates to evaluate.

    Returns
    -------
    omegas:
        1-D array of candidate angular frequencies.
    """
    omega_min = 2.0 * np.pi * min_reps / T
    omega_max = 2.0 * np.pi * max_reps / T
    return np.linspace(omega_min, omega_max, n_steps)


def find_best_frequency(
    z1: NDArray[np.float64],
    min_reps: int = 1,
    max_reps: int = 30,
    n_steps: int = 500,
) -> tuple[float, NDArray[np.float64], float]:
    """Grid-search for the angular frequency that best fits z₁.

    Evaluates fit_frequency() at each candidate ω and selects the one
    with the lowest residual.

    Parameters
    ----------
    z1:
        Time-series of shape (T,).
    min_reps:
        Lower bound on the repetition count search range.
    max_reps:
        Upper bound on the repetition count search range.
    n_steps:
        Number of candidate frequencies to try.

    Returns
    -------
    omega_hat:
        Best-fit angular frequency (radians per frame).
    w_hat:
        Coefficient vector [w₁, w₂, w₃] at omega_hat.
    min_residual:
        Residual ‖Aŵ − z₁‖₂ at omega_hat.

    Raises
    ------
    ValueError
        If z1 has fewer than 10 samples (too short to fit reliably).
    """
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


# ---------------------------------------------------------------------------
# Step 5 — Repetition count from best frequency
# ---------------------------------------------------------------------------

def count_repetitions(omega_hat: float, T: int) -> int:
    """Convert the best-fit angular frequency to a repetition count.

        r = round(ω̂ · T / (2π))

    Parameters
    ----------
    omega_hat:
        Best-fit angular frequency in radians per frame.
    T:
        Total number of frames.

    Returns
    -------
    r:
        Repetition count (non-negative integer).
    """
    r = round(omega_hat * T / (2.0 * np.pi))
    return max(0, int(r))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_exercise(
    Z: NDArray[np.float64],
    w_hat: NDArray[np.float64],
) -> str | None:
    """Classify the exercise type from the PCA projection matrix Z.

    Uses the first two PCA components and the fitted sinusoidal amplitude:

    - Amplitude A = sqrt(w₁² + w₂²) from the sinusoidal fit captures how
      large the dominant oscillation is.
    - z₂_std = std(Z[:,1]) measures secondary body-part activity.
    - z₁_mean = mean(Z[:,0]) reflects average posture (e.g. crouched vs upright).

    Threshold logic (tuned empirically on the three exercise types):
        squat:        large z₁_mean offset  (hips are lower than reference)
        pushup:       small z₂_std          (legs are static, only arms move)
        jumping_jack: large z₂_std          (arms and legs move together)

    Parameters
    ----------
    Z:
        PCA projection matrix of shape (T, k), k ≥ 2.
    w_hat:
        Sinusoidal coefficients [w₁, w₂, w₃] from fit_frequency().

    Returns
    -------
    label:
        One of 'squat', 'pushup', 'jumping_jack', or None (unrecognised).
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


# ---------------------------------------------------------------------------
# High-level pipeline function
# ---------------------------------------------------------------------------

def count_reps_and_classify(
    Z: NDArray[np.float64],
    min_reps: int = 1,
    max_reps: int = 30,
    n_steps: int = 500,
) -> tuple[int, str | None, float]:
    """Full Developer 3 pipeline: frequency fit → rep count → classification.

    Parameters
    ----------
    Z:
        PCA projection matrix of shape (T, k), k ≥ 1.
        z₁(t) = Z[:, 0] is used for frequency fitting.
    min_reps:
        Lower bound for frequency grid search.
    max_reps:
        Upper bound for frequency grid search.
    n_steps:
        Grid resolution.

    Returns
    -------
    r:
        Repetition count.
    label:
        Exercise label or None.
    omega_hat:
        Best-fit angular frequency (useful for debugging).
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D, got shape {Z.shape}")

    z1 = Z[:, 0]
    omega_hat, w_hat, _ = find_best_frequency(z1, min_reps, max_reps, n_steps)
    r = count_repetitions(omega_hat, len(z1))
    label = classify_exercise(Z, w_hat) if Z.shape[1] >= 2 else None

    return r, label, omega_hat
