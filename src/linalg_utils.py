"""
Shared linear algebra utilities used across the pipeline.

Functions here are intentionally low-level and have no dependencies on other
project modules, so any teammate can import them without circular imports.
"""

import numpy as np
from numpy.typing import NDArray


def center_matrix(
    X: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Subtract the column-wise mean from a data matrix.

    Computes µ = (1/T) Σ s(t) ∈ R^d and returns X̃ = X − 1µᵀ.

    Used by:
        - pca.py  (PCA centering step before SVD)
        - repetition.py  (centering the PCA time-signal before fitting)

    Parameters
    ----------
    X:
        Data matrix of shape (T, d).  Each row is one observation.

    Returns
    -------
    X_centered:
        Zero-mean matrix of shape (T, d).
    mu:
        Column mean vector of shape (d,).

    Raises
    ------
    ValueError
        If X is not a 2-D array.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    mu = X.mean(axis=0)
    return X - mu, mu


def flatten_poses(
    skeletons: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Flatten an array of skeleton matrices into row vectors.

    Converts S_hat ∈ R^{T×33×2} produced by Developer 1 into the pose
    matrix X ∈ R^{T×66} consumed by Developer 2.

    Note: for a *single* skeleton use ``features.flatten_skeleton`` directly.

    Parameters
    ----------
    skeletons:
        Array of shape (T, 33, 2) — one normalised skeleton per frame.

    Returns
    -------
    X:
        Matrix of shape (T, 66) where each row is a flattened pose vector
        s(t) = [x_0, y_0, x_1, y_1, …, x_32, y_32].

    Raises
    ------
    ValueError
        If ``skeletons`` does not have shape (T, 33, 2).
    """
    skeletons = np.asarray(skeletons, dtype=np.float64)
    if skeletons.ndim != 3 or skeletons.shape[1:] != (33, 2):
        raise ValueError(
            f"skeletons must have shape (T, 33, 2), got {skeletons.shape}"
        )
    T = skeletons.shape[0]
    return skeletons.reshape(T, -1)


def select_n_components(
    singular_values: NDArray[np.float64],
    variance_threshold: float = 0.95,
) -> int:
    """Return the minimum k that retains *variance_threshold* of total variance.

    The variance explained by component i is proportional to σᵢ², so:

        k = min{ k : (Σᵢ₌₁ᵏ σᵢ²) / (Σᵢ σᵢ²) ≥ threshold }

    Used by:
        - pca.py  (choosing how many principal components to keep)

    Parameters
    ----------
    singular_values:
        1-D array of non-negative singular values in *descending* order.
    variance_threshold:
        Fraction of variance to retain, in (0, 1].  Default is 0.95.

    Returns
    -------
    k:
        Number of components to keep.  Always ≥ 1 and ≤ len(singular_values).

    Raises
    ------
    ValueError
        If singular_values is not 1-D or variance_threshold is not in (0, 1].
    """
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.ndim != 1:
        raise ValueError("singular_values must be 1-D")
    if not (0.0 < variance_threshold <= 1.0):
        raise ValueError(
            f"variance_threshold must be in (0, 1], got {variance_threshold}"
        )

    variances = singular_values**2
    total_variance = variances.sum()

    if total_variance == 0.0:
        return 1

    cumulative_ratio = np.cumsum(variances) / total_variance
    # searchsorted finds the first index where cumulative_ratio >= threshold
    k = int(np.searchsorted(cumulative_ratio, variance_threshold)) + 1
    return min(k, len(singular_values))
