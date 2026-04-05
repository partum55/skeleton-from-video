"""Shared linear algebra helpers (no project-internal dependencies)."""

import numpy as np
from numpy.typing import NDArray


def center_matrix(
    X: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Subtract column-wise mean: returns (X - mu, mu)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    mu = X.mean(axis=0)
    return X - mu, mu


def flatten_poses(
    skeletons: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reshape (T, 33, 2) -> (T, 66) for PCA input."""
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
    """Smallest k such that cumulative sigma_i^2 / total >= threshold."""
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
