"""Shared linear algebra helpers, hand-rolled (no numpy.linalg).

Implements from scratch: vector L2 norm, 2x2 determinant, Gaussian
elimination with partial pivoting, and symmetric eigenvalue computation
(via power iteration with deflation, lazily importing src.svd to avoid a
cycle).
"""

import numpy as np
from numpy.typing import NDArray


def vector_norm(x: NDArray[np.float64]) -> float:
    """Euclidean norm: sqrt(sum_i x_i^2). Replaces np.linalg.norm for vectors."""
    x = np.asarray(x, dtype=np.float64).ravel()
    s = 0.0
    for xi in x:
        s += float(xi) * float(xi)
    return float(np.sqrt(s))


def frobenius_norm(M: NDArray[np.float64]) -> float:
    """Frobenius norm: sqrt(sum_ij M_ij^2). Replaces np.linalg.norm(M, 'fro')."""
    M = np.asarray(M, dtype=np.float64)
    s = 0.0
    for v in M.ravel():
        s += float(v) * float(v)
    return float(np.sqrt(s))


def det_2x2(M: NDArray[np.float64]) -> float:
    """Determinant of a 2x2 matrix: a*d - b*c. Replaces np.linalg.det for 2x2."""
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (2, 2):
        raise ValueError(f"det_2x2 expects 2x2 matrix, got {M.shape}")
    return float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])


def solve_linear_system(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve A x = b via Gaussian elimination with partial pivoting.

    Replaces np.linalg.solve. A must be square and non-singular.
    Cost ~ (2/3) n^3 flops; we only call it on small systems (3x3).
    """
    A = np.asarray(A, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square, got {A.shape}")
    if b.shape[0] != n:
        raise ValueError(f"shape mismatch: A is {A.shape}, b is {b.shape}")

    # Build augmented matrix [A | b]
    M = np.zeros((n, n + 1), dtype=np.float64)
    M[:, :n] = A
    M[:, n] = b

    # Forward elimination with partial pivoting
    for k in range(n):
        # Pivot: row with largest |M[i, k]| for i >= k
        pivot = k
        max_val = abs(M[k, k])
        for i in range(k + 1, n):
            if abs(M[i, k]) > max_val:
                max_val = abs(M[i, k])
                pivot = i
        if max_val < 1e-15:
            raise ValueError("singular matrix in solve_linear_system")
        if pivot != k:
            tmp = M[k].copy()
            M[k] = M[pivot]
            M[pivot] = tmp

        # Eliminate column k below the pivot
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            for j in range(k, n + 1):
                M[i, j] -= factor * M[k, j]

    # Back substitution
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = M[i, n]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
    return x


def eigvals_symmetric(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Eigenvalues of a symmetric matrix via power iteration, sorted ascending.

    Replaces np.linalg.eigvalsh. Imports src.svd lazily to avoid circular import.
    """
    from src.svd import symmetric_eig_power
    M = np.asarray(M, dtype=np.float64)
    eigenvalues, _ = symmetric_eig_power(M)
    return np.sort(eigenvalues)


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
