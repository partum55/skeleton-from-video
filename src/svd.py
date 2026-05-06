"""
Custom SVD via power iteration with deflation.

Algorithm
---------
Given A in R^{m x n} with m >= n we want A = U Sigma V^T.

Step 1.  Form B = A^T A in R^{n x n}.  B is symmetric and positive
         semi-definite.  Its eigenvalues are sigma_i^2 and its eigenvectors
         are the right singular vectors v_i of A.

Step 2.  Power iteration finds the dominant eigenpair (lambda_1, v_1) of B:

             v_{k+1} = B v_k / || B v_k ||.

         Decomposing v_0 in the eigenbasis of B,
         v_0 = sum_i alpha_i q_i, gives

             B^k v_0 = sum_i alpha_i lambda_i^k q_i,

         and after normalisation the term with the largest |lambda_i|
         dominates.  The error decays geometrically with ratio
         (lambda_2 / lambda_1)^k.  We monitor the Rayleigh quotient
         v^T B v as the eigenvalue estimate and stop when it stops moving.

Step 3.  Deflate B by subtracting the rank-one piece we just identified:

             B <- B - lambda_1 v_1 v_1^T.

         The eigenvalue at v_1 becomes 0; all other eigenpairs are unchanged.
         Power iteration on the deflated matrix produces (lambda_2, v_2),
         and so on.  Re-orthogonalisation against previously found vectors
         keeps numerical drift out of the already-computed subspace.

Step 4.  Recover singular values and left singular vectors:

             sigma_i = sqrt(lambda_i),     u_i = A v_i / sigma_i.

         For numerically zero singular values u_i is filled by Gram-Schmidt
         against a standard basis, so U remains orthonormal.

For wide matrices (m < n) the same algorithm runs on A^T and the result is
transposed.
"""

import numpy as np
from numpy.typing import NDArray


_EIG_TOL: float = 1e-12
_EIG_MAX_ITERS: int = 2000
_DEFAULT_SEED: int = 0
_ZERO_THRESHOLD: float = 1e-12


def _random_unit_orthogonal(
    columns: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw a random unit vector orthogonal to the columns of `columns`."""
    n = columns.shape[0]
    for _ in range(50):
        v = rng.standard_normal(n)
        if columns.shape[1] > 0:
            v -= columns @ (columns.T @ v)
        norm = float(np.sqrt(v @ v))
        if norm > 1e-10:
            return v / norm
    raise RuntimeError("failed to draw a vector orthogonal to existing columns")


def _orthonormalise(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Modified Gram-Schmidt orthonormalisation of the columns of `matrix`."""
    out = matrix.astype(np.float64, copy=True)
    n_cols = out.shape[1]
    for j in range(n_cols):
        v = out[:, j]
        for i in range(j):
            v -= float(out[:, i] @ v) * out[:, i]
        norm = float(np.sqrt(v @ v))
        if norm > 1e-12:
            out[:, j] = v / norm
        else:
            out[:, j] = 0.0
    return out


def _power_iteration_eigenpair(
    M: NDArray[np.float64],
    deflation_basis: NDArray[np.float64],
    rng: np.random.Generator,
    tol: float,
    max_iters: int,
) -> tuple[float, NDArray[np.float64]]:
    """Dominant eigenpair of symmetric M via power iteration.

    `deflation_basis` holds eigenvectors already extracted; the iterate is
    re-orthogonalised against them at every step to prevent drift back into
    the already-computed subspace.
    """
    v = _random_unit_orthogonal(deflation_basis, rng)
    eigenvalue = 0.0
    for _ in range(max_iters):
        Mv = M @ v
        if deflation_basis.shape[1] > 0:
            Mv -= deflation_basis @ (deflation_basis.T @ Mv)
        norm_Mv = float(np.sqrt(Mv @ Mv))
        if norm_Mv < tol:
            return 0.0, v
        v_next = Mv / norm_Mv
        eigenvalue_next = float(v_next @ (M @ v_next))
        residual = Mv - eigenvalue_next * v
        residual_norm = float(np.sqrt(residual @ residual))
        scale = max(abs(eigenvalue_next), 1.0)
        if residual_norm <= tol * scale:
            return eigenvalue_next, v_next
        v, eigenvalue = v_next, eigenvalue_next
    return eigenvalue, v


def symmetric_eig_power(
    B: NDArray[np.float64],
    seed: int | None = _DEFAULT_SEED,
    tol: float = _EIG_TOL,
    max_iters: int = _EIG_MAX_ITERS,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigendecomposition of symmetric B by power iteration + deflation.

    Returns (eigenvalues, V) such that B ≈ V diag(eigenvalues) V^T,
    with eigenvalues in descending order of magnitude and V orthonormal.
    """
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"B must be square, got shape {B.shape}")

    n = B.shape[0]
    M = B.copy()
    rng = np.random.default_rng(seed)
    eigenvalues = np.zeros(n, dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)

    for k in range(n):
        lam, v = _power_iteration_eigenpair(
            M, V[:, :k], rng, tol=tol, max_iters=max_iters,
        )
        eigenvalues[k] = lam
        V[:, k] = v
        M -= lam * np.outer(v, v)
        M = 0.5 * (M + M.T)

    return eigenvalues, V


def custom_svd(
    A: NDArray[np.float64],
    full_matrices: bool = False,
    seed: int | None = _DEFAULT_SEED,
    tol: float = _EIG_TOL,
    max_iters: int = _EIG_MAX_ITERS,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Economy SVD of A via power iteration on A^T A.

    Returns (U, s, Vt) with A ≈ U @ diag(s) @ Vt, s sorted in descending
    order, columns of U and rows of Vt being the corresponding singular
    vectors.  For wide matrices (m < n) the computation is done on A^T.
    """
    del full_matrices  # economy decomposition only; flag kept for API parity

    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")

    m, n = A.shape

    if m < n:
        U, s, Vt = custom_svd(A.T, seed=seed, tol=tol, max_iters=max_iters)
        return Vt.T, s, U.T

    rng = np.random.default_rng(seed)

    B = A.T @ A
    eigenvalues, V = symmetric_eig_power(B, seed=seed, tol=tol, max_iters=max_iters)

    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    V = V[:, order]
    s = np.sqrt(eigenvalues)

    AV = A @ V
    U = np.empty((m, n), dtype=np.float64)
    nonzero = s > _ZERO_THRESHOLD

    if np.any(nonzero):
        U[:, nonzero] = _orthonormalise(AV[:, nonzero])

    if not np.all(nonzero):
        existing = U[:, nonzero] if np.any(nonzero) else np.zeros((m, 0))
        for col in np.where(~nonzero)[0]:
            v = _random_unit_orthogonal(existing, rng)
            U[:, col] = v
            existing = np.column_stack([existing, v])

    return U, s, V.T
