"""
Custom SVD via Jacobi eigendecomposition of A^T A.

Algorithm (step by step)
------------------------
Given A ∈ R^{m×n} with m >= n:

Step 1  Form the symmetric PSD matrix  B = A^T A  ∈ R^{n×n}.
        Eigenvalues of B are σ_i^2; eigenvectors are the right singular vectors V.

Step 2  Jacobi cyclic sweeps on B.
        Each sweep visits every off-diagonal pair (i, j) with i < j and applies a
        Givens rotation G that zeroes out B[i, j]:

            B ← G^T B G,   V ← V G

        The Givens angle θ is chosen so that after the rotation B[i,j] = 0:

            τ = (B[j,j] − B[i,i]) / (2 B[i,j])
            t = sign(τ) / (|τ| + sqrt(1 + τ^2))   ← tan θ, always |t| ≤ 1
            c = 1 / sqrt(1 + t^2),   s = t c

        After enough sweeps, B converges to a diagonal matrix Λ = diag(λ_1, …, λ_n)
        and V is orthogonal with B_original = V Λ V^T.

Step 3  Singular values: σ_i = sqrt(max(λ_i, 0)).
        Sort in descending order; reorder columns of V accordingly.

Step 4  Left singular vectors: U_i = A v_i / σ_i  for σ_i > ε.
        For numerically zero singular values we fill U with an orthonormal basis
        (not needed for our use cases; included for interface completeness).

Result: A ≈ U Σ V^T   (exact up to floating-point).
"""

import numpy as np
from numpy.typing import NDArray


def _jacobi_sym_eig(
    B: NDArray[np.float64],
    tol: float = 1e-12,
    max_sweeps: int = 150,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Jacobi cyclic-sweep eigendecomposition of symmetric matrix B.

    Returns (eigenvalues, V) such that B = V diag(eigenvalues) V^T.
    Eigenvalues are NOT sorted here; sorting is done in custom_svd.
    """
    n = B.shape[0]
    A = B.astype(np.float64, copy=True)
    V = np.eye(n, dtype=np.float64)

    for _ in range(max_sweeps):
        # Off-diagonal Frobenius norm squared — convergence check.
        # We use the upper triangle only (matrix is symmetric).
        i_idx, j_idx = np.triu_indices(n, k=1)
        off_sq = float(np.sum(A[i_idx, j_idx] ** 2))
        if off_sq < tol ** 2:
            break

        # Cyclic sweep: one pass through all (i, j) pairs with i < j.
        for i in range(n - 1):
            for j in range(i + 1, n):
                a_ij = A[i, j]
                if abs(a_ij) < 1e-15:
                    continue  # already zero — skip rotation

                # --- Compute Givens angle ---
                # We want the rotation that zeroes A[i,j].
                # τ = cot(2θ) = (A[j,j] - A[i,i]) / (2 A[i,j])
                tau = (A[j, j] - A[i, i]) / (2.0 * a_ij)
                # t = tan(θ), choose smaller root for numerical stability
                t = np.sign(tau) / (abs(tau) + np.sqrt(1.0 + tau * tau))
                c = 1.0 / np.sqrt(1.0 + t * t)   # cos θ
                s = t * c                           # sin θ

                # --- Apply G^T A G in-place using numpy row/column ops ---
                # Update rows i and j of A first (left multiply by G^T)
                row_i = A[i, :].copy()
                row_j = A[j, :].copy()
                A[i, :] = c * row_i - s * row_j
                A[j, :] = s * row_i + c * row_j
                # Update columns i and j (right multiply by G)
                col_i = A[:, i].copy()
                col_j = A[:, j].copy()
                A[:, i] = c * col_i - s * col_j
                A[:, j] = s * col_i + c * col_j
                # Enforce exact zero and symmetry to prevent floating-point drift
                A[i, j] = 0.0
                A[j, i] = 0.0

                # --- Accumulate rotation in V: V <- V G ---
                v_i = V[:, i].copy()
                v_j = V[:, j].copy()
                V[:, i] = c * v_i - s * v_j
                V[:, j] = s * v_i + c * v_j

    return np.diag(A), V


def custom_svd(
    A: NDArray[np.float64],
    full_matrices: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Economy SVD of A via eigendecomposition of A^T A.

    Returns (U, s, Vt) satisfying  A ≈ U @ diag(s) @ Vt,
    with s sorted in descending order and columns of U and rows of Vt
    being the corresponding singular vectors.

    When m < n the computation is done on A^T and the result is transposed.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")

    m, n = A.shape

    # For wide matrices defer to the transpose path.
    if m < n:
        U, s, Vt = custom_svd(A.T, full_matrices=full_matrices)
        return Vt.T, s, U.T

    # -----------------------------------------------------------------
    # Step 1: form B = A^T A  (n×n, symmetric PSD)
    # -----------------------------------------------------------------
    B = A.T @ A

    # -----------------------------------------------------------------
    # Step 2: Jacobi eigendecomposition  B = V Λ V^T
    # -----------------------------------------------------------------
    eigenvalues, V = _jacobi_sym_eig(B)

    # -----------------------------------------------------------------
    # Step 3: singular values σ_i = sqrt(λ_i), sort descending
    # -----------------------------------------------------------------
    # Clamp tiny negatives caused by floating-point to zero.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    V = V[:, order]                   # right singular vectors, columns of V
    s = np.sqrt(eigenvalues)          # singular values

    # Economy: keep only r = min(m, n) = n components (already the case here).
    r = n

    # -----------------------------------------------------------------
    # Step 4: left singular vectors  U_i = A v_i / σ_i
    # -----------------------------------------------------------------
    AV = A @ V                        # m×r,  columns are A v_i
    U = np.empty((m, r), dtype=np.float64)

    zero_mask = s < 1e-12
    nonzero_mask = ~zero_mask

    # Nonzero singular values: straightforward normalisation.
    if np.any(nonzero_mask):
        U[:, nonzero_mask] = AV[:, nonzero_mask] / s[nonzero_mask]

    # Zero singular values: fill with arbitrary unit vectors orthogonal to the rest.
    # We use the null space of A^T (columns of U for zero σ come from the left null space).
    # Simple approach: use standard basis vectors not in the column span.
    if np.any(zero_mask):
        zero_cols = np.where(zero_mask)[0]
        # Gram-Schmidt against already-computed columns of U
        existing = U[:, nonzero_mask]
        e_idx = 0
        for col in zero_cols:
            # Find a standard basis vector e_k not already in the span.
            while e_idx < m:
                candidate = np.zeros(m, dtype=np.float64)
                candidate[e_idx] = 1.0
                e_idx += 1
                # Orthogonalise against existing U columns
                if existing.shape[1] > 0:
                    candidate -= existing @ (existing.T @ candidate)
                n_cand = np.linalg.norm(candidate)
                if n_cand > 1e-10:
                    U[:, col] = candidate / n_cand
                    existing = np.column_stack([existing, U[:, col]])
                    break
            else:
                U[:, col] = 0.0

    Vt = V.T                          # right singular vectors as rows
    return U, s, Vt
