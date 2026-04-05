"""
PCA via manual SVD (numpy.linalg.svd, no sklearn).

Centre X, do economy SVD, keep k components that retain >= 95% of variance,
project z(t) = V_k^T (s(t) - mu).
"""

import numpy as np
from numpy.typing import NDArray
from src.linalg_utils import center_matrix, select_n_components


# Default fraction of variance to retain when selecting k.
DEFAULT_VARIANCE_THRESHOLD: float = 0.95


def align_components_to_reference(
    components: NDArray[np.float64],
    reference_components: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Match order and sign of PCA components to a reference basis.
    Avoids sign/order flips when PCA is refit on a sliding window.
    """
    comps = np.asarray(components, dtype=np.float64)
    if reference_components is None:
        return comps

    ref = np.asarray(reference_components, dtype=np.float64)
    if comps.ndim != 2 or ref.ndim != 2 or comps.shape[0] != ref.shape[0]:
        return comps

    k_new = comps.shape[1]
    k_ref = ref.shape[1]
    if k_new == 0 or k_ref == 0:
        return comps

    sim = np.abs(ref.T @ comps)  # (k_ref, k_new)
    aligned = np.zeros_like(comps)

    used_ref: set[int] = set()
    used_new: set[int] = set()
    slots = min(k_new, k_ref)

    for out_idx in range(slots):
        best = None
        best_val = -np.inf
        for ref_idx in range(k_ref):
            if ref_idx in used_ref:
                continue
            for new_idx in range(k_new):
                if new_idx in used_new:
                    continue
                val = sim[ref_idx, new_idx]
                if val > best_val:
                    best_val = val
                    best = (ref_idx, new_idx)
        if best is None:
            break
        ref_idx, new_idx = best
        vec = comps[:, new_idx].copy()
        if float(np.dot(ref[:, ref_idx], vec)) < 0.0:
            vec *= -1.0
        aligned[:, out_idx] = vec
        used_ref.add(ref_idx)
        used_new.add(new_idx)

    out_idx = slots
    for new_idx in range(k_new):
        if new_idx in used_new:
            continue
        aligned[:, out_idx] = comps[:, new_idx]
        out_idx += 1

    return aligned


class PCA:
    """PCA via manual SVD — fit / transform / inverse_transform interface."""

    def __init__(self, variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD) -> None:
        if not (0.0 < variance_threshold <= 1.0):
            raise ValueError(
                f"variance_threshold must be in (0, 1], got {variance_threshold}"
            )
        self.variance_threshold = variance_threshold

        self.mean_: NDArray[np.float64] | None = None
        self.components_: NDArray[np.float64] | None = None
        self.singular_values_: NDArray[np.float64] | None = None
        self.n_components_: int | None = None
        self.variance_ratio_: NDArray[np.float64] | None = None

    def fit(
        self,
        X: NDArray[np.float64],
        reference_components: NDArray[np.float64] | None = None,
    ) -> "PCA":
        """Centre X, compute SVD, keep k components that meet variance threshold."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        X_centered, mu = center_matrix(X)

        # economy SVD: only compute min(T, d) singular vectors
        _, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        k = select_n_components(s, self.variance_threshold)

        self.mean_ = mu
        self.components_ = align_components_to_reference(Vt[:k].T, reference_components)
        self.singular_values_ = s[:k]        # shape (k,)
        self.n_components_ = k
        self.variance_ratio_ = (s[:k] ** 2) / (s**2).sum()  # shape (k,)

        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project X onto the top-k principal directions: Z = (X - mu) V_k."""
        self._assert_fitted()
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_

    def fit_transform(
        self,
        X: NDArray[np.float64],
        reference_components: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """fit() then transform() in one call."""
        X = np.asarray(X, dtype=np.float64)
        self.fit(X, reference_components=reference_components)
        return (X - self.mean_) @ self.components_

    def inverse_transform(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct X_hat = Z V_k^T + mu."""
        self._assert_fitted()
        Z = np.asarray(Z, dtype=np.float64)
        return Z @ self.components_.T + self.mean_

    @property
    def cumulative_variance_ratio(self) -> NDArray[np.float64]:
        """Cumulative explained variance ratio across the k kept components."""
        self._assert_fitted()
        return np.cumsum(self.variance_ratio_)

    def _assert_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError("PCA is not fitted yet.  Call fit() first.")


def fit_pca(
    X: NDArray[np.float64],
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> PCA:
    """Create and fit a PCA object in one call."""
    return PCA(variance_threshold=variance_threshold).fit(X)


def project_poses(
    skeletons_flat: NDArray[np.float64],
    pca: PCA,
) -> NDArray[np.float64]:
    """Project flattened pose vectors (T, 66) through a fitted PCA."""
    return pca.transform(skeletons_flat)
