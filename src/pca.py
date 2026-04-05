"""
PCA feature extraction via manual SVD.

Developer 2 module.

Input:  array of pose vectors s(t) ∈ R^66  (flattened, normalised skeletons)
Output: projection matrix Z ∈ R^{T×k}, mean vector µ ∈ R^66, basis V_k ∈ R^{66×k}

Algorithm
---------
1. Build data matrix X ∈ R^{T×66} from T flattened pose vectors.
2. Centre:  X̃ = X − 1µᵀ   where µ = (1/T) Σ s(t)
3. Economy SVD:  X̃ = U Σ Vᵀ   via numpy.linalg.svd
4. Select k columns of V that retain ≥ 95 % of variance  (σᵢ² / Σ σᵢ²).
5. Project:  z(t) = V_kᵀ (s(t) − µ)   →   Z = X̃ V_k ∈ R^{T×k}

The class follows a fit / transform / inverse_transform interface so
Developer 3 can call  pca.transform(new_frame)  on a single new pose at
inference time without re-fitting.
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
    """Align PCA components to a reference basis (order + sign).

    This reduces instability when PCA is re-fitted on a sliding window.
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
    """Principal Component Analysis via manual SVD.

    Attributes set after ``fit``
    ----------------------------
    mean_ : NDArray[np.float64], shape (d,)
        Column mean of the training data.
    components_ : NDArray[np.float64], shape (d, k)
        Top-k principal directions (columns of V).  Orthonormal: V_kᵀ V_k = I.
    singular_values_ : NDArray[np.float64], shape (k,)
        Singular values of X̃ corresponding to the kept components.
    n_components_ : int
        Number of components selected.
    variance_ratio_ : NDArray[np.float64], shape (k,)
        Fraction of total variance explained by each kept component.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 66))
    >>> pca = PCA().fit(X)
    >>> Z = pca.transform(X)          # shape (200, k)
    >>> X_hat = pca.inverse_transform(Z)  # approximate reconstruction
    """

    def __init__(self, variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD) -> None:
        """
        Parameters
        ----------
        variance_threshold:
            Fraction of variance to retain when selecting k.  Default 0.95.
        """
        if not (0.0 < variance_threshold <= 1.0):
            raise ValueError(
                f"variance_threshold must be in (0, 1], got {variance_threshold}"
            )
        self.variance_threshold = variance_threshold

        # set after fit()
        self.mean_: NDArray[np.float64] | None = None
        self.components_: NDArray[np.float64] | None = None
        self.singular_values_: NDArray[np.float64] | None = None
        self.n_components_: int | None = None
        self.variance_ratio_: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray[np.float64],
        reference_components: NDArray[np.float64] | None = None,
    ) -> "PCA":
        """Fit PCA on a data matrix X.

        Parameters
        ----------
        X:
            Data matrix of shape (T, d).  Rows are observations (frames),
            columns are features (pose coordinates).

        Returns
        -------
        self
            The fitted PCA object (enables method chaining).

        Raises
        ------
        ValueError
            If X is not 2-D or has fewer rows than columns.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        X_centered, mu = center_matrix(X)

        # Economy SVD: U (T×r), s (r,), Vt (r×d)  where r = min(T, d).
        # We do NOT use full_matrices=True to avoid allocating a T×T matrix.
        _, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        k = select_n_components(s, self.variance_threshold)

        self.mean_ = mu
        self.components_ = align_components_to_reference(Vt[:k].T, reference_components)
        self.singular_values_ = s[:k]        # shape (k,)
        self.n_components_ = k
        self.variance_ratio_ = (s[:k] ** 2) / (s**2).sum()  # shape (k,)

        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project X onto the principal subspace.

        z(t) = V_kᵀ (s(t) − µ)

        Parameters
        ----------
        X:
            Array of shape (T, d) or a single vector of shape (d,).

        Returns
        -------
        Z:
            Array of shape (T, k) or (k,).
        """
        self._assert_fitted()
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_

    def fit_transform(
        self,
        X: NDArray[np.float64],
        reference_components: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Fit and project in one call.

        Equivalent to ``self.fit(X).transform(X)`` but avoids centering twice.

        Parameters
        ----------
        X:
            Data matrix of shape (T, d).

        Returns
        -------
        Z:
            Projection matrix of shape (T, k).
        """
        X = np.asarray(X, dtype=np.float64)
        self.fit(X, reference_components=reference_components)
        # reuse the already-centred matrix implicitly through transform
        return (X - self.mean_) @ self.components_

    def inverse_transform(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct approximate original data from projections.

        X̂ = Z V_kᵀ + µ

        Parameters
        ----------
        Z:
            Projection array of shape (T, k) or (k,).

        Returns
        -------
        X_reconstructed:
            Array of shape (T, d) or (d,).
        """
        self._assert_fitted()
        Z = np.asarray(Z, dtype=np.float64)
        return Z @ self.components_.T + self.mean_

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cumulative_variance_ratio(self) -> NDArray[np.float64]:
        """Cumulative explained variance ratio across the k kept components."""
        self._assert_fitted()
        return np.cumsum(self.variance_ratio_)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError("PCA is not fitted yet.  Call fit() first.")


# ------------------------------------------------------------------
# Module-level convenience functions (stateless wrappers)
# ------------------------------------------------------------------

def fit_pca(
    X: NDArray[np.float64],
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> PCA:
    """Create and fit a PCA object in one call.

    Parameters
    ----------
    X:
        Data matrix of shape (T, d).
    variance_threshold:
        Fraction of variance to retain.  Default 0.95.

    Returns
    -------
    pca:
        Fitted PCA object.
    """
    return PCA(variance_threshold=variance_threshold).fit(X)


def project_poses(
    skeletons_flat: NDArray[np.float64],
    pca: PCA,
) -> NDArray[np.float64]:
    """Project an array of flattened poses through a fitted PCA.

    Convenience wrapper so Developer 3 can call a single function
    instead of constructing the PCA manually.

    Parameters
    ----------
    skeletons_flat:
        Matrix of shape (T, 66) — output of ``linalg_utils.flatten_poses``.
    pca:
        A fitted PCA object.

    Returns
    -------
    Z:
        Projection matrix of shape (T, k).
    """
    return pca.transform(skeletons_flat)
