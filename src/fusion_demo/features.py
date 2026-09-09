"""
Feature-image generation for cross-modality fusion.

For each reconstruction (one modality/algorithm array), this module
generates a set of engineered feature images (local statistics, gradients,
smoothed copies, an NMF-based expansion) that together form the predictor
columns used by the PLS fusion model in `fusion.py`.

Available filters (selected via ``filters``, a list of these names):
    - "rangefilt": local max-min (morphological gradient)
    - "stdfilt":   local standard deviation
    - "gradmag":   Gaussian gradient magnitude
    - "gausslap":  Gaussian-Laplacian (blob/edge detector)
    - "gaussf":    Gaussian smoothing at one or more scales (see `gauss`)
    - "maxf":      local maximum
    - "minf":      local minimum
    - "nmf":       non-negative matrix factorization expansion
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Filters independently applied to the reconstruction array
IMAGE_FILTERS = {
    "rangefilt",
    "stdfilt",
    "gradmag",
    "gausslap",
    "gaussf",
    "maxf",
    "minf",
}
# All available filters, including those that consumes the already-built features matrix (e.g., "nmf")
ALL_FILTERS = IMAGE_FILTERS | {"nmf"}


def _stdfilt(x: np.ndarray, w_sz: tuple[int, ...]) -> np.ndarray:
    """Local standard deviation filter (dimension-agnostic)."""
    # normalization factor to balance the mismatch between Matlab and Python
    coeff = np.prod(w_sz) / (np.prod(w_sz) - 1)
    c1 = ndimage.uniform_filter(x, w_sz, mode="reflect") * np.sqrt(coeff)
    c2 = ndimage.uniform_filter(x * x, w_sz, mode="reflect") * coeff
    return np.sqrt(np.maximum(c2 - c1 * c1, 0.0))


def generate_feature_images(
    reconstruction: np.ndarray,
    filters: list[str],
    gauss_base: float = 2**0.5,
    gauss_depth: int = 1,
    n_components: int | None = None,
) -> np.ndarray:
    """Generate feature images for a single reconstruction array.

    Parameters
    ----------
    reconstruction : numpy.ndarray
        A single 2-D or 3-D reconstruction array.
    filters : list[str]
        Which filters to apply (see module docstring for available
        names).
    gauss_base : float
        Base of the Gaussian scale-space progression used by "gaussf"
        (sigma_k = gauss_base ** k for k = 1..gauss_depth). Default is sqrt(2).
    gauss_depth : int
        Number of Gaussian scales to generate when "gaussf" is selected.
    n_components : int, optional
        Number of NMF components to generate when "nmf" is selected.
        Defaults to the number of non-NMF feature images generated.

    Returns
    -------
    numpy.ndarray
        Feature matrix of shape ``(reconstruction.size, n_features)``,
        i.e. one flattened feature image per column. The raw
        (unfiltered) reconstruction is always included as the first
        column.

    Raises
    ------
    ValueError
        If ``filters`` contains an unrecognized filter name.
    """
    unknown = set(filters) - ALL_FILTERS
    if unknown:
        raise ValueError(
            f"Unrecognized filter(s): {sorted(unknown)}. "
            f"Available filters: {sorted(ALL_FILTERS)}"
        )

    window_size = (3,) * reconstruction.ndim
    columns = [reconstruction.ravel()]

    if "rangefilt" in filters:
        columns.append(
            ndimage.morphological_gradient(reconstruction, size=window_size).ravel()
        )
    if "stdfilt" in filters:
        columns.append(_stdfilt(reconstruction, window_size).ravel())
    if "gradmag" in filters:
        columns.append(
            ndimage.gaussian_gradient_magnitude(reconstruction, sigma=1).ravel()
        )
    if "gausslap" in filters:
        columns.append(ndimage.gaussian_laplace(reconstruction, sigma=1).ravel())
    if "gaussf" in filters:
        sigmas = gauss_base ** np.arange(1, gauss_depth + 1)
        for sigma in sigmas:
            columns.append(
                ndimage.gaussian_filter(
                    reconstruction, sigma=sigma, truncate=3.0
                ).ravel()
            )
    if "maxf" in filters:
        columns.append(ndimage.maximum_filter(reconstruction, size=window_size).ravel())
    if "minf" in filters:
        columns.append(ndimage.minimum_filter(reconstruction, size=window_size).ravel())

    features = np.array(columns, dtype=np.float32).transpose()

    if "nmf" in filters:
        if n_components is None:
            n_components = features.shape[1]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)
        scaler = StandardScaler(with_mean=False)
        scaled = scaler.fit_transform(scaled)
        nmf = NMF(n_components=n_components, init="nndsvd", tol=5e-3)
        nmf_features = nmf.fit_transform(scaled).astype(np.float32)
        features = np.concatenate((features, nmf_features), axis=1)

    return features


def number_of_feature_images(
    filters: list[str],
    gauss_depth: int = 1,
    n_components: int | None = None,
) -> int:
    """Compute how many feature-image columns `generate_feature_images`
    will produce for a given filter configuration, without actually
    running the filters. Used for memory pre-allocation.

    Parameters
    ----------
    filters : list[str]
    gauss_depth : int
    n_components : int, optional

    Returns
    -------
    int
        Number of feature-image columns.
    """
    n_orig = 1  # raw reconstruction, always included
    for name in IMAGE_FILTERS:
        if name in filters:
            n_orig += gauss_depth if name == "gaussf" else 1

    if "nmf" in filters:
        if n_components is None:
            n_components = n_orig
        return n_orig + n_components
    return n_orig
