"""
Cross-modality PLS fusion.

Builds a predictor feature matrix from one or more modalities (via
`fusion_demo.features`) and fits a Partial Least Squares (PLS) regression
model to predict a target element's (denoised) signal from it.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.cross_decomposition import PLSRegression

from fusion_demo.features import generate_feature_images


def pls_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Partial Least Squares regression model and predict on X.

    Both X and y are normalized to zero mean and unit variance (z-scored)
    before fitting, so absolute scale differences between predictors
    (e.g. different modalities' native intensity ranges) or between
    predictors and the target do not affect the fit.

    Parameters
    ----------
    X : numpy.ndarray
        Predictor feature matrix, shape ``(n_samples, n_features)``.
    y : numpy.ndarray
        Target vector, shape ``(n_samples,)``.
    n_components : int, optional
        Number of PLS components. Defaults to ``min(n_samples - 1, n_features)``.

    Returns
    -------
    coef : numpy.ndarray
        PLS regression coefficients.
    y_pred : numpy.ndarray
        Predicted target values, shape ``(n_samples,)``.
    """
    X_z = stats.zscore(X, axis=0, ddof=1)
    y_mean, y_std = y.mean(), y.std(ddof=1)
    y_z = (y - y_mean) / y_std

    if n_components is None:
        n_components = min(X_z.shape[0] - 1, X_z.shape[1])

    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_z, y_z)
    y_pred_z = pls.predict(X_z)

    # Restore target value
    y_pred = y_pred_z * y_std + y_mean

    return pls.coef_, y_pred.ravel()


def build_cross_modality_model(
    predictors: dict[str, dict[str, np.ndarray]],
    target: np.ndarray,
    filters: list[str],
    gauss_base: float = 2**0.5,
    gauss_depth: int = 1,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the cross-modality fusion model and predict the target.

    Generates feature images for every reconstruction of every
    predictor modality, concatenates them into one feature
    matrix, and fits/predicts with `pls_regression`.

    Parameters
    ----------
    predictors : dict[str, dict[str, numpy.ndarray]]
        Maps modality name (e.g. "haadf", "Ag") to its reconstructions
        (algorithm name -> array), as returned by `fusion_demo.io.load_modality`.
        All arrays must share the same shape as ``target``.
    target : numpy.ndarray
        The (denoised) target element array to predict, same shape as
        each predictor array.
    filters : list[str]
        Feature filters to apply to each predictor reconstruction (see
        `fusion_demo.features` for available names).
    gauss_base : float
    gauss_depth : int
    n_components : int, optional
        Passed through to `pls_regression` (number of PLS components).

    Returns
    -------
    fused : numpy.ndarray
        Fusion result, reshaped back to ``target.shape``.
    coef : numpy.ndarray
        PLS regression coefficients (one per feature column, across all
        predictor reconstructions).

    Raises
    ------
    ValueError
        If any predictor reconstruction's shape does not match ``target.shape``.
    """
    shape = target.shape

    for modality_name, reconstructions in predictors.items():
        for algorithm_name, array in reconstructions.items():
            if array.shape != shape:
                raise ValueError(
                    f"Predictor '{modality_name}' algorithm '{algorithm_name}' "
                    f"has shape {array.shape}, expected {shape} (target's shape)."
                )

    feature_blocks = [
        generate_feature_images(array, filters, gauss_base, gauss_depth)
        for reconstructions in predictors.values()
        for array in reconstructions.values()
    ]
    X = np.concatenate(feature_blocks, axis=1)
    y = target.ravel()

    coef, y_pred = pls_regression(X, y, n_components=n_components)
    fused = y_pred.reshape(shape)

    return fused, coef
