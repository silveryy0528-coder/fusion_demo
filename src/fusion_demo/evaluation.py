"""
Evaluation metrics for the cross-modality fusion model.

Quantifies how well the fused prediction reproduces the (denoised)
target signal.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def pearson_correlation(target: np.ndarray, fused: np.ndarray) -> float:
    """Pearson correlation coefficient between the target and fusion result."""
    return stats.pearsonr(target.ravel(), fused.ravel()).statistic


def evaluate(target: np.ndarray, fused: np.ndarray) -> dict[str, float]:
    """Evaluate the reliability of a fusion result.

    Parameters
    ----------
    target : numpy.ndarray
        The (denoised) target array.
    fused : numpy.ndarray
        The fusion model's prediction, same shape as "target".

    Returns
    -------
    dict[str, float]
        "{"cc": ...}", ready to pass to `fusion_demo.io.save_fusion_result`.
    """
    return {
        "cc": pearson_correlation(target, fused),
    }

