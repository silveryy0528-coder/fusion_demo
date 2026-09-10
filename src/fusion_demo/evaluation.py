"""Evaluation metrics for the cross-modality fusion model."""

from __future__ import annotations

import numpy as np
from scipy import stats


def pearson_correlation(reference: np.ndarray, fused: np.ndarray) -> float:
    """Pearson correlation coefficient between the reference and fusion result."""
    return stats.pearsonr(reference.ravel(), fused.ravel()).statistic


def evaluate(reference: np.ndarray, fused: np.ndarray) -> dict[str, float]:
    """Evaluate the reliability of a fusion result.

    Parameters
    ----------
    reference : numpy.ndarray
        The reference array, either the ground truth or the denoised target array.
    fused : numpy.ndarray
        The fusion model's prediction, same shape as "reference".

    Returns
    -------
    dict[str, float]
        "{"cc": ...}", ready to pass to `fusion_demo.io.save_fusion_result`.
    """
    return {
        "cc": pearson_correlation(reference, fused),
    }
