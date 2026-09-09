"""
Top-level pipeline orchestration.

Wires together `fusion_demo.io`, `fusion_demo.denoise`, `fusion_demo.fusion`,
and `fusion_demo.evaluation` into a single "run the whole thing" entry point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fusion_demo.io import load_dataset, save_fusion_result
from fusion_demo.denoise import denoise
from fusion_demo.fusion import build_cross_modality_model
from fusion_demo.evaluation import evaluate


def run_pipeline(
    data_dir: Path,
    target_modality: str,
    predictor_modalities: list[str],
    filters: list[str],
    gauss_base: float = 2**0.5,
    gauss_depth: int = 1,
    n_components: int | None = None,
    output_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Run the full fusion pipeline: load, denoise, fuse, evaluate, (save).

    Parameters
    ----------
    data_dir : Path
        Dataset folder containing ``manifest.json`` (see `fusion_demo.io`).
    target_modality : str
        Name of the modality to predict (e.g. "Ti").
    predictor_modalities : list[str]
        Names of the modalities used as predictors (e.g. ["haadf", "Ag"]).
    filters : list[str]
        Feature filters passed to `fusion_demo.fusion.build_cross_modality_model`.
    gauss_base : float, optional
        Base for Gaussian filters.
    gauss_depth : int, optional
        Depth for Gaussian filters.
    n_components : int, optional
        Number of PLS components passed to `pls_regression`.
    output_dir : Path, optional
        If given, the fusion result, metrics, and run config are saved
        here via `fusion_demo.io.save_fusion_result`.

    Returns
    -------
    fused : numpy.ndarray
        The fusion result, same shape as the target modality.
    coef : numpy.ndarray
        PLS regression coefficients.
    metrics : dict[str, float]
        Evaluation metrics, e.g. ``{"cc": ...}``.
    """
    dataset = load_dataset(
        data_dir, modality_names=[target_modality, *predictor_modalities]
    )

    target = denoise(dataset[target_modality])
    predictors = {name: dataset[name] for name in predictor_modalities}
    print(predictors)

    fused, coef = build_cross_modality_model(
        predictors,
        target,
        filters,
        gauss_base,
        gauss_depth,
        n_components,
    )
    metrics = evaluate(target, fused)

    if output_dir is not None:
        config = {
            "target_modality": target_modality,
            "predictor_modalities": predictor_modalities,
            "filters": filters,
            "gauss_base": gauss_base,
            "gauss_depth": gauss_depth,
            "n_components": n_components,
        }
        save_fusion_result(output_dir, fused, metrics, config)

    return fused, coef, metrics
