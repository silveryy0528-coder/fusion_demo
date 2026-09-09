"""Plotting utilities for fusion_demo scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _to_2d(image: np.ndarray) -> np.ndarray:
    """Reduce a 3-D volume to its middle slice for display; pass 2-D through."""
    if image.ndim == 3:
        return image[image.shape[0] // 2]
    return image


def plot_fusion_summary(
    haadf: np.ndarray,
    target: np.ndarray,
    fused: np.ndarray,
    save_path: Path,
) -> None:
    """Save a 1x3 grayscale summary figure: HAADF | target | fusion result.

    3-D volumes are shown as their middle slice.

    Parameters
    ----------
    haadf : numpy.ndarray
        The (denoised) HAADF reference image.
    target : numpy.ndarray
        The (denoised) target element image being predicted.
    fused : numpy.ndarray
        The fusion model's prediction, same shape as ``target``.
    save_path : Path
        Where to save the figure (e.g. ``output_dir / "summary.png"``).
        The parent folder is created if it doesn't exist.
    """
    images = [_to_2d(haadf), _to_2d(target), _to_2d(fused)]
    titles = ["HAADF", "Target", "Fusion"]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
