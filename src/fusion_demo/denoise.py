"""NMF-based denoising for modalities with multiple reconstructions."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF


def denoise(reconstructions: dict[str, np.ndarray]) -> np.ndarray:
    """Collapse reconstructions into a single denoised array using NMF.
    If there is only one entry, it is returned unchanged.

    Parameters
    ----------
    reconstructions : dict[str, numpy.ndarray]
        Maps algorithm name to its (same-shaped) reconstruction array, as
        returned by `fusion_demo.io.load_modality`.

    Returns
    -------
    numpy.ndarray
        Single denoised array, same shape as each input reconstruction.
    """
    if len(reconstructions) == 1:
        (array,) = reconstructions.values()
        return array

    shape = next(iter(reconstructions.values())).shape

    # stack reconstructions as columns: [n_voxels, n_algorithms]
    channels = []
    for name, array in reconstructions.items():
        if (array < 0).any():
            print(f"Clipping negative values to zero in reconstruction '{name}'")
            array = np.clip(array, 0, None)
        channels.append(array.ravel())
    X = np.stack(channels, axis=1)

    nmf = NMF(n_components=1, init="nndsvd", tol=5e-3)
    w = nmf.fit_transform(X)
    h = nmf.components_
    h_norm = np.linalg.norm(h)
    denoised = (w * h_norm).reshape(shape)

    return denoised
