"""
I/O utilities for fusion_demo.

Handles loading the manifest + per-modality npz files that make up a
dataset (see `data/README.md`), and saving fusion results + run metadata.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Manifest:
    """Parsed contents of a dataset's manifest.json.

    Attributes
    ----------
    sample_id : str
        Identifier for the sample/measurement this dataset represents.
    shape : tuple[int, ...]
        Expected shape shared by every array in every modality (2-D slice
        or 3-D small volume).
    pixel_size_nm : float | None
        Physical pixel/voxel size, if known; None for synthetic/phantom
        data.
    modalities : dict[str, dict]
        Maps modality name (e.g. "haadf", "Ti") to its manifest entry,
        e.g. ``{"file": "haadf.npz", "algorithms": ["SIRT"]}``.
    source_path : Path
        Folder the manifest was loaded from (used to resolve relative
        npz file paths).
    """
    sample_id: str
    shape: tuple[int, ...]
    pixel_size_nm: float | None
    modalities: dict[str, dict]
    source_path: Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_manifest(data_dir: Path) -> Manifest:
    """Load and parse ``manifest.json`` from a dataset folder.

    Parameters
    ----------
    data_dir : Path
        Folder containing ``manifest.json`` and the modality npz files.

    Returns
    -------
    Manifest

    Raises
    ------
    FileNotFoundError
        If ``manifest.json`` is missing from ``data_dir``.
    """
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest.json found in {data_dir}. "
            "Expected a dataset folder containing manifest.json and one "
            "npz file per modality (see data/README.md)."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return Manifest(
        sample_id=raw["sample_id"],
        shape=tuple(raw["shape"]),
        pixel_size_nm=raw.get("pixel_size_nm"),
        modalities=raw["modalities"],
        source_path=data_dir,
    )


def load_modality(manifest: Manifest, name: str) -> dict[str, np.ndarray]:
    """Load one modality's npz file, checked against the manifest.

    Parameters
    ----------
    manifest : Manifest
    name : str
        Modality name as registered in ``manifest.modalities``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Maps reconstruction-algorithm name (e.g. "SIRT") to its array.

    Raises
    ------
    KeyError
        If ``name`` is not a modality registered in the manifest.
    FileNotFoundError
        If the modality's npz file does not exist on disk.
    ValueError
        If an array's shape does not match ``manifest.shape``.
    """
    if name not in manifest.modalities:
        raise KeyError(
            f"Modality '{name}' is not registered in the manifest. "
            f"Available modalities: {sorted(manifest.modalities)}"
        )

    modality_name = manifest.modalities[name]
    npz_path = manifest.source_path / modality_name["file"]
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Modality '{name}' references '{modality_name['file']}', "
            f"but no such file was found in {manifest.source_path}."
        )

    with np.load(npz_path) as npz_file:
        reconstructions = {}
        expected_shape = manifest.shape

        for algorithm in modality_name["algorithms"]:
            if algorithm not in npz_file:
                raise ValueError(
                    f"Modality '{name}' ({modality_name['file']}) is missing "
                    f"expected algorithm key '{algorithm}'. "
                    f"Available keys: {list(npz_file.keys())}"
                )
            array = npz_file[algorithm]
            if array.shape != expected_shape:
                raise ValueError(
                    f"Modality '{name}', algorithm '{algorithm}' has shape "
                    f"{array.shape}, expected {expected_shape} "
                    "(as declared in manifest.json)."
                )
            reconstructions[algorithm] = array

    return reconstructions


def load_dataset(
        data_dir: Path,
        modality_names: list[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Load a set of modalities from a dataset folder.

    Wrapper function: loads the manifest, then loads each requested
    modality (or all modalities registered in the manifest, if
    ``modality_names`` is None).

    Parameters
    ----------
    data_dir : Path
    modality_names : list[str], optional
        Subset of modalities to load (e.g. ``["haadf", "Ag"]`` as
        predictors, plus the target element). Defaults to all modalities
        listed in the manifest.

    Returns
    -------
    dict[str, dict[str, numpy.ndarray]]
        Maps modality name to its loaded reconstructions (algorithm name
        -> array), as returned by `load_modality`.

    Raises
    ------
    KeyError
        If a requested modality is not registered in the manifest.
    """
    manifest = load_manifest(data_dir)
    names = modality_names if modality_names is not None else list(manifest.modalities)
    return {name: load_modality(manifest, name) for name in names}


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_fusion_result(
        output_dir: Path,
        fused: np.ndarray,
        metrics: dict[str, float],
        config: dict
) -> Path:
    """Save fusion output + evaluation metrics + run config to disk.

    Parameters
    ----------
    output_dir : Path
        Destination folder for this run. Created if it doesn't exist.
    fused : numpy.ndarray
        The fused (PLS-predicted) image/volume.
    metrics : dict[str, float]
        Evaluation metrics to save, e.g. ``{"cc": ...}``.
    config : dict
        Run configuration (target/predictor elements, filters, gauss
        params, NMF n_components, etc.) to save alongside the result for
        reproducibility.

    Returns
    -------
    Path
        The ``output_dir`` the results were written to.

    Writes
    ------
    - ``<output_dir>/fusion.npy``: the fused array.
    - ``<output_dir>/metrics.json``: the metrics dict.
    - ``<output_dir>/config.json``: the config dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "fusion.npy", fused)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return output_dir
