"""End-to-end pipeline test: load -> denoise -> fuse -> evaluate -> save."""

import json

import numpy as np

from fusion_demo.pipeline import run_pipeline


def _write_correlated_dataset(data_dir, shape):
    """Write a synthetic dataset where Ti is strongly derived from haadf,
    so a well-behaved fusion pipeline should recover a high correlation.
    """
    manifest = {
        "sample_id": "pipeline_test",
        "pixel_size_nm": None,
        "shape": list(shape),
        "modalities": {
            "haadf": {"file": "haadf.npz", "algorithms": ["SIRT"]},
            "Ti": {"file": "Ti.npz", "algorithms": ["SIRT", "FBP"]},
        },
    }
    with open(data_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    x = np.linspace(0, 1, shape[0])
    xx, yy = np.meshgrid(x, x)
    base = xx + yy

    haadf = 2 * base + 0.01 * np.sin(3 * np.pi * xx)
    ti_sirt = 2 * base + 0.01 * np.sin(3 * np.pi * xx)
    ti_fbp = ti_sirt + 0.005 * np.sin(2 * np.pi * yy)

    np.savez(data_dir / "haadf.npz", SIRT=haadf)
    np.savez(data_dir / "Ti.npz", SIRT=ti_sirt, FBP=ti_fbp)


def test_run_pipeline_WHEN_run_on_correlated_synthetic_dataset_THEN_produces_high_correlation(
    tmp_path,
):
    shape = (16, 16)
    _write_correlated_dataset(tmp_path, shape=shape)
    output_dir = tmp_path / "results" / "run1"

    fused, _, metrics = run_pipeline(
        data_dir=tmp_path,
        target_modality="Ti",
        predictor_modalities=["haadf"],
        filters=["gaussf", "rangefilt"],
        gauss_depth=2,
        output_dir=output_dir,
    )

    assert fused.shape == shape
    assert metrics["cc"] > 0.8
    assert (output_dir / "fusion.npy").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "config.json").is_file()


def test_run_pipeline_WHEN_output_dir_omitted_THEN_does_not_write_files(tmp_path):
    _write_correlated_dataset(tmp_path, shape=(16, 16))

    run_pipeline(
        data_dir=tmp_path,
        target_modality="Ti",
        predictor_modalities=["haadf"],
        filters=["gaussf"],
    )

    assert not (tmp_path / "results").exists()
