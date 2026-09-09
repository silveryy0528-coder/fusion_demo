"""Tests for fusion_demo.io."""

import json

import numpy as np
import pytest

from fusion_demo.io import (
    load_manifest,
    load_modality,
    load_dataset,
    save_fusion_result,
)


def _write_dataset(data_dir, shape):
    """Write a small synthetic dataset (manifest + npz files) to data_dir."""
    manifest = {
        "sample_id": "test_sample",
        "pixel_size_nm": None,
        "shape": list(shape),
        "modalities": {
            "haadf": {"file": "haadf.npz", "algorithms": ["SIRT"]},
            "Ti": {"file": "Ti.npz", "algorithms": ["SIRT", "FBP"]},
        },
    }
    with open(data_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    sirt = np.ones(shape)
    fbp = np.eye(shape[0])
    np.savez(data_dir / "haadf.npz", SIRT=sirt)
    np.savez(data_dir / "Ti.npz", SIRT=sirt, FBP=fbp)

    return manifest


def test_load_manifest_WHEN_manifest_present_THEN_parses_expected_fields(tmp_path):
    shape = (4, 4)
    _write_dataset(tmp_path, shape=shape)
    manifest = load_manifest(tmp_path)
    assert manifest.sample_id == "test_sample"
    assert manifest.shape == shape
    assert set(manifest.modalities) == {"haadf", "Ti"}


def test_load_manifest_WHEN_manifest_missing_THEN_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path)


def test_load_modality_WHEN_modality_registered_THEN_returns_expected_arrays(tmp_path):
    shape = (4, 4)
    _write_dataset(tmp_path, shape=shape)
    manifest = load_manifest(tmp_path)
    result = load_modality(manifest, "Ti")
    assert set(result.keys()) == {"SIRT", "FBP"}
    assert result["SIRT"].shape == shape


def test_load_modality_WHEN_modality_not_registered_THEN_raises_key_error(tmp_path):
    _write_dataset(tmp_path, shape=(4, 4))
    manifest = load_manifest(tmp_path)
    with pytest.raises(KeyError):
        load_modality(manifest, "not_a_modality")


def test_load_modality_WHEN_npz_file_missing_THEN_raises_file_not_found_error(tmp_path):
    _write_dataset(tmp_path, shape=(4, 4))
    manifest_obj = load_manifest(tmp_path)
    (tmp_path / "haadf.npz").unlink()
    with pytest.raises(FileNotFoundError):
        load_modality(manifest_obj, "haadf")


def test_load_modality_WHEN_shape_mismatch_THEN_raises_value_error(tmp_path):
    _write_dataset(tmp_path, shape=(4, 4))
    manifest = load_manifest(tmp_path)
    # overwrite haadf.npz with a wrong shape
    np.savez(tmp_path / "haadf.npz", SIRT=np.ones((5, 5)))
    with pytest.raises(ValueError):
        load_modality(manifest, "haadf")


def test_load_dataset_WHEN_modality_names_given_THEN_loads_only_requested(tmp_path):
    _write_dataset(tmp_path, shape=(4, 4))
    result = load_dataset(tmp_path, modality_names=["haadf"])
    assert set(result.keys()) == {"haadf"}


def test_load_dataset_WHEN_modality_names_omitted_THEN_loads_all(tmp_path):
    _write_dataset(tmp_path, shape=(4, 4))
    result = load_dataset(tmp_path)
    assert set(result.keys()) == {"haadf", "Ti"}


def test_save_fusion_result_WHEN_called_THEN_writes_expected_files(tmp_path):
    output_dir = tmp_path / "run"
    fused = np.ones((4, 4))
    metrics = {"cc": 0.9}
    config = {"filters": ["gaussf"]}

    result_path = save_fusion_result(output_dir, fused, metrics, config)

    assert result_path == output_dir
    assert (output_dir / "fusion.npy").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "config.json").is_file()

    loaded_fused = np.load(output_dir / "fusion.npy")
    assert np.array_equal(loaded_fused, fused)

    with open(output_dir / "metrics.json") as f:
        assert json.load(f) == metrics

    with open(output_dir / "config.json") as f:
        assert json.load(f) == config
