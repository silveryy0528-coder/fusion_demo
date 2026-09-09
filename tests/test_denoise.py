"""Tests for fusion_demo.denoise."""

import numpy as np
import pytest

from fusion_demo.denoise import denoise


def test_denoise_WHEN_single_reconstruction_THEN_same_array_returns_unchanged():
    sirt = np.eye(8)
    result = denoise({"SIRT": sirt})
    assert result is sirt


def test_denoise_WHEN_multiple_reconstructions_THEN_output_shape_matches_input():
    sirt = np.eye(8)
    reconstructions = {
        "SIRT": sirt,
        "FBP": np.fliplr(sirt),
    }
    result = denoise(reconstructions)
    assert result.shape == sirt.shape


def test_denoise_WHEN_negative_values_in_reconstructions_THEN_output_is_non_negative():
    sirt = np.eye(8)
    fbp = np.fliplr(sirt)
    fbp[0, 0] = -1.0

    reconstructions = {
        "SIRT": sirt,
        "FBP": fbp,
    }
    result = denoise(reconstructions)
    assert (result >= 0).all()


def test_denoise_WHEN_correlated_reconstructions_THEN_output_correlates_with_inputs():
    sirt = np.eye(8)
    fbp = 0.9 * sirt + 0.06
    reconstructions = {
        "SIRT": sirt,
        "FBP": fbp,
    }
    result = denoise(reconstructions)
    corr = np.corrcoef(result.ravel(), sirt.ravel())[0, 1]
    assert corr > 0.99
