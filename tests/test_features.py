"""Tests for fusion_demo.features."""

import numpy as np
import pytest

from fusion_demo.features import (
    generate_feature_images,
    number_of_feature_images,
    IMAGE_FILTERS,
)


def test_number_of_feature_images_WHEN_no_filters_THEN_returns_one():
    assert number_of_feature_images(filters=[]) == 1


def test_number_of_feature_images_WHEN_gaussf_selected_THEN_counts_one_per_scale():
    n = number_of_feature_images(filters=["gaussf"], gauss_depth=3)
    assert n == 1 + 3  # raw + 3 gaussian scales


def test_generate_feature_images_WHEN_no_filters_THEN_returns_only_raw_column():
    array = np.arange(16, dtype=float).reshape(4, 4)
    features = generate_feature_images(array, filters=[])
    assert features.shape == (16, 1)
    assert np.all(features[:, 0] == array.ravel())


def test_generate_feature_images_WHEN_2d_input_THEN_output_shape_matches_expected_count():
    array = np.ones((8, 8), dtype=float)
    filters = IMAGE_FILTERS
    gauss_depth = 2

    features = generate_feature_images(array, filters, gauss_depth=gauss_depth)
    expected = number_of_feature_images(filters, gauss_depth=gauss_depth)
    assert features.shape == (array.size, expected)


def test_generate_feature_images_WHEN_3d_input_THEN_output_shape_matches_expected_count():
    array = np.ones((6, 6, 6), dtype=float)
    filters = IMAGE_FILTERS
    gauss_depth = 2

    features = generate_feature_images(array, filters, gauss_depth=gauss_depth)
    expected = number_of_feature_images(filters, gauss_depth=gauss_depth)
    assert features.shape == (array.size, expected)


def test_generate_feature_images_WHEN_nmf_included_THEN_output_shape_matches_expected_count():
    array = np.eye(8, dtype=float)
    filters = ["gaussf", "rangefilt", "nmf"]
    gauss_depth = 2
    n_components = 3

    features = generate_feature_images(
        array, filters, gauss_depth=gauss_depth, n_components=n_components
    )
    expected = number_of_feature_images(
        filters, gauss_depth=gauss_depth, n_components=n_components
    )
    assert features.shape == (array.size, expected)


def test_generate_feature_images_WHEN_valid_filters_THEN_output_has_no_nans():
    array = np.ones((8, 8), dtype=float)
    features = generate_feature_images(array, IMAGE_FILTERS, gauss_depth=2)
    assert not np.isnan(features).any()


def test_generate_feature_images_WHEN_unknown_filter_THEN_raises_value_error():
    array = np.ones((4, 4))
    with pytest.raises(ValueError):
        generate_feature_images(array, filters=["not_a_real_filter"])
