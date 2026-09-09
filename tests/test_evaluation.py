"""Tests for fusion_demo.evaluation."""

import numpy as np
import pytest

from fusion_demo.evaluation import pearson_correlation, evaluate


def test_pearson_correlation_WHEN_arrays_identical_THEN_returns_one():
    x = np.arange(5, dtype=float)
    assert pearson_correlation(x, x) == pytest.approx(1.0)


def test_pearson_correlation_WHEN_arrays_perfectly_anticorrelated_THEN_returns_minus_one():
    x = np.arange(5, dtype=float)
    assert pearson_correlation(x, -x) == pytest.approx(-1.0)


def test_pearson_correlation_WHEN_arrays_uncorrelated_THEN_returns_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=10000)
    y = rng.normal(size=10000)
    assert abs(pearson_correlation(x, y)) < 0.05


def test_pearson_correlation_GIVEN_2d_arrays_THEN_flattens_before_comparing():
    x = np.arange(9, dtype=float).reshape(3, 3)
    y = x.copy()
    assert pearson_correlation(x, y) == pytest.approx(1.0)


def test_evaluate_WHEN_called_THEN_returns_dict_with_cc_key():
    x = np.arange(5, dtype=float)
    metrics = evaluate(x, x)
    assert set(metrics.keys()) == {"cc"}
    assert metrics["cc"] == pytest.approx(1.0)
