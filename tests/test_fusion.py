"""Tests for fusion_demo.fusion."""

import numpy as np
import pytest

from fusion_demo.fusion import pls_regression, build_cross_modality_model


def test_pls_regression_WHEN_called_THEN_prediction_shape_matches_n_samples():
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.arange(4, dtype=float)

    _, y_pred = pls_regression(X, y, n_components=1)
    assert y_pred.shape == (4,)


def test_pls_regression_WHEN_known_linear_relation_THEN_recovers_high_correlation():
    t = np.linspace(-1, 1, 10)
    X = np.column_stack(
        [
            t,
            t**2,
            np.sin(np.pi * t),
            np.cos(np.pi * t),
            np.sin(2 * np.pi * t),
        ]
    )
    y = 2 * X[:, 0] - X[:, 1] + 0.01 * np.sin(3.7 * np.pi * t)

    _, y_pred = pls_regression(X, y)

    corr = np.corrcoef(y_pred, y)[0, 1]
    assert corr > 0.99


def test_pls_regression_WHEN_n_components_not_given_THEN_uses_default():
    t = np.linspace(-1, 1, 10)
    X = np.column_stack(
        [
            t,
            t**2,
            np.sin(np.pi * t),
            np.cos(np.pi * t),
        ]
    )
    y = 2 * t - t**2 + 0.1 * np.sin(2.3 * np.pi * t)

    coef_default, pred_default = pls_regression(X, y)
    coef_explicit, pred_explicit = pls_regression(X, y, n_components=4)

    np.testing.assert_allclose(coef_default, coef_explicit)
    np.testing.assert_allclose(pred_default, pred_explicit)


def test_build_cross_modality_model_WHEN_shape_mismatch_THEN_raises_value_error():
    target = np.ones((8, 8))
    predictors = {"bad": {"SIRT": np.ones((4, 4))}}
    with pytest.raises(ValueError):
        build_cross_modality_model(predictors, target, filters=["gaussf"])


def test_build_cross_modality_model_WHEN_called_THEN_fused_shape_matches_target():
    x = np.linspace(0, 1, 8)
    xx, yy = np.meshgrid(x, x)

    haadf = xx + yy
    ag = xx**2 + 0.5 * yy
    target = 0.6 * haadf - 0.4 * ag + 0.05 * np.sin(3 * np.pi * xx)

    predictors = {
        "haadf": {"SIRT": haadf},
        "Ag": {"SIRT": ag},
    }
    fused, _ = build_cross_modality_model(predictors, target, filters=["gaussf"])
    assert fused.shape == target.shape


def test_build_cross_modality_model_WHEN_multiple_reconstructions_per_modality_THEN_each_contributes_features():
    x = np.linspace(0, 1, 8)
    xx, yy = np.meshgrid(x, x)

    target = xx + yy
    predictors = {
        "haadf": {
            "SIRT": target.copy(),
            "FBP": target.copy(),
        },
    }
    _, coef = build_cross_modality_model(predictors, target, filters=[])
    # 2 algorithms x 1 feature column (raw, no filters) each = 2 columns.
    assert coef.shape[-1] == 2


def test_build_cross_modality_model_WHEN_predictor_equals_target_THEN_high_correlation():
    x = np.linspace(0, 1, 8)
    xx, yy = np.meshgrid(x, x)

    target = xx + yy
    predictors = {"haadf": {"SIRT": target.copy()}}

    fused, _ = build_cross_modality_model(predictors, target, filters=["gaussf"])
    corr = np.corrcoef(fused.ravel(), target.ravel())[0, 1]
    assert corr > 0.99
