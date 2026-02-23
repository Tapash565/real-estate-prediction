"""Tests for model evaluation functionality."""

import pytest
import numpy as np
from src.evaluate import evaluate_model


def test_evaluate_model_returns_dict(sample_target, sample_predictions):
    """Test that evaluate_model returns a dictionary."""
    metrics = evaluate_model(sample_target, sample_predictions)
    
    assert isinstance(metrics, dict)


def test_evaluate_perfect_predictions(sample_target):
    """Test evaluation with perfect predictions (R² should be 1.0)."""
    # Perfect predictions (identical to actual values)
    perfect_predictions = sample_target.copy()
    
    metrics = evaluate_model(sample_target, perfect_predictions)
    
    # R² should be 1.0 for perfect predictions
    assert abs(metrics['R^2 Score'] - 1.0) < 1e-10
    # MSE and MAE should be 0 for perfect predictions
    assert abs(metrics['Mean Squared Error']) < 1e-10
    assert abs(metrics['Mean Absolute Error']) < 1e-10


def test_evaluate_metric_keys(sample_target, sample_predictions):
    """Test that all expected metrics are present."""
    metrics = evaluate_model(sample_target, sample_predictions)
    
    expected_keys = ['Mean Squared Error', 'R^2 Score', 'Mean Absolute Error']
    
    for key in expected_keys:
        assert key in metrics, f"Expected metric '{key}' not found"


def test_evaluate_with_poor_predictions(sample_target):
    """Test evaluation with poor predictions (low R²)."""
    # Very poor predictions (constant value)
    poor_predictions = np.full_like(sample_target, sample_target.mean())
    
    metrics = evaluate_model(sample_target, poor_predictions)
    
    # R² should be close to 0 for constant predictions
    assert metrics['R^2 Score'] <= 0.01


def test_evaluate_metric_types(sample_target, sample_predictions):
    """Test that metrics are numeric values."""
    metrics = evaluate_model(sample_target, sample_predictions)
    
    for key, value in metrics.items():
        assert isinstance(value, (int, float, np.number)), f"Metric '{key}' is not numeric"


def test_evaluate_positive_mse_mae(sample_target, sample_predictions):
    """Test that MSE and MAE are non-negative."""
    metrics = evaluate_model(sample_target, sample_predictions)
    
    assert metrics['Mean Squared Error'] >= 0
    assert metrics['Mean Absolute Error'] >= 0


def test_evaluate_r2_range(sample_target, sample_predictions):
    """Test that R² is in a reasonable range."""
    metrics = evaluate_model(sample_target, sample_predictions)
    
    # R² can be negative for very poor models, but should be <= 1.0
    assert metrics['R^2 Score'] <= 1.0


def test_evaluate_with_arrays(sample_target):
    """Test evaluation with numpy arrays."""
    predictions = sample_target + np.random.normal(0, 10000, len(sample_target))
    
    metrics = evaluate_model(sample_target.values, predictions)
    
    assert isinstance(metrics, dict)
    assert len(metrics) == 3


def test_evaluate_zero_variance():
    """Test evaluation when predictions have zero variance."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([250, 250, 250, 250, 250])  # Constant predictions
    
    metrics = evaluate_model(y_true, y_pred)
    
    # Should still return valid metrics
    assert 'R^2 Score' in metrics
    assert 'Mean Squared Error' in metrics
    assert 'Mean Absolute Error' in metrics
