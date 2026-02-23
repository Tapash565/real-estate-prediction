"""Tests for visualization functionality."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from src import visualize
import os


def test_plot_predictions_vs_actual(sample_target, sample_predictions):
    """Test that predictions vs actual plot is created."""
    fig = visualize.plot_predictions_vs_actual(sample_target, sample_predictions)
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_residuals(sample_target, sample_predictions):
    """Test that residual plot is created."""
    fig = visualize.plot_residuals(sample_target, sample_predictions)
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_saves_to_path(sample_target, sample_predictions, tmp_path):
    """Test that plots are saved to specified path."""
    save_path = tmp_path / "test_plot.png"
    
    fig = visualize.plot_predictions_vs_actual(
        sample_target, sample_predictions, 
        save_path=str(save_path)
    )
    
    assert os.path.exists(save_path)
    plt.close(fig)


def test_plot_error_distribution(sample_target, sample_predictions):
    """Test that error distribution plot is created."""
    fig = visualize.plot_error_distribution(sample_target, sample_predictions)
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_cv_scores():
    """Test that CV scores plot is created."""
    scores = np.array([0.85, 0.82, 0.88, 0.84, 0.86])
    
    fig = visualize.plot_cv_scores(scores)
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_importance_with_tree_model():
    """Test feature importance plot with tree-based model."""
    from sklearn.tree import DecisionTreeRegressor
    
    # Create simple model with feature importances
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    
    feature_names = [f'feature_{i}' for i in range(5)]
    
    fig = visualize.plot_feature_importance(model, feature_names)
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_importance_with_linear_model():
    """Test feature importance plot with linear model (coefficients)."""
    from sklearn.linear_model import LinearRegression
    
    # Create simple linear model
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = LinearRegression()
    model.fit(X, y)
    
    feature_names = [f'feature_{i}' for i in range(5)]
    
    fig = visualize.plot_feature_importance(model, feature_names)
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_importance_top_n():
    """Test that top_n parameter limits features shown."""
    from sklearn.tree import DecisionTreeRegressor
    
    X = np.random.rand(100, 20)
    y = np.random.rand(100)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    
    feature_names = [f'feature_{i}' for i in range(20)]
    
    fig = visualize.plot_feature_importance(model, feature_names, top_n=10)
    
    assert fig is not None
    plt.close(fig)


def test_plot_with_custom_title(sample_target, sample_predictions):
    """Test plots with custom titles."""
    fig = visualize.plot_predictions_vs_actual(
        sample_target, sample_predictions,
        title="Custom Title Test"
    )
    
    assert fig is not None
    plt.close(fig)


def test_plot_error_distribution_saves(sample_target, sample_predictions, tmp_path):
    """Test that error distribution plot saves correctly."""
    save_path = tmp_path / "error_dist.png"
    
    fig = visualize.plot_error_distribution(
        sample_target, sample_predictions,
        save_path=str(save_path)
    )
    
    assert os.path.exists(save_path)
    plt.close(fig)


def test_plot_creates_directory_if_not_exists(sample_target, sample_predictions, tmp_path):
    """Test that plot functions create directory if it doesn't exist."""
    save_path = tmp_path / "nested" / "dir" / "plot.png"
    
    fig = visualize.plot_predictions_vs_actual(
        sample_target, sample_predictions,
        save_path=str(save_path)
    )
    
    assert os.path.exists(save_path)
    plt.close(fig)
