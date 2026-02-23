"""Tests for model training functionality."""

import pytest
import numpy as np
from src.train import train_model, train_and_evaluate, get_base_estimator
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def test_train_model_returns_model(sample_features, sample_target, numerical_features, categorical_features):
    """Test that train_model returns a trained model."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    model = train_model(X_processed, sample_target, model_type="regression", 
                       n_splits=3, evaluate=False, return_models=False)
    
    assert model is not None
    assert hasattr(model, 'predict')


def test_train_model_types(sample_features, sample_target, numerical_features, categorical_features):
    """Test different model types."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    
    # Test Linear Regression
    model_lr = train_model(X_processed, sample_target, model_type="regression", 
                          n_splits=2, evaluate=False, return_models=False)
    assert isinstance(model_lr, LinearRegression)
    
    # Test Decision Tree
    model_dt = train_model(X_processed, sample_target, model_type="decision_tree", 
                          n_splits=2, evaluate=False, return_models=False)
    assert isinstance(model_dt, DecisionTreeRegressor)


def test_invalid_model_type(sample_features, sample_target, numerical_features, categorical_features):
    """Test that invalid model type raises ValueError."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    
    with pytest.raises(ValueError):
        train_model(X_processed, sample_target, model_type="invalid_type", 
                   n_splits=2, evaluate=False, return_models=False)


def test_cross_validation_execution(sample_features, sample_target, numerical_features, categorical_features, capsys):
    """Test that cross-validation runs when evaluate=True."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    model = train_model(X_processed, sample_target, model_type="regression", 
                       n_splits=3, evaluate=True, return_models=False)
    
    # Check that CV output was printed
    captured = capsys.readouterr()
    assert "Fold" in captured.out
    assert "Cross‑validation R²" in captured.out


def test_return_models_list(sample_features, sample_target, numerical_features, categorical_features):
    """Test that return_models=True returns a list of models."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    n_splits = 3
    models = train_model(X_processed, sample_target, model_type="regression", 
                        n_splits=n_splits, evaluate=False, return_models=True)
    
    assert isinstance(models, list)
    assert len(models) == n_splits
    assert all(hasattr(m, 'predict') for m in models)


def test_model_can_predict(sample_features, sample_target, numerical_features, categorical_features):
    """Test that trained model can make predictions."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    model = train_model(X_processed, sample_target, model_type="regression", 
                       n_splits=2, evaluate=False, return_models=False)
    
    predictions = model.predict(X_processed)
    
    assert len(predictions) == len(sample_target)
    assert all(isinstance(p, (int, float, np.number)) for p in predictions)


def test_different_n_splits(sample_features, sample_target, numerical_features, categorical_features):
    """Test training with different number of CV splits."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    
    for n_splits in [2, 3, 5]:
        model = train_model(X_processed, sample_target, model_type="regression", 
                           n_splits=n_splits, evaluate=False, return_models=False)
        assert model is not None


def test_shuffle_parameter(sample_features, sample_target, numerical_features, categorical_features):
    """Test shuffle parameter in cross-validation."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    
    # Should work with both shuffle=True and shuffle=False
    model_shuffle = train_model(X_processed, sample_target, model_type="regression", 
                                n_splits=2, shuffle=True, evaluate=False, return_models=False)
    model_no_shuffle = train_model(X_processed, sample_target, model_type="regression", 
                                   n_splits=2, shuffle=False, evaluate=False, return_models=False)
    
    assert model_shuffle is not None
    assert model_no_shuffle is not None


# Tests for new train_and_evaluate function

def test_get_base_estimator():
    """Test get_base_estimator with different model types."""
    linear = get_base_estimator("linear", random_state=42)
    assert isinstance(linear, LinearRegression)
    
    dt = get_base_estimator("decision_tree", random_state=42)
    assert isinstance(dt, DecisionTreeRegressor)
    
    rf = get_base_estimator("random_forest", random_state=42)
    assert isinstance(rf, RandomForestRegressor)
    
    with pytest.raises(ValueError):
        get_base_estimator("invalid_type")


def test_train_and_evaluate_basic(sample_features, sample_target, numerical_features, categorical_features):
    """Test basic train_and_evaluate functionality."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    
    model, metrics = train_and_evaluate(
        X_processed,
        sample_target,
        model_type="linear",
        n_splits=3,
        random_state=42
    )
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert isinstance(metrics, dict)
    assert 'mean_r2' in metrics
    assert 'std_r2' in metrics
    assert 'mean_rmse' in metrics
    assert 'mean_mae' in metrics
    assert 'cv_scores' in metrics
    assert len(metrics['cv_scores']) == 3


def test_train_and_evaluate_with_hyperparameter_tuning(sample_features, sample_target, numerical_features, categorical_features):
    """Test train_and_evaluate with hyperparameter tuning."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    
    param_grid = {
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }
    
    model, metrics = train_and_evaluate(
        X_processed,
        sample_target,
        model_type="decision_tree",
        n_splits=2,
        param_grid=param_grid,
        search_type="grid",
        random_state=42
    )
    
    assert model is not None
    assert 'best_params' in metrics
    assert len(metrics['best_params']) > 0
    assert 'max_depth' in metrics['best_params']


def test_train_and_evaluate_different_models(sample_features, sample_target, numerical_features, categorical_features):
    """Test train_and_evaluate with different model types."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    
    model_types = ["linear", "decision_tree"]
    
    for model_type in model_types:
        model, metrics = train_and_evaluate(
            X_processed,
            sample_target,
            model_type=model_type,
            n_splits=2,
            random_state=42
        )
        
        assert model is not None
        assert metrics['mean_r2'] is not None
        assert metrics['n_samples'] == len(X_processed)


def test_train_and_evaluate_metrics_format(sample_features, sample_target, numerical_features, categorical_features):
    """Test that metrics are in correct format."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    
    model, metrics = train_and_evaluate(
        X_processed,
        sample_target,
        model_type="linear",
        n_splits=3,
        random_state=42
    )
    
    # Check metric types
    assert isinstance(metrics['mean_r2'], (float, np.floating))
    assert isinstance(metrics['std_r2'], (float, np.floating))
    assert isinstance(metrics['mean_rmse'], (float, np.floating))
    assert isinstance(metrics['mean_mae'], (float, np.floating))
    assert isinstance(metrics['cv_scores'], list)
    assert isinstance(metrics['n_samples'], int)
    assert isinstance(metrics['n_features'], int)
