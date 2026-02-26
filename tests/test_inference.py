"""Tests for inference functionality."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.inference import (
    ModelPredictor,
    load_model_bundle,
    predict,
    batch_predict
)
from src.pipeline import build_full_pipeline


@pytest.fixture
def trained_model(tmp_path, sample_features, sample_target, numerical_features, categorical_features):
    """Create and save a trained model for testing."""
    from src.pipeline import build_full_pipeline
    
    # Train a simple model
    pipeline = build_full_pipeline(
        numerical_features,
        categorical_features,
        model_type='linear',
        random_state=42
    )
    pipeline.fit(sample_features, sample_target)
    
    # Save model
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(pipeline.named_steps['model'], model_path)
    
    # Save preprocessor
    pipeline_path = tmp_path / "test_pipeline.joblib"
    joblib.dump(pipeline.named_steps['preprocessor'], pipeline_path)
    
    return str(model_path), str(pipeline_path)


def test_model_predictor_load(trained_model):
    """Test that ModelPredictor can load model."""
    model_path, pipeline_path = trained_model
    
    predictor = ModelPredictor(model_path, pipeline_path)
    predictor.load()
    
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    assert predictor._is_loaded is True


def test_model_predictor_predict(trained_model, sample_features):
    """Test ModelPredictor prediction."""
    model_path, pipeline_path = trained_model
    
    predictor = ModelPredictor(model_path, pipeline_path)
    predictor.load()
    
    predictions = predictor.predict(sample_features)
    
    assert len(predictions) == len(sample_features)
    assert predictions.dtype in [np.float32, np.float64]


def test_model_predictor_predict_without_load(trained_model, sample_features):
    """Test that predicting without loading raises error."""
    model_path, pipeline_path = trained_model
    
    predictor = ModelPredictor(model_path, pipeline_path)
    
    with pytest.raises(RuntimeError, match="Model not loaded"):
        predictor.predict(sample_features)


def test_model_predictor_return_dataframe(trained_model, sample_features):
    """Test returning predictions as DataFrame."""
    model_path, pipeline_path = trained_model
    
    predictor = ModelPredictor(model_path, pipeline_path)
    predictor.load()
    
    predictions = predictor.predict(sample_features, return_dataframe=True)
    
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(sample_features)
    assert 'prediction' in predictions.columns


def test_load_model_bundle(trained_model):
    """Test load_model_bundle function."""
    model_path, pipeline_path = trained_model
    
    model, preprocessor = load_model_bundle(model_path, pipeline_path)
    
    assert model is not None
    assert preprocessor is not None


def test_load_model_bundle_file_not_found():
    """Test that load_model_bundle raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model_bundle("nonexistent.joblib", "nonexistent_pipeline.joblib")


def test_predict_function(trained_model, sample_features):
    """Test convenience predict function."""
    model_path, pipeline_path = trained_model
    
    predictions = predict(sample_features, model_path, pipeline_path)
    
    assert len(predictions) == len(sample_features)
    assert predictions.dtype in [np.float32, np.float64]


def test_batch_predict(trained_model, numerical_features, categorical_features):
    """Test batch prediction for large datasets."""
    model_path, pipeline_path = trained_model
    
    # Create larger dataset
    n_samples = 1000
    data = {
        'bed': np.random.randint(1, 6, n_samples),
        'bath': np.random.randint(1, 4, n_samples),
        'house_size': np.random.randint(1000, 5000, n_samples),
        'acre_lot': np.random.uniform(0.1, 2.0, n_samples),
        'city': np.random.choice(['Manchester', 'Nashua', 'Concord'], n_samples),
        'state': np.random.choice(['NH', 'MA', 'VT'], n_samples),
        'zip_code': np.random.choice(['03101', '03060', '03301'], n_samples).astype(str),
        'brokered_by': np.random.choice(['Keller Williams', 'RE/MAX', 'Century 21'], n_samples),
        'status': np.random.choice(['for_sale', 'sold'], n_samples)
    }
    large_features = pd.DataFrame(data)
    
    predictions = batch_predict(
        large_features,
        model_path,
        pipeline_path,
        batch_size=100
    )
    
    assert len(predictions) == n_samples
    assert predictions.dtype in [np.float32, np.float64]


def test_predictor_without_pipeline(tmp_path, sample_features, sample_target, numerical_features, categorical_features):
    """Test predictor with only model (no separate preprocessing)."""
    from src.pipeline import build_full_pipeline
    
    # Train full pipeline
    pipeline = build_full_pipeline(
        numerical_features,
        categorical_features,
        model_type='linear',
        random_state=42
    )
    pipeline.fit(sample_features, sample_target)
    
    # Save entire pipeline as model
    model_path = tmp_path / "full_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    
    # Load and predict (no separate preprocessing)
    loaded_pipeline = joblib.load(model_path)
    predictions = loaded_pipeline.predict(sample_features)
    
    assert len(predictions) == len(sample_features)
