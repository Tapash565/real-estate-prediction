"""Tests for pipeline construction."""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.pipeline import (
    build_preprocessing_pipeline,
    build_full_pipeline,
    get_feature_names
)


def test_build_preprocessing_pipeline(numerical_features, categorical_features):
    """Test building preprocessing pipeline."""
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    
    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 2  # numerical and categorical


def test_preprocessing_pipeline_transform(sample_features, numerical_features, categorical_features):
    """Test that preprocessing pipeline transforms data correctly."""
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    
    X_transformed = preprocessor.fit_transform(sample_features)
    
    assert X_transformed.shape[0] == len(sample_features)
    assert X_transformed.shape[1] > len(numerical_features)  # includes one-hot encoded
    assert not np.isnan(X_transformed).any()  # no missing values


def test_preprocessing_handles_missing_values(numerical_features, categorical_features):
    """Test that preprocessing handles missing values."""
    data = {
        'bed': [3.0, np.nan, 4.0],
        'bath': [2.0, 2.0, np.nan],
        'house_size': [2000.0, 1500.0, 2500.0],
        'acre_lot': [0.25, np.nan, 0.3],
        'city': ['Manchester', 'Nashua', np.nan],
        'state': ['NH', 'MA', 'NH'],
        'zip_code': ['03101', '03060', np.nan],
        'brokered_by': ['Keller Williams', np.nan, 'RE/MAX'],
        'status': ['for_sale', 'sold', np.nan]
    }
    df = pd.DataFrame(data)
    
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    X_transformed = preprocessor.fit_transform(df)
    
    assert not np.isnan(X_transformed).any()


def test_build_full_pipeline(numerical_features, categorical_features):
    """Test building full pipeline with model."""
    pipeline = build_full_pipeline(
        numerical_features,
        categorical_features,
        model_type='linear',
        random_state=42
    )
    
    assert isinstance(pipeline, Pipeline)
    assert 'preprocessor' in pipeline.named_steps
    assert 'model' in pipeline.named_steps


def test_full_pipeline_fit_predict(sample_features, sample_target, numerical_features, categorical_features):
    """Test that full pipeline can fit and predict."""
    pipeline = build_full_pipeline(
        numerical_features,
        categorical_features,
        model_type='linear',
        random_state=42
    )
    
    pipeline.fit(sample_features, sample_target)
    predictions = pipeline.predict(sample_features)
    
    assert len(predictions) == len(sample_target)
    assert predictions.dtype in [np.float32, np.float64]


def test_full_pipeline_different_models(sample_features, sample_target, numerical_features, categorical_features):
    """Test building pipelines with different model types."""
    model_types = ['linear', 'decision_tree', 'random_forest']
    
    for model_type in model_types:
        pipeline = build_full_pipeline(
            numerical_features,
            categorical_features,
            model_type=model_type,
            random_state=42
        )
        
        pipeline.fit(sample_features, sample_target)
        predictions = pipeline.predict(sample_features)
        
        assert len(predictions) == len(sample_target)


def test_pipeline_with_custom_params(sample_features, sample_target, numerical_features, categorical_features):
    """Test pipeline with custom model parameters."""
    model_params = {'n_estimators': 50, 'max_depth': 5}
    
    pipeline = build_full_pipeline(
        numerical_features,
        categorical_features,
        model_type='random_forest',
        random_state=42,
        model_params=model_params
    )
    
    pipeline.fit(sample_features, sample_target)
    
    # Verify parameters were set
    assert pipeline.named_steps['model'].n_estimators == 50
    assert pipeline.named_steps['model'].max_depth == 5


def test_get_feature_names(sample_features, numerical_features, categorical_features):
    """Test extracting feature names from fitted preprocessor."""
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    preprocessor.fit(sample_features)
    
    feature_names = get_feature_names(preprocessor)
    
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
