"""Tests for model persistence utilities."""

import pytest
import os
from src.model_utils import (
    save_model, load_model,
    save_preprocessing_pipeline, load_preprocessing_pipeline,
    save_model_bundle, load_model_bundle
)


def test_save_and_load_model(trained_model, temp_model_path):
    """Test saving and loading a model."""
    metadata = {'test_key': 'test_value', 'version': '1.0'}
    
    # Save model
    save_model(trained_model, temp_model_path, metadata)
    assert os.path.exists(temp_model_path)
    
    # Load model
    loaded_model, loaded_metadata = load_model(temp_model_path)
    
    assert loaded_model is not None
    assert loaded_metadata is not None
    assert loaded_metadata['test_key'] == 'test_value'


def test_load_nonexistent_model():
    """Test loading a model that doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_model("nonexistent_model.joblib")


def test_saved_model_predictions_match(trained_model, sample_features, 
                                        sample_target, temp_model_path,
                                        numerical_features, categorical_features):
    """Test that loaded model produces same predictions as original."""
    from src.preprocessing import preprocess_data
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    
    # Get predictions from original model
    original_predictions = trained_model.predict(X_processed)
    
    # Save and load model
    save_model(trained_model, temp_model_path)
    loaded_model, _ = load_model(temp_model_path)
    
    # Get predictions from loaded model
    loaded_predictions = loaded_model.predict(X_processed)
    
    # Should be identical
    assert all(original_predictions == loaded_predictions)


def test_save_preprocessing_pipeline(sample_features, temp_pipeline_path,
                                     numerical_features, categorical_features):
    """Test saving preprocessing pipeline."""
    from src.preprocessing import preprocess_data
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    # Create pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    pipeline.fit(sample_features)
    
    # Save pipeline
    save_preprocessing_pipeline(pipeline, temp_pipeline_path)
    
    assert os.path.exists(temp_pipeline_path)


def test_load_preprocessing_pipeline(sample_features, temp_pipeline_path,
                                     numerical_features, categorical_features):
    """Test loading preprocessing pipeline."""
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    # Create and save pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    pipeline.fit(sample_features)
    
    save_preprocessing_pipeline(pipeline, temp_pipeline_path)
    
    # Load pipeline
    loaded_pipeline = load_preprocessing_pipeline(temp_pipeline_path)
    
    assert loaded_pipeline is not None
    assert hasattr(loaded_pipeline, 'transform')


def test_save_model_bundle(trained_model, sample_features, temp_model_path,
                           temp_pipeline_path, numerical_features, categorical_features):
    """Test saving model and pipeline together."""
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    # Create pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    pipeline.fit(sample_features)
    
    metadata = {'test': 'bundle'}
    
    # Save bundle
    model_path, pipeline_path = save_model_bundle(
        trained_model, pipeline, temp_model_path, temp_pipeline_path, metadata
    )
    
    assert os.path.exists(model_path)
    assert os.path.exists(pipeline_path)


def test_load_model_bundle(trained_model, sample_features, temp_model_path,
                           temp_pipeline_path, numerical_features, categorical_features):
    """Test loading model and pipeline together."""
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    # Create and save bundle
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    pipeline.fit(sample_features)
    
    metadata = {'test': 'bundle'}
    save_model_bundle(trained_model, pipeline, temp_model_path, temp_pipeline_path, metadata)
    
    # Load bundle
    loaded_model, loaded_pipeline, loaded_metadata = load_model_bundle(
        temp_model_path, temp_pipeline_path
    )
    
    assert loaded_model is not None
    assert loaded_pipeline is not None
    assert loaded_metadata['test'] == 'bundle'


def test_model_metadata_persistence(trained_model, temp_model_path):
    """Test that metadata is correctly saved and loaded."""
    metadata = {
        'model_type': 'LinearRegression',
        'metrics': {'r2': 0.85},
        'features': ['bed', 'bath'],
        'version': '1.0.0'
    }
    
    save_model(trained_model, temp_model_path, metadata)
    loaded_model, loaded_metadata = load_model(temp_model_path)
    
    assert loaded_metadata['model_type'] == 'LinearRegression'
    assert loaded_metadata['metrics']['r2'] == 0.85
    assert loaded_metadata['features'] == ['bed', 'bath']
    assert loaded_metadata['version'] == '1.0.0'


def test_load_pipeline_nonexistent():
    """Test loading pipeline that doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_preprocessing_pipeline("nonexistent_pipeline.joblib")
