"""Pytest configuration and shared fixtures."""

import os
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import tempfile


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with correct column names for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'bed': np.random.randint(1, 6, n_samples).astype(float),
        'bath': np.random.randint(1, 4, n_samples).astype(float),
        'house_size': np.random.randint(800, 4000, n_samples).astype(float),
        'acre_lot': np.random.uniform(0.1, 2.0, n_samples),
        'city': np.random.choice(['Manchester', 'Nashua', 'Concord'], n_samples),
        'state': np.random.choice(['New Hampshire', 'Massachusetts'], n_samples),
        'zip_code': np.random.choice(['03101', '03060', '03301'], n_samples).astype(str),
        'brokered_by': np.random.choice(['Keller Williams', 'RE/MAX', 'Century 21'], n_samples),
        'status': np.random.choice(['for_sale', 'sold', 'pending'], n_samples),
        'price': np.random.randint(150000, 600000, n_samples).astype(float)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_features(sample_dataframe):
    """Extract features (X) from sample DataFrame."""
    return sample_dataframe.drop(columns=['price'])


@pytest.fixture
def sample_target(sample_dataframe):
    """Extract target (y) from sample DataFrame."""
    return sample_dataframe['price']


@pytest.fixture
def numerical_features():
    """List of numerical feature names."""
    return ['bed', 'bath', 'house_size', 'acre_lot']


@pytest.fixture
def categorical_features():
    """List of categorical feature names."""
    return ['city', 'state', 'zip_code', 'brokered_by', 'status']


@pytest.fixture
def trained_model(sample_features, sample_target):
    """Create a simple trained model for testing."""
    from src.preprocessing import preprocess_data
    
    numerical_features = ['bed', 'bath', 'house_size', 'acre_lot']
    categorical_features = ['city', 'state', 'zip_code', 'brokered_by', 'status']
    
    X_processed = preprocess_data(sample_features, categorical_features, numerical_features)
    
    model = LinearRegression()
    model.fit(X_processed, sample_target)
    
    return model


@pytest.fixture
def temp_model_path():
    """Create a temporary file path for model saving/loading tests."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, 'test_model.joblib')
    
    yield model_path
    
    # Cleanup after test
    if os.path.exists(model_path):
        os.remove(model_path)
    try:
        os.rmdir(temp_dir)
    except:
        pass


@pytest.fixture
def temp_pipeline_path():
    """Create a temporary file path for pipeline saving/loading tests."""
    temp_dir = tempfile.mkdtemp()
    pipeline_path = os.path.join(temp_dir, 'test_pipeline.joblib')
    
    yield pipeline_path
    
    # Cleanup after test
    if os.path.exists(pipeline_path):
        os.remove(pipeline_path)
    try:
        os.rmdir(temp_dir)
    except:
        pass


@pytest.fixture
def sample_csv_path(sample_dataframe):
    """Create a temporary CSV file for testing data loading."""
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, 'test_data.csv')
    
    sample_dataframe.to_csv(csv_path, index=False)
    
    yield csv_path
    
    # Cleanup after test
    if os.path.exists(csv_path):
        os.remove(csv_path)
    try:
        os.rmdir(temp_dir)
    except:
        pass


@pytest.fixture
def sample_predictions(sample_target):
    """Generate sample predictions for testing evaluation."""
    np.random.seed(42)
    # Add some noise to the actual values to simulate predictions
    noise = np.random.normal(0, 20000, len(sample_target))
    return sample_target + noise
