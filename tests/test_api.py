"""Tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the model loading to avoid file dependencies
import api.main as api_main

# Create a mock model and pipeline for testing
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def mock_model_and_pipeline(monkeypatch):
    """Mock the model and pipeline for testing."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([300000.0])
    
    mock_pipeline = MagicMock()
    mock_pipeline.transform.return_value = np.array([[1, 2, 3, 4, 5]])
    
    mock_metadata = {
        'model_type': 'LinearRegression',
        'saved_at': '2026-02-21T10:00:00',
        'metrics': {
            'R^2 Score': 0.85,
            'Mean Squared Error': 12500000.0,
            'Mean Absolute Error': 25000.0
        },
        'features': {
            'numerical': ['bed', 'bath', 'house_size', 'acre_lot'],
            'categorical': ['city', 'state', 'zip_code', 'brokered_by', 'status']
        },
        'num_samples': 50000,
        'cv_mean': 0.84,
        'cv_std': 0.02
    }
    
    monkeypatch.setattr(api_main, 'model', mock_model)
    monkeypatch.setattr(api_main, 'preprocessing_pipeline', mock_pipeline)
    monkeypatch.setattr(api_main, 'model_metadata', mock_metadata)
    
    return mock_model, mock_pipeline, mock_metadata


@pytest.fixture
def client():
    """Create a test client."""
    from api.main import app
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data


def test_predict_endpoint_with_model(client, mock_model_and_pipeline):
    """Test prediction endpoint with mocked model."""
    property_data = {
        "bed": 3.0,
        "bath": 2.0,
        "house_size": 2000.0,
        "acre_lot": 0.25,
        "city": "Manchester",
        "state": "New Hampshire",
        "zip_code": "03101",
        "brokered_by": "Keller Williams",
        "status": "for_sale"
    }
    
    response = client.post("/predict", json=property_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert data["currency"] == "USD"


def test_predict_endpoint_invalid_input(client, mock_model_and_pipeline):
    """Test prediction endpoint with invalid input."""
    invalid_data = {
        "bed": -1,  # Invalid: negative value
        "bath": 2.0,
        "house_size": 2000.0,
        "acre_lot": 0.25,
        "city": "Manchester",
        "state": "New Hampshire",
        "zip_code": "03101",
        "brokered_by": "Keller Williams",
        "status": "for_sale"
    }
    
    response = client.post("/predict", json=invalid_data)
    
    # Should return validation error
    assert response.status_code == 422


def test_batch_predict_endpoint(client, mock_model_and_pipeline):
    """Test batch prediction endpoint."""
    batch_data = {
        "properties": [
            {
                "bed": 3.0,
                "bath": 2.0,
                "house_size": 2000.0,
                "acre_lot": 0.25,
                "city": "Manchester",
                "state": "New Hampshire",
                "zip_code": "03101",
                "brokered_by": "Keller Williams",
                "status": "for_sale"
            },
            {
                "bed": 4.0,
                "bath": 3.0,
                "house_size": 2500.0,
                "acre_lot": 0.5,
                "city": "Nashua",
                "state": "New Hampshire",
                "zip_code": "03060",
                "brokered_by": "RE/MAX",
                "status": "for_sale"
            }
        ]
    }
    
    response = client.post("/predict/batch", json=batch_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "count" in data
    assert data["count"] == 2


def test_model_info_endpoint(client, mock_model_and_pipeline):
    """Test model info endpoint."""
    response = client.get("/model/info")
    
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "metrics" in data


def test_model_features_endpoint(client, mock_model_and_pipeline):
    """Test model features endpoint."""
    response = client.get("/model/features")
    
    assert response.status_code == 200
    data = response.json()
    assert "numerical_features" in data
    assert "categorical_features" in data
    assert "total_features" in data


def test_model_metrics_endpoint(client, mock_model_and_pipeline):
    """Test model metrics endpoint."""
    response = client.get("/model/metrics")
    
    assert response.status_code == 200
    data = response.json()
    assert "evaluation_metrics" in data
    assert "cross_validation" in data


def test_predict_without_model_loaded(client, monkeypatch):
    """Test prediction when model is not loaded."""
    # Set model to None
    monkeypatch.setattr(api_main, 'model', None)
    monkeypatch.setattr(api_main, 'preprocessing_pipeline', None)
    
    property_data = {
        "bed": 3.0,
        "bath": 2.0,
        "house_size": 2000.0,
        "acre_lot": 0.25,
        "city": "Manchester",
        "state": "New Hampshire",
        "zip_code": "03101",
        "brokered_by": "Keller Williams",
        "status": "for_sale"
    }
    
    response = client.post("/predict", json=property_data)
    
    # Should return service unavailable
    assert response.status_code == 503


def test_predict_missing_fields(client, mock_model_and_pipeline):
    """Test prediction with missing required fields."""
    incomplete_data = {
        "bed": 3.0,
        "bath": 2.0
        # Missing other required fields
    }
    
    response = client.post("/predict", json=incomplete_data)
    
    # Should return validation error
    assert response.status_code == 422


def test_health_when_model_not_loaded(client, monkeypatch):
    """Test health check when model is not loaded."""
    monkeypatch.setattr(api_main, 'model', None)
    
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] == False
    assert data["status"] == "degraded"
