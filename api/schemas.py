"""Pydantic schemas for API request/response validation."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator


class PropertyFeatures(BaseModel):
    """Input schema for property features."""
    
    bed: float = Field(..., description="Number of bedrooms", ge=0, le=50)
    bath: float = Field(..., description="Number of bathrooms", ge=0, le=50)
    house_size: float = Field(..., description="House size in square feet", gt=0, le=50000)
    acre_lot: float = Field(..., description="Lot size in acres", ge=0, le=10000)
    city: str = Field(..., description="City name", min_length=1, max_length=100)
    state: str = Field(..., description="State name", min_length=2, max_length=50)
    zip_code: str = Field(..., description="ZIP code", min_length=1, max_length=20)
    brokered_by: str = Field(..., description="Broker information", min_length=1, max_length=200)
    status: str = Field(..., description="Property status (e.g., 'for_sale')", min_length=1, max_length=50)
    
    @field_validator('bed', 'bath')
    @classmethod
    def validate_rooms(cls, v):
        """Ensure room counts are reasonable."""
        if v < 0:
            raise ValueError("Room count cannot be negative")
        return v
    
    @field_validator('house_size', 'acre_lot')
    @classmethod
    def validate_size(cls, v):
        """Ensure sizes are positive."""
        if v <= 0:
            raise ValueError("Size must be positive")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "bed": 3.0,
                "bath": 2.0,
                "house_size": 2000.0,
                "acre_lot": 0.25,
                "city": "Manchester",
                "state": "New Hampshire",
                "zip_code": "03101",
                "brokered_by": "Keller Williams Realty",
                "status": "for_sale"
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for single prediction."""
    
    predicted_price: float = Field(..., description="Predicted property price in USD")
    currency: str = Field(default="USD", description="Currency of the price")
    model_info: Optional[Dict[str, Any]] = Field(default=None, description="Model metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 325000.50,
                "currency": "USD",
                "model_info": {
                    "model_type": "LinearRegression",
                    "version": "1.0"
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions."""
    
    properties: List[PropertyFeatures] = Field(..., description="List of properties to predict")
    
    class Config:
        json_schema_extra = {
            "example": {
                "properties": [
                    {
                        "bed": 3.0,
                        "bath": 2.0,
                        "house_size": 2000.0,
                        "acre_lot": 0.25,
                        "city": "Manchester",
                        "state": "New Hampshire",
                        "zip_code": "03101",
                        "brokered_by": "Keller Williams Realty",
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
        }


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions made")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {"predicted_price": 325000.50, "currency": "USD"},
                    {"predicted_price": 425000.75, "currency": "USD"}
                ],
                "count": 2
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    timestamp: str = Field(..., description="Current server timestamp")
    version: str = Field(default="1.0.0", description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2026-02-21T10:30:00",
                "version": "1.0.0"
            }
        }


class ModelInfoResponse(BaseModel):
    """Model information response schema."""
    
    model_type: str = Field(..., description="Type of the model")
    training_date: Optional[str] = Field(default=None, description="When the model was trained")
    metrics: Optional[Dict[str, float]] = Field(default=None, description="Model performance metrics")
    features: Optional[Dict[str, List[str]]] = Field(default=None, description="Features used by the model")
    num_samples: Optional[int] = Field(default=None, description="Number of training samples")
    cv_mean: Optional[float] = Field(default=None, description="Cross-validation mean score")
    cv_std: Optional[float] = Field(default=None, description="Cross-validation standard deviation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "LinearRegression",
                "training_date": "2026-02-21T10:00:00",
                "metrics": {
                    "R^2 Score": 0.85,
                    "Mean Squared Error": 12500000.0,
                    "Mean Absolute Error": 25000.0
                },
                "features": {
                    "numerical": ["bed", "bath", "house_size", "acre_lot"],
                    "categorical": ["city", "state", "zip_code", "brokered_by", "status"]
                },
                "num_samples": 50000,
                "cv_mean": 0.84,
                "cv_std": 0.02
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Prediction failed",
                "detail": "Invalid input data format"
            }
        }


class Token(BaseModel):
    """Token response schema."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "token_type": "bearer"
            }
        }


class PredictionExplanationResponse(BaseModel):
    """Prediction explanation response schema."""

    predicted_price: float = Field(..., description="Predicted property price in USD")
    currency: str = Field(default="USD", description="Currency of the price")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    model_type: str = Field(..., description="Type of model used")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 325000.50,
                "currency": "USD",
                "feature_importance": {
                    "house_size": 0.45,
                    "bed": 0.25,
                    "bath": 0.20,
                    "acre_lot": 0.10
                },
                "model_type": "RandomForestRegressor"
            }
        }


class DriftMetricsResponse(BaseModel):
    """Model drift metrics response schema."""

    drift_detected: bool = Field(..., description="Whether data drift was detected")
    drift_score: float = Field(..., description="Drift detection score")
    feature_drift: Dict[str, float] = Field(..., description="Per-feature drift scores")
    timestamp: str = Field(..., description="Timestamp of the check")

    class Config:
        json_schema_extra = {
            "example": {
                "drift_detected": False,
                "drift_score": 0.12,
                "feature_drift": {
                    "house_size": 0.05,
                    "bed": 0.15,
                    "city": 0.20
                },
                "timestamp": "2026-02-21T10:30:00"
            }
        }


class ABRoutingResponse(BaseModel):
    """A/B test routing information response."""

    bucket: str = Field(..., description="A/B test bucket assignment")
    model_version: str = Field(..., description="Model version for this bucket")
    experiment_id: str = Field(..., description="Experiment ID")

    class Config:
        json_schema_extra = {
            "example": {
                "bucket": "A",
                "model_version": "1.0.0",
                "experiment_id": "exp-001"
            }
        }
