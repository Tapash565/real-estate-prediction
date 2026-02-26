"""Enhanced FastAPI application for real estate price prediction."""

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import sentry_sdk
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.auth import (
    TokenData,
    User,
    authenticate_user,
    create_access_token,
    get_api_key,
    get_current_active_user,
    get_current_user_from_token,
    require_read,
)
from api.caching import PredictionCache, cache
from api.logging_config import configure_logging, get_logger
from api.middleware import (
    CORSMiddlewareConfig,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    TimeoutMiddleware,
)
from api.monitoring import (
    MetricsEndpoint,
    PrometheusMiddleware,
    health_checker,
    instrument_fastapi,
    register_health_check,
    setup_tracing,
    span,
    track_prediction,
    update_model_metrics,
)
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionExplanationResponse,
    PredictionResponse,
    PropertyFeatures,
    Token,
)
from src.model_utils import load_model_bundle

# Configure logging
configure_logging(
    log_level=os.getenv("LOG_LEVEL", "WARNING"),
    json_format=False,
)
logger = get_logger(__name__)

# =============================================================================
# SENTRY CONFIGURATION
# =============================================================================
if sentry_dsn := os.getenv("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.1")),
        environment=os.getenv("ENVIRONMENT", "development"),
        release=os.getenv("APP_VERSION", "1.0.0"),
    )
    # logger.info("sentry_initialized")

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = os.path.join("models", "latest_model.joblib")
PIPELINE_PATH = os.path.join("models", "preprocessing_pipeline.joblib")
# API_PREFIX = "/api"  # Not needed with APIRouter

# Global variables for model and pipeline
model = None
preprocessing_pipeline = None
model_metadata: Optional[dict[str, Any]] = None
model_type: str = "unknown"

# =============================================================================
# RATE LIMITING
# =============================================================================
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"],
    storage_uri=os.getenv("REDIS_URL", "memory://"),
)

# =============================================================================
# HEALTH CHECKS
# =============================================================================
@register_health_check("model")
def check_model_health() -> tuple[bool, str]:
    """Check if model is loaded."""
    if model is not None and preprocessing_pipeline is not None:
        return True, f"Model loaded ({model_type})"
    return False, "Model not loaded"


@register_health_check("cache")
def check_cache_health() -> tuple[bool, str]:
    """Check if Redis cache is connected."""
    if cache.is_connected():
        stats = cache.get_stats()
        return True, f"Redis connected ({stats.get('used_memory', 'N/A')} used)"
    return True, "Redis not connected (cache disabled)"


# =============================================================================
# LIFESPAN EVENTS
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    global model, preprocessing_pipeline, model_metadata, model_type

    # logger.info("api_startup")

    # Connect to Redis cache
    cache.connect()

    # Setup tracing
    setup_tracing()

    # Load model
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(PIPELINE_PATH):
            model, preprocessing_pipeline, model_metadata = load_model_bundle(
                MODEL_PATH, PIPELINE_PATH
            )
            model_type = model_metadata.get("model_type", "unknown")
            update_model_metrics(model_type, loaded=True)
            # logger.info(
            #     "model_loaded",
            #     model_type=model_type,
            #     r2_score=model_metadata.get("metrics", {}).get("R^2 Score", "N/A"),
            # )
        else:
            # logger.warning("model_files_not_found")
            update_model_metrics("none", loaded=False)
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
        update_model_metrics("error", loaded=False)

    yield

    # Shutdown
    # logger.info("api_shutdown")


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="Real Estate Price Prediction API",
    description="API for predicting real estate prices using machine learning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Create API router for /api prefix
api_router = APIRouter(prefix="/api")

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middlewares
# app.add_middleware(RequestLoggingMiddleware)  # Disabled to prevent console logging
app.add_middleware(PrometheusMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(TimeoutMiddleware, timeout_seconds=60.0)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORSMiddlewareConfig.ALLOW_ORIGINS,
    allow_credentials=CORSMiddlewareConfig.ALLOW_CREDENTIALS,
    allow_methods=CORSMiddlewareConfig.ALLOW_METHODS,
    allow_headers=CORSMiddlewareConfig.ALLOW_HEADERS,
)

# Instrument with OpenTelemetry
instrument_fastapi(app)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def check_model_loaded() -> None:
    """Check if the model is loaded and raise an error if not."""
    if model is None or preprocessing_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first.",
        )


def prepare_input_dataframe(property_features: PropertyFeatures) -> pd.DataFrame:
    """Convert PropertyFeatures to DataFrame for prediction."""
    data = {
        "bed": [property_features.bed],
        "bath": [property_features.bath],
        "house_size": [property_features.house_size],
        "acre_lot": [property_features.acre_lot],
        "city": [property_features.city],
        "state": [property_features.state],
        "zip_code": [property_features.zip_code],
        "brokered_by": [property_features.brokered_by],
        "status": [property_features.status],
    }
    return pd.DataFrame(data)


def features_to_dict(features: PropertyFeatures) -> dict[str, Any]:
    """Convert PropertyFeatures to dictionary."""
    return {
        "bed": features.bed,
        "bath": features.bath,
        "house_size": features.house_size,
        "acre_lot": features.acre_lot,
        "city": features.city,
        "state": features.state,
        "zip_code": features.zip_code,
        "brokered_by": features.brokered_by,
        "status": features.status,
    }


# =============================================================================
# PUBLIC ENDPOINTS (no auth required)
# =============================================================================
@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with API information."""
    return {
        "message": "Welcome to the Real Estate Price Prediction API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Enhanced health check endpoint with dependency status."""
    checks = health_checker.run_checks()

    return HealthResponse(
        status=checks["overall_status"],
        model_loaded=model is not None and preprocessing_pipeline is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        details=checks,
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return await MetricsEndpoint.metrics()


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================
@api_router.post("/auth/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(
    username: str = Query(..., description="Username"),
    password: str = Query(..., description="Password"),
):
    """Get JWT access token for API authentication."""
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.username, "scopes": user.scopes})
    return {"access_token": access_token, "token_type": "bearer"}


# =============================================================================
# PROTECTED ENDPOINTS (auth required)
# =============================================================================
@api_router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    dependencies=[Depends(require_read)],
)
async def get_model_info():
    """Get information about the loaded model."""
    check_model_loaded()

    return ModelInfoResponse(
        model_type=model_metadata.get("model_type", "Unknown"),
        training_date=model_metadata.get("saved_at", None),
        metrics=model_metadata.get("metrics", None),
        features=model_metadata.get("features", None),
        num_samples=model_metadata.get("num_samples", None),
        cv_mean=model_metadata.get("cv_mean", None),
        cv_std=model_metadata.get("cv_std", None),
    )


@api_router.get(
    "/model/features",
    tags=["Model"],
    dependencies=[Depends(require_read)],
)
async def get_model_features():
    """Get the list of features used by the model."""
    check_model_loaded()

    features = model_metadata.get("features", {})

    return {
        "numerical_features": features.get("numerical", []),
        "categorical_features": features.get("categorical", []),
        "total_features": len(features.get("numerical", [])) + len(features.get("categorical", [])),
    }


@api_router.get(
    "/model/metrics",
    tags=["Model"],
    dependencies=[Depends(require_read)],
)
async def get_model_metrics():
    """Get detailed performance metrics of the model."""
    check_model_loaded()

    metrics = model_metadata.get("metrics", {})
    cv_scores = model_metadata.get("cv_scores", [])

    return {
        "evaluation_metrics": metrics,
        "cross_validation": {
            "mean_score": model_metadata.get("cv_mean", None),
            "std_score": model_metadata.get("cv_std", None),
            "fold_scores": cv_scores,
        },
    }


@api_router.get(
    "/model/categories",
    tags=["Model"],
    dependencies=[Depends(require_read)],
)
async def get_model_categories():
    """Get unique categories for each categorical feature."""
    check_model_loaded()

    try:
        # Get the ColumnTransformer
        preprocessor = preprocessing_pipeline.named_steps["preprocessor"]
        cat_transformer = preprocessor.named_transformers_["cat"]
        ohe = cat_transformer.named_steps["onehot"]
        categories = ohe.categories_
        features = model_metadata.get("features", {}).get("categorical", [])

        result = {}
        for i, feature in enumerate(features):
            if i < len(categories):
                result[feature] = categories[i].tolist()

        # Add cities from dataset
        if "city" not in result:
            DATA_PATH = os.path.join("data", "raw", "realtor-data.csv")
            if os.path.exists(DATA_PATH):
                df_temp = pd.read_csv(DATA_PATH, usecols=["city"])
                result["city"] = sorted(df_temp["city"].dropna().unique().tolist())

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract categories: {e!s}",
        )


@api_router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    dependencies=[Depends(require_read)],
)
@limiter.limit("60/minute")
async def predict_price(
    request: Request,
    property_features: PropertyFeatures,
    use_cache: bool = Query(True, description="Use cached prediction if available"),
    current_user: User = Depends(get_current_active_user),
):
    """
    Predict the price of a single property.

    Provide property features including:
    - Number of bedrooms and bathrooms
    - House size in square feet
    - Lot size in acres
    - Location information (city, state, zip code)
    - Broker and status information
    """
    check_model_loaded()
    start_time = datetime.now()

    # Convert features to dict for caching
    features_dict = features_to_dict(property_features)

    # Check cache
    if use_cache:
        cached_result = PredictionCache.get(features_dict)
        if cached_result:
            latency = (datetime.now() - start_time).total_seconds()
            track_prediction(
                model_version=model_metadata.get("model_type", "unknown"),
                prediction_value=cached_result["predicted_price"],
                latency=latency,
                cached=True,
            )
            # logger.info(
            #     "prediction_cached",
            #     user=current_user.username,
            #     price=cached_result["predicted_price"],
            # )
            return PredictionResponse(**cached_result)

    try:
        with span("prediction", {"model_type": model_type}) as s:
            # Prepare input data
            input_df = prepare_input_dataframe(property_features)

            # Preprocess
            X_processed = preprocessing_pipeline.transform(input_df)

            # Make prediction
            prediction = model.predict(X_processed)[0]
            prediction = max(0, float(prediction))

            s.set_attribute("prediction_value", prediction)

        latency = (datetime.now() - start_time).total_seconds()

        result = {
            "predicted_price": round(prediction, 2),
            "currency": "USD",
            "model_info": {
                "model_type": model_metadata.get("model_type", "Unknown"),
                "version": "1.0",
            },
        }

        # Cache result
        PredictionCache.set(features_dict, result)

        # Track metrics
        track_prediction(
            model_version=model_metadata.get("model_type", "unknown"),
            prediction_value=prediction,
            latency=latency,
            cached=False,
        )

        # logger.info(
        #     "prediction_made",
        #     user=current_user.username,
        #     price=prediction,
        #     latency=latency,
        # )

        return PredictionResponse(**result)

    except Exception as e:
        track_prediction(
            model_version=model_metadata.get("model_type", "unknown"),
            prediction_value=0,
            latency=(datetime.now() - start_time).total_seconds(),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e!s}",
        )


@api_router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    dependencies=[Depends(require_read)],
)
@limiter.limit("10/minute")
async def predict_batch(
    request: Request,
    batch_request: BatchPredictionRequest,
    current_user: User = Depends(get_current_active_user),
):
    """
    Predict prices for multiple properties at once.

    Provide a list of property features to get predictions for all of them.
    """
    check_model_loaded()
    start_time = datetime.now()

    if len(batch_request.properties) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size cannot exceed 100 properties",
        )

    try:
        predictions = []

        for property_features in batch_request.properties:
            input_df = prepare_input_dataframe(property_features)
            X_processed = preprocessing_pipeline.transform(input_df)
            prediction = model.predict(X_processed)[0]
            prediction = max(0, float(prediction))

            predictions.append(
                PredictionResponse(
                    predicted_price=round(prediction, 2),
                    currency="USD",
                )
            )

        latency = (datetime.now() - start_time).total_seconds()

        # logger.info(
        #     "batch_prediction_made",
        #     user=current_user.username,
        #     count=len(predictions),
        #     latency=latency,
        # )

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {e!s}",
        )


@api_router.post(
    "/predict/explain",
    response_model=PredictionExplanationResponse,
    tags=["Prediction"],
    dependencies=[Depends(require_read)],
)
async def explain_prediction(
    property_features: PropertyFeatures,
    current_user: User = Depends(get_current_active_user),
):
    """
    Explain a prediction with feature importance.

    Returns the predicted price along with feature contributions.
    """
    check_model_loaded()

    try:
        # Prepare input data
        input_df = prepare_input_dataframe(property_features)

        # Get feature importance if available
        feature_importance = {}
        if hasattr(model, "feature_importances_"):
            # Tree-based models
            importances = model.feature_importances_
            feature_names = model_metadata.get("features", {}).get("all", [])
            for name, importance in zip(feature_names, importances):
                feature_importance[name] = round(importance, 4)
        elif hasattr(model, "coef_"):
            # Linear models
            coefs = np.abs(model.coef_)
            feature_names = model_metadata.get("features", {}).get("all", [])
            total = np.sum(coefs)
            for name, coef in zip(feature_names, coefs):
                feature_importance[name] = round(coef / total, 4) if total > 0 else 0

        # Preprocess
        X_processed = preprocessing_pipeline.transform(input_df)

        # Make prediction
        prediction = model.predict(X_processed)[0]
        prediction = max(0, float(prediction))

        # logger.info(
        #     "prediction_explained",
        #     user=current_user.username,
        #     price=prediction,
        # )

        return PredictionExplanationResponse(
            predicted_price=round(prediction, 2),
            currency="USD",
            feature_importance=feature_importance,
            model_type=model_metadata.get("model_type", "Unknown"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain prediction: {e!s}",
        )


# =============================================================================
# INCLUDE ROUTERS
# =============================================================================
app.include_router(api_router)


# =============================================================================
# ERROR HANDLERS
# =============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    # logger.warning(
    #     "http_exception",
    #     path=request.url.path,
    #     status_code=exc.status_code,
    #     detail=exc.detail,
    # )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors."""
    logger.error(
        "unexpected_error",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
