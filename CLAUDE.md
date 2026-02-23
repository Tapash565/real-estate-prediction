# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**Real Estate Price Prediction API** - A production-ready machine learning platform for predicting real estate prices.

### Project Structure

```
D:\projects\real-estate-prediction/
├── api/                          # FastAPI backend application
│   ├── __init__.py
│   ├── auth.py                   # JWT/API key authentication
│   ├── caching.py                # Redis caching layer
│   ├── logging_config.py         # Structured logging with structlog
│   ├── main.py                   # Enhanced FastAPI app with monitoring
│   ├── middleware.py             # Request logging, CORS, security headers
│   ├── monitoring.py             # Prometheus metrics and OpenTelemetry tracing
│   └── schemas.py                # Pydantic request/response models
├── src/                          # ML pipeline source code
│   ├── __init__.py
│   ├── ab_testing.py             # A/B testing framework
│   ├── config.py                 # Configuration management
│   ├── data_loader.py            # Data loading utilities
│   ├── evaluate.py               # Model evaluation
│   ├── inference.py              # Prediction inference
│   ├── mlflow_tracking.py        # MLflow experiment tracking
│   ├── model_monitoring.py       # Model drift detection
│   ├── model_utils.py            # Model utilities
│   ├── pipeline.py               # ML training pipeline
│   ├── preprocessing.py          # Data preprocessing
│   ├── retraining_pipeline.py   # Automated retraining
│   ├── train.py                  # Model training with CV
│   └── visualize.py              # Visualization utilities
├── frontend/                     # React + TypeScript + Vite frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── ErrorBoundary.tsx    # React error boundaries
│   │   │   ├── FormValidation.tsx   # Form validation with user feedback
│   │   │   ├── SearchableDropdown.tsx
│   │   │   └── Skeleton.tsx         # Loading skeletons
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── vite-env.d.ts
│   ├── package.json
│   └── vite.config.ts
├── tests/                        # Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_auth.py
│   ├── test_caching.py
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_evaluate.py
│   ├── test_inference.py
│   ├── test_model_utils.py
│   ├── test_monitoring.py
│   ├── test_pipeline.py
│   ├── test_preprocessing.py
│   ├── test_train.py
│   └── test_visualize.py
├── data/                         # Data directory
│   ├── raw/realtor-data.csv
│   └── processed/
├── models/                       # Saved model artifacts
├── notebooks/                    # Jupyter notebooks for EDA
├── reports/                      # Analysis reports
├── .github/workflows/            # GitHub Actions CI/CD
├── .pre-commit-config.yaml       # Pre-commit hooks
├── pyproject.toml                # Project configuration
├── requirements.txt              # Python dependencies
├── Makefile                      # Development shortcuts
├── docker-compose.yml            # Docker services
├── Dockerfile                    # Container definition
├── config.yaml                   # Application config
└── README.md                     # Project documentation
```

## Architecture

### Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI with Pydantic v2 |
| ML/DS | scikit-learn, pandas, numpy |
| Frontend | React 18, TypeScript, Vite |
| Cache | Redis |
| Monitoring | Prometheus, OpenTelemetry, Sentry |
| CI/CD | GitHub Actions |
| Containerization | Docker, Docker Compose |

### Key Features Implemented

1. **Code Quality**
   - Ruff for linting and formatting
   - mypy for type checking
   - pre-commit hooks
   - Comprehensive test coverage

2. **API Enhancements**
   - Rate limiting with slowapi (100/min default)
   - JWT and API key authentication
   - Redis caching layer
   - API versioning (/api/v1/)
   - Structured logging with request IDs
   - Prometheus metrics endpoint
   - OpenTelemetry distributed tracing
   - Sentry error tracking

3. **ML Infrastructure**
   - MLflow for experiment tracking
   - A/B testing framework
   - Model drift detection
   - Automated retraining pipeline
   - Feature importance explanations

4. **Security**
   - JWT token authentication
   - API key support
   - Security headers middleware
   - CORS configuration
   - Request timeout handling

5. **Monitoring & Observability**
   - Prometheus metrics
   - Distributed tracing
   - Structured JSON logging
   - Health checks with dependency status
   - Sentry integration

6. **Frontend**
   - React Error Boundaries
   - Loading skeletons
   - Form validation with user feedback

## Common Commands

### Development

```bash
# Setup
make install-dev          # Install all dependencies
make pre-commit-install   # Install git hooks

# Code Quality
make lint                 # Run Ruff linter
make format               # Format code
make type-check           # Run mypy
make check-all            # Run all checks

# Testing
make test                 # Run tests
make test-cov             # Run with coverage
make test-fast            # Skip slow tests

# Run Applications
make run-api              # Start FastAPI server
make run-frontend         # Start React dev server
make run-all              # Run full ML pipeline

# Docker
make docker-build         # Build Docker image
make docker-compose-up    # Start all services
make docker-compose-down  # Stop services
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Get token
curl -X POST "http://localhost:8000/api/v1/auth/token?username=admin&password=admin123"

# Make prediction (authenticated)
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "bed": 3,
    "bath": 2,
    "house_size": 2000,
    "acre_lot": 0.25,
    "city": "Manchester",
    "state": "New Hampshire",
    "zip_code": "03101",
    "brokered_by": "Test Broker",
    "status": "for_sale"
  }'

# Metrics endpoint
curl http://localhost:8000/metrics
```

### Environment Variables

```bash
# Required
LOG_LEVEL=INFO
LOG_FORMAT=json

# Optional
REDIS_HOST=localhost
REDIS_PORT=6379
SENTRY_DSN=your-sentry-dsn
MLFLOW_TRACKING_URI=./mlruns
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Auth
JWT_SECRET_KEY=your-secret-key
```

## API Endpoints

### Public (No Auth)
- `GET /` - API info
- `GET /health` - Health check with dependency status
- `GET /metrics` - Prometheus metrics

### Authentication
- `POST /api/v1/auth/token` - Get JWT token

### Protected (Auth Required)
- `POST /api/v1/predict` - Single prediction (60/min)
- `POST /api/v1/predict/batch` - Batch predictions (10/min)
- `POST /api/v1/predict/explain` - Prediction with feature importance
- `GET /api/v1/model/info` - Model metadata
- `GET /api/v1/model/features` - Feature list
- `GET /api/v1/model/metrics` - Performance metrics
- `GET /api/v1/model/categories` - Categorical values

## Documentation

- API docs: `http://localhost:8000/api/v1/docs` (Swagger UI)
- ReDoc: `http://localhost:8000/api/v1/redoc`
- OpenAPI spec: `http://localhost:8000/api/v1/openapi.json`

## Development Workflow

1. Activate virtual environment: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
2. Make changes
3. Run checks: `make check-all`
4. Commit (pre-commit hooks run automatically)
5. Push (CI/CD runs tests automatically)

## Notes

- All API endpoints are versioned under `/api/v1/`
- Rate limits: 100/min for predictions, 10/min for batch
- Authentication via Bearer token or X-API-Key header
- Redis caching improves prediction response times
- MLflow tracking requires MLflow server running for full features
