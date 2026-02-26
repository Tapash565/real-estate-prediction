# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready real estate price prediction system using machine learning with scikit-learn, FastAPI for the REST API, and React + TypeScript for the frontend. The project follows a modular architecture with comprehensive testing and Docker deployment support.

## Common Commands

### Development Setup
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running the Application
```bash
# Full ML pipeline (train, evaluate, visualize)
python run_all.py

# Start FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Start React frontend
cd frontend && npm run dev

# Run Jupyter notebook
jupyter notebook
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov=api --cov-report=html

# Run specific test file
pytest tests/test_train.py -v

# Run tests matching pattern
pytest tests/ -k "test_model" -v
```

### Code Quality
```bash
# Lint with Ruff
ruff check src/ api/ tests/

# Auto-fix lint issues
ruff check --fix src/ api/ tests/

# Format code
ruff format src/ api/ tests/

# Type checking
mypy src/ api/

# Run all checks
make check-all
```

### Docker
```bash
# Build Docker image
docker build -t real-estate-prediction:latest .

# Run with Docker Compose
docker-compose up -d api

# Run training in Docker
docker-compose up train
```

## Architecture

### Tech Stack
- **Backend**: FastAPI (Python 3.13+)
- **ML Framework**: scikit-learn with hyperparameter tuning
- **Frontend**: React 18 + TypeScript + Vite
- **Database**: CSV-based (raw data), supports PostgreSQL for production
- **Caching**: Redis
- **MLOps**: MLflow for experiment tracking

### Directory Structure
```
real-estate-prediction/
├── api/                    # FastAPI REST API
│   ├── main.py            # App entry point, route definitions
│   ├── schemas.py         # Pydantic request/response models
│   ├── auth.py            # Authentication (API keys, JWT)
│   ├── caching.py         # Redis prediction caching
│   ├── middleware.py      # Rate limiting, CORS, logging
│   ├── monitoring.py      # Prometheus metrics, Sentry
│   └── logging_config.py  # Logging configuration
├── src/                   # ML pipeline modules
│   ├── config.py          # YAML config + env var overrides
│   ├── data_loader.py     # CSV data loading
│   ├── preprocessing.py   # Feature transformations
│   ├── pipeline.py        # sklearn Pipeline construction
│   ├── train.py           # Model training with cross-validation
│   ├── inference.py       # ModelPredictor for production predictions
│   ├── evaluate.py        # R², RMSE, MAE metrics
│   ├── visualize.py       # Matplotlib/Seaborn plots
│   ├── model_utils.py     # Model persistence (joblib)
│   ├── mlflow_tracking.py # Experiment tracking
│   ├── model_monitoring.py# Drift detection
│   ├── retraining_pipeline.py
│   └── ab_testing.py
├── frontend/              # React + TypeScript app
│   ├── src/
│   │   ├── App.tsx
│   │   └── components/
│   └── package.json
├── tests/                 # pytest test suite (90%+ coverage)
├── models/                # Saved model files (*.joblib)
├── data/                  # Raw and processed data
├── docs/                  # Architecture documentation
├── notebooks/             # Jupyter notebooks (EDA)
├── reports/figures/       # Generated visualizations
├── config.yaml            # Main configuration
├── run_all.py            # Main pipeline script
├── Makefile              # Build automation
└── Dockerfile            # Docker image
```

### Configuration
All settings are in `config.yaml`, overridable via environment variables:
- `REAL_ESTATE_DATA_PATH` - Dataset location
- `REAL_ESTATE_MODEL_TYPE` - Algorithm (linear, sgd, decision_tree, random_forest, gradient_boosting)
- `REAL_ESTATE_N_SPLITS` - Cross-validation folds
- `REAL_ESTATE_LOG_LEVEL` - Logging level
- `REAL_ESTATE_MODEL_PATH` / `REAL_ESTATE_PIPELINE_PATH` - Model files for inference

### Supported Models
- Linear Regression (baseline)
- SGD Regressor (large datasets)
- Decision Tree
- Random Forest (recommended default)
- Gradient Boosting

### API Endpoints
- `GET /health` - Health check
- `GET /model/info` - Model metadata
- `GET /model/features` - Feature list
- `GET /model/metrics` - Performance metrics
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

## Key Design Patterns

1. **Configuration-driven**: All settings in `config.yaml`, no hardcoded values
2. **Pipeline pattern**: sklearn Pipeline for reproducible preprocessing + model
3. **Type hints**: Full typing for IDE support and error detection
4. **Dependency injection**: Config objects passed as parameters, no global state
5. **Test fixtures**: Shared test data in `tests/conftest.py`
6. **Structured logging**: Use `logging` module, not `print()`

## Data Flow

1. **Training**: `run_all.py` → `data_loader.py` → `preprocessing.py` → `train.py` (CV) → `model_utils.py` (save)
2. **Inference**: `api/main.py` → `inference.py` (load model + pipeline) → prediction → response
3. **Caching**: Redis stores predictions by input hash to avoid redundant computation

## Files Not Tracked in Git

The following are excluded via `.gitignore`:
- `venv/` - Python virtual environment
- `models/*.joblib` - Trained model files (large binaries)
- `data/processed/*.csv` - Processed data files
- `reports/figures/*.png` - Generated visualizations
- `.env` - Environment variables (sensitive)
- `htmlcov/` - Coverage reports
- `.pytest_cache/` - Test cache
- `mlruns/` - MLflow experiment data
- `frontend/node_modules/` - NPM dependencies
- `frontend/dist/` - Built frontend assets