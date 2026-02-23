# Real Estate Price Prediction

A production-ready machine learning project for predicting real estate prices using scikit-learn, featuring a complete data science pipeline, REST API, Docker deployment, and comprehensive testing.

## 🎯 Features

- **Production-Ready Pipeline**: Configuration-driven, modular architecture with logging
- **Multiple Model Support**: Linear, SGD, Decision Tree, Random Forest, Gradient Boosting
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV integration
- **Cross-Validation**: K-Fold cross-validation with comprehensive metrics
- **REST API**: FastAPI-based prediction service with interactive documentation
- **Docker Support**: Multi-stage Dockerfile and docker-compose configuration
- **Configuration Management**: YAML-based config with environment variable overrides
- **Model Inference**: Dedicated inference module for production predictions
- **Visualization Tools**: Comprehensive plots for model performance analysis
- **Model Persistence**: Save and load trained models with preprocessing pipelines
- **Comprehensive Testing**: Full test coverage with pytest (90%+ coverage)
- **Type Hints**: Fully typed codebase for better IDE support and maintainability
- **Structured Logging**: Replace print statements with proper logging
- **MLflow Integration**: Optional experiment tracking and model registry

## 📁 Project Structure

```
real-estate-prediction/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   └── schemas.py           # Pydantic models for request/response validation
├── data/
│   ├── raw/
│   │   └── realtor-data.csv # Raw dataset
│   └── processed/           # Preprocessed data files
├── models/                  # Saved trained models and pipelines
├── notebooks/
│   └── eda.ipynb           # Exploratory data analysis
├── reports/
│   └── figures/            # Generated visualizations
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessing.py    # Data preprocessing functions
│   ├── pipeline.py         # sklearn Pipeline construction
│   ├── train.py            # Model training with cross-validation and tuning
│   ├── inference.py        # Model loading and prediction
│   ├── evaluate.py         # Model evaluation metrics
│   ├── visualize.py        # Visualization functions
│   └── model_utils.py      # Model persistence utilities
├── tests/                  # Comprehensive test suite
│   ├── conftest.py         # Pytest fixtures
│   ├── test_config.py      # Configuration tests
│   ├── test_pipeline.py    # Pipeline tests
│   ├── test_inference.py   # Inference tests
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_visualize.py
│   ├── test_model_utils.py
│   └── test_api.py
├── .dockerignore           # Docker build exclusions
├── .gitignore
├── config.yaml             # Main configuration file
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker Compose services
├── requirements.txt        # Python dependencies
├── run_all.py             # Main pipeline script
├── CLAUDE.md              # AI assistant guidance
├── QUICKSTART.md          # Quick start guide
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd real-estate-prediction
   ```

2. **Create a virtual environment**
   ```bash
   # W

4. **Configure the pipeline** (optional)
   Edit `config.yaml` to customize:
   - Data paths and features
   - Model type and hyperparameters
   - Cross-validation settings
   - Output paths
   
   Or use environment variables:
   ```bash
   export REAL_ESTATE_MODEL_TYPE=random_forest
   export REAL_ESTATE_N_SPLITS=10
   ```

## ⚙️ Configuration

The pipeline is driven by `config.yaml`. Key settings:

```yaml
# Data Settings
data_path: "data/raw/realtor-data.csv"
target_column: "price"
numerical_features: ["bed", "bath", "house_size", "acre_lot"]
categorical_features: ["state", "status"]
sample_size: 100000  # null for all data

# Model Configuration
model_type: "random_forest"  # linear, sgd, decision_tree, random_forest, gradient_boosting
n_splits: 5
random_state: 42

# Hyperparameter Tuning (optional)
hyperparameter_grid:
  random_forest:
    n_estimators: [50, 100, 200]
    max_depth: [10, 20, null]
    min_samples_split: [2, 5]

# MLflow (optional)
mlflow:
  enabled: false
  tracking_uri: "file:./mlruns"
  experiment_name: "real_estate_prediction"
```

**Environment Variable Overrides:**
- `REAL_ESTATE_DATA_PATH` - Override data path
- `REAL_ESTATE_MODEL_TYPE` - Override model type
- `REAL_ESTATE_N_SPLITS` - Override CV splits
- `REAL_ESTATE_LOG_LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)indows
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Information

The project uses the `realtor-data.csv` dataset with the following features:

**Numerical Features:**
- `bed` - Number of bedrooms
- `bath` - Number of bathrooms
- `house_size` - Square footage of the house
- `acre_lot` - Lot size in acres

**Categorical Features:**
- `city` - City location
- `state` - State location
- `zip_code` - ZIP code
- `brokered_by` - Real estate broker
- `status` - Property status (e.g., for_sale, sold)

**Target Variable:**
- `price` - Property price in USD

## 🎓 Usage

### Option 1: Local Development

#### 1. Train the Model

Run the complete ML pipeline to train a model, generate visualizations, and save artifacts:

```bash
python run_all.py
```

This will:
- Load and clean the data based on `config.yaml`
- Preprocess features (imputation, scaling, encoding)
- Train model with K-fold cross-validation
- Optionally perform hyperparameter tuning
- Evaluate with R², RMSE, MAE metrics
- Generate visualization plots
- Save model and preprocessing pipeline to `models/`
- Save processed data to `data/processed/`
- Create performance visualizations in `reports/figures/`

**Expected Output:**
```
======================================================================
REAL ESTATE PRICE PREDICTION PIPELINE
======================================================================

[1/7] Loading data...
   ✓ Loaded 50000+ records with 12 columns
   ✓ Cleaned data: 48532 records remain

[2/7] Preprocessing data...
   ✓ Preprocessed shape: (48532, 156)

[3/7] Training random_forest model with 5‑fold cross-validation...
   INFO: Performing hyperparameter tuning using grid search
   Fold 1/5 - R²: 0.8534, RMSE: 45231.23, MAE: 32145.67
   ...
   Cross-validation results:
     R² = 0.8534 ± 0.0123
     RMSE = 45231.23
     MAE = 32145.67

[4/7] Training final model on full dataset
   ✓ Model trained successfully

[5/7] Evaluating model...
   ✓ Test R²: 0.8542

[6/7] Generating visualizations...
   ✓ Created 5 visualization plots

[7/7] Saving artifacts...
   ✓ Model saved: models/latest_model.joblib
   ✓ Pipeline saved: models/preprocessing_pipeline.joblib

Pipeline completed successfully!
```

#### 2. Make Predictions (Programmatic)

Use the inference module to load models and make predictions:

```python
from src.inference import ModelPredictor
import pandas as pd

# Load the trained model
predictor = ModelPredictor(
    model_path="models/latest_model.joblib",
    pipeline_path="models/preprocessing_pipeline.joblib"
)
predictor.load()

# Prepare input data
data = pd.DataFrame([{
    'bed': 3.0,
    'bath': 2.0,
    'house_size': 2000.0,
    'acre_lot': 0.25,
    'state': 'New Hampshire',
    'status': 'for_sale'
}])

# Make prediction
predictions = predictor.predict(data)
print(f"Predicted price: ${predictions[0]:,.2f}")

# Or use the convenience function
from src.inference import predict
predictions = predict(data)
```

#### 3. Start the API Server

Launch the FastAPI server for making predictions:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Option 2: Docker Deployment

#### Quick Start with Docker Compose

```bash
# Run training
docker-compose up train

# Start API server
docker-compose up -d api

# Run tests
docker-compose up test

# Development with Jupyter
docker-compose up dev
```

#### Manual Docker Usage

```bash
# Build the image
docker build -t real-estate-prediction:latest .

# Run training
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           real-estate-prediction:latest

# Run API server
docker run -p 8000:8000 \
           -v $(pwd)/models:/app/models:ro \
           real-estate-prediction:latest \
           uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run with custom config
docker run -v $(pwd)/config.yaml:/app/config.yaml \
           real-estate-prediction:latest
```

### 3. API Endpoints

#### Health Check
```bash
GET http://localhost:8000/health
```

#### Get Model Information
```bash
GET http://localhost:8000/model/info
```

#### Single Prediction
```bash
POST http://localhost:8000/predict
Content-Type: application/json

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
}
```

**Response:**
```json
{
  "predicted_price": 325000.50,
  "currency": "USD",
  "model_info": {
    "model_type": "LinearRegression",
    "version": "1.0"
  }
}
```

#### Batch Predictions
```bash
POST http://localhost:8000/predict/batch
Content-Type: application/json

{
  "properties": [
    { /* property 1 features */ },
    { /* property 2 features */ }
  ]
}
```

#### Get Model Features
```bash
GET http://localhost:8000/model/features
```

#### Get Model Metrics
```bash
GET http://localhost:8000/model/metrics
```

### 4. Run Tests

Execute the full test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov=api --cov-report=html

# Run specific test file
pytest tests/test_train.py -v

# Run tests matching a pattern
pytest tests/ -k "test_model" -v
```

**Expected Test Output:**
```
tests/test_data_loader.py::test_load_data_success PASSED
tests/test_preprocessing.py::test_preprocess_data_output_shape PASSED
tests/test_train.py::test_train_model_returns_model PASSED
...
======================== XX passed in X.XXs ========================
```

### 5. Interactive API Testing

Visit http://localhost:8000/docs for the Swagger UI interface where you can:
- View all available endpoints
- Test API calls interactively
- See request/response schemas
- View example payloads

## 🔧 Model Details

### Supported Algorithms

The pipeline supports multiple regression algorithms, configurable via `config.yaml`:

1. **Linear Regression** (`linear`)
   - Fast training and prediction
   - Interpretable coefficients
   - Good baseline model
   - No hyperparameters to tune

2. **SGD Regressor** (`sgd`)
   - Efficient for large datasets (100k+ samples)
   - Online learning capable
   - Configurable regularization (L1, L2, ElasticNet)
   - Suitable for streaming data

3. **Decision Tree** (`decision_tree`)
   - Captures non-linear relationships
   - Feature importance available
   - No feature scaling required
   - Tunable: max_depth, min_samples_split, min_samples_leaf

4. **Random Forest** (`random_forest`) - **Recommended**
   - Ensemble of decision trees
   - Robust to overfitting
   - Handles non-linearity well
   - Tunable: n_estimators, max_depth, min_samples_split

5. **Gradient Boosting** (`gradient_boosting`)
   - State-of-the-art performance
   - Sequential ensemble learning
   - Handles complex patterns
   - Tunable: n_estimators, learning_rate, max_depth

### Hyperparameter Tuning

Configure hyperparameter search in `config.yaml`:

```yaml
hyperparameter_grid:
  random_forest:
    n_estimators: [50, 100, 200]
    max_depth: [10, 20, null]
    min_samples_split: [2, 5]
    min_samples_leaf: [1, 2]
```

The pipeline automatically:
- Uses GridSearchCV or RandomizedSearchCV
- Performs cross-validated hyperparameter search
- Selects best parameters based on R² score
- Trains final model with optimal parameters
- Logs best parameters and performance

### Evaluation Metrics

The pipeline reports comprehensive metrics:

- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **RMSE**: Root Mean Squared Error in price units (lower is better)
- **MAE**: Mean Absolute Error in price units (lower is better)
- **Cross-Validation Scores**: Per-fold R² scores with mean and std
- **Sample Count**: Training samples and features used

### Preprocessing Pipeline

The `src/pipeline.py` module builds sklearn Pipeline objects:

1. **ColumnTransformer**: Separate handling for numerical and categorical features
   
2. **Numerical Pipeline**:
   - SimpleImputer (median strategy)
   - StandardScaler (zero mean, unit variance)
   
3. **Categorical Pipeline**:
   - SimpleImputer (most frequent strategy)
   - OneHotEncoder (handles unknown categories gracefully)

4. **Full Pipeline**:
   - Preprocessing + Model in single Pipeline object
   - Ensures consistent transformations
   - Simplifies deployment and inference

### Model Persistence

Models are saved with their preprocessing pipelines:

```python
from src.model_utils import save_model_bundle, load_model_bundle

# Save
save_model_bundle(
    model,
    preprocessor, 
    "models/latest_model.joblib",
    "models/preprocessing_pipeline.joblib",
    metadata={"r2_score": 0.85, "trained_on": "2026-02-21"}
)

# Load
model, preprocessor = load_model_bundle(
    "models/latest_model.joblib",
    "models/preprocessing_pipeline.joblib"
)
```

## 📈 Visualizations

The pipeline generates the following visualizations in `reports/figures/`:

1. **predictions_vs_actual.png** - Scatter plot comparing predictions to actual values
2. **residuals.png** - Residual plot to check model assumptions
3. **error_distribution.png** - Histogram of prediction errors
4. **cv_scores.png** - Cross-validation performance across folds
5. **feature_importance.png** - Top features by importance/coefficients

## 🧪 Testing

The project includes comprehensive test coverage (90%+):

### Test Organization

- **test_config.py**: Configuration loading, YAML parsing, environment variables
- **test_pipeline.py**: Pipeline construction, preprocessing transformations
- **test_inference.py**: Model loading, predictions, batch processing
- **test_data_loader.py**: CSV file handling, missing files, invalid data
- **test_preprocessing.py**: Scaling, encoding, missing values, unknown categories
- **test_train.py**: Model training, cross-validation, hyperparameter tuning
- **test_evaluate.py**: Metrics calculation, edge cases, perfect predictions
- **test_visualize.py**: Plot generation, file saving, custom parameters
- **test_model_utils.py**: Save/load operations, metadata handling
- **test_api.py**: All endpoints, validation, error handling, mock responses

### Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov=api --cov-report=html --cov-report=term

# Run specific test module
pytest tests/test_train.py -v

# Run tests matching a pattern
pytest tests/ -k "test_model" -v

# Run with detailed output for debugging
pytest tests/ -vv -s

# Run only failed tests from last run
pytest --lf

# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto
```

### Test Coverage

View coverage report:
```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Scheduled nightly builds

See `.github/workflows/` for CI configuration (if available).

## 🛠️ Development

### Code Organization

The codebase follows a modular, production-ready architecture:

- **src/config.py**: Configuration management with YAML and env vars
- **src/pipeline.py**: sklearn Pipeline construction utilities
- **src/train.py**: Model training with cross-validation and tuning (type hints, logging)
- **src/inference.py**: Production inference with ModelPredictor class
- **src/data_loader.py**: Data loading and validation
- **src/preprocessing.py**: Legacy preprocessing functions
- **src/evaluate.py**: Metrics and evaluation
- **src/visualize.py**: Plotting and visualization
- **src/model_utils.py**: Model persistence utilities
- **api/**: REST API implementation (FastAPI, Pydantic)
- **tests/**: Comprehensive test suite (pytest)
- **run_all.py**: Main pipeline orchestration

### Key Design Patterns

1. **Configuration as Code**: All settings in `config.yaml`, overridable via environment
2. **Pipeline Pattern**: sklearn Pipelines for reproducible transformations
3. **Dependency Injection**: Pass config objects, not global state
4. **Type Safety**: Full type hints for IDE support and error detection
5. **Separation of Concerns**: Training, inference, and API are independent
6. **Testability**: Mock-friendly interfaces, comprehensive fixtures

### Adding a New Model

1. **Add to `src/train.py`**:
   ```python
   def get_base_estimator(model_type: str, random_state: int = 42):
       estimators = {
           # ... existing models ...
           "xgboost": XGBRegressor(random_state=random_state, n_jobs=-1)
       }
   ```

2. **Add hyperparameter grid to `config.yaml`**:
   ```yaml
   hyperparameter_grid:
     xgboost:
       n_estimators: [100, 200, 500]
       max_depth: [3, 5, 7]
       learning_rate: [0.01, 0.1, 0.3]
   ```

3. **Add tests in `tests/test_train.py`**:
   ```python
   def test_xgboost_training(sample_features, sample_target, ...):
       model, metrics = train_and_evaluate(X, y, model_type="xgboost")
       assert model is not None
   ```

4. **Update documentation**

### Adding a New Feature

1. **Update `config.yaml`**:
   ```yaml
   numerical_features:
     - "bed"
     - "bath"
     - "price_per_sqft"  # New feature
   ```

2. **Add feature engineering in data loading**:
   ```python
   df['price_per_sqft'] = df['price'] / df['house_size']
   ```

3. **Update tests and documentation**

### Extending the API

1. **Define schema in `api/schemas.py`**:
   ```python
   class NewRequest(BaseModel):
       # fields
   ```

2. **Implement endpoint in `api/main.py`**:
   ```python
   @app.post("/new-endpoint")
   async def new_endpoint(request: NewRequest):
       # implementation
   ```

3. **Add tests in `tests/test_api.py`**:
   ```python
   def test_new_endpoint(client):
       response = client.post("/new-endpoint", json={...})
       assert response.status_code == 200
   ```

### Code Style

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use for all function signatures
- **Docstrings**: Google-style docstrings for public APIs
- **Logging**: Use `logging` module, not `print`
- **Error Handling**: Specific exceptions with helpful messages
- **Testing**: Write tests before or alongside new code

## 🚧 Production Deployment

### Environment Variables

Set these in production:

```bash
# Required
export REAL_ESTATE_DATA_PATH=/data/realtor-data.csv
export REAL_ESTATE_MODEL_PATH=/models/production_model.joblib
export REAL_ESTATE_PIPELINE_PATH=/models/production_pipeline.joblib

# Optional
export REAL_ESTATE_LOG_LEVEL=INFO
export REAL_ESTATE_MODEL_TYPE=random_forest
export REAL_ESTATE_N_SPLITS=10
```

### Health Monitoring

The API includes health check endpoints:
- `/health` - Basic health status
- `/model/info` - Model metadata and version
- `/model/metrics` - Model performance metrics

### Scaling Considerations

- **API**: Deploy multiple instances behind a load balancer
- **Training**: Use batch processing for large datasets
- **Caching**: Consider Redis for frequent predictions on same inputs
- **Async**: API supports async endpoints for concurrent requests

### MLflow Integration (Optional)

Enable experiment tracking:

1. **Update config.yaml**:
   ```yaml
   mlflow:
     enabled: true
     tracking_uri: "http://mlflow-server:5000"
     experiment_name: "real_estate_production"
   ```

2. **Track experiments**:
   ```python
   # Automatically logs metrics, parameters, and artifacts
   python run_all.py
   ```

3. **View results**:
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```

## 🚀 Future Enhancements

### Completed ✅
- [x] Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- [x] Additional model algorithms (Random Forest, Gradient Boosting)
- [x] Docker containerization for deployment
- [x] Configuration management with YAML
- [x] Production-ready inference module
- [x] Comprehensive logging
- [x] Type hints throughout codebase
- [x] Extended test coverage (90%+)

### Planned 🔜
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Model monitoring and drift detection
- [ ] Interactive web dashboard (Streamlit/Gradio)
- [ ] Database integration (PostgreSQL)
- [ ] Feature store for feature engineering
- [ ] A/B testing framework
- [ ] Model explainability (SHAP, LIME)
- [ ] XGBoost and LightGBM support
- [ ] Automated retraining pipeline
- [ ] Kubernetes deployment manifests
- [ ] Prometheus metrics export
- [ ] GraphQL API option

## 📝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Update documentation as needed
6. Follow code style guidelines (PEP 8, type hints)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Development Workflow

```bash
# Setup
git clone <repo>
cd real-estate-prediction
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# Make changes
# ... edit code ...

# Run tests
pytest tests/ -v --cov=src

# Format code (optional, requires black)
black src/ tests/ api/

# Lint code (optional, requires flake8)
flake8 src/ tests/ api/

# Commit
git add .
git commit -m "Description of changes"
git push origin feature-branch
```

## 📚 Quick Reference

### Common Commands

```bash
# Training
python run_all.py                          # Run full pipeline
python -c "from src.config import get_config; print(get_config())"  # View config

# API
uvicorn api.main:app --reload              # Development server
uvicorn api.main:app --host 0.0.0.0 --port 8000  # Production server
curl http://localhost:8000/health          # Health check

# Testing
pytest tests/ -v                           # Run all tests
pytest tests/test_train.py -v              # Run specific test file
pytest -k "test_model" -v                  # Run tests matching pattern
pytest --cov=src --cov-report=html         # Generate coverage report

# Docker
docker-compose up train                    # Run training in Docker
docker-compose up -d api                   # Start API server
docker-compose down                        # Stop all services
docker-compose logs -f api                 # View API logs

# Configuration
export REAL_ESTATE_MODEL_TYPE=random_forest  # Override model type
export REAL_ESTATE_LOG_LEVEL=DEBUG          # Set log level
```

### Module Imports

```python
# Configuration
from src.config import get_config, load_config

# Training
from src.train import train_and_evaluate, get_base_estimator

# Pipeline
from src.pipeline import build_full_pipeline, build_preprocessing_pipeline

# Inference
from src.inference import ModelPredictor, predict, batch_predict

# Data Loading
from src.data_loader import load_data

# Evaluation
from src.evaluate import evaluate_model

# Visualization
from src.visualize import plot_predictions_vs_actual, plot_cv_scores

# Model Utils
from src.model_utils import save_model_bundle, load_model_bundle
```

### Configuration Paths

| File | Purpose |
|------|---------|
| `config.yaml` | Main configuration file |
| `.env` | Environment variables (create if needed) |
| `models/latest_model.joblib` | Trained model |
| `models/preprocessing_pipeline.joblib` | Preprocessing pipeline |
| `data/raw/realtor-data.csv` | Input dataset |
| `data/processed/processed_data.csv` | Processed features |
| `reports/figures/` | Generated visualizations |
| `mlruns/` | MLflow experiment tracking |

## 📄 License

[Add your license here - e.g., MIT, Apache 2.0, etc.]

## 👥 Authors

[Add author information]

## 🙏 Acknowledgments

- Dataset: [Credit the data source if applicable]
- Libraries: scikit-learn, FastAPI, pandas, matplotlib, seaborn
- Community: Contributors and users

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: [Your contact information]

---

**Built with ❤️ using Python, scikit-learn, and FastAPI**
