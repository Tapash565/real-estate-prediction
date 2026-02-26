# Project Specifications: Real Estate Price Prediction Agent

## 1. What the user can send as input
- Property features for prediction (bed, bath, house_size, acre_lot, city, state, zip_code, brokered_by, status).
- Configuration overrides (e.g., model type, dataset path) via API or environment variables.

## 2. What workflows exist
- **Training Workflow**: Loads raw data, preprocesses, trains ML model (Random Forest, linear, etc.), evaluates, and saves the model.
- **Inference Workflow**: Serves the trained model via a FastAPI REST API for single or batch predictions.

## 3. What tools are being used
- Python & scikit-learn (Machine Learning)
- FastAPI (REST API)
- Docker (Deployment)
- pytest (Testing)

## 4. What outputs are expected
- Predicted property price (in USD).
- Model performance metrics (R², RMSE, MAE) and visualization plots.
- Saved model artifacts (`.joblib`).

## 5. Where data is stored
- Raw/Processed Data: `data/raw/` and `data/processed/`
- Trained Models: `models/`
- Agent temporary files: `.tmp/`

## 6. Where the system will be deployed
- Locally via Docker Compose.
- Remotely on cloud deployment using either Docker Render or Frontend Vercel and Backend Render.

## 7. What "done" looks like
- The agent structure (`instructions/`, `execution/`) is fully set up.
- The `project_specs.md` is approved.
- The system correctly handles the model training and prediction workflows.
- API is fully operational and tests are passing.
