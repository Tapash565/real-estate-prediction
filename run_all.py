"""Run the full data science pipeline.

This script demonstrates how the utilities in the ``src`` package can be
used together: loading the raw CSV, preprocessing the data, training a
model with optional K‑Fold cross‑validation, evaluating the model, and
visualizing the results.

The script is intentionally minimal and is suitable for quick
experimentation or as a template for a larger workflow.
"""

import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.model_utils import save_model_bundle
from src import visualize

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "raw", "realtor-data.csv")

# Target column
TARGET_COLUMN = "price"

# Features - updated to match actual CSV columns
# Note: Removed high-cardinality features (zip_code, brokered_by, city) to prevent memory issues
NUMERICAL_FEATURES = ["bed", "bath", "house_size", "acre_lot"]
CATEGORICAL_FEATURES = ["state", "status"]  # Only low-cardinality categorical features

# Model output paths
MODEL_PATH = os.path.join("models", "latest_model.joblib")
PIPELINE_PATH = os.path.join("models", "preprocessing_pipeline.joblib")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "processed_data.csv")

# Visualization output directory
FIGURES_DIR = os.path.join("reports", "figures")

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("REAL ESTATE PRICE PREDICTION PIPELINE")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[1/7] Loading data...")
    df = load_data(DATA_PATH)
    if df is None:
        raise RuntimeError("Data loading failed.")
    print(f"   ✓ Loaded {len(df)} records with {len(df.columns)} columns")

    # Clean data: Remove rows with missing critical features
    print("\n   Cleaning data...")
    initial_count = len(df)
    
    # Drop rows with missing values in key features
    df = df.dropna(subset=['bed', 'bath', 'house_size', TARGET_COLUMN])
    print(f"   ✓ Removed {initial_count - len(df)} rows with missing critical features")
    
    # Remove price outliers (unrealistic prices)
    df = df[(df[TARGET_COLUMN] > 10000) & (df[TARGET_COLUMN] < 10000000)]
    print(f"   ✓ Filtered to realistic prices ($10k - $10M): {len(df)} records remain")
    
    # Sample 100k high-quality records for faster training
    SAMPLE_SIZE = 100000
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"   ✓ Sampled {SAMPLE_SIZE:,} records for training")
    
    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Step 2: Preprocess data
    print("\n[2/7] Preprocessing data...")
    X_processed = preprocess_data(X, CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
    print(f"   ✓ Preprocessed shape: {X_processed.shape}")
    
    # Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    pd.DataFrame(X_processed).to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"   ✓ Saved processed data to {PROCESSED_DATA_PATH}")

    # Step 3: Train model with cross-validation
    print("\n[3/7] Training model with 5‑fold cross-validation...")
    
    # Collect CV scores for visualization
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed), 1):
        temp_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        temp_model.fit(X_processed[train_idx], y.iloc[train_idx])
        preds = temp_model.predict(X_processed[val_idx])
        score = r2_score(y.iloc[val_idx], preds)
        cv_scores.append(score)
        print(f"   Fold {fold}/{n_splits} R²: {score:.4f}")
    
    mean_score = sum(cv_scores) / len(cv_scores)
    std_score = (sum((s - mean_score) ** 2 for s in cv_scores) / len(cv_scores)) ** 0.5
    print(f"   ✓ Cross-validation R²: mean={mean_score:.4f}, std={std_score:.4f}")
    
    # Train final model on all data
    print("\n[4/7] Training final model on full dataset...")
    model = train_model(
        X_processed,
        y,
        model_type="random_forest",
        n_splits=5,
        evaluate=False,
        return_models=False,
    )
    print(f"   ✓ Model trained: {type(model).__name__}")

    # Step 4: Make predictions
    print("\n[5/7] Making predictions on the training data...")
    y_pred = model.predict(X_processed)
    print(f"   ✓ Generated {len(y_pred)} predictions")

    # Step 5: Evaluate model
    print("\n[6/7] Evaluating model performance...")
    metrics = evaluate_model(y, y_pred)
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")

    # Step 6: Save model and preprocessing pipeline
    print("\n[7/7] Saving model and preprocessing pipeline...")
    
    # Recreate the preprocessing pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    # Recreate the preprocessing pipeline to save it
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
    preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    preprocessing_pipeline.fit(X)
    
    # Create metadata
    metadata = {
        'model_type': type(model).__name__,
        'metrics': metrics,
        'cv_scores': cv_scores,
        'cv_mean': mean_score,
        'cv_std': std_score,
        'features': {
            'numerical': NUMERICAL_FEATURES,
            'categorical': CATEGORICAL_FEATURES
        },
        'target': TARGET_COLUMN,
        'num_samples': len(df)
    }
    
    # Save model and pipeline
    save_model_bundle(model, preprocessing_pipeline, MODEL_PATH, PIPELINE_PATH, metadata)
    print(f"   ✓ Model saved to {MODEL_PATH}")
    print(f"   ✓ Pipeline saved to {PIPELINE_PATH}")

    # Step 7: Generate visualizations
    print("\n[BONUS] Generating visualizations...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Plot 1: Predictions vs Actual
    visualize.plot_predictions_vs_actual(
        y, y_pred, 
        save_path=os.path.join(FIGURES_DIR, "predictions_vs_actual.png")
    )
    
    # Plot 2: Residuals
    visualize.plot_residuals(
        y, y_pred,
        save_path=os.path.join(FIGURES_DIR, "residuals.png")
    )
    
    # Plot 3: Error Distribution
    visualize.plot_error_distribution(
        y, y_pred,
        save_path=os.path.join(FIGURES_DIR, "error_distribution.png")
    )
    
    # Plot 4: CV Scores
    visualize.plot_cv_scores(
        cv_scores,
        save_path=os.path.join(FIGURES_DIR, "cv_scores.png")
    )
    
    # Plot 5: Feature Importance (if applicable)
    try:
        # Get feature names after preprocessing
        feature_names = NUMERICAL_FEATURES.copy()
        if hasattr(preprocessing_pipeline.named_steps['preprocessor'].named_transformers_['cat'], 'get_feature_names_out'):
            cat_features = preprocessing_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
            feature_names.extend(cat_features)
        
        visualize.plot_feature_importance(
            model, feature_names,
            save_path=os.path.join(FIGURES_DIR, "feature_importance.png"),
            top_n=20
        )
    except Exception as e:
        print(f"   ⚠ Could not generate feature importance plot: {e}")
    
    print(f"   ✓ All visualizations saved to {FIGURES_DIR}/")

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Model Performance: R² = {metrics['R^2 Score']:.4f}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Visualizations: {FIGURES_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
