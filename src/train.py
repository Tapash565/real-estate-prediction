"""Model training with KFold cross-validation and hyperparameter tuning."""

import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from numpy.typing import NDArray

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


def get_base_estimator(
    model_type: str,
    random_state: int = 42
) -> BaseEstimator:
    """
    Get a base estimator instance based on model type.
    
    Args:
        model_type: One of "linear", "sgd", "decision_tree", "random_forest", "gradient_boosting".
        random_state: Random seed for reproducibility.
        
    Returns:
        Scikit-learn estimator instance.
        
    Raises:
        ValueError: If model_type is not supported.
    """
    estimators = {
        "linear": LinearRegression(),
        "sgd": SGDRegressor(max_iter=1000, tol=1e-3, random_state=random_state),
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=100, 
            random_state=random_state, 
            n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, 
            random_state=random_state
        )
    }
    
    if model_type not in estimators:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Choose from: {list(estimators.keys())}"
        )
    
    return estimators[model_type]


def train_and_evaluate(
    X: NDArray,
    y: NDArray,
    model_type: str = "random_forest",
    n_splits: int = 5,
    random_state: int = 42,
    param_grid: Optional[Dict[str, Any]] = None,
    search_type: str = "grid",
    n_iter: int = 10
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Train a model with KFold cross-validation and optional hyperparameter tuning.
    
    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        model_type: Type of model to train.
        n_splits: Number of KFold splits for cross-validation.
        random_state: Random seed for reproducibility.
        param_grid: Hyperparameter grid for tuning. If None, uses default parameters.
        search_type: "grid" for GridSearchCV or "random" for RandomizedSearchCV.
        n_iter: Number of iterations for RandomizedSearchCV (ignored for grid search).
        
    Returns:
        Tuple of (best_estimator, metrics_summary) where metrics_summary contains:
            - cv_scores: List of R² scores for each fold
            - mean_r2: Mean R² across folds
            - std_r2: Standard deviation of R² across folds
            - mean_rmse: Mean RMSE across folds
            - mean_mae: Mean MAE across folds
            - best_params: Best hyperparameters (if tuning was performed)
    """
    logger.info(f"Training {model_type} model with {n_splits}-fold cross-validation")
    
    # Get base estimator
    base_estimator = get_base_estimator(model_type, random_state)
    
    # Perform hyperparameter tuning if param_grid is provided
    if param_grid is not None:
        logger.info(f"Performing hyperparameter tuning using {search_type} search")
        
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        if search_type == "grid":
            search = GridSearchCV(
                base_estimator,
                param_grid,
                cv=kfold,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
        elif search_type == "random":
            search = RandomizedSearchCV(
                base_estimator,
                param_grid,
                n_iter=n_iter,
                cv=kfold,
                scoring='r2',
                n_jobs=-1,
                random_state=random_state,
                verbose=1
            )
        else:
            raise ValueError(f"search_type must be 'grid' or 'random', got {search_type}")
        
        search.fit(X, y)
        best_estimator = search.best_estimator_
        best_params = search.best_params_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV R²: {search.best_score_:.4f}")
    else:
        best_params = {}
        logger.info("No hyperparameter tuning - using default parameters")
    
    # Perform detailed cross-validation to get metrics
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores_r2 = []
    cv_scores_rmse = []
    cv_scores_mae = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Use best estimator if tuning was done, otherwise fit base estimator
        if param_grid is not None:
            estimator = best_estimator.__class__(**best_params)
            if hasattr(estimator, 'random_state'):
                estimator.random_state = random_state
        else:
            estimator = get_base_estimator(model_type, random_state)
        
        estimator.fit(X_train_fold, y_train_fold)
        y_pred = estimator.predict(X_val_fold)
        
        r2 = r2_score(y_val_fold, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        mae = mean_absolute_error(y_val_fold, y_pred)
        
        cv_scores_r2.append(r2)
        cv_scores_rmse.append(rmse)
        cv_scores_mae.append(mae)
        
        logger.info(f"Fold {fold}/{n_splits} - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    mean_r2 = np.mean(cv_scores_r2)
    std_r2 = np.std(cv_scores_r2)
    mean_rmse = np.mean(cv_scores_rmse)
    mean_mae = np.mean(cv_scores_mae)
    
    logger.info(f"Cross-validation results:")
    logger.info(f"  R² = {mean_r2:.4f} ± {std_r2:.4f}")
    logger.info(f"  RMSE = {mean_rmse:.2f}")
    logger.info(f"  MAE = {mean_mae:.2f}")
    
    # Train final model on full dataset
    logger.info("Training final model on full dataset")
    if param_grid is not None:
        final_model = best_estimator.__class__(**best_params)
        if hasattr(final_model, 'random_state'):
            final_model.random_state = random_state
    else:
        final_model = get_base_estimator(model_type, random_state)
    
    final_model.fit(X, y)
    
    # Prepare metrics summary
    metrics_summary = {
        'cv_scores': cv_scores_r2,
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'mean_rmse': mean_rmse,
        'mean_mae': mean_mae,
        'best_params': best_params,
        'n_samples': len(X),
        'n_features': X.shape[1]
    }
    
    return final_model, metrics_summary


def train_model(
    X_train: NDArray,
    y_train: NDArray,
    model_type: str = "regression",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    evaluate: bool = False,
    return_models: bool = False,
) -> Union[BaseEstimator, List[BaseEstimator]]:
    """
    Legacy training function for backward compatibility.
    
    For new code, prefer using train_and_evaluate() instead.
    
    Args:
        X_train: Feature matrix.
        y_train: Target vector.
        model_type: Model type (supports legacy names: "regression" -> "linear").
        n_splits: Number of folds for KFold.
        shuffle: Whether to shuffle data before splitting.
        random_state: Random seed.
        evaluate: If True, compute and log R² score for each fold.
        return_models: If True, return list of fold models; else return single model.
    
    Returns:
        Trained model or list of models.
    """
    logger.warning("train_model() is deprecated. Use train_and_evaluate() for new code.")
    
    # Map legacy model names
    if model_type == "regression":
        model_type = "linear"
    
    base_estimator = get_base_estimator(model_type, random_state)
    
    if evaluate:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            estimator = get_base_estimator(model_type, random_state)
            estimator.fit(X_train[train_idx], y_train[train_idx])
            preds = estimator.predict(X_train[val_idx])
            score = r2_score(y_train[val_idx], preds)
            fold_scores.append(score)
            logger.info(f"Fold {fold}/{n_splits} R²: {score:.4f}")
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        logger.info(f"Cross-validation R²: mean={mean_score:.4f}, std={std_score:.4f}")
    
    if return_models:
        models = []
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_idx, _ in kf.split(X_train):
            estimator = get_base_estimator(model_type, random_state)
            estimator.fit(X_train[train_idx], y_train[train_idx])
            models.append(estimator)
        return models
    
    # Train on the full dataset
    model = get_base_estimator(model_type, random_state)
    model.fit(X_train, y_train)
    return model
