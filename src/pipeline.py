"""Build scikit-learn pipelines for preprocessing and model training.

This module creates a unified Pipeline that combines preprocessing steps
(imputation, scaling, encoding) with the chosen estimator.
"""

import logging
from typing import List, Optional

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.train import get_base_estimator

logger = logging.getLogger(__name__)


def build_preprocessing_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
    numerical_strategy: str = 'median',
    categorical_strategy: str = 'most_frequent'
) -> ColumnTransformer:
    """
    Build a preprocessing pipeline with separate transformers for numerical and categorical features.
    
    Args:
        numerical_features: List of numerical feature column names.
        categorical_features: List of categorical feature column names.
        numerical_strategy: Imputation strategy for numerical features ('mean', 'median', 'most_frequent').
        categorical_strategy: Imputation strategy for categorical features ('most_frequent', 'constant').
        
    Returns:
        ColumnTransformer configured with numerical and categorical pipelines.
    """
    logger.info(f"Building preprocessing pipeline:")
    logger.info(f"  Numerical features ({len(numerical_features)}): {numerical_features}")
    logger.info(f"  Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Numerical pipeline: impute missing values, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numerical_strategy)),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute missing values, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_strategy)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor


def build_full_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
    model_type: str = 'random_forest',
    random_state: int = 42,
    model_params: Optional[dict] = None
) -> Pipeline:
    """
    Build a complete pipeline combining preprocessing and model.
    
    Args:
        numerical_features: List of numerical feature column names.
        categorical_features: List of categorical feature column names.
        model_type: Type of model to use ('linear', 'sgd', 'decision_tree', 'random_forest', 'gradient_boosting').
        random_state: Random seed for reproducibility.
        model_params: Optional dictionary of model hyperparameters.
        
    Returns:
        Complete Pipeline with preprocessing and estimator.
    """
    logger.info(f"Building full pipeline with {model_type} model")
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    
    # Get base estimator
    estimator = get_base_estimator(model_type, random_state)
    
    # Apply custom parameters if provided
    if model_params:
        logger.info(f"Applying custom model parameters: {model_params}")
        estimator.set_params(**model_params)
    
    # Combine into full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', estimator)
    ])
    
    logger.info("Pipeline built successfully")
    
    return pipeline


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract feature names from a fitted ColumnTransformer.
    
    Args:
        preprocessor: Fitted ColumnTransformer.
        
    Returns:
        List of feature names after transformation.
    """
    feature_names = []
    
    for name, transformer, features in preprocessor.transformers_:
        if name == 'remainder':
            continue
            
        if hasattr(transformer, 'get_feature_names_out'):
            # For transformers that support get_feature_names_out
            names = transformer.get_feature_names_out(features)
            feature_names.extend(names)
        else:
            # Fallback for transformers without this method
            if isinstance(features, list):
                feature_names.extend(features)
            else:
                feature_names.append(features)
    
    return feature_names
