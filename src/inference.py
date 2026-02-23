"""Model inference module for loading trained models and making predictions.

This module provides functionality to load saved model bundles and perform
predictions on new data.
"""

import logging
from pathlib import Path
from typing import Union, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Wrapper class for loading and using trained models for inference."""
    
    def __init__(
        self,
        model_path: str,
        pipeline_path: Optional[str] = None
    ):
        """
        Initialize the predictor with model and optional preprocessing pipeline.
        
        Args:
            model_path: Path to the saved model (.joblib file).
            pipeline_path: Optional path to preprocessing pipeline (.joblib file).
                          If None, assumes model_path contains a full Pipeline.
        """
        self.model_path = Path(model_path)
        self.pipeline_path = Path(pipeline_path) if pipeline_path else None
        self.model: Optional[BaseEstimator] = None
        self.preprocessor: Optional[Pipeline] = None
        self._is_loaded = False
        
    def load(self) -> None:
        """Load the model and optional preprocessing pipeline from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        
        if self.pipeline_path:
            if not self.pipeline_path.exists():
                raise FileNotFoundError(f"Pipeline file not found: {self.pipeline_path}")
            
            logger.info(f"Loading preprocessing pipeline from {self.pipeline_path}")
            self.preprocessor = joblib.load(self.pipeline_path)
        
        self._is_loaded = True
        logger.info("Model loaded successfully")
    
    def predict(
        self,
        X: Union[pd.DataFrame, NDArray],
        return_dataframe: bool = False
    ) -> Union[NDArray, pd.DataFrame]:
        """
        Make predictions on new data.
        
        Args:
            X: Input features as DataFrame or array.
            return_dataframe: If True and X is a DataFrame, return predictions as DataFrame.
            
        Returns:
            Predictions as array or DataFrame.
            
        Raises:
            RuntimeError: If model hasn't been loaded yet.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Preprocess if preprocessor is available
        if self.preprocessor is not None:
            logger.debug("Applying preprocessing pipeline")
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        # Make predictions
        logger.debug(f"Making predictions for {len(X)} samples")
        predictions = self.model.predict(X_processed)
        
        # Return as DataFrame if requested and input was DataFrame
        if return_dataframe and isinstance(X, pd.DataFrame):
            return pd.DataFrame(predictions, columns=['prediction'], index=X.index)
        
        return predictions
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, NDArray]
    ) -> NDArray:
        """
        Predict class probabilities (for classifiers).
        
        Args:
            X: Input features.
            
        Returns:
            Class probabilities.
            
        Raises:
            RuntimeError: If model hasn't been loaded.
            AttributeError: If model doesn't support predict_proba.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("Model does not support predict_proba")
        
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        return self.model.predict_proba(X_processed)


def load_model_bundle(
    model_path: str,
    pipeline_path: Optional[str] = None
) -> Tuple[BaseEstimator, Optional[Pipeline]]:
    """
    Load model and preprocessing pipeline from disk.
    
    Args:
        model_path: Path to the saved model.
        pipeline_path: Optional path to preprocessing pipeline.
        
    Returns:
        Tuple of (model, preprocessor). preprocessor is None if pipeline_path is None.
        
    Raises:
        FileNotFoundError: If files don't exist.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    preprocessor = None
    if pipeline_path:
        pipeline_file = Path(pipeline_path)
        if not pipeline_file.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
        
        logger.info(f"Loading preprocessing pipeline from {pipeline_path}")
        preprocessor = joblib.load(pipeline_path)
    
    return model, preprocessor


def predict(
    X: Union[pd.DataFrame, NDArray],
    model_path: str = "models/latest_model.joblib",
    pipeline_path: Optional[str] = "models/preprocessing_pipeline.joblib"
) -> NDArray:
    """
    Convenience function to load model and make predictions in one step.
    
    Args:
        X: Input features.
        model_path: Path to saved model.
        pipeline_path: Path to preprocessing pipeline. Set to None if not using preprocessing.
        
    Returns:
        Predictions as numpy array.
    """
    model, preprocessor = load_model_bundle(model_path, pipeline_path)
    
    if preprocessor is not None:
        X_processed = preprocessor.transform(X)
    else:
        X_processed = X
    
    predictions = model.predict(X_processed)
    
    return predictions


def batch_predict(
    X: pd.DataFrame,
    model_path: str,
    pipeline_path: Optional[str] = None,
    batch_size: int = 10000
) -> NDArray:
    """
    Make predictions in batches for large datasets.
    
    Args:
        X: Input features DataFrame.
        model_path: Path to saved model.
        pipeline_path: Path to preprocessing pipeline.
        batch_size: Number of samples per batch.
        
    Returns:
        Predictions as numpy array.
    """
    model, preprocessor = load_model_bundle(model_path, pipeline_path)
    
    n_samples = len(X)
    predictions = np.zeros(n_samples)
    
    logger.info(f"Processing {n_samples} samples in batches of {batch_size}")
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X.iloc[start_idx:end_idx]
        
        if preprocessor is not None:
            X_batch = preprocessor.transform(X_batch)
        
        predictions[start_idx:end_idx] = model.predict(X_batch)
        
        if (end_idx % (batch_size * 10)) == 0 or end_idx == n_samples:
            logger.info(f"Processed {end_idx}/{n_samples} samples")
    
    return predictions
