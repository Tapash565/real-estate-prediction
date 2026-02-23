"""Model persistence utilities for saving and loading trained models and preprocessing pipelines."""

import os
from datetime import datetime
import joblib


def save_model(model, filepath, metadata=None):
    """
    Save a trained model to disk using joblib.

    Parameters
    ----------
    model : estimator object
        The trained scikit-learn model to save.
    filepath : str
        The path where the model should be saved.
    metadata : dict, optional
        Additional metadata to save with the model (e.g., training date, metrics, model type).

    Returns
    -------
    str
        The path to the saved model file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create a bundle with model and metadata
    model_bundle = {
        'model': model,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat()
    }
    
    # Save using joblib
    joblib.dump(model_bundle, filepath)
    print(f"Model saved to {filepath}")
    
    return filepath


def load_model(filepath):
    """
    Load a trained model from disk.

    Parameters
    ----------
    filepath : str
        The path to the saved model file.

    Returns
    -------
    model : estimator object
        The loaded scikit-learn model.
    metadata : dict
        The metadata that was saved with the model.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Load the model bundle
    model_bundle = joblib.load(filepath)
    
    # Handle both old format (just model) and new format (bundle)
    if isinstance(model_bundle, dict) and 'model' in model_bundle:
        model = model_bundle['model']
        metadata = model_bundle.get('metadata', {})
        print(f"Model loaded from {filepath}")
        print(f"Saved at: {model_bundle.get('saved_at', 'Unknown')}")
        return model, metadata
    else:
        # Legacy format - just the model
        print(f"Model loaded from {filepath} (legacy format)")
        return model_bundle, {}


def save_preprocessing_pipeline(pipeline, filepath):
    """
    Save a fitted preprocessing pipeline to disk.

    Parameters
    ----------
    pipeline : Pipeline or ColumnTransformer
        The fitted preprocessing pipeline.
    filepath : str
        The path where the pipeline should be saved.

    Returns
    -------
    str
        The path to the saved pipeline file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save using joblib
    joblib.dump(pipeline, filepath)
    print(f"Preprocessing pipeline saved to {filepath}")
    
    return filepath


def load_preprocessing_pipeline(filepath):
    """
    Load a preprocessing pipeline from disk.

    Parameters
    ----------
    filepath : str
        The path to the saved pipeline file.

    Returns
    -------
    pipeline : Pipeline or ColumnTransformer
        The loaded preprocessing pipeline.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pipeline file not found: {filepath}")
    
    pipeline = joblib.load(filepath)
    print(f"Preprocessing pipeline loaded from {filepath}")
    
    return pipeline


def save_model_bundle(model, pipeline, model_path, pipeline_path, metadata=None):
    """
    Save both model and preprocessing pipeline together.

    This is a convenience function to save both the model and pipeline
    in one call, ensuring they stay synchronized.

    Parameters
    ----------
    model : estimator object
        The trained scikit-learn model.
    pipeline : Pipeline or ColumnTransformer
        The fitted preprocessing pipeline.
    model_path : str
        The path where the model should be saved.
    pipeline_path : str
        The path where the pipeline should be saved.
    metadata : dict, optional
        Additional metadata to save with the model.

    Returns
    -------
    tuple
        (model_path, pipeline_path) tuple of saved file paths.
    """
    save_model(model, model_path, metadata)
    save_preprocessing_pipeline(pipeline, pipeline_path)
    
    return model_path, pipeline_path


def load_model_bundle(model_path, pipeline_path):
    """
    Load both model and preprocessing pipeline together.

    This is a convenience function to load both the model and pipeline
    in one call.

    Parameters
    ----------
    model_path : str
        The path to the saved model file.
    pipeline_path : str
        The path to the saved pipeline file.

    Returns
    -------
    tuple
        (model, pipeline, metadata) tuple.
    """
    model, metadata = load_model(model_path)
    pipeline = load_preprocessing_pipeline(pipeline_path)
    
    return model, pipeline, metadata
