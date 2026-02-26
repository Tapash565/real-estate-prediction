"""Configuration management for the real estate prediction pipeline.

This module loads configuration from config.yaml and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data-related configuration."""
    data_path: str
    target_column: str
    numerical_features: List[str]
    categorical_features: List[str]
    sample_size: Optional[int] = None
    price_min: float = 10000
    price_max: float = 10000000
    required_columns: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Model-related configuration."""
    model_type: str = "random_forest"
    n_splits: int = 5
    random_state: int = 42
    hyperparameter_grid: Optional[Dict[str, Any]] = None


@dataclass
class PathConfig:
    """Output path configuration."""
    model_path: str = "models/latest_model.joblib"
    pipeline_path: str = "models/preprocessing_pipeline.joblib"
    processed_data_path: str = "data/processed/processed_data.csv"
    figures_dir: str = "reports/figures"


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    enabled: bool = False
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "real_estate_prediction"


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig
    model: ModelConfig
    paths: PathConfig
    mlflow: MLflowConfig
    log_level: str = "INFO"


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file and environment variables.
    
    Environment variables take precedence over config file values.
    Use the format: REAL_ESTATE_<SECTION>_<KEY> (e.g., REAL_ESTATE_DATA_PATH)
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Config object with all settings.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required config values are missing.
    """
    # Load YAML config
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Override with environment variables
    def get_env_or_yaml(section: str, key: str, default=None):
        """Get value from environment variable or YAML config."""
        env_key = f"REAL_ESTATE_{section.upper()}_{key.upper()}" if section else f"REAL_ESTATE_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value
        if section:
            return yaml_config.get(section, {}).get(key, default)
        return yaml_config.get(key, default)
    
    # Build DataConfig
    data_config = DataConfig(
        data_path=get_env_or_yaml('', 'data_path', 'data/raw/realtor-data.csv'),
        target_column=get_env_or_yaml('', 'target_column', 'price'),
        numerical_features=yaml_config.get('numerical_features', []),
        categorical_features=yaml_config.get('categorical_features', []),
        sample_size=yaml_config.get('sample_size'),
        price_min=yaml_config.get('price_min', 10000),
        price_max=yaml_config.get('price_max', 10000000),
        required_columns=yaml_config.get('required_columns', [])
    )
    
    # Build ModelConfig
    model_config = ModelConfig(
        model_type=get_env_or_yaml('', 'model_type', 'random_forest'),
        n_splits=int(get_env_or_yaml('', 'n_splits', 5)),
        random_state=int(get_env_or_yaml('', 'random_state', 42)),
        hyperparameter_grid=yaml_config.get('hyperparameter_grid')
    )
    
    # Build PathConfig
    path_config = PathConfig(
        model_path=get_env_or_yaml('', 'model_path', 'models/latest_model.joblib'),
        pipeline_path=get_env_or_yaml('', 'pipeline_path', 'models/preprocessing_pipeline.joblib'),
        processed_data_path=get_env_or_yaml('', 'processed_data_path', 'data/processed/processed_data.csv'),
        figures_dir=get_env_or_yaml('', 'figures_dir', 'reports/figures')
    )
    
    # Build MLflowConfig
    mlflow_settings = yaml_config.get('mlflow', {})
    mlflow_config = MLflowConfig(
        enabled=mlflow_settings.get('enabled', False),
        tracking_uri=mlflow_settings.get('tracking_uri', 'file:./mlruns'),
        experiment_name=mlflow_settings.get('experiment_name', 'real_estate_prediction')
    )
    
    # Build main Config
    config = Config(
        data=data_config,
        model=model_config,
        paths=path_config,
        mlflow=mlflow_config,
        log_level=get_env_or_yaml('', 'log_level', 'INFO')
    )
    
    return config


# Singleton config instance
_config: Optional[Config] = None


def get_config(config_path: str = "config.yaml", reload: bool = False) -> Config:
    """Get the global configuration instance.
    
    Args:
        config_path: Path to config file (used on first load).
        reload: Force reload configuration from file.
        
    Returns:
        Global Config instance.
    """
    global _config
    if _config is None or reload:
        _config = load_config(config_path)
    return _config
