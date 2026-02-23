"""Tests for configuration management."""

import pytest
import os
from pathlib import Path
import yaml
from src.config import (
    load_config, get_config, Config, DataConfig, ModelConfig, PathConfig, MLflowConfig
)


def test_load_config_from_yaml(tmp_path):
    """Test loading configuration from YAML file."""
    config_content = """
data_path: "data/test.csv"
target_column: "price"
numerical_features: ["bed", "bath"]
categorical_features: ["state"]
model_type: "linear"
n_splits: 3
random_state: 123
log_level: "DEBUG"
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    
    config = load_config(str(config_file))
    
    assert isinstance(config, Config)
    assert config.data.data_path == "data/test.csv"
    assert config.data.target_column == "price"
    assert config.data.numerical_features == ["bed", "bath"]
    assert config.data.categorical_features == ["state"]
    assert config.model.model_type == "linear"
    assert config.model.n_splits == 3
    assert config.model.random_state == 123
    assert config.log_level == "DEBUG"


def test_load_config_with_defaults(tmp_path):
    """Test that missing values use defaults."""
    config_content = """
data_path: "data/test.csv"
target_column: "price"
numerical_features: ["bed"]
categorical_features: ["state"]
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    
    config = load_config(str(config_file))
    
    # Check defaults
    assert config.model.model_type == "random_forest"
    assert config.model.n_splits == 5
    assert config.model.random_state == 42
    assert config.log_level == "INFO"


def test_load_config_file_not_found():
    """Test that FileNotFoundError is raised for missing config."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_get_config_singleton(tmp_path):
    """Test that get_config returns singleton instance."""
    config_content = """
data_path: "data/test.csv"
target_column: "price"
numerical_features: ["bed"]
categorical_features: ["state"]
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    
    # Reset singleton
    import src.config
    src.config._config = None
    
    config1 = get_config(str(config_file))
    config2 = get_config(str(config_file))
    
    assert config1 is config2


def test_config_dataclasses():
    """Test that configuration dataclasses work correctly."""
    data_config = DataConfig(
        data_path="test.csv",
        target_column="price",
        numerical_features=["bed"],
        categorical_features=["state"]
    )
    
    assert data_config.data_path == "test.csv"
    assert data_config.sample_size is None
    
    model_config = ModelConfig(model_type="linear", n_splits=3)
    assert model_config.random_state == 42  # default
    
    path_config = PathConfig()
    assert path_config.model_path == "models/latest_model.joblib"
    
    mlflow_config = MLflowConfig(enabled=True)
    assert mlflow_config.tracking_uri == "file:./mlruns"
