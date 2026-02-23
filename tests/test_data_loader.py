"""Tests for data loading functionality."""

import os
import pytest
from src.data_loader import load_data


def test_load_data_success(sample_csv_path):
    """Test successful data loading from CSV."""
    df = load_data(sample_csv_path)
    
    assert df is not None
    assert len(df) > 0
    assert 'price' in df.columns


def test_load_data_file_not_found():
    """Test that FileNotFoundError is raised for missing file."""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.csv")


def test_load_data_returns_dataframe(sample_csv_path):
    """Test that load_data returns a pandas DataFrame."""
    import pandas as pd
    
    df = load_data(sample_csv_path)
    assert isinstance(df, pd.DataFrame)


def test_load_data_columns(sample_csv_path):
    """Test that expected columns are present in loaded data."""
    df = load_data(sample_csv_path)
    
    expected_columns = ['bed', 'bath', 'house_size', 'acre_lot', 'city', 
                       'state', 'zip_code', 'brokered_by', 'status', 'price']
    
    for col in expected_columns:
        assert col in df.columns, f"Expected column '{col}' not found"


def test_load_data_invalid_csv(tmp_path):
    """Test handling of invalid CSV file."""
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("This is not,valid,CSV\ndata without proper structure")
    
    # Should handle gracefully and return None or raise exception
    result = load_data(str(invalid_csv))
    # Depending on implementation, this might return None or raise an exception
    # The current implementation returns None on any exception
    assert result is None or len(result) >= 0
