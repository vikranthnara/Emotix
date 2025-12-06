"""Unit tests for ingestion layer."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json
from src.ingest import (
    ingest_csv, ingest_jsonl, ingest_json, ingest_from_dict, validate_dataframe
)


class TestIngest:
    """Test ingestion functions."""
    
    def test_validate_dataframe_success(self):
        """Test successful validation."""
        df = pd.DataFrame({
            'UserID': ['user1', 'user2'],
            'Text': ['hello', 'world'],
            'Timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00']
        })
        result = validate_dataframe(df)
        assert len(result) == 2
        assert list(result.columns) == ['UserID', 'Text', 'Timestamp']
        assert pd.api.types.is_datetime64_any_dtype(result['Timestamp'])
    
    def test_validate_dataframe_missing_column(self):
        """Test validation fails with missing column."""
        df = pd.DataFrame({
            'UserID': ['user1'],
            'Text': ['hello']
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df)
    
    def test_validate_dataframe_null_values(self):
        """Test validation fails with null values."""
        df = pd.DataFrame({
            'UserID': ['user1', None],
            'Text': ['hello', 'world'],
            'Timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00']
        })
        with pytest.raises(ValueError, match="UserID cannot contain null"):
            validate_dataframe(df)
    
    def test_ingest_csv(self):
        """Test CSV ingestion."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("UserID,Text,Timestamp\n")
            f.write("user1,hello world,2024-01-01 10:00:00\n")
            f.write("user2,test message,2024-01-01 11:00:00\n")
            temp_path = f.name
        
        try:
            df = ingest_csv(temp_path)
            assert len(df) == 2
            assert list(df.columns) == ['UserID', 'Text', 'Timestamp']
        finally:
            Path(temp_path).unlink()
    
    def test_ingest_jsonl(self):
        """Test JSONL ingestion."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"UserID": "user1", "Text": "hello", "Timestamp": "2024-01-01 10:00:00"}\n')
            f.write('{"UserID": "user2", "Text": "world", "Timestamp": "2024-01-01 11:00:00"}\n')
            temp_path = f.name
        
        try:
            df = ingest_jsonl(temp_path)
            assert len(df) == 2
        finally:
            Path(temp_path).unlink()
    
    def test_ingest_json(self):
        """Test JSON ingestion."""
        data = [
            {"UserID": "user1", "Text": "hello", "Timestamp": "2024-01-01 10:00:00"},
            {"UserID": "user2", "Text": "world", "Timestamp": "2024-01-01 11:00:00"}
        ]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            df = ingest_json(temp_path)
            assert len(df) == 2
        finally:
            Path(temp_path).unlink()
    
    def test_ingest_from_dict(self):
        """Test ingestion from dict list."""
        records = [
            {"UserID": "user1", "Text": "hello", "Timestamp": "2024-01-01 10:00:00"},
            {"UserID": "user2", "Text": "world", "Timestamp": "2024-01-01 11:00:00"}
        ]
        df = ingest_from_dict(records)
        assert len(df) == 2
        assert list(df.columns) == ['UserID', 'Text', 'Timestamp']

