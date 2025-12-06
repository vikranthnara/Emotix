"""Unit tests for preprocessing layer."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json
from src.preprocess import Preprocessor, preprocess_pipeline


class TestPreprocess:
    """Test preprocessing functions."""
    
    def test_slang_normalization(self):
        """Test slang dictionary normalization."""
        slang_dict = {
            "lol": "laughing out loud",
            "omg": "oh my god",
            "tbh": "to be honest"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(slang_dict, f)
            temp_path = f.name
        
        try:
            preprocessor = Preprocessor(temp_path)
            text = "lol that's funny tbh"
            normalized, flags = preprocessor.normalize_slang(text)
            assert "laughing out loud" in normalized.lower()
            assert "to be honest" in normalized.lower()
            assert len(flags['slang_replacements']) > 0
        finally:
            Path(temp_path).unlink()
    
    def test_demojize(self):
        """Test emoji demojization."""
        preprocessor = Preprocessor()
        text = "I'm happy 😊 and excited 🎉"
        demojized, flags = preprocessor.demojize(text)
        assert "😊" not in demojized
        assert "🎉" not in demojized
        assert len(flags['emojis_found']) > 0
    
    def test_minimal_syntactic_fix(self):
        """Test syntactic fixes."""
        preprocessor = Preprocessor()
        text = "hello  world  ,  how are you?"
        fixed, flags = preprocessor.minimal_syntactic_fix(text)
        assert "  " not in fixed  # No double spaces
        assert fixed.count(" ") < text.count(" ")
    
    def test_preprocess_text_full_pipeline(self):
        """Test full preprocessing pipeline."""
        slang_dict = {"lol": "laughing out loud"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(slang_dict, f)
            temp_path = f.name
        
        try:
            preprocessor = Preprocessor(temp_path)
            text = "lol that's great 😊"
            normalized, flags = preprocessor.preprocess_text(text)
            assert "laughing out loud" in normalized.lower()
            assert "😊" not in normalized
            assert 'slang_replacements' in flags
            assert 'emojis_found' in flags
        finally:
            Path(temp_path).unlink()
    
    def test_preprocess_dataframe(self):
        """Test DataFrame preprocessing."""
        slang_dict = {"lol": "laughing out loud"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(slang_dict, f)
            temp_path = f.name
        
        try:
            df = pd.DataFrame({
                'UserID': ['user1', 'user2'],
                'Text': ['lol funny', 'hello 😊'],
                'Timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00']
            })
            
            preprocessor = Preprocessor(temp_path)
            result = preprocessor.preprocess_dataframe(df)
            
            assert 'NormalizedText' in result.columns
            assert 'NormalizationFlags' in result.columns
            assert len(result) == 2
        finally:
            Path(temp_path).unlink()
    
    def test_preprocess_pipeline_function(self):
        """Test convenience preprocessing function."""
        slang_dict = {"lol": "laughing out loud"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(slang_dict, f)
            temp_path = f.name
        
        try:
            df = pd.DataFrame({
                'UserID': ['user1'],
                'Text': ['lol funny'],
                'Timestamp': ['2024-01-01 10:00:00']
            })
            
            result = preprocess_pipeline(df, temp_path)
            assert 'NormalizedText' in result.columns
        finally:
            Path(temp_path).unlink()

