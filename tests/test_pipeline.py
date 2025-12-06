"""Unit tests for end-to-end pipeline."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from src.pipeline import run_full_pipeline


class TestPipeline:
    """Test end-to-end pipeline."""
    
    def test_run_full_pipeline_csv(self):
        """Test pipeline with CSV input."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("UserID,Text,Timestamp\n")
            f.write("user1,hello world,2024-01-01 10:00:00\n")
            f.write("user1,test message,2024-01-01 11:00:00\n")
            temp_input = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump({"lol": "laughing out loud"}, f)
            temp_slang = f.name
        
        try:
            # Mock the model to avoid downloading - patch at the pipeline import level
            with patch('src.pipeline.EmotionModel') as mock_model_class:
                mock_model = MagicMock()
                mock_model.predict_batch.return_value = [
                    {'label': 'joy', 'intensity': 0.8},
                    {'label': 'neutral', 'intensity': 0.5}
                ]
                mock_model_class.return_value = mock_model
                
                # Mock run_inference_pipeline to avoid model initialization
                with patch('src.pipeline.run_inference_pipeline') as mock_inference:
                    mock_inference.return_value = pd.DataFrame({
                        'UserID': ['user1', 'user1'],
                        'Timestamp': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00']),
                        'NormalizedText': ['hello world', 'test message'],
                        'PrimaryEmotionLabel': ['joy', 'neutral'],
                        'IntensityScore_Primary': [0.8, 0.5],
                        'Sequence': ['hello world', 'test message']
                    })
                    
                    df = run_full_pipeline(
                        temp_input,
                        temp_db,
                        slang_dict_path=temp_slang
                    )
                    
                    assert len(df) == 2
                    assert 'PrimaryEmotionLabel' in df.columns
        finally:
            Path(temp_input).unlink()
            Path(temp_db).unlink()
            Path(temp_slang).unlink()
    
    def test_run_full_pipeline_jsonl(self):
        """Test pipeline with JSONL input."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"UserID": "user1", "Text": "hello", "Timestamp": "2024-01-01 10:00:00"}\n')
            f.write('{"UserID": "user1", "Text": "world", "Timestamp": "2024-01-01 11:00:00"}\n')
            temp_input = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            with patch('src.pipeline.EmotionModel') as mock_model_class:
                mock_model = MagicMock()
                mock_model_class.return_value = mock_model
                
                with patch('src.pipeline.run_inference_pipeline') as mock_inference:
                    mock_inference.return_value = pd.DataFrame({
                        'UserID': ['user1', 'user1'],
                        'Timestamp': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00']),
                        'NormalizedText': ['hello', 'world'],
                        'PrimaryEmotionLabel': ['joy', 'neutral'],
                        'IntensityScore_Primary': [0.8, 0.5],
                        'Sequence': ['hello', 'world']
                    })
                    
                    df = run_full_pipeline(temp_input, temp_db)
                    assert len(df) == 2
                    assert 'PrimaryEmotionLabel' in df.columns
        finally:
            Path(temp_input).unlink()
            Path(temp_db).unlink()
    
    def test_run_full_pipeline_json(self):
        """Test pipeline with JSON input."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            data = [
                {"UserID": "user1", "Text": "hello", "Timestamp": "2024-01-01 10:00:00"},
                {"UserID": "user1", "Text": "world", "Timestamp": "2024-01-01 11:00:00"}
            ]
            json.dump(data, f)
            temp_input = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            with patch('src.pipeline.EmotionModel') as mock_model_class:
                mock_model = MagicMock()
                mock_model_class.return_value = mock_model
                
                with patch('src.pipeline.run_inference_pipeline') as mock_inference:
                    mock_inference.return_value = pd.DataFrame({
                        'UserID': ['user1', 'user1'],
                        'Timestamp': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00']),
                        'NormalizedText': ['hello', 'world'],
                        'PrimaryEmotionLabel': ['joy', 'neutral'],
                        'IntensityScore_Primary': [0.8, 0.5],
                        'Sequence': ['hello', 'world']
                    })
                    
                    df = run_full_pipeline(temp_input, temp_db)
                    assert len(df) == 2
                    assert 'PrimaryEmotionLabel' in df.columns
        finally:
            Path(temp_input).unlink()
            Path(temp_db).unlink()
    
    def test_run_full_pipeline_unsupported_format(self):
        """Test pipeline with unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_input = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                run_full_pipeline(temp_input, temp_db)
        finally:
            Path(temp_input).unlink()
            Path(temp_db).unlink()
    
    def test_run_full_pipeline_with_checkpoint(self):
        """Test pipeline with checkpoint directory."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("UserID,Text,Timestamp\n")
            f.write("user1,hello,2024-01-01 10:00:00\n")
            temp_input = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        with tempfile.TemporaryDirectory() as temp_checkpoint:
            try:
                with patch('src.pipeline.EmotionModel') as mock_model_class:
                    mock_model = MagicMock()
                    mock_model_class.return_value = mock_model
                    
                    with patch('src.pipeline.run_inference_pipeline') as mock_inference:
                        mock_inference.return_value = pd.DataFrame({
                            'UserID': ['user1'],
                            'Timestamp': pd.to_datetime(['2024-01-01 10:00:00']),
                            'NormalizedText': ['hello'],
                            'PrimaryEmotionLabel': ['joy'],
                            'IntensityScore_Primary': [0.8],
                            'Sequence': ['hello']
                        })
                        
                        df = run_full_pipeline(
                            temp_input,
                            temp_db,
                            checkpoint_dir=temp_checkpoint
                        )
                        
                        # Check checkpoints were created
                        checkpoint_files = list(Path(temp_checkpoint).glob("*.parquet"))
                        assert len(checkpoint_files) > 0
            finally:
                Path(temp_input).unlink()
                Path(temp_db).unlink()
    
    def test_run_full_pipeline_custom_model(self):
        """Test pipeline with custom model name."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("UserID,Text,Timestamp\n")
            f.write("user1,hello,2024-01-01 10:00:00\n")
            temp_input = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            with patch('src.pipeline.run_inference_pipeline') as mock_inference:
                mock_inference.return_value = pd.DataFrame({
                    'UserID': ['user1'],
                    'Timestamp': pd.to_datetime(['2024-01-01 10:00:00']),
                    'NormalizedText': ['hello'],
                    'PrimaryEmotionLabel': ['joy'],
                    'IntensityScore_Primary': [0.8],
                    'Sequence': ['hello']
                })
                
                # Mock EmotionModel to verify it's called with custom model
                with patch('src.pipeline.EmotionModel') as mock_model_class:
                    mock_model = MagicMock()
                    mock_model_class.return_value = mock_model
                    
                    df = run_full_pipeline(
                        temp_input,
                        temp_db,
                        model_name="custom-model"
                    )
                    
                    # Verify custom model was used
                    mock_model_class.assert_called_once()
                    call_args = mock_model_class.call_args
                    assert call_args[1]['model_name'] == "custom-model"
        finally:
            Path(temp_input).unlink()
            Path(temp_db).unlink()

