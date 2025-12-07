"""Unit tests for modeling layer."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
import importlib


# Mock transformers before importing EmotionModel
def mock_transformers():
    """Create mock transformers module."""
    mock_transformers_module = MagicMock()
    mock_transformers_module.AutoTokenizer = MagicMock()
    mock_transformers_module.AutoModelForSequenceClassification = MagicMock()
    mock_transformers_module.pipeline = MagicMock()
    return mock_transformers_module


class TestModeling:
    """Test modeling functions."""
    
    @patch('src.modeling.torch')
    def test_model_init_success(self, mock_torch):
        """Test successful model initialization."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        
        # Mock transformers imports
        with patch.dict('sys.modules', {'transformers': mock_transformers()}):
            with patch('src.modeling.TRANSFORMERS_AVAILABLE', True):
                # Reload module to pick up mocks
                import src.modeling
                importlib.reload(src.modeling)
                from src.modeling import EmotionModel
                
                # Setup transformer mocks
                mock_tokenizer = MagicMock()
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                mock_model.eval.return_value = None
                mock_pipe = MagicMock()
                
                import transformers
                transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
                transformers.AutoModelForSequenceClassification.from_pretrained = MagicMock(return_value=mock_model)
                transformers.pipeline = MagicMock(return_value=mock_pipe)
                
                model = EmotionModel()
                
                assert model.model_name == "j-hartmann/emotion-english-distilroberta-base"
                assert model.device == "cpu"
    
    @patch('src.modeling.torch')
    def test_model_predict_emotion(self, mock_torch):
        """Test emotion prediction."""
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'transformers': mock_transformers()}):
            with patch('src.modeling.TRANSFORMERS_AVAILABLE', True):
                import src.modeling
                importlib.reload(src.modeling)
                from src.modeling import EmotionModel
                
                mock_tokenizer = MagicMock()
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                mock_model.eval.return_value = None
                
                mock_pipe = MagicMock()
                mock_pipe.return_value = [[
                    {'label': 'joy', 'score': 0.9},
                    {'label': 'sadness', 'score': 0.05}
                ]]
                
                import transformers
                transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
                transformers.AutoModelForSequenceClassification.from_pretrained = MagicMock(return_value=mock_model)
                transformers.pipeline = MagicMock(return_value=mock_pipe)
                
                model = EmotionModel()
                result = model.predict_emotion("I'm happy")
                
                assert 'label' in result
                assert 'intensity' in result
                assert 0.0 <= result['intensity'] <= 1.0
    
    @patch('src.modeling.torch')
    def test_model_predict_emotion_with_probs(self, mock_torch):
        """Test emotion prediction with probabilities."""
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'transformers': mock_transformers()}):
            with patch('src.modeling.TRANSFORMERS_AVAILABLE', True):
                import src.modeling
                importlib.reload(src.modeling)
                from src.modeling import EmotionModel
                
                mock_tokenizer = MagicMock()
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                mock_model.eval.return_value = None
                
                mock_pipe = MagicMock()
                mock_pipe.return_value = [[
                    {'label': 'joy', 'score': 0.9},
                    {'label': 'sadness', 'score': 0.1}
                ]]
                
                import transformers
                transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
                transformers.AutoModelForSequenceClassification.from_pretrained = MagicMock(return_value=mock_model)
                transformers.pipeline = MagicMock(return_value=mock_pipe)
                
                model = EmotionModel()
                result = model.predict_emotion("test", return_probs=True)
                
                assert 'probabilities' in result
                assert isinstance(result['probabilities'], dict)
    
    @patch('src.modeling.torch')
    def test_model_predict_batch(self, mock_torch):
        """Test batch prediction."""
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'transformers': mock_transformers()}):
            with patch('src.modeling.TRANSFORMERS_AVAILABLE', True):
                import src.modeling
                importlib.reload(src.modeling)
                from src.modeling import EmotionModel
                
                mock_tokenizer = MagicMock()
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                mock_model.eval.return_value = None
                
                mock_pipe = MagicMock()
                def mock_pipe_call(texts):
                    return [[{'label': 'joy', 'score': 0.9}] for _ in texts]
                mock_pipe.side_effect = mock_pipe_call
                
                import transformers
                transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
                transformers.AutoModelForSequenceClassification.from_pretrained = MagicMock(return_value=mock_model)
                transformers.pipeline = MagicMock(return_value=mock_pipe)
                
                model = EmotionModel()
                texts = ["I'm happy", "I'm sad"]
                results = model.predict_batch(texts, batch_size=2)
                
                assert len(results) == 2
                assert all('label' in r for r in results)
                assert all('intensity' in r for r in results)
    
    @patch('src.modeling.torch')
    def test_model_detect_ambiguity(self, mock_torch):
        """Test ambiguity detection."""
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'transformers': mock_transformers()}):
            with patch('src.modeling.TRANSFORMERS_AVAILABLE', True):
                import src.modeling
                importlib.reload(src.modeling)
                from src.modeling import EmotionModel
                
                mock_tokenizer = MagicMock()
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                mock_model.eval.return_value = None
                
                # Create a callable mock pipeline
                # The pipeline is created by transformers.pipeline() and is callable
                def mock_pipe_call(text):
                    return [[
                        {'label': 'joy', 'score': 0.4},
                        {'label': 'sadness', 'score': 0.35},
                        {'label': 'anger', 'score': 0.25}
                    ]]
                
                import transformers
                transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
                transformers.AutoModelForSequenceClassification.from_pretrained = MagicMock(return_value=mock_model)
                # pipeline() returns a callable - use the function directly
                transformers.pipeline = MagicMock(return_value=mock_pipe_call)
                
                model = EmotionModel()
                # Ensure pipeline is set correctly - manually set it to our mock function
                # The transformers.pipeline() call might not have worked correctly
                model.pipeline = mock_pipe_call
                
                # Low difference between top 2 probs (0.4 vs 0.35 = 0.05 diff) should be ambiguous
                # With threshold 0.3, diff of 0.05 < 0.3, so ambiguous
                is_ambiguous, score = model.detect_ambiguity("I'm fine", ambiguity_threshold=0.3)
                assert is_ambiguous is True
                
                # High confidence = not ambiguous (large difference)
                def mock_pipe_call2(text):
                    return [[
                        {'label': 'joy', 'score': 0.9},
                        {'label': 'sadness', 'score': 0.1}
                    ]]
                # Update the model's pipeline directly
                model.pipeline = mock_pipe_call2
                is_ambiguous, score = model.detect_ambiguity("I'm happy", ambiguity_threshold=0.3)
                assert is_ambiguous is False
    
    @patch('src.modeling.torch')
    def test_run_inference_pipeline(self, mock_torch):
        """Test inference pipeline."""
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'transformers': mock_transformers()}):
            with patch('src.modeling.TRANSFORMERS_AVAILABLE', True):
                import src.modeling
                importlib.reload(src.modeling)
                from src.modeling import EmotionModel, run_inference_pipeline
                
                mock_tokenizer = MagicMock()
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                mock_model.eval.return_value = None
                
                mock_pipe = MagicMock()
                mock_pipe.return_value = [[{'label': 'joy', 'score': 0.9}]]
                
                import transformers
                transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
                transformers.AutoModelForSequenceClassification.from_pretrained = MagicMock(return_value=mock_model)
                transformers.pipeline = MagicMock(return_value=mock_pipe)
                
                df = pd.DataFrame({
                    'Sequence': ["I'm happy", "I'm sad"],
                    'NormalizedText': ["I'm happy", "I'm sad"]
                })
                
                model = EmotionModel()
                result = run_inference_pipeline(df, model, sequence_col='Sequence')
                
                assert 'PrimaryEmotionLabel' in result.columns
                assert 'IntensityScore_Primary' in result.columns
                assert len(result) == 2
    
    @patch('src.modeling.torch')
    def test_run_inference_pipeline_with_postprocessing(self, mock_torch):
        """Test inference pipeline with post-processing."""
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'transformers': mock_transformers()}):
            with patch('src.modeling.TRANSFORMERS_AVAILABLE', True):
                import src.modeling
                importlib.reload(src.modeling)
                from src.modeling import EmotionModel, run_inference_pipeline
                
                mock_tokenizer = MagicMock()
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                mock_model.eval.return_value = None
                
                # Create a callable mock pipeline
                # The pipeline is created by transformers.pipeline() and is callable
                # Need to handle both predict_batch (multiple calls) and detect_ambiguity (single call)
                def mock_pipe_call(text):
                    # Always return sadness for the text prediction
                    return [[
                        {'label': 'sadness', 'score': 0.99},
                        {'label': 'joy', 'score': 0.01}
                    ]]
                mock_pipe = mock_pipe_call  # Use the function directly
                
                import transformers
                transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
                transformers.AutoModelForSequenceClassification.from_pretrained = MagicMock(return_value=mock_model)
                # pipeline() returns a callable
                transformers.pipeline = MagicMock(return_value=mock_pipe)
                
                df = pd.DataFrame({
                    'Sequence': ["thx for help"],
                    'NormalizedText': ["thx for help"]
                })
                
                model = EmotionModel()
                # Manually set pipeline to ensure it works
                model.pipeline = mock_pipe_call
                
                result = run_inference_pipeline(
                    df, model, 
                    sequence_col='Sequence',
                    apply_post_processing=True
                )
                
                assert 'PrimaryEmotionLabel' in result.columns
                # Post-processing should override sadness to joy for "thx"
                # The text "thx for help" should trigger gratitude override
                assert result['PrimaryEmotionLabel'].iloc[0] == 'joy'
    
    @patch('src.modeling.torch')
    def test_model_custom_device(self, mock_torch):
        """Test model with custom device."""
        mock_torch.cuda.is_available.return_value = True
        
        with patch.dict('sys.modules', {'transformers': mock_transformers()}):
            with patch('src.modeling.TRANSFORMERS_AVAILABLE', True):
                import src.modeling
                importlib.reload(src.modeling)
                from src.modeling import EmotionModel
                
                mock_tokenizer = MagicMock()
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                mock_model.eval.return_value = None
                mock_pipe = MagicMock()
                
                import transformers
                transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
                transformers.AutoModelForSequenceClassification.from_pretrained = MagicMock(return_value=mock_model)
                transformers.pipeline = MagicMock(return_value=mock_pipe)
                
                model = EmotionModel(device="cpu")
                assert model.device == "cpu"
    
    @patch('src.modeling.torch')
    def test_model_custom_model_name(self, mock_torch):
        """Test model with custom model name."""
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'transformers': mock_transformers()}):
            with patch('src.modeling.TRANSFORMERS_AVAILABLE', True):
                import src.modeling
                importlib.reload(src.modeling)
                from src.modeling import EmotionModel
                
                mock_tokenizer = MagicMock()
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                mock_model.eval.return_value = None
                mock_pipe = MagicMock()
                
                import transformers
                transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
                transformers.AutoModelForSequenceClassification.from_pretrained = MagicMock(return_value=mock_model)
                transformers.pipeline = MagicMock(return_value=mock_pipe)
                
                model = EmotionModel(model_name="custom-model")
                assert model.model_name == "custom-model"
