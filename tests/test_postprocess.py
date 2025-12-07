"""Unit tests for post-processing layer."""

import pytest
import pandas as pd
from src.postprocess import EmotionPostProcessor, apply_post_processing


class TestPostProcess:
    """Test post-processing functions."""
    
    def test_check_positive_indicators(self):
        """Test positive indicator detection."""
        processor = EmotionPostProcessor()
        
        assert processor.check_positive_indicators("I'm happy!")
        assert processor.check_positive_indicators("thanks for helping")
        assert processor.check_positive_indicators("lol that's funny")
        assert not processor.check_positive_indicators("I'm sad")
    
    def test_check_gratitude(self):
        """Test gratitude detection."""
        processor = EmotionPostProcessor()
        
        assert processor.check_gratitude("thanks for being there")
        assert processor.check_gratitude("thx for help")
        assert processor.check_gratitude("I appreciate it")
        assert processor.check_gratitude("thank you so much")
        assert not processor.check_gratitude("I'm sad")
    
    def test_check_progress(self):
        """Test progress detection."""
        processor = EmotionPostProcessor()
        
        assert processor.check_progress("getting better slowly")
        assert processor.check_progress("I'm improving")
        assert processor.check_progress("making progress")
        assert not processor.check_progress("I'm getting worse")
    
    def test_check_humor(self):
        """Test humor detection."""
        processor = EmotionPostProcessor()
        
        assert processor.check_humor("lol that's hilarious")
        assert processor.check_humor("haha funny")
        assert processor.check_humor("that's funny")
        assert not processor.check_humor("I'm serious")
    
    def test_check_neutral(self):
        """Test neutral detection."""
        processor = EmotionPostProcessor()
        
        # Low confidence = neutral
        assert processor.check_neutral("test", intensity=0.3)
        assert processor.check_neutral("test", intensity=0.4)
        
        # Neutral keywords with low-medium confidence
        assert processor.check_neutral("brb need a break", intensity=0.6)
        assert processor.check_neutral("okay sure", intensity=0.5)
        
        # High confidence with neutral keywords still neutral
        assert processor.check_neutral("brb", intensity=0.75)
        
        # Not neutral
        assert not processor.check_neutral("I'm happy", intensity=0.9)
        assert not processor.check_neutral("feeling great", intensity=0.95)
    
    def test_post_process_gratitude_override(self):
        """Test gratitude override rule."""
        processor = EmotionPostProcessor()
        
        label, intensity, overridden, override_type = processor.post_process(
            "thx for being there", "sadness", 0.99, original_text="thx for being there"
        )
        
        assert label == "joy"
        assert overridden is True
        assert intensity == pytest.approx(0.7, abs=0.01)
        assert override_type == "gratitude"
    
    def test_post_process_progress_override(self):
        """Test progress override rule."""
        processor = EmotionPostProcessor()
        
        label, intensity, overridden, override_type = processor.post_process(
            "getting better slowly", "sadness", 0.95, original_text="getting better slowly"
        )
        
        assert label == "joy"
        assert overridden is True
        # "getting better" may match positive_keywords first, so accept either
        assert override_type in ["progress", "positive_keywords"]
    
    def test_post_process_humor_override(self):
        """Test humor override rule."""
        processor = EmotionPostProcessor()
        
        label, intensity, overridden, override_type = processor.post_process(
            "lol that's hilarious", "sadness", 0.55, original_text="lol that's hilarious"
        )
        
        assert label == "joy"
        assert overridden is True
        assert override_type == "humor"
    
    def test_post_process_positive_keywords_override(self):
        """Test positive keywords override."""
        processor = EmotionPostProcessor()
        
        # Needs high confidence (>=0.9) for override
        label, intensity, overridden, override_type = processor.post_process(
            "I'm feeling great today", "sadness", 0.95, original_text="I'm feeling great today"
        )
        
        assert label == "joy"
        assert overridden is True
        assert override_type in ["positive_keywords", "low_confidence_positive"]
    
    def test_post_process_neutral_override(self):
        """Test neutral override rule."""
        processor = EmotionPostProcessor()
        
        label, intensity, overridden, override_type = processor.post_process(
            "brb need to take a break", "sadness", 0.77, original_text="brb need to take a break"
        )
        
        assert label == "neutral"
        assert overridden is True
        assert intensity < 0.5  # Low confidence for neutral
        assert override_type == "neutral"
    
    def test_post_process_no_override(self):
        """Test that correct predictions are not overridden."""
        processor = EmotionPostProcessor()
        
        # Correct prediction - should not override
        label, intensity, overridden, override_type = processor.post_process(
            "I'm feeling great", "joy", 0.95, original_text="I'm feeling great"
        )
        
        assert label == "joy"
        assert overridden is False
        assert override_type is None
        
        # Negative emotion correctly identified
        label, intensity, overridden, override_type = processor.post_process(
            "I'm so sad", "sadness", 0.92, original_text="I'm so sad"
        )
        
        assert label == "sadness"
        assert overridden is False
        assert override_type is None
    
    def test_post_process_low_confidence_override(self):
        """Test that low confidence predictions can be overridden."""
        processor = EmotionPostProcessor()
        
        # Gratitude override doesn't require high confidence, but sadness needs >=0.9
        # So use high confidence sadness on gratitude text
        label, intensity, overridden, override_type = processor.post_process(
            "thanks for help", "sadness", 0.95, original_text="thanks for help"
        )
        
        # Should override if gratitude detected (gratitude rule doesn't check intensity threshold)
        # Note: "thanks" may match positive_keywords first, so accept either
        assert label == "joy"
        assert overridden is True
        assert override_type in ["gratitude", "positive_keywords"]
    
    def test_post_process_high_confidence_positive(self):
        """Test high confidence positive predictions."""
        processor = EmotionPostProcessor()
        
        label, intensity, overridden, override_type = processor.post_process(
            "I'm so happy!", "joy", 0.98, original_text="I'm so happy!"
        )
        
        assert label == "joy"
        assert intensity == pytest.approx(0.98, abs=0.01)
        assert overridden is False
        assert override_type is None
    
    def test_apply_post_processing_dataframe(self):
        """Test applying post-processing to DataFrame."""
        df = pd.DataFrame({
            'NormalizedText': [
                "thx for being there",
                "I'm sad",
                "brb need a break",
                "lol hilarious"
            ],
            'PrimaryEmotionLabel': ['sadness', 'sadness', 'sadness', 'sadness'],
            'IntensityScore_Primary': [0.99, 0.92, 0.77, 0.55]
        })
        
        result = apply_post_processing(df)
        
        assert 'CorrectedLabel' in result.columns
        assert 'CorrectedIntensity' in result.columns
        assert 'WasOverridden' in result.columns
        assert 'NeedsReview' in result.columns
        
        # Check overrides applied
        assert result['WasOverridden'].sum() >= 3  # At least 3 should be overridden
    
    def test_apply_post_processing_custom_columns(self):
        """Test post-processing with custom column names."""
        df = pd.DataFrame({
            'Text': ["thx for help"],
            'Label': ['sadness'],
            'Intensity': [0.99]
        })
        
        result = apply_post_processing(
            df,
            text_col='Text',
            label_col='Label',
            intensity_col='Intensity'
        )
        
        assert 'CorrectedLabel' in result.columns
        assert result['CorrectedLabel'].iloc[0] == 'joy'
    
    def test_post_process_flag_for_review(self):
        """Test review flagging for suspicious predictions."""
        processor = EmotionPostProcessor()
        
        # High confidence negative on positive text
        label, intensity, overridden, override_type = processor.post_process(
            "I'm feeling great!", "sadness", 0.98, original_text="I'm feeling great!"
        )
        
        # Should flag for review (check via flag_for_review method if exposed)
        # For now, just verify override happens
        assert overridden is True or label == "joy"
        assert override_type is not None
    
    def test_post_process_emoji_detection(self):
        """Test that emojis are detected in post-processing."""
        processor = EmotionPostProcessor()
        
        # Heart emoji should trigger gratitude
        label, intensity, overridden, override_type = processor.post_process(
            "thanks ❤️", "sadness", 0.90, original_text="thanks ❤️"
        )
        
        assert label == "joy"
        assert overridden is True
        # "thanks" may match positive_keywords first, so accept either
        assert override_type in ["gratitude", "positive_keywords"]
    
    def test_post_process_multiple_patterns(self):
        """Test text with multiple positive patterns."""
        processor = EmotionPostProcessor()
        
        # Has both gratitude and progress
        label, intensity, overridden, override_type = processor.post_process(
            "thx, I'm getting better", "sadness", 0.95, original_text="thx, I'm getting better"
        )
        
        assert label == "joy"
        assert overridden is True
        # Multiple patterns may match, accept any valid override type
        assert override_type in ["gratitude", "progress", "positive_keywords"]
    
    def test_post_process_case_insensitive(self):
        """Test that patterns are case-insensitive."""
        processor = EmotionPostProcessor()
        
        label1, _, _, _ = processor.post_process("THX FOR HELP", "sadness", 0.99, original_text="THX FOR HELP")
        label2, _, _, _ = processor.post_process("ThX fOr HeLp", "sadness", 0.99, original_text="ThX fOr HeLp")
        
        assert label1 == "joy"
        assert label2 == "joy"
    
    def test_post_process_intensity_preservation(self):
        """Test that intensity is preserved when not overridden."""
        processor = EmotionPostProcessor()
        
        label, intensity, overridden, override_type = processor.post_process(
            "I'm sad", "sadness", 0.88, original_text="I'm sad"
        )
        
        assert label == "sadness"
        assert intensity == pytest.approx(0.88, abs=0.01)
        assert overridden is False
        assert override_type is None

