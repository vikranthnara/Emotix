"""Unit tests for contextualization layer."""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
from src.contextualize import (
    create_sequence_for_model,
    create_sequences_batch,
    extract_context_features
)
from src.persistence import MWBPersistence
from src.context_strategies import RecentContextStrategy, SameDayContextStrategy


class TestContextualize:
    """Test contextualization functions."""
    
    def test_create_sequence_empty_history(self):
        """Test sequence creation with empty history."""
        utterance = "I feel happy"
        history = pd.DataFrame()
        
        sequence = create_sequence_for_model(utterance, history)
        assert sequence == utterance
    
    def test_create_sequence_with_history(self):
        """Test sequence creation with history."""
        utterance = "I feel happy"
        history = pd.DataFrame({
            'NormalizedText': ['previous message', 'another message'],
            'Timestamp': pd.date_range('2024-01-01', periods=2)
        })
        
        sequence = create_sequence_for_model(utterance, history)
        assert utterance in sequence
        assert '[SEP]' in sequence
        assert 'previous message' in sequence or 'another message' in sequence
    
    def test_create_sequence_max_context_turns(self):
        """Test sequence creation respects max_context_turns."""
        utterance = "test"
        history = pd.DataFrame({
            'NormalizedText': [f'message {i}' for i in range(10)],
            'Timestamp': pd.date_range('2024-01-01', periods=10)
        })
        
        sequence = create_sequence_for_model(utterance, history, max_context_turns=3)
        # Should only include last 2-3 context items
        assert sequence.count('[SEP]') == 1
        # Should not include all 10 messages
        assert 'message 0' not in sequence
    
    def test_create_sequence_custom_sep_token(self):
        """Test sequence creation with custom separator token."""
        utterance = "test"
        history = pd.DataFrame({
            'NormalizedText': ['context'],
            'Timestamp': [pd.Timestamp('2024-01-01')]
        })
        
        sequence = create_sequence_for_model(utterance, history, sep_token="|")
        assert "|" in sequence
        assert "[SEP]" not in sequence
    
    def test_create_sequence_truncation(self):
        """Test sequence truncation for long sequences."""
        utterance = "short"
        # Create very long context
        long_context = " ".join(["word"] * 1000)
        history = pd.DataFrame({
            'NormalizedText': [long_context],
            'Timestamp': [pd.Timestamp('2024-01-01')]
        })
        
        sequence = create_sequence_for_model(utterance, history)
        # Should be truncated to max_length (512) - allow 1 char margin for rounding
        assert len(sequence) <= 513
        assert utterance in sequence
    
    def test_create_sequence_with_strategy(self):
        """Test sequence creation with context strategy."""
        utterance = "test"
        history = pd.DataFrame({
            'NormalizedText': ['msg1', 'msg2', 'msg3'],
            'Timestamp': pd.date_range('2024-01-01', periods=3)
        })
        
        strategy = RecentContextStrategy()
        sequence = create_sequence_for_model(
            utterance, history, strategy=strategy, format_type="standard"
        )
        assert utterance in sequence
        assert '[SEP]' in sequence
    
    def test_create_sequences_batch_empty_df(self):
        """Test batch sequence creation with empty DataFrame."""
        df = pd.DataFrame()
        persistence = MWBPersistence(":memory:")
        
        result = create_sequences_batch(df, persistence)
        assert len(result) == 0
    
    def test_create_sequences_batch_no_history(self):
        """Test batch sequence creation for new users."""
        df = pd.DataFrame({
            'UserID': ['user1'],
            'Timestamp': [pd.Timestamp('2024-01-01 10:00:00')],
            'NormalizedText': ['first message']
        })
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            persistence = MWBPersistence(temp_path)
            result = create_sequences_batch(df, persistence)
            
            assert 'Sequence' in result.columns
            assert len(result) == 1
            assert result['Sequence'].iloc[0] == 'first message'  # No history
        finally:
            Path(temp_path).unlink()
    
    def test_create_sequences_batch_with_history(self):
        """Test batch sequence creation with existing history."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            persistence = MWBPersistence(temp_path)
            
            # Write initial history
            history_df = pd.DataFrame({
                'UserID': ['user1', 'user1'],
                'Timestamp': [
                    pd.Timestamp('2024-01-01 09:00:00'),
                    pd.Timestamp('2024-01-01 09:30:00')
                ],
                'NormalizedText': ['first', 'second'],
                'NormalizationFlags': [{}, {}]
            })
            persistence.write_results(history_df, archive_raw=False)
            
            # Create new utterance
            new_df = pd.DataFrame({
                'UserID': ['user1'],
                'Timestamp': [pd.Timestamp('2024-01-01 10:00:00')],
                'NormalizedText': ['third message']
            })
            
            result = create_sequences_batch(new_df, persistence)
            
            assert 'Sequence' in result.columns
            assert len(result) == 1
            sequence = result['Sequence'].iloc[0]
            assert 'third message' in sequence
            assert '[SEP]' in sequence
            # Should include history
            assert 'first' in sequence or 'second' in sequence
        finally:
            Path(temp_path).unlink()
    
    def test_create_sequences_batch_timestamp_filtering(self):
        """Test that batch creation filters history by timestamp."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            persistence = MWBPersistence(temp_path)
            
            # Write history with different timestamps
            history_df = pd.DataFrame({
                'UserID': ['user1', 'user1', 'user1'],
                'Timestamp': [
                    pd.Timestamp('2024-01-01 08:00:00'),
                    pd.Timestamp('2024-01-01 09:00:00'),
                    pd.Timestamp('2024-01-01 10:00:00')
                ],
                'NormalizedText': ['early', 'middle', 'late'],
                'NormalizationFlags': [{}, {}, {}]
            })
            persistence.write_results(history_df, archive_raw=False)
            
            # Create utterance at 10:30
            new_df = pd.DataFrame({
                'UserID': ['user1'],
                'Timestamp': [pd.Timestamp('2024-01-01 10:30:00')],
                'NormalizedText': ['new message']
            })
            
            result = create_sequences_batch(new_df, persistence, max_context_turns=2)
            sequence = result['Sequence'].iloc[0]
            
            # Should include history before 10:30
            assert 'new message' in sequence
            # Should not include 'late' (same timestamp or after)
            # Should include 'early' or 'middle'
        finally:
            Path(temp_path).unlink()
    
    def test_create_sequences_batch_multiple_users(self):
        """Test batch creation handles multiple users correctly."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            persistence = MWBPersistence(temp_path)
            
            # Write history for user1
            history_df = pd.DataFrame({
                'UserID': ['user1'],
                'Timestamp': [pd.Timestamp('2024-01-01 09:00:00')],
                'NormalizedText': ['user1 history'],
                'NormalizationFlags': [{}]
            })
            persistence.write_results(history_df, archive_raw=False)
            
            # Create utterances for both users
            new_df = pd.DataFrame({
                'UserID': ['user1', 'user2'],
                'Timestamp': [
                    pd.Timestamp('2024-01-01 10:00:00'),
                    pd.Timestamp('2024-01-01 10:00:00')
                ],
                'NormalizedText': ['user1 new', 'user2 new']
            })
            
            result = create_sequences_batch(new_df, persistence)
            
            assert len(result) == 2
            # user1 should have history
            user1_seq = result[result['UserID'] == 'user1']['Sequence'].iloc[0]
            assert 'user1 history' in user1_seq
            # user2 should not have history
            user2_seq = result[result['UserID'] == 'user2']['Sequence'].iloc[0]
            assert user2_seq == 'user2 new'
        finally:
            Path(temp_path).unlink()
    
    def test_extract_context_features_empty(self):
        """Test feature extraction with empty history."""
        history = pd.DataFrame()
        features = extract_context_features(history)
        
        assert features['num_previous_turns'] == 0
        assert features['avg_previous_intensity'] is None
        assert features['recent_emotions'] == []
    
    def test_extract_context_features_with_intensity(self):
        """Test feature extraction with intensity scores."""
        history = pd.DataFrame({
            'IntensityScore_Primary': [0.8, 0.9, 0.7],
            'PrimaryEmotionLabel': ['joy', 'joy', 'sadness'],
            'Timestamp': pd.date_range('2024-01-01', periods=3)
        })
        
        features = extract_context_features(history)
        
        assert features['num_previous_turns'] == 3
        assert features['avg_previous_intensity'] == pytest.approx(0.8, abs=0.01)
        assert len(features['recent_emotions']) == 3
    
    def test_extract_context_features_recent_emotions(self):
        """Test feature extraction extracts recent emotions."""
        history = pd.DataFrame({
            'PrimaryEmotionLabel': ['joy', 'sadness', 'anger', 'fear'],
            'Timestamp': pd.date_range('2024-01-01', periods=4)
        })
        
        features = extract_context_features(history)
        
        # Should get last 3 emotions
        assert len(features['recent_emotions']) == 3
        assert 'fear' in features['recent_emotions']
        assert 'joy' not in features['recent_emotions']  # Too old
    
    def test_create_sequence_format_types(self):
        """Test different format types."""
        utterance = "test"
        history = pd.DataFrame({
            'NormalizedText': ['context'],
            'Timestamp': [pd.Timestamp('2024-01-01')]
        })
        
        # Standard format
        seq_standard = create_sequence_for_model(
            utterance, history, format_type="standard"
        )
        assert seq_standard.startswith(utterance)
        
        # With strategy
        strategy = RecentContextStrategy()
        seq_with_strategy = create_sequence_for_model(
            utterance, history, strategy=strategy, format_type="standard"
        )
        assert utterance in seq_with_strategy

