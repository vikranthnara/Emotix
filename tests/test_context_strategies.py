"""Unit tests for context strategies."""

import pytest
import pandas as pd
from datetime import datetime
from src.context_strategies import (
    ContextStrategy,
    RecentContextStrategy,
    SameDayContextStrategy,
    EmotionalContextStrategy,
    WeightedContextStrategy,
    HybridContextStrategy,
    create_sequence_with_strategy,
    get_model_separator_token
)


class TestContextStrategies:
    """Test context selection strategies."""
    
    def test_recent_context_strategy_empty(self):
        """Test recent strategy with empty history."""
        strategy = RecentContextStrategy()
        history = pd.DataFrame()
        timestamp = pd.Timestamp('2024-01-01')
        
        result = strategy.select_context(history, timestamp, max_turns=3)
        assert result.empty
    
    def test_recent_context_strategy(self):
        """Test recent strategy selects most recent turns."""
        strategy = RecentContextStrategy()
        history = pd.DataFrame({
            'NormalizedText': [f'msg{i}' for i in range(10)],
            'Timestamp': pd.date_range('2024-01-01', periods=10)
        })
        timestamp = pd.Timestamp('2024-01-01 12:00:00')
        
        result = strategy.select_context(history, timestamp, max_turns=3)
        assert len(result) == 3
        # Should get last 3
        assert 'msg9' in result['NormalizedText'].values
        assert 'msg7' in result['NormalizedText'].values
        assert 'msg6' not in result['NormalizedText'].values
    
    def test_same_day_context_strategy(self):
        """Test same-day strategy filters by date."""
        strategy = SameDayContextStrategy()
        history = pd.DataFrame({
            'NormalizedText': ['yesterday', 'today1', 'today2'],
            'Timestamp': [
                pd.Timestamp('2024-01-01 23:00:00'),
                pd.Timestamp('2024-01-02 10:00:00'),
                pd.Timestamp('2024-01-02 11:00:00')
            ]
        })
        timestamp = pd.Timestamp('2024-01-02 12:00:00')
        
        result = strategy.select_context(history, timestamp, max_turns=5)
        # Should only include same-day entries
        assert len(result) == 2
        assert 'today1' in result['NormalizedText'].values
        assert 'today2' in result['NormalizedText'].values
        assert 'yesterday' not in result['NormalizedText'].values
    
    def test_same_day_context_strategy_empty(self):
        """Test same-day strategy with no same-day entries."""
        strategy = SameDayContextStrategy()
        history = pd.DataFrame({
            'NormalizedText': ['yesterday'],
            'Timestamp': [pd.Timestamp('2024-01-01')]
        })
        timestamp = pd.Timestamp('2024-01-02')
        
        result = strategy.select_context(history, timestamp, max_turns=3)
        assert result.empty
    
    def test_emotional_context_strategy(self):
        """Test emotional strategy filters neutral."""
        strategy = EmotionalContextStrategy()
        history = pd.DataFrame({
            'NormalizedText': ['happy', 'neutral', 'sad'],
            'PrimaryEmotionLabel': ['joy', 'neutral', 'sadness'],
            'IntensityScore_Primary': [0.9, 0.3, 0.8]
        })
        timestamp = pd.Timestamp('2024-01-01')
        
        result = strategy.select_context(history, timestamp, max_turns=5)
        # Should filter out neutral
        assert len(result) == 2
        assert 'neutral' not in result['NormalizedText'].values
    
    def test_emotional_context_strategy_intensity_filter(self):
        """Test emotional strategy filters by intensity."""
        strategy = EmotionalContextStrategy()
        history = pd.DataFrame({
            'NormalizedText': ['high', 'low'],
            'PrimaryEmotionLabel': ['joy', 'sadness'],
            'IntensityScore_Primary': [0.8, 0.3]  # Low intensity
        })
        timestamp = pd.Timestamp('2024-01-01')
        
        result = strategy.select_context(history, timestamp, max_turns=5)
        # Should filter out low intensity
        assert len(result) == 1
        assert 'high' in result['NormalizedText'].values
    
    def test_emotional_context_strategy_no_labels(self):
        """Test emotional strategy without emotion labels."""
        strategy = EmotionalContextStrategy()
        history = pd.DataFrame({
            'NormalizedText': ['msg1', 'msg2'],
            'Timestamp': pd.date_range('2024-01-01', periods=2)
        })
        timestamp = pd.Timestamp('2024-01-01')
        
        result = strategy.select_context(history, timestamp, max_turns=3)
        # Should return all if no labels available
        assert len(result) == 2
    
    def test_weighted_context_strategy(self):
        """Test weighted strategy weights by recency."""
        strategy = WeightedContextStrategy()
        history = pd.DataFrame({
            'NormalizedText': ['old', 'recent'],
            'Timestamp': [
                pd.Timestamp('2024-01-01 08:00:00'),
                pd.Timestamp('2024-01-01 11:00:00')
            ]
        })
        timestamp = pd.Timestamp('2024-01-01 12:00:00')
        
        result = strategy.select_context(history, timestamp, max_turns=2)
        # Should prioritize recent
        assert len(result) == 2
        # Most recent should be first
        assert result['NormalizedText'].iloc[0] == 'recent'
    
    def test_hybrid_context_strategy(self):
        """Test hybrid strategy."""
        recent = RecentContextStrategy()
        strategies = [(recent, 1.0)]
        hybrid = HybridContextStrategy(strategies)
        
        history = pd.DataFrame({
            'NormalizedText': ['msg1', 'msg2'],
            'Timestamp': pd.date_range('2024-01-01', periods=2)
        })
        timestamp = pd.Timestamp('2024-01-01')
        
        result = hybrid.select_context(history, timestamp, max_turns=2)
        assert len(result) == 2
    
    def test_create_sequence_with_strategy_standard(self):
        """Test sequence creation with standard format."""
        strategy = RecentContextStrategy()
        history = pd.DataFrame({
            'NormalizedText': ['context1', 'context2'],
            'Timestamp': pd.date_range('2024-01-01', periods=2)
        })
        timestamp = pd.Timestamp('2024-01-01')
        
        sequence = create_sequence_with_strategy(
            "utterance", history, strategy, timestamp,
            max_context_turns=3, format_type="standard"
        )
        
        assert sequence.startswith("utterance")
        assert "[SEP]" in sequence
        assert "context" in sequence.lower()
    
    def test_create_sequence_with_strategy_reverse(self):
        """Test sequence creation with reverse format."""
        strategy = RecentContextStrategy()
        history = pd.DataFrame({
            'NormalizedText': ['context'],
            'Timestamp': [pd.Timestamp('2024-01-01')]
        })
        timestamp = pd.Timestamp('2024-01-01')
        
        sequence = create_sequence_with_strategy(
            "utterance", history, strategy, timestamp,
            max_context_turns=3, format_type="reverse"
        )
        
        assert sequence.endswith("utterance")
        assert "[SEP]" in sequence
    
    def test_create_sequence_with_strategy_weighted(self):
        """Test sequence creation with weighted format."""
        strategy = RecentContextStrategy()
        history = pd.DataFrame({
            'NormalizedText': ['context'],
            'Timestamp': [pd.Timestamp('2024-01-01')]
        })
        timestamp = pd.Timestamp('2024-01-01')
        
        sequence = create_sequence_with_strategy(
            "utterance", history, strategy, timestamp,
            max_context_turns=3, format_type="weighted"
        )
        
        # Should repeat utterance
        assert sequence.count("utterance") >= 2
    
    def test_create_sequence_with_strategy_concatenated(self):
        """Test sequence creation with concatenated format."""
        strategy = RecentContextStrategy()
        history = pd.DataFrame({
            'NormalizedText': ['context'],
            'Timestamp': [pd.Timestamp('2024-01-01')]
        })
        timestamp = pd.Timestamp('2024-01-01')
        
        sequence = create_sequence_with_strategy(
            "utterance", history, strategy, timestamp,
            max_context_turns=3, format_type="concatenated"
        )
        
        assert "[SEP]" not in sequence
        assert "utterance" in sequence
        assert "context" in sequence
    
    def test_create_sequence_with_strategy_truncation(self):
        """Test sequence truncation."""
        strategy = RecentContextStrategy()
        long_context = " ".join(["word"] * 1000)
        history = pd.DataFrame({
            'NormalizedText': [long_context],
            'Timestamp': [pd.Timestamp('2024-01-01')]
        })
        timestamp = pd.Timestamp('2024-01-01')
        
        sequence = create_sequence_with_strategy(
            "utterance", history, strategy, timestamp,
            max_context_turns=3, format_type="standard"
        )
        
        # Allow 1 char margin for rounding
        assert len(sequence) <= 513
    
    def test_create_sequence_with_strategy_empty_history(self):
        """Test sequence creation with empty history."""
        strategy = RecentContextStrategy()
        history = pd.DataFrame()
        timestamp = pd.Timestamp('2024-01-01')
        
        sequence = create_sequence_with_strategy(
            "utterance", history, strategy, timestamp,
            max_context_turns=3, format_type="standard"
        )
        
        assert sequence == "utterance"
    
    def test_get_model_separator_token_no_model(self):
        """Test separator token without model."""
        token = get_model_separator_token(None)
        assert token == "[SEP]"
    
    def test_get_model_separator_token_with_tokenizer(self):
        """Test separator token from tokenizer."""
        mock_model = type('MockModel', (), {
            'tokenizer': type('MockTokenizer', (), {
                'sep_token': '<sep>',
                'special_tokens_map': {}
            })()
        })()
        
        token = get_model_separator_token(mock_model)
        assert token == "<sep>"
    
    def test_get_model_separator_token_from_special_tokens(self):
        """Test separator token from special tokens map."""
        mock_model = type('MockModel', (), {
            'tokenizer': type('MockTokenizer', (), {
                'sep_token': None,
                'special_tokens_map': {'sep_token': '<SEP>'}
            })()
        })()
        
        token = get_model_separator_token(mock_model)
        assert token == "<SEP>"

