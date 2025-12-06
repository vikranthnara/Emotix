"""
Context selection and formatting strategies for improved accuracy.
"""

import pandas as pd
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ContextStrategy:
    """Base class for context selection strategies."""
    
    def select_context(self, history: pd.DataFrame, 
                      current_timestamp: pd.Timestamp,
                      max_turns: int = 3) -> pd.DataFrame:
        """Select relevant context from history."""
        raise NotImplementedError


class RecentContextStrategy(ContextStrategy):
    """Select most recent context turns."""
    
    def select_context(self, history: pd.DataFrame,
                      current_timestamp: pd.Timestamp,
                      max_turns: int = 3) -> pd.DataFrame:
        """Select most recent N turns."""
        if history.empty:
            return history
        
        # Get most recent turns
        return history.tail(max_turns)


class SameDayContextStrategy(ContextStrategy):
    """Select context from same day only."""
    
    def select_context(self, history: pd.DataFrame,
                      current_timestamp: pd.Timestamp,
                      max_turns: int = 3) -> pd.DataFrame:
        """Select context from same day."""
        if history.empty:
            return history
        
        # Filter to same day
        history['Date'] = pd.to_datetime(history['Timestamp']).dt.date
        current_date = pd.to_datetime(current_timestamp).date()
        
        same_day = history[history['Date'] == current_date].copy()
        same_day = same_day.drop(columns=['Date'])
        
        # Get most recent from same day
        return same_day.tail(max_turns)


class EmotionalContextStrategy(ContextStrategy):
    """Select only emotional context (skip neutral)."""
    
    def select_context(self, history: pd.DataFrame,
                      current_timestamp: pd.Timestamp,
                      max_turns: int = 3) -> pd.DataFrame:
        """Select context with clear emotions (skip neutral)."""
        if history.empty:
            return history
        
        # Filter out neutral/low-intensity predictions
        if 'PrimaryEmotionLabel' in history.columns:
            # Skip neutral labels
            emotional = history[history['PrimaryEmotionLabel'] != 'neutral'].copy()
        else:
            emotional = history.copy()
        
        # Filter by intensity if available
        if 'IntensityScore_Primary' in history.columns:
            emotional = emotional[emotional['IntensityScore_Primary'] >= 0.5]
        
        # Get most recent emotional context
        return emotional.tail(max_turns)


class WeightedContextStrategy(ContextStrategy):
    """Select context with recency weighting."""
    
    def select_context(self, history: pd.DataFrame,
                      current_timestamp: pd.Timestamp,
                      max_turns: int = 3) -> pd.DataFrame:
        """Select context with recency weighting."""
        if history.empty:
            return history
        
        # Calculate time differences
        history = history.copy()
        history['TimeDiff'] = (current_timestamp - pd.to_datetime(history['Timestamp'])).dt.total_seconds()
        
        # Weight by recency (more recent = higher weight)
        # Recent = last hour gets weight 1.0, older gets less
        history['RecencyWeight'] = 1.0 / (1.0 + history['TimeDiff'] / 3600)  # Decay over hours
        
        # Sort by recency weight and take top N
        history = history.sort_values('RecencyWeight', ascending=False)
        
        return history.head(max_turns).drop(columns=['TimeDiff', 'RecencyWeight'])


class HybridContextStrategy(ContextStrategy):
    """Combine multiple strategies."""
    
    def __init__(self, strategies: List[Tuple[ContextStrategy, float]]):
        """
        Initialize with weighted strategies.
        
        Args:
            strategies: List of (strategy, weight) tuples
        """
        self.strategies = strategies
    
    def select_context(self, history: pd.DataFrame,
                      current_timestamp: pd.Timestamp,
                      max_turns: int = 3) -> pd.DataFrame:
        """Select context using hybrid approach."""
        if history.empty:
            return history
        
        # For now, use first strategy (can be enhanced)
        if self.strategies:
            return self.strategies[0][0].select_context(history, current_timestamp, max_turns)
        return history.tail(max_turns)


def create_sequence_with_strategy(utterance: str,
                                 history: pd.DataFrame,
                                 strategy: ContextStrategy,
                                 current_timestamp: pd.Timestamp,
                                 max_context_turns: int = 3,
                                 sep_token: str = "[SEP]",
                                 format_type: str = "standard") -> str:
    """
    Create sequence using specified context strategy and format.
    
    Args:
        utterance: Current utterance
        history: Full history DataFrame
        strategy: Context selection strategy
        current_timestamp: Current timestamp
        max_context_turns: Max turns to include
        sep_token: Separator token
        format_type: "standard", "weighted", "reverse", "repeated"
        
    Returns:
        Formatted sequence string
    """
    if history.empty:
        return utterance
    
    # Select context using strategy
    selected_context = strategy.select_context(history, current_timestamp, max_context_turns)
    
    if selected_context.empty:
        return utterance
    
    # Get context texts
    context_texts = selected_context['NormalizedText'].tolist()
    
    # Apply format type
    if format_type == "standard":
        # Standard: utterance [SEP] context
        context = ' '.join(context_texts[-2:])  # Last 2 for focus
        sequence = f"{utterance} {sep_token} {context}"
    
    elif format_type == "reverse":
        # Reverse: context [SEP] utterance
        context = ' '.join(context_texts[-2:])
        sequence = f"{context} {sep_token} {utterance}"
    
    elif format_type == "weighted":
        # Weighted: utterance [SEP] utterance context (repeat utterance)
        context = ' '.join(context_texts[-2:])
        sequence = f"{utterance} {sep_token} {utterance} {context}"
    
    elif format_type == "concatenated":
        # Concatenated: utterance context (no separator)
        context = ' '.join(context_texts[-2:])
        sequence = f"{utterance} {context}"
    
    else:
        # Default to standard
        context = ' '.join(context_texts[-2:])
        sequence = f"{utterance} {sep_token} {context}"
    
    # Truncate if too long
    max_length = 512
    if len(sequence) > max_length:
        utterance_len = len(utterance) + len(sep_token) + 1
        available = max_length - utterance_len
        if available > 0:
            context = context[:available]
            if format_type == "standard":
                sequence = f"{utterance} {sep_token} {context}"
            elif format_type == "reverse":
                sequence = f"{context} {sep_token} {utterance}"
            else:
                sequence = utterance[:max_length]
        else:
            sequence = utterance[:max_length]
    
    return sequence


def get_model_separator_token(model) -> str:
    """
    Get the model's actual separator token.
    
    Args:
        model: EmotionModel instance
        
    Returns:
        Separator token string
    """
    try:
        if hasattr(model, 'tokenizer'):
            # Try to get sep_token from tokenizer
            if hasattr(model.tokenizer, 'sep_token') and model.tokenizer.sep_token:
                return model.tokenizer.sep_token
            # Try special_tokens_map
            if hasattr(model.tokenizer, 'special_tokens_map'):
                special = model.tokenizer.special_tokens_map
                if 'sep_token' in special:
                    return special['sep_token']
    except:
        pass
    
    # Default fallback
    return "[SEP]"

