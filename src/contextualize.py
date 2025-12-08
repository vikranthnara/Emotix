"""
Layer 3: Contextualization
Multi-turn retrieval and sequence formatting for model input.
"""

import pandas as pd
from typing import Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def create_sequence_for_model(utterance: str, 
                             history: pd.DataFrame,
                             max_context_turns: int = 3,  # Reduced from 5 for better focus
                             sep_token: str = "[SEP]",
                             strategy: Optional[any] = None,
                             format_type: str = "standard") -> str:
    """
    Create sequence format: UTTERANCE [SEP] CONTEXT for model input.
    
    Retrieves recent history and formats it as a context sequence.
    The context helps resolve ambiguity in the current utterance.
    
    Args:
        utterance: Current utterance text (normalized)
        history: DataFrame of previous utterances (from fetch_history)
        max_context_turns: Maximum number of previous turns to include
        sep_token: Separator token between utterance and context
        strategy: Optional context selection strategy (from context_strategies)
        format_type: Format type ("standard", "reverse", "weighted", "concatenated")
        
    Returns:
        Formatted sequence string: "UTTERANCE [SEP] CONTEXT"
    """
    if history.empty:
        return utterance
    
    # Use strategy if provided
    if strategy is not None:
        from src.context_strategies import create_sequence_with_strategy
        current_timestamp = pd.Timestamp.now()  # Could be passed as parameter
        return create_sequence_with_strategy(
            utterance, history, strategy, current_timestamp,
            max_context_turns, sep_token, format_type
        )
    
    # Default behavior (backward compatible)
    # Get most recent turns (history is already ordered by timestamp ASC)
    # Use fewer turns to avoid diluting the main utterance
    recent_history = history.tail(min(max_context_turns, 3))  # Cap at 3
    
    # Concatenate historical normalized texts (only last 2 for focus)
    context_texts = recent_history['NormalizedText'].tolist()
    # Use only last 2 context items to keep sequence focused
    context = ' '.join(context_texts[-2:]) if len(context_texts) > 2 else ' '.join(context_texts)
    
    # Format: UTTERANCE [SEP] CONTEXT
    # Ensure utterance is prominent (comes first)
    sequence = f"{utterance} {sep_token} {context}"
    
    # Truncate if too long (models have token limits)
    # Keep utterance intact, truncate context if needed
    max_length = 512  # Typical model limit
    if len(sequence) > max_length:
        # Keep full utterance, truncate context
        utterance_len = len(utterance) + len(sep_token) + 1
        available = max_length - utterance_len
        if available > 0:
            context = context[:available]
            sequence = f"{utterance} {sep_token} {context}"
        else:
            # If utterance itself is too long, just return it
            sequence = utterance[:max_length]
    
    logger.debug(f"Created sequence with {len(recent_history)} context turns")
    return sequence


def create_sequences_batch(df: pd.DataFrame,
                          persistence,
                          max_context_turns: int = 5,
                          sep_token: str = "[SEP]",
                          journal_id: Optional[int] = None) -> pd.DataFrame:
    """
    Create sequences for a batch of utterances with their context.
    
    Args:
        df: DataFrame with UserID, Timestamp, NormalizedText columns
        persistence: MWBPersistence instance for history retrieval
        max_context_turns: Maximum number of previous turns to include
        sep_token: Separator token
        journal_id: Optional journal ID to filter context by journal
        
    Returns:
        DataFrame with added 'Sequence' column
    """
    logger.info(f"Creating sequences for {len(df)} utterances")
    
    # Early return for empty DataFrame
    if df.empty or len(df) == 0:
        logger.warning("Empty DataFrame provided to create_sequences_batch")
        df = df.copy()
        if 'Sequence' not in df.columns:
            df['Sequence'] = []
        return df
    
    # Ensure input DataFrame Timestamp column is datetime type upfront
    if 'Timestamp' in df.columns:
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        # Drop rows with invalid timestamps
        df = df.dropna(subset=['Timestamp'])
    
    # Check again after timestamp conversion
    if df.empty or len(df) == 0:
        logger.warning("DataFrame became empty after timestamp conversion")
        df = df.copy()
        if 'Sequence' not in df.columns:
            df['Sequence'] = []
        return df
    
    # Ensure required columns exist
    required_cols = ['UserID', 'Timestamp', 'NormalizedText']
    missing = set(required_cols) - set(df.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}")
        df = df.copy()
        if 'Sequence' not in df.columns:
            df['Sequence'] = []
        return df
    
    # Batch fetch user histories to avoid N+1 query problem
    unique_users = df['UserID'].unique()
    logger.info(f"Batch fetching histories for {len(unique_users)} unique users (journal_id={journal_id})")
    user_histories = {}
    for user_id in unique_users:
        user_histories[user_id] = persistence.fetch_history(
            user_id,
            journal_id=journal_id,  # Filter by journal for context isolation
            limit=max_context_turns * 2  # Fetch more to filter by timestamp
        )
    logger.info(f"Fetched histories for {len(user_histories)} users")
    
    sequences = []
    
    for _, row in df.iterrows():
        user_id = row['UserID']
        # Timestamp should already be datetime from above, but ensure it
        timestamp = pd.to_datetime(row['Timestamp'])
        utterance = row['NormalizedText']
        
        # Use cached history instead of fetching per utterance
        history = user_histories.get(user_id, pd.DataFrame())
        
        # ALWAYS convert timestamps before any comparison - SQLite returns strings
        if not history.empty and 'Timestamp' in history.columns:
            history = history.copy()
            
            # Convert each timestamp individually to ensure all are converted
            converted_timestamps = []
            for ts_val in history['Timestamp']:
                if isinstance(ts_val, str):
                    converted = pd.to_datetime(ts_val, errors='coerce')
                elif isinstance(ts_val, pd.Timestamp):
                    converted = ts_val
                else:
                    converted = pd.to_datetime(ts_val, errors='coerce')
                converted_timestamps.append(converted)
            
            # Replace the column with converted values - force datetime64[ns] dtype
            history['Timestamp'] = pd.Series(converted_timestamps, index=history.index, dtype='datetime64[ns]')
            
            # Drop any rows where timestamp conversion failed (NaT values)
            history = history.dropna(subset=['Timestamp'])
        
        # Filter to only include history before current timestamp
        if not history.empty and 'Timestamp' in history.columns:
            # Double-check dtype is datetime before comparison
            # This is a safety check in case conversion above didn't work
            if history['Timestamp'].dtype == 'object' or str(history['Timestamp'].dtype) != 'datetime64[ns]':
                # Force conversion one more time if needed
                logger.warning(f"Timestamp dtype is {history['Timestamp'].dtype}, forcing conversion")
                history['Timestamp'] = pd.to_datetime(history['Timestamp'], errors='coerce')
                history = history.dropna(subset=['Timestamp'])
            
            # Ensure timestamp is also datetime
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.to_datetime(timestamp)
            
            # Now both sides are guaranteed to be datetime64[ns], safe to compare
            if not history.empty:
                try:
                    history = history[history['Timestamp'] < timestamp]
                except TypeError as e:
                    # Last resort error handling - log and continue without filtering
                    logger.error(f"Timestamp comparison failed: {e}. History dtype: {history['Timestamp'].dtype}, timestamp type: {type(timestamp)}")
                    # Continue without filtering rather than crashing the pipeline
                    pass
        
        # Create sequence
        sequence = create_sequence_for_model(
            utterance,
            history,
            max_context_turns=max_context_turns,
            sep_token=sep_token
        )
        sequences.append(sequence)
    
    df = df.copy()
    df['Sequence'] = sequences
    
    logger.info("Sequence creation complete")
    return df


def extract_context_features(history: pd.DataFrame) -> dict:
    """
    Extract contextual features from history for enhanced modeling.
    
    Args:
        history: DataFrame of previous utterances
        
    Returns:
        Dictionary of contextual features
    """
    features = {
        'num_previous_turns': len(history),
        'avg_previous_intensity': None,
        'recent_emotions': [],
        'time_since_last': None
    }
    
    if history.empty:
        return features
    
    # Average intensity from previous predictions
    if 'IntensityScore_Primary' in history.columns:
        intensities = history['IntensityScore_Primary'].dropna()
        if len(intensities) > 0:
            features['avg_previous_intensity'] = float(intensities.mean())
    
    # Recent emotion labels
    if 'PrimaryEmotionLabel' in history.columns:
        recent_labels = history['PrimaryEmotionLabel'].dropna().tail(3).tolist()
        features['recent_emotions'] = recent_labels
    
    # Time since last utterance
    if 'Timestamp' in history.columns and len(history) > 0:
        last_timestamp = pd.to_datetime(history['Timestamp'].iloc[-1])
        # This would need current timestamp to calculate, placeholder for now
        features['time_since_last'] = None
    
    return features

