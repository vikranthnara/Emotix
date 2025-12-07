"""
Utility functions for logging, checkpointing, and audit trails.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def checkpoint_dataframe(df: pd.DataFrame,
                        checkpoint_dir: Path,
                        stage: str,
                        suffix: Optional[str] = None) -> Path:
    """
    Save DataFrame checkpoint as parquet for audit trail.
    
    Args:
        df: DataFrame to checkpoint
        checkpoint_dir: Directory to save checkpoints
        stage: Stage name (e.g., 'ingestion', 'preprocessing')
        suffix: Optional suffix for filename
        
    Returns:
        Path to saved checkpoint file
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stage}_{timestamp}"
    if suffix:
        filename += f"_{suffix}"
    filename += ".parquet"
    
    checkpoint_path = checkpoint_dir / filename
    df.to_parquet(checkpoint_path, index=False)
    
    logger.info(f"Checkpoint saved: {checkpoint_path} ({len(df)} rows)")
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path) -> pd.DataFrame:
    """
    Load DataFrame from checkpoint.
    
    Args:
        checkpoint_path: Path to parquet checkpoint file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_parquet(checkpoint_path)


def get_emotion_sentiment(emotion: str) -> str:
    """
    Get sentiment polarity for an emotion label.
    
    Args:
        emotion: Emotion label (e.g., 'joy', 'sadness', 'anger')
        
    Returns:
        '(positive)', '(negative)', or '(neutral)'
    """
    if not emotion or emotion.lower() == 'n/a':
        return ''
    
    emotion_lower = emotion.lower()
    
    # Positive emotions
    positive_emotions = {'joy', 'love', 'gratitude', 'excitement', 'happiness', 'optimism'}
    if emotion_lower in positive_emotions:
        return '(positive)'
    
    # Negative emotions
    negative_emotions = {'sadness', 'anger', 'fear', 'disgust', 'anxiety', 'depression', 'stress', 'worry'}
    if emotion_lower in negative_emotions:
        return '(negative)'
    
    # Neutral
    if emotion_lower == 'neutral':
        return '(neutral)'
    
    # Default to empty if unknown
    return ''

