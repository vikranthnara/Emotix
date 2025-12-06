"""
Complete Phase 2 pipeline: Ingestion → Preprocessing → Contextualization → Modeling → Persistence
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Union

from src.ingest import ingest_csv, ingest_jsonl, ingest_json
from src.preprocess import preprocess_pipeline
from src.contextualize import create_sequences_batch
from src.modeling import EmotionModel, run_inference_pipeline
from src.persistence import MWBPersistence
from src.utils import setup_logging, checkpoint_dataframe
from src.context_strategies import RecentContextStrategy

logger = logging.getLogger(__name__)


def run_full_pipeline(
    input_path: Union[str, Path],
    db_path: Union[str, Path],
    slang_dict_path: Optional[Union[str, Path]] = None,
    model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    max_context_turns: int = 5,
    batch_size: int = 32,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    archive_raw: bool = True
) -> pd.DataFrame:
    """
    Run complete Phase 2 pipeline end-to-end.
    
    Args:
        input_path: Path to input CSV/JSON/JSONL file
        db_path: Path to SQLite database
        slang_dict_path: Path to slang dictionary JSON
        model_name: Hugging Face model identifier
        max_context_turns: Maximum context turns for sequences
        batch_size: Batch size for model inference
        checkpoint_dir: Directory for checkpoints (optional)
        archive_raw: Whether to archive raw text
        
    Returns:
        Final DataFrame with all predictions
    """
    logger.info("Starting Phase 2 full pipeline")
    
    # Step 1: Ingestion
    logger.info("[1/5] Ingestion...")
    input_path = Path(input_path)
    if input_path.suffix == '.csv':
        df = ingest_csv(input_path)
    elif input_path.suffix == '.jsonl':
        df = ingest_jsonl(input_path)
    elif input_path.suffix == '.json':
        df = ingest_json(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    if checkpoint_dir:
        checkpoint_dataframe(df, checkpoint_dir, "ingestion")
    
    # Step 2: Preprocessing
    logger.info("[2/5] Preprocessing...")
    df = preprocess_pipeline(df, slang_dict_path=slang_dict_path)
    
    if checkpoint_dir:
        checkpoint_dataframe(df, checkpoint_dir, "preprocessing")
    
    # Step 3: Contextualization
    logger.info("[3/5] Contextualization...")
    persistence = MWBPersistence(db_path)
    
    # Use optimized context strategy (recent, focused)
    # Can be customized: RecentContextStrategy, SameDayContextStrategy, etc.
    context_strategy = RecentContextStrategy()
    
    df = create_sequences_batch(
        df,
        persistence,
        max_context_turns=max_context_turns
    )
    
    if checkpoint_dir:
        checkpoint_dataframe(df, checkpoint_dir, "contextualization")
    
    # Step 4: Modeling
    logger.info("[4/5] Modeling...")
    model = EmotionModel(model_name=model_name)
    df = run_inference_pipeline(
        df,
        model,
        sequence_col='Sequence',
        batch_size=batch_size,
        apply_post_processing=True,  # Enable post-processing by default
        use_context_aware=True  # Use context-aware post-processing
    )
    
    if checkpoint_dir:
        checkpoint_dataframe(df, checkpoint_dir, "modeling")
    
    # Step 5: Persistence
    logger.info("[5/5] Persistence...")
    rows_written = persistence.write_results(df, archive_raw=archive_raw)
    logger.info(f"Pipeline complete. Wrote {rows_written} rows to database.")
    
    return df

