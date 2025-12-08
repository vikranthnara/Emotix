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
from src.anomaly_detection import UserPatternAnalyzer
from src.suicidal_detection import SuicidalIdeationDetector

logger = logging.getLogger(__name__)


def run_full_pipeline(
    input_path: Union[str, Path],
    db_path: Union[str, Path],
    slang_dict_path: Optional[Union[str, Path]] = None,
    model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    max_context_turns: int = 5,
    batch_size: int = 32,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    archive_raw: bool = True,
    temperature: float = 1.5,
    journal_id: Optional[int] = None
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
        temperature: Temperature scaling for confidence calibration (default: 1.5)
                    Higher values (>1.0) reduce confidence, lower values (<1.0) increase confidence
                    Increased from 1.3 to 1.5 to further reduce overconfidence
        journal_id: Optional journal ID to filter context and store entries
        
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
        max_context_turns=max_context_turns,
        journal_id=journal_id
    )
    
    if checkpoint_dir:
        checkpoint_dataframe(df, checkpoint_dir, "contextualization")
    
    # Step 4: Modeling
    logger.info("[4/5] Modeling...")
    model = EmotionModel(model_name=model_name, temperature=temperature)  # Calibrate confidence
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
    
    # Step 4.5: Suicidal Ideation Detection (before persistence)
    logger.info("[4.5/5] Suicidal Ideation Detection...")
    detector = SuicidalIdeationDetector()
    suicidal_flags = []
    suicidal_confidences = []
    suicidal_patterns = []
    
    # Use original Text column if available, otherwise use NormalizedText
    text_col = 'Text' if 'Text' in df.columns else 'NormalizedText'
    
    for idx, row in df.iterrows():
        text = row.get(text_col, '')
        user_id = row.get('UserID', None)
        timestamp = row.get('Timestamp', None)
        
        is_detected, confidence, pattern_type = detector.process_text(
            text, user_id=user_id, timestamp=timestamp
        )
        
        suicidal_flags.append(is_detected)
        suicidal_confidences.append(confidence)
        suicidal_patterns.append(pattern_type)
    
    df['SuicidalIdeationFlag'] = suicidal_flags
    df['SuicidalIdeationConfidence'] = suicidal_confidences
    df['SuicidalIdeationPattern'] = suicidal_patterns
    
    suicidal_count = sum(suicidal_flags)
    if suicidal_count > 0:
        logger.warning(f"Detected {suicidal_count} record(s) with suicidal ideation patterns")
    
    # Step 4.6: Anomaly Detection (before persistence)
    logger.info("[4.6/5] Anomaly Detection...")
    analyzer = UserPatternAnalyzer()
    user_analysis = analyzer.analyze_all_users(df)
    flagged_users = analyzer.flag_high_risk_users(df)
    
    # Store anomaly detection flag separately
    if len(flagged_users) > 0:
        logger.warning(f"Flagged {len(flagged_users)} users for review: {flagged_users}")
        df['AnomalyDetectionFlag'] = df['UserID'].isin(flagged_users)
    else:
        df['AnomalyDetectionFlag'] = False
    
    # Combine all flag sources into FlagForReview
    # Post-processing review flag (if exists)
    post_processing_flag = df.get('PostProcessingReviewFlag', pd.Series([False] * len(df)))
    if isinstance(post_processing_flag, pd.Series):
        post_processing_flag = post_processing_flag.fillna(False).astype(bool)
    else:
        post_processing_flag = pd.Series([False] * len(df))
    
    # Anomaly detection flag
    anomaly_flag = df.get('AnomalyDetectionFlag', pd.Series([False] * len(df)))
    if isinstance(anomaly_flag, pd.Series):
        anomaly_flag = anomaly_flag.fillna(False).astype(bool)
    else:
        anomaly_flag = pd.Series([False] * len(df))
    
    # High confidence flag (if exists)
    high_conf_flag = df.get('HighConfidenceFlag', pd.Series([False] * len(df)))
    if isinstance(high_conf_flag, pd.Series):
        high_conf_flag = high_conf_flag.fillna(False).astype(bool)
    else:
        high_conf_flag = pd.Series([False] * len(df))
    
    # Suicidal ideation flag (if exists)
    suicidal_flag = df.get('SuicidalIdeationFlag', pd.Series([False] * len(df)))
    if isinstance(suicidal_flag, pd.Series):
        suicidal_flag = suicidal_flag.fillna(False).astype(bool)
    else:
        suicidal_flag = pd.Series([False] * len(df))
    
    # Combine: flag if any source flags it (suicidal ideation always flags for review)
    df['FlagForReview'] = post_processing_flag | anomaly_flag | high_conf_flag | suicidal_flag
    
    # Step 5: Persistence
    logger.info("[5/5] Persistence...")
    
    # Verify flag columns exist and log counts before persistence
    post_processing_count = df.get('PostProcessingReviewFlag', pd.Series([0] * len(df))).sum() if 'PostProcessingReviewFlag' in df.columns else 0
    anomaly_count = df.get('AnomalyDetectionFlag', pd.Series([0] * len(df))).sum() if 'AnomalyDetectionFlag' in df.columns else 0
    high_conf_count = df.get('HighConfidenceFlag', pd.Series([0] * len(df))).sum() if 'HighConfidenceFlag' in df.columns else 0
    suicidal_count = df.get('SuicidalIdeationFlag', pd.Series([0] * len(df))).sum() if 'SuicidalIdeationFlag' in df.columns else 0
    total_flag_count = df.get('FlagForReview', pd.Series([0] * len(df))).sum() if 'FlagForReview' in df.columns else 0
    
    logger.info(f"Flag counts before persistence: PostProcessing={post_processing_count}, Anomaly={anomaly_count}, HighConf={high_conf_count}, Suicidal={suicidal_count}, Total={total_flag_count}")
    
    # Ensure all flag columns are present (set to False if missing)
    if 'PostProcessingReviewFlag' not in df.columns:
        df['PostProcessingReviewFlag'] = False
        logger.warning("PostProcessingReviewFlag column missing, setting to False")
    if 'AnomalyDetectionFlag' not in df.columns:
        df['AnomalyDetectionFlag'] = False
        logger.warning("AnomalyDetectionFlag column missing, setting to False")
    if 'HighConfidenceFlag' not in df.columns:
        df['HighConfidenceFlag'] = False
        logger.warning("HighConfidenceFlag column missing, setting to False")
    if 'SuicidalIdeationFlag' not in df.columns:
        df['SuicidalIdeationFlag'] = False
        logger.warning("SuicidalIdeationFlag column missing, setting to False")
    if 'FlagForReview' not in df.columns:
        df['FlagForReview'] = False
        logger.warning("FlagForReview column missing, setting to False")
    
    rows_written = persistence.write_results(df, archive_raw=archive_raw, journal_id=journal_id)
    logger.info(f"Pipeline complete. Wrote {rows_written} rows to database.")
    
    return df

