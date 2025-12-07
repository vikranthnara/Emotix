"""
Layer 5: Persistence
SQLite wrapper with schema creation, transaction-safe writes, and history retrieval.
"""

import sqlite3
import pandas as pd
import json
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MWBPersistence:
    """SQLite persistence layer for MWB Log with ACID transactions."""
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize persistence layer.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # MWB Log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mwb_log (
                    LogID INTEGER PRIMARY KEY AUTOINCREMENT,
                    UserID TEXT NOT NULL,
                    Timestamp DATETIME NOT NULL,
                    NormalizedText TEXT NOT NULL,
                    PrimaryEmotionLabel TEXT,
                    IntensityScore_Primary REAL,
                    OriginalEmotionLabel TEXT,
                    OriginalIntensityScore REAL,
                    AmbiguityFlag INTEGER DEFAULT 0,
                    NormalizationFlags TEXT,
                    PostProcessingOverride TEXT,
                    FlagForReview INTEGER DEFAULT 0,
                    PostProcessingReviewFlag INTEGER DEFAULT 0,
                    AnomalyDetectionFlag INTEGER DEFAULT 0,
                    HighConfidenceFlag INTEGER DEFAULT 0,
                    CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Migrate existing tables to add new columns if they don't exist
            self._migrate_schema(cursor)
            
            # Create indexes for mwb_log
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_timestamp 
                ON mwb_log (UserID, Timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON mwb_log (Timestamp)
            """)
            
            # Raw archive table (data lake)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_archive (
                    ArchiveID INTEGER PRIMARY KEY AUTOINCREMENT,
                    UserID TEXT NOT NULL,
                    Timestamp DATETIME NOT NULL,
                    RawText TEXT NOT NULL,
                    Metadata TEXT,
                    CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for raw_archive
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_archive_user_timestamp 
                ON raw_archive (UserID, Timestamp)
            """)
            
            conn.commit()
            logger.info("Database schema initialized")
        finally:
            conn.close()
    
    def _migrate_schema(self, cursor) -> None:
        """Migrate existing schema to add new columns if needed."""
        try:
            # Check if PostProcessingOverride column exists
            cursor.execute("PRAGMA table_info(mwb_log)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'PostProcessingOverride' not in columns:
                cursor.execute("""
                    ALTER TABLE mwb_log 
                    ADD COLUMN PostProcessingOverride TEXT
                """)
                logger.info("Added PostProcessingOverride column to mwb_log")
            
            if 'FlagForReview' not in columns:
                cursor.execute("""
                    ALTER TABLE mwb_log 
                    ADD COLUMN FlagForReview INTEGER DEFAULT 0
                """)
                logger.info("Added FlagForReview column to mwb_log")
            
            if 'OriginalEmotionLabel' not in columns:
                cursor.execute("""
                    ALTER TABLE mwb_log 
                    ADD COLUMN OriginalEmotionLabel TEXT
                """)
                logger.info("Added OriginalEmotionLabel column to mwb_log")
            
            if 'OriginalIntensityScore' not in columns:
                cursor.execute("""
                    ALTER TABLE mwb_log 
                    ADD COLUMN OriginalIntensityScore REAL
                """)
                logger.info("Added OriginalIntensityScore column to mwb_log")
            
            if 'PostProcessingReviewFlag' not in columns:
                cursor.execute("""
                    ALTER TABLE mwb_log 
                    ADD COLUMN PostProcessingReviewFlag INTEGER DEFAULT 0
                """)
                logger.info("Added PostProcessingReviewFlag column to mwb_log")
            
            if 'AnomalyDetectionFlag' not in columns:
                cursor.execute("""
                    ALTER TABLE mwb_log 
                    ADD COLUMN AnomalyDetectionFlag INTEGER DEFAULT 0
                """)
                logger.info("Added AnomalyDetectionFlag column to mwb_log")
            
            if 'HighConfidenceFlag' not in columns:
                cursor.execute("""
                    ALTER TABLE mwb_log 
                    ADD COLUMN HighConfidenceFlag INTEGER DEFAULT 0
                """)
                logger.info("Added HighConfidenceFlag column to mwb_log")
        except Exception as e:
            logger.warning(f"Schema migration warning (may be expected for new tables): {e}")
    
    def write_results(self, df: pd.DataFrame, 
                     archive_raw: bool = True,
                     batch_size: int = 100) -> int:
        """
        Write preprocessed results to SQLite with transaction safety.
        
        Args:
            df: DataFrame with columns: UserID, Timestamp, NormalizedText,
                NormalizationFlags, and optionally PrimaryEmotionLabel,
                IntensityScore_Primary, AmbiguityFlag
            archive_raw: If True, also archive raw text to raw_archive table
            batch_size: Number of rows to commit per transaction
            
        Returns:
            Number of rows written
        """
        required_cols = {'UserID', 'Timestamp', 'NormalizedText'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        conn = self._get_connection()
        rows_written = 0
        
        try:
            cursor = conn.cursor()
            
            # Process in batches for transaction safety
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                try:
                    for _, row in batch.iterrows():
                        # Convert timestamp to string format for SQLite
                        timestamp = pd.to_datetime(row['Timestamp'])
                        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Archive raw text if original Text column exists
                        if archive_raw and 'Text' in row:
                            cursor.execute("""
                                INSERT INTO raw_archive 
                                (UserID, Timestamp, RawText, Metadata)
                                VALUES (?, ?, ?, ?)
                            """, (
                                str(row['UserID']),
                                timestamp_str,
                                str(row['Text']),
                                json.dumps(row.get('Metadata', {}))
                            ))
                        
                        # Write to MWB log
                        normalization_flags = row.get('NormalizationFlags', {})
                        if isinstance(normalization_flags, dict):
                            normalization_flags = json.dumps(normalization_flags)
                        
                        cursor.execute("""
                            INSERT INTO mwb_log 
                            (UserID, Timestamp, NormalizedText, 
                             PrimaryEmotionLabel, IntensityScore_Primary,
                             OriginalEmotionLabel, OriginalIntensityScore,
                             AmbiguityFlag, NormalizationFlags,
                             PostProcessingOverride, FlagForReview,
                             PostProcessingReviewFlag, AnomalyDetectionFlag, HighConfidenceFlag)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(row['UserID']),
                            timestamp_str,
                            str(row['NormalizedText']),
                            row.get('PrimaryEmotionLabel'),
                            row.get('IntensityScore_Primary'),
                            row.get('OriginalEmotionLabel'),
                            row.get('OriginalIntensityScore'),
                            int(row.get('AmbiguityFlag', 0)),
                            normalization_flags,
                            row.get('PostProcessingOverride'),
                            int(row.get('FlagForReview', 0)),
                            int(row.get('PostProcessingReviewFlag', 0)),
                            int(row.get('AnomalyDetectionFlag', 0)),
                            int(row.get('HighConfidenceFlag', 0))
                        ))
                    
                    # Commit batch
                    conn.commit()
                    rows_written += len(batch)
                    logger.info(f"Committed batch: {len(batch)} rows (total: {rows_written})")
                
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error in batch {i}:{i+batch_size}, rolling back: {e}")
                    raise
        
        finally:
            conn.close()
        
        logger.info(f"Successfully wrote {rows_written} rows to database")
        return rows_written
    
    def fetch_history(self, 
                     user_id: str,
                     since_timestamp: Optional[datetime] = None,
                     limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch user history for context retrieval.
        Optimized for <100ms latency with indexing.
        
        Args:
            user_id: User identifier
            since_timestamp: Optional datetime to filter history
            limit: Optional limit on number of records
            
        Returns:
            DataFrame with history records, ordered by Timestamp (ascending)
        """
        conn = self._get_connection()
        
        try:
            query = """
                SELECT LogID, UserID, Timestamp, NormalizedText,
                       PrimaryEmotionLabel, IntensityScore_Primary,
                       OriginalEmotionLabel, OriginalIntensityScore,
                       AmbiguityFlag, NormalizationFlags,
                       PostProcessingOverride, FlagForReview,
                       PostProcessingReviewFlag, AnomalyDetectionFlag, HighConfidenceFlag
                FROM mwb_log
                WHERE UserID = ?
            """
            params = [str(user_id)]
            
            if since_timestamp:
                query += " AND Timestamp >= ?"
                # Convert datetime to string format for SQLite
                if isinstance(since_timestamp, datetime):
                    params.append(since_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    params.append(since_timestamp)
            
            query += " ORDER BY Timestamp ASC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            # Ensure Timestamp is datetime type - force conversion
            # SQLite stores as TEXT, so we need to explicitly convert
            if 'Timestamp' in df.columns and len(df) > 0:
                # Use parse_dates parameter or explicit conversion
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=False)
                # Drop any rows where conversion failed
                df = df.dropna(subset=['Timestamp'])
            
            # Parse NormalizationFlags JSON
            if 'NormalizationFlags' in df.columns:
                df['NormalizationFlags'] = df['NormalizationFlags'].apply(
                    lambda x: json.loads(x) if x and isinstance(x, str) else {}
                )
            
            logger.info(f"Fetched {len(df)} records for user {user_id}")
            return df
        
        finally:
            conn.close()
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a user."""
        conn = self._get_connection()
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_logs,
                    MIN(Timestamp) as first_log,
                    MAX(Timestamp) as last_log,
                    AVG(IntensityScore_Primary) as avg_intensity
                FROM mwb_log
                WHERE UserID = ?
            """, (str(user_id),))
            
            row = cursor.fetchone()
            return {
                'total_logs': row['total_logs'],
                'first_log': row['first_log'],
                'last_log': row['last_log'],
                'avg_intensity': row['avg_intensity']
            }
        
        finally:
            conn.close()


# Note: Layer 3 (Contextualization) and Layer 4 (Modeling) functions
# are now implemented in src/contextualize.py and src/modeling.py
# Import them as needed:
# from src.contextualize import create_sequence_for_model, create_sequences_batch
# from src.modeling import batch_for_inference, run_inference_pipeline

