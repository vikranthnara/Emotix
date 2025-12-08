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
            
            # Journals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS journals (
                    JournalID INTEGER PRIMARY KEY AUTOINCREMENT,
                    UserID TEXT NOT NULL,
                    JournalName TEXT NOT NULL,
                    CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP,
                    IsArchived INTEGER DEFAULT 0,
                    UNIQUE(UserID, JournalName, IsArchived)
                )
            """)
            
            # Create index for journals
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_journals_user_archived 
                ON journals (UserID, IsArchived)
            """)
            
            # MWB Log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mwb_log (
                    LogID INTEGER PRIMARY KEY AUTOINCREMENT,
                    UserID TEXT NOT NULL,
                    JournalID INTEGER,
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
                    SuicidalIdeationFlag INTEGER DEFAULT 0,
                    CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Migrate existing tables to add new columns if they don't exist
            self._migrate_schema(cursor)
            
            # Create indexes for mwb_log
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_journal_timestamp 
                ON mwb_log (UserID, JournalID, Timestamp)
            """)
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
                    JournalID INTEGER,
                    Timestamp DATETIME NOT NULL,
                    RawText TEXT NOT NULL,
                    Metadata TEXT,
                    CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for raw_archive
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_archive_user_journal_timestamp 
                ON raw_archive (UserID, JournalID, Timestamp)
            """)
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
            
            if 'SuicidalIdeationFlag' not in columns:
                cursor.execute("""
                    ALTER TABLE mwb_log 
                    ADD COLUMN SuicidalIdeationFlag INTEGER DEFAULT 0
                """)
                logger.info("Added SuicidalIdeationFlag column to mwb_log")
            
            if 'JournalID' not in columns:
                cursor.execute("""
                    ALTER TABLE mwb_log 
                    ADD COLUMN JournalID INTEGER
                """)
                logger.info("Added JournalID column to mwb_log")
            
            # Check raw_archive columns
            cursor.execute("PRAGMA table_info(raw_archive)")
            archive_columns = [row[1] for row in cursor.fetchall()]
            
            if 'JournalID' not in archive_columns:
                cursor.execute("""
                    ALTER TABLE raw_archive 
                    ADD COLUMN JournalID INTEGER
                """)
                logger.info("Added JournalID column to raw_archive")
        except Exception as e:
            logger.warning(f"Schema migration warning (may be expected for new tables): {e}")
    
    def write_results(self, df: pd.DataFrame, 
                     archive_raw: bool = True,
                     batch_size: int = 100,
                     journal_id: Optional[int] = None) -> int:
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
                                (UserID, JournalID, Timestamp, RawText, Metadata)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                str(row['UserID']),
                                journal_id,
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
                            (UserID, JournalID, Timestamp, NormalizedText, 
                             PrimaryEmotionLabel, IntensityScore_Primary,
                             OriginalEmotionLabel, OriginalIntensityScore,
                             AmbiguityFlag, NormalizationFlags,
                             PostProcessingOverride, FlagForReview,
                             PostProcessingReviewFlag, AnomalyDetectionFlag, HighConfidenceFlag, SuicidalIdeationFlag)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(row['UserID']),
                            journal_id,
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
                            int(row.get('HighConfidenceFlag', 0)),
                            int(row.get('SuicidalIdeationFlag', 0))
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
                     journal_id: Optional[int] = None,
                     since_timestamp: Optional[datetime] = None,
                     limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch user history for context retrieval.
        Optimized for <100ms latency with indexing.
        
        Args:
            user_id: User identifier
            journal_id: Optional journal ID to filter by journal
            since_timestamp: Optional datetime to filter history
            limit: Optional limit on number of records
            
        Returns:
            DataFrame with history records, ordered by Timestamp (ascending)
        """
        conn = self._get_connection()
        
        try:
            query = """
                SELECT LogID, UserID, JournalID, Timestamp, NormalizedText,
                       PrimaryEmotionLabel, IntensityScore_Primary,
                       OriginalEmotionLabel, OriginalIntensityScore,
                       AmbiguityFlag, NormalizationFlags,
                       PostProcessingOverride, FlagForReview,
                       PostProcessingReviewFlag, AnomalyDetectionFlag, HighConfidenceFlag, SuicidalIdeationFlag
                FROM mwb_log
                WHERE UserID = ?
            """
            params = [str(user_id)]
            
            # Filter by journal if provided
            if journal_id is not None:
                query += " AND JournalID = ?"
                params.append(journal_id)
            
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
    
    def get_user_stats(self, user_id: str, journal_id: Optional[int] = None) -> Dict:
        """Get statistics for a user, optionally filtered by journal."""
        conn = self._get_connection()
        
        try:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    COUNT(*) as total_logs,
                    MIN(Timestamp) as first_log,
                    MAX(Timestamp) as last_log,
                    AVG(IntensityScore_Primary) as avg_intensity
                FROM mwb_log
                WHERE UserID = ?
            """
            params = [str(user_id)]
            
            if journal_id is not None:
                query += " AND JournalID = ?"
                params.append(journal_id)
            
            cursor.execute(query, tuple(params))
            
            row = cursor.fetchone()
            return {
                'total_logs': row['total_logs'],
                'first_log': row['first_log'],
                'last_log': row['last_log'],
                'avg_intensity': row['avg_intensity']
            }
        
        finally:
            conn.close()
    
    def get_last_3_summary(self, user_id: str, journal_id: Optional[int] = None) -> str:
        """
        Return a formatted summary of the last 3 entries with their tags.
        
        Args:
            user_id: User identifier
            journal_id: Optional journal ID to filter by journal
            
        Returns:
            Formatted string with last 3 entries, one per line.
            Format: "[Timestamp] Text → Emotion (Intensity)"
            Returns "No entries found." if user has no history.
        """
        # Fetch more entries than needed, then take the last 3 (most recent)
        history = self.fetch_history(user_id, journal_id=journal_id, limit=10)
        
        if history.empty:
            return "No entries found."
        
        # Get last 3 entries (most recent) - history is ordered ASC, so tail(3) gets most recent
        last_3 = history.tail(3)
        
        lines = []
        for _, row in last_3.iterrows():
            timestamp = row['Timestamp']
            text = row['NormalizedText']
            # Truncate text if too long (keep first 60 chars)
            if len(text) > 60:
                text = text[:57] + "..."
            emotion = row.get('PrimaryEmotionLabel', 'N/A')
            intensity = row.get('IntensityScore_Primary', 0.0)
            
            # Format timestamp nicely
            if isinstance(timestamp, pd.Timestamp):
                ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                ts_str = str(timestamp)
            
            # Format intensity to 2 decimal places
            if isinstance(intensity, (int, float)):
                intensity_str = f"{intensity:.2f}"
            else:
                intensity_str = str(intensity)
            
            # Get sentiment indicator
            from src.utils import get_emotion_sentiment
            sentiment = get_emotion_sentiment(emotion)
            sentiment_str = f" {sentiment}" if sentiment else ""
            
            lines.append(f"[{ts_str}] {text} → {emotion}{sentiment_str} ({intensity_str})")
        
        return "\n".join(lines)
    
    def create_journal(self, user_id: str, journal_name: str) -> int:
        """
        Create a new journal for a user.
        
        Args:
            user_id: User identifier
            journal_name: Name of the journal (must be unique per user, case-insensitive)
            
        Returns:
            JournalID of the created journal
            
        Raises:
            ValueError: If journal name already exists for user
        """
        conn = self._get_connection()
        
        try:
            cursor = conn.cursor()
            
            # Check if journal with same name already exists (case-insensitive, not archived)
            cursor.execute("""
                SELECT JournalID FROM journals
                WHERE UserID = ? AND LOWER(JournalName) = LOWER(?) AND IsArchived = 0
            """, (str(user_id), journal_name.strip()))
            
            existing = cursor.fetchone()
            if existing:
                raise ValueError(f"Journal '{journal_name}' already exists for user {user_id}")
            
            # Create new journal
            cursor.execute("""
                INSERT INTO journals (UserID, JournalName, IsArchived)
                VALUES (?, ?, 0)
            """, (str(user_id), journal_name.strip()))
            
            journal_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Created journal '{journal_name}' (ID: {journal_id}) for user {user_id}")
            return journal_id
            
        finally:
            conn.close()
    
    def list_journals(self, user_id: str, include_archived: bool = False) -> pd.DataFrame:
        """
        List all journals for a user.
        
        Args:
            user_id: User identifier
            include_archived: If True, include archived journals
            
        Returns:
            DataFrame with columns: JournalID, JournalName, CreatedAt, IsArchived, EntryCount
        """
        conn = self._get_connection()
        
        try:
            query = """
                SELECT 
                    j.JournalID,
                    j.JournalName,
                    j.CreatedAt,
                    j.IsArchived,
                    COUNT(l.LogID) as EntryCount
                FROM journals j
                LEFT JOIN mwb_log l ON j.JournalID = l.JournalID
                WHERE j.UserID = ?
            """
            params = [str(user_id)]
            
            if not include_archived:
                query += " AND j.IsArchived = 0"
            
            query += " GROUP BY j.JournalID, j.JournalName, j.CreatedAt, j.IsArchived"
            query += " ORDER BY j.CreatedAt DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            # Convert CreatedAt to datetime
            if 'CreatedAt' in df.columns and len(df) > 0:
                df['CreatedAt'] = pd.to_datetime(df['CreatedAt'], errors='coerce')
            
            logger.info(f"Found {len(df)} journals for user {user_id}")
            return df
            
        finally:
            conn.close()
    
    def archive_journal(self, user_id: str, journal_name: str) -> bool:
        """
        Soft delete (archive) a journal.
        
        Args:
            user_id: User identifier
            journal_name: Name of the journal to archive
            
        Returns:
            True if journal was archived, False if not found
        """
        conn = self._get_connection()
        
        try:
            cursor = conn.cursor()
            
            # Find journal (case-insensitive)
            cursor.execute("""
                SELECT JournalID FROM journals
                WHERE UserID = ? AND LOWER(JournalName) = LOWER(?) AND IsArchived = 0
            """, (str(user_id), journal_name.strip()))
            
            journal = cursor.fetchone()
            if not journal:
                logger.warning(f"Journal '{journal_name}' not found for user {user_id}")
                return False
            
            # Archive the journal
            cursor.execute("""
                UPDATE journals
                SET IsArchived = 1
                WHERE JournalID = ?
            """, (journal['JournalID'],))
            
            conn.commit()
            
            logger.info(f"Archived journal '{journal_name}' (ID: {journal['JournalID']}) for user {user_id}")
            return True
            
        finally:
            conn.close()
    
    def get_journal_id(self, user_id: str, journal_name: str) -> Optional[int]:
        """
        Get JournalID for a given user and journal name.
        
        Args:
            user_id: User identifier
            journal_name: Name of the journal (case-insensitive)
            
        Returns:
            JournalID if found, None otherwise
        """
        conn = self._get_connection()
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT JournalID FROM journals
                WHERE UserID = ? AND LOWER(JournalName) = LOWER(?) AND IsArchived = 0
            """, (str(user_id), journal_name.strip()))
            
            result = cursor.fetchone()
            return result['JournalID'] if result else None
            
        finally:
            conn.close()
    
    def get_active_journal_id(self, user_id: str, journal_name: str) -> Optional[int]:
        """
        Get JournalID for an active (non-archived) journal.
        Alias for get_journal_id (already filters by IsArchived=0).
        
        Args:
            user_id: User identifier
            journal_name: Name of the journal (case-insensitive)
            
        Returns:
            JournalID if found and active, None otherwise
        """
        return self.get_journal_id(user_id, journal_name)
    
    def clear_all_data(self) -> Dict[str, int]:
        """
        Clear all entries from both mwb_log and raw_archive tables.
        
        Returns:
            Dictionary with counts of deleted entries
        """
        conn = self._get_connection()
        
        try:
            cursor = conn.cursor()
            
            # Count before deletion
            cursor.execute("SELECT COUNT(*) FROM mwb_log")
            log_count_before = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM raw_archive")
            archive_count_before = cursor.fetchone()[0]
            
            # Delete from mwb_log
            cursor.execute("DELETE FROM mwb_log")
            deleted_logs = cursor.rowcount
            
            # Delete from raw_archive
            cursor.execute("DELETE FROM raw_archive")
            deleted_archive = cursor.rowcount
            
            conn.commit()
            
            logger.info(f"Cleared {deleted_logs} mwb_log entries and {deleted_archive} raw_archive entries")
            
            return {
                'deleted_logs': deleted_logs,
                'deleted_archive': deleted_archive,
                'total_deleted': deleted_logs + deleted_archive
            }
        
        finally:
            conn.close()


# Note: Layer 3 (Contextualization) and Layer 4 (Modeling) functions
# are now implemented in src/contextualize.py and src/modeling.py
# Import them as needed:
# from src.contextualize import create_sequence_for_model, create_sequences_batch
# from src.modeling import batch_for_inference, run_inference_pipeline

