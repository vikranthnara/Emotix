"""
Layer 1: Ingestion
Accepts raw CSV/JSON lines and converts to canonical pd.DataFrame format.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Union, List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate ingested DataFrame has required columns: UserID, Text, Timestamp.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Validated DataFrame with standardized column names
        
    Raises:
        ValueError: If required columns are missing or invalid
    """
    required_cols = {'UserID', 'Text', 'Timestamp'}
    
    # Normalize column names (case-insensitive)
    df.columns = df.columns.str.strip()
    col_mapping = {col: col for col in df.columns}
    
    # Try to map common variations
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['userid', 'user_id', 'user']:
            col_mapping[col] = 'UserID'
        elif col_lower in ['text', 'message', 'content', 'utterance']:
            col_mapping[col] = 'Text'
        elif col_lower in ['timestamp', 'time', 'date', 'datetime', 'created_at']:
            col_mapping[col] = 'Timestamp'
    
    df = df.rename(columns=col_mapping)
    
    # Check required columns exist
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Validate data types and non-null
    if df['UserID'].isna().any():
        raise ValueError("UserID cannot contain null values")
    if df['Text'].isna().any():
        raise ValueError("Text cannot contain null values")
    if df['Timestamp'].isna().any():
        raise ValueError("Timestamp cannot contain null values")
    
    # Convert Timestamp to datetime if not already
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    except Exception as e:
        raise ValueError(f"Invalid timestamp format: {e}")
    
    # Ensure UserID is string
    df['UserID'] = df['UserID'].astype(str)
    
    # Ensure Text is string
    df['Text'] = df['Text'].astype(str)
    
    logger.info(f"Validated DataFrame with {len(df)} rows")
    return df[['UserID', 'Text', 'Timestamp']]


def ingest_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Ingest data from CSV file.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        Validated DataFrame with columns: UserID, Text, Timestamp
    """
    logger.info(f"Ingesting CSV from {file_path}")
    df = pd.read_csv(file_path, **kwargs)
    return validate_dataframe(df)


def ingest_jsonl(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Ingest data from JSON Lines file (one JSON object per line).
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        Validated DataFrame with columns: UserID, Text, Timestamp
    """
    logger.info(f"Ingesting JSONL from {file_path}")
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
    
    if not records:
        raise ValueError("No valid records found in JSONL file")
    
    df = pd.DataFrame(records)
    return validate_dataframe(df)


def ingest_json(file_path: Union[str, Path], records_key: Optional[str] = None) -> pd.DataFrame:
    """
    Ingest data from JSON file (array of objects or single object with records_key).
    
    Args:
        file_path: Path to JSON file
        records_key: Optional key to extract records from nested JSON
        
    Returns:
        Validated DataFrame with columns: UserID, Text, Timestamp
    """
    logger.info(f"Ingesting JSON from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if records_key:
        records = data.get(records_key, [])
    elif isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # Try to find array values
        records = [data] if all(k in data for k in ['UserID', 'Text', 'Timestamp']) else []
    else:
        raise ValueError("JSON must be an array or object with records")
    
    if not records:
        raise ValueError("No records found in JSON file")
    
    df = pd.DataFrame(records)
    return validate_dataframe(df)


def ingest_from_dict(records: List[Dict]) -> pd.DataFrame:
    """
    Ingest data from list of dictionaries.
    
    Args:
        records: List of dictionaries with UserID, Text, Timestamp keys
        
    Returns:
        Validated DataFrame with columns: UserID, Text, Timestamp
    """
    logger.info(f"Ingesting {len(records)} records from dict list")
    df = pd.DataFrame(records)
    return validate_dataframe(df)

