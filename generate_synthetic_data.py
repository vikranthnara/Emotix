#!/usr/bin/env python3
"""
Generate synthetic data for testing the MWB pipeline.
This script creates diverse examples that test all features:
- Various emotions (joy, sadness, fear, anger, surprise, neutral)
- Positive examples that trigger overrides
- Negative examples that should be flagged
- Low-intensity predictions
- High-confidence predictions
"""

import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path

def generate_synthetic_data(num_records: int = 50, num_users: int = 5) -> pd.DataFrame:
    """
    Generate synthetic journal entries with diverse emotions and scenarios.
    
    Args:
        num_records: Number of records to generate
        num_users: Number of unique users
        
    Returns:
        DataFrame with UserID, Text, Timestamp columns
    """
    users = [f"user_{i:03d}" for i in range(1, num_users + 1)]
    
    # Diverse test cases
    test_cases = [
        # Positive examples (should trigger joy overrides)
        ("thx for the help!", "gratitude"),
        ("laughing out loud at that joke", "humor"),
        ("made progress on my project today", "progress"),
        ("feeling grateful for everything", "gratitude"),
        ("haha that's hilarious", "humor"),
        ("finally finished my task!", "progress"),
        ("thank you so much", "gratitude"),
        
        # Negative examples (should be flagged)
        ("I'm feeling really sad and hopeless", "sadness_high"),
        ("terrified about what might happen", "fear_high"),
        ("so angry I can't think straight", "anger_high"),
        ("completely devastated by the news", "sadness_high"),
        ("anxious about tomorrow's meeting", "anxiety"),
        ("worried sick about the results", "anxiety"),
        
        # Neutral examples
        ("just checking in", "neutral"),
        ("nothing much happening", "neutral"),
        ("same as usual", "neutral"),
        ("just another day", "neutral"),
        
        # Mixed/ambiguous examples
        ("I'm fine", "ambiguous"),
        ("could be better", "ambiguous"),
        ("not sure how I feel", "ambiguous"),
        
        # High confidence examples
        ("I am absolutely ecstatic!", "joy_high"),
        ("completely heartbroken", "sadness_high"),
        ("extremely frightened", "fear_high"),
        
        # Low confidence examples (shouldn't be flagged)
        ("maybe feeling okay", "low_confidence"),
        ("sort of happy I guess", "low_confidence"),
        ("a bit down maybe", "low_confidence"),
        
        # Sarcasm examples
        ("oh great, another problem", "sarcasm"),
        ("wonderful, just what I needed", "sarcasm"),
        
        # Context-dependent examples
        ("feeling better now", "context_dependent"),
        ("things are improving", "context_dependent"),
        ("getting through it", "context_dependent"),
        
        # Standard emotions
        ("feeling happy today", "joy"),
        ("feeling sad", "sadness"),
        ("scared about the future", "fear"),
        ("surprised by the outcome", "surprise"),
        ("disgusted by what happened", "disgust"),
        ("feeling angry", "anger"),
    ]
    
    records = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(num_records):
        # Select a test case
        text, case_type = random.choice(test_cases)
        
        # Assign to a user (some users get more records)
        user_id = random.choice(users)
        
        # Create timestamp (spread over last 7 days)
        hours_ago = random.randint(0, 7 * 24)
        timestamp = base_time + timedelta(hours=hours_ago)
        
        records.append({
            'UserID': user_id,
            'Text': text,
            'Timestamp': timestamp
        })
    
    df = pd.DataFrame(records)
    
    # Sort by timestamp
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    return df

def clear_database(db_path: Path):
    """Clear all existing data from the database."""
    import sqlite3
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear both tables
    cursor.execute("DELETE FROM mwb_log")
    cursor.execute("DELETE FROM raw_archive")
    
    conn.commit()
    conn.close()
    
    print(f"✓ Cleared all data from {db_path}")

def main():
    """Generate synthetic data and save to CSV."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic data for MWB pipeline')
    parser.add_argument('--records', type=int, default=50, help='Number of records to generate')
    parser.add_argument('--users', type=int, default=5, help='Number of unique users')
    parser.add_argument('--clear-db', action='store_true', help='Clear existing database before generating')
    parser.add_argument('--output', type=str, default='data/synthetic_data.csv', help='Output CSV file path')
    args = parser.parse_args()
    
    # Generate synthetic data
    print(f"Generating {args.records} synthetic records for {args.users} users...")
    df = generate_synthetic_data(num_records=args.records, num_users=args.users)
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved synthetic data to {output_path}")
    print(f"  Records: {len(df)}")
    print(f"  Users: {df['UserID'].nunique()}")
    print(f"  Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    # Clear database if requested
    if args.clear_db:
        db_path = Path("data/mwb_log.db")
        clear_database(db_path)
    
    print("\nNext steps:")
    print(f"1. Run pipeline: python test_pipeline.py --input {output_path}")
    print("2. Analyze results: python analyze_results.py")

if __name__ == "__main__":
    main()

