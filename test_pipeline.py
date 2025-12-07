#!/usr/bin/env python3
"""
End-to-end test script for the complete 5-layer MWB pipeline.
Tests: Ingestion → Preprocessing → Contextualization → Modeling → Persistence
"""

import sys
from pathlib import Path
import time
import pandas as pd

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import run_full_pipeline
from src.persistence import MWBPersistence
from src.utils import setup_logging, get_emotion_sentiment

def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 70}")
    print(f"{title}")
    print(f"{char * 70}")

def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * 70)

def analyze_results(df: pd.DataFrame):
    """Analyze and display pipeline results."""
    print_subsection("📊 Emotion Distribution")
    if 'PrimaryEmotionLabel' in df.columns:
        emotion_counts = df['PrimaryEmotionLabel'].value_counts()
        print("\nEmotion counts:")
        for emotion, count in emotion_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {emotion:15s}: {count:3d} ({pct:5.1f}%)")
    else:
        print("  No emotion predictions found")
    
    print_subsection("📈 Intensity Statistics")
    if 'IntensityScore_Primary' in df.columns:
        intensity = df['IntensityScore_Primary']
        print(f"\n  Mean intensity: {intensity.mean():.3f}")
        print(f"  Median intensity: {intensity.median():.3f}")
        print(f"  Min intensity: {intensity.min():.3f}")
        print(f"  Max intensity: {intensity.max():.3f}")
    else:
        print("  No intensity scores found")
    
    print_subsection("🔍 Post-Processing Analysis")
    if 'PostProcessingOverride' in df.columns:
        overrides = df['PostProcessingOverride'].value_counts()
        print("\nPost-processing overrides:")
        for override_type, count in overrides.items():
            if pd.notna(override_type):
                print(f"  {override_type}: {count}")
        
        # Show samples of overrides
        override_samples = df[df['PostProcessingOverride'].notna()]
        if len(override_samples) > 0:
            print(f"\n  Sample overrides ({len(override_samples)} total):")
            for idx, row in override_samples.head(5).iterrows():
                # Use OriginalEmotionLabel (model prediction) not PrimaryEmotionLabel (final label)
                original = row.get('OriginalEmotionLabel', row.get('PrimaryEmotionLabel', 'N/A'))
                current = row.get('PrimaryEmotionLabel', 'N/A')
                override_type = row.get('PostProcessingOverride', 'N/A')
                text = row.get('NormalizedText', 'N/A')[:50]
                print(f"    '{text}...'")
                original_sentiment = get_emotion_sentiment(original)
                current_sentiment = get_emotion_sentiment(current)
                original_sentiment_str = f" {original_sentiment}" if original_sentiment else ""
                current_sentiment_str = f" {current_sentiment}" if current_sentiment else ""
                if original == current:
                    print(f"      Original: {original}{original_sentiment_str} → Current: {current}{current_sentiment_str} (via {override_type}, no change)")
                else:
                    print(f"      Original: {original}{original_sentiment_str} → Current: {current}{current_sentiment_str} (via {override_type})")
    else:
        print("  No post-processing overrides found")
    
    print_subsection("⚠️  Ambiguity Detection")
    if 'AmbiguityFlag' in df.columns:
        ambiguous = df['AmbiguityFlag'].sum()
        print(f"\n  Ambiguous predictions: {ambiguous} / {len(df)} ({ambiguous/len(df)*100:.1f}%)")
    else:
        print("  No ambiguity flags found")
    
    print_subsection("🚩 Human Review Flags")
    if 'FlagForReview' in df.columns:
        flagged = df['FlagForReview'].sum()
        if flagged > 0:
            print(f"\n  ⚠️  {flagged} predictions flagged for human review")
            flagged_samples = df[df['FlagForReview'] == True]
            for idx, row in flagged_samples.head(3).iterrows():
                text = row.get('NormalizedText', 'N/A')[:60]
                emotion = row.get('PrimaryEmotionLabel', 'N/A')
                sentiment = get_emotion_sentiment(emotion)
                sentiment_str = f" {sentiment}" if sentiment else ""
                intensity = row.get('IntensityScore_Primary', 'N/A')
                print(f"    - '{text}...' → {emotion}{sentiment_str} ({intensity:.2f})")
        else:
            print("  ✓ No predictions flagged for review")
    else:
        print("  No review flags found")
    
    print_subsection("🚨 Suicidal Ideation Detection")
    if 'SuicidalIdeationFlag' in df.columns:
        detected = df['SuicidalIdeationFlag'].sum()
        if detected > 0:
            print(f"\n  ⚠️  URGENT: {detected} record(s) with suicidal ideation patterns detected")
            print(f"  ⚠️  These require immediate human review and crisis support")
            detected_samples = df[df['SuicidalIdeationFlag'] == True]
            for idx, row in detected_samples.head(3).iterrows():
                text = row.get('NormalizedText', 'N/A')[:60]
                user_id = row.get('UserID', 'N/A')
                pattern = row.get('SuicidalIdeationPattern', 'N/A')
                confidence = row.get('SuicidalIdeationConfidence', 0.0)
                print(f"    - User: {user_id} | Pattern: {pattern} | Confidence: {confidence:.2f}")
                print(f"      Text: '{text}...'")
        else:
            print("  ✓ No suicidal ideation patterns detected")
    else:
        print("  No suicidal ideation detection data available")

def show_sample_predictions(df: pd.DataFrame, n: int = 5):
    """Show sample predictions with details."""
    print_subsection(f"📝 Sample Predictions (showing {min(n, len(df))} of {len(df)})")
    
    for idx, row in df.head(n).iterrows():
        print(f"\n  [{idx+1}] User: {row.get('UserID', 'N/A')}")
        print(f"      Text: {row.get('NormalizedText', 'N/A')}")
        if 'PrimaryEmotionLabel' in row:
            emotion = row.get('PrimaryEmotionLabel', 'N/A')
            sentiment = get_emotion_sentiment(emotion)
            sentiment_str = f" {sentiment}" if sentiment else ""
            intensity = row.get('IntensityScore_Primary', 'N/A')
            print(f"      Emotion: {emotion}{sentiment_str} (intensity: {intensity:.3f})")
        if 'PostProcessingOverride' in row and pd.notna(row['PostProcessingOverride']):
            print(f"      ⚡ Override: {row['PostProcessingOverride']}")
        if 'AmbiguityFlag' in row and row.get('AmbiguityFlag', False):
            print(f"      ⚠️  Ambiguous")
        if 'FlagForReview' in row and row.get('FlagForReview', False):
            print(f"      🚩 Flagged for review")
        if 'SuicidalIdeationFlag' in row and row.get('SuicidalIdeationFlag', False):
            print(f"      🚨 URGENT: Suicidal ideation detected")

def test_history_retrieval(persistence: MWBPersistence, user_ids: list):
    """Test history retrieval performance."""
    print_section("🔍 History Retrieval Test")
    
    for user_id in user_ids:
        start_time = time.time()
        history = persistence.fetch_history(user_id)
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"\n  User: {user_id}")
        print(f"  Records: {len(history)}")
        print(f"  Latency: {elapsed_ms:.2f}ms {'✓' if elapsed_ms < 100 else '✗'} (requirement: <100ms)")
        
        if len(history) > 0:
            print(f"  Recent emotions:")
            recent = history.tail(3)
            for _, row in recent.iterrows():
                emotion = row.get('PrimaryEmotionLabel', 'N/A')
                sentiment = get_emotion_sentiment(emotion)
                sentiment_str = f" {sentiment}" if sentiment else ""
                intensity = row.get('IntensityScore_Primary', 'N/A')
                timestamp = row.get('Timestamp', 'N/A')
                print(f"    {timestamp} → {emotion}{sentiment_str} ({intensity:.3f})")

def main():
    """Run end-to-end pipeline test."""
    setup_logging(log_level="INFO")
    
    print_section("🧠 Emotix Full Pipeline Test (5 Layers)")
    print("\nThis test runs the complete pipeline:")
    print("  1. Ingestion (CSV/JSON/JSONL)")
    print("  2. Preprocessing (slang, emoji, normalization)")
    print("  3. Contextualization (multi-turn sequences)")
    print("  4. Modeling (emotion classification)")
    print("  5. Persistence (SQLite storage)")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run MWB pipeline end-to-end test')
    parser.add_argument('--input', type=str, help='Input CSV file path (default: data/sample_data.csv)')
    args = parser.parse_args()
    
    # Configuration
    if args.input:
        data_path = Path(args.input)
    else:
        data_path = project_root / "data" / "sample_data.csv"
    
    db_path = project_root / "data" / "mwb_log.db"
    slang_dict_path = project_root / "data" / "slang_dictionary.json"
    checkpoint_dir = project_root / "checkpoints"
    
    # Check if data file exists
    if not data_path.exists():
        print(f"\n❌ Error: Data file not found: {data_path}")
        print("   Please ensure the input file exists")
        return 1
    
    print(f"\n📁 Input: {data_path}")
    print(f"💾 Database: {db_path}")
    print(f"📚 Slang dict: {slang_dict_path}")
    
    # Run pipeline
    print_section("🚀 Running Pipeline")
    start_time = time.time()
    
    try:
        df_results = run_full_pipeline(
            input_path=data_path,
            db_path=db_path,
            slang_dict_path=slang_dict_path if slang_dict_path.exists() else None,
            model_name="j-hartmann/emotion-english-distilroberta-base",
            max_context_turns=3,
            batch_size=32,
            checkpoint_dir=checkpoint_dir,
            archive_raw=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Pipeline completed in {elapsed_time:.2f} seconds")
        print(f"✓ Processed {len(df_results)} records")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Analyze results
    print_section("📊 Results Analysis")
    analyze_results(df_results)
    
    # Show sample predictions
    show_sample_predictions(df_results, n=8)
    
    # Test history retrieval
    persistence = MWBPersistence(db_path)
    unique_users = df_results['UserID'].unique().tolist()
    test_history_retrieval(persistence, unique_users[:3])  # Test first 3 users
    
    # Summary
    print_section("✅ Test Complete")
    print("\nNext steps:")
    print("  - View full results: df_results (in Python)")
    print("  - Check database: sqlite3 data/mwb_log.db")
    print("  - View checkpoints: ls checkpoints/")
    print("  - Run unit tests: pytest tests/ -v")
    print("  - Open demo notebook: jupyter notebook notebooks/phase2_demo.ipynb")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

