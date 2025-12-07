#!/usr/bin/env python3
"""
Interactive CLI for Emotix Mental Well-Being Tracking Pipeline.
Accepts streaming text entries and processes them through the full 5-layer pipeline.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import warnings

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.preprocess import preprocess_pipeline
from src.contextualize import create_sequences_batch
from src.modeling import EmotionModel, run_inference_pipeline
from src.persistence import MWBPersistence
from src.utils import setup_logging
from src.postprocess import EmotionPostProcessor
from src.utils import get_emotion_sentiment
from src.suicidal_detection import SuicidalIdeationDetector


def process_single_entry(
    text: str,
    user_id: str,
    timestamp: datetime,
    persistence: MWBPersistence,
    preprocessor,
    model: EmotionModel,
    postprocessor: EmotionPostProcessor,
    suicidal_detector: SuicidalIdeationDetector,
    slang_dict_path: Path = None
) -> dict:
    """
    Process a single text entry through the full pipeline.
    
    Args:
        text: Raw text entry
        user_id: User identifier
        timestamp: Timestamp for the entry
        persistence: MWBPersistence instance
        preprocessor: Preprocessor instance (or None to create one)
        model: EmotionModel instance
        postprocessor: EmotionPostProcessor instance
        slang_dict_path: Path to slang dictionary
        
    Returns:
        Dictionary with processing results
    """
    # Create DataFrame for single entry
    df = pd.DataFrame({
        'UserID': [user_id],
        'Text': [text],
        'Timestamp': [timestamp]
    })
    
    # Step 1: Preprocessing
    if preprocessor is None:
        from src.preprocess import Preprocessor
        preprocessor = Preprocessor(slang_dict_path=slang_dict_path)
        df = preprocessor.preprocess_dataframe(df)
    else:
        df = preprocessor.preprocess_dataframe(df)
    
    # Step 2: Contextualization
    df = create_sequences_batch(df, persistence, max_context_turns=3)
    
    # Step 3: Modeling
    df = run_inference_pipeline(df, model)
    
    # Step 4: Post-processing
    row = df.iloc[0]
    # Pass original text for keyword checking (to avoid issues with corrupted normalized text)
    original_text = row.get('Text', row['NormalizedText'])
    corrected_label, corrected_intensity, was_overridden, override_type = postprocessor.post_process(
        row['NormalizedText'],
        row['PrimaryEmotionLabel'],
        row['IntensityScore_Primary'],
        original_text=original_text
    )
    
    # Update DataFrame with post-processed results
    df['PrimaryEmotionLabel'] = corrected_label
    df['IntensityScore_Primary'] = corrected_intensity
    df['PostProcessingOverride'] = override_type if was_overridden else None
    df['OriginalEmotionLabel'] = row['PrimaryEmotionLabel']
    df['OriginalIntensityScore'] = row['IntensityScore_Primary']
    
    # Step 4.5: Suicidal Ideation Detection
    is_detected, confidence, pattern_type = suicidal_detector.process_text(
        original_text, user_id=user_id, timestamp=timestamp
    )
    df['SuicidalIdeationFlag'] = is_detected
    df['SuicidalIdeationConfidence'] = confidence
    df['SuicidalIdeationPattern'] = pattern_type
    df['FlagForReview'] = is_detected  # Always flag for review if detected
    
    # Step 5: Persistence
    persistence.write_results(df, archive_raw=True)
    
    return {
        'text': text,
        'normalized_text': df['NormalizedText'].iloc[0],
        'emotion': corrected_label,
        'intensity': corrected_intensity,
        'original_emotion': row['PrimaryEmotionLabel'],
        'was_overridden': was_overridden,
        'override_type': override_type,
        'suicidal_detected': is_detected,
        'suicidal_pattern': pattern_type,
        'suicidal_confidence': confidence
    }


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 70)
    print("🧠 Emotix: Mental Well-Being Tracking Pipeline")
    print("=" * 70)
    print("\nInteractive CLI for processing text entries through the full pipeline.")
    print("Type your entries and press Enter to process them.")
    print("\nCommands:")
    print("  'exit' or 'quit' - Exit the program")
    print("  'summary' - Show summary of last 3 entries")
    print("  'history' - Show full history")
    print("  'help' - Show this help message")
    print("=" * 70 + "\n")


def print_result(result: dict):
    """Print processing result."""
    print(f"\n✓ Processed: {result['text']}")
    print(f"  Normalized: {result['normalized_text']}")
    emotion = result['emotion']
    sentiment = get_emotion_sentiment(emotion)
    sentiment_str = f" {sentiment}" if sentiment else ""
    print(f"  Emotion: {emotion}{sentiment_str} (intensity: {result['intensity']:.2f})")
    if result['was_overridden']:
        original_emotion = result['original_emotion']
        original_sentiment = get_emotion_sentiment(original_emotion)
        original_sentiment_str = f" {original_sentiment}" if original_sentiment else ""
        new_sentiment_str = f" {sentiment}" if sentiment else ""
        print(f"  ⚡ Override: {original_emotion}{original_sentiment_str} → {emotion}{new_sentiment_str} (via {result['override_type']})")
    if result.get('suicidal_detected', False):
        print(f"  🚨 URGENT: Suicidal ideation detected (pattern: {result.get('suicidal_pattern', 'N/A')}, confidence: {result.get('suicidal_confidence', 0.0):.2f})")


def interactive_loop(
    user_id: str,
    db_path: Path,
    slang_dict_path: Path = None,
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
):
    """Main interactive loop."""
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*return_all_scores.*")
    
    # Initialize components
    print(f"\nInitializing pipeline components...")
    persistence = MWBPersistence(db_path)
    
    from src.preprocess import Preprocessor
    preprocessor = Preprocessor(slang_dict_path=slang_dict_path)
    
    print("Loading emotion model (this may take a moment)...")
    model = EmotionModel(model_name=model_name, temperature=1.5)
    
    postprocessor = EmotionPostProcessor()
    suicidal_detector = SuicidalIdeationDetector()
    
    print("✓ Pipeline ready!\n")
    
    # Main loop
    try:
        while True:
            try:
                # Get user input
                user_input = input(f"[{user_id}] > ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit']:
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'summary':
                    summary = persistence.get_last_3_summary(user_id)
                    print(f"\n📊 Last 3 Entries Summary:\n{summary}\n")
                    continue
                
                elif user_input.lower() == 'history':
                    history = persistence.fetch_history(user_id, limit=20)
                    if history.empty:
                        print("\n📝 No history found.\n")
                    else:
                        print(f"\n📝 History ({len(history)} entries):")
                        for _, row in history.tail(10).iterrows():
                            emotion = row.get('PrimaryEmotionLabel', 'N/A')
                            sentiment = get_emotion_sentiment(emotion)
                            sentiment_str = f" {sentiment}" if sentiment else ""
                            intensity = row.get('IntensityScore_Primary', 'N/A')
                            text = row.get('NormalizedText', 'N/A')[:50]
                            timestamp = row.get('Timestamp', 'N/A')
                            print(f"  [{timestamp}] {text} → {emotion}{sentiment_str} ({intensity:.2f if isinstance(intensity, (int, float)) else intensity})")
                        print()
                    continue
                
                elif user_input.lower() == 'help':
                    print_welcome()
                    continue
                
                # Process entry
                timestamp = datetime.now()
                result = process_single_entry(
                    text=user_input,
                    user_id=user_id,
                    timestamp=timestamp,
                    persistence=persistence,
                    preprocessor=preprocessor,
                    model=model,
                    postprocessor=postprocessor,
                    suicidal_detector=suicidal_detector,
                    slang_dict_path=slang_dict_path
                )
                
                print_result(result)
                
                # Show summary after each entry
                summary = persistence.get_last_3_summary(user_id)
                print(f"\n📊 Last 3 Entries:\n{summary}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing entry: {e}")
                import traceback
                traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Interactive CLI for Emotix Mental Well-Being Tracking Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompt for user ID)
  python emotix_cli.py
  
  # Specify user ID
  python emotix_cli.py --user user001
  
  # Custom database and slang dictionary
  python emotix_cli.py --user user001 --db data/custom.db --slang-dict data/custom_slang.json
        """
    )
    
    parser.add_argument(
        '--user',
        type=str,
        help='User ID (if not provided, will prompt for it)'
    )
    
    parser.add_argument(
        '--db',
        type=str,
        default='data/mwb_log.db',
        help='Path to SQLite database (default: data/mwb_log.db)'
    )
    
    parser.add_argument(
        '--slang-dict',
        type=str,
        default='data/slang_dictionary.json',
        help='Path to slang dictionary JSON (default: data/slang_dictionary.json)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='j-hartmann/emotion-english-distilroberta-base',
        help='Hugging Face model name (default: j-hartmann/emotion-english-distilroberta-base)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='WARNING',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: WARNING)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Get user ID
    if args.user:
        user_id = args.user
    else:
        print_welcome()
        user_id = input("Enter your User ID: ").strip()
        if not user_id:
            print("❌ User ID cannot be empty. Exiting.")
            return 1
    
    # Validate paths
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    slang_dict_path = None
    if args.slang_dict:
        slang_dict_path = Path(args.slang_dict)
        if not slang_dict_path.exists():
            print(f"⚠️  Warning: Slang dictionary not found at {slang_dict_path}")
            print("   Continuing without slang dictionary...")
            slang_dict_path = None
    
    # Run interactive loop
    interactive_loop(
        user_id=user_id,
        db_path=db_path,
        slang_dict_path=slang_dict_path,
        model_name=args.model
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

