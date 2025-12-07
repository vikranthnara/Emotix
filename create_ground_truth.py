#!/usr/bin/env python3
"""
Create ground truth CSV from synthetic data by mapping text patterns to expected emotions.
"""

import pandas as pd
from pathlib import Path
import re
import sys

def map_text_to_expected_emotion(text: str) -> str:
    """
    Map text to expected emotion based on patterns from generate_synthetic_data.py.
    """
    text_lower = text.lower()
    
    # Gratitude patterns → joy
    if any(pattern in text_lower for pattern in ['thx', 'thank you', 'thank', 'grateful', 'gratitude', 'appreciate']):
        return 'joy'
    
    # Humor patterns → joy
    if any(pattern in text_lower for pattern in ['laughing out loud', 'hilarious', 'haha', 'lol', 'funny']):
        return 'joy'
    
    # Progress patterns → joy
    if any(pattern in text_lower for pattern in ['progress', 'finished', 'improving', 'better', 'easier']):
        return 'joy'
    
    # High sadness → sadness
    if any(pattern in text_lower for pattern in ['sad and hopeless', 'devastated', 'heartbroken']):
        return 'sadness'
    
    # High fear → fear
    if any(pattern in text_lower for pattern in ['terrified', 'extremely frightened', 'frightened']):
        return 'fear'
    
    # High anger → anger
    if any(pattern in text_lower for pattern in ['angry', 'so angry']):
        return 'anger'
    
    # Anxiety → fear (or anxiety if model supports it)
    if any(pattern in text_lower for pattern in ['anxious', 'worried sick', 'worried']):
        return 'fear'  # Model may predict fear for anxiety
    
    # Neutral patterns → neutral
    if any(pattern in text_lower for pattern in ['just checking in', 'nothing much', 'same as usual', 'just another day']):
        return 'neutral'
    
    # Ambiguous → neutral (or could be sadness/anger depending on context)
    if any(pattern in text_lower for pattern in ["i'm fine", "could be better", "not sure how i feel"]):
        return 'neutral'  # Ambiguous cases default to neutral
    
    # High confidence joy → joy
    if 'ecstatic' in text_lower:
        return 'joy'
    
    # Low confidence → neutral (uncertain)
    if any(pattern in text_lower for pattern in ['maybe', 'sort of', 'i guess', 'a bit']):
        return 'neutral'
    
    # Sarcasm → could be sadness or anger (context-dependent)
    if any(pattern in text_lower for pattern in ['oh great, another', 'wonderful, just what']):
        return 'sadness'  # Sarcasm often indicates negative emotion
    
    # Context-dependent → joy (positive context)
    if any(pattern in text_lower for pattern in ['feeling better', 'improving', 'getting through']):
        return 'joy'
    
    # Standard emotions
    if 'happy' in text_lower:
        return 'joy'
    if 'sad' in text_lower:
        return 'sadness'
    if 'scared' in text_lower or 'fear' in text_lower:
        return 'fear'
    if 'surprised' in text_lower:
        return 'surprise'
    if 'disgusted' in text_lower:
        return 'disgust'
    if 'angry' in text_lower:
        return 'anger'
    
    # Default to neutral if no pattern matches
    return 'neutral'

def create_ground_truth_csv(synthetic_csv_path: str, output_path: str):
    """
    Create ground truth CSV from synthetic data.
    
    Args:
        synthetic_csv_path: Path to synthetic_data_large.csv
        output_path: Path to save ground truth CSV
    """
    # Read synthetic data
    df = pd.read_csv(synthetic_csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Map each text to expected emotion
    df['TrueEmotionLabel'] = df['Text'].apply(map_text_to_expected_emotion)
    
    # Create ground truth DataFrame
    ground_truth = df[['UserID', 'Timestamp', 'TrueEmotionLabel']].copy()
    
    # Save to CSV
    ground_truth.to_csv(output_path, index=False)
    
    print(f"✓ Created ground truth CSV: {output_path}")
    print(f"  Records: {len(ground_truth)}")
    print(f"  Emotion distribution:")
    emotion_counts = ground_truth['TrueEmotionLabel'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"    {emotion}: {count} ({count/len(ground_truth)*100:.1f}%)")
    
    return ground_truth

if __name__ == "__main__":
    synthetic_path = "data/synthetic_data_large.csv"
    output_path = "data/ground_truth_large.csv"
    
    if len(sys.argv) > 1:
        synthetic_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    create_ground_truth_csv(synthetic_path, output_path)
    print(f"\nNext step: python analyze_results.py --ground-truth {output_path}")

