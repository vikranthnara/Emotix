#!/usr/bin/env python3
"""
Comprehensive analysis of pipeline results from the database.
"""

import sys
from pathlib import Path
import pandas as pd
import sqlite3
from collections import Counter

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.persistence import MWBPersistence
from src.anomaly_detection import UserPatternAnalyzer
from src.validation import MetricsTracker

def print_section(title: str, char: "=" = "="):
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}")

def analyze_emotion_distribution(df: pd.DataFrame):
    """Analyze emotion distribution."""
    print_section("📊 Emotion Distribution Analysis")
    
    if 'PrimaryEmotionLabel' not in df.columns:
        print("  ❌ No emotion labels found")
        return
    
    emotion_counts = df['PrimaryEmotionLabel'].value_counts()
    total = len(df)
    
    print(f"\nTotal predictions: {total}")
    print(f"\nEmotion breakdown:")
    print(f"{'Emotion':<20} {'Count':<10} {'Percentage':<12} {'Intensity (avg)':<15}")
    print("-" * 60)
    
    for emotion, count in emotion_counts.items():
        pct = (count / total) * 100
        emotion_df = df[df['PrimaryEmotionLabel'] == emotion]
        avg_intensity = emotion_df['IntensityScore_Primary'].mean() if 'IntensityScore_Primary' in df.columns else 0.0
        print(f"{emotion:<20} {count:<10} {pct:>6.1f}%      {avg_intensity:>6.3f}")
    
    # Check for class imbalance
    max_count = emotion_counts.max()
    min_count = emotion_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n⚠️  Class imbalance ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 3:
        print("   → Significant class imbalance detected")
    else:
        print("   → Relatively balanced distribution")

def analyze_intensity_scores(df: pd.DataFrame):
    """Analyze intensity score distribution."""
    print_section("📈 Intensity Score Analysis")
    
    if 'IntensityScore_Primary' not in df.columns:
        print("  ❌ No intensity scores found")
        return
    
    intensity = df['IntensityScore_Primary']
    
    print(f"\nBasic Statistics:")
    print(f"  Mean:   {intensity.mean():.4f}")
    print(f"  Median: {intensity.median():.4f}")
    print(f"  Std:    {intensity.std():.4f}")
    print(f"  Min:    {intensity.min():.4f}")
    print(f"  Max:    {intensity.max():.4f}")
    
    # Distribution by quartiles
    q25 = intensity.quantile(0.25)
    q50 = intensity.quantile(0.50)
    q75 = intensity.quantile(0.75)
    
    print(f"\nQuartiles:")
    print(f"  Q1 (25%): {q25:.4f}")
    print(f"  Q2 (50%): {q50:.4f}")
    print(f"  Q3 (75%): {q75:.4f}")
    
    # Intensity distribution
    low = (intensity < 0.5).sum()
    medium = ((intensity >= 0.5) & (intensity < 0.8)).sum()
    high = (intensity >= 0.8).sum()
    very_high = (intensity >= 0.95).sum()
    
    print(f"\nIntensity categories:")
    print(f"  Low (<0.5):    {low:3d} ({low/len(df)*100:5.1f}%)")
    print(f"  Medium (0.5-0.8): {medium:3d} ({medium/len(df)*100:5.1f}%)")
    print(f"  High (≥0.8):   {high:3d} ({high/len(df)*100:5.1f}%)")
    print(f"  Very High (≥0.95): {very_high:3d} ({very_high/len(df)*100:5.1f}%) ⚠️ Flagged for review")

def analyze_post_processing(df: pd.DataFrame):
    """Analyze post-processing overrides."""
    print_section("🔍 Post-Processing Analysis")
    
    if 'PostProcessingOverride' not in df.columns:
        print("  ℹ️  Post-processing override data not persisted in database")
        print("     (Overrides are applied during inference but not stored)")
        return
    
    overrides = df[df['PostProcessingOverride'].notna()]
    total = len(df)
    override_count = len(overrides)
    
    print(f"\nTotal overrides: {override_count} / {total} ({override_count/total*100:.1f}%)")
    
    if override_count > 0:
        override_types = overrides['PostProcessingOverride'].value_counts()
        print(f"\nOverride types (by frequency):")
        for override_type, count in override_types.items():
            pct = (count / override_count) * 100
            print(f"  {override_type}: {count} ({pct:.1f}% of overrides)")
        
        # Show what emotions were overridden
        print(f"\nOriginal emotions (before override):")
        # Use OriginalEmotionLabel if available, fallback to PrimaryEmotionLabel
        original_col = 'OriginalEmotionLabel' if 'OriginalEmotionLabel' in df.columns else 'PrimaryEmotionLabel'
        original_emotions = df.loc[overrides.index, original_col].value_counts()
        for emotion, count in original_emotions.items():
            print(f"  {emotion}: {count}")
        
        # Show sample overrides
        print(f"\nSample overrides:")
        for idx, row in overrides.head(5).iterrows():
            text = row.get('NormalizedText', 'N/A')[:60]
            original = row.get('OriginalEmotionLabel', row.get('PrimaryEmotionLabel', 'N/A'))
            current = row.get('PrimaryEmotionLabel', 'N/A')
            override_type = row.get('PostProcessingOverride', 'N/A')
            intensity = row.get('IntensityScore_Primary', 'N/A')
            print(f"  '{text}...'")
            if original == current:
                print(f"    {original} (no change, override_type: {override_type}, intensity: {intensity:.3f})")
            else:
                print(f"    {original} → {current} (via {override_type}, intensity: {intensity:.3f})")
    else:
        print("  ✓ No post-processing overrides applied")

def analyze_ambiguity(df: pd.DataFrame):
    """Analyze ambiguity detection."""
    print_section("⚠️  Ambiguity Detection Analysis")
    
    if 'AmbiguityFlag' not in df.columns:
        print("  ❌ No ambiguity flags found")
        return
    
    ambiguous = df['AmbiguityFlag'].sum()
    total = len(df)
    
    print(f"\nAmbiguous predictions: {ambiguous} / {total} ({ambiguous/total*100:.1f}%)")
    
    if ambiguous > 0:
        ambiguous_df = df[df['AmbiguityFlag'] == True]
        print(f"\nAmbiguous emotion distribution:")
        amb_emotions = ambiguous_df['PrimaryEmotionLabel'].value_counts()
        for emotion, count in amb_emotions.items():
            print(f"  {emotion}: {count}")
        
        print(f"\nSample ambiguous predictions:")
        for idx, row in ambiguous_df.head(3).iterrows():
            text = row.get('NormalizedText', 'N/A')[:60]
            emotion = row.get('PrimaryEmotionLabel', 'N/A')
            intensity = row.get('IntensityScore_Primary', 'N/A')
            print(f"  '{text}...' → {emotion} ({intensity:.3f})")

def analyze_review_flags(df: pd.DataFrame):
    """Analyze human review flags."""
    print_section("🚩 Human Review Flag Analysis")
    
    if 'FlagForReview' not in df.columns and 'HighConfidenceFlag' not in df.columns:
        print("  ℹ️  Review flag data not persisted in database")
        print("     (Flags are set during inference but not stored)")
        return
    
    total = len(df)
    flagged = df['FlagForReview'].sum() if 'FlagForReview' in df.columns else 0
    high_conf = df['HighConfidenceFlag'].sum() if 'HighConfidenceFlag' in df.columns else 0
    
    print(f"\nTotal flagged for review: {flagged} / {total} ({flagged/total*100:.1f}%)")
    
    # Break down by flag source
    post_processing_flag = df['PostProcessingReviewFlag'].sum() if 'PostProcessingReviewFlag' in df.columns else 0
    anomaly_flag = df['AnomalyDetectionFlag'].sum() if 'AnomalyDetectionFlag' in df.columns else 0
    
    print(f"\nFlag breakdown by source:")
    print(f"  Post-processing flags: {post_processing_flag} ({post_processing_flag/total*100:.1f}%)")
    print(f"  Anomaly detection flags: {anomaly_flag} ({anomaly_flag/total*100:.1f}%)")
    print(f"  High confidence flags (≥0.95): {high_conf} ({high_conf/total*100:.1f}%)")
    
    # Show intensity distribution of flagged predictions
    if flagged > 0:
        flagged_df = df[df['FlagForReview'] == True]
        
        if 'IntensityScore_Primary' in flagged_df.columns:
            intensities = flagged_df['IntensityScore_Primary']
            print(f"\nIntensity distribution of flagged predictions:")
            print(f"  Mean: {intensities.mean():.3f}")
            print(f"  Median: {intensities.median():.3f}")
            print(f"  Min: {intensities.min():.3f}")
            print(f"  Max: {intensities.max():.3f}")
            
            # Count low-intensity flags (potential false positives)
            low_intensity = (intensities < 0.7).sum()
            medium_intensity = ((intensities >= 0.7) & (intensities < 0.85)).sum()
            high_intensity = (intensities >= 0.85).sum()
            
            print(f"\n  Low intensity (<0.7): {low_intensity} ({low_intensity/flagged*100:.1f}%) ⚠️")
            print(f"  Medium intensity (0.7-0.85): {medium_intensity} ({medium_intensity/flagged*100:.1f}%)")
            print(f"  High intensity (≥0.85): {high_intensity} ({high_intensity/flagged*100:.1f}%)")
        
        print(f"\nFlagged emotion distribution:")
        flag_emotions = flagged_df['PrimaryEmotionLabel'].value_counts()
        for emotion, count in flag_emotions.items():
            print(f"  {emotion}: {count}")
        
        print(f"\nSample flagged predictions:")
        for idx, row in flagged_df.head(5).iterrows():
            text = row.get('NormalizedText', 'N/A')[:60]
            emotion = row.get('PrimaryEmotionLabel', 'N/A')
            intensity = row.get('IntensityScore_Primary', 'N/A')
            
            # Show which flags are set
            flags = []
            if row.get('PostProcessingReviewFlag', False):
                flags.append('post-processing')
            if row.get('AnomalyDetectionFlag', False):
                flags.append('anomaly')
            if row.get('HighConfidenceFlag', False):
                flags.append('high-confidence')
            flag_source = ', '.join(flags) if flags else 'unknown'
            print(f"  '{text}...' → {emotion} ({intensity:.3f}) [flags: {flag_source}]")
    else:
        print("  ✓ No predictions flagged for review")

def analyze_user_patterns(df: pd.DataFrame):
    """Analyze patterns by user."""
    print_section("👤 User Pattern Analysis")
    
    if 'UserID' not in df.columns:
        print("  ❌ No user data found")
        return
    
    users = df['UserID'].unique()
    print(f"\nTotal users: {len(users)}")
    
    for user_id in sorted(users):
        user_df = df[df['UserID'] == user_id]
        print(f"\n  User: {user_id} ({len(user_df)} records)")
        
        if 'PrimaryEmotionLabel' in user_df.columns:
            emotions = user_df['PrimaryEmotionLabel'].value_counts()
            print(f"    Top emotions:")
            for emotion, count in emotions.head(3).items():
                print(f"      {emotion}: {count}")
        
        if 'IntensityScore_Primary' in user_df.columns:
            avg_intensity = user_df['IntensityScore_Primary'].mean()
            print(f"    Avg intensity: {avg_intensity:.3f}")

def analyze_temporal_patterns(df: pd.DataFrame):
    """Analyze temporal patterns."""
    print_section("📅 Temporal Pattern Analysis")
    
    if 'Timestamp' not in df.columns:
        print("  ❌ No timestamp data found")
        return
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    print(f"\nTime range:")
    print(f"  Start: {df['Timestamp'].min()}")
    print(f"  End:   {df['Timestamp'].max()}")
    print(f"  Span:  {(df['Timestamp'].max() - df['Timestamp'].min()).days} days")
    
    # Daily distribution
    df['Date'] = df['Timestamp'].dt.date
    daily_counts = df['Date'].value_counts().sort_index()
    
    print(f"\nDaily record counts:")
    for date, count in daily_counts.items():
        print(f"  {date}: {count} records")

def analyze_text_quality(df: pd.DataFrame):
    """Analyze text quality and normalization."""
    print_section("📝 Text Quality Analysis")
    
    if 'NormalizedText' not in df.columns:
        print("  ❌ No normalized text found")
        return
    
    if 'NormalizationFlags' not in df.columns:
        print("  ❌ No normalization flags found")
        return
    
    # Text length statistics
    df['TextLength'] = df['NormalizedText'].str.len()
    
    print(f"\nText length statistics:")
    print(f"  Mean:   {df['TextLength'].mean():.1f} chars")
    print(f"  Median: {df['TextLength'].median():.1f} chars")
    print(f"  Min:    {df['TextLength'].min()} chars")
    print(f"  Max:    {df['TextLength'].max()} chars")
    
    # Normalization flags
    flags = df['NormalizationFlags'].value_counts()
    print(f"\nNormalization flags:")
    for flag, count in flags.items():
        if pd.notna(flag):
            print(f"  {flag}: {count}")

def main():
    """Load and analyze results from database."""
    print_section("🧠 Emotix Pipeline Results Analysis")
    
    db_path = project_root / "data" / "mwb_log.db"
    
    if not db_path.exists():
        print(f"\n❌ Database not found: {db_path}")
        return 1
    
    # Load all results from database
    persistence = MWBPersistence(db_path)
    
    # Query all records (only columns that exist in schema)
    conn = sqlite3.connect(db_path)
    # Try to get all columns, including new ones
    try:
        df = pd.read_sql_query("""
            SELECT 
                LogID, UserID, Timestamp, NormalizedText,
                PrimaryEmotionLabel, IntensityScore_Primary,
                OriginalEmotionLabel, OriginalIntensityScore,
                AmbiguityFlag, NormalizationFlags,
                PostProcessingOverride, FlagForReview,
                PostProcessingReviewFlag, AnomalyDetectionFlag, HighConfidenceFlag
            FROM mwb_log
            ORDER BY Timestamp
        """, conn)
    except:
        # Fallback if new columns don't exist yet
        df = pd.read_sql_query("""
            SELECT 
                LogID, UserID, Timestamp, NormalizedText,
                PrimaryEmotionLabel, IntensityScore_Primary,
                OriginalEmotionLabel, OriginalIntensityScore,
                AmbiguityFlag, NormalizationFlags,
                PostProcessingOverride, FlagForReview
            FROM mwb_log
            ORDER BY Timestamp
        """, conn)
    conn.close()
    
    # Convert types
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    if 'AmbiguityFlag' in df.columns:
        df['AmbiguityFlag'] = df['AmbiguityFlag'].astype(bool)
    if 'FlagForReview' in df.columns:
        df['FlagForReview'] = df['FlagForReview'].astype(bool)
    
    print(f"\n✓ Loaded {len(df)} records from database")
    print(f"✓ Columns: {', '.join(df.columns)}")
    
    # Run all analyses
    analyze_emotion_distribution(df)
    analyze_intensity_scores(df)
    analyze_post_processing(df)
    analyze_ambiguity(df)
    analyze_review_flags(df)
    analyze_user_patterns(df)
    analyze_temporal_patterns(df)
    analyze_text_quality(df)
    
    # Evaluation Metrics
    print_section("📊 Evaluation Metrics")
    tracker = MetricsTracker()
    
    # Override effectiveness
    override_metrics = tracker.track_override_effectiveness(df)
    if override_metrics:
        print(f"\nOverride Effectiveness:")
        print(f"  Total overrides: {override_metrics.get('total_overrides', 0)}")
        print(f"  Override rate: {override_metrics.get('override_rate', 0)*100:.1f}%")
        if override_metrics.get('precision') is not None:
            print(f"  Precision: {override_metrics['precision']:.3f}")
            print(f"  Recall: {override_metrics['recall']:.3f}")
    
    # Calibration quality
    calibration_metrics = tracker.track_calibration_quality(df)
    if calibration_metrics:
        print(f"\nCalibration Quality:")
        print(f"  Mean intensity: {calibration_metrics.get('mean_intensity', 0):.3f}")
        print(f"  High confidence (≥0.95): {calibration_metrics.get('high_confidence_rate', 0)*100:.1f}%")
        if 'expected_calibration_error' in calibration_metrics:
            print(f"  Expected Calibration Error (ECE): {calibration_metrics['expected_calibration_error']:.3f}")
            print(f"    (Lower is better, <0.1 is well-calibrated)")
    
    # Emotion distribution tracking
    dist_metrics = tracker.track_emotion_distribution(df)
    if dist_metrics:
        print(f"\nEmotion Distribution Tracking:")
        print(f"  Total predictions: {dist_metrics.get('total_predictions', 0)}")
        print(f"  Unique emotions: {len(dist_metrics.get('emotion_counts', {}))}")
    
    # Anomaly Detection
    print_section("🚨 User Anomaly Detection")
    analyzer = UserPatternAnalyzer()
    user_analysis = analyzer.analyze_all_users(df)
    
    if len(user_analysis) > 0:
        flagged_users = user_analysis[user_analysis['flag_for_review'] == True]
        print(f"\nUsers flagged for review: {len(flagged_users)}")
        
        if len(flagged_users) > 0:
            print("\nFlagged users:")
            for _, user_row in flagged_users.iterrows():
                print(f"\n  User: {user_row['user_id']}")
                print(f"    Risk Level: {user_row['risk_level'].upper()}")
                print(f"    Records: {user_row['total_records']}")
                print(f"    Avg Intensity: {user_row['avg_intensity']:.3f}")
                print(f"    Anomalies:")
                for anomaly in user_row['anomalies']:
                    print(f"      - {anomaly}")
        
        # Show all user risk levels
        print(f"\nUser risk distribution:")
        risk_counts = user_analysis['risk_level'].value_counts()
        for risk, count in risk_counts.items():
            print(f"  {risk}: {count}")
    
    # Summary
    print_section("✅ Analysis Complete")
    print("\nKey insights:")
    
    if 'PrimaryEmotionLabel' in df.columns:
        emotion_counts = df['PrimaryEmotionLabel'].value_counts()
        top_emotion = emotion_counts.index[0]
        print(f"  • Most common emotion: {top_emotion} ({emotion_counts[top_emotion]} occurrences)")
    
    if 'IntensityScore_Primary' in df.columns:
        avg_intensity = df['IntensityScore_Primary'].mean()
        print(f"  • Average intensity: {avg_intensity:.3f}")
    
    if 'AmbiguityFlag' in df.columns:
        ambiguous_count = df['AmbiguityFlag'].sum()
        print(f"  • Ambiguous predictions: {ambiguous_count} ({ambiguous_count/len(df)*100:.1f}%)")
    
    if 'PostProcessingOverride' in df.columns:
        override_count = df['PostProcessingOverride'].notna().sum()
        if override_count > 0:
            print(f"  • Post-processing overrides: {override_count} ({override_count/len(df)*100:.1f}%)")
    
    if 'FlagForReview' in df.columns:
        flagged_count = df['FlagForReview'].sum()
        if flagged_count > 0:
            print(f"  • Flagged for review: {flagged_count} ({flagged_count/len(df)*100:.1f}%)")
    
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())

