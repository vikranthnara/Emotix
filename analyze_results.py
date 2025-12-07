#!/usr/bin/env python3
"""
Comprehensive analysis of pipeline results from the database.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
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
    print_section("📊 Emotion Distribution")
    
    if 'PrimaryEmotionLabel' not in df.columns:
        print("  ❌ No emotion labels found")
        return
    
    emotion_counts = df['PrimaryEmotionLabel'].value_counts()
    total = len(df)
    
    print(f"  Total: {total} predictions")
    print(f"  {'Emotion':<15} {'Count':<8} {'%':<8} {'Avg Intensity':<12}")
    print(f"  {'-'*45}")
    
    for emotion, count in emotion_counts.items():
        pct = (count / total) * 100
        emotion_df = df[df['PrimaryEmotionLabel'] == emotion]
        avg_intensity = emotion_df['IntensityScore_Primary'].mean() if 'IntensityScore_Primary' in df.columns else 0.0
        print(f"  {emotion:<15} {count:<8} {pct:>6.1f}%  {avg_intensity:>10.3f}")
    
    # Check for class imbalance
    max_count = emotion_counts.max()
    min_count = emotion_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 3:
        print(f"  ⚠️  Class imbalance: {imbalance_ratio:.1f}x (significant)")

def analyze_intensity_scores(df: pd.DataFrame):
    """Analyze intensity score distribution."""
    print_section("📈 Intensity Score Analysis")
    
    if 'IntensityScore_Primary' not in df.columns:
        print("  ❌ No intensity scores found")
        return
    
    intensity = df['IntensityScore_Primary']
    
    print(f"  Mean: {intensity.mean():.3f}  |  Median: {intensity.median():.3f}  |  Range: [{intensity.min():.3f}, {intensity.max():.3f}]")
    
    # Intensity distribution
    low = (intensity < 0.5).sum()
    medium = ((intensity >= 0.5) & (intensity < 0.8)).sum()
    high = (intensity >= 0.8).sum()
    very_high = (intensity >= 0.95).sum()
    
    print(f"  Low (<0.5): {low} ({low/len(df)*100:.1f}%)  |  Medium (0.5-0.8): {medium} ({medium/len(df)*100:.1f}%)  |  High (≥0.8): {high} ({high/len(df)*100:.1f}%)  |  Very High (≥0.95): {very_high} ({very_high/len(df)*100:.1f}%)")

def analyze_post_processing(df: pd.DataFrame):
    """Analyze post-processing overrides."""
    print_section("🔍 Post-Processing")
    
    if 'PostProcessingOverride' not in df.columns:
        return
    
    # Only count actual overrides where label changed
    overrides = df[df['PostProcessingOverride'].notna()].copy()
    
    # Filter to only include records where label actually changed
    if 'OriginalEmotionLabel' in overrides.columns and 'PrimaryEmotionLabel' in overrides.columns:
        overrides['OriginalEmotionLabel'] = overrides['OriginalEmotionLabel'].fillna(overrides['PrimaryEmotionLabel'])
        overrides = overrides[overrides['OriginalEmotionLabel'] != overrides['PrimaryEmotionLabel']]
        overrides = overrides[
            (overrides['OriginalEmotionLabel'].notna()) & 
            (overrides['PrimaryEmotionLabel'].notna()) &
            (overrides['OriginalEmotionLabel'] != '') &
            (overrides['PrimaryEmotionLabel'] != '')
        ]
    
    total = len(df)
    override_count = len(overrides)
    
    if override_count > 0:
        print(f"  Overrides: {override_count} / {total} ({override_count/total*100:.1f}%)")
        override_types = overrides['PostProcessingOverride'].value_counts()
        top_types = ', '.join([f"{t}({c})" for t, c in override_types.head(3).items()])
        print(f"  Top types: {top_types}")
    else:
        print("  ✓ No overrides applied")

def analyze_ambiguity(df: pd.DataFrame):
    """Analyze ambiguity detection."""
    print_section("⚠️  Ambiguity Detection")
    
    if 'AmbiguityFlag' not in df.columns:
        return
    
    ambiguous = df['AmbiguityFlag'].sum()
    total = len(df)
    
    if ambiguous > 0:
        print(f"  Ambiguous: {ambiguous} / {total} ({ambiguous/total*100:.1f}%)")
        ambiguous_df = df[df['AmbiguityFlag'] == True]
        amb_emotions = ambiguous_df['PrimaryEmotionLabel'].value_counts()
        top_amb = ', '.join([f'{e}({c})' for e, c in amb_emotions.head(3).items()])
        print(f"  Top emotions: {top_amb}")
    else:
        print("  ✓ No ambiguous predictions")

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
    
    print(f"  Total flagged: {flagged} / {total} ({flagged/total*100:.1f}%)")
    
    if flagged > 0:
        # Break down by flag source
        post_processing_flag = df['PostProcessingReviewFlag'].sum() if 'PostProcessingReviewFlag' in df.columns else 0
        anomaly_flag = df['AnomalyDetectionFlag'].sum() if 'AnomalyDetectionFlag' in df.columns else 0
        suicidal_flag = df['SuicidalIdeationFlag'].sum() if 'SuicidalIdeationFlag' in df.columns else 0
        
        print(f"  Sources: Post-processing ({post_processing_flag}), Anomaly ({anomaly_flag}), High-confidence ({high_conf}), Suicidal ideation ({suicidal_flag})")
        
        flagged_df = df[df['FlagForReview'] == True]
        if 'IntensityScore_Primary' in flagged_df.columns:
            avg_intensity = flagged_df['IntensityScore_Primary'].mean()
            low_intensity = (flagged_df['IntensityScore_Primary'] < 0.7).sum()
            print(f"  Avg intensity: {avg_intensity:.3f}  |  Low intensity (<0.7): {low_intensity} ({low_intensity/flagged*100:.1f}%) ⚠️")
    else:
        print("  ✓ No predictions flagged for review")

def analyze_suicidal_ideation(df: pd.DataFrame):
    """Analyze suicidal ideation detections."""
    print_section("🚨 Suicidal Ideation Detection")
    
    if 'SuicidalIdeationFlag' not in df.columns:
        print("  ℹ️  Suicidal ideation detection data not available")
        return
    
    total = len(df)
    detected = df['SuicidalIdeationFlag'].sum() if 'SuicidalIdeationFlag' in df.columns else 0
    
    if detected > 0:
        print(f"  ⚠️  Detected: {detected} / {total} ({detected/total*100:.1f}%)")
        print(f"  ⚠️  URGENT: These records require immediate human review")
        
        # Get detected records
        detected_df = df[df['SuicidalIdeationFlag'] == True].copy()
        
        # Show pattern types if available
        if 'SuicidalIdeationPattern' in detected_df.columns:
            pattern_counts = detected_df['SuicidalIdeationPattern'].value_counts()
            print(f"\n  Pattern types:")
            for pattern, count in pattern_counts.items():
                if pd.notna(pattern):
                    print(f"    {pattern}: {count}")
        
        # Show confidence scores if available
        if 'SuicidalIdeationConfidence' in detected_df.columns:
            avg_confidence = detected_df['SuicidalIdeationConfidence'].mean()
            min_confidence = detected_df['SuicidalIdeationConfidence'].min()
            max_confidence = detected_df['SuicidalIdeationConfidence'].max()
            print(f"\n  Confidence scores:")
            print(f"    Average: {avg_confidence:.2f}  |  Range: [{min_confidence:.2f}, {max_confidence:.2f}]")
        
        # Show sample detections (with privacy considerations - truncated text)
        print(f"\n  Sample detections (showing up to 3):")
        for idx, row in detected_df.head(3).iterrows():
            text = row.get('NormalizedText', 'N/A')[:50]
            user_id = row.get('UserID', 'N/A')
            timestamp = row.get('Timestamp', 'N/A')
            pattern = row.get('SuicidalIdeationPattern', 'N/A')
            confidence = row.get('SuicidalIdeationConfidence', 0.0)
            print(f"    User: {user_id} | Pattern: {pattern} | Confidence: {confidence:.2f}")
            print(f"    Text: '{text}...'")
            print(f"    Timestamp: {timestamp}")
    else:
        print("  ✓ No suicidal ideation patterns detected")

def analyze_user_patterns(df: pd.DataFrame):
    """Analyze patterns by user."""
    print_section("👤 User Pattern Analysis")
    
    if 'UserID' not in df.columns:
        print("  ❌ No user data found")
        return
    
    users = df['UserID'].unique()
    print(f"  Total users: {len(users)}")
    
    if 'PrimaryEmotionLabel' in df.columns and 'IntensityScore_Primary' in df.columns:
        user_stats = df.groupby('UserID').agg({
            'PrimaryEmotionLabel': 'count',
            'IntensityScore_Primary': 'mean'
        }).rename(columns={'PrimaryEmotionLabel': 'records', 'IntensityScore_Primary': 'avg_intensity'})
        print(f"  Records per user: {user_stats['records'].min()}-{user_stats['records'].max()} (avg: {user_stats['records'].mean():.1f})")
        print(f"  Avg intensity per user: {user_stats['avg_intensity'].min():.3f}-{user_stats['avg_intensity'].max():.3f} (avg: {user_stats['avg_intensity'].mean():.3f})")

def analyze_temporal_patterns(df: pd.DataFrame):
    """Analyze temporal patterns."""
    print_section("📅 Temporal Pattern Analysis")
    
    if 'Timestamp' not in df.columns:
        print("  ❌ No timestamp data found")
        return
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    time_span = (df['Timestamp'].max() - df['Timestamp'].min()).days
    print(f"  Time range: {df['Timestamp'].min().strftime('%Y-%m-%d')} to {df['Timestamp'].max().strftime('%Y-%m-%d')} ({time_span} days)")
    
    df['Date'] = df['Timestamp'].dt.date
    daily_counts = df['Date'].value_counts()
    print(f"  Records per day: {daily_counts.min()}-{daily_counts.max()} (avg: {daily_counts.mean():.1f})")


def calculate_classification_metrics(df: pd.DataFrame, ground_truth_path: Path):
    """Calculate accuracy, false positive, false negative metrics using ground truth labels."""
    print_section("📊 Classification Metrics")
    
    # Load ground truth
    try:
        gt_df = pd.read_csv(ground_truth_path)
        print(f"  Loaded {len(gt_df)} ground truth records")
    except Exception as e:
        print(f"  ❌ Error loading ground truth: {e}")
        return
    
    # Check required columns
    required_cols = ['UserID', 'Timestamp', 'TrueEmotionLabel']
    missing = set(required_cols) - set(gt_df.columns)
    if missing:
        print(f"  ❌ Missing columns: {missing}")
        return
    
    # Convert timestamps for matching
    gt_df['Timestamp'] = pd.to_datetime(gt_df['Timestamp'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Merge predictions with ground truth on UserID and Timestamp
    # Use a small time window for matching (within 1 second) in case of timestamp precision differences
    merged_list = []
    for _, gt_row in gt_df.iterrows():
        # Find matching prediction
        matches = df[
            (df['UserID'] == gt_row['UserID']) &
            (abs((df['Timestamp'] - gt_row['Timestamp']).dt.total_seconds()) < 1)
        ]
        if len(matches) > 0:
            # Take the closest match
            matches = matches.copy()
            matches['time_diff'] = abs((matches['Timestamp'] - gt_row['Timestamp']).dt.total_seconds())
            best_match = matches.loc[matches['time_diff'].idxmin()]
            merged_row = best_match.to_dict()
            merged_row['TrueEmotionLabel'] = gt_row['TrueEmotionLabel']
            if 'TrueIntensity' in gt_df.columns:
                merged_row['TrueIntensity'] = gt_row['TrueIntensity']
            merged_list.append(merged_row)
    
    if len(merged_list) == 0:
        print("  ⚠️  No matching records found (check UserID/Timestamp)")
        return
    
    merged = pd.DataFrame(merged_list)
    
    # Get predictions and true labels
    y_pred = merged['PrimaryEmotionLabel'].fillna('').astype(str).tolist()
    y_true = merged['TrueEmotionLabel'].fillna('').astype(str).tolist()
    
    # Filter out any empty labels
    valid_mask = (pd.Series(y_pred) != '') & (pd.Series(y_true) != '')
    merged_valid = merged[valid_mask].copy()
    y_pred = merged_valid['PrimaryEmotionLabel'].fillna('').astype(str).tolist()
    y_true = merged_valid['TrueEmotionLabel'].fillna('').astype(str).tolist()
    
    if len(y_pred) == 0:
        print("  ⚠️  No valid labels found")
        return
    
    print(f"  Evaluating {len(y_pred)} matched predictions")
    
    # Calculate overall metrics using MetricsTracker
    tracker = MetricsTracker()
    y_true_intensity = None
    y_pred_intensity = None
    
    if 'TrueIntensity' in merged_valid.columns:
        y_true_intensity = merged_valid['TrueIntensity'].fillna(0.0).tolist()
    if 'IntensityScore_Primary' in merged_valid.columns:
        y_pred_intensity = merged_valid['IntensityScore_Primary'].fillna(0.0).tolist()
    
    metrics = tracker.compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_true_intensity=y_true_intensity,
        y_pred_intensity=y_pred_intensity
    )
    
    # Display overall metrics
    print(f"\n  Overall Metrics:")
    print(f"    Accuracy:  {metrics['accuracy']:.3f}  |  Precision: {metrics['precision_weighted']:.3f}  |  Recall: {metrics['recall_weighted']:.3f}  |  F1: {metrics['f1_weighted']:.3f}")
    
    # Display per-class metrics (top 5 by support)
    if 'classification_report' in metrics:
        report = metrics['classification_report']
        emotion_scores = []
        for emotion in sorted(set(y_true + y_pred)):
            if emotion in report and isinstance(report[emotion], dict):
                prec = report[emotion].get('precision', 0)
                rec = report[emotion].get('recall', 0)
                f1 = report[emotion].get('f1-score', 0)
                supp = report[emotion].get('support', 0)
                emotion_scores.append((emotion, prec, rec, f1, supp))
        
        if emotion_scores:
            emotion_scores.sort(key=lambda x: x[4], reverse=True)  # Sort by support
            print(f"\n  Per-Class (top 5):")
            print(f"    {'Emotion':<12} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Supp':<8}")
            print(f"    {'-'*45}")
            for emotion, prec, rec, f1, supp in emotion_scores[:5]:
                print(f"    {emotion:<12} {prec:<8.3f} {rec:<8.3f} {f1:<8.3f} {supp:<8}")
    
    # Calculate false positive and false negative rates per emotion (top 5 by support)
    all_emotions = sorted(set(y_true + y_pred))
    emotion_fp_metrics = []
    for emotion in all_emotions:
        fp_metrics = tracker.track_false_positive_rate(
            merged_valid,
            true_labels_col='TrueEmotionLabel',
            emotion=emotion
        )
        if fp_metrics and fp_metrics['true_positives'] + fp_metrics['false_negatives'] > 0:
            emotion_fp_metrics.append((emotion, fp_metrics))
    
    if emotion_fp_metrics:
        # Sort by total support (TP + FN)
        emotion_fp_metrics.sort(key=lambda x: x[1]['true_positives'] + x[1]['false_negatives'], reverse=True)
        print(f"\n  False Positive/Negative Rates (top 5):")
        print(f"    {'Emotion':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'FPR':<8} {'FNR':<8}")
        print(f"    {'-'*50}")
        for emotion, fp_metrics in emotion_fp_metrics[:5]:
            print(f"    {emotion:<12} {fp_metrics['true_positives']:<6} {fp_metrics['false_positives']:<6} {fp_metrics['false_negatives']:<6} {fp_metrics['false_positive_rate']:<8.3f} {fp_metrics['false_negative_rate']:<8.3f}")
    
    # Display confusion matrix summary (top 5)
    if 'confusion_matrix' in metrics and 'labels' in metrics:
        cm = metrics['confusion_matrix']
        labels = metrics['labels']
        
        # Show which emotions are most confused
        confusion_pairs = []
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                if i != j and cm[i][j] > 0:  # Not diagonal and has errors
                    confusion_pairs.append((true_label, pred_label, cm[i][j]))
        
        if confusion_pairs:
            confusion_pairs.sort(key=lambda x: x[2], reverse=True)
            print(f"\n  Top Confusions (True → Predicted):")
            for true_l, pred_l, count in confusion_pairs[:5]:
                print(f"    {true_l} → {pred_l}: {count}")
    
    # Intensity metrics if available
    if 'intensity_mse' in metrics:
        print(f"\n  Intensity Metrics: MSE={metrics['intensity_mse']:.4f}, MAE={metrics['intensity_mae']:.4f}, R²={metrics['intensity_r2']:.4f}")
    
    print()

def main():
    """Load and analyze results from database."""
    parser = argparse.ArgumentParser(description='Analyze Emotix pipeline results')
    parser.add_argument('--since', type=str, help='Only analyze records since this date (YYYY-MM-DD or days ago, e.g., "7" for last 7 days)')
    parser.add_argument('--db', type=str, help='Path to database file (default: data/mwb_log.db)')
    parser.add_argument('--ground-truth', type=str, help='Path to CSV file with ground truth labels (columns: UserID, Timestamp, TrueEmotionLabel, optional: TrueIntensity)')
    args = parser.parse_args()
    
    print_section("🧠 Emotix Pipeline Results Analysis")
    
    db_path = project_root / "data" / "mwb_log.db"
    if args.db:
        db_path = Path(args.db)
    
    if not db_path.exists():
        print(f"\n❌ Database not found: {db_path}")
        return 1
    
    # Parse date filter
    date_filter = None
    if args.since:
        try:
            # Try parsing as days ago
            days_ago = int(args.since)
            date_filter = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            # Try parsing as date string
            try:
                date_filter = datetime.strptime(args.since, '%Y-%m-%d').strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                print(f"⚠️  Invalid date format: {args.since}. Use YYYY-MM-DD or number of days.")
                return 1
    
    # Load all results from database
    persistence = MWBPersistence(db_path)
    
    # Query all records (only columns that exist in schema)
    conn = sqlite3.connect(db_path)
    
    # Build query with optional date filter
    base_query = """
        SELECT 
            LogID, UserID, Timestamp, NormalizedText,
            PrimaryEmotionLabel, IntensityScore_Primary,
            OriginalEmotionLabel, OriginalIntensityScore,
            AmbiguityFlag, NormalizationFlags,
            PostProcessingOverride, FlagForReview,
            PostProcessingReviewFlag, AnomalyDetectionFlag, HighConfidenceFlag, SuicidalIdeationFlag
        FROM mwb_log
    """
    
    if date_filter:
        base_query += f" WHERE Timestamp >= '{date_filter}'"
    
    base_query += " ORDER BY Timestamp"
    
    # Try to get all columns, including new ones
    try:
        df = pd.read_sql_query(base_query, conn)
    except:
        # Fallback if new columns don't exist yet
        fallback_query = """
            SELECT 
                LogID, UserID, Timestamp, NormalizedText,
                PrimaryEmotionLabel, IntensityScore_Primary,
                OriginalEmotionLabel, OriginalIntensityScore,
                AmbiguityFlag, NormalizationFlags,
                PostProcessingOverride, FlagForReview
            FROM mwb_log
        """
        if date_filter:
            fallback_query += f" WHERE Timestamp >= '{date_filter}'"
        fallback_query += " ORDER BY Timestamp"
        df = pd.read_sql_query(fallback_query, conn)
    conn.close()
    
    # Convert flag columns to bool (they're stored as integers in SQLite)
    if 'PostProcessingReviewFlag' in df.columns:
        df['PostProcessingReviewFlag'] = df['PostProcessingReviewFlag'].astype(bool)
    if 'AnomalyDetectionFlag' in df.columns:
        df['AnomalyDetectionFlag'] = df['AnomalyDetectionFlag'].astype(bool)
    if 'HighConfidenceFlag' in df.columns:
        df['HighConfidenceFlag'] = df['HighConfidenceFlag'].astype(bool)
    if 'SuicidalIdeationFlag' in df.columns:
        df['SuicidalIdeationFlag'] = df['SuicidalIdeationFlag'].astype(bool)
    if 'FlagForReview' in df.columns:
        df['FlagForReview'] = df['FlagForReview'].astype(bool)
    if 'AmbiguityFlag' in df.columns:
        df['AmbiguityFlag'] = df['AmbiguityFlag'].astype(bool)
    
    # Convert types
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Detect old records (missing new columns or None OriginalEmotionLabel)
    old_records = 0
    if len(df) > 0:
        # Count records missing flag sources (old records)
        if 'PostProcessingReviewFlag' in df.columns:
            old_records = (~df['PostProcessingReviewFlag'] & ~df['AnomalyDetectionFlag'] & 
                          ~df['HighConfidenceFlag'] & df['FlagForReview']).sum()
        # Count records with None OriginalEmotionLabel
        if 'OriginalEmotionLabel' in df.columns:
            none_original = df['OriginalEmotionLabel'].isna().sum()
            if none_original > 0:
                old_records = max(old_records, none_original)
    
    print(f"✓ Loaded {len(df)} records from database")
    
    if old_records > 0 and not date_filter:
        print(f"⚠️  Warning: {old_records} old records detected. Use --since 7 to filter recent records.")
    
    # Run all analyses
    analyze_emotion_distribution(df)
    analyze_intensity_scores(df)
    analyze_post_processing(df)
    analyze_ambiguity(df)
    analyze_review_flags(df)
    analyze_suicidal_ideation(df)
    analyze_user_patterns(df)
    analyze_temporal_patterns(df)
    
    # Calculate classification metrics if ground truth provided
    if args.ground_truth:
        ground_truth_path = Path(args.ground_truth)
        if ground_truth_path.exists():
            calculate_classification_metrics(df, ground_truth_path)
        else:
            print(f"\n⚠️  Ground truth file not found: {ground_truth_path}")
            print("   Skipping classification metrics calculation")
    
    
    # Anomaly Detection
    print_section("🚨 User Anomaly Detection")
    analyzer = UserPatternAnalyzer()
    user_analysis = analyzer.analyze_all_users(df)
    
    if len(user_analysis) > 0:
        flagged_users = user_analysis[user_analysis['flag_for_review'] == True]
        print(f"  Flagged users: {len(flagged_users)} / {len(user_analysis)}")
        
        if len(flagged_users) > 0:
            risk_levels = flagged_users['risk_level'].value_counts()
            risk_summary = ', '.join([f"{r}({c})" for r, c in risk_levels.items()])
            print(f"  Risk levels: {risk_summary}")
        
        # Show all user risk levels
        risk_counts = user_analysis['risk_level'].value_counts()
        risk_dist = ', '.join([f"{r}({c})" for r, c in risk_counts.items()])
        print(f"  Risk distribution: {risk_dist}")
    
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())

