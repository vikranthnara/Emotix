"""
Validation and metrics tracking for emotion classification.
Includes synthetic ambiguity dataset generation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AmbiguityGenerator:
    """
    Generate synthetic ambiguous utterances for validation.
    Uses LLM-style prompt engineering to create ambiguous cases.
    """
    
    # Templates for ambiguous utterances
    AMBIGUOUS_TEMPLATES = [
        # Sarcasm/irony
        ("I'm fine", ["neutral", "sadness", "anger"], "Sarcastic 'fine' can indicate distress"),
        ("Whatever", ["neutral", "anger", "sadness"], "Dismissive response can hide emotions"),
        ("That's great", ["joy", "sarcasm", "anger"], "Sarcastic positive can be negative"),
        
        # Context-dependent
        ("I don't know", ["confusion", "sadness", "anxiety"], "Uncertainty can indicate multiple states"),
        ("Maybe", ["neutral", "anxiety", "confusion"], "Hedging can mask true feelings"),
        ("I guess", ["neutral", "uncertainty", "anxiety"], "Tentative agreement can hide concerns"),
        
        # Polysemous expressions
        ("I'm okay", ["neutral", "sadness", "anxiety"], "Okay can mean 'not great'"),
        ("It's nothing", ["neutral", "sadness", "anger"], "Dismissing can hide real issues"),
        ("I'm good", ["joy", "neutral", "sarcasm"], "Good can be genuine or dismissive"),
        
        # Emotional masking
        ("I'm tired", ["neutral", "sadness", "stress"], "Tired can be physical or emotional"),
        ("It's fine", ["neutral", "anger", "sadness"], "Fine often means 'not fine'"),
        ("No worries", ["calm", "neutral", "dismissive"], "Can be genuine or dismissive"),
    ]
    
    def generate_ambiguous_cases(self, 
                                num_cases: int = 50,
                                include_context: bool = True) -> pd.DataFrame:
        """
        Generate synthetic ambiguous cases.
        
        Args:
            num_cases: Number of cases to generate
            include_context: If True, include example contexts
            
        Returns:
            DataFrame with ambiguous cases and ground truth labels
        """
        logger.info(f"Generating {num_cases} ambiguous cases")
        
        cases = []
        
        # Repeat templates to reach num_cases
        templates = self.AMBIGUOUS_TEMPLATES * ((num_cases // len(self.AMBIGUOUS_TEMPLATES)) + 1)
        
        for i, (text, possible_labels, description) in enumerate(templates[:num_cases]):
            case = {
                'CaseID': f'amb_{i+1:04d}',
                'Text': text,
                'NormalizedText': text,  # Assume already normalized
                'PossibleLabels': possible_labels,
                'PrimaryLabel': possible_labels[0],  # First is most likely
                'SecondaryLabel': possible_labels[1] if len(possible_labels) > 1 else None,
                'Description': description,
                'IsAmbiguous': True,
                'RequiresContext': include_context
            }
            
            if include_context:
                case['ContextExamples'] = self._generate_context_examples(text, possible_labels)
            
            cases.append(case)
        
        df = pd.DataFrame(cases)
        logger.info(f"Generated {len(df)} ambiguous cases")
        return df
    
    def _generate_context_examples(self, 
                                   text: str,
                                   possible_labels: List[str]) -> List[Dict]:
        """Generate example contexts that disambiguate the text."""
        examples = []
        
        # Example contexts for each possible label
        for label in possible_labels[:2]:  # Top 2 labels
            if label == 'sadness':
                context = "I've been feeling down lately. " + text
            elif label == 'anger':
                context = "I'm frustrated with everything. " + text
            elif label == 'sarcasm':
                context = "After everything that happened, " + text.lower()
            elif label == 'neutral':
                context = "Everything is going well. " + text
            elif label == 'anxiety':
                context = "I'm worried about what might happen. " + text
            else:
                context = text  # Default
        
            examples.append({
                'context': context,
                'disambiguated_label': label
            })
        
        return examples


class MetricsTracker:
    """Track validation metrics for emotion classification."""
    
    def __init__(self):
        self.metrics_history = []
    
    def compute_metrics(self,
                       y_true: List[str],
                       y_pred: List[str],
                       y_true_intensity: Optional[List[float]] = None,
                       y_pred_intensity: Optional[List[float]] = None,
                       with_context: bool = False) -> Dict:
        """
        Compute classification and regression metrics.
        
        Args:
            y_true: True emotion labels
            y_pred: Predicted emotion labels
            y_true_intensity: True intensity scores (optional)
            y_pred_intensity: Predicted intensity scores (optional)
            with_context: Whether context was used
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            confusion_matrix, classification_report
        )
        
        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'with_context': with_context,
            'num_samples': len(y_true)
        }
        
        # Intensity metrics (if available)
        if y_true_intensity and y_pred_intensity:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(y_true_intensity, y_pred_intensity)
            mae = mean_absolute_error(y_true_intensity, y_pred_intensity)
            r2 = r2_score(y_true_intensity, y_pred_intensity)
            
            metrics.update({
                'intensity_mse': float(mse),
                'intensity_mae': float(mae),
                'intensity_r2': float(r2)
            })
        
        # Confusion matrix
        labels = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['labels'] = labels
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        self.metrics_history.append(metrics)
        return metrics
    
    def compare_with_without_context(self,
                                    metrics_with: Dict,
                                    metrics_without: Dict) -> Dict:
        """
        Compare metrics with and without context.
        
        Args:
            metrics_with: Metrics with context
            metrics_without: Metrics without context
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            'accuracy_improvement': metrics_with['accuracy'] - metrics_without['accuracy'],
            'f1_improvement': metrics_with['f1_weighted'] - metrics_without['f1_weighted'],
            'precision_improvement': metrics_with['precision_weighted'] - metrics_without['precision_weighted'],
            'recall_improvement': metrics_with['recall_weighted'] - metrics_without['recall_weighted'],
        }
        
        if 'intensity_mse' in metrics_with:
            comparison['intensity_mse_improvement'] = (
                metrics_without['intensity_mse'] - metrics_with['intensity_mse']
            )
        
        return comparison
    
    def save_metrics(self, filepath: Path):
        """Save metrics history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Saved metrics to {filepath}")
    
    def load_metrics(self, filepath: Path):
        """Load metrics history from JSON file."""
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)
        logger.info(f"Loaded metrics from {filepath}")
    
    def track_override_effectiveness(self,
                                    df: pd.DataFrame,
                                    true_labels_col: Optional[str] = None) -> Dict:
        """
        Track effectiveness of post-processing overrides.
        
        Args:
            df: DataFrame with predictions, overrides, and optionally true labels
            true_labels_col: Column name for true labels (if available)
            
        Returns:
            Dictionary with override effectiveness metrics
        """
        if 'PostProcessingOverride' not in df.columns or 'WasOverridden' not in df.columns:
            logger.warning("Override columns not found in DataFrame")
            return {}
        
        overrides = df[df['WasOverridden'] == True]
        total_overrides = len(overrides)
        
        if total_overrides == 0:
            return {
                'total_overrides': 0,
                'override_rate': 0.0,
                'override_types': {},
                'precision': None,
                'recall': None
            }
        
        # Override type distribution
        override_types = overrides['PostProcessingOverride'].value_counts().to_dict()
        
        metrics = {
            'total_overrides': total_overrides,
            'override_rate': total_overrides / len(df),
            'override_types': override_types
        }
        
        # If true labels available, compute precision/recall
        if true_labels_col and true_labels_col in df.columns:
            # Precision: of all overrides, how many were correct?
            correct_overrides = 0
            for idx, row in overrides.iterrows():
                predicted = row.get('PrimaryEmotionLabel', '')
                true_label = row.get(true_labels_col, '')
                if predicted == true_label:
                    correct_overrides += 1
            
            precision = correct_overrides / total_overrides if total_overrides > 0 else 0.0
            
            # Recall: of all misclassifications, how many were caught by overrides?
            # This requires knowing which predictions were wrong
            misclassifications = df[df['PrimaryEmotionLabel'] != df[true_labels_col]]
            caught_by_override = len(misclassifications[misclassifications['WasOverridden'] == True])
            recall = caught_by_override / len(misclassifications) if len(misclassifications) > 0 else 0.0
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['correct_overrides'] = correct_overrides
            metrics['caught_misclassifications'] = caught_by_override
        
        return metrics
    
    def track_emotion_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Track emotion distribution over time.
        
        Args:
            df: DataFrame with emotion predictions and timestamps
            
        Returns:
            Dictionary with emotion distribution metrics
        """
        if 'PrimaryEmotionLabel' not in df.columns:
            logger.warning("PrimaryEmotionLabel column not found")
            return {}
        
        emotion_counts = df['PrimaryEmotionLabel'].value_counts().to_dict()
        total = len(df)
        emotion_percentages = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}
        
        # Temporal distribution if timestamp available
        temporal_dist = {}
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df = df.dropna(subset=['Timestamp'])
            if len(df) > 0:
                df = df.sort_values('Timestamp')
                # Group by date
                df['Date'] = df['Timestamp'].dt.date
                daily_dist = df.groupby('Date')['PrimaryEmotionLabel'].value_counts().unstack(fill_value=0)
                temporal_dist = daily_dist.to_dict()
        
        return {
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'total_predictions': total,
            'temporal_distribution': temporal_dist
        }
    
    def track_calibration_quality(self,
                                  df: pd.DataFrame,
                                  true_labels_col: Optional[str] = None,
                                  intensity_col: str = 'IntensityScore_Primary') -> Dict:
        """
        Track calibration quality using Expected Calibration Error (ECE).
        
        Args:
            df: DataFrame with predictions and intensities
            true_labels_col: Column name for true labels (if available)
            intensity_col: Column name for intensity scores
            
        Returns:
            Dictionary with calibration metrics
        """
        if intensity_col not in df.columns:
            logger.warning(f"Intensity column {intensity_col} not found")
            return {}
        
        intensities = df[intensity_col]
        
        metrics = {
            'mean_intensity': float(intensities.mean()),
            'median_intensity': float(intensities.median()),
            'std_intensity': float(intensities.std()),
            'high_confidence_rate': float((intensities >= 0.95).sum() / len(df)),
            'medium_confidence_rate': float(((intensities >= 0.5) & (intensities < 0.95)).sum() / len(df)),
            'low_confidence_rate': float((intensities < 0.5).sum() / len(df))
        }
        
        # Compute ECE if true labels available
        if true_labels_col and true_labels_col in df.columns:
            try:
                ece = self._compute_ece(df, true_labels_col, intensity_col)
                metrics['expected_calibration_error'] = ece
            except Exception as e:
                logger.warning(f"Could not compute ECE: {e}")
        
        # Intensity distribution by emotion
        if 'PrimaryEmotionLabel' in df.columns:
            intensity_by_emotion = df.groupby('PrimaryEmotionLabel')[intensity_col].agg(['mean', 'std', 'count']).to_dict('index')
            metrics['intensity_by_emotion'] = intensity_by_emotion
        
        return metrics
    
    def _compute_ece(self, df: pd.DataFrame, true_col: str, intensity_col: str, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = sum(|accuracy(bin) - confidence(bin)| * proportion(bin))
        """
        df = df.copy()
        df['Correct'] = (df['PrimaryEmotionLabel'] == df[true_col]).astype(int)
        
        # Bin predictions by confidence
        df['Bin'] = pd.cut(df[intensity_col], bins=n_bins, labels=False)
        
        ece = 0.0
        for bin_idx in range(n_bins):
            bin_data = df[df['Bin'] == bin_idx]
            if len(bin_data) == 0:
                continue
            
            accuracy = bin_data['Correct'].mean()
            confidence = bin_data[intensity_col].mean()
            proportion = len(bin_data) / len(df)
            
            ece += abs(accuracy - confidence) * proportion
        
        return float(ece)
    
    def track_false_positive_rate(self,
                                 df: pd.DataFrame,
                                 true_labels_col: str,
                                 emotion: str) -> Dict:
        """
        Track false positive rate for a specific emotion.
        
        Args:
            df: DataFrame with predictions and true labels
            true_labels_col: Column name for true labels
            emotion: Emotion label to check
            
        Returns:
            Dictionary with false positive metrics
        """
        if 'PrimaryEmotionLabel' not in df.columns or true_labels_col not in df.columns:
            logger.warning("Required columns not found")
            return {}
        
        # True positives: predicted emotion AND true emotion
        tp = len(df[(df['PrimaryEmotionLabel'] == emotion) & (df[true_labels_col] == emotion)])
        
        # False positives: predicted emotion BUT not true emotion
        fp = len(df[(df['PrimaryEmotionLabel'] == emotion) & (df[true_labels_col] != emotion)])
        
        # True negatives: not predicted emotion AND not true emotion
        tn = len(df[(df['PrimaryEmotionLabel'] != emotion) & (df[true_labels_col] != emotion)])
        
        # False negatives: not predicted emotion BUT true emotion
        fn = len(df[(df['PrimaryEmotionLabel'] != emotion) & (df[true_labels_col] == emotion)])
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return {
            'emotion': emotion,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr),
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        }


def validate_on_ambiguous_dataset(model,
                                 ambiguous_df: pd.DataFrame,
                                 persistence,
                                 with_context: bool = True) -> Dict:
    """
    Validate model on synthetic ambiguous dataset.
    
    Args:
        model: EmotionModel instance
        ambiguous_df: DataFrame of ambiguous cases
        persistence: MWBPersistence instance (for context if needed)
        with_context: Whether to use context
        
    Returns:
        Dictionary of validation metrics
    """
    logger.info(f"Validating on {len(ambiguous_df)} ambiguous cases")
    
    predictions = []
    true_labels = []
    
    for _, row in ambiguous_df.iterrows():
        text = row['NormalizedText']
        true_label = row['PrimaryLabel']
        
        if with_context:
            # In real scenario, would fetch user history
            # For synthetic data, use sequence as-is
            sequence = text
        else:
            sequence = text
        
        pred = model.predict_emotion(sequence)
        predictions.append(pred['label'])
        true_labels.append(true_label)
    
    # Compute metrics
    tracker = MetricsTracker()
    metrics = tracker.compute_metrics(
        true_labels,
        predictions,
        with_context=with_context
    )
    
    return metrics

