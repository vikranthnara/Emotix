"""
Layer 4: Modeling
Fine-tuned transformer model for emotion classification and intensity prediction.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers library not available. Install with: pip install transformers torch")

logger = logging.getLogger(__name__)


class EmotionModel:
    """
    Emotion classification and intensity prediction model.
    
    Uses a pre-trained transformer model (default: j-hartmann/emotion-english-distilroberta-base)
    for emotion classification, with intensity prediction via softmax probabilities.
    """
    
    # Emotion labels mapping (based on common emotion taxonomies)
    EMOTION_LABELS = [
        'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
        'neutral', 'anxiety', 'stress', 'excitement', 'calm', 'confusion'
    ]
    
    def __init__(self, 
                 model_name: str = "j-hartmann/emotion-english-distilroberta-base",
                 device: Optional[str] = None,
                 use_intensity: bool = True,
                 temperature: float = 1.0):
        """
        Initialize emotion model.
        
        Args:
            model_name: Hugging Face model identifier
            device: 'cuda', 'cpu', or None (auto-detect)
            use_intensity: If True, compute intensity scores from probabilities
            temperature: Temperature scaling for confidence calibration (1.0 = no calibration)
                         Higher values (>1.0) = lower confidence, Lower values (<1.0) = higher confidence
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.use_intensity = use_intensity
        self.temperature = temperature
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        logger.info(f"Loading model: {model_name} on {device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            self.model.to(device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                return_all_scores=True
            )
            
            logger.info(f"Model loaded successfully")
            
            # Inspect model's actual label space
            self._inspect_model_labels()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _inspect_model_labels(self) -> None:
        """Inspect the model's actual label space and log discrepancies."""
        try:
            # Get model's label mapping from config
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
                model_labels = list(self.model.config.id2label.values())
                model_labels_lower = [label.lower() for label in model_labels]
                
                logger.info(f"Model supports {len(model_labels)} labels: {model_labels}")
                
                # Compare with expected emotions
                expected_lower = [e.lower() for e in self.EMOTION_LABELS]
                missing_in_model = [e for e in expected_lower if e not in model_labels_lower]
                extra_in_model = [l for l in model_labels_lower if l not in expected_lower]
                
                if missing_in_model:
                    logger.warning(f"Expected emotions not found in model: {missing_in_model}")
                    logger.warning(f"Model may not output these emotions, which could explain class imbalance")
                
                if extra_in_model:
                    logger.info(f"Model has additional labels not in expected list: {extra_in_model}")
                
                # Store model labels for later comparison with predictions
                self._model_labels = model_labels
                self._model_labels_lower = model_labels_lower
            else:
                logger.warning("Could not inspect model label space (config.id2label not available)")
                self._model_labels = []
                self._model_labels_lower = []
        except Exception as e:
            logger.warning(f"Error inspecting model labels: {e}")
            self._model_labels = []
            self._model_labels_lower = []
    
    def log_model_limitations(self, predictions: List[Dict]) -> None:
        """
        Log model limitations by comparing supported labels with actual predictions.
        
        Args:
            predictions: List of prediction dictionaries with 'label' key
        """
        if not hasattr(self, '_model_labels') or not self._model_labels:
            logger.warning("Model labels not available for limitation analysis")
            return
        
        predicted_labels = [p.get('label', '').lower() for p in predictions]
        unique_predicted = set(predicted_labels)
        
        # Find emotions that model supports but were never predicted
        supported_but_not_predicted = [
            label for label in self._model_labels_lower 
            if label not in unique_predicted
        ]
        
        if supported_but_not_predicted:
            logger.warning(f"⚠️  Model Limitation: Model supports these emotions but they were never predicted:")
            logger.warning(f"    {supported_but_not_predicted}")
            logger.warning(f"    This suggests model bias - these emotions may need fine-tuning or a different model")
            
            # Check if these are rare emotions that should be predicted
            rare_emotions = ['anger', 'disgust', 'neutral']
            rare_missing = [e for e in supported_but_not_predicted if e in rare_emotions]
            if rare_missing:
                logger.warning(f"    Rare emotions missing: {rare_missing}")
                logger.warning(f"    Consider: fine-tuning, class weights, or alternative model")
    
    def get_model_labels(self) -> List[str]:
        """
        Get the actual labels supported by the model.
        
        Returns:
            List of label strings supported by the model
        """
        try:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
                return list(self.model.config.id2label.values())
            else:
                logger.warning("Model config.id2label not available, returning expected labels")
                return self.EMOTION_LABELS
        except Exception as e:
            logger.warning(f"Error getting model labels: {e}")
            return self.EMOTION_LABELS
    
    def predict_emotion(self, 
                       text: str,
                       return_probs: bool = False) -> Dict:
        """
        Predict emotion label and intensity for a single text.
        
        Args:
            text: Input text (should be preprocessed sequence)
            return_probs: If True, return all class probabilities
            
        Returns:
            Dictionary with:
            - 'label': Primary emotion label
            - 'intensity': Intensity score (0.0-1.0)
            - 'probabilities': All class probabilities (if return_probs=True)
        """
        if not text or not text.strip():
            return {
                'label': 'neutral',
                'intensity': 0.0,
                'probabilities': {}
            }
        
        try:
            # Get predictions from pipeline
            results = self.pipeline(text)
            
            # Extract probabilities (results is list of dicts)
            probs = {item['label']: item['score'] for item in results[0]}
            
            # Apply temperature scaling for confidence calibration
            if self.temperature != 1.0:
                probs = self._apply_temperature_scaling(probs)
            
            # Get primary emotion (highest probability)
            primary_label = max(probs, key=probs.get)
            primary_prob = probs[primary_label]
            
            # Map to our emotion taxonomy if needed
            mapped_label = self._map_emotion_label(primary_label)
            
            # Compute intensity (0.0-1.0) from probability
            # Higher probability = higher intensity
            intensity = float(primary_prob) if self.use_intensity else 0.5
            
            result = {
                'label': mapped_label,
                'intensity': intensity,
            }
            
            if return_probs:
                result['probabilities'] = probs
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'label': 'neutral',
                'intensity': 0.0,
                'probabilities': {}
            }
    
    def predict_batch(self, 
                     texts: List[str],
                     batch_size: int = 32,
                     return_probs: bool = False) -> List[Dict]:
        """
        Predict emotions for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            return_probs: If True, return all class probabilities
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Predicting emotions for {len(texts)} texts")
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = [self.predict_emotion(text, return_probs) for text in batch]
            results.extend(batch_results)
        
        logger.info(f"Completed predictions for {len(results)} texts")
        return results
    
    def _apply_temperature_scaling(self, probs: Dict[str, float]) -> Dict[str, float]:
        """
        Apply temperature scaling to calibrate confidence scores.
        
        Temperature scaling: softmax(logits / temperature)
        - temperature > 1.0: Reduces confidence (makes probabilities more uniform)
        - temperature < 1.0: Increases confidence (makes probabilities more peaked)
        - temperature = 1.0: No change
        
        Args:
            probs: Dictionary of label -> probability
            
        Returns:
            Calibrated probabilities
        """
        if self.temperature == 1.0:
            return probs
        
        # Convert probabilities to logits (inverse softmax)
        # Use log to avoid numerical issues
        logits = {label: np.log(prob + 1e-10) for label, prob in probs.items()}
        
        # Apply temperature scaling: logits / temperature
        scaled_logits = {label: logit / self.temperature for label, logit in logits.items()}
        
        # Convert back to probabilities using softmax
        exp_logits = np.array([np.exp(logit) for logit in scaled_logits.values()])
        sum_exp = exp_logits.sum()
        calibrated_probs = {
            label: float(np.exp(logit) / sum_exp)
            for label, logit in zip(scaled_logits.keys(), scaled_logits.values())
        }
        
        return calibrated_probs
    
    def check_emotion_support(self, predictions: List[Dict]) -> Dict:
        """
        Check which emotions are predicted and detect class imbalance.
        
        Args:
            predictions: List of prediction dictionaries with 'label' key
            
        Returns:
            Dictionary with emotion counts and warnings
        """
        emotion_counts = {}
        for pred in predictions:
            label = pred.get('label', 'unknown')
            emotion_counts[label] = emotion_counts.get(label, 0) + 1
        
        total = len(predictions)
        emotion_percentages = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}
        
        # Get model's actual supported labels
        model_labels = self.get_model_labels()
        model_labels_lower = [l.lower() for l in model_labels]
        
        # Expected emotions from the model (use actual model labels if available)
        expected_emotions = [e.lower() for e in model_labels] if model_labels else ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
        predicted_labels_lower = [e.lower() for e in emotion_counts.keys()]
        missing_emotions = [e for e in expected_emotions if e not in predicted_labels_lower]
        
        warnings = []
        if missing_emotions:
            warnings.append(f"Missing emotions (not predicted): {', '.join(missing_emotions)}")
            # Check if these emotions are even supported by the model
            unsupported = [e for e in missing_emotions if e not in model_labels_lower]
            if unsupported:
                warnings.append(f"⚠️  These emotions may not be supported by the model: {', '.join(unsupported)}")
        
        # Check for class imbalance (if one emotion is >50% or another is <5%)
        if total > 0 and len(emotion_counts) > 1:
            max_pct = max(emotion_percentages.values())
            min_pct = min(emotion_percentages.values())
            max_emotion = max(emotion_percentages, key=emotion_percentages.get)
            min_emotion = min(emotion_percentages, key=emotion_percentages.get)
            
            imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
            
            if max_pct > 50:
                warnings.append(f"Class imbalance: {max_emotion} is {max_pct:.1f}% of predictions")
            if min_pct < 5:
                warnings.append(f"Rare emotion: {min_emotion} is only {min_pct:.1f}% of predictions")
            if imbalance_ratio > 10:
                warnings.append(f"⚠️  Severe class imbalance: {imbalance_ratio:.1f}x ratio ({max_emotion} vs {min_emotion})")
        
        return {
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'missing_emotions': missing_emotions,
            'model_labels': model_labels,
            'warnings': warnings
        }
    
    def _map_emotion_label(self, label: str) -> str:
        """
        Map model's emotion labels to our taxonomy.
        
        The j-hartmann/emotion-english-distilroberta-base model uses:
        - joy, sadness, anger, fear, surprise, disgust, neutral
        
        We map these to our extended taxonomy.
        
        Args:
            label: Original model label
            
        Returns:
            Mapped label in our taxonomy
        """
        # Common mappings (case-insensitive)
        label_lower = label.lower().strip()
        
        # Direct mappings from j-hartmann model
        mapping = {
            # Standard emotions
            'joy': 'joy',
            'happiness': 'joy',
            'happy': 'joy',
            'sadness': 'sadness',
            'sad': 'sadness',
            'anger': 'anger',
            'angry': 'anger',
            'fear': 'fear',
            'afraid': 'fear',
            'surprise': 'surprise',
            'surprised': 'surprise',
            'disgust': 'disgust',
            'disgusted': 'disgust',
            'neutral': 'neutral',
            
            # Extended taxonomy
            'anxiety': 'anxiety',
            'anxious': 'anxiety',
            'stress': 'stress',
            'stressed': 'stress',
            'excitement': 'excitement',
            'excited': 'excitement',
            'calm': 'calm',
            'confusion': 'confusion',
            'confused': 'confusion',
            
            # Common variations
            'love': 'joy',
            'hate': 'anger',
            'worried': 'anxiety',
            'nervous': 'anxiety',
            'frustrated': 'anger',
            'upset': 'sadness',
            'depressed': 'sadness',
        }
        
        # Return mapped label or original if not found
        return mapping.get(label_lower, label_lower)
    
    def detect_ambiguity(self, 
                         text: str,
                         history: Optional[pd.DataFrame] = None,
                         ambiguity_threshold: float = 0.3) -> Tuple[bool, float]:
        """
        Detect if an utterance is ambiguous (multiple emotions with similar probabilities).
        
        Args:
            text: Input text
            history: Optional history for context
            ambiguity_threshold: Threshold for ambiguity (difference between top 2 probs)
            
        Returns:
            Tuple of (is_ambiguous, ambiguity_score)
        """
        result = self.predict_emotion(text, return_probs=True)
        probs = result.get('probabilities', {})
        
        if len(probs) < 2:
            return False, 0.0
        
        # Get top 2 probabilities
        sorted_probs = sorted(probs.values(), reverse=True)
        top_prob = sorted_probs[0]
        second_prob = sorted_probs[1]
        
        # Ambiguity score: difference between top 2
        diff = top_prob - second_prob
        
        # If difference is small, it's ambiguous
        is_ambiguous = diff < ambiguity_threshold
        ambiguity_score = 1.0 - diff  # Higher score = more ambiguous
        
        return is_ambiguous, ambiguity_score


class CalibrationModel:
    """
    Calibration model for probability calibration using Platt scaling or isotonic regression.
    
    Example usage with synthetic data:
        ```python
        from src.modeling import CalibrationModel
        import numpy as np
        
        # Generate synthetic predictions and true labels
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])  # Example probabilities
        true_labels = np.array([0, 0, 1])  # True class indices
        
        # Fit Platt scaling
        calibrator = CalibrationModel()
        calibrator.fit_platt_scaling(probs, true_labels)
        
        # Calibrate new predictions
        new_probs = np.array([[0.85, 0.15], [0.75, 0.25]])
        calibrated = calibrator.calibrate(new_probs)
        ```
    """
    
    def __init__(self):
        """Initialize calibration model."""
        self.calibrator = None
        self.method = None
        self.is_fitted = False
    
    def fit_platt_scaling(self, probs: np.ndarray, true_labels: np.ndarray) -> None:
        """
        Fit Platt scaling (logistic regression) for probability calibration.
        
        Args:
            probs: Array of predicted probabilities (n_samples, n_classes)
            true_labels: Array of true class labels (n_samples,)
        """
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            logger.warning("scikit-learn not available. Install with: pip install scikit-learn")
            return
        
        try:
            # Platt scaling is essentially logistic regression
            # We'll use sklearn's CalibratedClassifierCV with method='sigmoid'
            base_estimator = LogisticRegression()
            self.calibrator = CalibratedClassifierCV(base_estimator, method='sigmoid', cv='prefit')
            
            # For Platt scaling, we need to fit on probabilities
            # Create a dummy classifier that outputs the probabilities
            from sklearn.base import BaseEstimator, ClassifierMixin
            class ProbClassifier(BaseEstimator, ClassifierMixin):
                def __init__(self, probs):
                    self.probs = probs
                def predict_proba(self, X):
                    return self.probs
                def fit(self, X, y):
                    return self
            
            dummy_clf = ProbClassifier(probs)
            self.calibrator.fit(dummy_clf, true_labels)
            self.method = 'platt'
            self.is_fitted = True
            logger.info("Platt scaling fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting Platt scaling: {e}")
    
    def fit_isotonic_regression(self, probs: np.ndarray, true_labels: np.ndarray) -> None:
        """
        Fit isotonic regression for probability calibration.
        
        Args:
            probs: Array of predicted probabilities (n_samples, n_classes)
            true_labels: Array of true class labels (n_samples,)
        """
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            logger.warning("scikit-learn not available. Install with: pip install scikit-learn")
            return
        
        try:
            # For isotonic regression, we calibrate per class
            n_classes = probs.shape[1]
            self.calibrator = []
            
            for class_idx in range(n_classes):
                ir = IsotonicRegression(out_of_bounds='clip')
                class_probs = probs[:, class_idx]
                class_labels = (true_labels == class_idx).astype(int)
                ir.fit(class_probs, class_labels)
                self.calibrator.append(ir)
            
            self.method = 'isotonic'
            self.is_fitted = True
            logger.info("Isotonic regression fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting isotonic regression: {e}")
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.
        
        Args:
            probs: Array of predicted probabilities (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted or self.calibrator is None:
            logger.warning("Calibrator not fitted. Returning original probabilities.")
            return probs
        
        try:
            if self.method == 'platt':
                # For Platt scaling with CalibratedClassifierCV
                # We need to use predict_proba, but our dummy classifier approach
                # may not work directly. For now, return original probabilities
                # with a note that this needs a proper implementation
                logger.warning("Platt scaling calibration not fully implemented. Returning original probabilities.")
                return probs
            elif self.method == 'isotonic':
                # Apply isotonic regression per class
                calibrated = np.zeros_like(probs)
                for class_idx, ir in enumerate(self.calibrator):
                    calibrated[:, class_idx] = ir.transform(probs[:, class_idx])
                # Renormalize to ensure probabilities sum to 1
                calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
                return calibrated
            else:
                return probs
        except Exception as e:
            logger.error(f"Error applying calibration: {e}")
            return probs


def batch_for_inference(df: pd.DataFrame, 
                       sequence_col: str = 'Sequence',
                       batch_size: int = 32) -> List[pd.DataFrame]:
    """
    Prepare batches for model inference.
    
    Args:
        df: DataFrame with sequences ready for inference
        sequence_col: Name of sequence column
        batch_size: Batch size for inference
        
    Returns:
        List of DataFrame batches
    """
    batches = []
    for i in range(0, len(df), batch_size):
        batches.append(df.iloc[i:i+batch_size])
    return batches


def run_inference_pipeline(df: pd.DataFrame,
                          model: EmotionModel,
                          sequence_col: str = 'Sequence',
                          batch_size: int = 32,
                          apply_post_processing: bool = True,
                          use_context_aware: bool = True) -> pd.DataFrame:
    """
    Run full inference pipeline on DataFrame.
    
    Args:
        df: DataFrame with sequences
        model: EmotionModel instance
        sequence_col: Name of sequence column
        batch_size: Batch size for processing
        
    Returns:
        DataFrame with added columns:
        - PrimaryEmotionLabel
        - IntensityScore_Primary
        - AmbiguityFlag
    """
    logger.info(f"Running inference on {len(df)} sequences")
    
    sequences = df[sequence_col].tolist()
    
    # Get predictions
    predictions = model.predict_batch(sequences, batch_size=batch_size)
    
    # Check for class imbalance
    emotion_analysis = model.check_emotion_support(predictions)
    if emotion_analysis['warnings']:
        for warning in emotion_analysis['warnings']:
            logger.warning(f"Emotion distribution: {warning}")
    logger.info(f"Emotion distribution: {emotion_analysis['emotion_counts']}")
    
    # Log model limitations (emotions supported but never predicted)
    model.log_model_limitations(predictions)
    
    # Extract results
    df = df.copy()
    df['PrimaryEmotionLabel'] = [p['label'] for p in predictions]
    df['IntensityScore_Primary'] = [p['intensity'] for p in predictions]
    
    # Store original model predictions before post-processing
    df['OriginalEmotionLabel'] = df['PrimaryEmotionLabel'].copy()
    df['OriginalIntensityScore'] = df['IntensityScore_Primary'].copy()
    
    # Flag high-confidence predictions (intensity >0.95)
    df['HighConfidenceFlag'] = df['IntensityScore_Primary'] > 0.95
    high_conf_count = df['HighConfidenceFlag'].sum()
    if high_conf_count > 0:
        logger.info(f"Flagged {high_conf_count} high-confidence predictions (intensity >0.95) for review")
    
    # Detect ambiguity
    ambiguity_flags = []
    for seq in sequences:
        is_amb, _ = model.detect_ambiguity(seq)
        ambiguity_flags.append(1 if is_amb else 0)
    
    df['AmbiguityFlag'] = ambiguity_flags
    
    # Apply post-processing if requested
    if apply_post_processing:
        try:
            if use_context_aware and sequence_col in df.columns:
                # Use context-aware post-processing
                from src.context_aware_postprocess import apply_context_aware_post_processing
                logger.info("Applying context-aware post-processing...")
                df = apply_context_aware_post_processing(
                    df, text_col='NormalizedText',
                    label_col='PrimaryEmotionLabel',
                    intensity_col='IntensityScore_Primary',
                    context_col=sequence_col
                )
            else:
                # Use standard post-processing
                from src.postprocess import apply_post_processing as apply_pp
                logger.info("Applying post-processing rules...")
                df = apply_pp(df, text_col='NormalizedText',
                             label_col='PrimaryEmotionLabel',
                             intensity_col='IntensityScore_Primary')
            
            # Use corrected labels if available
            if 'CorrectedLabel' in df.columns:
                df['PrimaryEmotionLabel'] = df['CorrectedLabel']
                df['IntensityScore_Primary'] = df['CorrectedIntensity']
                # Ensure PostProcessingOverride column exists
                if 'PostProcessingOverride' not in df.columns:
                    df['PostProcessingOverride'] = None
                overrides = df['WasOverridden'].sum() if 'WasOverridden' in df.columns else 0
                reviews = df['NeedsReview'].sum() if 'NeedsReview' in df.columns else 0
                logger.info(f"Post-processing applied: {overrides} overrides, {reviews} flagged for review")
        except ImportError:
            logger.warning("Post-processing module not available, skipping")
    
    logger.info("Inference complete")
    return df

