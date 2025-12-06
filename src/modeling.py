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
                 use_intensity: bool = True):
        """
        Initialize emotion model.
        
        Args:
            model_name: Hugging Face model identifier
            device: 'cuda', 'cpu', or None (auto-detect)
            use_intensity: If True, compute intensity scores from probabilities
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.use_intensity = use_intensity
        
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
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
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
    
    # Extract results
    df = df.copy()
    df['PrimaryEmotionLabel'] = [p['label'] for p in predictions]
    df['IntensityScore_Primary'] = [p['intensity'] for p in predictions]
    
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
                overrides = df['WasOverridden'].sum() if 'WasOverridden' in df.columns else 0
                reviews = df['NeedsReview'].sum() if 'NeedsReview' in df.columns else 0
                logger.info(f"Post-processing applied: {overrides} overrides, {reviews} flagged for review")
        except ImportError:
            logger.warning("Post-processing module not available, skipping")
    
    logger.info("Inference complete")
    return df

