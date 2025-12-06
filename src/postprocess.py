"""
Post-processing rules to fix common model misclassifications.
Implements rule-based overrides for obvious cases.
"""

import re
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class EmotionPostProcessor:
    """Post-process emotion predictions with rule-based corrections."""
    
    # Positive emotion indicators
    POSITIVE_KEYWORDS = [
        'thanks', 'thank you', 'grateful', 'gratitude', 'appreciate',
        'hilarious', 'funny', 'laugh', 'lol', 'haha', 'hehe',
        'better', 'improving', 'progress', 'getting better',
        'happy', 'glad', 'excited', 'great', 'awesome', 'amazing',
        'love', '❤️', '💕', '😊', '😄', '😃', '😂', '🎉', '🎊',
        'yay', 'woohoo', 'yes!', 'let\'s do this', 'ready',
        'feeling good', 'feeling great', 'feeling better'
    ]
    
    # Negative emotion indicators (for validation)
    NEGATIVE_KEYWORDS = [
        'sad', 'depressed', 'down', 'upset', 'angry', 'mad',
        'anxious', 'worried', 'stressed', 'overwhelmed',
        'tired', 'exhausted', 'drained', 'hopeless',
        '😢', '😭', '😫', '😰', '😨', '😟'
    ]
    
    # Neutral indicators
    NEUTRAL_KEYWORDS = [
        'brb', 'be right back', 'need to', 'going to',
        'just', 'okay', 'ok', 'sure', 'alright',
        'update', 'checking', 'noting'
    ]
    
    # Gratitude patterns (should be joy, not sadness)
    GRATITUDE_PATTERNS = [
        r'thanks?\s+for',
        r'thank\s+you',
        r'\bthx\b',  # Added abbreviation
        r'grateful',
        r'appreciate',
        r'means?\s+a\s+lot',
        r'being\s+there',  # "for being there"
        r'❤️',
        r'💕',
        r'red_heart'  # Demojized heart
    ]
    
    # Positive progress patterns (should be joy/positive, not sadness)
    PROGRESS_PATTERNS = [
        r'getting\s+better',
        r'improving',
        r'making\s+progress',
        r'feeling\s+better',
        r'doing\s+better'
    ]
    
    # Humor patterns (should be joy, not sadness)
    HUMOR_PATTERNS = [
        r'hilarious',
        r'funny',
        r'lol',
        r'haha',
        r'😂',
        r'😄',
        r'😃'
    ]
    
    def __init__(self, 
                 override_confidence: float = 0.7,
                 min_confidence_for_override: float = 0.9):
        """
        Initialize post-processor.
        
        Args:
            override_confidence: Confidence to assign to rule-based overrides
            min_confidence_for_override: Minimum model confidence to consider override
        """
        self.override_confidence = override_confidence
        self.min_confidence_for_override = min_confidence_for_override
    
    def check_positive_indicators(self, text: str) -> bool:
        """Check if text contains positive emotion indicators."""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.POSITIVE_KEYWORDS)
    
    def check_gratitude(self, text: str) -> bool:
        """Check if text expresses gratitude."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.GRATITUDE_PATTERNS)
    
    def check_progress(self, text: str) -> bool:
        """Check if text indicates positive progress."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.PROGRESS_PATTERNS)
    
    def check_humor(self, text: str) -> bool:
        """Check if text contains humor/laughter."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.HUMOR_PATTERNS)
    
    def check_neutral(self, text: str, intensity: float = None) -> bool:
        """Check if text is neutral (no clear emotion)."""
        text_lower = text.lower()
        
        # Low confidence = likely neutral
        if intensity is not None and intensity < 0.5:
            return True
        
        # Neutral keywords - check if text contains neutral phrases
        has_neutral = any(keyword.lower() in text_lower for keyword in self.NEUTRAL_KEYWORDS)
        
        # If has neutral keywords, check confidence threshold
        if has_neutral:
            # For neutral keywords, even medium confidence might be neutral
            # because model may be forcing an emotion
            if intensity is None or intensity < 0.8:  # Raised threshold for neutral keywords
                return True
        
        return False
    
    def post_process(self, 
                    text: str,
                    predicted_label: str,
                    predicted_intensity: float,
                    original_probs: Optional[Dict] = None) -> Tuple[str, float, bool]:
        """
        Post-process emotion prediction with rule-based corrections.
        
        Args:
            text: Input text (normalized)
            predicted_label: Model's predicted emotion label
            predicted_intensity: Model's predicted intensity
            original_probs: Original probability distribution (optional)
            
        Returns:
            Tuple of (corrected_label, corrected_intensity, was_overridden)
        """
        text_lower = text.lower()
        was_overridden = False
        corrected_label = predicted_label
        corrected_intensity = predicted_intensity
        
        # Rule 1: Positive keywords → joy (override sadness/fear)
        if self.check_positive_indicators(text_lower):
            if predicted_label in ['sadness', 'fear'] and predicted_intensity >= self.min_confidence_for_override:
                corrected_label = 'joy'
                corrected_intensity = self.override_confidence
                was_overridden = True
                logger.info(f"Override: Positive indicators detected. {predicted_label} → joy")
        
        # Rule 2: Gratitude → joy (override sadness)
        if self.check_gratitude(text_lower):
            if predicted_label == 'sadness' and predicted_intensity >= self.min_confidence_for_override:
                corrected_label = 'joy'
                corrected_intensity = self.override_confidence
                was_overridden = True
                logger.info(f"Override: Gratitude detected. sadness → joy")
        
        # Rule 3: Progress → joy/positive (override sadness)
        if self.check_progress(text_lower):
            if predicted_label == 'sadness' and predicted_intensity >= self.min_confidence_for_override:
                corrected_label = 'joy'
                corrected_intensity = self.override_confidence
                was_overridden = True
                logger.info(f"Override: Progress detected. sadness → joy")
        
        # Rule 4: Humor → joy (override sadness)
        if self.check_humor(text_lower):
            if predicted_label == 'sadness':
                corrected_label = 'joy'
                corrected_intensity = self.override_confidence
                was_overridden = True
                logger.info(f"Override: Humor detected. sadness → joy")
        
        # Rule 5: Neutral text → neutral (enhanced detection)
        if self.check_neutral(text_lower, predicted_intensity):
            # Override to neutral if detected
            corrected_label = 'neutral'
            corrected_intensity = min(0.3, predicted_intensity * 0.5)  # Lower confidence for neutral
            was_overridden = True
            logger.info(f"Override: Neutral detected. {predicted_label} → neutral")
        
        return corrected_label, corrected_intensity, was_overridden
    
    def flag_for_review(self,
                       text: str,
                       predicted_label: str,
                       predicted_intensity: float) -> bool:
        """
        Flag prediction for human review.
        
        Args:
            text: Input text
            predicted_label: Predicted emotion
            predicted_intensity: Predicted intensity
            
        Returns:
            True if should be flagged for review
        """
        text_lower = text.lower()
        
        # Flag high-confidence negative predictions on positive-sounding text
        if predicted_label in ['sadness', 'fear'] and predicted_intensity >= 0.95:
            if self.check_positive_indicators(text_lower):
                return True
        
        # Flag high-confidence predictions that were overridden
        if predicted_intensity >= 0.95 and self.check_positive_indicators(text_lower):
            if predicted_label in ['sadness', 'fear']:
                return True
        
        return False


def apply_post_processing(df, text_col='NormalizedText', 
                         label_col='PrimaryEmotionLabel',
                         intensity_col='IntensityScore_Primary'):
    """
    Apply post-processing to a DataFrame of predictions.
    
    Args:
        df: DataFrame with predictions
        text_col: Column name for text
        label_col: Column name for predicted label
        intensity_col: Column name for intensity
        
    Returns:
        DataFrame with corrected predictions and override flags
    """
    processor = EmotionPostProcessor()
    
    corrected_labels = []
    corrected_intensities = []
    override_flags = []
    review_flags = []
    
    for _, row in df.iterrows():
        text = row[text_col]
        label = row[label_col]
        intensity = row[intensity_col]
        
        corrected_label, corrected_intensity, was_overridden = processor.post_process(
            text, label, intensity
        )
        
        needs_review = processor.flag_for_review(text, label, intensity)
        
        corrected_labels.append(corrected_label)
        corrected_intensities.append(corrected_intensity)
        override_flags.append(1 if was_overridden else 0)
        review_flags.append(1 if needs_review else 0)
    
    df = df.copy()
    df['CorrectedLabel'] = corrected_labels
    df['CorrectedIntensity'] = corrected_intensities
    df['WasOverridden'] = override_flags
    df['NeedsReview'] = review_flags
    
    return df

