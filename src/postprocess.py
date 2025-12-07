"""
Post-processing rules to fix common model misclassifications.
Implements rule-based overrides for obvious cases.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

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
    
    # Anxiety patterns (map to fear)
    ANXIETY_PATTERNS = [
        r'\banxious\b',
        r'\bworried\b',
        r'\bnervous\b',
        r'\buneasy\b',
        r'\bapprehensive\b'
    ]
    
    # Depression patterns (map to sadness)
    DEPRESSION_PATTERNS = [
        r'\bdepressed\b',
        r'\bdepression\b',
        r'\bhopeless\b',
        r'\bdespair\b',
        r'\bhelpless\b'
    ]
    
    # Sarcasm patterns
    SARCASM_PATTERNS = [
        r'\bsure\b.*',
        r'yeah\s+right',
        r'as\s+if',
        r'whatever',
        r'oh\s+great'
    ]
    
    # Mixed emotion indicators
    MIXED_EMOTION_INDICATORS = [
        'but', 'however', 'although', 'even though', 'though',
        'yet', 'still', 'despite', 'while'
    ]
    
    # Negation patterns
    NEGATION_PATTERNS = [
        r'\bnot\s+(happy|sad|angry|afraid|excited|glad)',
        r'\bno\s+(joy|happiness|excitement)',
        r'\bnever\s+(felt|feeling)\s+(better|worse)',
        r'\bcan\'?t\s+(believe|stand)'
    ]
    
    def __init__(self, 
                 override_confidence: float = 0.7,
                 min_confidence_for_override: float = 0.75):
        """
        Initialize post-processor.
        
        Args:
            override_confidence: Confidence to assign to rule-based overrides
            min_confidence_for_override: Minimum model confidence to consider override (default: 0.75)
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
    
    def check_anxiety(self, text: str) -> bool:
        """Check if text contains anxiety indicators."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.ANXIETY_PATTERNS)
    
    def check_depression(self, text: str) -> bool:
        """Check if text contains depression indicators."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.DEPRESSION_PATTERNS)
    
    def check_sarcasm(self, text: str) -> bool:
        """Check if text contains sarcasm indicators."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.SARCASM_PATTERNS)
    
    def check_mixed_emotions(self, text: str) -> bool:
        """Check if text contains mixed emotion indicators."""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in self.MIXED_EMOTION_INDICATORS)
    
    def check_negation(self, text: str) -> bool:
        """Check if text contains negation patterns."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.NEGATION_PATTERNS)
    
    def check_neutral(self, text: str, intensity: float = None) -> bool:
        """Check if text is neutral (no clear emotion)."""
        text_lower = text.lower()
        
        # Low confidence = likely neutral (updated threshold)
        if intensity is not None and intensity < 0.6:
            # Check if no strong emotion keywords present
            has_strong_emotion = (
                any(keyword in text_lower for keyword in self.POSITIVE_KEYWORDS) or
                any(keyword in text_lower for keyword in self.NEGATIVE_KEYWORDS)
            )
            if not has_strong_emotion:
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
                    original_probs: Optional[Dict] = None) -> Tuple[str, float, bool, Optional[str]]:
        """
        Post-process emotion prediction with rule-based corrections.
        
        Args:
            text: Input text (normalized)
            predicted_label: Model's predicted emotion label
            predicted_intensity: Model's predicted intensity
            original_probs: Original probability distribution (optional)
            
        Returns:
            Tuple of (corrected_label, corrected_intensity, was_overridden, override_type)
            override_type: 'positive_keywords', 'gratitude', 'progress', 'humor', 'neutral', or None
        """
        text_lower = text.lower()
        was_overridden = False
        corrected_label = predicted_label
        corrected_intensity = predicted_intensity
        override_type = None
        
        # Rule 1: Positive keywords → joy (override sadness/fear, but not if already joy)
        if self.check_positive_indicators(text_lower) and predicted_label != 'joy':
            if predicted_label in ['sadness', 'fear'] and predicted_intensity >= self.min_confidence_for_override:
                corrected_label = 'joy'
                corrected_intensity = self.override_confidence
                was_overridden = True
                override_type = 'positive_keywords'
                logger.info(f"Override: Positive indicators detected. {predicted_label} → joy")
        
        # Rule 2: Gratitude → joy (override sadness, but not if already joy)
        if self.check_gratitude(text_lower) and not was_overridden and predicted_label != 'joy':
            if predicted_label == 'sadness' and predicted_intensity >= self.min_confidence_for_override:
                corrected_label = 'joy'
                corrected_intensity = self.override_confidence
                was_overridden = True
                override_type = 'gratitude'
                logger.info(f"Override: Gratitude detected. sadness → joy")
        
        # Rule 3: Progress → joy/positive (override sadness, but not if already joy)
        if self.check_progress(text_lower) and not was_overridden and predicted_label != 'joy':
            if predicted_label == 'sadness' and predicted_intensity >= self.min_confidence_for_override:
                corrected_label = 'joy'
                corrected_intensity = self.override_confidence
                was_overridden = True
                override_type = 'progress'
                logger.info(f"Override: Progress detected. sadness → joy")
        
        # Rule 4: Humor → joy (override sadness, but not if already joy)
        if self.check_humor(text_lower) and not was_overridden and predicted_label != 'joy':
            if predicted_label == 'sadness':
                corrected_label = 'joy'
                corrected_intensity = self.override_confidence
                was_overridden = True
                override_type = 'humor'
                logger.info(f"Override: Humor detected. sadness → joy")
        
        # Rule 5: Intensity-based override (low confidence + positive keywords → joy, but not if already joy)
        if not was_overridden and predicted_intensity < 0.6 and predicted_label != 'joy':
            if self.check_positive_indicators(text_lower):
                corrected_label = 'joy'
                corrected_intensity = self.override_confidence
                was_overridden = True
                override_type = 'low_confidence_positive'
                logger.info(f"Override: Low confidence positive detected. {predicted_label} → joy")
        
        # Rule 6: Anxiety patterns (validate fear predictions)
        if self.check_anxiety(text_lower) and not was_overridden:
            if predicted_label == 'fear':
                # Keep as fear, but log for validation
                logger.debug(f"Anxiety pattern detected, keeping fear label")
        
        # Rule 7: Depression patterns (validate sadness predictions)
        if self.check_depression(text_lower) and not was_overridden:
            if predicted_label == 'sadness':
                # Keep as sadness, but log for validation
                logger.debug(f"Depression pattern detected, keeping sadness label")
        
        # Rule 8: Neutral text → neutral (enhanced detection, but not if already neutral)
        if self.check_neutral(text_lower, predicted_intensity) and not was_overridden and predicted_label != 'neutral':
            # Override to neutral if detected
            corrected_label = 'neutral'
            corrected_intensity = min(0.3, predicted_intensity * 0.5)  # Lower confidence for neutral
            was_overridden = True
            override_type = 'neutral'
            logger.info(f"Override: Neutral detected. {predicted_label} → neutral")
        
        # Final validation: only mark as override if label actually changed
        # This prevents joy→joy, sadness→sadness, etc. overrides
        if corrected_label == predicted_label:
            if was_overridden:  # Only log if an override was attempted
                logger.debug(f"Override validation: {predicted_label}→{corrected_label} reset (no change, override_type was: {override_type})")
            was_overridden = False
            override_type = None
            corrected_intensity = predicted_intensity  # Keep original intensity
        
        return corrected_label, corrected_intensity, was_overridden, override_type
    
    def flag_for_review(self,
                       text: str,
                       predicted_label: str,
                       predicted_intensity: float,
                       high_confidence_flag: bool = False,
                       previous_emotions: Optional[List[str]] = None) -> bool:
        """
        Flag prediction for human review.
        
        Args:
            text: Input text
            predicted_label: Predicted emotion
            predicted_intensity: Predicted intensity
            high_confidence_flag: Flag from model indicating intensity >0.95
            previous_emotions: List of previous emotion labels for temporal pattern detection
            
        Returns:
            True if should be flagged for review
        """
        text_lower = text.lower()
        
        # Don't flag low-intensity predictions (likely false positives)
        # Exception: high-confidence flags (≥0.95) are still flagged
        if predicted_intensity < 0.8 and not high_confidence_flag:
            return False
        
        # Flag high-confidence negative emotions (sadness, fear, anger)
        if predicted_intensity >= 0.95:
            if predicted_label in ['sadness', 'fear', 'anger']:
                return True
        
        # Flag high-confidence predictions on positive-sounding text (potential misclassification)
        if predicted_intensity >= 0.95:
            if predicted_label in ['sadness', 'fear'] and self.check_positive_indicators(text_lower):
                return True
        
        # Flag high-confidence positive emotions when context suggests negative
        # (This would be handled by context-aware post-processing, but flag here too)
        if predicted_intensity >= 0.95 and predicted_label == 'joy':
            # Check for potential sarcasm or mixed signals
            if self.check_sarcasm(text_lower):
                return True
        
        # Temporal pattern: sudden emotion shifts
        if previous_emotions and len(previous_emotions) >= 2:
            recent_emotions = previous_emotions[-2:]  # Last 2 emotions
            if len(set(recent_emotions)) > 1:  # Different emotions in recent history
                # Check for dramatic shifts
                negative_emotions = ['sadness', 'fear', 'anger']
                positive_emotions = ['joy', 'surprise']
                
                if recent_emotions[-1] in positive_emotions and predicted_label in negative_emotions:
                    # Sudden shift from positive to negative
                    return True
                if recent_emotions[-1] in negative_emotions and predicted_label in positive_emotions:
                    # Sudden shift from negative to positive (could be genuine or sarcasm)
                    if predicted_intensity >= 0.9:  # High confidence on shift
                        return True
        
        # Flag sarcasm detected with joy prediction (only if high confidence)
        if self.check_sarcasm(text_lower) and predicted_label == 'joy':
            if predicted_intensity >= 0.85:  # Only flag high-confidence sarcasm
                return True
        
        # Flag mixed emotions (only if high confidence)
        if self.check_mixed_emotions(text_lower):
            if predicted_intensity >= 0.85:  # Only flag high-confidence mixed emotions
                return True
        
        # Flag negation patterns (only if high confidence)
        if self.check_negation(text_lower):
            if predicted_intensity >= 0.85:  # Only flag high-confidence negation
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
    override_types = []
    review_flags = []
    
    # Track previous emotions per user for temporal pattern detection
    user_emotion_history = {}
    
    for idx, row in df.iterrows():
        text = row[text_col]
        label = row[label_col]
        intensity = row[intensity_col]
        high_confidence = row.get('HighConfidenceFlag', False) if 'HighConfidenceFlag' in df.columns else False
        user_id = row.get('UserID', None)
        
        # Get previous emotions for this user (for temporal pattern detection)
        previous_emotions = None
        if user_id:
            previous_emotions = user_emotion_history.get(user_id, [])
        
        corrected_label, corrected_intensity, was_overridden, override_type = processor.post_process(
            text, label, intensity
        )
        
        needs_review = processor.flag_for_review(
            text, label, intensity,
            high_confidence_flag=high_confidence,
            previous_emotions=previous_emotions
        )
        
        # Update emotion history for this user
        if user_id:
            if user_id not in user_emotion_history:
                user_emotion_history[user_id] = []
            user_emotion_history[user_id].append(corrected_label)
            # Keep only last 5 emotions to avoid memory issues
            if len(user_emotion_history[user_id]) > 5:
                user_emotion_history[user_id] = user_emotion_history[user_id][-5:]
        
        corrected_labels.append(corrected_label)
        corrected_intensities.append(corrected_intensity)
        override_flags.append(1 if was_overridden else 0)
        override_types.append(override_type if was_overridden else None)
        review_flags.append(1 if needs_review else 0)
    
    df = df.copy()
    df['CorrectedLabel'] = corrected_labels
    df['CorrectedIntensity'] = corrected_intensities
    df['WasOverridden'] = override_flags
    df['PostProcessingOverride'] = override_types
    df['NeedsReview'] = review_flags
    # Store post-processing review flag separately
    df['PostProcessingReviewFlag'] = review_flags
    
    return df

