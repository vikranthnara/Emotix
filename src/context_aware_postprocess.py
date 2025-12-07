"""
Context-aware post-processing that uses context to validate overrides.
"""

import logging
from typing import Optional, Tuple
import re

from src.postprocess import EmotionPostProcessor

logger = logging.getLogger(__name__)


class ContextAwarePostProcessor(EmotionPostProcessor):
    """Post-processor that considers context when making corrections."""
    
    def post_process_with_context(self,
                                 text: str,
                                 predicted_label: str,
                                 predicted_intensity: float,
                                 context_text: Optional[str] = None,
                                 context_emotions: Optional[list] = None) -> Tuple[str, float, bool, bool, Optional[str]]:
        """
        Post-process with context awareness.
        
        Args:
            text: Current utterance text
            predicted_label: Model's predicted label
            predicted_intensity: Model's predicted intensity
            context_text: Previous context text (concatenated)
            context_emotions: List of previous emotion labels
            
        Returns:
            Tuple of (corrected_label, corrected_intensity, was_overridden, needs_review, override_type)
        """
        # First apply standard post-processing
        # Note: original_text not available in context-aware post-process, use normalized text
        corrected_label, corrected_intensity, was_overridden, override_type = self.post_process(
            text, predicted_label, predicted_intensity, original_text=None
        )
        
        needs_review = False
        
        # If context is available, use it to validate
        if context_text:
            context_lower = context_text.lower()
            
            # Check for context-emotion consistency
            if was_overridden:
                # If we overrode to joy, check if context supports it
                if corrected_label == 'joy':
                    # Check if context has negative indicators
                    negative_indicators = ['sad', 'down', 'depressed', 'struggling', 'difficult', 
                                          'hard', 'tough', 'bad', 'worst', 'terrible']
                    has_negative_context = any(ind in context_lower for ind in negative_indicators)
                    
                    if has_negative_context:
                        # Context is negative but text is positive
                        # This could be genuine improvement or sarcasm
                        # Flag for review but keep the override (text is more recent)
                        needs_review = True
                        logger.info(f"Context conflict: Negative context but positive text. Flagging for review.")
                
                # If we overrode to neutral, check context
                if corrected_label == 'neutral':
                    # If context shows strong emotion, neutral might be wrong
                    if context_emotions:
                        strong_emotions = [e for e in context_emotions if e and e != 'neutral']
                        if len(strong_emotions) >= 2:
                            # Multiple strong emotions in context, neutral might be transition
                            # Keep neutral but flag if confidence was high
                            if predicted_intensity > 0.7:
                                needs_review = True
            
            # Check for high-confidence suspicious predictions
            if predicted_intensity >= 0.95:
                # High confidence negative on positive-sounding text
                if predicted_label in ['sadness', 'fear']:
                    if self.check_positive_indicators(text):
                        needs_review = True
                        logger.info(f"High confidence negative on positive text. Flagging for review.")
                
                # High confidence positive on negative context
                if predicted_label == 'joy':
                    negative_indicators = ['sad', 'down', 'depressed', 'struggling']
                    if any(ind in context_lower for ind in negative_indicators):
                        # Could be sarcasm or genuine improvement
                        needs_review = True
                        logger.info(f"High confidence positive on negative context. Flagging for review.")
        
        # Also check standard review flags (inherited from EmotionPostProcessor)
        # Pass high_confidence_flag correctly to respect intensity threshold exception
        # Check if intensity >= 0.95 to determine high_confidence_flag
        if not needs_review:
            high_conf_flag = predicted_intensity >= 0.95
            needs_review = self.flag_for_review(text, predicted_label, predicted_intensity, high_confidence_flag=high_conf_flag)
        
        return corrected_label, corrected_intensity, was_overridden, needs_review, override_type


def apply_context_aware_post_processing(df,
                                        text_col='NormalizedText',
                                        label_col='PrimaryEmotionLabel',
                                        intensity_col='IntensityScore_Primary',
                                        context_col='Sequence'):  # Use Sequence column for context
    """
    Apply context-aware post-processing to DataFrame.
    
    Args:
        df: DataFrame with predictions
        text_col: Column name for text
        label_col: Column name for predicted label
        intensity_col: Column name for intensity
        context_col: Column name for sequence (contains context)
        
    Returns:
        DataFrame with corrected predictions and review flags
    """
    processor = ContextAwarePostProcessor()
    
    corrected_labels = []
    corrected_intensities = []
    override_flags = []
    override_types = []
    review_flags = []
    
    for _, row in df.iterrows():
        text = row[text_col]
        label = row[label_col]
        intensity = row[intensity_col]
        
        # Extract context from sequence if available
        context_text = None
        if context_col in df.columns:
            sequence = row[context_col]
            # Extract context part (after [SEP])
            if '[SEP]' in sequence:
                parts = sequence.split('[SEP]', 1)
                if len(parts) > 1:
                    context_text = parts[1].strip()
        
        # Get context emotions from history if available
        context_emotions = None
        # Could extract from history if we have it
        
        high_confidence = row.get('HighConfidenceFlag', False) if 'HighConfidenceFlag' in df.columns else False
        
        # Extract previous emotions from context if available
        previous_emotions = None
        if context_emotions:
            previous_emotions = context_emotions
        elif context_text:
            # Try to extract emotions from context text (if we have history)
            # This is a fallback if context_emotions is not provided
            pass
        
        corrected_label, corrected_intensity, was_overridden, needs_review, override_type = processor.post_process_with_context(
            text, label, intensity, context_text, context_emotions
        )
        
        # Also check high confidence flag and temporal patterns
        if not needs_review:
            needs_review = processor.flag_for_review(
                text, label, intensity,
                high_confidence_flag=high_confidence,
                previous_emotions=previous_emotions
            )
        
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

