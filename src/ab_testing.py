"""
A/B testing framework for context strategies and formats.
"""

import pandas as pd
from typing import Dict, List, Tuple
import logging

from src.context_strategies import (
    RecentContextStrategy, SameDayContextStrategy, 
    EmotionalContextStrategy, WeightedContextStrategy,
    create_sequence_with_strategy
)
from src.modeling import EmotionModel
from src.postprocess import EmotionPostProcessor

logger = logging.getLogger(__name__)


class ABTester:
    """A/B testing framework for context strategies."""
    
    def __init__(self, model: EmotionModel):
        """Initialize with model."""
        self.model = model
        self.processor = EmotionPostProcessor()
    
    def test_context_strategies(self,
                               test_cases: List[Dict],
                               strategies: Dict[str, any],
                               format_types: List[str] = ["standard"]) -> pd.DataFrame:
        """
        Test different context strategies and formats.
        
        Args:
            test_cases: List of dicts with 'text', 'expected', 'context' keys
            strategies: Dict of {name: strategy} pairs
            format_types: List of format types to test
            
        Returns:
            DataFrame with results for each strategy/format combination
        """
        results = []
        
        for case in test_cases:
            text = case['text']
            expected = case['expected']
            context_history = case.get('context', pd.DataFrame())
            current_timestamp = case.get('timestamp', pd.Timestamp.now())
            
            # Test each strategy
            for strategy_name, strategy in strategies.items():
                # Test each format
                for format_type in format_types:
                    # Create sequence
                    sequence = create_sequence_with_strategy(
                        text, context_history, strategy, current_timestamp,
                        max_context_turns=3, format_type=format_type
                    )
                    
                    # Predict
                    pred = self.model.predict_emotion(sequence)
                    
                    # Post-process
                    corrected_label, corrected_intensity, overridden, _ = self.processor.post_process(
                        text, pred['label'], pred['intensity'], original_text=None
                    )
                    
                    is_correct = corrected_label.lower() == expected.lower()
                    
                    results.append({
                        'Text': text,
                        'Expected': expected,
                        'Strategy': strategy_name,
                        'Format': format_type,
                        'OriginalLabel': pred['label'],
                        'OriginalIntensity': pred['intensity'],
                        'CorrectedLabel': corrected_label,
                        'CorrectedIntensity': corrected_intensity,
                        'WasOverridden': overridden,
                        'IsCorrect': is_correct,
                        'SequenceLength': len(sequence)
                    })
        
        return pd.DataFrame(results)
    
    def compare_strategies(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare strategy performance.
        
        Args:
            results_df: Results from test_context_strategies
            
        Returns:
            Summary DataFrame with accuracy by strategy/format
        """
        summary = results_df.groupby(['Strategy', 'Format']).agg({
            'IsCorrect': ['mean', 'count'],
            'WasOverridden': 'sum',
            'SequenceLength': 'mean'
        }).round(3)
        
        summary.columns = ['Accuracy', 'Count', 'Overrides', 'AvgSequenceLength']
        summary = summary.sort_values('Accuracy', ascending=False)
        
        return summary
    
    def test_format_types(self,
                         test_cases: List[Dict],
                         format_types: List[str]) -> pd.DataFrame:
        """
        Test different sequence formats.
        
        Args:
            test_cases: List of test cases
            format_types: List of format types
            
        Returns:
            Results DataFrame
        """
        strategy = RecentContextStrategy()
        strategies = {'recent': strategy}
        
        return self.test_context_strategies(test_cases, strategies, format_types)


def run_ab_tests(test_cases: List[Dict]) -> Dict:
    """
    Run comprehensive A/B tests.
    
    Args:
        test_cases: List of test cases with text, expected, context
        
    Returns:
        Dictionary with test results
    """
    try:
        model = EmotionModel()
    except ImportError:
        logger.warning("Model not available for A/B testing")
        return {}
    
    tester = ABTester(model)
    
    # Define strategies
    strategies = {
        'recent': RecentContextStrategy(),
        'same_day': SameDayContextStrategy(),
        'emotional': EmotionalContextStrategy(),
        'weighted': WeightedContextStrategy()
    }
    
    # Test formats
    format_types = ["standard", "reverse", "weighted", "concatenated"]
    
    # Run tests
    results = tester.test_context_strategies(test_cases, strategies, format_types)
    
    # Compare
    summary = tester.compare_strategies(results)
    
    return {
        'results': results,
        'summary': summary,
        'best_strategy': summary.index[0] if len(summary) > 0 else None
    }

