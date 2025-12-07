"""
User pattern anomaly detection for clinical flagging.
Detects concerning patterns in user emotion data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class UserPatternAnalyzer:
    """Analyze user emotion patterns and detect anomalies."""
    
    # Concerning emotion thresholds
    HIGH_FEAR_THRESHOLD = 0.40  # 40% of records classified as fear
    HIGH_SADNESS_THRESHOLD = 0.50  # 50% of records classified as sadness
    HIGH_ANGER_THRESHOLD = 0.30  # 30% of records classified as anger
    
    # Intensity thresholds
    VERY_HIGH_INTENSITY_THRESHOLD = 0.95  # Average intensity > 0.95
    LOW_INTENSITY_THRESHOLD = 0.50  # Average intensity < 0.50 (may indicate disengagement)
    
    # Minimum records for pattern detection
    MIN_RECORDS_FOR_ANALYSIS = 5
    
    def __init__(self, 
                 fear_threshold: float = None,
                 sadness_threshold: float = None,
                 anger_threshold: float = None):
        """
        Initialize pattern analyzer.
        
        Args:
            fear_threshold: Custom threshold for fear percentage (default: 0.40)
            sadness_threshold: Custom threshold for sadness percentage (default: 0.50)
            anger_threshold: Custom threshold for anger percentage (default: 0.30)
        """
        self.fear_threshold = fear_threshold or self.HIGH_FEAR_THRESHOLD
        self.sadness_threshold = sadness_threshold or self.HIGH_SADNESS_THRESHOLD
        self.anger_threshold = anger_threshold or self.HIGH_ANGER_THRESHOLD
    
    def analyze_user(self, user_df: pd.DataFrame) -> Dict:
        """
        Analyze a single user's emotion patterns.
        
        Args:
            user_df: DataFrame with columns: UserID, PrimaryEmotionLabel, IntensityScore_Primary
            
        Returns:
            Dictionary with:
            - 'user_id': User identifier
            - 'total_records': Number of records
            - 'emotion_distribution': Dict of emotion -> count
            - 'avg_intensity': Average intensity score
            - 'anomalies': List of detected anomalies
            - 'risk_level': 'low', 'medium', 'high', or 'critical'
            - 'flag_for_review': Boolean indicating if user should be flagged
        """
        if len(user_df) < self.MIN_RECORDS_FOR_ANALYSIS:
            return {
                'user_id': user_df['UserID'].iloc[0] if 'UserID' in user_df.columns else 'unknown',
                'total_records': len(user_df),
                'emotion_distribution': {},
                'avg_intensity': 0.0,
                'anomalies': ['Insufficient data for analysis'],
                'risk_level': 'low',
                'flag_for_review': False
            }
        
        user_id = user_df['UserID'].iloc[0] if 'UserID' in user_df.columns else 'unknown'
        
        # Calculate emotion distribution
        if 'PrimaryEmotionLabel' in user_df.columns:
            emotion_counts = user_df['PrimaryEmotionLabel'].value_counts()
            emotion_distribution = emotion_counts.to_dict()
            total = len(user_df)
            emotion_percentages = {emotion: count / total for emotion, count in emotion_distribution.items()}
        else:
            emotion_distribution = {}
            emotion_percentages = {}
        
        # Calculate average intensity
        if 'IntensityScore_Primary' in user_df.columns:
            avg_intensity = user_df['IntensityScore_Primary'].mean()
        else:
            avg_intensity = 0.0
        
        # Detect anomalies
        anomalies = []
        risk_level = 'low'
        
        # Check for high fear percentage
        fear_pct = emotion_percentages.get('fear', 0.0)
        if fear_pct >= self.fear_threshold:
            anomalies.append(f"High fear frequency: {fear_pct:.1%} of records")
            risk_level = 'high' if risk_level == 'low' else 'critical'
        
        # Check for high sadness percentage
        sadness_pct = emotion_percentages.get('sadness', 0.0)
        if sadness_pct >= self.sadness_threshold:
            anomalies.append(f"High sadness frequency: {sadness_pct:.1%} of records")
            risk_level = 'high' if risk_level == 'low' else 'critical'
        
        # Check for high anger percentage
        anger_pct = emotion_percentages.get('anger', 0.0)
        if anger_pct >= self.anger_threshold:
            anomalies.append(f"High anger frequency: {anger_pct:.1%} of records")
            risk_level = 'medium' if risk_level == 'low' else 'high'
        
        # Check for very high intensity (may indicate distress)
        if avg_intensity >= self.VERY_HIGH_INTENSITY_THRESHOLD:
            anomalies.append(f"Very high average intensity: {avg_intensity:.3f}")
            if risk_level == 'low':
                risk_level = 'medium'
        
        # Check for low intensity (may indicate disengagement)
        if avg_intensity <= self.LOW_INTENSITY_THRESHOLD:
            anomalies.append(f"Low average intensity: {avg_intensity:.3f} (possible disengagement)")
        
        # Check for lack of positive emotions
        positive_emotions = ['joy', 'surprise', 'excitement', 'calm']
        positive_pct = sum(emotion_percentages.get(emotion, 0.0) for emotion in positive_emotions)
        if positive_pct < 0.10 and total >= 10:  # Less than 10% positive emotions
            anomalies.append(f"Low positive emotion frequency: {positive_pct:.1%}")
            if risk_level == 'low':
                risk_level = 'medium'
        
        # Determine if should flag for review
        # Less aggressive: require critical risk OR 3+ anomalies (instead of 2)
        # Also check intensity: don't flag if average intensity is low (<0.7)
        # Low intensity suggests low-confidence predictions, which may be less reliable
        intensity_threshold = 0.7
        has_sufficient_intensity = avg_intensity >= intensity_threshold
        
        # Flag if:
        # 1. Critical risk level (regardless of intensity or anomalies)
        # 2. High risk + 3+ anomalies + sufficient intensity
        # 3. 3+ anomalies + sufficient intensity (even if risk is medium)
        flag_for_review = (
            risk_level == 'critical' or
            (risk_level == 'high' and len(anomalies) >= 3 and has_sufficient_intensity) or
            (len(anomalies) >= 3 and has_sufficient_intensity)
        )
        
        return {
            'user_id': user_id,
            'total_records': total,
            'emotion_distribution': emotion_distribution,
            'emotion_percentages': emotion_percentages,
            'avg_intensity': avg_intensity,
            'anomalies': anomalies,
            'risk_level': risk_level,
            'flag_for_review': flag_for_review
        }
    
    def analyze_all_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze patterns for all users in a DataFrame.
        
        Args:
            df: DataFrame with columns: UserID, PrimaryEmotionLabel, IntensityScore_Primary
            
        Returns:
            DataFrame with analysis results for each user
        """
        if 'UserID' not in df.columns:
            logger.warning("No UserID column found, cannot analyze user patterns")
            return pd.DataFrame()
        
        results = []
        for user_id in df['UserID'].unique():
            user_df = df[df['UserID'] == user_id]
            analysis = self.analyze_user(user_df)
            results.append(analysis)
        
        return pd.DataFrame(results)
    
    def flag_high_risk_users(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of user IDs that should be flagged for review.
        
        Args:
            df: DataFrame with user emotion data
            
        Returns:
            List of user IDs flagged for review
        """
        analysis_df = self.analyze_all_users(df)
        flagged = analysis_df[analysis_df['flag_for_review'] == True]
        return flagged['user_id'].tolist()


def detect_user_anomalies(df: pd.DataFrame, 
                          user_id: Optional[str] = None) -> Dict:
    """
    Convenience function to detect anomalies for a user or all users.
    
    Args:
        df: DataFrame with emotion data
        user_id: Optional specific user ID to analyze
        
    Returns:
        Analysis results dictionary or DataFrame
    """
    analyzer = UserPatternAnalyzer()
    
    if user_id:
        user_df = df[df['UserID'] == user_id]
        return analyzer.analyze_user(user_df)
    else:
        return analyzer.analyze_all_users(df)

