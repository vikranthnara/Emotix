"""
Layer 2: Preprocessing
Normalization pipeline: slang/abbrev dictionary lookup, fuzzy matching,
demojization, tokenization, preserve stopwords.
"""

import pandas as pd
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
from difflib import get_close_matches
import emoji

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Tokenization will use simple regex.")

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocessing pipeline for conversational text normalization."""
    
    def __init__(self, slang_dict_path: Optional[Union[str, Path]] = None):
        """
        Initialize preprocessor with slang dictionary.
        
        Args:
            slang_dict_path: Path to JSON file with slang/abbrev mappings
        """
        self.slang_dict = {}
        if slang_dict_path:
            self.load_slang_dict(slang_dict_path)
        
        # Initialize NLTK resources if available
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            self.stopwords_set = set(stopwords.words('english'))
        else:
            self.stopwords_set = set()
    
    def load_slang_dict(self, path: Union[str, Path]) -> None:
        """Load slang/abbreviation dictionary from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            self.slang_dict = json.load(f)
        logger.info(f"Loaded {len(self.slang_dict)} slang/abbrev mappings")
    
    def normalize_slang(self, text: str, fuzzy_threshold: float = 0.8) -> Tuple[str, Dict]:
        """
        Normalize slang and abbreviations using dictionary lookup and fuzzy matching.
        
        Args:
            text: Input text
            fuzzy_threshold: Minimum similarity for fuzzy matching (0-1)
            
        Returns:
            Tuple of (normalized_text, normalization_flags_dict)
        """
        flags = {
            'slang_replacements': [],
            'fuzzy_replacements': [],
            'unknown_slang': []
        }
        
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Remove punctuation for lookup (preserve original)
            word_clean = re.sub(r'[^\w]', '', word.lower())
            
            # Only match if it's an exact key match (prevent substring matching)
            # This prevents "am" from matching "ama" or "i'm" from matching "imo"
            if word_clean in self.slang_dict:
                # Additional check: ensure it's not a substring match
                # Only replace if the cleaned word exactly matches a dictionary key
                replacement = self.slang_dict[word_clean]
                normalized_words.append(replacement)
                flags['slang_replacements'].append({
                    'original': word,
                    'replacement': replacement
                })
            elif self.slang_dict:
                # Try fuzzy matching, but only if word lengths are similar
                # This prevents "am" from matching "ama" or "i'm" from matching "imo"
                matches = get_close_matches(
                    word_clean,
                    self.slang_dict.keys(),
                    n=1,
                    cutoff=fuzzy_threshold
                )
                if matches:
                    matched_key = matches[0]
                    # Only use fuzzy match if lengths are similar (within 1 character for short words, 2 for longer)
                    # This prevents "am" (2 chars) from matching "ama" (3 chars) or "i'm" from matching "imo"
                    length_diff = abs(len(word_clean) - len(matched_key))
                    # For words 3 chars or less, require exact length match or difference of 1
                    # For longer words, allow difference of 2
                    max_diff = 1 if len(word_clean) <= 3 else 2
                    if length_diff <= max_diff and len(word_clean) >= len(matched_key) * 0.7:
                        replacement = self.slang_dict[matched_key]
                        normalized_words.append(replacement)
                        flags['fuzzy_replacements'].append({
                            'original': word,
                            'matched': matched_key,
                            'replacement': replacement
                        })
                    else:
                        normalized_words.append(word)
                else:
                    normalized_words.append(word)
                    # Check if word looks like slang (short, all caps, etc.)
                    if len(word_clean) <= 4 and word_clean.isalpha():
                        flags['unknown_slang'].append(word)
            else:
                normalized_words.append(word)
        
        normalized_text = ' '.join(normalized_words)
        return normalized_text, flags
    
    def demojize(self, text: str) -> Tuple[str, Dict]:
        """
        Convert emojis to text descriptions.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (demojized_text, flags_dict)
        """
        flags = {'emojis_found': []}
        
        # Find all emojis
        emoji_chars = [c for c in text if c in emoji.EMOJI_DATA]
        
        if emoji_chars:
            # Replace emojis with their descriptions
            demojized = emoji.demojize(text, delimiters=(" ", " "))
            flags['emojis_found'] = list(set(emoji_chars))
        else:
            demojized = text
        
        return demojized, flags
    
    def minimal_syntactic_fix(self, text: str) -> Tuple[str, Dict]:
        """
        Apply minimal syntactic fixes (common typos, spacing).
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (fixed_text, flags_dict)
        """
        flags = {'fixes_applied': []}
        fixed = text
        
        # Fix common spacing issues
        fixed = re.sub(r'\s+', ' ', fixed)  # Multiple spaces to single
        fixed = re.sub(r'\s+([.,!?;:])', r'\1', fixed)  # Space before punctuation
        fixed = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', fixed)  # Space after punctuation
        
        # Fix common typos (basic examples - can be expanded)
        typo_fixes = {
            r'\bu\b': 'you',
            r'\bur\b': 'your',
            r'\bur\b': "you're",
            r'\bthru\b': 'through',
            r'\btho\b': 'though',
        }
        
        for pattern, replacement in typo_fixes.items():
            if re.search(pattern, fixed, re.IGNORECASE):
                fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)
                flags['fixes_applied'].append(f"{pattern} -> {replacement}")
        
        return fixed.strip(), flags
    
    def tokenize(self, text: str, preserve_stopwords: bool = True) -> List[str]:
        """
        Tokenize text using NLTK or simple regex.
        
        Args:
            text: Input text
            preserve_stopwords: If True, keep stopwords in output
            
        Returns:
            List of tokens
        """
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text)
        else:
            # Simple regex tokenization
            tokens = re.findall(r'\b\w+\b', text)
        
        if not preserve_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stopwords_set]
        
        return tokens
    
    def preprocess_text(self, text: str, fuzzy_threshold: float = 0.8) -> Tuple[str, Dict]:
        """
        Full preprocessing pipeline for a single text.
        
        Args:
            text: Raw input text
            fuzzy_threshold: Fuzzy matching threshold
            
        Returns:
            Tuple of (normalized_text, normalization_flags_dict)
        """
        flags = {}
        current_text = text
        
        # Step 1: Demojization
        current_text, emoji_flags = self.demojize(current_text)
        flags.update(emoji_flags)
        
        # Step 2: Slang normalization
        current_text, slang_flags = self.normalize_slang(current_text, fuzzy_threshold)
        flags.update(slang_flags)
        
        # Step 3: Minimal syntactic fixes
        current_text, fix_flags = self.minimal_syntactic_fix(current_text)
        flags.update(fix_flags)
        
        return current_text, flags
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                            text_col: str = 'Text',
                            fuzzy_threshold: float = 0.8) -> pd.DataFrame:
        """
        Preprocess entire DataFrame.
        
        Args:
            df: Input DataFrame with text column
            text_col: Name of text column
            fuzzy_threshold: Fuzzy matching threshold
            
        Returns:
            DataFrame with added columns: NormalizedText, NormalizationFlags
        """
        logger.info(f"Preprocessing {len(df)} rows")
        
        results = df.apply(
            lambda row: self.preprocess_text(row[text_col], fuzzy_threshold),
            axis=1
        )
        
        df['NormalizedText'] = [r[0] for r in results]
        df['NormalizationFlags'] = [r[1] for r in results]
        
        logger.info("Preprocessing complete")
        return df


def preprocess_pipeline(df: pd.DataFrame,
                       slang_dict_path: Optional[Union[str, Path]] = None,
                       text_col: str = 'Text',
                       fuzzy_threshold: float = 0.8) -> pd.DataFrame:
    """
    Convenience function to run preprocessing pipeline on DataFrame.
    
    Args:
        df: Input DataFrame
        slang_dict_path: Path to slang dictionary JSON
        text_col: Name of text column
        fuzzy_threshold: Fuzzy matching threshold
        
    Returns:
        Preprocessed DataFrame with NormalizedText and NormalizationFlags
    """
    preprocessor = Preprocessor(slang_dict_path)
    return preprocessor.preprocess_dataframe(df, text_col, fuzzy_threshold)

