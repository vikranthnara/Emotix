"""
Suicidal ideation detection for crisis support.
Detects patterns indicating suicidal thoughts and provides immediate help resources.
"""

import re
import logging
from typing import Optional, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)


class SuicidalIdeationDetector:
    """Detect suicidal ideation patterns in text and provide crisis support resources."""
    
    # Direct suicidal statements (highest priority - catch all variations)
    DIRECT_PATTERNS = [
        # Abbreviations first (before full phrases)
        r'\b(kms|k\s*m\s*s)\b',  # "kms" = "kill myself"
        # Want/wish/hope to die variations (including common typos)
        # Note: "kill my self" (with space) is handled by making "myself" optional space
        r'\b(want|wanna|wana|wan|wishing|hoping|need|should|have\s+to|must)\s+(to\s+)?(die|kill\s+my\s*self|kill\s+myself|end\s+it|end\s+my\s+life|end\s+everything|end\s+it\s+all|off\s+myself|offing\s+myself)\b',
        # Going to/planning to variations
        r'\b(going\s+to|planning\s+to|gonna|will|\'ll|am\s+going\s+to|\'m\s+going\s+to)\s+(kill\s+my\s*self|kill\s+myself|end\s+it|end\s+my\s+life|end\s+everything|die|commit\s+suicide|take\s+my\s+life|off\s+myself)\b',
        # Present continuous (in progress)
        r'\b(am|\'m|is|are)\s+(killing\s+myself|ending\s+it|ending\s+my\s+life|ending\s+everything|committing\s+suicide|taking\s+my\s+life|offing\s+myself)\b',
        # Direct statements without helper verbs
        r'\b(kill\s+my\s*self|kill\s+myself|end\s+it\s+all|end\s+everything|end\s+my\s+life|off\s+myself|offing\s+myself)\b',
        # Suicide-related terms
        r'\b(commit\s+suicide|take\s+my\s+life|end\s+my\s+life|suicide|suicidal|ending\s+it|ending\s+everything)\b',
        # Not wanting to live
        r'\b(not\s+wanting\s+to\s+live|don\'t\s+want\s+to\s+live|can\'t\s+go\s+on|won\'t\s+go\s+on)\b',
        # Should/need/have to die
        r'\b(should|need|have\s+to|must|gotta)\s+(just\s+)?(die|kill\s+myself|end\s+it|end\s+my\s+life)\b',
        # Future tense variations
        r'\b(i\'ll|i\s+will|i\'m\s+gonna|im\s+gonna)\s+(kill\s+my\s*self|kill\s+myself|end\s+it|end\s+my\s+life|die|commit\s+suicide)\b',
    ]
    
    # Indirect suicidal statements (expanded for comprehensive coverage)
    INDIRECT_PATTERNS = [
        # Worthlessness patterns
        r'\b(not\s+worth\s+living|not\s+worth\s+it|life\s+isn\'t\s+worth|not\s+worth\s+anything|worthless|useless)\b',
        # Burden on others
        r'\b(better\s+off\s+without\s+me|everyone\s+would\s+be\s+better|world\s+better\s+without|(i\'m|i\s+am|im)\s+(a\s+)?burden|everyone\s+hates\s+me|no\s+one\s+would\s+miss\s+me)\b',
        # Pointlessness
        r'\b(no\s+point|nothing\s+matters|doesn\'t\s+matter|what\'s\s+the\s+point|what\s+is\s+the\s+point|no\s+reason\s+to\s+live)\b',
        # Can't continue
        r'\b(can\'t\s+go\s+on|can\'t\s+do\s+this|can\'t\s+handle\s+it\s+anymore|can\'t\s+take\s+this|can\'t\s+take\s+it\s+anymore|can\'t\s+deal\s+with\s+this)\b',
        # Wish to be dead/gone
        r'\b(wish\s+I\s+was\s+dead|wish\s+I\s+was\s+gone|wish\s+I\s+didn\'t\s+exist|wish\s+I\s+was\s+never\s+born|wish\s+to\s+disappear|want\s+to\s+disappear)\b',
        # Shouldn't exist / should disappear
        r'\b(shouldn\'t\s+be\s+here|shouldn\'t\s+exist|don\'t\s+deserve\s+to\s+live|don\'t\s+deserve\s+to\s+be\s+here|should\s+(just\s+)?disappear|need\s+to\s+disappear|want\s+to\s+disappear)\b',
        # Don't want to be here (with and without apostrophe)
        r'\b((don\'t|dont)\s+want\s+to\s+be\s+here|don\'t\s+want\s+to\s+exist|don\'t\s+belong\s+here|don\'t\s+belong\s+anywhere|dont\s+belong\s+here|dont\s+belong\s+anywhere)\b',
        # Better off dead
        r'\b(better\s+off\s+dead|would\s+be\s+better\s+if\s+I\s+was\s+dead|better\s+if\s+I\s+was\s+gone)\b',
        # Self-hatred
        r'\b(hate\s+myself|hate\s+my\s+life|hate\s+everything|hate\s+it\s+all|i\'m\s+terrible|i\'m\s+awful)\b',
        # Done/finished (only in suicidal context - require additional indicators)
        # "i'm done" alone is too ambiguous, need context like "done with life", "done with everything"
        r'\b((i\'m|i\s+am|im)\s+(done\s+with\s+(life|everything|it\s+all)|finished\s+with\s+(life|everything|it\s+all)|over\s+it|over\s+this|through|through\s+with\s+it))\b',
    ]
    
    # Hopelessness patterns (expanded)
    HOPELESSNESS_PATTERNS = [
        # No hope
        r'\b(no\s+hope|hopeless|nothing\s+will\s+change|never\s+get\s+better|no\s+hope\s+left|all\s+hope\s+is\s+gone)\b',
        # Never improving
        r'\b(always\s+be\s+like\s+this|never\s+going\s+to\s+improve|stuck\s+forever|never\s+going\s+to\s+change|always\s+going\s+to\s+be\s+this\s+way)\b',
        # Trapped
        r'\b(no\s+way\s+out|trapped|no\s+escape|no\s+options|no\s+way\s+forward|stuck|no\s+exit)\b',
        # Giving up
        r'\b(giving\s+up|done\s+trying|can\'t\s+fight\s+anymore|gave\s+up|i\'ve\s+given\s+up|i\'m\s+giving\s+up)\b',
        # Nothing will help
        r'\b(nothing\s+will\s+help|nothing\s+can\s+help|nothing\s+helps|nothing\s+works|nothing\s+matters)\b',
        # Always be this way
        r'\b(always\s+like\s+this|always\s+going\s+to\s+be\s+this\s+way|never\s+changes|never\s+going\s+to\s+change)\b',
    ]
    
    # Planning language (expanded)
    PLANNING_PATTERNS = [
        # Thinking about it
        r'\b(thinking\s+about\s+(ending|killing|dying|suicide)|thoughts\s+of\s+(ending|killing|dying|suicide)|thought\s+about\s+(ending|killing|dying|suicide))\b',
        # Planning
        r'\b(planning\s+(to\s+)?(end|kill|die|suicide)|made\s+plans|have\s+a\s+plan|got\s+a\s+plan|my\s+plan)\b',
        # Decided
        r'\b(decided\s+to|made\s+up\s+my\s+mind|going\s+through\s+with|i\'ve\s+decided|i\'m\s+going\s+through\s+with)\b',
        # Tonight/today/soon
        r'\b(tonight|today|soon|this\s+week|this\s+weekend)\s+(i\'m|i\s+will|i\'ll|going\s+to)\s+(kill|end|die|do\s+it)\b',
        # Method-specific (high risk indicators)
        r'\b(have\s+pills|have\s+a\s+gun|have\s+the\s+means|got\s+everything\s+ready|everything\s+is\s+ready)\b',
    ]
    
    # Goodbye/final messages (expanded)
    GOODBYE_PATTERNS = [
        # Goodbye variations
        r'\b(goodbye\s+forever|see\s+you\s+never|this\s+is\s+goodbye|final\s+goodbye|goodbye\s+for\s+good|goodbye\s+everyone)\b',
        # Last message
        r'\b(last\s+message|final\s+message|this\s+is\s+it|my\s+last|this\s+is\s+my\s+last|my\s+final)\b',
        # Telling others
        r'\b(tell\s+(everyone|them|family|people)\s+I\s+(love|said|thanks|thank\s+you)|say\s+goodbye\s+to|tell\s+.*\s+I\s+love)\b',
        # Final words
        r'\b(final\s+words|last\s+words|my\s+final\s+words|this\s+is\s+goodbye|farewell)\b',
        # I love you (in context of goodbye)
        r'\b(i\s+love\s+you\s+all|love\s+you\s+guys|love\s+you\s+all)\s+(goodbye|this\s+is\s+it|this\s+is\s+goodbye)\b',
    ]
    
    # False positive exclusions (common phrases that might trigger false positives)
    FALSE_POSITIVE_PATTERNS = [
        r'\b(kill\s+for\s+(a|some|that))\b',  # "I could kill for a pizza"
        r'\b(die\s+for\s+(a|some|that))\b',  # "I would die for that"
        r'\b(dying\s+to\s+(see|try|go))\b',  # "I'm dying to see that"
        r'\b(kill\s+me\s+(now|with))\b',  # "Kill me now" (exasperation)
        r'\b(just\s+kill\s+me)\b',  # "Just kill me" (exasperation, but could be real - keep for review)
    ]
    
    # Negation patterns (positive statements indicating NOT having suicidal thoughts)
    NEGATION_PATTERNS = [
        r'\b(not|isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|can\'t|couldn\'t)\s+(on\s+my\s+mind|thinking\s+about|going\s+to|planning\s+to|want\s+to|wanna|gonna)',
        r'\b(no\s+longer|not\s+anymore|anymore|not\s+thinking|not\s+going|not\s+planning|not\s+wanting)',
        r'\b(suicide|kill\s+myself|kill\s+my\s+self|kms|end\s+my\s+life|die)\s+(isn\'t|aren\'t|wasn\'t|weren\'t|not|no\s+longer|anymore)',
        r'\b(isn\'t|aren\'t|wasn\'t|weren\'t|not|no\s+longer|anymore)\s+(on\s+my\s+mind|thinking\s+about|going\s+to|planning\s+to)\s+(suicide|kill|die|end)',
        r'\b(over|past|behind|recovered|recovering|better|improved|healing|healed)\s+(suicide|suicidal|thoughts|thinking)',
        r'\b(suicide|suicidal|thoughts|thinking)\s+(is|are|was|were)\s+(over|past|behind|gone|done|finished)',
    ]
    
    def __init__(self, case_sensitive: bool = False):
        """
        Initialize detector.
        
        Args:
            case_sensitive: Whether pattern matching should be case-sensitive (default: False)
        """
        self.case_sensitive = case_sensitive
        self.flags = re.IGNORECASE if not case_sensitive else 0
    
    def detect(self, text: str) -> Tuple[bool, float, Optional[str]]:
        """
        Detect suicidal ideation in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (is_detected, confidence_score, pattern_type)
            - is_detected: True if suicidal ideation detected
            - confidence_score: Confidence level (0.0-1.0)
            - pattern_type: Type of pattern detected ('direct', 'indirect', 'hopelessness', 'planning', 'goodbye')
        """
        if not text or not text.strip():
            return False, 0.0, None
        
        text_lower = text.lower()
        
        # Check for negation patterns FIRST (positive statements about NOT having suicidal thoughts)
        # These should completely exclude detection
        for pattern in self.NEGATION_PATTERNS:
            if re.search(pattern, text_lower, self.flags):
                # This is a positive statement (e.g., "suicide isn't on my mind anymore")
                # Do NOT flag as suicidal ideation
                logger.debug(f"Negation pattern detected, excluding from suicidal ideation detection: {text}")
                return False, 0.0, None
        
        # Check for false positives
        for pattern in self.FALSE_POSITIVE_PATTERNS:
            if re.search(pattern, text_lower, self.flags):
                # False positive detected - still flag for review but with lower confidence
                # Some phrases like "just kill me" could be real, so we don't exclude completely
                pass  # Continue checking other patterns
        
        # Check direct patterns (highest confidence - 0.95)
        # These are explicit statements of intent
        for pattern in self.DIRECT_PATTERNS:
            if re.search(pattern, text_lower, self.flags):
                return True, 0.95, 'direct'
        
        # Check planning patterns (high confidence - 0.90)
        # Indicates active planning, which is high risk
        for pattern in self.PLANNING_PATTERNS:
            if re.search(pattern, text_lower, self.flags):
                return True, 0.90, 'planning'
        
        # Check goodbye patterns (high confidence - 0.90)
        # Final messages are strong indicators
        for pattern in self.GOODBYE_PATTERNS:
            if re.search(pattern, text_lower, self.flags):
                return True, 0.90, 'goodbye'
        
        # Check indirect patterns (medium-high confidence - 0.80)
        # Still concerning but less explicit
        for pattern in self.INDIRECT_PATTERNS:
            if re.search(pattern, text_lower, self.flags):
                return True, 0.80, 'indirect'
        
        # Check hopelessness patterns (medium confidence - 0.75)
        # Lower confidence but still important to catch
        for pattern in self.HOPELESSNESS_PATTERNS:
            if re.search(pattern, text_lower, self.flags):
                return True, 0.75, 'hopelessness'
        
        # Context-aware check for ambiguous phrases like "i'm done" or "i'm finished"
        # These only trigger if there are additional concerning indicators
        ambiguous_phrases = [
            r'\b((i\'m|i\s+am|im)\s+(done|finished))\b',
        ]
        for pattern in ambiguous_phrases:
            if re.search(pattern, text_lower, self.flags):
                # Check for additional concerning context
                concerning_context = [
                    'life', 'everything', 'it all', 'all of this', 'all this',
                    'hopeless', 'no hope', 'worthless', 'burden', 'hate',
                    'wish', 'die', 'death', 'kill', 'suicide', 'end'
                ]
                has_context = any(context in text_lower for context in concerning_context)
                if has_context:
                    # Has concerning context - flag it
                    return True, 0.75, 'ambiguous_with_context'
                # No concerning context - don't flag (too ambiguous)
                # Continue to other checks
        
        # Additional safety check: Look for multiple concerning keywords
        # Even if no single pattern matches, multiple indicators increase risk
        # This is a safety net to catch combinations that might not match specific patterns
        concerning_keywords = [
            'die', 'death', 'dead', 'kill', 'suicide', 'end', 'hopeless', 
            'worthless', 'burden', 'better off', 'no point', 'give up',
            'hate myself', 'hate my life', 'wish i was', 'shouldn\'t exist',
            'no hope', 'trapped', 'no way out', 'nothing matters'
        ]
        keyword_count = sum(1 for keyword in concerning_keywords if keyword in text_lower)
        if keyword_count >= 2:
            # Multiple concerning keywords detected - flag for review
            # Lower confidence but still important to catch
            return True, 0.70, 'multiple_indicators'
        
        # Additional safety: Check for very strong single indicators that might be missed
        # These are high-risk single words/phrases that should always trigger
        # BUT: Skip if they appear in a negated context (already checked above, but double-check)
        critical_single_indicators = [
            'suicidal', 'kill myself', 'kill my self', 'kms', 'end my life', 'take my life',
            'commit suicide', 'ending it all', 'ending everything'
        ]
        for indicator in critical_single_indicators:
            if indicator in text_lower:
                # Double-check: make sure it's not negated
                # Look for negation words near the indicator
                indicator_pos = text_lower.find(indicator)
                if indicator_pos >= 0:
                    # Check context around the indicator (20 chars before and after)
                    start = max(0, indicator_pos - 20)
                    end = min(len(text_lower), indicator_pos + len(indicator) + 20)
                    context = text_lower[start:end]
                    
                    # Check for negation in context
                    negation_words = ['not', "isn't", "aren't", "wasn't", "weren't", "don't", 
                                    "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
                                    "no longer", "not anymore", "anymore", "over", "past", "behind"]
                    has_negation = any(neg in context for neg in negation_words)
                    
                    if not has_negation:
                        return True, 0.85, 'critical_indicator'
        
        return False, 0.0, None
    
    def get_help_message(self) -> str:
        """
        Get crisis support help message with international resources.
        
        Returns:
            Formatted help message string
        """
        return """⚠️  CRISIS SUPPORT DETECTED ⚠️

If you or someone you know is experiencing a mental health crisis or 
having thoughts of suicide, please reach out for help immediately:

🚨 IMMEDIATE HELP (24/7):
• Your local emergency services: Call your country's emergency number
  (911 in US/Canada, 999 in UK, 112 in EU, etc.)

🌍 International Crisis Resources:
• International Association for Suicide Prevention (IASP)
  Website: https://www.iasp.info/resources/Crisis_Centres/
  Find crisis centers in your country
  
• Befrienders Worldwide
  Website: https://www.befrienders.org/
  Confidential emotional support worldwide
  
• Crisis Text Line (available in many countries)
  Text: HOME to 741741 (US/UK/Canada/Ireland)
  Available 24/7 via text message

💬 You are not alone. Professional help is available 24/7.
   Reaching out is a sign of strength, not weakness.

📞 If this is an emergency, please contact your local emergency 
   services immediately. Your life matters.

[This message was automatically triggered by content analysis. 
If you are in immediate danger, please call emergency services now.]"""

    def process_text(self, text: str, user_id: Optional[str] = None, 
                    timestamp: Optional[datetime] = None) -> Tuple[bool, float, Optional[str]]:
        """
        Process text and detect suicidal ideation, logging if detected.
        
        Args:
            text: Text to analyze
            user_id: Optional user ID for logging
            timestamp: Optional timestamp for logging
            
        Returns:
            Tuple of (is_detected, confidence_score, pattern_type)
        """
        is_detected, confidence, pattern_type = self.detect(text)
        
        if is_detected:
            user_info = f"User: {user_id}" if user_id else "Unknown user"
            time_info = f" at {timestamp}" if timestamp else ""
            logger.warning(
                f"Suicidal ideation detected ({pattern_type}, confidence: {confidence:.2f}) "
                f"for {user_info}{time_info}"
            )
            # Print help message immediately
            print("\n" + self.get_help_message() + "\n")
        
        return is_detected, confidence, pattern_type
