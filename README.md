# Emotix: Mental Well-Being Tracking Pipeline

A multi-layer NLP pipeline for Mental Well-Being tracking from raw conversational text. The pipeline processes noisy conversational data through ingestion, preprocessing, contextualization, emotion modeling, and persistence layers.

## Architecture Overview

The pipeline consists of 5 layers:

1. **Layer 1: Ingestion** - Accepts raw CSV/JSON, validates, and converts to canonical `pd.DataFrame` format
2. **Layer 2: Preprocessing** - Normalizes slang/abbreviations, demojizes, applies syntactic fixes, preserves stopwords
3. **Layer 3: Contextualization** - Multi-turn retrieval and sequence formatting for context-aware inference
4. **Layer 4: Modeling** - Emotion classification and intensity prediction using transformer models
5. **Layer 5: Persistence** - Two-tier storage (raw archive + structured SQLite MWB Log)

### Data Flow

```
Raw Text (CSV/JSON) 
  → [Ingestion] → pd.DataFrame (UserID, Text, Timestamp)
  → [Preprocessing] → pd.DataFrame (+ NormalizedText, NormalizationFlags)
  → [Contextualization] → pd.DataFrame (+ Sequence with context)
  → [Modeling] → pd.DataFrame (+ PrimaryEmotionLabel, IntensityScore_Primary, AmbiguityFlag)
  → [Persistence] → SQLite (mwb_log + raw_archive tables)
```

## Layer-by-Layer Architecture

### Layer 1: Ingestion

**Purpose**: Accepts raw data from multiple formats and converts to a standardized DataFrame format.

**Key Features**:
- **Multi-format Support**: Handles CSV, JSON, and JSON Lines formats
- **Flexible Column Mapping**: Automatically detects and maps common column name variations (e.g., `user_id`, `UserID`, `user` → `UserID`)
- **Data Validation**: 
  - Ensures required columns exist: `UserID`, `Text`, `Timestamp`
  - Validates no null values in required fields
  - Converts timestamps to datetime format (handles ISO 8601, SQL format, Unix timestamps)
  - Type coercion for UserID and Text columns
- **Error Handling**: Provides clear error messages for missing columns or invalid data

**Output**: Standardized DataFrame with columns: `UserID` (string), `Text` (string), `Timestamp` (datetime)

**Example**:
```python
from src.ingest import ingest_csv

df = ingest_csv("data/sample_data.csv")
# Returns DataFrame with UserID, Text, Timestamp columns
```

### Layer 2: Preprocessing

**Purpose**: Normalizes conversational text to improve model accuracy while preserving semantic meaning.

**Key Features**:
- **Slang Normalization**: 
  - Dictionary-based lookup for common abbreviations (e.g., "lol" → "laughing out loud")
  - Fuzzy matching fallback for typos and variations
  - Prevents false matches (e.g., "am" won't match "ama")
- **Emoji Demojization**: Converts emojis to text descriptions (e.g., 😊 → ":smiling_face:")
- **Syntactic Fixes**: Handles common typos and spacing issues
- **Stopword Preservation**: Keeps stopwords by default (configurable) to maintain context
- **Flag Tracking**: Records all transformations in `NormalizationFlags` JSON column for auditability

**Output**: DataFrame with added columns: `NormalizedText` (string), `NormalizationFlags` (JSON dict)

**Example**:
```python
from src.preprocess import Preprocessor

preprocessor = Preprocessor(slang_dict_path="data/slang_dictionary.json")
normalized, flags = preprocessor.preprocess_text("lol that's great 😊")
# Returns: ("laughing out loud that's great :smiling_face:", {...})
```

### Layer 3: Contextualization

**Purpose**: Retrieves conversation history and formats sequences for context-aware emotion classification.

**Key Features**:
- **History Retrieval**: Fast indexed queries (<100ms) to fetch previous user entries from database
- **Journal Isolation**: Context retrieval can be filtered by `JournalID` to maintain separate context per journal
- **Context Selection Strategies**:
  - **RecentContextStrategy**: Most recent N turns (default)
  - **SameDayContextStrategy**: Only same-day context (good for journaling)
  - **EmotionalContextStrategy**: Filters out neutral context
  - **WeightedContextStrategy**: Weights context by recency
- **Sequence Formatting**: Multiple format types:
  - `standard`: `"utterance [SEP] context"` (default)
  - `reverse`: `"context [SEP] utterance"`
  - `weighted`: `"utterance [SEP] utterance context"` (repeats utterance for emphasis)
  - `concatenated`: `"utterance context"` (no separator)
- **Batch Processing**: Efficient batch fetching to avoid N+1 query problems
- **Temporal Filtering**: Only includes history before current timestamp to prevent data leakage

**Output**: DataFrame with added column: `Sequence` (string) containing formatted context-aware input

**Example**:
```python
from src.contextualize import create_sequences_batch
from src.persistence import MWBPersistence

persistence = MWBPersistence("data/mwb_log.db")
# Create sequences with journal-specific context
df = create_sequences_batch(df, persistence, max_context_turns=5, journal_id=1)
# Adds 'Sequence' column with context-aware formatted text
# Context only includes entries from the specified journal
```

### Layer 4: Modeling

**Purpose**: Emotion classification and intensity prediction using transformer models.

**Key Features**:
- **Model**: Pre-trained `j-hartmann/emotion-english-distilroberta-base` (DistilRoBERTa)
- **Emotion Classes**: 7 emotions (joy, sadness, anger, fear, surprise, disgust, neutral)
- **Intensity Prediction**: Softmax probabilities provide intensity scores (0.0-1.0)
- **Confidence Calibration**: Temperature scaling (default: 1.5) to reduce overconfidence
- **Ambiguity Detection**: Identifies low-confidence predictions that may need review
- **Batch Inference**: Efficient batch processing for multiple texts
- **Context-Aware Input**: Uses sequences from Layer 3 that include conversation history

**Output**: DataFrame with added columns: `PrimaryEmotionLabel` (string), `IntensityScore_Primary` (float), `AmbiguityFlag` (boolean)

**Example**:
```python
from src.modeling import EmotionModel, run_inference_pipeline

model = EmotionModel(temperature=1.5)
df = run_inference_pipeline(df, model)
# Adds emotion labels and intensity scores
```

### Layer 5: Persistence

**Purpose**: Two-tier storage system for audit trail and queryable structured data.

**Key Features**:
- **Two-Tier Architecture**:
  - `raw_archive`: Immutable audit trail of original input (data lake pattern)
  - `mwb_log`: Structured, indexed results for fast queries
- **Multi-Journal Support**: 
  - `journals` table for journal metadata
  - `JournalID` column in both `mwb_log` and `raw_archive` for journal association
  - Composite indexes on `(UserID, JournalID, Timestamp)` for journal-specific queries
- **ACID Transactions**: Batch writes with rollback on errors
- **Fast Queries**: Composite indexes enable <100ms history retrieval (per journal or globally)
- **Schema Migration**: Automatic schema evolution without data loss
- **Comprehensive Schema**: Stores original predictions, post-processed results, flags, and metadata

**Output**: Data persisted to SQLite database with both raw archive and structured log tables

**Example**:
```python
from src.persistence import MWBPersistence

persistence = MWBPersistence("data/mwb_log.db")
persistence.write_results(df, archive_raw=True)
history = persistence.fetch_history("user001", limit=10)
```

## Context-Aware System Design

### Overview

The Emotix pipeline is designed to be **context-aware**, meaning it uses conversation history to improve emotion classification accuracy. This is critical for mental well-being tracking because emotions are often expressed ambiguously and require context to interpret correctly.

### How Context-Awareness Works

**1. History Retrieval (Layer 3)**
- When processing a new text entry, the system queries the database for that user's previous entries
- Uses indexed queries on `(UserID, Timestamp)` for fast retrieval (<100ms)
- Only includes history **before** the current timestamp to prevent data leakage
- Retrieves up to N previous turns (configurable, default: 5)

**2. Context Selection Strategies**

The system provides multiple strategies for selecting relevant context:

- **RecentContextStrategy** (Default): Selects the most recent N turns
  - Best for: Real-time conversations where recent context is most relevant
  - Example: "I'm fine" after "I'm feeling terrible" → sadness (not neutral)

- **SameDayContextStrategy**: Only includes context from the same day
  - Best for: Daily journaling where each day is independent
  - Example: Morning entry doesn't use context from previous day

- **EmotionalContextStrategy**: Filters out neutral/low-intensity entries
  - Best for: Focusing on emotional transitions
  - Example: Only uses entries with clear emotions (intensity ≥ 0.5)

- **WeightedContextStrategy**: Weights context by recency
  - Best for: Balancing recent and historical context
  - Example: Recent entries get higher weight than older ones

**3. Sequence Formatting**

The selected context is formatted into a sequence that the model can process:

- **Standard Format**: `"current utterance [SEP] context"`
  - Current utterance comes first (most prominent)
  - Context provides background information
  - Example: `"I'm fine [SEP] I'm feeling terrible today nothing is working"`

- **Reverse Format**: `"context [SEP] current utterance"`
  - Context comes first, then current utterance
  - Useful when context is more important than current text

- **Weighted Format**: `"current utterance [SEP] current utterance context"`
  - Repeats current utterance for emphasis
  - Ensures model focuses on current text while having context

**4. Model Processing**

The transformer model processes the context-aware sequence:
- The model sees both the current utterance and historical context
- This enables it to resolve ambiguities (e.g., "I'm fine" can mean different things based on context)
- Context helps distinguish between similar phrases with different emotional meanings

### Impact of Context-Awareness

**Accuracy Improvement**:
- Resolves ambiguous phrases: "I'm fine" → sadness (when context is negative) vs. neutral (when context is positive)
- Captures emotional transitions: Detects when emotions change over time
- Reduces false positives: Context helps distinguish sarcasm from genuine statements

**Example Scenarios**:

1. **Ambiguous "I'm fine"**:
   - Without context: Often classified as neutral (low confidence)
   - With context ("I'm feeling terrible" → "I'm fine"): Classified as sadness (higher confidence)

2. **Emotional Transitions**:
   - Context: "I'm so sad" → "Feeling better now"
   - Without context: "Feeling better now" → joy (0.72)
   - With context: "Feeling better now" → joy (0.85) - higher confidence due to positive transition

3. **Sarcasm Detection**:
   - Context: Multiple positive entries → "Oh great, another problem"
   - Without context: May be classified as joy
   - With context: Correctly flagged for review (sarcasm pattern)

### Technical Implementation

**Database Integration**:
- History is stored in SQLite with composite indexes for fast retrieval
- Batch fetching prevents N+1 query problems
- Temporal filtering ensures no future data leakage

**Sequence Construction**:
```python
# Example sequence creation
utterance = "I'm fine"
history = [
    "I'm feeling terrible today",
    "Nothing is working for me"
]
sequence = "I'm fine [SEP] I'm feeling terrible today Nothing is working for me"
```

**Model Input**:
- The sequence is tokenized and fed to the transformer model
- Model processes the entire sequence (utterance + context) together
- Output includes emotion label and intensity score

### Performance Considerations

- **Query Optimization**: Composite indexes enable <100ms history retrieval
- **Batch Processing**: Batch fetching reduces database round trips
- **Sequence Length**: Truncation at 512 tokens to fit model limits
- **Context Window**: Configurable (default: 5 turns) to balance accuracy and performance

This context-aware design significantly improves emotion classification accuracy, especially for ambiguous or context-dependent expressions common in mental well-being tracking.

## Multi-Journal Support

### Overview

The Emotix pipeline supports **multiple journals per user**, allowing users to maintain separate, isolated journals for different purposes (e.g., "Work Journal", "Personal Journal", "Therapy Journal"). Each journal maintains its own context history, ensuring that entries in one journal don't affect emotion classification in another.

### Key Features

- **Journal Isolation**: Context is isolated per journal - entries in one journal only use context from that same journal
- **Multiple Journals Per User**: Users can create and manage multiple named journals
- **Journal Management**: Create, list, switch between, and archive journals
- **Context-Aware Per Journal**: Each journal maintains its own conversation history for context-aware emotion classification
- **Soft Deletion**: Journals can be archived (soft-deleted) while preserving all entries

### How It Works

**Database Schema**:
- `journals` table stores journal metadata (JournalID, UserID, JournalName, IsArchived)
- `mwb_log` and `raw_archive` tables include `JournalID` column to associate entries with journals
- Composite indexes on `(UserID, JournalID, Timestamp)` enable fast journal-specific queries

**Context Isolation**:
- When processing an entry, the system only retrieves history from the same journal
- This ensures that "I'm fine" in a work journal doesn't use context from a personal journal
- Each journal maintains independent emotion tracking and patterns

**Journal Operations**:
- **Create**: Create a new journal with a unique name (case-insensitive)
- **List**: View all active journals with entry counts
- **Switch**: Change the active journal for new entries
- **Archive**: Soft-delete a journal (entries preserved, journal hidden)

### Usage

**CLI Usage with Journal Commands**:

```bash
python emotix_cli.py --user user001
```

Once in the CLI:
```
[user001@My Journal] > journal create
Enter journal name: Work Journal
✓ Created journal: Work Journal (ID: 1)

[user001@Work Journal] > I had a great meeting today
✓ Processed: I had a great meeting today
  Emotion: joy (intensity: 0.85)

[user001@Work Journal] > journal switch
📔 Available Journals:
  1. Work Journal (1 entries)
  2. Create new journal
  3. Cancel
Select journal (1-3): 2
Enter journal name: Personal Journal
✓ Created journal: Personal Journal (ID: 2)
✓ Switched to journal: Personal Journal

[user001@Personal Journal] > Feeling stressed about family
✓ Processed: Feeling stressed about family
  Emotion: fear (intensity: 0.72)
```

**Programmatic Usage**:

```python
from src.persistence import MWBPersistence

persistence = MWBPersistence("data/mwb_log.db")

# Create a new journal
journal_id = persistence.create_journal("user001", "Work Journal")
print(f"Created journal with ID: {journal_id}")

# List all journals for a user
journals_df = persistence.list_journals("user001")
print(journals_df)
# Output:
#    JournalID  JournalName  CreatedAt  IsArchived  EntryCount
# 0          1  Work Journal  2024-01-15           0          5
# 1          2  Personal Journal  2024-01-15           0          3

# Get journal ID by name
journal_id = persistence.get_journal_id("user001", "Work Journal")

# Fetch history for a specific journal
history = persistence.fetch_history("user001", journal_id=journal_id, limit=10)

# Get summary for a specific journal
summary = persistence.get_last_3_summary("user001", journal_id=journal_id)

# Write results to a specific journal
from src.pipeline import run_full_pipeline

df_results = run_full_pipeline(
    input_path="data/sample_data.csv",
    db_path="data/mwb_log.db",
    journal_id=journal_id  # Associate entries with specific journal
)
```

**Pipeline Integration**:

The journal feature is integrated throughout the pipeline:
- **Contextualization**: `create_sequences_batch()` accepts `journal_id` parameter to filter context
- **Persistence**: `write_results()` accepts `journal_id` to associate entries with journals
- **History Retrieval**: `fetch_history()` filters by journal when `journal_id` is provided
- **Statistics**: `get_user_stats()` can filter statistics by journal

### Benefits

1. **Privacy**: Separate journals allow users to keep different aspects of their life separate
2. **Organization**: Users can organize entries by topic, purpose, or time period
3. **Context Accuracy**: Journal isolation ensures context-aware classification uses relevant history
4. **Flexibility**: Users can create as many journals as needed and switch between them easily
5. **Data Preservation**: Archived journals preserve all entries for future analysis

### Example Use Cases

- **Work vs Personal**: Separate work-related stress from personal emotions
- **Therapy Journal**: Maintain a dedicated journal for therapy sessions
- **Daily Reflection**: Create a new journal each month for monthly reviews
- **Project-Specific**: Track emotions related to specific projects or life events
- **Family Journal**: Separate family-related entries from individual entries

The multi-journal feature enhances the pipeline's flexibility and makes it suitable for diverse mental well-being tracking scenarios.

## Suicidal Detection

### Overview

The pipeline includes a critical safety feature for detecting suicidal ideation patterns in user text. This feature operates as an additional layer (Step 4.5) after emotion modeling and before persistence, providing immediate crisis support resources when concerning patterns are detected.

### How It Works

The `SuicidalIdeationDetector` uses pattern-based detection with multiple confidence levels:

1. **Direct Patterns** (Confidence: 0.95)
   - Explicit statements: "I want to die", "I'm going to kill myself", "I need to end my life"
   - Suicide-related terms: "commit suicide", "take my life", "ending it all"
   - Future tense variations: "I'll kill myself", "I'm going to end it"

2. **Planning Patterns** (Confidence: 0.90)
   - Active planning language: "thinking about ending it", "planning to kill myself"
   - Method-specific indicators: "have pills", "have a gun", "everything is ready"
   - Time-specific indicators: "tonight", "today", "this week"

3. **Goodbye Patterns** (Confidence: 0.90)
   - Final messages: "goodbye forever", "this is my last message", "tell everyone I love them"
   - Farewell statements in context of crisis indicators

4. **Indirect Patterns** (Confidence: 0.80)
   - Worthlessness: "not worth living", "life isn't worth it", "I'm a burden"
   - Burden on others: "everyone would be better without me", "no one would miss me"
   - Pointlessness: "no point", "nothing matters", "no reason to live"
   - Self-hatred: "hate myself", "hate my life", "I'm terrible"

5. **Hopelessness Patterns** (Confidence: 0.75)
   - No hope: "hopeless", "nothing will change", "never get better"
   - Trapped feelings: "no way out", "trapped", "no escape"
   - Giving up: "giving up", "done trying", "can't fight anymore"

6. **Multiple Indicators** (Confidence: 0.70)
   - Safety net: Detects when multiple concerning keywords appear together
   - Catches combinations that might not match specific patterns

### False Positive Prevention

The detector includes safeguards to reduce false positives:
- Excludes common phrases: "I could kill for a pizza", "I'm dying to see that"
- Context-aware detection: Ambiguous phrases like "I'm done" only trigger with additional concerning context
- Pattern validation: Requires specific linguistic structures, not just keyword presence

### Integration & Impact

**Pipeline Integration:**
- Runs automatically after emotion modeling (Step 4.5)
- Analyzes original text (before normalization) to preserve semantic meaning
- Adds three columns to the DataFrame:
  - `SuicidalIdeationFlag`: Boolean indicating detection
  - `SuicidalIdeationConfidence`: Confidence score (0.0-1.0)
  - `SuicidalIdeationPattern`: Type of pattern detected

**Automatic Flagging:**
- Any detected suicidal ideation automatically sets `FlagForReview = True`
- Logs warnings with user ID and timestamp for audit trail
- Immediately displays crisis support resources to the user

**Crisis Support Resources:**
When detection occurs, the system immediately displays:
- 24/7 emergency contact information
- International crisis resources (IASP, Befrienders Worldwide)
- Crisis Text Line information
- Encouraging message emphasizing that help is available

**Impact:**
- **Immediate Response**: Provides crisis resources instantly when concerning patterns are detected
- **Early Intervention**: Catches both explicit and indirect expressions of suicidal ideation
- **Comprehensive Coverage**: Detects patterns across multiple linguistic styles and variations
- **Privacy-Preserving**: Operates locally without external API calls
- **Audit Trail**: Logs all detections with user context for follow-up care

**Clinical Considerations:**
- This is a screening tool, not a diagnostic tool
- All detections should be reviewed by mental health professionals
- High-confidence detections (≥0.90) require immediate human review
- The system prioritizes sensitivity (catching true cases) over specificity (avoiding false positives) for safety

### Usage Example

```python
from src.suicidal_detection import SuicidalIdeationDetector

detector = SuicidalIdeationDetector()
is_detected, confidence, pattern_type = detector.process_text(
    "I don't want to live anymore",
    user_id="user001",
    timestamp=datetime.now()
)

if is_detected:
    print(f"Detected: {pattern_type} (confidence: {confidence:.2f})")
    # Help message is automatically displayed
```

The detector is automatically integrated into the full pipeline and CLI, providing seamless crisis detection for all processed text entries.

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/vikranthnara/Emotix.git
cd Emotix
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install package in development mode:
```bash
pip install -e .
```

## Quick Start

### Run the Demo Notebooks

**Phase 2 Demo (Complete Pipeline):**
```bash
jupyter notebook notebooks/phase2_demo.ipynb
```

### Interactive CLI

**Run the interactive CLI:**
```bash
python emotix_cli.py --user user001
```

The CLI processes text entries through the full pipeline and displays emotion predictions with context summaries.

**Commands:**
- Type any text and press Enter to process it
- `summary` - Show summary of last 3 entries
- `history` - Show full history (last 10 entries)
- `journal create` - Create a new journal
- `journal list` - List all your journals
- `journal switch` - Switch to a different journal
- `journal remove` - Archive (remove) a journal
- `exit` or `quit` - Exit the program

### Programmatic Usage

**Complete Pipeline:**
```python
from src.pipeline import run_full_pipeline
from src.persistence import MWBPersistence

# Optional: Create a journal first
persistence = MWBPersistence("data/mwb_log.db")
journal_id = persistence.create_journal("user001", "My Journal")

# Run pipeline with journal support
df_results = run_full_pipeline(
    input_path="data/sample_data.csv",
    db_path="data/mwb_log.db",
    slang_dict_path="data/slang_dictionary.json",
    model_name="j-hartmann/emotion-english-distilroberta-base",
    temperature=1.5,
    journal_id=journal_id  # Optional: associate entries with journal
)
```

## Technical Decisions & Rationale

### Model Selection

**Selected Model:** `j-hartmann/emotion-english-distilroberta-base`

**Rationale:**
- **DistilRoBERTa Architecture**: 40% faster than RoBERTa-base while maintaining 97% of performance
- **Pre-trained for Emotion**: Fine-tuned specifically for emotion classification (not general sentiment)
- **Resource Efficiency**: Runs on CPU with acceptable latency (<500ms per inference)
- **Intensity Support**: Softmax probabilities provide natural intensity scores (0.0-1.0)
- **Hugging Face Integration**: Well-maintained, actively updated model

**Alternative Models Considered:**
- **RoBERTa-base**: More accurate but 40% slower and 2x memory usage
- **BERT-base**: Older architecture, less efficient, lower accuracy on emotion tasks
- **cardiffnlp/twitter-roberta-base-emotion**: Twitter-specific, may not generalize to journaling

### Architecture Decisions

**5-Layer Pipeline Design:**
- **Separation of Concerns**: Each layer has a single, well-defined responsibility
- **DataFrame Interchange**: Pandas DataFrames as canonical format enables easy debugging, checkpointing, and integration with data science tools
- **Two-Tier Persistence**: 
  - `raw_archive`: Immutable audit trail of original input
  - `mwb_log`: Structured, queryable results for analysis
- **Context-Aware Processing**: Multi-turn sequences enable understanding of conversational flow

**Confidence Calibration:**
- **Problem**: Initial model showed overconfidence (intensity >0.95 for many predictions)
- **Solution**: Temperature scaling (default: 1.5) to reduce overconfidence
- **Result**: More realistic confidence scores, better flagging of high-confidence predictions

**Post-Processing Strategy:**
- **Problem**: Model misclassified obvious positive emotions (gratitude → sadness, progress → neutral)
- **Solution**: Rule-based post-processing with override validation
- **Validation**: Overrides only applied when label actually changes (prevents false positives)
- **Impact**: Post-processing corrects ~25% of misclassifications in test data

### Library Choices

**Pandas**: Chosen for DataFrame operations due to:
- Standard data science tool with rich functionality
- Easy integration with SQLite and Parquet checkpoints
- Supports future migration to Bodo for parallel processing

**SQLite**: Chosen for persistence due to:
- Zero-configuration database suitable for Phase 1
- ACID-compliant transactions
- Fast indexed queries (<100ms requirement met)
- Easy migration path to PostgreSQL for production

**Transformers (Hugging Face)**: Chosen for modeling due to:
- Industry-standard library for transformer models
- Easy model loading and inference
- Active maintenance and community support
- Supports multiple model architectures

## Methodology

### Tools Used

1. **Emotion Classification Model**: `j-hartmann/emotion-english-distilroberta-base`
   - Pre-trained DistilRoBERTa model fine-tuned for emotion classification
   - Supports 7 emotions: joy, sadness, anger, fear, surprise, disgust, neutral
   - Intensity prediction via softmax probabilities

2. **Preprocessing Tools**:
   - `emoji` library: Converts emojis to text descriptions (demojization)
   - Custom slang dictionary: Normalizes abbreviations and slang terms
   - Fuzzy string matching: Handles typos and variations in slang

3. **Contextualization**:
   - SQLite database: Stores and retrieves conversation history
   - Composite indexing: Enables fast history retrieval (<100ms)
   - Multiple context strategies: Recent, SameDay, Emotional, Weighted

4. **Post-Processing**:
   - Rule-based pattern matching: Corrects common misclassifications
   - Context-aware validation: Validates overrides against conversation history
   - Anomaly detection: Identifies high-risk user patterns

5. **Crisis Detection**:
   - Suicidal ideation detection: Pattern-based detection with multiple confidence levels
   - Real-time crisis support: Immediate display of help resources when detected
   - False positive prevention: Context-aware validation to reduce false alarms

6. **Validation & Testing**:
   - pytest: Unit testing framework (104 tests across 9 test files)
   - Synthetic data generation: Creates diverse test cases for validation
   - Metrics tracking: Precision, recall, F1-score, intensity correlation

### Verification of Output

**Model Output Verification:**
- Verified emotion labels match model's supported classes (7 emotions)
- Validated intensity scores are in valid range (0.0-1.0)
- Tested ambiguity detection on known ambiguous phrases
- Compared predictions with and without context to measure improvement

**Preprocessing Verification:**
- Tested slang normalization on 30+ common abbreviations
- Verified emoji demojization preserves semantic meaning
- Validated that stopwords are preserved (configurable)
- Tested fuzzy matching prevents false positives (e.g., "am" not matching "ama")

**Post-Processing Verification:**
- Validated overrides only apply when label actually changes
- Tested pattern matching on positive keywords, gratitude, progress indicators
- Verified context-aware validation prevents inappropriate overrides
- Measured accuracy improvement (~25% correction rate on test data)

**Persistence Verification:**
- Validated transaction safety (rollback on errors)
- Tested history retrieval performance (<100ms requirement)
- Verified schema migration handles new columns gracefully
- Confirmed two-tier storage (raw_archive + mwb_log) maintains data integrity

**Testing Coverage:**
- 103 unit tests across all layers
- Integration tests for complete pipeline
- Performance tests for history retrieval
- Synthetic data validation with known ground truth

### Synthetic Dataset Generation

**AI-Assisted Dataset Creation:**
We leveraged AI tools (ChatGPT) to generate a comprehensive synthetic dataset for pipeline validation and evaluation. The synthetic data generation process involved:

- **Dataset Size**: 500 records across 10 users
- **Diversity**: Includes all 7 emotion classes (joy, sadness, anger, fear, surprise, disgust, neutral)
- **Test Cases**: Covers positive examples (gratitude, humor, progress), negative examples (high-intensity emotions), ambiguous cases, sarcasm, and context-dependent scenarios
- **Temporal Distribution**: Records span 7 days with realistic timestamps
- **Ground Truth**: Pattern-based mapping to expected emotions for evaluation

The synthetic dataset (`data/synthetic_data_large.csv`) was generated using `generate_synthetic_data.py` and validated against ground truth labels (`data/ground_truth_large.csv`) created via pattern matching rules. This dataset enables comprehensive evaluation of the pipeline's performance across diverse emotion classification scenarios.

### Performance Metrics

**Overall Performance (500-record synthetic dataset):**
- **Accuracy**: 0.712
- **Precision**: 0.740
- **Recall**: 0.712
- **F1-Score**: 0.718

**Per-Class Performance (Top 5 Emotions):**

| Emotion  | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| joy      | 0.863     | 0.760  | 0.808    | 183     |
| neutral  | 0.637     | 0.760  | 0.693    | 104     |
| sadness  | 0.683     | 0.589  | 0.633    | 95      |
| fear     | 0.828     | 0.716  | 0.768    | 74      |
| anger    | 0.864     | 0.950  | 0.905    | 20      |

**False Positive/Negative Rates (Top 5):**

| Emotion  | TP   | FP  | FN  | FPR    | FNR    |
|----------|------|-----|-----|--------|--------|
| joy      | 139  | 22  | 44  | 0.069  | 0.240  |
| neutral  | 79   | 45  | 25  | 0.114  | 0.240  |
| sadness  | 56   | 26  | 39  | 0.064  | 0.411  |
| fear     | 53   | 11  | 21  | 0.026  | 0.284  |
| anger    | 19   | 3   | 1   | 0.006  | 0.050  |

**Top Confusion Patterns:**
- sadness → surprise: 33 cases
- joy → neutral: 30 cases
- fear → sadness: 16 cases
- neutral → joy: 14 cases
- disgust → neutral: 12 cases

These metrics demonstrate the pipeline's effectiveness in emotion classification, with particularly strong performance on joy and anger detection. The confusion patterns highlight areas for improvement, particularly in distinguishing between sadness and surprise, and handling neutral emotions.

## AI Disclosure

This project was developed with the assistance of AI tools, primarily **Cursor IDE** (an AI-powered code editor), along with ChatGPT and GitHub Copilot. AI assistance was strategically leveraged throughout the development process.

### AI Usage

**1. Code Documentation and Comments**
- AI was used to generate comprehensive documentation and inline comments for each source file
- Every module includes detailed docstrings, function descriptions, and usage examples
- This ensures code maintainability and helps future developers understand the system architecture

**2. Synthetic Dataset Generation**
- Leveraged ChatGPT to generate diverse test cases and patterns for the 500-record synthetic dataset (`data/synthetic_data_large.csv`)
- AI helped create realistic conversational text covering all emotion classes, edge cases, and ambiguous scenarios
- The synthetic dataset includes positive examples (gratitude, humor, progress), negative examples (high-intensity emotions), neutral cases, sarcasm, and context-dependent scenarios

**3. Multi-Agent Development System**
- **Cursor IDE's multi-agent capabilities** were leveraged to create a specialized "team" of AI agents, each focusing on specific features:
  - **Ingestion Agent**: Specialized in Layer 1 (data ingestion, validation, format handling)
  - **Preprocessing Agent**: Focused on Layer 2 (slang normalization, emoji handling, text cleaning)
  - **Contextualization Agent**: Dedicated to Layer 3 (context retrieval, sequence formatting, strategy implementation)
  - **Modeling Agent**: Handled Layer 4 (emotion classification, intensity prediction, model integration)
  - **Persistence Agent**: Managed Layer 5 (database schema, query optimization, transaction safety)
  - **Post-Processing Agent**: Specialized in rule-based corrections and override validation
  - **Crisis Detection Agent**: Focused on suicidal ideation detection patterns and help resources
  - **Testing Agent**: Dedicated exclusively to unit tests, integration tests, and test coverage
- This multi-agent approach enabled parallel development and specialization, similar to having a team of workers where each agent became an expert in their domain
- The testing agent ensured comprehensive test coverage (104 tests across 9 test files) and validated all features independently

### Validation of AI Output

All AI-generated code, documentation, and synthetic data were rigorously validated:

**Code Validation**:
- **Manual Review**: All AI-generated code was reviewed line-by-line
- **Unit Testing**: The testing agent created comprehensive test suites that validated functionality
- **Integration Testing**: End-to-end pipeline tests ensured all layers work together correctly
- **Performance Testing**: History retrieval performance (<100ms) was verified through benchmarks
- **Edge Case Testing**: Synthetic data was used to test edge cases and error handling

**Synthetic Data Validation**:
- **Pattern Matching**: Created ground truth labels (`data/ground_truth_large.csv`) using pattern-based rules to validate synthetic data quality
- **Emotion Distribution**: Verified synthetic dataset covers all 7 emotion classes with realistic distributions
- **Pipeline Testing**: Ran synthetic data through the complete pipeline to ensure it produces valid results
- **Metrics Validation**: Compared pipeline performance on synthetic data against expected patterns

**Documentation Validation**:
- **Code-Documentation Alignment**: Verified all documentation matches the actual implementation
- **Example Testing**: All code examples in the documentation were tested to ensure they work correctly
- **Architecture Accuracy**: Confirmed technical descriptions accurately reflect the system design

## Project Structure

```
Emotix/
├── src/
│   ├── ingest.py          # Layer 1: Ingestion
│   ├── preprocess.py      # Layer 2: Preprocessing
│   ├── contextualize.py   # Layer 3: Contextualization
│   ├── modeling.py        # Layer 4: Emotion Modeling
│   ├── persistence.py     # Layer 5: Persistence
│   ├── postprocess.py     # Post-processing rules
│   ├── suicidal_detection.py  # Crisis detection and support
│   ├── pipeline.py        # Complete pipeline integration
│   └── utils.py           # Utilities
├── tests/                 # Unit tests
├── notebooks/             # Demo notebooks
├── data/                  # Sample data and slang dictionary
└── requirements.txt
```

## Key Features

- **Handles Messy Data**: Normalizes slang, emojis, typos, and inconsistent formatting
- **Context-Aware**: Uses conversation history to improve emotion classification
- **Multi-Journal Support**: Users can create multiple isolated journals (e.g., "Work", "Personal", "Therapy") with separate context histories
- **Rule-Based Corrections**: Post-processing corrects common misclassifications
- **Anomaly Detection**: Identifies high-risk user patterns automatically
- **Suicidal Ideation Detection**: Real-time detection of crisis indicators with immediate help resources
- **Fast Queries**: History retrieval under 100ms with indexed database
- **Two-Tier Storage**: Immutable raw archive + structured queryable log

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```
