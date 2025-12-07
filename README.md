# Emotix: Mental Well-Being Tracking Pipeline

A multi-layer NLP pipeline for longitudinal Mental Well-Being (MWB) tracking from raw conversational text. Phase 1 implements Ingestion, Preprocessing, and Persistence layers with Pandas DataFrames as canonical interchange format.

## Architecture Overview

The pipeline consists of 5 layers:

1. **Layer 1: Ingestion** - Accepts raw CSV/JSON lines, validates, and converts to canonical `pd.DataFrame` format
2. **Layer 2: Preprocessing** - Normalizes slang/abbreviations, demojizes, applies minimal syntactic fixes, preserves stopwords
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

## Project Structure

```
Emotix/
├── src/
│   ├── __init__.py
│   ├── ingest.py          # Layer 1: Ingestion
│   ├── preprocess.py      # Layer 2: Preprocessing
│   ├── contextualize.py   # Layer 3: Contextualization
│   ├── modeling.py        # Layer 4: Emotion Modeling
│   ├── persistence.py     # Layer 5: Persistence
│   ├── validation.py      # Validation and metrics
│   ├── pipeline.py        # Complete pipeline integration
│   └── utils.py           # Logging and checkpointing utilities
├── tests/
│   ├── test_ingest.py
│   ├── test_preprocess.py
│   └── test_persistence.py
├── notebooks/
│   └── phase1_demo.ipynb  # End-to-end demo
├── data/
│   ├── slang_dictionary.json  # Slang/abbrev mappings
│   ├── sample_data.csv        # Sample noisy chat data
│   └── mwb_log.db             # SQLite database (created on first run)
├── checkpoints/               # Parquet checkpoints (created on run)
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
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

**Phase 1 Demo:**
```bash
jupyter notebook notebooks/phase1_demo.ipynb
```

**Phase 2 Demo (Complete Pipeline):**
```bash
jupyter notebook notebooks/phase2_demo.ipynb
```

The Phase 2 notebook demonstrates:
- Complete 5-layer pipeline
- Emotion classification and intensity prediction
- Contextualization with multi-turn sequences
- Ambiguity detection
- Synthetic validation dataset

### Run Unit Tests

```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

### Programmatic Usage

**Complete Pipeline (Recommended):**
```python
from src.pipeline import run_full_pipeline

# Run all 5 layers
df_results = run_full_pipeline(
    input_path="data/sample_data.csv",
    db_path="data/mwb_log.db",
    slang_dict_path="data/slang_dictionary.json",
    model_name="j-hartmann/emotion-english-distilroberta-base",
    temperature=1.5  # Confidence calibration (default: 1.5)
)
```

**Step-by-Step Usage:**
```python
from src.ingest import ingest_csv
from src.preprocess import preprocess_pipeline
from src.contextualize import create_sequences_batch
from src.modeling import EmotionModel, run_inference_pipeline
from src.persistence import MWBPersistence

# 1. Ingest
df = ingest_csv("data/sample_data.csv")

# 2. Preprocess
df = preprocess_pipeline(df, slang_dict_path="data/slang_dictionary.json")

# 3. Contextualize
persistence = MWBPersistence("data/mwb_log.db")
df = create_sequences_batch(df, persistence, max_context_turns=5)

# 4. Model
model = EmotionModel()
df = run_inference_pipeline(df, model)

# 5. Persist
persistence.write_results(df, archive_raw=True)
```

## Layer Details

### Layer 1: Ingestion (`src/ingest.py`)

**Functions:**
- `ingest_csv(file_path)` - Load from CSV
- `ingest_jsonl(file_path)` - Load from JSON Lines
- `ingest_json(file_path)` - Load from JSON array/object
- `ingest_from_dict(records)` - Load from list of dicts
- `validate_dataframe(df)` - Validate required columns and types

**Validation:**
- Required columns: `UserID`, `Text`, `Timestamp`
- Auto-converts timestamp to datetime
- Ensures no null values in required fields

### Layer 2: Preprocessing (`src/preprocess.py`)

**Features:**
- **Slang normalization**: Dictionary lookup + fuzzy matching fallback
- **Demojization**: Converts emojis to text descriptions
- **Syntactic fixes**: Common typos, spacing issues
- **Stopword preservation**: Keeps stopwords for context (configurable)
- **Flag tracking**: Records all transformations in `NormalizationFlags`

**Usage:**
```python
from src.preprocess import Preprocessor

preprocessor = Preprocessor(slang_dict_path="data/slang_dictionary.json")
normalized, flags = preprocessor.preprocess_text("lol that's great 😊")
# Returns: ("laughing out loud that's great :smiling_face:", {...})
```

### Layer 5: Persistence (`src/persistence.py`)

**SQLite Schema:**

**`mwb_log` table:**
- `LogID` (INTEGER PRIMARY KEY)
- `UserID` (TEXT)
- `Timestamp` (DATETIME)
- `NormalizedText` (TEXT)
- `PrimaryEmotionLabel` (TEXT, nullable) - Final emotion after post-processing
- `IntensityScore_Primary` (REAL, nullable) - Final intensity after post-processing
- `OriginalEmotionLabel` (TEXT, nullable) - Model prediction before post-processing
- `OriginalIntensityScore` (REAL, nullable) - Model intensity before post-processing
- `AmbiguityFlag` (INTEGER, default 0)
- `NormalizationFlags` (TEXT, JSON)
- `PostProcessingOverride` (TEXT, nullable) - Override type if label was changed
- `FlagForReview` (INTEGER, default 0) - Combined flag from all sources
- `PostProcessingReviewFlag` (INTEGER, default 0) - Flag from post-processing rules
- `AnomalyDetectionFlag` (INTEGER, default 0) - Flag from user pattern analysis
- `HighConfidenceFlag` (INTEGER, default 0) - Flag for intensity ≥0.95
- `CreatedAt` (DATETIME)

**`raw_archive` table:**
- `ArchiveID` (INTEGER PRIMARY KEY)
- `UserID` (TEXT)
- `Timestamp` (DATETIME)
- `RawText` (TEXT)
- `Metadata` (TEXT, JSON)
- `CreatedAt` (DATETIME)

**APIs:**
- `write_results(df, archive_raw=True, batch_size=100)` - Transaction-safe batch writes
- `fetch_history(user_id, since_timestamp=None, limit=None)` - Fast indexed retrieval (<100ms)

**Indexing:**
- Composite index on `(UserID, Timestamp)` for fast history queries
- Index on `Timestamp` for time-range queries

## Layer 3: Contextualization (`src/contextualize.py`)

**Functions:**
- `create_sequence_for_model(utterance, history, max_context_turns=3, strategy=None, format_type="standard")` - Create sequence with context
- `create_sequences_batch(df, persistence, max_context_turns=3)` - Batch sequence creation
- `extract_context_features(history)` - Extract contextual features

**Context Strategies** (`src/context_strategies.py`):
- `RecentContextStrategy()` - Most recent N turns (default)
- `SameDayContextStrategy()` - Only same-day context (good for journaling)
- `EmotionalContextStrategy()` - Filter out neutral context
- `WeightedContextStrategy()` - Weight by recency

**Sequence Formats:**
- `"standard"`: `"utterance [SEP] context"` (default)
- `"reverse"`: `"context [SEP] utterance"`
- `"weighted"`: `"utterance [SEP] utterance context"` (repeats utterance)
- `"concatenated"`: `"utterance context"` (no separator)

**Usage:**
```python
from src.contextualize import create_sequence_for_model
from src.context_strategies import SameDayContextStrategy

# Use same-day strategy for journaling
strategy = SameDayContextStrategy()
history = persistence.fetch_history("user001", limit=10)
sequence = create_sequence_for_model("I'm fine", history, strategy=strategy, format_type="standard")
```

## Layer 4: Modeling (`src/modeling.py`)

**Model:** `j-hartmann/emotion-english-distilroberta-base`
- Pre-trained DistilRoBERTa model for emotion classification
- Resource-efficient (faster than full RoBERTa)
- Supports emotion labels: joy, sadness, anger, fear, surprise, disgust, neutral
- Intensity prediction via softmax probabilities (0.0-1.0)

**Model Limitations:**
The base model may exhibit biases in emotion prediction:
- **Rare emotions**: Anger, disgust, and neutral are rarely predicted even though the model supports them
- **Class imbalance**: The model tends to favor sadness, fear, joy, and surprise over other emotions
- **Root cause**: This is a limitation of the pre-trained model, not a calibration issue

**Mitigation strategies:**
- Fine-tune the model on domain-specific data with class weights
- Use alternative models (e.g., `cardiffnlp/twitter-roberta-base-emotion`)
- Apply class weights during inference
- Consider ensemble methods combining multiple models

The pipeline automatically detects and logs these limitations during inference.

**Usage:**
```python
from src.modeling import EmotionModel

# Initialize with temperature scaling for confidence calibration
model = EmotionModel(temperature=1.5)  # Default: 1.5 (reduces overconfidence)
result = model.predict_emotion("I'm feeling great today!")
# Returns: {'label': 'joy', 'intensity': 0.95, ...}

# Batch prediction
results = model.predict_batch(["text1", "text2"], batch_size=32)

# Ambiguity detection
is_ambiguous, score = model.detect_ambiguity("I'm fine")

# Check emotion distribution and class imbalance
analysis = model.check_emotion_support(results)
print(analysis['warnings'])  # Shows missing emotions or imbalance
```

**Confidence Calibration:**
- Temperature scaling: Adjust `temperature` parameter (default: 1.5)
  - Higher values (>1.0): Reduce confidence (softer probabilities)
  - Lower values (<1.0): Increase confidence (sharper probabilities)
  - Default increased to 1.5 to reduce overconfidence
- Platt scaling and isotonic regression: Available via `CalibrationModel` class
  - Requires scikit-learn: `pip install scikit-learn`
  - See `CalibrationModel` docstring for example usage

**Alternative Models:**
The `EmotionModel` class can use any Hugging Face emotion classification model:
```python
model = EmotionModel(model_name="cardiffnlp/twitter-roberta-base-emotion", temperature=1.5)
```

**Post-Processing Rules:**
The pipeline includes rule-based post-processing to correct common misclassifications:
- **Positive keywords** → joy (overrides sadness/fear when confidence ≥0.75)
- **Gratitude patterns** → joy (overrides sadness)
- **Progress indicators** → joy (overrides sadness)
- **Humor patterns** → joy (overrides sadness)
- **Low confidence + positive** → joy (intensity <0.6 with positive keywords)
- **Anxiety patterns** → validates fear predictions
- **Depression patterns** → validates sadness predictions
- **Sarcasm detection** → flags for review (intensity ≥0.85)
- **Mixed emotions** → flags for review (intensity ≥0.85)
- **Negation patterns** → flags for review (intensity ≥0.85)
- **Neutral detection** → overrides to neutral for low-confidence predictions

**Override Validation:**
- Overrides only applied when label actually changes (prevents joy→joy, sadness→sadness)
- Original model predictions stored in `OriginalEmotionLabel` for analysis
- Override types tracked: `positive_keywords`, `gratitude`, `progress`, `humor`, `neutral`, `low_confidence_positive`

**Review Flagging:**
- High-confidence predictions (intensity ≥0.95) flagged for review
- Low-intensity predictions (<0.8) NOT flagged (reduces false positives)
- Flag sources tracked separately:
  - `PostProcessingReviewFlag`: From post-processing rules
  - `AnomalyDetectionFlag`: From user pattern analysis
  - `HighConfidenceFlag`: From intensity threshold
  - `FlagForReview`: Combined flag (OR of all sources)

## Scaling Considerations

### Current (Phase 1)
- **Data Processing**: Pandas (single-threaded)
- **Database**: SQLite (file-based)

### Production Scaling Path
- **Pandas → Bodo**: Replace `pd.DataFrame` operations with Bodo for parallel processing
  - Location: All DataFrame operations in `ingest.py`, `preprocess.py`
  - Migration: Minimal changes due to Bodo's pandas-compatible API
- **SQLite → PostgreSQL/ElastiCache**: 
  - Replace `sqlite3` with `psycopg2` or Redis client
  - Update connection pooling in `persistence.py`
  - Add connection retry logic for distributed systems

## Logging and Audit Trail

The pipeline includes checkpointing for auditability:

```python
from src.utils import checkpoint_dataframe

# Save checkpoint after each transformation
checkpoint_dataframe(df, checkpoint_dir="checkpoints", stage="preprocessing")
```

Checkpoints are saved as Parquet files with timestamps for full traceability.

## Sample Data

- **`data/sample_data.csv`**: 20 rows of noisy conversational text with:
  - Emojis (😊, 😫, 🎉, etc.)
  - Slang/abbreviations (lol, omg, tbh, idk, etc.)
  - Typos and colloquialisms
  - Multiple users and timestamps

- **`data/slang_dictionary.json`**: 30+ common slang/abbreviation mappings

## Post-Processing (`src/postprocess.py`)

**Rule-based corrections** for common misclassifications:
- Positive emotion detection (gratitude, humor, progress)
- Neutral detection
- Human review flagging

**Usage:**
```python
from src.postprocess import apply_post_processing

df_processed = apply_post_processing(df)
# Adds: CorrectedLabel, CorrectedIntensity, WasOverridden, NeedsReview
```

**Context-Aware Post-Processing** (`src/context_aware_postprocess.py`):
- Uses context to validate overrides
- Detects context-text conflicts
- Enhanced review flagging

```python
from src.context_aware_postprocess import apply_context_aware_post_processing

df_processed = apply_context_aware_post_processing(df, context_col='Sequence')
```

## Validation (`src/validation.py`)

**Synthetic Ambiguity Generator:**
```python
from src.validation import AmbiguityGenerator

generator = AmbiguityGenerator()
ambiguous_df = generator.generate_ambiguous_cases(num_cases=50)
```

**Metrics Tracking:**
```python
from src.validation import MetricsTracker

tracker = MetricsTracker()
metrics = tracker.compute_metrics(y_true, y_pred, y_true_intensity, y_pred_intensity)
comparison = tracker.compare_with_without_context(metrics_with, metrics_without)
```

## A/B Testing (`src/ab_testing.py`)

**Test different context strategies and formats:**
```python
from src.ab_testing import ABTester, run_ab_tests

tester = ABTester(model)
results = tester.test_context_strategies(test_cases, strategies, formats)
summary = tester.compare_strategies(results)
```

## Anomaly Detection (`src/anomaly_detection.py`)

**User Pattern Analysis:**
Automatically detects high-risk user emotion patterns:
- Persistent negative emotions (sadness, fear, anger)
- Low positive emotion frequency
- Sudden emotion shifts
- High average intensity

**Usage:**
```python
from src.anomaly_detection import UserPatternAnalyzer

analyzer = UserPatternAnalyzer()
user_analysis = analyzer.analyze_all_users(df)
flagged_users = analyzer.flag_high_risk_users(df)
# Returns users with critical risk or 3+ anomalies
```

**Risk Levels:**
- `low`: Normal patterns
- `medium`: Some concerns
- `high`: Multiple negative patterns
- `critical`: Severe patterns requiring immediate attention

## Analysis Tools

### Results Analysis (`analyze_results.py`)

Comprehensive analysis of pipeline results from the database:

```bash
# Analyze all records
python analyze_results.py

# Analyze only recent records (last 7 days)
python analyze_results.py --since 7

# Analyze since specific date
python analyze_results.py --since 2024-01-15

# Use custom database
python analyze_results.py --db /path/to/database.db --since 7
```

**Features:**
- Emotion distribution analysis
- Intensity score statistics
- Post-processing override analysis (with original emotion tracking)
- Ambiguity detection analysis
- Review flag breakdown by source
- User pattern analysis
- Temporal pattern analysis
- Text quality analysis
- Evaluation metrics

### Synthetic Data Generation (`generate_synthetic_data.py`)

Generate diverse test data for pipeline validation:

```bash
# Generate 500 records for 10 users
python generate_synthetic_data.py --records 500 --users 10 --output data/test_data.csv

# Clear database and generate new data
python generate_synthetic_data.py --records 100 --users 5 --clear-db --output data/synthetic_data.csv
```

**Test Cases Included:**
- Positive examples (gratitude, humor, progress)
- Negative examples (high-intensity emotions)
- Neutral examples
- Ambiguous examples
- Low/high confidence examples
- Sarcasm examples
- Context-dependent examples

## Model Selection

**Selected Model:** `j-hartmann/emotion-english-distilroberta-base`

**Rationale:**
- Based on DistilRoBERTa (resource-efficient, as specified)
- Pre-trained specifically for emotion classification
- Readily available on Hugging Face
- Good performance on conversational text
- Supports intensity prediction via probability scores

**Alternative Models Considered:**
- **multiMentalRoBERTa**: Specialized for mental health but very new (2024) and may not be on Hugging Face
- **RoBERTa-base**: More accurate but slower and more resource-intensive
- **BERT-base**: Older architecture, less efficient

## Next Steps

### Fine-tuning
1. **Collect Labeled Data**:
   - 1000+ labeled conversations from clinical partners
   - Annotate with emotion labels and intensity scores (0.0-1.0)
   - Include ambiguous cases requiring context

2. **Fine-tune Model**:
   - Fine-tune on labeled MWB dataset
   - Validate on synthetic ambiguity dataset
   - A/B test context window sizes (1, 3, 5, 10 turns)

3. **Human-in-the-Loop**:
   - Implement feedback mechanism for ambiguous cases
   - Update slang dictionary based on user corrections
   - Continuous model improvement

### Production Readiness
1. **Performance**:
   - Load testing for <100ms history retrieval at scale
   - Batch optimization for model inference
   - Caching layer (Redis) for frequent users

2. **Monitoring**:
   - Track precision/recall with and without context
   - Monitor clinical precision (false positive/negative rates)
   - Alert on pipeline failures

3. **API**:
   - Create REST API endpoints for real-time inference
   - WebSocket support for streaming conversations
   - Rate limiting and authentication

## Success Criteria (Phase 1) ✅

- ✅ End-to-end demo runs locally and commits structured rows to SQLite
- ✅ `fetch_history()` returns ordered context within <100ms for sample DB
- ✅ Preprocessing preserves stopwords and records `NormalizationFlags` per row
- ✅ Unit tests cover all three layers
- ✅ Documentation includes architecture overview and run instructions

## License

[Specify license]

## Contributing

[Contributing guidelines]
