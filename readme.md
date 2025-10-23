````markdown
# Agentic Movie Recommender — Debate + Judge (Boilerplate)

This is a **scaffold** for the Debate + Judge recommender with an **Orchestrator agent**,
**Critic tool-agents**, **Judge tool-agents**, a **Calibrator** head, and a **Router** (bandit policy).
It is intentionally minimal so you can plug in your own prompts, rules, and update logic.

## 🎯 Key Features

- ✅ **Train/Test Split Support**: Prevents data contamination during evaluation
- ✅ **Multi-Critic Debate**: Multiple critic personas analyze movies from different perspectives
- ✅ **Judge Aggregation**: Judges synthesize critic opinions with skill tracking
- ✅ **Online Learning**: Calibrator and judges update from user feedback
- ✅ **Relative Path Resolution**: Works consistently regardless of execution directory

## Layout
```
agentic_rec/
  ├─ docs/
  │   ├─ arquitectura.md
  │   └─ train_test_split.md  # 📖 Train/test split documentation
  ├─ src/
  │   ├─ data/
  │   │   └─ splits/
  │   │       ├─ train_users_1.csv  # Training data
  │   │       └─ test_users_1.csv   # Test data (no contamination)
  │   ├─ resources/
  │   │   ├─ movie_critics/    # Critic persona prompts
  │   │   └─ judges/           # Judge prompts
  │   ├─ orchestrator.py       # primary agent: routing, debate, aggregation, updates
  │   ├─ critics.py            # critic personae + manager
  │   ├─ judges.py             # judges + skill tracking
  │   ├─ calibrator.py         # online regressor + uncertainty head
  │   ├─ router.py             # routing/bandit policy for critics/judges
  │   ├─ retriever.py          # user/movie context retrieval
  │   ├─ data_store.py         # 🔒 Train/test aware data storage
  │   ├─ features.py           # feature builders for calibrator
  │   ├─ logging_store.py      # event logging and replay helpers
  │   ├─ types.py              # dataclasses for typed IO
  │   ├─ llm_client.py         # interface to your LLM provider
  │   ├─ validate_split.py     # 🔍 Data contamination checker
  │   └─ main_demo.py          # runnable demo
  └─ requirements.txt
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key
```bash
export OPENAI_API_KEY='sk-...'
# or create a .env file with OPENAI_API_KEY=sk-...
```

### 3. Run Demo (Automatic Train/Test Split)
```bash
python3 -m src.main_demo
```

The system will:
- Automatically detect `src/data/splits/train_users_1.csv` and `test_users_1.csv`
- Build user context ONLY from train data (no contamination)
- Evaluate predictions on test data
- Display predictions vs ground truth

### 4. Validate Data Integrity
```bash
python3 src/validate_split.py
```

This ensures no (userId, movieId) pairs overlap between train and test.

## 🔒 Train/Test Split Architecture

The system prevents data contamination by:

- **Train data** → Used for user history, neighbors, popularity metrics
- **Test data** → Only movie metadata accessible; ratings NEVER exposed during prediction
- **Movie metadata** → Available from both train and test (title, synopsis, genres)

See [docs/train_test_split.md](docs/train_test_split.md) for detailed documentation.

## Usage Examples

### Default Mode (Train/Test Split)
```bash
# Uses src/data/splits/train_users_1.csv + test_users_1.csv
python3 -m src.main_demo
```

### Single File Mode (Backward Compatible)
```bash
# Use single file as train data
python3 -m src.main_demo data/my_ratings.csv
```

### Custom Resources Directory
```bash
# Specify custom critic/judge prompts location
python3 -m src.main_demo --resources path/to/resources
```

### With Verbose Output
```bash
# See detailed critic/judge outputs
VERBOSE=1 python3 -m src.main_demo
```

## Development

### Adding New Critics

1. Create a new prompt file in `src/resources/movie_critics/`:
```bash
echo "You are a Horror Movie Expert critic..." > src/resources/movie_critics/horror_expert.txt
```

2. The system auto-discovers and loads it on next run.

### Adding New Judges

1. Create a new prompt file in `src/resources/judges/`:
```bash
echo "You are a Consensus Judge..." > src/resources/judges/consensus_v1.txt
```

2. Auto-discovered on next run.

### Testing Changes

```bash
# Check for syntax errors
python3 -m py_compile src/*.py

# Validate data integrity
python3 src/validate_split.py

# Run demo with verbose output
VERBOSE=1 python3 -m src.main_demo
```

## Architecture Notes

- **Calibrator**: Tiny online linear regressor (no external ML deps), updates from user feedback
- **Router**: Bandit-style policy for selecting critics/judges based on genre and performance
- **Judge Skill Tracking**: Judges maintain skill estimates that improve with feedback
- **Critic Performance**: Track critic accuracy over time (exponential moving average)
- **Logging**: All predictions/debates logged to `./logs/events.jsonl`

## Data Format

### Required CSV Columns

```
userId,movieId,rating,title,overview,genres,genre_list,personality
1,858,5.0,Sleepless in Seattle,"A young boy...",<json>,"['Comedy','Drama','Romance']","Drama fan..."
```

- `userId`: User identifier (any type, converted to string)
- `movieId`: Movie identifier (any type, converted to string)
- `rating`: User rating (float, typically 0-5)
- `title`: Movie title (string)
- `overview`: Movie synopsis (string)
- `genre_list`: Parsed list of genre strings
- `personality`: User personality/profile description (optional)

## Notes

- Everything is typed for clarity but simplified for portability
- Relative paths work regardless of execution directory
- Train/test split is optional but highly recommended for evaluation
- System is backward compatible with single-file datasets

## Next Steps

1. ✅ Set up train/test splits in `src/data/splits/`
2. ✅ Add your LLM API key
3. ⚠️ Customize critic prompts in `src/resources/movie_critics/`
4. ⚠️ Customize judge prompts in `src/resources/judges/`
5. ⚠️ Implement advanced update logic in `orchestrator.py`
6. ⚠️ Add evaluation metrics (MAE, RMSE, coverage)

````
