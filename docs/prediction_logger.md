# Prediction Logger - Quick Guide

## What is it?

A simple CSV logger that saves every prediction made by the system for later analysis.

## Log File Location

Logs are saved in `logs/predictions_YYYYMMDD_HHMMSS.csv`

## Log Format

```csv
timestamp,movie_id,movie_title,user_id,predicted_rating,predicted_sigma,true_rating,error,genres
2025-10-23T16:29:56,86637,Ricco,45811,3.000,0.750,2.500,0.500,"Action, Foreign"
2025-10-23T16:32:05,831,This Island Earth,45811,2.871,0.750,3.000,0.129,"Mystery, Adventure"
```

## Usage

### Automatic Logging (Built-in)

When you run the demo, predictions are automatically logged:

```bash
python -m src.main_demo --samples 20
```

This creates a new log file: `logs/predictions_YYYYMMDD_HHMMSS.csv`

### Analyze Logs

View detailed statistics from your prediction logs:

```bash
# Analyze most recent log
python analyze_predictions.py

# Analyze specific log file
python analyze_predictions.py logs/predictions_20251023_162822.csv
```

### Sample Output

```
================================================================================
PREDICTION LOG ANALYSIS: predictions_20251023_162822.csv
================================================================================

📊 OVERALL STATISTICS
  Total predictions:     20
  Average error:         0.541
  Min error:             0.012
  Max error:             1.234
  Average prediction:    3.425
  Average true rating:   3.482

🏆 BEST PREDICTION
  Movie: Inception
  Predicted: 4.51 | True: 4.50 | Error: 0.012

⚠️  WORST PREDICTION
  Movie: The Room
  Predicted: 3.20 | True: 2.00 | Error: 1.234

🎭 ERROR BY GENRE (Top 10)
  Drama                     | Avg Error: 0.342 | Count: 15
  Action                    | Avg Error: 0.521 | Count: 8
  Comedy                    | Avg Error: 0.678 | Count: 6
```

## Manual Integration

```python
from src.prediction_logger import PredictionLogger

# Create logger
logger = PredictionLogger()

# Log a prediction
logger.log_prediction(
    movie_id="123",
    movie_title="The Matrix",
    user_id="user_1",
    predicted_rating=4.5,
    predicted_sigma=0.5,
    true_rating=4.0,
    genres="Action, Sci-Fi"
)

# Print summary
logger.print_summary()
```

## What Gets Logged?

| Field | Description | Example |
|-------|-------------|---------|
| `timestamp` | When prediction was made | `2025-10-23T16:29:56.594187` |
| `movie_id` | Movie identifier | `86637` |
| `movie_title` | Movie title | `Inception` |
| `user_id` | User identifier | `45811` |
| `predicted_rating` | System's prediction (ŷ) | `4.250` |
| `predicted_sigma` | Uncertainty estimate (σ) | `0.500` |
| `true_rating` | Ground truth rating | `4.000` |
| `error` | Absolute error \|ŷ - y\| | `0.250` |
| `genres` | Movie genres | `Action, Sci-Fi` |

## Use Cases

1. **Performance Tracking** - Monitor how the system improves over time
2. **Error Analysis** - Identify which movies/genres are hardest to predict
3. **Debugging** - Trace prediction history for specific users/movies
4. **Research** - Export data for external analysis (Excel, Python, R, etc.)
5. **A/B Testing** - Compare different model configurations

## Tips

- Log files are timestamped, so each run creates a new file
- CSV format is easy to import into Excel, Google Sheets, or pandas
- Use `analyze_predictions.py` for quick insights
- For custom analysis, read the CSV with pandas:
  ```python
  import pandas as pd
  df = pd.read_csv('logs/predictions_20251023_162822.csv')
  ```

## Files

- `src/prediction_logger.py` - Logger implementation
- `analyze_predictions.py` - Analysis script
- `logs/predictions_*.csv` - Log files (gitignored)
