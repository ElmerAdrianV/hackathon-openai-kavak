# Reviewer Agent - Meta-Analysis System

## Overview

The **Reviewer** is a meta-learning agent that analyzes the performance of judges and critics in the movie recommendation system. It provides insights into system calibration, identifies strengths and weaknesses, and suggests improvements.

## Purpose

The Reviewer addresses the question: *"How well are our judges performing, and how can we improve?"*

By periodically analyzing prediction history, the Reviewer:
- Tracks judge accuracy and consistency
- Monitors critic utilization (alpha weights)
- Identifies performance trends
- Suggests calibrator adjustments
- Provides actionable recommendations for prompt improvements

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Orchestrator                       │
│  ┌─────────┐  ┌─────────┐  ┌──────────────┐        │
│  │ Critics │→ │ Judges  │→ │ Calibrator   │        │
│  └─────────┘  └─────────┘  └──────────────┘        │
│       ↓            ↓               ↓                 │
│  ┌──────────────────────────────────────────┐      │
│  │         Prediction Logs                   │      │
│  └──────────────────────────────────────────┘      │
└────────────────────┬────────────────────────────────┘
                     ↓
            ┌────────────────┐
            │    Reviewer    │
            │  - Statistics  │
            │  - Analysis    │
            │  - Suggestions │
            └────────────────┘
```

## Key Features

### 1. Judge Performance Tracking
- **Average Error**: Mean absolute error for each judge
- **Consistency**: Standard deviation of errors (lower = more consistent)
- **Prediction Count**: Number of predictions made

### 2. Critic Utilization Analysis
- **Alpha Weights**: How heavily each critic is weighted by judges
- **Utilization Patterns**: Which critics are most/least trusted
- **Correlation with Performance**: Do certain critics correlate with better predictions?

### 3. System-Level Metrics
- **Overall Error**: Average across all judges and predictions
- **Error Trends**: Is performance improving over time?
- **Calibration Quality**: How well is the calibrator working?

### 4. Recommendations
The Reviewer generates actionable suggestions:
- 📊 **Prompt Improvements**: Identify judges that need better instructions
- 🔧 **Calibrator Tuning**: Suggest dimension increases or learning rate changes
- ⚡ **Critic Optimization**: Highlight underutilized or overutilized critics
- ✅ **Performance Validation**: Confirm when system is well-calibrated

## Usage

### Basic Integration

```python
from src.reviewer import Reviewer

# Initialize with review interval (every N predictions)
reviewer = Reviewer(review_interval=5)

# During prediction loop
for prediction in predictions:
    # Make prediction
    yhat, sigma, aux = orch.predict(user_id=u, movie_id=m)
    
    # Record for reviewer
    events = orch.logger.read_all()
    last_event = events[-1]
    reviewer.record_prediction(
        judge_outputs=last_event['judge_outputs'],
        true_rating=true_rating,
        critic_ids=critic_ids
    )
    
    # Check if review should run
    if reviewer.should_review():
        report = reviewer.run_review()
        reviewer.print_report(report)
```

### Command Line

```bash
# Default: review every 5 predictions
python -m src.main_demo --samples 20

# Custom review interval
python -m src.main_demo --samples 30 --review-interval 10

# Multiple options
python -m src.main_demo --samples 50 --review-interval 5 --resources ./resources
```

## Output Example

```
================================================================================
🔍 REVIEWER ANALYSIS - 2025-10-23T14:30:00
================================================================================
Total predictions analyzed: 10
Overall avg error: 0.847 ± 0.423

📊 Judge Performance:
  • grounded_v1           | Avg Error: 0.782 | Std: 0.345 | Predictions: 10
  • balanced_moderate     | Avg Error: 0.912 | Std: 0.501 | Predictions: 10

🏆 Best performing judge: grounded_v1
⚠️  Needs improvement: balanced_moderate

🎭 Critic Utilization (avg alpha weights):
  • cinephile             [████████████░░░░░░░░] 0.623
  • character_focused     [██████░░░░░░░░░░░░░░] 0.312
  • technical_expert      [███░░░░░░░░░░░░░░░░░] 0.157
  • comedy_specialist     [█░░░░░░░░░░░░░░░░░░░] 0.089

💡 Recommendations:
  ✅ Excellent overall performance! System is well-calibrated.
  🔍 Underutilized critics: comedy_specialist. These critics may need prompt improvements.
  ⭐ Heavily weighted critics: cinephile. These critics are highly trusted by judges.
================================================================================
```

## Data Structures

### JudgeStats
```python
@dataclass
class JudgeStats:
    judge_id: str
    predictions: List[float]  # r_tilde values
    errors: List[float]       # abs(r_tilde - true_rating)
    avg_error: float
    std_error: float
    consistency: float
```

### ReviewReport
```python
@dataclass
class ReviewReport:
    timestamp: str
    num_predictions: int
    judge_stats: List[JudgeStats]
    overall_avg_error: float
    overall_std_error: float
    best_judge: Optional[str]
    worst_judge: Optional[str]
    recommendations: List[str]
    critic_utilization: Dict[str, float]
```

## Future Extensions

### Automatic Prompt Refinement
Currently, the Reviewer provides recommendations. Future versions could:
- Automatically generate improved prompts for underperforming judges
- Use LLM to analyze error patterns and suggest specific changes
- A/B test different prompt variations

### Calibrator Auto-Tuning
The Reviewer could:
- Automatically adjust calibrator dimension based on performance
- Modify learning rates dynamically
- Implement early stopping when performance plateaus

### Critic Selection Optimization
Advanced features could include:
- Dynamic critic selection based on movie genre/features
- Automatic critic pool expansion/pruning
- Meta-learning to identify optimal critic combinations

### Historical Analysis
- Track performance across multiple sessions
- Identify long-term trends
- Compare performance across different user segments or movie types

## Best Practices

1. **Review Interval**: 
   - Too frequent (every prediction): Noisy, unstable statistics
   - Too infrequent (every 50 predictions): Delayed feedback
   - Recommended: 5-10 predictions

2. **Minimum Data**: 
   - Wait for at least 3-5 predictions before running first review
   - More data = more reliable statistics

3. **Interpretation**:
   - Focus on trends, not individual predictions
   - Consider context (e.g., difficult movies may have higher errors)
   - Use recommendations as starting points, not absolute truth

4. **Action Items**:
   - Review underperforming judge prompts
   - Investigate underutilized critics
   - Monitor calibrator suggestions but test before applying

## Technical Notes

### Thread Safety
Current implementation is **not thread-safe**. If using in multi-threaded environment, add appropriate locks.

### Memory Usage
The Reviewer stores full prediction history. For long-running systems, consider:
- Periodic reset with `reviewer.reset()`
- Sliding window of recent predictions
- Archiving old reports to disk

### Performance Impact
- Minimal overhead during `record_prediction()`
- Review computation is O(n) where n = predictions since last review
- Typically < 100ms for review intervals of 5-10 predictions

## Related Components

- **Orchestrator**: Coordinates predictions, source of logs
- **Judges**: Evaluated by Reviewer
- **Critics**: Utilization tracked by Reviewer  
- **Calibrator**: Target of optimization suggestions
- **LoggingStore**: Source of prediction event data
