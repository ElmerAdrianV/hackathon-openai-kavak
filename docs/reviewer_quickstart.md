# Reviewer Agent - Quick Start

## What is it?

The **Reviewer** is a meta-analysis agent that monitors judge and critic performance, providing insights and recommendations for system improvement.

## Quick Usage

```bash
# Run with default settings (review every 5 predictions)
python -m src.main_demo --samples 20

# Custom review interval
python -m src.main_demo --samples 30 --review-interval 10
```

## What You'll See

### During Execution
```
[1/10] The Matrix    | Pred: 4.25 ± 0.50 | Real: 4.50 | Error: 0.25
[2/10] Inception     | Pred: 4.10 ± 0.45 | Real: 4.00 | Error: 0.10
...
```

### Reviewer Analysis (every N predictions)
```
================================================================================
🔍 REVIEWER ANALYSIS
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

💡 Recommendations:
  ✅ Excellent overall performance! System is well-calibrated.
  ⭐ Heavily weighted critics: cinephile. These critics are highly trusted.
================================================================================
```

## Key Metrics

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **Avg Error** | Mean absolute error | < 0.8 |
| **Std Error** | Consistency (lower = better) | < 0.5 |
| **Alpha Weight** | How much judges trust a critic | Balanced distribution |

## Recommendations Explained

| Icon | Meaning |
|------|---------|
| ✅ | System performing well |
| ⚠️ | High error - needs attention |
| 📊 | Inconsistent judge - refine prompts |
| ⚡ | Underperforming - review configuration |
| 🔍 | Underutilized critic - may need better prompts |
| ⭐ | Highly trusted critic |
| 🔧 | Calibrator adjustment suggested |

## Integration Example

```python
from src.reviewer import Reviewer

# Initialize
reviewer = Reviewer(review_interval=5)

# In prediction loop
for movie in movies:
    yhat, sigma, aux = orch.predict(user_id=u, movie_id=m)
    
    # Record for analysis
    reviewer.record_prediction(judge_outputs, true_rating, critic_ids)
    
    # Run review if interval reached
    if reviewer.should_review():
        report = reviewer.run_review()
        reviewer.print_report(report)
```

## Future Enhancements (Boilerplate Ready)

The Reviewer provides a foundation for:

1. **Automatic Prompt Refinement** - Use LLM to improve underperforming judge prompts
2. **Calibrator Auto-Tuning** - Dynamically adjust parameters based on performance
3. **Critic Selection Optimization** - Choose optimal critics per movie/genre
4. **A/B Testing** - Compare different prompt versions
5. **Historical Trends** - Track long-term performance evolution

## Files

- `src/reviewer.py` - Main Reviewer implementation
- `docs/reviewer.md` - Detailed documentation
- `src/main_demo.py` - Integration example

## Learn More

See [docs/reviewer.md](reviewer.md) for detailed architecture, data structures, and extension points.
