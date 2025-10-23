# Train/Test Split Architecture

## Overview

The system now supports proper train/test splitting to prevent data contamination during evaluation. This ensures that:

1. **Training data** is used to build user context (history, neighbors)
2. **Test data** ratings are NEVER exposed to the model during prediction
3. Movie metadata (title, synopsis, genres) from test set IS accessible

## Data Contamination Prevention

### What is Protected

- ✅ User rating history: built ONLY from train data
- ✅ Movie neighbors: computed ONLY from train data
- ✅ Popularity metrics: calculated ONLY from train data

### What is Accessible

- ✅ Movie metadata (title, overview, genre_list) from BOTH train and test
- ✅ User personality attributes (non-rating-based user info)

## File Structure

```
src/data/splits/
  ├── train_users_1.csv   # Training data: used for context building
  └── test_users_1.csv    # Test data: used for evaluation only
```

## Usage

### Automatic Mode (Recommended)

```bash
# Automatically loads train/test split from src/data/splits/
python3 -m src.main_demo
```

The system will:
1. Detect `train_users_1.csv` and `test_users_1.csv`
2. Build context using only train data
3. Evaluate predictions on test data
4. Display ground truth ratings for comparison

### Single File Mode (Backward Compatible)

```bash
# Use a single file as train data (evaluates on same data)
python3 -m src.main_demo data/my_data.csv
```

### Custom Resources

```bash
# Specify custom resources directory
python3 -m src.main_demo --resources src/resources
```

## Implementation Details

### DataStore (`src/data_store.py`)

```python
# Constructor accepts train and optional test DataFrames
store = DataStore(train_df, test_df)

# Methods guarantee no test contamination:
store.get_user_history(user_id, k=10)  # TRAIN ONLY
store.get_neighbors(movie_id, k=8)     # TRAIN ONLY
store.get_movie(movie_id)              # TRAIN + TEST metadata
```

### Key Design Decisions

1. **Movie Metadata Merging**: Movie metadata (title, overview, genres) is merged from both train and test to ensure complete coverage. This is safe because:
   - Synopsis and genre are properties of the movie, not user-specific
   - They don't reveal user ratings or preferences

2. **User History**: Built exclusively from train data to prevent leaking test ratings into the model's context.

3. **Neighbors**: Computed only from train data using Jaccard similarity on genres + popularity from train.

4. **Personality**: Merged from both train and test since it's a user attribute (not rating-based), ensuring consistency.

## Validation

To verify no data contamination:

```python
# Check that test ratings are not in train
train_pairs = set(zip(train_df['userId'], train_df['movieId']))
test_pairs = set(zip(test_df['userId'], test_df['movieId']))
assert len(train_pairs & test_pairs) == 0, "Overlap detected!"
```

## Example Output

```
[Init] Loading train/test split:
  Train: /path/to/src/data/splits/train_users_1.csv
  Test:  /path/to/src/data/splits/test_users_1.csv
[Demo] Train/Test split loaded: evaluating on test set (no contamination)

[Demo] Evaluating row 0: userId=1, movieId=58559, title=Confession of a Child of the Century
[Predict 1] -> 3.45 ± 0.82 | aux={...}
[Ground Truth] true_rating=4.00
```

## Migration Guide

### From Single DataFrame

**Before:**
```python
df = pd.read_csv("data.csv")
store = DataStore(df)
```

**After:**
```python
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
store = DataStore(train_df, test_df)
```

### Backward Compatibility

The system remains fully backward compatible:
```python
# Still works (test_df defaults to None)
store = DataStore(train_df)
```

## Best Practices

1. **Always use train/test splits** for production evaluation
2. **Verify no overlap** between train and test (userId, movieId) pairs
3. **Use same user across splits** to test generalization to new movies
4. **Keep personality consistent** across train/test for the same user
5. **Document your split strategy** (temporal, random, stratified, etc.)

## Future Enhancements

- [ ] Add validation split support
- [ ] Implement k-fold cross-validation
- [ ] Add metrics tracking (MAE, RMSE, coverage)
- [ ] Support temporal splits (train on past, test on future)
- [ ] Add cold-start user/movie handling
