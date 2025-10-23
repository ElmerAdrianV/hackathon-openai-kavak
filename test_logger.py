#!/usr/bin/env python3
"""
Quick test script for PredictionLogger
"""

from src.prediction_logger import PredictionLogger

# Create logger
logger = PredictionLogger(filename="test_predictions.csv")

# Log a few test predictions
test_data = [
    ("123", "The Matrix", "user_1", 4.5, 0.5, 4.0, "Action, Sci-Fi"),
    ("456", "Inception", "user_1", 4.2, 0.4, 4.5, "Action, Thriller"),
    ("789", "The Godfather", "user_1", 4.8, 0.3, 5.0, "Crime, Drama"),
]

print("Logging test predictions...")
for movie_id, title, user_id, pred, sigma, true, genres in test_data:
    logger.log_prediction(
        movie_id=movie_id,
        movie_title=title,
        user_id=user_id,
        predicted_rating=pred,
        predicted_sigma=sigma,
        true_rating=true,
        genres=genres
    )
    print(f"  ✓ Logged: {title}")

print("\nTest complete!")
logger.print_summary()
