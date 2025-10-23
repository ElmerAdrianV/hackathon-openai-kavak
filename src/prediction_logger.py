"""
Simple prediction logger that writes predictions to CSV file.
"""

import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class PredictionLogger:
    """
    Logs predictions to a CSV file for later analysis.
    
    Columns:
    - timestamp: When the prediction was made
    - movie_id: Movie identifier
    - movie_title: Movie title
    - user_id: User identifier
    - predicted_rating: System's prediction (yhat)
    - predicted_sigma: Uncertainty estimate
    - true_rating: Actual rating from user
    - error: Absolute error |predicted - true|
    - genres: Movie genres (comma-separated)
    """
    
    def __init__(self, log_dir: str = "logs", filename: Optional[str] = None):
        """
        Initialize prediction logger.
        
        Args:
            log_dir: Directory to store log files
            filename: Custom filename (default: predictions_YYYYMMDD_HHMMSS.csv)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.csv"
        
        self.log_path = self.log_dir / filename
        self.fieldnames = [
            'timestamp',
            'movie_id',
            'movie_title',
            'user_id',
            'predicted_rating',
            'predicted_sigma',
            'true_rating',
            'error',
            'genres'
        ]
        
        # Create file with header if it doesn't exist
        if not self.log_path.exists():
            self._write_header()
        
        print(f"[PredictionLogger] Logging to: {self.log_path}")
    
    def _write_header(self):
        """Write CSV header."""
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def log_prediction(
        self,
        movie_id: str,
        movie_title: str,
        user_id: str,
        predicted_rating: float,
        predicted_sigma: float,
        true_rating: float,
        genres: Optional[str] = None
    ):
        """
        Log a single prediction to CSV.
        
        Args:
            movie_id: Movie identifier
            movie_title: Movie title
            user_id: User identifier
            predicted_rating: System's prediction
            predicted_sigma: Uncertainty estimate
            true_rating: Ground truth rating
            genres: Movie genres (comma-separated string)
        """
        error = abs(predicted_rating - true_rating)
        
        row = {
            'timestamp': datetime.now().isoformat(),
            'movie_id': movie_id,
            'movie_title': movie_title,
            'user_id': user_id,
            'predicted_rating': f"{predicted_rating:.3f}",
            'predicted_sigma': f"{predicted_sigma:.3f}",
            'true_rating': f"{true_rating:.3f}",
            'error': f"{error:.3f}",
            'genres': genres or ''
        }
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Read log file and compute summary statistics.
        
        Returns:
            Dictionary with statistics (avg_error, count, etc.)
        """
        if not self.log_path.exists():
            return {}
        
        errors = []
        predictions = []
        true_ratings = []
        
        with open(self.log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    errors.append(float(row['error']))
                    predictions.append(float(row['predicted_rating']))
                    true_ratings.append(float(row['true_rating']))
                except (ValueError, KeyError):
                    continue
        
        if not errors:
            return {}
        
        return {
            'count': len(errors),
            'avg_error': sum(errors) / len(errors),
            'min_error': min(errors),
            'max_error': max(errors),
            'avg_prediction': sum(predictions) / len(predictions),
            'avg_true_rating': sum(true_ratings) / len(true_ratings)
        }
    
    def print_summary(self):
        """Print summary statistics from log file."""
        stats = self.get_summary_stats()
        
        if not stats:
            print("No predictions logged yet.")
            return
        
        print(f"\n{'='*60}")
        print(f"PREDICTION LOG SUMMARY: {self.log_path.name}")
        print(f"{'='*60}")
        print(f"Total predictions:     {stats['count']}")
        print(f"Average error:         {stats['avg_error']:.3f}")
        print(f"Min error:             {stats['min_error']:.3f}")
        print(f"Max error:             {stats['max_error']:.3f}")
        print(f"Average prediction:    {stats['avg_prediction']:.3f}")
        print(f"Average true rating:   {stats['avg_true_rating']:.3f}")
        print(f"{'='*60}\n")
