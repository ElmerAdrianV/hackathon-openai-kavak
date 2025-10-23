#!/usr/bin/env python3
"""
Analyze prediction logs from CSV files.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict


def analyze_log(log_path: str):
    """Analyze a prediction log file."""
    
    path = Path(log_path)
    if not path.exists():
        print(f"Error: File not found: {log_path}")
        return
    
    # Read data
    predictions = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                predictions.append({
                    'timestamp': row['timestamp'],
                    'movie_id': row['movie_id'],
                    'movie_title': row['movie_title'],
                    'user_id': row['user_id'],
                    'predicted': float(row['predicted_rating']),
                    'sigma': float(row['predicted_sigma']),
                    'true': float(row['true_rating']),
                    'error': float(row['error']),
                    'genres': row.get('genres', '')
                })
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid row: {e}")
                continue
    
    if not predictions:
        print("No valid predictions found in log.")
        return
    
    # Calculate statistics
    errors = [p['error'] for p in predictions]
    avg_error = sum(errors) / len(errors)
    max_error = max(errors)
    min_error = min(errors)
    
    predicted_ratings = [p['predicted'] for p in predictions]
    true_ratings = [p['true'] for p in predictions]
    avg_predicted = sum(predicted_ratings) / len(predicted_ratings)
    avg_true = sum(true_ratings) / len(true_ratings)
    
    # Genre analysis
    genre_errors = defaultdict(list)
    for p in predictions:
        if p['genres']:
            genres = [g.strip() for g in p['genres'].split(',')]
            for genre in genres:
                genre_errors[genre].append(p['error'])
    
    # Print report
    print(f"\n{'='*80}")
    print(f"PREDICTION LOG ANALYSIS: {path.name}")
    print(f"{'='*80}")
    print(f"\n📊 OVERALL STATISTICS")
    print(f"  Total predictions:     {len(predictions)}")
    print(f"  Average error:         {avg_error:.3f}")
    print(f"  Min error:             {min_error:.3f}")
    print(f"  Max error:             {max_error:.3f}")
    print(f"  Average prediction:    {avg_predicted:.3f}")
    print(f"  Average true rating:   {avg_true:.3f}")
    
    # Best and worst predictions
    best = min(predictions, key=lambda p: p['error'])
    worst = max(predictions, key=lambda p: p['error'])
    
    print(f"\n🏆 BEST PREDICTION")
    print(f"  Movie: {best['movie_title']}")
    print(f"  Predicted: {best['predicted']:.2f} | True: {best['true']:.2f} | Error: {best['error']:.3f}")
    
    print(f"\n⚠️  WORST PREDICTION")
    print(f"  Movie: {worst['movie_title']}")
    print(f"  Predicted: {worst['predicted']:.2f} | True: {worst['true']:.2f} | Error: {worst['error']:.3f}")
    
    # Genre analysis
    if genre_errors:
        print(f"\n🎭 ERROR BY GENRE (Top 10)")
        sorted_genres = sorted(
            genre_errors.items(),
            key=lambda x: sum(x[1]) / len(x[1])
        )[:10]
        
        for genre, errs in sorted_genres:
            avg_genre_error = sum(errs) / len(errs)
            count = len(errs)
            print(f"  {genre:25s} | Avg Error: {avg_genre_error:.3f} | Count: {count}")
    
    # Detailed predictions table
    print(f"\n📋 DETAILED PREDICTIONS")
    print(f"  {'Movie':<40} | {'Pred':>5} | {'True':>5} | {'Error':>6} | Genres")
    print(f"  {'-'*40}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+{'-'*20}")
    
    for p in predictions:
        title = p['movie_title'][:40]
        genres = p['genres'][:20] if p['genres'] else 'N/A'
        print(f"  {title:40} | {p['predicted']:5.2f} | {p['true']:5.2f} | {p['error']:6.3f} | {genres}")
    
    print(f"\n{'='*80}\n")


def main():
    if len(sys.argv) < 2:
        # Find most recent log file
        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("Error: logs/ directory not found")
            sys.exit(1)
        
        pred_logs = list(logs_dir.glob("predictions_*.csv"))
        if not pred_logs:
            print("Error: No prediction logs found in logs/")
            sys.exit(1)
        
        # Use most recent
        log_file = max(pred_logs, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent log: {log_file}")
    else:
        log_file = sys.argv[1]
    
    analyze_log(log_file)


if __name__ == "__main__":
    main()
