#!/usr/bin/env python3
"""
Quick validation script to verify train/test split has no data contamination.
"""
from pathlib import Path
import pandas as pd
import sys

def validate_split(train_path: str, test_path: str) -> bool:
    """Validate that train and test splits have no overlap in (userId, movieId) pairs."""
    
    print(f"[Validation] Loading files...")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"\n[Stats]")
    print(f"  Train rows: {len(train_df)}")
    print(f"  Test rows:  {len(test_df)}")
    
    # Check required columns
    required = ["userId", "movieId", "rating"]
    for col in required:
        if col not in train_df.columns:
            print(f"❌ Train missing column: {col}")
            return False
        if col not in test_df.columns:
            print(f"❌ Test missing column: {col}")
            return False
    
    # Check for overlap in (userId, movieId) pairs
    train_pairs = set(zip(train_df["userId"].astype(str), train_df["movieId"].astype(str)))
    test_pairs = set(zip(test_df["userId"].astype(str), test_df["movieId"].astype(str)))
    
    overlap = train_pairs & test_pairs
    
    print(f"\n[Overlap Check]")
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Test pairs:  {len(test_pairs)}")
    print(f"  Overlap:     {len(overlap)}")
    
    if overlap:
        print(f"\n❌ DATA CONTAMINATION DETECTED!")
        print(f"   {len(overlap)} (userId, movieId) pairs appear in BOTH train and test")
        print(f"   Example overlaps (first 5):")
        for i, (uid, mid) in enumerate(list(overlap)[:5]):
            print(f"     {i+1}. userId={uid}, movieId={mid}")
        return False
    
    # Check user coverage
    train_users = set(train_df["userId"].astype(str))
    test_users = set(test_df["userId"].astype(str))
    common_users = train_users & test_users
    
    print(f"\n[User Coverage]")
    print(f"  Train users: {len(train_users)}")
    print(f"  Test users:  {len(test_users)}")
    print(f"  Common:      {len(common_users)}")
    
    if len(common_users) == 0:
        print(f"  ⚠️  Warning: No common users between train and test (cold-start scenario)")
    else:
        print(f"  ✅ Good: {len(common_users)} users appear in both (testing generalization to new movies)")
    
    # Check movie coverage
    train_movies = set(train_df["movieId"].astype(str))
    test_movies = set(test_df["movieId"].astype(str))
    common_movies = train_movies & test_movies
    test_only_movies = test_movies - train_movies
    
    print(f"\n[Movie Coverage]")
    print(f"  Train movies: {len(train_movies)}")
    print(f"  Test movies:  {len(test_movies)}")
    print(f"  Common:       {len(common_movies)}")
    print(f"  Test-only:    {len(test_only_movies)}")
    
    if len(test_only_movies) > 0:
        print(f"  ✅ Good: {len(test_only_movies)} movies only in test (evaluating on new items)")
    
    print(f"\n✅ VALIDATION PASSED: No data contamination detected!")
    return True


if __name__ == "__main__":
    # Default paths
    here = Path(__file__).resolve().parent
    default_train = here / "data" / "splits" / "train_users_10.csv"
    default_test = here / "data" / "splits" / "test_users_10.csv"
    
    if len(sys.argv) == 3:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
    elif default_train.exists() and default_test.exists():
        train_path = str(default_train)
        test_path = str(default_test)
    else:
        print("Usage: python validate_split.py <train.csv> <test.csv>")
        print(f"  or place files at:")
        print(f"    {default_train}")
        print(f"    {default_test}")
        sys.exit(1)
    
    success = validate_split(train_path, test_path)
    sys.exit(0 if success else 1)
