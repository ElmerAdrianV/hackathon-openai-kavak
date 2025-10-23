from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class DataStore:
    """
    Wrapper for train/test split data to prevent data contamination.

    Required columns:
      - userId, movieId, rating
      - title, overview
      - genre_list  (list[str] per row)
      - personality (string; may repeat per user)

    Design:
      - train_df: Used for building user history, neighbors, and all context features
      - test_df: Only movie metadata (title, overview, genres) accessible; ratings NEVER exposed
      - Movie metadata is merged from both train and test for complete coverage
    """

    def __init__(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None):
        # Basic cleanup / normalization for train
        needed = ["userId", "movieId", "rating", "title", "overview", "genre_list"]
        for col in needed:
            if col not in train_df.columns:
                raise ValueError(f"train_df missing required column: {col}")

        self.train_df = train_df.copy()
        # If personality column is missing, add a default
        if "personality" not in self.train_df.columns:
            self.train_df["personality"] = ""

        # Normalize types
        self.train_df["userId"] = self.train_df["userId"].astype(str)
        self.train_df["movieId"] = self.train_df["movieId"].astype(str)
        # Ensure genre_list is a list[str]
        self.train_df["genre_list"] = self.train_df["genre_list"].apply(
            lambda g: g if isinstance(g, list) else []
        )

        # Deduplicate (userId, movieId) keeping the last occurrence
        self.train_df = (
            self.train_df.sort_index()
            .drop_duplicates(subset=["userId", "movieId"], keep="last")
            .reset_index(drop=True)
        )

        # Process test_df if provided (for movie metadata only, no ratings exposed)
        self.test_df = None
        if test_df is not None:
            self.test_df = test_df.copy()
            if "personality" not in self.test_df.columns:
                self.test_df["personality"] = ""
            self.test_df["userId"] = self.test_df["userId"].astype(str)
            self.test_df["movieId"] = self.test_df["movieId"].astype(str)
            self.test_df["genre_list"] = self.test_df["genre_list"].apply(
                lambda g: g if isinstance(g, list) else []
            )

        # Keep reference to "full df" for backward compatibility (train only)
        self.df = self.train_df

        # Precompute per-movie metadata from BOTH train and test (metadata only, no ratings from test)
        movie_cols = ["movieId", "title", "overview", "genre_list"]
        
        # Start with train movies
        train_movies = (
            self.train_df[movie_cols]
            .drop_duplicates(subset=["movieId"], keep="first")
        )
        
        # Add test movies (metadata only) if available
        if self.test_df is not None:
            test_movies = (
                self.test_df[movie_cols]
                .drop_duplicates(subset=["movieId"], keep="first")
            )
            # Concatenate and deduplicate (train takes priority)
            all_movies = pd.concat([train_movies, test_movies], ignore_index=True)
            all_movies = all_movies.drop_duplicates(subset=["movieId"], keep="first")
        else:
            all_movies = train_movies
        
        self.movies = all_movies.set_index("movieId").to_dict(orient="index")

        # Basic popularity proxy = count of ratings per movie (TRAIN ONLY)
        counts = self.train_df.groupby("movieId")["rating"].size().rename("count").to_frame()
        self.movie_popularity = counts["count"].to_dict()

        # Per-user aggregates for quick access (TRAIN ONLY - no test contamination)
        # user history sorted by rating desc, then arbitrary
        self.user_histories = (
            self.train_df.sort_values(["userId", "rating"], ascending=[True, False])
            .groupby("userId")
            .apply(
                lambda g: g[["movieId", "rating", "title", "genre_list"]].to_dict(
                    orient="records"
                )
            )
            .to_dict()
        )

        # Per-user "personality": take the *most frequent* non-empty personality string (from train or test)
        def _mode_non_empty(vals: List[str]) -> str:
            vals = [v for v in vals if isinstance(v, str) and v.strip()]
            if not vals:
                return ""
            # mode
            uniq, counts = np.unique(vals, return_counts=True)
            return uniq[int(np.argmax(counts))]

        # Merge personality from both train and test (personality is user attribute, not rating-based)
        all_pers_df = self.train_df[["userId", "personality"]].copy()
        if self.test_df is not None:
            test_pers = self.test_df[["userId", "personality"]].copy()
            all_pers_df = pd.concat([all_pers_df, test_pers], ignore_index=True)
        
        per_user_pers = (
            all_pers_df.groupby("userId")["personality"]
            .apply(lambda s: _mode_non_empty(s.tolist()))
            .to_dict()
        )
        self.user_personality = per_user_pers

    # ---------- Query helpers ----------

    def get_movie(self, movie_id: str) -> Optional[Dict[str, Any]]:
        return self.movies.get(str(movie_id))

    def get_user_history(self, user_id: str, k: int = 20) -> List[Dict[str, Any]]:
        hist = self.user_histories.get(str(user_id), [])
        return hist[:k]

    def get_user_personality(self, user_id: str) -> str:
        return self.user_personality.get(str(user_id), "")

    def get_neighbors(self, movie_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Very light KNN using TRAIN DATA ONLY (no test contamination):
          - Jaccard similarity over genre_list
          - + small popularity prior (from train)
        
        Returns neighbors from train set that user has actually rated.
        """
        m = self.get_movie(movie_id)
        if not m:
            return []

        g0 = set(m.get("genre_list", []))
        if not g0:
            g0 = set()

        # Pull candidate set from TRAIN only
        df = self.train_df
        mask = (
            df["genre_list"].apply(lambda gl: any(g in gl for g in g0))
            if g0
            else pd.Series(True, index=df.index)
        )
        cand = (
            df.loc[mask, ["movieId", "title", "genre_list"]]
            .drop_duplicates("movieId")
            .copy()
        )

        # Compute similarity
        def jaccard(glist: List[str]) -> float:
            s = set(glist or [])
            if not g0 and not s:
                return 0.0
            inter = len(g0 & s)
            union = len(g0 | s)
            return float(inter) / float(union) if union else 0.0

        cand["sim"] = cand["genre_list"].apply(jaccard)
        cand["pop"] = cand["movieId"].map(self.movie_popularity).fillna(1.0)
        # Score = sim + 0.05 * log(pop)
        cand["score"] = cand["sim"] + 0.05 * np.log1p(cand["pop"])
        # Exclude the movie itself
        cand = cand[cand["movieId"] != str(movie_id)]
        # Top-k
        top = cand.sort_values("score", ascending=False).head(k)
        return top[["movieId", "title", "sim"]].to_dict(orient="records")
