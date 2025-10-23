from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class DataStore:
    """
    Thin wrapper over your ratings+metadata dataframe.

    Required columns:
      - userId, movieId, rating
      - title, overview
      - genre_list  (list[str] per row)
      - personality (string; may repeat per user)
    """

    def __init__(self, df: pd.DataFrame):
        # Basic cleanup / normalization
        needed = ["userId", "movieId", "rating", "title", "overview", "genre_list"]
        for col in needed:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        self.df = df.copy()
        # If personality column is missing, add a default
        if "personality" not in self.df.columns:
            self.df["personality"] = ""

        # Normalize types
        self.df["userId"] = self.df["userId"].astype(str)
        self.df["movieId"] = self.df["movieId"].astype(str)
        # Ensure genre_list is a list[str]
        self.df["genre_list"] = self.df["genre_list"].apply(
            lambda g: g if isinstance(g, list) else []
        )

        # Deduplicate (userId, movieId) keeping the last occurrence
        self.df = (
            self.df.sort_index()
            .drop_duplicates(subset=["userId", "movieId"], keep="last")
            .reset_index(drop=True)
        )

        # Precompute per-movie metadata
        movie_cols = ["movieId", "title", "overview", "genre_list"]
        # If multiple rows per movieId remain (different users), keep the first
        self.movies = (
            self.df[movie_cols]
            .drop_duplicates(subset=["movieId"], keep="first")
            .set_index("movieId")
            .to_dict(orient="index")
        )

        # Basic popularity proxy = count of ratings per movie
        counts = self.df.groupby("movieId")["rating"].size().rename("count").to_frame()
        self.movie_popularity = counts["count"].to_dict()

        # Per-user aggregates for quick access
        # user history sorted by rating desc, then arbitrary
        self.user_histories = (
            self.df.sort_values(["userId", "rating"], ascending=[True, False])
            .groupby("userId")
            .apply(
                lambda g: g[["movieId", "rating", "title", "genre_list"]].to_dict(
                    orient="records"
                )
            )
            .to_dict()
        )

        # Per-user "personality": take the *most frequent* non-empty personality string
        def _mode_non_empty(vals: List[str]) -> str:
            vals = [v for v in vals if isinstance(v, str) and v.strip()]
            if not vals:
                return ""
            # mode
            uniq, counts = np.unique(vals, return_counts=True)
            return uniq[int(np.argmax(counts))]

        per_user_pers = (
            self.df.groupby("userId")["personality"]
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
        Very light KNN:
          - Jaccard similarity over genre_list
          - + small popularity prior
        """
        m = self.get_movie(movie_id)
        if not m:
            return []

        g0 = set(m.get("genre_list", []))
        if not g0:
            g0 = set()

        # Pull a small candidate set by shared genres to stay fast
        # (You could speed this further by pre-indexing by genre.)
        df = self.df
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
