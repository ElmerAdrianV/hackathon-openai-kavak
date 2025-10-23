from __future__ import annotations
from typing import Optional
from .types import ContextPack
from .data_store import DataStore


class Retriever:
    """
    Real retriever backed by a DataStore.
    """

    def __init__(self, store: DataStore):
        self.store = store

    def get_context(
        self,
        user_id: str,
        movie_id: str,
        k_history: int = 10,
        k_neighbors: int = 8,
    ) -> ContextPack:
        movie = self.store.get_movie(movie_id)
        if not movie:
            movie = {"title": f"Movie {movie_id}", "overview": "", "genre_list": []}

        # Primary genre for quick routing feature
        genre = movie.get("genre_list", [])
        primary_genre = genre[0] if genre else "Unknown"

        user_hist = self.store.get_user_history(user_id, k=k_history)
        personality = self.store.get_user_personality(user_id)

        neighbors = self.store.get_neighbors(movie_id, k=k_neighbors)

        user_profile = {
            "history": user_hist,
            "personality": personality,
        }
        movie_profile = {
            "title": movie.get("title", ""),
            "overview": movie.get("overview", ""),
            "genres": movie.get("genre_list", []),
        }
        retrieved = {"neighbors": neighbors}

        return ContextPack(
            user_id=str(user_id),
            movie_id=str(movie_id),
            genre=primary_genre,
            user_profile=user_profile,
            movie_profile=movie_profile,
            retrieved=retrieved,
        )
