from __future__ import annotations
from typing import Dict, Any
from .types import ContextPack

def retrieve_context(user_id: str, movie_id: str) -> ContextPack:
    # TODO: plug real retrieval (profiles, metadata, KNN, etc.)
    user_profile = {"history": [{"movie_id": "m123", "rating": 4.5}, {"movie_id": "m777", "rating": 2.0}]}
    movie_profile = {"title": f"Movie {movie_id}", "year": 2019, "genres": ["Drama"]}
    genre = movie_profile["genres"][0]
    retrieved = {"neighbors": [{"movie_id": "m123", "sim": 0.8}, {"movie_id": "m888", "sim": 0.7}]}
    return ContextPack(user_id=user_id, movie_id=movie_id, genre=genre,
                       user_profile=user_profile, movie_profile=movie_profile, retrieved=retrieved)
