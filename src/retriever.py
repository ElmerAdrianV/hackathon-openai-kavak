from __future__ import annotations
from typing import Dict, Any
from .types import ContextPack


def retrieve_context(user_id: str, movie_id: str) -> ContextPack:
    # TODO: replace with real store/embeddings/metadata
    user_profile = {
        "history": [
            {
                "movie_id": "m101",
                "title": "Indie Love",
                "rating": 4.5,
                "genres": ["Drama", "Romance"],
            },
            {
                "movie_id": "m205",
                "title": "Action Max",
                "rating": 2.0,
                "genres": ["Action"],
            },
        ]
    }
    movie_profile = {
        "title": f"Movie {movie_id}",
        "year": 2021,
        "genres": ["Drama"],
        "cast": ["A. Actor", "B. Star"],
    }
    genre = movie_profile["genres"][0] if movie_profile.get("genres") else "Unknown"
    retrieved = {
        "neighbors": [
            {"movie_id": "m101", "sim": 0.78, "title": "Indie Love"},
            {"movie_id": "m333", "sim": 0.64, "title": "Quiet Evenings"},
        ]
    }
    return ContextPack(
        user_id=user_id,
        movie_id=movie_id,
        genre=genre,
        user_profile=user_profile,
        movie_profile=movie_profile,
        retrieved=retrieved,
    )
