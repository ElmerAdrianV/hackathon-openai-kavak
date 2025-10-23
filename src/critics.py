from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from .types import CriticOutput, ContextPack
from .llm_client import LLMClient, extract_json_block


def _load_system_prompt(resources_dir: str, critic_id: str) -> str:
    """
    Loads the system prompt text from resources/movie_critics/{critic_id}.(txt|md)
    """
    base = os.path.join(resources_dir, "movie_critics")
    for ext in (".txt", ".md"):
        path = os.path.join(base, critic_id + ext)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    # fallback
    return f"You are a movie critic persona: {critic_id}. Be concise and grounded."


def _critic_user_prompt(ctx: ContextPack) -> str:
    """
    Builds the user prompt for a critic. We require strict JSON output.
    """
    facts = {
        "movie_profile": ctx.movie_profile,
        "user_history": ctx.user_profile.get("history", []),
        "retrieved": ctx.retrieved,
    }
    return (
        "You will predict how much the USER would like MOVIE on a 0–5 scale.\n"
        "Rules: use only the provided FACTS; do not invent details; be concise.\n\n"
        f"FACTS (JSON):\n```json\n{json.dumps(facts, ensure_ascii=False)}\n```\n\n"
        "Output JSON ONLY in the following schema (no prose):\n"
        "{\n"
        '  "score": float,           // 0..5\n'
        '  "confidence": float,      // 0..1 (self-reported)\n'
        '  "rationale": "≤25 words citing FACTS keys (e.g., user_history[0])"\n'
        "}\n"
    )


@dataclass
class Critic:
    critic_id: str
    resources_dir: str
    llm: LLMClient
    model: str = "gpt-5"

    def score(self, ctx: ContextPack) -> CriticOutput:
        system_prompt = _load_system_prompt(self.resources_dir, self.critic_id)
        user_prompt = _critic_user_prompt(ctx)
        txt = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            settings={"max_completion_tokens": 350},
        )
        data = extract_json_block(txt) or {}
        score = float(np.clip(data.get("score", 3.0), 0.0, 5.0))
        conf = float(np.clip(data.get("confidence", 0.5), 0.0, 1.0))
        rationale = str(data.get("rationale", ""))
        return CriticOutput(self.critic_id, score, conf, rationale)


class CriticManager:
    def __init__(self, critics: List[Critic]):
        self.critics = critics

    def run(
        self, ctx: ContextPack, critic_ids: Optional[List[str]] = None
    ) -> List[CriticOutput]:
        chosen = [
            c for c in self.critics if (critic_ids is None or c.critic_id in critic_ids)
        ]
        return [c.score(ctx) for c in chosen]
