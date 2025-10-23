from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass
import random

from .types import CriticOutput, ContextPack
from .llm_client import LLMClient

@dataclass
class Critic:
    critic_id: str
    persona: str
    prompt_template: str = "You are {persona}. Given CONTEXT, predict 0-5 score."
    llm: LLMClient = None

    def score(self, ctx: ContextPack) -> CriticOutput:
        # TODO: Build a real prompt using ctx + persona; call LLM for JSON output.
        # Dummy implementation for pipeline connectivity:
        rnd = random.Random(f"{self.critic_id}-{ctx.user_id}-{ctx.movie_id}")
        score = max(0.0, min(5.0, rnd.uniform(2.0, 4.5)))
        conf = rnd.uniform(0.3, 0.9)
        rationale = f"{self.persona} rationale about {ctx.movie_id} for {ctx.user_id}."
        return CriticOutput(self.critic_id, score, conf, rationale)

class CriticManager:
    def __init__(self, critics: List[Critic]):
        self.critics = critics

    def run(self, ctx: ContextPack, critic_ids: List[str] = None) -> List[CriticOutput]:
        chosen = [c for c in self.critics if (critic_ids is None or c.critic_id in critic_ids)]
        return [c.score(ctx) for c in chosen]
