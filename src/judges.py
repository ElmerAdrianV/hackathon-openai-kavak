from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass, field
import math, random

from .types import JudgeOutput, ContextPack, CriticOutput
from .llm_client import LLMClient

@dataclass
class Judge:
    judge_id: str
    style: str = "Grounded"
    llm: LLMClient = None
    # Exponentially-weighted skill score used by router/calibrator features
    skill_score: float = 0.0

    def evaluate(self, critics: List[CriticOutput], ctx: ContextPack) -> JudgeOutput:
        # TODO: Implement grounded evaluation with flags & alphas from LLM.
        # Dummy: weight higher confidence critics slightly more.
        eps = 1e-9
        confs = [max(eps, c.confidence) for c in critics]
        s = sum(confs)
        alphas = [c/s for c in confs]
        r_tilde = sum(a * c.score for a, c in zip(alphas, critics))
        flags = [0 for _ in critics]  # no flags in dummy
        justification = f"{self.style} judge blend of {len(critics)} critics."
        return JudgeOutput(self.judge_id, r_tilde, alphas, flags, justification)

    def update_skill(self, true_rating: float, r_tilde: float, rho: float = 0.1):
        err = (true_rating - r_tilde) ** 2
        # Skill increases as error decreases (negative MSE)
        self.skill_score = (1 - rho) * self.skill_score + rho * (-err)

class JudgePool:
    def __init__(self, judges: List[Judge]):
        self.judges = judges

    def run(self, critics: List[CriticOutput], ctx: ContextPack, judge_ids: List[str] = None) -> List[JudgeOutput]:
        chosen = [j for j in self.judges if (judge_ids is None or j.judge_id in judge_ids)]
        return [j.evaluate(critics, ctx) for j in chosen]

    def get_skill_table(self) -> Dict[str, float]:
        return {j.judge_id: j.skill_score for j in self.judges}
