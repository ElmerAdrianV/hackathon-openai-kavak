from __future__ import annotations
from typing import Dict, List, Tuple
import random


class Router:
    """
    Routing policy over critics/judges. MVP:
      - critics: uniform random top-k
      - judges: top-1 by skill
    Replace with LinUCB/Thompson later.
    """

    def __init__(
        self,
        critic_ids: List[str],
        judge_ids: List[str],
        k_critics: int = 4,
        k_judges: int = 1,
    ):
        self.critic_ids = critic_ids
        self.judge_ids = judge_ids
        self.k_critics = k_critics
        self.k_judges = k_judges

    def select(
        self, genre: str, judge_skill: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        chosen_critics = random.sample(
            self.critic_ids, k=min(self.k_critics, len(self.critic_ids))
        )
        ranked = sorted(
            self.judge_ids, key=lambda j: judge_skill.get(j, 0.0), reverse=True
        )
        chosen_judges = (
            ranked[: self.k_judges] if ranked else self.judge_ids[: self.k_judges]
        )
        return chosen_critics, chosen_judges
