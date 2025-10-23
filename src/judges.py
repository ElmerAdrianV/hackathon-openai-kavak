from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from .types import JudgeOutput, ContextPack, CriticOutput
from .llm_client import LLMClient, extract_json_block


def _load_system_prompt(resources_dir: str, judge_id: str) -> str:
    """
    Loads system prompt from resources/judges/{judge_id}.(txt|md)
    """
    base = os.path.join(resources_dir, "judges")
    for ext in (".txt", ".md"):
        path = os.path.join(base, judge_id + ext)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    # fallback
    return f"You are a grounded debate judge: {judge_id}. Check support and calibrate."


def _judge_user_prompt(
    critics: List[CriticOutput], ctx: ContextPack, critic_track: Dict[str, float]
) -> str:
    """
    Judge gets critics' outputs, plus simple performance priors.
    Requires strict JSON output.
    """
    packed = {
        "critics": [
            {
                "critic_id": c.critic_id,
                "score": c.score,
                "confidence": c.confidence,
                "rationale": c.rationale,
            }
            for c in critics
        ],
        "movie_profile": ctx.movie_profile,
        "user_history": ctx.user_profile.get("history", []),
        "retrieved": ctx.retrieved,
        "critic_track": critic_track,  # e.g., historical skill (can be zeros initially)
    }
    schema = (
        "{\n"
        '  "r_tilde": float,         // judge’s calibrated 0..5\n'
        '  "alphas": [float,...],    // one per critic, >=0, sum≈1\n'
        '  "flags": [0|1,...],       // one per critic (1=unsupported claims)\n'
        '  "justification": "≤20 words"\n'
        "}\n"
    )
    return (
        "Read critics’ JSON and FACTS. Flag unsupported claims. Weight reliable critics.\n"
        "Output strictly the JSON schema below (no prose):\n\n"
        f"INPUT (JSON):\n```json\n{json.dumps(packed, ensure_ascii=False)}\n```\n\n"
        f"OUTPUT SCHEMA:\n{schema}"
    )


@dataclass
class Judge:
    judge_id: str
    resources_dir: str
    llm: LLMClient
    model: str = "gpt-5"
    # EMA of negative MSE as a crude skill score
    skill_score: float = 0.0

    def evaluate(
        self,
        critics: List[CriticOutput],
        ctx: ContextPack,
        critic_track: Dict[str, float],
    ) -> JudgeOutput:
        system_prompt = _load_system_prompt(self.resources_dir, self.judge_id)
        user_prompt = _judge_user_prompt(critics, ctx, critic_track)
        txt = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            settings={"max_completion_tokens": 350},
        )
        data = extract_json_block(txt) or {}
        # Parse with safe fallbacks
        r_tilde = float(
            np.clip(
                data.get(
                    "r_tilde", np.mean([c.score for c in critics]) if critics else 3.0
                ),
                0.0,
                5.0,
            )
        )
        alphas = data.get("alphas", [])
        if (not isinstance(alphas, list)) or (len(alphas) != len(critics)):
            # fallback: proportional to critic confidence
            confs = np.array([max(1e-6, c.confidence) for c in critics], dtype=float)
            alphas = (confs / confs.sum()).tolist()
        else:
            alphas = [max(0.0, float(a)) for a in alphas]
            s = sum(alphas)
            alphas = [a / s if s > 0 else 1.0 / max(1, len(alphas)) for a in alphas]

        flags = data.get("flags", [0] * len(critics))
        if (not isinstance(flags, list)) or (len(flags) != len(critics)):
            flags = [0] * len(critics)
        flags = [1 if int(f) == 1 else 0 for f in flags]

        justification = str(data.get("justification", "weighted blend"))
        return JudgeOutput(self.judge_id, r_tilde, alphas, flags, justification)

    def update_skill(self, true_rating: float, r_tilde: float, rho: float = 0.1):
        err = (true_rating - r_tilde) ** 2
        self.skill_score = (1 - rho) * self.skill_score + rho * (-err)


class JudgePool:
    def __init__(self, judges: List[Judge]):
        self.judges = judges

    def run(
        self,
        critics: List[CriticOutput],
        ctx: ContextPack,
        judge_ids: Optional[List[str]] = None,
        critic_track: Optional[Dict[str, float]] = None,
    ) -> List[JudgeOutput]:
        critic_track = critic_track or {}
        chosen = [
            j for j in self.judges if (judge_ids is None or j.judge_id in judge_ids)
        ]
        return [j.evaluate(critics, ctx, critic_track) for j in chosen]

    def get_skill_table(self) -> Dict[str, float]:
        return {j.judge_id: j.skill_score for j in self.judges}
