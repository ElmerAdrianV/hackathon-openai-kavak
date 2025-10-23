from __future__ import annotations

import os
from typing import List, Optional, Dict, Any
import numpy as np

from .llm_client import LLMClient, extract_json_block
from .types import ContextPack, CriticOutput, JudgeOutput


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _find_persona_file(resources_dir: str, subdir: str, agent_id: str) -> Optional[str]:
    base = os.path.join(resources_dir, subdir, agent_id)
    for ext in (".txt", ".md"):
        p = base + ext
        if os.path.exists(p):
            return p
    return None


# Appended to any judge system prompt to enforce a JSON contract.
_JUDGE_JSON_SPEC = """
Return STRICT JSON only with these exact keys:
{
  "r_tilde": number,            // calibrated score 0..5 from the debate
  "alphas": [number, ...],      // weights for each critic, length MUST = number of critics, sums ~ 1
  "flags": [0 or 1, ...],       // 1 if the critic made unsupported claims, else 0, length MUST = number of critics
  "justification": string       // brief explanation of weighting and any penalties
}
No extra keys. No markdown. No prose outside the JSON.
""".strip()


class Judge:
    """
    A single judge persona. Reads prompt from:
      {resources_dir}/judges/{judge_id}.txt|md
    """

    def __init__(
        self, judge_id: str, resources_dir: str, llm: LLMClient, model: str = "gpt-5"
    ):
        self.judge_id = judge_id
        self.resources_dir = resources_dir
        self.llm = llm
        self.model = model
        self.skill: float = 0.0  # EMA skill used by router
        self._system_prompt_cache: Optional[str] = None

    # ---- skill updates ----
    def update_skill(self, true_rating: float, judge_pred: float, rho: float = 0.1):
        # negative MSE as a score -> higher is better
        err2 = float((true_rating - judge_pred) ** 2)
        self.skill = (1.0 - rho) * self.skill + rho * (-err2)

    def get_skill(self) -> float:
        return self.skill

    # ---- prompt build ----
    def _load_system_prompt(self) -> str:
        if self._system_prompt_cache is not None:
            return self._system_prompt_cache

        path = _find_persona_file(self.resources_dir, "judges", self.judge_id)
        if path:
            persona = _read_text_file(path).strip()
        else:
            persona = f"You are a grounded debate judge named '{self.judge_id}'. Verify claims; penalize unsupported ones."

        system_prompt = (persona + "\n\n" + _JUDGE_JSON_SPEC).strip()
        self._system_prompt_cache = system_prompt
        return system_prompt

    def _build_user_prompt(
        self,
        critics: List[CriticOutput],
        ctx: ContextPack,
        critic_track: Dict[str, float],
    ) -> str:
        movie = ctx.movie_profile or {}
        up = ctx.user_profile or {}
        lines = [
            "Context:",
            f"- User personality: {up.get('personality', '')}",
            f"- Movie title: {movie.get('title', '')}",
        ]

        genres = movie.get("genres", [])
        if isinstance(genres, list):
            lines.append(f"- Movie genres: {', '.join(genres)}")
        else:
            lines.append(f"- Movie genres: {genres}")

        lines.append("- Critics (id, score, confidence, rationale) in debate order:")
        for c in critics:
            lines.append(
                f"  * {c.critic_id} | s={c.score:.2f} | q={c.confidence:.2f} | rationale: {(c.rationale or '').strip()} | track={critic_track.get(c.critic_id, 0.0):.3f}"
            )

        lines.append(
            "\nTask: Produce a calibrated score r_tilde in 0..5."
            " Choose alphas (weights) for each critic (length must equal number of critics, sum ~ 1)."
            " flags[i]=1 if critic i used unsupported claims; else 0."
            " Explain briefly in justification."
        )
        return "\n".join(lines)

    # ---- judging ----
    def evaluate(
        self,
        critics: List[CriticOutput],
        ctx: ContextPack,
        critic_track: Dict[str, float],
    ) -> JudgeOutput:
        system_prompt = self._load_system_prompt()
        user_prompt = self._build_user_prompt(critics, ctx, critic_track)

        raw = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            settings={"max_completion_tokens": 350},
            force_json=True,
        )

        data: Dict[str, Any] = extract_json_block(raw) or {}

        # r_tilde
        r_tilde = float(
            np.clip(
                data.get(
                    "r_tilde", np.mean([c.score for c in critics]) if critics else 3.0
                ),
                0.0,
                5.0,
            )
        )

        # alphas
        alphas = data.get("alphas", [])
        if not isinstance(alphas, list) or len(alphas) != len(critics):
            # fallback: proportional to critic confidence, with tiny smoothing
            qs = np.array(
                [max(1e-3, float(c.confidence)) for c in critics], dtype=float
            )
            alphas = (
                (qs / qs.sum()).tolist()
                if qs.sum() > 0
                else [1.0 / max(1, len(critics))] * len(critics)
            )
        else:
            try:
                arr = np.array([float(a) for a in alphas], dtype=float)
                s = arr.sum()
                if s <= 0:
                    arr = np.ones_like(arr) / len(arr)
                else:
                    arr = arr / s
                alphas = arr.tolist()
            except Exception:
                alphas = [1.0 / max(1, len(critics))] * len(critics)

        # flags
        flags = data.get("flags", [])
        if not isinstance(flags, list) or len(flags) != len(critics):
            flags = [0] * len(critics)
        else:
            try:
                flags = [int(bool(f)) for f in flags]
            except Exception:
                flags = [0] * len(critics)

        justification = str(data.get("justification", "")).strip()

        jo = JudgeOutput(self.judge_id, r_tilde, alphas, flags, justification)

        # Attach raw for verbose logging (main_demo can print if orchestrator logs jo.__dict__)
        try:
            setattr(jo, "debug_raw", raw)
        except Exception:
            pass

        return jo


class JudgePool:
    """
    Wraps many judges and exposes:
      - run(...): evaluate with a selected subset of judges
      - get_skill_table(): map judge_id -> skill
    """

    def __init__(self, judges: List[Judge]):
        self.judges = judges
        self.map = {j.judge_id: j for j in judges}

    def run(
        self,
        critics: List[CriticOutput],
        ctx: ContextPack,
        judge_ids: Optional[List[str]] = None,
        critic_track: Optional[Dict[str, float]] = None,
    ) -> List[JudgeOutput]:
        chosen = [
            self.map[jid]
            for jid in (judge_ids or [j.judge_id for j in self.judges])
            if jid in self.map
        ]
        track = critic_track or {}
        outs: List[JudgeOutput] = []
        for j in chosen:
            out = j.evaluate(critics, ctx, track)
            outs.append(out)
        return outs

    def get_skill_table(self) -> Dict[str, float]:
        return {j.judge_id: j.get_skill() for j in self.judges}
