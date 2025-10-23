from __future__ import annotations

import os
from typing import List, Optional, Dict, Any
import numpy as np

from .llm_client import LLMClient, extract_json_block
from .types import ContextPack, CriticOutput


# -------- utilities --------


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


# Appended to any critic system prompt to enforce a JSON contract.
_CRITIC_JSON_SPEC = """
Return STRICT JSON only with these exact keys:
{
  "score": number,            // 0..5 (may be fractional)
  "confidence": number,       // 0..1
  "rationale": string         // 1-3 sentences explaining why this user would or wouldn't like it
}
No extra keys. No markdown. No prose outside the JSON.
""".strip()


# -------- classes --------


class Critic:
    """
    A single persona critic. Reads a persona prompt from:
      {resources_dir}/movie_critics/{critic_id}.txt|md
    """

    def __init__(
        self, critic_id: str, resources_dir: str, llm: LLMClient, model: str = "gpt-5"
    ):
        self.critic_id = critic_id
        self.resources_dir = resources_dir
        self.llm = llm
        self.model = model
        self._system_prompt_cache: Optional[str] = None

    # ---- prompt build ----
    def _load_system_prompt(self) -> str:
        if self._system_prompt_cache is not None:
            return self._system_prompt_cache

        path = _find_persona_file(self.resources_dir, "movie_critics", self.critic_id)
        if path:
            persona = _read_text_file(path).strip()
        else:
            persona = f"You are a movie critic persona named '{self.critic_id}'. Rely only on provided context."

        # Enforce JSON response
        system_prompt = (persona + "\n\n" + _CRITIC_JSON_SPEC).strip()
        self._system_prompt_cache = system_prompt
        return system_prompt

    def _build_user_prompt(self, ctx: ContextPack) -> str:
        movie = ctx.movie_profile or {}
        up = ctx.user_profile or {}
        neighbors = (ctx.retrieved or {}).get("neighbors", [])

        lines = [
            "Context:",
            f"- User personality: {up.get('personality', '')}",
            f"- Movie title: {movie.get('title', '')}",
            f"- Movie overview: {movie.get('overview', '')}",
        ]

        genres = movie.get("genres", [])
        if isinstance(genres, list):
            lines.append(f"- Movie genres: {', '.join(genres)}")
        else:
            lines.append(f"- Movie genres: {genres}")

        lines.append("- User top history (title, rating) [up to 10]:")
        for h in (up.get("history", []) or [])[:10]:
            lines.append(f"  * {h.get('title', '?')} — {h.get('rating', '?')}")

        if neighbors:
            lines.append("- Nearest neighbor movies (title, sim) [up to 6]:")
            for n in neighbors[:6]:
                title = n.get("title", "?")
                sim = n.get("sim", 0.0)
                try:
                    sim = float(sim)
                except Exception:
                    sim = 0.0
                lines.append(f"  * {title} — sim={sim:.2f}")

        lines.append(
            "\nTask: Predict how much this user would like this movie on a 0..5 scale."
            " Provide a short rationale grounded in the context."
        )
        return "\n".join(lines)

    # ---- scoring ----
    def score(self, ctx: ContextPack) -> CriticOutput:
        system_prompt = self._load_system_prompt()
        user_prompt = self._build_user_prompt(ctx)

        raw = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            settings={"max_completion_tokens": 350},
            force_json=True,  # enforce JSON mode on supported models
        )

        data: Dict[str, Any] = extract_json_block(raw) or {}
        score = float(np.clip(data.get("score", 3.0), 0.0, 5.0))
        conf = float(np.clip(data.get("confidence", 0.5), 0.0, 1.0))
        rationale = str(data.get("rationale", "")).strip()

        out = CriticOutput(self.critic_id, score, conf, rationale)
        # Attach raw model text for verbose debug (main_demo reads flags.llm_raw)
        try:
            flags = getattr(out, "flags", None)
            if isinstance(flags, dict):
                flags["llm_raw"] = raw
                if "rationale" not in data:
                    flags["parse_note"] = "missing_rationale_in_json"
            else:
                # if CriticOutput has no flags dict, add one dynamically
                setattr(out, "flags", {"llm_raw": raw})
        except Exception:
            pass

        return out


class CriticManager:
    """
    Wraps many critics and runs a selected subset.
    """

    def __init__(self, critics: List[Critic]):
        self.critics = critics
        self.map = {c.critic_id: c for c in critics}

    def run(
        self, ctx: ContextPack, critic_ids: Optional[List[str]] = None
    ) -> List[CriticOutput]:
        chosen = [
            self.map[cid]
            for cid in (critic_ids or [c.critic_id for c in self.critics])
            if cid in self.map
        ]
        return [c.score(ctx) for c in chosen]
