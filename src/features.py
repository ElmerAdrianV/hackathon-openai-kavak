from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from .types import CriticOutput, JudgeOutput, ContextPack


def featurize(
    critics: List[CriticOutput],
    judges: List[JudgeOutput],
    ctx: ContextPack,
    judge_skill: Dict[str, float],
) -> Tuple[np.ndarray, float]:
    # Critic stats
    scores = (
        np.array([c.score for c in critics], dtype=float)
        if critics
        else np.array([0.0])
    )
    confs = (
        np.array([c.confidence for c in critics], dtype=float)
        if critics
        else np.array([0.0])
    )
    s_mean, s_std = float(scores.mean()), float(scores.std() + 1e-9)
    c_mean, c_std = float(confs.mean()), float(confs.std() + 1e-9)

    # Judge stats
    jt = (
        np.array([j.r_tilde for j in judges], dtype=float)
        if judges
        else np.array([0.0])
    )
    j_mean, j_std = float(jt.mean()), float(jt.std() + 1e-9)
    skill_feats = (
        [judge_skill.get(j.judge_id, 0.0) for j in judges] if judges else [0.0]
    )
    skill_mean = float(np.mean(skill_feats))

    # Simple genre hash as a stable scalar feature
    genre_hash = (hash(ctx.genre) % 17) / 17.0

    x = np.array(
        [s_mean, s_std, c_mean, c_std, j_mean, j_std, skill_mean, genre_hash],
        dtype=float,
    )
    disagreement = float(0.5 * (s_std + j_std))
    return x, disagreement
