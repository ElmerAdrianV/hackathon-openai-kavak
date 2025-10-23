from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time

@dataclass
class CriticOutput:
    critic_id: str
    score: float                # in [0,5]
    confidence: float           # in [0,1]
    rationale: str
    flags: Dict[str, Any] = field(default_factory=dict)  # optional, e.g., self-checks

@dataclass
class JudgeOutput:
    judge_id: str
    r_tilde: float              # calibrated judge rating in [0,5]
    alphas: List[float]         # weights over critics (sum to ~1)
    flags: List[int]            # 0/1 per critic, unsupported claims etc.
    justification: str

@dataclass
class ContextPack:
    user_id: str
    movie_id: str
    genre: str
    user_profile: Dict[str, Any]     # e.g., history, embeddings
    movie_profile: Dict[str, Any]    # e.g., metadata, embeddings
    retrieved: Dict[str, Any]        # slim top-K similar items, etc.

@dataclass
class EventLog:
    ts: float
    user_id: str
    movie_id: str
    context: Dict[str, Any]
    critic_outputs: List[Dict[str, Any]]
    judge_outputs: List[Dict[str, Any]]
    yhat: float
    yhat_sigma: float
    feedback: Optional[Dict[str, Any]] = None

def now_ts() -> float:
    return time.time()
