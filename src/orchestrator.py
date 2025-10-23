from __future__ import annotations
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .types import ContextPack, EventLog, now_ts
from .critics import CriticManager, Critic
from .judges import JudgePool, Judge
from .calibrator import OnlineCalibrator
from .router import Router
from .retriever import retrieve_context
from .features import featurize
from .logging_store import EventLogger

@dataclass
class OrchestratorConfig:
    k_critics: int = 4
    k_judges: int = 1
    calibrator_dim: int = 8   # must match features.featurize output size
    lr: float = 1e-2
    l2: float = 1e-4

class Orchestrator:
    def __init__(self, critics: List[Critic], judges: List[Judge], cfg: OrchestratorConfig = OrchestratorConfig()):
        self.critics = CriticManager(critics)
        self.judges = JudgePool(judges)
        self.router = Router([c.critic_id for c in critics], [j.judge_id for j in judges],
                             k_critics=cfg.k_critics, k_judges=cfg.k_judges)
        self.calibrator = OnlineCalibrator(dim=cfg.calibrator_dim, lr=cfg.lr, l2=cfg.l2)
        self.logger = EventLogger()

    def predict(self, user_id: str, movie_id: str) -> Tuple[float, float, Dict[str, Any]]:
        ctx = retrieve_context(user_id, movie_id)
        judge_skill = self.judges.get_skill_table()
        chosen_critics, chosen_judges = self.router.select(ctx.genre, judge_skill)

        critic_outs = self.critics.run(ctx, critic_ids=chosen_critics)
        judge_outs = self.judges.run(critic_outs, ctx, judge_ids=chosen_judges)

        x, disagree = featurize(critic_outs, judge_outs, ctx, judge_skill)
        yhat, sigma = self.calibrator.predict(x, disagreement=disagree)

        # Log without feedback yet
        ev = EventLog(ts=now_ts(), user_id=user_id, movie_id=movie_id,
                      context={"genre": ctx.genre},
                      critic_outputs=[c.__dict__ for c in critic_outs],
                      judge_outputs=[j.__dict__ for j in judge_outs],
                      yhat=yhat, yhat_sigma=sigma, feedback=None)
        self.logger.append(ev)

        aux = {"chosen_critics": chosen_critics, "chosen_judges": chosen_judges, "disagreement": disagree}
        return yhat, sigma, aux

    def online_update(self, true_rating: float):
        """Update calibrator and judge skills using the last logged event (simple demo).
        In a real system, you’d join by (user,movie,timestamp) or use a message queue.
        """
        events = self.logger.read_all()
        if not events: 
            return
        last = events[-1]
        # rebuild x features from logged structures (here we recompute cheaply)
        # In production, store features in the log to avoid drift.
        # We'll approximate by using the same featurizer again.
        # Re-instantiate lightweight objects:
        from .types import CriticOutput, JudgeOutput, ContextPack
        ctx = ContextPack(user_id=last["user_id"], movie_id=last["movie_id"],
                          genre=last["context"]["genre"], user_profile={}, movie_profile={}, retrieved={})
        critics = [CriticOutput(**co) for co in last["critic_outputs"]]
        judges = [JudgeOutput(**jo) for jo in last["judge_outputs"]]
        judge_skill = self.judges.get_skill_table()

        x, _ = featurize(critics, judges, ctx, judge_skill)
        self.calibrator.partial_fit(x, true_rating)

        # update judge skills
        for jo in judges:
            # find the judge object
            for j in self.judges.judges:
                if j.judge_id == jo["judge_id"] if isinstance(jo, dict) else (j.judge_id == jo.judge_id):
                    r_tilde = jo["r_tilde"] if isinstance(jo, dict) else jo.r_tilde
                    j.update_skill(true_rating, r_tilde)

    def nightly_evolution(self):
        """Placeholder for prompt evolution/offline replay selection.
        Implement:
          - identify weak judges/critics by skill/utility
          - mutate prompts, run offline replay, promote winners
        """
        pass
