from __future__ import annotations
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .types import ContextPack, EventLog, now_ts, CriticOutput, JudgeOutput
from .critics import CriticManager, Critic
from .judges import JudgePool, Judge
from .calibrator import OnlineCalibrator
from .router import Router
from .features import featurize
from .logging_store import EventLogger
from .retriever import Retriever


@dataclass
class OrchestratorConfig:
    resources_dir: str = "./resources"
    k_critics: int = 4
    k_judges: int = 1
    calibrator_dim: int = 8
    lr: float = 1e-2
    l2: float = 1e-4


class Orchestrator:
    def __init__(
        self,
        critics: List[Critic],
        judges: List[Judge],
        retriever: Retriever,
        cfg: OrchestratorConfig = OrchestratorConfig(),
    ):
        self.cfg = cfg
        self.critics = CriticManager(critics)
        self.judges = JudgePool(judges)
        self.router = Router(
            [c.critic_id for c in critics],
            [j.judge_id for j in judges],
            k_critics=cfg.k_critics,
            k_judges=cfg.k_judges,
        )
        self.calibrator = OnlineCalibrator(dim=cfg.calibrator_dim, lr=cfg.lr, l2=cfg.l2)
        self.logger = EventLogger()
        self.retriever = retriever
        self.critic_track: Dict[str, float] = {c.critic_id: 0.0 for c in critics}

    def predict(
        self, user_id: str, movie_id: str
    ) -> Tuple[float, float, Dict[str, Any]]:
        ctx = self.retriever.get_context(user_id, movie_id)
        judge_skill = self.judges.get_skill_table()
        chosen_critics, chosen_judges = self.router.select(ctx.genre, judge_skill)

        critic_outs: List[CriticOutput] = self.critics.run(
            ctx, critic_ids=chosen_critics
        )
        judge_outs: List[JudgeOutput] = self.judges.run(
            critic_outs, ctx, judge_ids=chosen_judges, critic_track=self.critic_track
        )

        judge_dicts = []
        for jo in judge_outs:
            d = jo.__dict__.copy()
            # If judges.py attached 'debug_raw', keep it to inspect later
            raw = getattr(jo, "debug_raw", None)
            if raw:
                d["raw"] = raw
            judge_dicts.append(d)
        
        print("judge_dicts="+str(judge_dicts)) # debug

        # ev = EventLog(
        #     ts=now_ts(),
        #     user_id=user_id,
        #     movie_id=movie_id,
        #     context={...},
        #     critic_outputs=[c.__dict__ for c in critic_outs],
        #     judge_outputs=judge_dicts,  # <-- use judge_dicts instead of [j.__dict__ ...]
        #     yhat=yhat,
        #     yhat_sigma=sigma,
        #     feedback=None,
        # )

        x, disagreement = featurize(critic_outs, judge_outs, ctx, judge_skill)
        yhat, sigma = self.calibrator.predict(x, disagreement=disagreement)

        ev = EventLog(
            ts=now_ts(),
            user_id=user_id,
            movie_id=movie_id,
            context={
                "genre": ctx.genre,
                "movie_profile": ctx.movie_profile,
                "user_personality": ctx.user_profile.get("personality", ""),
            },
            critic_outputs=[c.__dict__ for c in critic_outs],
            judge_outputs=[j.__dict__ for j in judge_outs],
            yhat=yhat,
            yhat_sigma=sigma,
            feedback=None,
        )
        self.logger.append(ev)

        aux = {
            "chosen_critics": chosen_critics,
            "chosen_judges": chosen_judges,
            "disagreement": disagreement,
            "user_personality": ctx.user_profile.get("personality", ""),
        }
        return yhat, sigma, aux

    def online_update(self, true_rating: float):
        events = self.logger.read_all()
        if not events:
            return
        last = events[-1]

        # Rebuild light objects to featurize (we skip full retrieval here)
        ctx = ContextPack(
            user_id=last["user_id"],
            movie_id=last["movie_id"],
            genre=last["context"]["genre"],
            user_profile={"personality": last["context"].get("user_personality", "")},
            movie_profile=last["context"]["movie_profile"],
            retrieved={"neighbors": []},
        )
        critics = [CriticOutput(**co) for co in last["critic_outputs"]]
        judges = [JudgeOutput(**jo) for jo in last["judge_outputs"]]
        judge_skill = self.judges.get_skill_table()

        x, _ = featurize(critics, judges, ctx, judge_skill)
        self.calibrator.partial_fit(x, true_rating)

        for jo in judges:
            for j in self.judges.judges:
                if j.judge_id == jo.judge_id:
                    j.update_skill(true_rating, jo.r_tilde)

        for co in critics:
            err = abs(true_rating - co.score)
            self.critic_track[co.critic_id] = 0.9 * self.critic_track.get(
                co.critic_id, 0.0
            ) + 0.1 * (-err)

    def nightly_evolution(self):
        pass
