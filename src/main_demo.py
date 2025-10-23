from __future__ import annotations

# If run as a script (not as a module), make the package importable
if __name__ == "__main__" and __package__ is None:
    import os, sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # After this, absolute imports like `from src...` work.

from src.orchestrator import Orchestrator, OrchestratorConfig
from src.critics import Critic
from src.judges import Judge
from src.llm_client import LLMClient


def build_system(resources_dir: str = "./resources"):
    llm = LLMClient()

    critics = [
        Critic(critic_id="cinephile", resources_dir=resources_dir, llm=llm),
        Critic(critic_id="mcu_fan", resources_dir=resources_dir, llm=llm),
        Critic(critic_id="romcom", resources_dir=resources_dir, llm=llm),
        Critic(critic_id="stats", resources_dir=resources_dir, llm=llm),
        Critic(critic_id="horrorhead", resources_dir=resources_dir, llm=llm),
    ]

    judges = [
        Judge(judge_id="grounded_v1", resources_dir=resources_dir, llm=llm),
        Judge(judge_id="strict_v1", resources_dir=resources_dir, llm=llm),
    ]

    cfg = OrchestratorConfig(
        resources_dir=resources_dir, k_critics=4, k_judges=1, calibrator_dim=8
    )
    return Orchestrator(critics, judges, cfg)


def demo_flow():
    orch = build_system()

    # 1) Predict
    yhat, sigma, aux = orch.predict(user_id="u42", movie_id="m9001")
    print(f"Prediction: {yhat:.2f} ± {sigma:.2f} | aux={aux}")

    # 2) Simulate feedback and update
    orch.online_update(true_rating=4.0)

    # 3) Predict again
    yhat2, sigma2, aux2 = orch.predict(user_id="u42", movie_id="m9002")
    print(f"Prediction: {yhat2:.2f} ± {sigma2:.2f} | aux={aux2}")


if __name__ == "__main__":
    demo_flow()
