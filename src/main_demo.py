from __future__ import annotations
from .orchestrator import Orchestrator, OrchestratorConfig
from .critics import Critic
from .judges import Judge
from .llm_client import LLMClient

def build_demo_system():
    llm = LLMClient()

    critics = [
        Critic(critic_id="cinephile", persona="Arthouse Cinephile", llm=llm),
        Critic(critic_id="mcu_fan", persona="MCU Enjoyer", llm=llm),
        Critic(critic_id="romcom", persona="Middle-aged Rom-Com Fan", llm=llm),
        Critic(critic_id="stats", persona="Data-Driven Analyst", llm=llm),
        Critic(critic_id="horrorhead", persona="Horror Specialist", llm=llm),
    ]

    judges = [
        Judge(judge_id="grounded_v1", style="Grounded"),
        Judge(judge_id="strict_v1", style="Strict Grounding"),
    ]

    cfg = OrchestratorConfig(k_critics=4, k_judges=1, calibrator_dim=8)
    return Orchestrator(critics, judges, cfg)

def demo_flow():
    orch = build_demo_system()

    # Inference (no real LLMs yet; dummy outputs)
    yhat, sigma, aux = orch.predict(user_id="u42", movie_id="m9001")
    print(f"Prediction: {yhat:.2f} ± {sigma:.2f}  | aux={aux}")

    # Simulate feedback (pretend true rating is 4.0)
    orch.online_update(true_rating=4.0)

    # Another prediction after update
    yhat2, sigma2, aux2 = orch.predict(user_id="u42", movie_id="m9002")
    print(f"Prediction: {yhat2:.2f} ± {sigma2:.2f}  | aux={aux2}")

if __name__ == "__main__":
    demo_flow()
