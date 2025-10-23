# Agentic Movie Recommender — Debate + Judge (Boilerplate)

This is a **scaffold** for the Debate + Judge recommender with an **Orchestrator agent**,
**Critic tool-agents**, **Judge tool-agents**, a **Calibrator** head, and a **Router** (bandit policy).
It is intentionally minimal so you can plug in your own prompts, rules, and update logic.

## Layout
```
agentic_rec/
  ├─ docs/
      ├─
  ├─ src/
  │   ├─ orchestrator.py       # primary agent: routing, debate, aggregation, updates
  │   ├─ critics.py            # critic personae + manager
  │   ├─ judges.py             # judges + skill tracking
  │   ├─ calibrator.py         # online regressor + uncertainty head (stub)
  │   ├─ router.py             # routing/bandit policy for critics/judges
  │   ├─ retriever.py          # user/movie context retrieval (stub)
  │   ├─ features.py           # feature builders for calibrator
  │   ├─ logging_store.py      # event logging and replay helpers
  │   ├─ types.py              # dataclasses for typed IO
  │   ├─ llm_client.py         # interface to your LLM provider (stub)
  │   └─ main_demo.py          # runnable demo with dummy components
  └─ requirements.txt
```

## Quick start
1) Open `src/main_demo.py` and run it. It uses dummy critics/judges to prove the pipeline works.
2) Replace stubs in `llm_client.py` with your actual LLM calls and add real prompts in `critics.py` / `judges.py`.
3) Implement your **update loops** inside `orchestrator.py` (`online_update`, `nightly_evolution`).

## Notes
- The calibrator is a tiny online linear regressor written from scratch (no external ML deps).
- Everything is typed but simplified to keep the boilerplate portable.
- Logging writes JSONL to `./logs/events.jsonl` by default.
