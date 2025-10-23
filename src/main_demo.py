from __future__ import annotations

# Allow "python src/main_demo.py ..." without package context
if __name__ == "__main__" and __package__ is None:
    import os, sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import ast
import pandas as pd
from typing import Dict, Any, List

from src.orchestrator import Orchestrator, OrchestratorConfig
from src.critics import Critic
from src.judges import Judge
from src.llm_client import LLMClient
from src.data_store import DataStore
from src.retriever import Retriever

# Toggle verbose debug prints (can also set env VERBOSE=1)
VERBOSE = os.getenv("VERBOSE", "1") not in ("0", "false", "False")


# --------- data loading helpers ---------
def _parse_genre_list_column(df: pd.DataFrame) -> pd.DataFrame:
    if (
        "genre_list" in df.columns
        and len(df) > 0
        and isinstance(df["genre_list"].iloc[0], str)
    ):

        def _to_list(x):
            try:
                return ast.literal_eval(x)
            except Exception:
                return []

        df["genre_list"] = df["genre_list"].apply(_to_list)
    if "personality" not in df.columns:
        df["personality"] = ""
    return df


def load_df_from_path(path: str) -> pd.DataFrame:
    """
    Load ratings+metadata dataframe from CSV or PKL.
    Required: userId, movieId, rating, title, overview, genre_list
    Optional: personality
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        return _parse_genre_list_column(df)
    if ext in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
        return _parse_genre_list_column(df)
    raise ValueError(f"Unsupported data file extension: {ext}")


def load_default_df() -> pd.DataFrame:
    here = os.path.dirname(os.path.abspath(__file__))
    cand = [
        os.path.join(here, "data", "movie_user_data.pkl"),
        os.path.join(here, "data", "movie_user_data.csv"),
    ]
    for p in cand:
        if os.path.exists(p):
            return load_df_from_path(p)
    raise FileNotFoundError(
        "No default dataset found. Provide a path argument or place a file at:\n"
        "  src/data/movie_user_data.pkl  or  src/data/movie_user_data.csv"
    )


# --------- pretty printers ---------
def _print_context_summary(ev: Dict[str, Any]) -> None:
    ctx = ev.get("context", {})
    movie = ctx.get("movie_profile", {}) or {}
    title = movie.get("title", "")
    genres = movie.get("genres", [])
    personality = ctx.get("user_personality", "")
    print("  Context:")
    print(f"    title: {title}")
    if isinstance(genres, list):
        print(f"    genres: {', '.join(genres)}")
    else:
        print(f"    genres: {genres}")
    if personality:
        print(f"    personality: {personality}")


def _print_critics(ev: Dict[str, Any]) -> List[str]:
    cos = ev.get("critic_outputs", []) or []
    print("  Critics:")
    ids = []
    if not cos:
        print("    (none)")
        return ids
    for c in cos:
        cid = c.get("critic_id", "?")
        ids.append(cid)
        score = c.get("score")
        conf = c.get("confidence")
        rationale = (c.get("rationale") or "").strip().replace("\n", " ")
        print(
            f"    - {cid:12s} | score={score:.2f} | conf={conf:.2f} | rationale: {rationale}"
        )
    return ids


def _print_judges(ev: Dict[str, Any], critic_ids: List[str]) -> None:
    jos = ev.get("judge_outputs", []) or []
    if not jos:
        print("  Judges: (none)")
        return
    print("  Judges:")
    for j in jos:
        jid = j.get("judge_id", "?")
        rtilde = j.get("r_tilde")
        just = (j.get("justification") or "").strip().replace("\n", " ")
        print(f"    - {jid:12s} | r_tilde={rtilde:.2f} | justification: {just}")
        alphas = j.get("alphas", [])
        flags = j.get("flags", [])
        # align and print per-critic alphas/flags
        if alphas and len(alphas) == len(critic_ids):
            print("      critic weights (alpha) and flags:")
            for cid, a, f in zip(
                critic_ids,
                alphas,
                flags if isinstance(flags, list) else [0] * len(alphas),
            ):
                print(f"        * {cid:12s} | alpha={float(a):.3f} | flag={int(f)}")
        else:
            print(
                "      (alphas/flags length mismatch or empty; judge fell back to defaults)"
            )


def print_verbose_from_last_log(orch: Orchestrator, header: str) -> None:
    events = orch.logger.read_all()
    if not events:
        print("  (no events logged)")
        return
    ev = events[-1]
    print(header)
    _print_context_summary(ev)
    critic_ids = _print_critics(ev)
    _print_judges(ev, critic_ids)
    print(
        f"  Final: yhat={ev.get('yhat', None)} | sigma={ev.get('yhat_sigma', None)}\n"
    )


# --------- system builder ---------
def build_system(df: pd.DataFrame, resources_dir: str = "./resources") -> Orchestrator:
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

    store = DataStore(df)
    retriever = Retriever(store)
    return Orchestrator(critics, judges, retriever, cfg)


# --------- demo flow ---------
def demo_flow(path_to_data: str | None = None):
    if path_to_data:
        df = load_df_from_path(path_to_data)
    else:
        df = load_default_df()

    orch = build_system(df)

    # pick two rows to demonstrate predict -> update -> predict
    if len(df) == 0:
        print("Dataset is empty.")
        return

    # ---- First example ----
    row = df.iloc[0]
    u = str(row["userId"])
    m = str(row["movieId"])
    print(
        f"\n[Demo] Using row 0: userId={u}, movieId={m}, title={row.get('title', '')}"
    )

    yhat, sigma, aux = orch.predict(user_id=u, movie_id=m)
    print(f"[Predict 1] -> {yhat:.2f} ± {sigma:.2f} | aux={aux}")
    if VERBOSE:
        print_verbose_from_last_log(orch, "  --- Verbose details (after Predict 1) ---")

    # feedback
    true_rating = float(row.get("rating", 3.0))
    orch.online_update(true_rating=true_rating)
    print(f"[Update] online_update with true_rating={true_rating:.2f}")

    # ---- Second example ----
    idx2 = 1 if len(df) > 1 else 0
    row2 = df.iloc[idx2]
    u2 = str(row2["userId"])
    m2 = str(row2["movieId"])
    print(
        f"\n[Demo] Using row {idx2}: userId={u2}, movieId={m2}, title={row2.get('title', '')}"
    )

    yhat2, sigma2, aux2 = orch.predict(user_id=u2, movie_id=m2)
    print(f"[Predict 2] -> {yhat2:.2f} ± {sigma2:.2f} | aux={aux2}")
    if VERBOSE:
        print_verbose_from_last_log(orch, "  --- Verbose details (after Predict 2) ---")

    true_rating2 = float(row2.get("rating", 3.0))
    orch.online_update(true_rating=true_rating2)
    print(f"[Update] online_update with true_rating={true_rating2:.2f}\n")


if __name__ == "__main__":
    # Usage:
    #   python -m src.main_demo /abs/path/to/your.csv
    #   python -m src.main_demo /abs/path/to/your.pkl
    #   python -m src.main_demo               # looks under src/data/
    arg_path = sys.argv[1] if len(sys.argv) > 1 else None
    demo_flow(arg_path)
