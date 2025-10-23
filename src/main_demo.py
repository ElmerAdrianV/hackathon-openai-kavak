from __future__ import annotations

# Allow "python src/main_demo.py ..." without package context
if __name__ == "__main__" and __package__ is None:
    import os, sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import ast
import pandas as pd

from src.orchestrator import Orchestrator, OrchestratorConfig
from src.critics import Critic
from src.judges import Judge
from src.llm_client import LLMClient
from src.data_store import DataStore
from src.retriever import Retriever


# --------- data loading helpers ---------
def load_df_from_path(path: str) -> pd.DataFrame:
    """
    Load your ratings+metadata dataframe from CSV or pickle.
    Required columns: userId, movieId, rating, title, overview, genre_list
    Optional: personality
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    elif ext in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported data file extension: {ext}")

    # Convert genre_list strings like "['Drama','Romance']" into lists
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

    # Ensure required columns exist
    needed = ["userId", "movieId", "rating", "title", "overview", "genre_list"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Optional column: personality
    if "personality" not in df.columns:
        df["personality"] = ""

    return df


def load_default_df() -> pd.DataFrame:
    """
    Try a sensible default path:
      - src/data/movie_user_data.pkl
      - or src/data/movie_user_data.csv
    """
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


# --------- system builder ---------
def build_system(df: pd.DataFrame, resources_dir: str = "./resources") -> Orchestrator:
    llm = LLMClient()

    # You have cinephile + grounded_v1 prompts already; others will use fallback system prompts if files are missing.
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

    row = df.iloc[0]
    u = str(row["userId"])
    m = str(row["movieId"])
    print(
        f"\n[Demo] Using row 0: userId={u}, movieId={m}, title={row.get('title', '')}"
    )

    yhat, sigma, aux = orch.predict(user_id=u, movie_id=m)
    print(f"[Predict 1] -> {yhat:.2f} ± {sigma:.2f} | aux={aux}")

    # if rating exists, feed it as feedback
    true_rating = float(row.get("rating", 3.0))
    orch.online_update(true_rating=true_rating)
    print(f"[Update] online_update with true_rating={true_rating:.2f}")

    # second example: prefer same user with a different movie if possible
    # else, just take next row
    idx2 = 1 if len(df) > 1 else 0
    row2 = df.iloc[idx2]
    u2 = str(row2["userId"])
    m2 = str(row2["movieId"])
    print(
        f"\n[Demo] Using row {idx2}: userId={u2}, movieId={m2}, title={row2.get('title', '')}"
    )

    yhat2, sigma2, aux2 = orch.predict(user_id=u2, movie_id=m2)
    print(f"[Predict 2] -> {yhat2:.2f} ± {sigma2:.2f} | aux={aux2}")

    # Optional: update again
    true_rating2 = float(row2.get("rating", 3.0))
    orch.online_update(true_rating=true_rating2)
    print(f"[Update] online_update with true_rating={true_rating2:.2f}\n")


if __name__ == "__main__":
    # Usage:
    #   python -m src.main_demo /abs/path/to/your.csv
    # or:
    #   python src/main_demo.py /abs/path/to/your.pkl
    # or (with default data under src/data/):
    #   python -m src.main_demo
    arg_path = (
        "/Users/alef/Documents/GitHub/hackathon-openai-kavak/src/data/user_1_data.csv"
    )
    demo_flow(arg_path)
