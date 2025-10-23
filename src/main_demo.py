from __future__ import annotations

# Allow "python src/main_demo.py ..." without package context
if __name__ == "__main__" and __package__ is None:
    import os, sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import glob
import ast
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

from src.orchestrator import Orchestrator, OrchestratorConfig
from src.critics import Critic
from src.judges import Judge
from src.llm_client import LLMClient
from src.data_store import DataStore
from src.retriever import Retriever

VERBOSE = os.getenv("VERBOSE", "1") not in ("0", "false", "False")


# ---------------- data loading helpers ----------------
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
    Load DataFrame from path. If path is relative, resolve it relative to src/ directory.
    """
    p = Path(path)
    if not p.is_absolute():
        # Resolve relative to src/ directory
        src_dir = Path(__file__).resolve().parent
        p = (src_dir / path).resolve()
    
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    
    ext = p.suffix.lower()
    if ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(str(p), sep=sep)
        return _parse_genre_list_column(df)
    if ext in {".pkl", ".pickle"}:
        df = pd.read_pickle(str(p))
        return _parse_genre_list_column(df)
    raise ValueError(f"Unsupported data file extension: {ext}")


def load_default_df() -> pd.DataFrame:
    """
    Load default dataset from src/data/ directory relative to this script.
    Tries multiple file patterns in order of preference.
    """
    here = Path(__file__).resolve().parent  # src/ directory
    cand = [
        here / "data" / "user_1_data.csv",
        here / "data" / "movie_user_data.pkl",
        here / "data" / "movie_user_data.csv",
    ]
    for p in cand:
        if p.exists():
            print(f"[Init] Loading default data from: {p}")
            return load_df_from_path(str(p))
    raise FileNotFoundError(
        "No default dataset found. Provide a path argument or place a file at:\n"
        "  src/data/user_1_data.csv, src/data/movie_user_data.pkl, or src/data/movie_user_data.csv"
    )


# ---------------- discovery helpers ----------------
def _discover_ids_under(dir_path: Path) -> List[str]:
    ids: List[str] = []
    for pattern in ("*.txt", "*.md"):
        for p in dir_path.glob(pattern):
            base = p.stem
            if base not in ids:
                ids.append(base)
    ids.sort()
    return ids


def discover_critics(resources_dir: Path) -> List[str]:
    mc_dir = resources_dir / "movie_critics"
    if not mc_dir.is_dir():
        return []
    return _discover_ids_under(mc_dir)


def discover_judges(resources_dir: Path) -> List[str]:
    j_dir = resources_dir / "judges"
    if not j_dir.is_dir():
        return []
    return _discover_ids_under(j_dir)


# ---------------- pretty printers ----------------
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
        score = float(c.get("score", 0.0))
        conf = float(c.get("confidence", 0.0))
        rationale = (c.get("rationale") or "").strip().replace("\n", " ")
        print(
            f"    - {cid:12s} | score={score:.2f} | conf={conf:.2f} | rationale: {rationale}"
        )
        raw = (c.get("flags", {}) or {}).get("llm_raw")
        if raw:
            print(
                "      raw:",
                raw[:300].replace("\n", " "),
                "..." if len(raw) > 300 else "",
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
        rtilde = float(j.get("r_tilde", 0.0))
        just = (j.get("justification") or "").strip().replace("\n", " ")
        print(f"    - {jid:12s} | r_tilde={rtilde:.2f} | justification: {just}")
        alphas = j.get("alphas", [])
        flags = j.get("flags", [])
        if alphas and len(alphas) == len(critic_ids):
            print("      critic weights (alpha) and flags:")
            for cid, a, f in zip(
                critic_ids,
                alphas,
                flags if isinstance(flags, list) else [0] * len(alphas),
            ):
                try:
                    a_f = float(a)
                except:
                    a_f = 0.0
                try:
                    f_i = int(f)
                except:
                    f_i = 0
                print(f"        * {cid:12s} | alpha={a_f:.3f} | flag={f_i}")
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


# ---------------- system builder ----------------
def resolve_resources_dir(cli_override: str | None = None) -> Path:
    """
    Always resolve resources directory relative to the project structure.
    Default: <project_root>/src/resources (project_root = parent of src/)
    CLI override is still supported but converted to absolute path relative to project root.
    """
    project_root = Path(__file__).resolve().parent.parent  # repo root (contains src/)
    
    if cli_override:
        # If it's an absolute path, use it; otherwise resolve relative to project root
        p = Path(cli_override)
        if p.is_absolute():
            return p.resolve()
        return (project_root / cli_override).resolve()
    
    # Default: use src/resources (not resources at root level)
    return (project_root / "src" / "resources").resolve()


def debug_list_dir(path: Path, label: str):
    try:
        print(f"[Debug] Listing {label} at: {path}")
        if not path.exists():
            print("  (path does not exist)")
            return
        if not path.is_dir():
            print("  (path exists but is not a directory)")
            return
        for entry in sorted(path.iterdir()):
            print("  -", entry.name)
    except Exception as e:
        print(f"[Debug] Could not list {label}: {e}")


def build_system(df: pd.DataFrame, resources_dir: str | None = None) -> Orchestrator:
    resources_path = resolve_resources_dir(resources_dir)
    print(f"[Init] resources_dir resolved to: {resources_path}")

    mc_dir = resources_path / "movie_critics"
    j_dir = resources_path / "judges"

    critic_ids = discover_critics(resources_path)
    judge_ids = discover_judges(resources_path)

    if not critic_ids:
        debug_list_dir(resources_path, "resources_dir")
        debug_list_dir(mc_dir, "movie_critics")
        raise RuntimeError(
            f"No critic prompts found under {mc_dir}. "
            f"Add files like 'cinephile.txt' there."
        )
    if not judge_ids:
        debug_list_dir(resources_path, "resources_dir")
        debug_list_dir(j_dir, "judges")
        raise RuntimeError(
            f"No judge prompts found under {j_dir}. "
            f"Add files like 'grounded_v1.txt' there."
        )

    llm = LLMClient()

    # Optional ordering preference
    preferred_critics = ["cinephile", "romcom", "stats", "mcu_fan", "horrorhead"]
    ordered_critics = [c for c in preferred_critics if c in critic_ids] + [
        c for c in critic_ids if c not in preferred_critics
    ]

    critics = [
        Critic(critic_id=cid, resources_dir=str(resources_path), llm=llm)
        for cid in ordered_critics
    ]
    judges = [
        Judge(judge_id=jid, resources_dir=str(resources_path), llm=llm)
        for jid in judge_ids
    ]

    print(f"[Init] Loaded critics: {', '.join([c.critic_id for c in critics])}")
    print(f"[Init] Loaded judges : {', '.join([j.judge_id  for j in judges ])}")

    cfg = OrchestratorConfig(
        resources_dir=str(resources_path),
        k_critics=min(4, len(critics)),
        k_judges=1,
        calibrator_dim=8,
    )

    store = DataStore(df)
    retriever = Retriever(store)
    return Orchestrator(critics, judges, retriever, cfg)


# ---------------- demo flow ----------------
def demo_flow(path_to_data: str | None = None, resources_dir: str | None = None):
    if path_to_data:
        df = load_df_from_path(path_to_data)
    else:
        df = load_default_df()

    orch = build_system(df, resources_dir=resources_dir)

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
    # CLI:
    #   python -m src.main_demo /abs/path/to/your.csv
    #   python -m src.main_demo --resources /abs/path/to/resources  /abs/path/to/your.csv
    args = sys.argv[1:]
    res_arg = None
    data_arg = None
    if args:
        if args[0] == "--resources":
            if len(args) < 2:
                print(
                    "Usage: python -m src.main_demo --resources /path/to/resources  [data_path]"
                )
                sys.exit(1)
            res_arg = args[1]
            data_arg = args[2] if len(args) > 2 else None
        else:
            data_arg = args[0]
    demo_flow(data_arg, resources_dir=res_arg)
