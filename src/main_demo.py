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
from src.reviewer import Reviewer
from src.prediction_logger import PredictionLogger

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
    
    Returns the train dataframe for backward compatibility with single-df mode.
    """
    here = Path(__file__).resolve().parent  # src/ directory
    cand = [
        here / "data" / "splits" / "user_45811_test.csv"
    ]
    for p in cand:
        if p.exists():
            print(f"[Init] Loading default data from: {p}")
            return load_df_from_path(str(p))
    raise FileNotFoundError(
        "No default dataset found. Provide a path argument or place a file at:\n"
        "  src/data/splits/user_45811_test.csv, src/data/user_1_data.csv, \n"
        "  src/data/movie_user_data.pkl, or src/data/movie_user_data.csv"
    )


def load_train_test_split() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load train/test split if available, otherwise return (train, None).
    Looks for:
      - src/data/splits/user_45811_test.csv + user_45811_train.csv
      - Falls back to single dataset as train-only
    """
    here = Path(__file__).resolve().parent
    train_path = here / "data" / "splits" / "user_45811_test.csv"
    test_path = here / "data" / "splits" / "user_45811_train.csv"
    
    if train_path.exists() and test_path.exists():
        print(f"[Init] Loading train/test split:")
        print(f"  Train: {train_path}")
        print(f"  Test:  {test_path}")
        train_df = load_df_from_path(str(train_path))
        test_df = load_df_from_path(str(test_path))
        return train_df, test_df
    
    # Fallback: single dataset as train
    print("[Init] No train/test split found, using default dataset as train-only")
    train_df = load_default_df()
    return train_df, None


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


def build_system(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    resources_dir: str | None = None,
) -> Orchestrator:
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
        k_critics=min(2, len(critics)),
        k_judges=1,
        calibrator_dim=8,
    )

    # Build DataStore with train/test split
    store = DataStore(train_df, test_df)
    retriever = Retriever(store)
    return Orchestrator(critics, judges, retriever, cfg)


# ---------------- demo flow ----------------
def demo_flow(resources_dir: str | None = None, n_samples: int = 10, review_interval: int = 5):
    """
    Demo flow with train/test split support.
    
    Iterates over multiple movies from the test set, showing how the system
    learns and improves over time.
    
    Args:
        resources_dir: Path to resources directory
        n_samples: Number of test samples to evaluate (default: 10)
        review_interval: How often to run Reviewer analysis (default: 5)
    """

    # Try to load train/test split
    train_df, test_df = load_train_test_split()
    eval_df = test_df if test_df is not None else train_df
    if test_df is not None:
        print("[Demo] Train/Test split loaded: evaluating on test set (no contamination)")
    else:
        print("[Demo] No test set: evaluating on train data (demo mode)")

    orch = build_system(train_df, test_df, resources_dir=resources_dir)
    
    # Get resources path and LLM for Reviewer
    resources_path = resolve_resources_dir(resources_dir)
    llm = LLMClient()
    
    # Initialize Reviewer with capabilities to improve judges
    reviewer = Reviewer(
        review_interval=review_interval,
        resources_dir=str(resources_path),
        llm=llm
    )
    
    # Initialize Prediction Logger
    pred_logger = PredictionLogger()

    if len(eval_df) == 0:
        print("Evaluation dataset is empty.")
        return

    # Limit samples to available data
    n_samples = min(n_samples, len(eval_df))
    print(f"\n{'='*80}")
    print(f"🎬 Evaluando {n_samples} películas para demostrar el aprendizaje del sistema")
    print(f"{'='*80}\n")

    errors = []
    predictions = []
    
    # Iterate over multiple samples
    for idx in range(n_samples):
        row = eval_df.iloc[idx]
        u = str(row["userId"])
        m = str(row["movieId"])
        title = row.get('title', 'Unknown')
        true_rating = float(row.get("rating", 3.0))
        
        # Get genres if available
        genres_list = row.get('genre_list', [])
        if isinstance(genres_list, list):
            genres = ', '.join(genres_list)
        else:
            genres = str(genres_list) if genres_list else ''
        
        # Make prediction
        yhat, sigma, aux = orch.predict(user_id=u, movie_id=m)
        error = abs(yhat - true_rating)
        errors.append(error)
        predictions.append(yhat)
        
        # Log to CSV file
        pred_logger.log_prediction(
            movie_id=m,
            movie_title=title,
            user_id=u,
            predicted_rating=yhat,
            predicted_sigma=sigma,
            true_rating=true_rating,
            genres=genres
        )
        
        # Show progress
        print(f"[{idx+1}/{n_samples}] {title[:40]:40s} | Pred: {yhat:.2f} ± {sigma:.2f} | Real: {true_rating:.2f} | Error: {error:.2f}")
        
        # Record prediction for reviewer analysis
        events = orch.logger.read_all()
        if events:
            last_event = events[-1]
            judge_outputs = last_event.get('judge_outputs', [])
            critic_outputs = last_event.get('critic_outputs', [])
            critic_ids = [c.get('critic_id') for c in critic_outputs] if critic_outputs else []
            reviewer.record_prediction(judge_outputs, true_rating, critic_ids)
        
        # Update system with ground truth
        orch.online_update(true_rating=true_rating)
        
        # Check if Reviewer should run
        if reviewer.should_review():
            report = reviewer.run_review()
            reviewer.print_report(report)
            
            # Automatically improve worst performing judge
            improvement_info = reviewer.improve_worst_judge(report)
            if improvement_info:
                print(f"\n{'='*80}")
                print(f"🔄 JUDGE IMPROVEMENT")
                print(f"{'='*80}")
                print(f"Replaced: {improvement_info['original_judge']}")
                print(f"New version: {improvement_info['new_judge']}")
                print(f"Reason: {improvement_info['reason']}")
                print(f"{'='*80}\n")
            
            # Check for calibrator suggestions
            suggestions = reviewer.suggest_calibrator_update()
            if suggestions:
                print(f"🔧 Calibrator Update Suggestions: {suggestions}")
                print()
        
        # Show learning progress every 3 samples
        elif (idx + 1) % 3 == 0 or idx == n_samples - 1:
            recent_errors = errors[max(0, idx-2):idx+1]
            avg_recent_error = sum(recent_errors) / len(recent_errors)
            print(f"  📊 Error promedio últimas {len(recent_errors)} predicciones: {avg_recent_error:.3f}")
            if idx >= 3:
                prev_errors = errors[max(0, idx-5):max(1, idx-2)]
                if prev_errors:
                    avg_prev_error = sum(prev_errors) / len(prev_errors)
                    improvement = avg_prev_error - avg_recent_error
                    symbol = "📈" if improvement > 0 else "📉"
                    print(f"  {symbol} Mejora: {improvement:+.3f} (calibrando críticos y jueces)")
            print()
    
    # Final summary
    avg_error = sum(errors) / len(errors)
    first_half_error = sum(errors[:n_samples//2]) / max(1, n_samples//2)
    second_half_error = sum(errors[n_samples//2:]) / max(1, len(errors[n_samples//2:]))
    
    print(f"\n{'='*80}")
    print(f"📊 RESUMEN DEL APRENDIZAJE")
    print(f"{'='*80}")
    print(f"Error promedio total:           {avg_error:.3f}")
    print(f"Error primera mitad:            {first_half_error:.3f}")
    print(f"Error segunda mitad:            {second_half_error:.3f}")
    print(f"Mejora observada:               {first_half_error - second_half_error:+.3f}")
    print(f"{'='*80}\n")
    
    # Print prediction log summary
    pred_logger.print_summary()
    
    # Run final review if we haven't just done one
    if not reviewer.should_review() and reviewer.prediction_count > 0:
        print("🔍 Ejecutando análisis final del Reviewer...")
        final_report = reviewer.run_review()
        reviewer.print_report(final_report)
        
        # Try to improve worst judge one more time
        final_improvement = reviewer.improve_worst_judge(final_report)
        if final_improvement:
            print(f"\n{'='*80}")
            print(f"🔄 FINAL JUDGE IMPROVEMENT")
            print(f"{'='*80}")
            print(f"Replaced: {final_improvement['original_judge']}")
            print(f"New version: {final_improvement['new_judge']}")
            print(f"Reason: {final_improvement['reason']}")
            print(f"{'='*80}\n")
    
    # Print improvement history if any
    if reviewer.improvement_history:
        print(f"\n{'='*80}")
        print(f"📜 JUDGE IMPROVEMENT HISTORY")
        print(f"{'='*80}")
        for i, imp in enumerate(reviewer.improvement_history, 1):
            print(f"{i}. {imp['original_judge']} → {imp['new_judge']}")
            print(f"   Error: {imp['original_error']:.3f} | Reason: {imp['reason']}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # CLI:
    #   python -m src.main_demo
    #   python -m src.main_demo --resources /path/to/resources
    #   python -m src.main_demo --samples 20
    #   python -m src.main_demo --samples 20 --review-interval 10
    #   python -m src.main_demo --resources /path/to/resources --samples 20 --review-interval 10
    args = sys.argv[1:]
    res_arg = None
    n_samples = 10  # default
    review_interval = 5  # default
    
    i = 0
    while i < len(args):
        if args[i] == "--resources":
            if i + 1 >= len(args):
                print("Usage: python -m src.main_demo [--resources /path/to/resources] [--samples N] [--review-interval N]")
                sys.exit(1)
            res_arg = args[i + 1]
            i += 2
        elif args[i] == "--samples":
            if i + 1 >= len(args):
                print("Usage: python -m src.main_demo [--resources /path/to/resources] [--samples N] [--review-interval N]")
                sys.exit(1)
            try:
                n_samples = int(args[i + 1])
            except ValueError:
                print(f"Error: --samples debe ser un número entero, recibido: {args[i + 1]}")
                sys.exit(1)
            i += 2
        elif args[i] in ("--review-interval", "--review"):
            if i + 1 >= len(args):
                print("Usage: python -m src.main_demo [--resources /path/to/resources] [--samples N] [--review-interval N]")
                sys.exit(1)
            try:
                review_interval = int(args[i + 1])
            except ValueError:
                print(f"Error: --review-interval debe ser un número entero, recibido: {args[i + 1]}")
                sys.exit(1)
            i += 2
        else:
            i += 1
    
    demo_flow(resources_dir=res_arg, n_samples=n_samples, review_interval=review_interval)
