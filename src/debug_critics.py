from __future__ import annotations

# Allow "python -m src.debug_critics ..." without package context
if __name__ == "__main__" and __package__ is None:
    import os, sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import ast
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd

from src.llm_client import LLMClient, extract_json_block
from src.critics import Critic
from src.data_store import DataStore
from src.retriever import Retriever


# ---------- data & resources helpers ----------
def parse_genre_list_column(df: pd.DataFrame) -> pd.DataFrame:
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


def load_df(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    elif ext in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported data extension: {ext}")
    return parse_genre_list_column(df)


def resolve_resources_dir(cli_override: Optional[str]) -> Path:
    if cli_override:
        return Path(cli_override).expanduser().resolve()
    env = os.getenv("RESOURCES_DIR")
    if env:
        return Path(env).expanduser().resolve()
    repo_root = Path(__file__).resolve().parent.parent
    primary = (repo_root / "resources").resolve()
    if primary.exists():
        return primary
    fallback = (repo_root / "src" / "resources").resolve()
    if fallback.exists():
        return fallback
    return (Path.cwd() / "resources").resolve()


def find_persona_text(resources_dir: Path, critic_id: str) -> str:
    base = resources_dir / "movie_critics" / critic_id
    for ext in (".txt", ".md"):
        p = base.with_suffix(ext)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8").strip()
            except Exception:
                pass
    return f"You are a movie critic persona named '{critic_id}'. Rely only on provided context."


# ---------- prompt builders (fallback path) ----------
CRITIC_JSON_SPEC = """
Return STRICT JSON only with these exact keys:
{
  "score": number,
  "confidence": number,
  "rationale": string
}
No extra keys. No markdown. No prose outside the JSON.
""".strip()


def build_user_prompt_from_ctx(ctx) -> str:
    movie = ctx.movie_profile or {}
    up = ctx.user_profile or {}
    neighbors = (ctx.retrieved or {}).get("neighbors", [])

    lines = [
        "Context:",
        f"- User personality: {up.get('personality', '')}",
        f"- Movie title: {movie.get('title', '')}",
        f"- Movie overview: {movie.get('overview', '')}",
    ]
    genres = movie.get("genres", [])
    if isinstance(genres, list):
        lines.append(f"- Movie genres: {', '.join(genres)}")
    else:
        lines.append(f"- Movie genres: {genres}")

    lines.append("- User top history (title, rating) [up to 10]:")
    for h in (up.get("history", []) or [])[:10]:
        lines.append(f"  * {h.get('title', '?')} — {h.get('rating', '?')}")

    if neighbors:
        lines.append("- Nearest neighbors (title, sim) [up to 6]:")
        for n in neighbors[:6]:
            title = n.get("title", "?")
            sim = n.get("sim", 0.0)
            try:
                sim = float(sim)
            except Exception:
                sim = 0.0
            lines.append(f"  * {title} — sim={sim:.2f}")

    lines.append(
        "\nTask: Predict how much this user would like this movie on a 0..5 scale."
        " Provide a short rationale grounded in the context."
    )
    return "\n".join(lines)


# ---------- printing ----------
def print_bar(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def print_parsed_and_raw(raw: str, maxraw: int = 1200):
    parsed = extract_json_block(raw) or {}
    score = parsed.get("score")
    conf = parsed.get("confidence")
    rationale = parsed.get("rationale")
    print("[Parsed]")
    print(f"  score     : {score}")
    print(f"  confidence: {conf}")
    print(f"  rationale : {rationale}\n")

    print("[Raw]")
    rtxt = raw if isinstance(raw, str) else str(raw)
    cut = rtxt[:maxraw]
    print(cut)
    if len(rtxt) > maxraw:
        print("... [truncated] ...")


# ---------- main ----------
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Debug LLM critics: print raw + parsed outputs."
    )
    ap.add_argument(
        "data_path",
        help="CSV/PKL with: userId, movieId, title, overview, genre_list, rating[, personality]",
    )
    ap.add_argument(
        "--resources",
        help="Path to resources dir (contains movie_critics/ and judges/).",
    )
    ap.add_argument("--user", help="User ID (default: from first row).")
    ap.add_argument("--movie", help="Movie ID (default: from first row).")
    ap.add_argument(
        "--critic",
        required=True,
        help="Critic ID to run (basename of the persona file).",
    )
    ap.add_argument(
        "--maxraw", type=int, default=1200, help="Max chars of raw output to print."
    )
    args = ap.parse_args()

    df = load_df(args.data_path)
    if len(df) == 0:
        print("Dataset is empty.")
        sys.exit(1)

    if args.user and args.movie:
        user_id = str(args.user)
        movie_id = str(args.movie)
    else:
        row = df.iloc[0]
        user_id = str(row["userId"])
        movie_id = str(row["movieId"])

    resources_dir = resolve_resources_dir(args.resources)
    print(f"[Init] resources_dir: {resources_dir}")
    print(f"[Init] Critic to run: {args.critic}")
    print(f"[Init] userId={user_id}, movieId={movie_id}")

    # Build context
    store = DataStore(df)
    retriever = Retriever(store)
    ctx = retriever.get_context(user_id=user_id, movie_id=movie_id)

    # Build critic instance
    llm = LLMClient()
    critic = Critic(critic_id=args.critic, resources_dir=str(resources_dir), llm=llm)

    # -------- Path A: try public API first (critic.score) and show parsed + any raw flag --------
    print_bar(f"CRITIC (public API): {args.critic}")
    try:
        out = critic.score(ctx)
        print("[Parsed from critic.score()]")
        print(f"  score     : {getattr(out, 'score', None)}")
        print(f"  confidence: {getattr(out, 'confidence', None)}")
        print(f"  rationale : {(getattr(out, 'rationale', '') or '').strip()}")
        raw = None
        flags = getattr(out, "flags", None)
        if isinstance(flags, dict):
            raw = flags.get("llm_raw")
        if raw:
            print("\n[Raw from critic.flags.llm_raw]")
            rtxt = raw if isinstance(raw, str) else str(raw)
            cut = rtxt[: args.maxraw]
            print(cut)
            if len(rtxt) > args.maxraw:
                print("... [truncated] ...")
        else:
            print("\n[Raw] (critic did not attach raw text)")
    except Exception as e:
        print(f"[Warn] critic.score() raised {type(e).__name__}: {e}")

    # -------- Path B: force a manual call (robust even if critic lacks helpers) --------
    print_bar(f"CRITIC (manual call with enforced JSON): {args.critic}")
    try:
        persona = find_persona_text(resources_dir, args.critic)
        system_prompt = (persona.strip() + "\n\n" + CRITIC_JSON_SPEC).strip()
        user_prompt = build_user_prompt_from_ctx(ctx)

        raw = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=getattr(critic, "model", "gpt-5"),
            settings={"max_completion_tokens": 700},
            force_json=True,
        )
        print_parsed_and_raw(raw, maxraw=args.maxraw)
    except Exception as e:
        print(f"[Err] Manual call failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
