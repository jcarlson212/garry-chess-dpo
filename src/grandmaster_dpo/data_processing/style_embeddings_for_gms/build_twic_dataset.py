#!/usr/bin/env python3
"""
Build a globally balanced TWIC dataset from one giant PGN file.

Behavior:
1) Stream a very large PGN file efficiently.
2) Keep games only if:
   - standard chess
   - at least 5 plies
   - at least one player has Elo >= 2500
3) De-duplicate games.
4) Classify each kept game as blitz / rapid / classical.
5) Randomly sample a FINAL GLOBAL mix of:
      60% blitz
      30% rapid
      10% classical
6) Partition output PGNs by the higher-rated player's first two letters.
7) Preserve original raw PGN text and metadata.

Notes:
- No minimum player-games threshold.
- No maximum games per player.
- Global random balancing, not per-player balancing.

Example:
python build_twic_global_mix.py \
    --input /path/to/all_twic_games.pgn \
    --output-dir /path/to/out_pgns \
    --sqlite /path/to/twic_global.db \
    --min-elo 2500 \
    --min-plies 5 \
    --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}


# ---------------------------------------------------
# PGN streaming
# ---------------------------------------------------

def iter_raw_pgn_games(path: str) -> Generator[Tuple[str, Dict[str, str], str], None, None]:
    """
    Stream PGN games as raw text blocks:
      raw_game_text, headers_dict, movetext_string
    """
    with open(path, "r", encoding="utf-8", errors="ignore", buffering=1024 * 1024 * 16) as f:
        header_lines: List[str] = []
        move_lines: List[str] = []
        in_headers = False

        for line in f:
            stripped = line.rstrip("\n")

            if stripped.startswith("["):
                if move_lines and header_lines:
                    raw = "".join(header_lines) + "\n" + "".join(move_lines).rstrip() + "\n\n"
                    headers = parse_header_lines(header_lines)
                    movetext = "".join(move_lines).strip()
                    yield raw, headers, movetext
                    header_lines = []
                    move_lines = []

                in_headers = True
                header_lines.append(line)
                continue

            if in_headers and stripped == "":
                if header_lines:
                    move_lines.append(line)
                in_headers = False
                continue

            if header_lines:
                move_lines.append(line)

        if header_lines:
            raw = "".join(header_lines) + "\n" + "".join(move_lines).rstrip() + "\n"
            headers = parse_header_lines(header_lines)
            movetext = "".join(move_lines).strip()
            yield raw, headers, movetext


def parse_header_lines(header_lines: List[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for line in header_lines:
        m = HEADER_RE.match(line.strip())
        if m:
            headers[m.group(1)] = m.group(2)
    return headers


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def safe_int(x: Optional[str], default: int = -1) -> int:
    if x is None:
        return default
    x = x.strip()
    if not x:
        return default
    try:
        return int(x)
    except Exception:
        return default


def classify_time_control(headers: Dict[str, str]) -> str:
    """
    Uses EventType heuristic:
      blitz -> blitz
      rapid -> rapid
      else -> classical
    """
    et = (headers.get("EventType") or "").strip().lower()
    if "blitz" in et:
        return "blitz"
    if "rapid" in et:
        return "rapid"
    return "classical"


def is_standard_game(headers: Dict[str, str]) -> bool:
    variant = (headers.get("Variant") or "").strip().lower()
    if variant and variant not in {"standard", "chess"}:
        return False
    return True


def normalize_movetext_for_hash(movetext: str) -> str:
    s = movetext
    s = re.sub(r"\{[^}]*\}", " ", s)        # PGN comments
    s = re.sub(r";[^\n]*", " ", s)          # semicolon comments

    # Remove simple non-nested variations repeatedly
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\([^()]*\)", " ", s)

    s = re.sub(r"\$\d+", " ", s)            # NAGs
    s = re.sub(r"\b\d+\.(\.\.)?", " ", s)   # move numbers
    s = re.sub(r"\s+", " ", s).strip()
    return s


def approximate_ply_count(headers: Dict[str, str], movetext: str) -> int:
    pc = safe_int(headers.get("PlyCount"), default=-1)
    if pc >= 0:
        return pc

    norm = normalize_movetext_for_hash(movetext)
    if not norm:
        return 0
    tokens = [t for t in norm.split(" ") if t and t not in RESULT_TOKENS]
    return len(tokens)


def clean_player_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    return name.strip()


def get_partition_key(headers: Dict[str, str], movetext: str) -> str:
    name = headers.get("White", "") or headers.get("Black", "")
    variant = (headers.get("Variant") or "").strip().lower()
    partition_bytes = (clean_player_name(name).lower() + variant.lower() + normalize_movetext_for_hash(movetext)).encode("utf-8")
    h = hashlib.md5(partition_bytes).hexdigest()
    return h[:1]


def higher_rated_player(headers: Dict[str, str]) -> Tuple[str, int]:
    w = headers.get("White", "").strip()
    b = headers.get("Black", "").strip()
    we = safe_int(headers.get("WhiteElo"), default=-1)
    be = safe_int(headers.get("BlackElo"), default=-1)

    if we > be:
        return w, we
    if be > we:
        return b, be
    return (w, we) if w.lower() <= b.lower() else (b, be)


def game_hash(headers: Dict[str, str], movetext: str) -> str:
    """
    Dedupe signature using players + date + result + normalized movetext.
    """
    white = clean_player_name(headers.get("White", ""))
    black = clean_player_name(headers.get("Black", ""))
    date = headers.get("Date", "").strip()
    result = headers.get("Result", "").strip()
    norm_moves = normalize_movetext_for_hash(movetext)
    payload = "\x1f".join([white, black, date, result, norm_moves])
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class CandidateGame:
    game_id: int
    time_class: str
    partition: str
    white: str
    black: str
    white_elo: int
    black_elo: int
    raw_pgn: str


# ---------------------------------------------------
# SQLite
# ---------------------------------------------------

def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS games (
        game_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        game_hash    TEXT UNIQUE,
        time_class   TEXT NOT NULL,
        partition    TEXT NOT NULL,
        white        TEXT NOT NULL,
        black        TEXT NOT NULL,
        white_elo    INTEGER NOT NULL,
        black_elo    INTEGER NOT NULL,
        raw_pgn      TEXT NOT NULL
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_games_tc ON games(time_class);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_games_partition ON games(partition);")
    conn.commit()


# ---------------------------------------------------
# Candidate extraction
# ---------------------------------------------------

def store_candidates(
    pgn_path: str,
    conn: sqlite3.Connection,
    min_elo: int,
    min_plies: int,
    stderr_every: int = 25000,
) -> None:
    cur = conn.cursor()
    n = 0
    kept = 0
    deduped = 0

    for raw, headers, movetext in iter_raw_pgn_games(pgn_path):
        n += 1
        if n % stderr_every == 0:
            print(f"[scan] scanned {n:,} games, kept {kept:,}, deduped {deduped:,}", file=sys.stderr)

        if not is_standard_game(headers):
            continue

        plies = approximate_ply_count(headers, movetext)
        if plies < min_plies:
            continue

        white = clean_player_name(headers.get("White", ""))
        black = clean_player_name(headers.get("Black", ""))
        if not white or not black:
            continue

        we = safe_int(headers.get("WhiteElo"), default=-1)
        be = safe_int(headers.get("BlackElo"), default=-1)

        if max(we, be) < min_elo:
            continue

        tc = classify_time_control(headers)
        hr_name, _ = higher_rated_player(headers)
        partition = get_partition_key(headers, movetext)
        gh = game_hash(headers, movetext)

        cur.execute("""
            INSERT OR IGNORE INTO games
            (game_hash, time_class, partition, white, black, white_elo, black_elo, raw_pgn)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (gh, tc, partition, white, black, we, be, raw))

        if cur.rowcount == 1:
            kept += 1
        else:
            deduped += 1

        if n % 5000 == 0:
            conn.commit()

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM games")
    total_games = cur.fetchone()[0]
    print(f"[scan] finished: unique kept games = {total_games:,}", file=sys.stderr)


# ---------------------------------------------------
# Global balancing
# ---------------------------------------------------

def get_counts_by_time_class(conn: sqlite3.Connection) -> Dict[str, int]:
    cur = conn.cursor()
    cur.execute("""
        SELECT time_class, COUNT(*)
        FROM games
        GROUP BY time_class
    """)
    counts = {"blitz": 0, "rapid": 0, "classical": 0}
    for tc, c in cur.fetchall():
        counts[tc] = int(c)
    return counts


def compute_global_sample_plan(counts: Dict[str, int]) -> Dict[str, int]:
    """
    Find the largest possible dataset size N such that:
      blitz_needed     = ceil/floor-compatible for 60%
      rapid_needed     = 30%
      classical_needed = 10%
    and all buckets have enough games.

    Since exact integer rounding can be annoying, we use:
      blitz = round(0.6 * N)
      rapid = round(0.3 * N)
      classical = N - blitz - rapid

    Then binary search for max feasible N.
    """
    def feasible(n: int) -> bool:
        b = int(round(0.60 * n))
        r = int(round(0.30 * n))
        c = n - b - r
        return (
            counts["blitz"] >= b and
            counts["rapid"] >= r and
            counts["classical"] >= c
        )

    lo, hi = 0, sum(counts.values())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid - 1

    n = lo
    b = int(round(0.60 * n))
    r = int(round(0.30 * n))
    c = n - b - r
    return {"total": n, "blitz": b, "rapid": r, "classical": c}


def sample_game_ids_by_time_class(
    conn: sqlite3.Connection,
    sample_plan: Dict[str, int],
    seed: int,
) -> List[int]:
    rng = random.Random(seed)
    cur = conn.cursor()

    selected_ids: List[int] = []

    for tc in ("blitz", "rapid", "classical"):
        needed = sample_plan[tc]
        cur.execute("SELECT game_id FROM games WHERE time_class = ?", (tc,))
        ids = [int(row[0]) for row in cur.fetchall()]
        rng.shuffle(ids)
        selected_ids.extend(ids[:needed])

    rng.shuffle(selected_ids)
    return selected_ids


# ---------------------------------------------------
# Output
# ---------------------------------------------------

def write_partitioned_outputs(
    conn: sqlite3.Connection,
    selected_game_ids: List[int],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if not selected_game_ids:
        print("[write] no selected games to write", file=sys.stderr)
        return

    cur = conn.cursor()
    partition_to_rows = defaultdict(list)

    chunk = 900
    for i in range(0, len(selected_game_ids), chunk):
        subset = selected_game_ids[i:i+chunk]
        placeholders = ",".join("?" for _ in subset)
        cur.execute(f"""
            SELECT game_id, partition, raw_pgn
            FROM games
            WHERE game_id IN ({placeholders})
            ORDER BY partition, game_id
        """, subset)
        for game_id, partition, raw_pgn in cur.fetchall():
            partition_to_rows[partition].append((int(game_id), raw_pgn))

    for partition, rows in partition_to_rows.items():
        out_path = os.path.join(out_dir, f"{partition}.pgn")
        with open(out_path, "w", encoding="utf-8", buffering=1024 * 1024 * 16) as f:
            for _, raw_pgn in rows:
                f.write(raw_pgn)
                if not raw_pgn.endswith("\n\n"):
                    f.write("\n\n")
        print(f"[write] {partition}: wrote {len(rows):,} games -> {out_path}", file=sys.stderr)


def write_manifest(
    out_dir: str,
    counts_before: Dict[str, int],
    sample_plan: Dict[str, int],
    seed: int,
) -> None:
    path = os.path.join(out_dir, "manifest.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Global TWIC dataset manifest\n")
        f.write("===========================\n\n")
        f.write(f"Random seed: {seed}\n\n")
        f.write("Available candidate games before balancing:\n")
        f.write(f"  blitz:     {counts_before['blitz']}\n")
        f.write(f"  rapid:     {counts_before['rapid']}\n")
        f.write(f"  classical: {counts_before['classical']}\n")
        f.write(f"  total:     {sum(counts_before.values())}\n\n")
        f.write("Selected final global mix:\n")
        f.write(f"  blitz:     {sample_plan['blitz']}\n")
        f.write(f"  rapid:     {sample_plan['rapid']}\n")
        f.write(f"  classical: {sample_plan['classical']}\n")
        f.write(f"  total:     {sample_plan['total']}\n")

    print(f"[write] manifest -> {path}", file=sys.stderr)


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main() -> None:
    # python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/build_twic_dataset.py --input ./final_experiments_for_paper/experiment2_style_model/raw_pgns_twic/twic_1k_plus_games.pgn --output-dir ./final_experiments_for_paper/experiment2_style_model/filtered_pgns_twic_1k_plus --sqlite ./final_experiments_for_paper/experiment2_style_model/twic_work_1k_plus.db --min-elo 2500 --min-plies 5 --seed 42 --rebuild-db
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to giant TWIC PGN file")
    ap.add_argument("--output-dir", required=True, help="Directory for partitioned output PGNs")
    ap.add_argument("--sqlite", required=True, help="Path to working SQLite database")
    ap.add_argument("--min-elo", type=int, default=2500, help="At least one player in game must have Elo >= this")
    ap.add_argument("--min-plies", type=int, default=5, help="Minimum plies in a game")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for global sampling")
    ap.add_argument("--rebuild-db", action="store_true", help="Delete and rebuild SQLite database")
    args = ap.parse_args()

    if args.rebuild_db and os.path.exists(args.sqlite):
        os.remove(args.sqlite)

    os.makedirs(args.output_dir, exist_ok=True)

    conn = sqlite3.connect(args.sqlite)
    init_db(conn)

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM games")
    existing_games = cur.fetchone()[0]

    if existing_games == 0:
        print("[main] scanning PGN and storing candidates...", file=sys.stderr)
        store_candidates(
            pgn_path=args.input,
            conn=conn,
            min_elo=args.min_elo,
            min_plies=args.min_plies,
        )
    else:
        print(f"[main] sqlite already contains {existing_games:,} games; skipping scan", file=sys.stderr)

    counts = get_counts_by_time_class(conn)
    print(f"[main] candidate counts by time class: {counts}", file=sys.stderr)

    sample_plan = compute_global_sample_plan(counts)
    print(f"[main] final global sample plan: {sample_plan}", file=sys.stderr)

    selected_game_ids = sample_game_ids_by_time_class(
        conn=conn,
        sample_plan=sample_plan,
        seed=args.seed,
    )

    print(f"[main] selected {len(selected_game_ids):,} total games", file=sys.stderr)

    write_partitioned_outputs(
        conn=conn,
        selected_game_ids=selected_game_ids,
        out_dir=args.output_dir,
    )

    write_manifest(
        out_dir=args.output_dir,
        counts_before=counts,
        sample_plan=sample_plan,
        seed=args.seed,
    )

    conn.close()
    print("[main] done", file=sys.stderr)


if __name__ == "__main__":
    main()
