#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Generator, List, Optional, Tuple

import chess

HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
MOVE_NUM_RE = re.compile(r'^\d+\.(\.\.)?$')


def clean_player_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name or "")
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    return name.strip()


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
    et = (headers.get("EventType") or "").strip().lower()
    if "blitz" in et:
        return "blitz"
    if "rapid" in et:
        return "rapid"
    return "classical"


def eco_to_bucket(eco: str, mode: str = "family2") -> str:
    eco = (eco or "").strip().upper()
    if not eco:
        return "UNKNOWN"
    if mode == "letter":
        return eco[0] if len(eco) >= 1 else "UNKNOWN"
    if mode == "family2":
        return eco[:2] if len(eco) >= 2 else eco
    if mode == "full":
        return eco
    raise ValueError(f"Unsupported eco bucket mode: {mode}")


def ply_to_phase(ply_abs: int) -> str:
    if ply_abs < 20:
        return "opening"
    if ply_abs < 60:
        return "middlegame"
    return "endgame"


def make_game_id(partition_name: str, game_idx: int) -> str:
    return f"{partition_name}__g{game_idx}"


def make_example_id(game_id: str, ply_idx: int) -> str:
    return f"{game_id}_{ply_idx}"


def iter_partition_files(input_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and name.lower().endswith(".pgn"):
            files.append(path)
    return files


def iter_raw_games(path: str) -> Generator[Tuple[Dict[str, str], str], None, None]:
    """
    Robust PGN splitter:
    - collect consecutive header lines
    - then collect movetext until next header block
    - return (headers_dict, movetext_str)
    """
    with open(path, "r", encoding="utf-8", errors="ignore", buffering=1024 * 1024 * 16) as f:
        header_lines: List[str] = []
        move_lines: List[str] = []
        in_headers = False

        for line in f:
            stripped = line.rstrip("\n")

            if stripped.startswith("["):
                # start of next game only if we already have headers and some movetext
                if header_lines and move_lines:
                    headers = parse_header_lines(header_lines)
                    movetext = "".join(move_lines).strip()
                    yield headers, movetext
                    header_lines = []
                    move_lines = []

                in_headers = True
                header_lines.append(line)
                continue

            if in_headers and stripped == "":
                in_headers = False
                continue

            if header_lines:
                move_lines.append(line)

        if header_lines:
            headers = parse_header_lines(header_lines)
            movetext = "".join(move_lines).strip()
            yield headers, movetext


def parse_header_lines(header_lines: List[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for line in header_lines:
        m = HEADER_RE.match(line.strip())
        if m:
            headers[m.group(1)] = m.group(2)
    return headers


def clean_movetext(text: str) -> str:
    # remove comments
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r";[^\n]*", " ", text)

    # remove simple nested variations iteratively
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\([^()]*\)", " ", text)

    # remove NAGs
    text = re.sub(r"\$\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_san_movetext(movetext: str) -> List[str]:
    text = clean_movetext(movetext)
    raw_tokens = text.split()

    tokens: List[str] = []
    for tok in raw_tokens:
        tok = tok.strip()

        if not tok:
            continue

        # skip move numbers like "1." or "23..."
        if MOVE_NUM_RE.match(tok):
            continue

        # skip pure results
        if tok in RESULT_TOKENS:
            continue

        # handle combined tokens like "31...Qh4" or "31... Qh4"
        tok = re.sub(r'^\d+\.\.\.', '', tok)
        tok = re.sub(r'^\d+\.', '', tok)

        if not tok or tok in RESULT_TOKENS:
            continue

        tokens.append(tok)

    return tokens


def emit_examples_for_game(
    headers: Dict[str, str],
    movetext: str,
    partition_name: str,
    game_idx: int,
    min_elo: int,
    eco_bucket_mode: str,
) -> Tuple[List[dict], str]:
    white = clean_player_name(headers.get("White", ""))
    black = clean_player_name(headers.get("Black", ""))

    players_to_skip = ["Keymer, Vincent", "Wei Yi", "Carlsen, M.", "Firouzja, Alireza", "Giri, A.", "Gukesh, D.", "Praggnanandhaa, R.", "Caruana, F.", "Nakamura, Hi", "Nakamura, H."]

    if white in players_to_skip or black in players_to_skip:
        return [], "skipped_player"

    if not white or not black or white == "?" or black == "?":
        return [], "missing_headers"

    white_elo = safe_int(headers.get("WhiteElo"), default=-1)
    black_elo = safe_int(headers.get("BlackElo"), default=-1)

    if max(white_elo, black_elo) < min_elo:
        return [], "below_elo"

    san_tokens = tokenize_san_movetext(movetext)
    if not san_tokens:
        return [], "empty"

    game_type = classify_time_control(headers)
    opening_bucket = eco_to_bucket(headers.get("ECO", ""), mode=eco_bucket_mode)
    game_id = make_game_id(partition_name, game_idx)

    board = chess.Board()
    pre_move_fens: List[str] = [board.fen()]
    rows: List[dict] = []

    ply_idx = 0
    for san in san_tokens:
        try:
            move = board.parse_san(san)
        except Exception:
            return [], "illegal_mainline"

        ply_idx += 1
        mover_is_white = board.turn == chess.WHITE
        move_color = "white" if mover_is_white else "black"

        if mover_is_white:
            player_id = white
            opponent_id = black
            player_elo = white_elo
        else:
            player_id = black
            opponent_id = white
            player_elo = black_elo

        if player_elo >= min_elo:
            hist = pre_move_fens[-6:]
            if len(hist) < 6:
                hist = [hist[0]] * (6 - len(hist)) + hist

            rows.append({
                "example_id": make_example_id(game_id, ply_idx),
                "player_id": player_id,
                "opponent_id": opponent_id,
                "game_id": game_id,
                "ply_idx": ply_idx,
                "move_color": move_color,
                "game_type": game_type,
                "opening_bucket": opening_bucket,
                "phase": ply_to_phase(ply_idx),
                "board_t_minus_5": hist[0],
                "board_t_minus_4": hist[1],
                "board_t_minus_3": hist[2],
                "board_t_minus_2": hist[3],
                "board_t_minus_1": hist[4],
                "board_t": hist[5],
                "move_played": move.uci(),
            })

        board.push(move)
        pre_move_fens.append(board.fen())

    return rows, "ok"


def process_partition(
    input_path: str,
    output_dir: str,
    min_elo: int,
    eco_bucket_mode: str,
    flush_every: int = 10000,
) -> Dict[str, object]:
    partition_filename = os.path.basename(input_path)
    partition_stem, _ = os.path.splitext(partition_filename)
    output_path = os.path.join(output_dir, f"{partition_stem}.jsonl")

    os.makedirs(output_dir, exist_ok=True)

    total_games = 0
    total_rows = 0
    parse_errors = 0
    skipped_missing_headers = 0
    skipped_below_elo = 0
    skipped_empty = 0
    skipped_illegal_mainline = 0
    skipped_player = 0

    with open(output_path, "w", encoding="utf-8", buffering=1024 * 1024 * 16) as out_f:
        for game_idx, (headers, movetext) in enumerate(iter_raw_games(input_path), start=1):
            total_games += 1

            if total_games == 1:
                preview = movetext[:400]
                print("FIRST HEADERS:", headers, file=sys.stderr)
                print("FIRST MOVETEXT PREVIEW:", preview, file=sys.stderr)
                print("FIRST TOKENS:", tokenize_san_movetext(movetext)[:20], file=sys.stderr)

            try:
                rows, status = emit_examples_for_game(
                    headers=headers,
                    movetext=movetext,
                    partition_name=partition_stem,
                    game_idx=game_idx,
                    min_elo=min_elo,
                    eco_bucket_mode=eco_bucket_mode,
                )

                if status == "missing_headers":
                    skipped_missing_headers += 1
                    continue
                if status == "below_elo":
                    skipped_below_elo += 1
                    continue
                if status == "empty":
                    skipped_empty += 1
                    continue
                if status == "illegal_mainline":
                    skipped_illegal_mainline += 1
                    continue
                if status == "skipped_player":
                    skipped_player += 1
                    continue

                for row in rows:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += len(rows)

                if total_games % flush_every == 0:
                    out_f.flush()
                    print(
                        f"[{partition_stem}] games={total_games:,} rows={total_rows:,} "
                        f"parse_errors={parse_errors:,} "
                        f"missing_headers={skipped_missing_headers:,} "
                        f"below_elo={skipped_below_elo:,} "
                        f"empty={skipped_empty:,} "
                        f"illegal_mainline={skipped_illegal_mainline:,}",
                        file=sys.stderr,
                    )

            except Exception as e:
                parse_errors += 1
                print(f"[{partition_stem}] ERROR on game {game_idx}: {e}", file=sys.stderr)

    return {
        "partition": partition_stem,
        "input_path": input_path,
        "output_path": output_path,
        "games": total_games,
        "rows": total_rows,
        "parse_errors": parse_errors,
        "skipped_missing_headers": skipped_missing_headers,
        "skipped_below_elo": skipped_below_elo,
        "skipped_empty": skipped_empty,
        "skipped_illegal_mainline": skipped_illegal_mainline,
        "skipped_player": skipped_player,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-elo", type=int, default=2500)
    ap.add_argument("--eco-bucket-mode", choices=["letter", "family2", "full"], default="family2")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))))
    args = ap.parse_args()

    input_files = iter_partition_files(args.input_dir)
    if not input_files:
        raise FileNotFoundError(f"No .pgn files found in: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[main] found {len(input_files)} partition files", file=sys.stderr)
    print(f"[main] using {args.workers} worker processes", file=sys.stderr)

    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
    ) as ex:
        futures = [
            ex.submit(
                process_partition,
                input_path,
                args.output_dir,
                args.min_elo,
                args.eco_bucket_mode,
            )
            for input_path in input_files
        ]

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(
                f"[done] {res['partition']}: games={res['games']:,}, rows={res['rows']:,}, "
                f"parse_errors={res['parse_errors']:,}, "
                f"missing_headers={res['skipped_missing_headers']:,}, "
                f"below_elo={res['skipped_below_elo']:,}, "
                f"empty={res['skipped_empty']:,}, "
                f"illegal_mainline={res['skipped_illegal_mainline']:,} "
                f"skipped_player={res['skipped_player']:,} "
                f"-> {res['output_path']}",
                file=sys.stderr,
            )

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": args.input_dir,
                "output_dir": args.output_dir,
                "min_elo": args.min_elo,
                "eco_bucket_mode": args.eco_bucket_mode,
                "workers": args.workers,
                "partitions": sorted(results, key=lambda x: x["partition"]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[main] manifest -> {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()