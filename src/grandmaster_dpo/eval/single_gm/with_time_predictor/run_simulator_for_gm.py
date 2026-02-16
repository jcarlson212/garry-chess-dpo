#!/usr/bin/env python3
"""
Timed single-GM Maia2+DPO simulator (Tkinter) with a learned "think-time" model.

What this does
- Loads Maia2 (blitz) + your single-GM fine-tuned weights (policy_dpo_best.pt preferred, else policy_best.pt)
- Loads your trained timer head (timer_head_best.pt)
- For each ply, runs Maia2 *once* to get:
    (a) masked policy logits (for move distribution + SF biasing)
    (b) a feature vector for the timer model (hook layer if available, else logits fallback)
- Uses timer prediction to set Stockfish LIMIT = time=pred_seconds (instead of depth)
- Tracks clocks + last-5 ply times, and feeds those to the timer model
- Human plays via click-to-move; model moves in a background thread

Usage (only argument is gm name):
  python kasparovnet_timed_gui.py magnus
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine
import torch
import tkinter as tk
from tkinter import messagebox

from maia2 import inference, model as maia_model
from maia2.utils import mirror_move

# Optional: your helper (preferred). If not present, we fall back to direct stockfish.play()
try:
    from grandmaster_dpo.eval.stockfish_helpers import batch_choose_moves_sf_topk_biased_by_policy  # type: ignore
except Exception:
    batch_choose_moves_sf_topk_biased_by_policy = None


# ----------------------------
# Small utilities
# ----------------------------

PREFERRED_CHESS_FONTS = [
    "Noto Sans Symbols 2",
    "Noto Sans Symbols",
    "DejaVu Sans",
    "Segoe UI Symbol",
    "Apple Symbols",
    "Symbola",
    "Arial Unicode MS",
    "Helvetica",
]


def pick_glyph_font(size: int) -> tuple[str, int]:
    try:
        import tkinter.font as tkfont

        available = set(tkfont.families())
        for f in PREFERRED_CHESS_FONTS:
            if f in available:
                return (f, size)
    except Exception:
        pass
    return ("Helvetica", size)


def device_auto() -> str:
    # Pick the best available without CLI args
    if torch.cuda.is_available():
        return "cuda"
    # MPS checks are safe even if not built
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_state_dict_fuzzy(pt: Path, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    sd = torch.load(pt, map_location=map_location)
    if isinstance(sd, dict) and "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
        sd = sd["model_state_dict"]
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported checkpoint format in {pt}: {type(sd)}")

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def get_module_by_dotted_name(root: torch.nn.Module, dotted: str) -> Optional[torch.nn.Module]:
    cur: Any = root
    for part in dotted.split("."):
        if part.isdigit():
            idx = int(part)
            if isinstance(cur, (torch.nn.ModuleList, list, tuple)):
                if idx < 0 or idx >= len(cur):
                    return None
                cur = cur[idx]
            else:
                return None
        else:
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
    return cur if isinstance(cur, torch.nn.Module) else None


def max_elo_supported(elo_dict: dict) -> int:
    mx = None
    for k in elo_dict.keys():
        m = re.match(r"^>=\s*(\d+)$", str(k))
        if m:
            mx = max(mx or 0, int(m.group(1)))
    return mx if mx is not None else 3000


def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(legal_moves > 0, logits, torch.full_like(logits, neg_inf))


def fmt_clock(ms: int) -> str:
    if ms < 0:
        ms = 0
    s = ms // 1000
    mm = s // 60
    ss = s % 60
    return f"{mm:02d}:{ss:02d}"


def find_stockfish() -> str:
    env = os.environ.get("STOCKFISH_PATH", "").strip()
    if env and Path(env).exists():
        return env

    for p in [
        shutil.which("stockfish"),
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        "/usr/bin/stockfish",
    ]:
        if p and Path(p).exists():
            return str(p)

    raise FileNotFoundError(
        "Could not find Stockfish. Install it and/or set STOCKFISH_PATH to the binary path."
    )


def make_stockfish_engine(path: str, threads: int = 16, hash_mb: int = 2048) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(path)
    # Best effort configure
    try:
        engine.configure({"Threads": int(threads)})
    except Exception:
        pass
    try:
        engine.configure({"Hash": int(hash_mb)})
    except Exception:
        pass
    return engine


# ----------------------------
# Timer head (load-only)
# ----------------------------

class TimerHead(torch.nn.Module):
    def __init__(self, in_dim: int, hidden1: int, hidden2: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden1),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TimerConfig:
    hook_layer: str = "last_ln"
    logits_feature: str = "masked_logits"  # "masked_logits" or "logprobs"
    ply_norm: float = 120.0
    min_think_ms: int = 50
    max_think_ms: int = 10_000
    safety_ms: int = 50  # keep a tiny buffer from flagging


class MaiaOncePerPly:
    """
    Runs Maia2 once per ply, capturing:
      - masked_logits [1, V]
      - feats [1, D] from hook layer if available, else logits fallback
    """

    def __init__(
        self,
        maia: torch.nn.Module,
        all_moves_dict: Dict[str, int],
        elo_dict: Dict[str, int],
        device: str,
        timer_cfg: TimerConfig,
    ):
        self.maia = maia
        self.all_moves_dict = all_moves_dict
        self.elo_dict = elo_dict
        self.device = device
        self.cfg = timer_cfg

        self._hook_buf: Optional[torch.Tensor] = None
        self._hooked = False
        self._hook_handle = None

        if timer_cfg.hook_layer:
            mod = get_module_by_dotted_name(self.maia, timer_cfg.hook_layer)
            if mod is not None:
                self._hook_handle = mod.register_forward_hook(self._forward_hook)
                self._hooked = True
                print(f"[timer] Hooking Maia2 layer: {timer_cfg.hook_layer}")
            else:
                print(f"[timer] WARNING: hook_layer not found: {timer_cfg.hook_layer}. Will fallback to logits.")

    def _forward_hook(self, module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        out = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(out):
            self._hook_buf = None
            return
        if out.dim() == 4:
            out = out.mean(dim=(2, 3))
        self._hook_buf = out

    @torch.no_grad()
    def forward_once(
        self,
        fen: str,
        elo_self: int,
        elo_oppo: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          feats: [1, D]
          masked_logits: [1, V] (legal masked)
        """
        mx = max_elo_supported(self.elo_dict)
        elo_self = min(int(elo_self), mx)
        elo_oppo = min(int(elo_oppo), mx)

        board_input, es_cat, eo_cat, legal_moves = inference.preprocessing(
            fen, elo_self, elo_oppo, self.elo_dict, self.all_moves_dict
        )

        board_input = board_input.unsqueeze(0).to(self.device)   # [1, C, 8, 8]
        legal_moves = legal_moves.unsqueeze(0).to(self.device)   # [1, V]
        es_t = torch.tensor([int(es_cat)], device=self.device).long()
        eo_t = torch.tensor([int(eo_cat)], device=self.device).long()

        self._hook_buf = None
        logits_maia, _, _ = self.maia(board_input, es_t, eo_t)  # [1, V]

        masked_logits = apply_legal_mask(logits_maia, legal_moves)

        # Features: hook if present, else logits fallback
        if self._hooked and self._hook_buf is not None:
            feats = self._hook_buf
        else:
            if self.cfg.logits_feature == "logprobs":
                feats = torch.log_softmax(masked_logits, dim=-1)
            else:
                feats = masked_logits

        return feats, masked_logits


def build_timer_head_from_ckpt(timer_pt: Path, device: str) -> Tuple[TimerHead, int]:
    sd = load_state_dict_fuzzy(timer_pt, map_location="cpu")
    # Infer shapes
    w0 = sd.get("net.0.weight", None)
    w1 = sd.get("net.3.weight", None)
    if w0 is None or w1 is None:
        raise ValueError(f"timer head checkpoint missing expected keys (net.0.weight, net.3.weight): {timer_pt}")

    hidden1 = int(w0.shape[0])
    in_dim = int(w0.shape[1])
    hidden2 = int(w1.shape[0])

    # Infer dropout from presence only (not stored); set a sane default
    head = TimerHead(in_dim=in_dim, hidden1=hidden1, hidden2=hidden2, dropout=0.1)
    head.load_state_dict(sd, strict=True)
    head.to(device)
    head.eval()
    feat_dim = in_dim - 7  # prev5 (5) + clock (1) + ply (1)
    if feat_dim <= 0:
        raise ValueError(f"Timer in_dim={in_dim} implies feat_dim={feat_dim}, which is invalid.")
    return head, feat_dim


# ----------------------------
# Tk UI
# ----------------------------

UNICODE_PIECES = {
    chess.Piece(chess.PAWN, chess.WHITE): "♙",
    chess.Piece(chess.KNIGHT, chess.WHITE): "♘",
    chess.Piece(chess.BISHOP, chess.WHITE): "♗",
    chess.Piece(chess.ROOK, chess.WHITE): "♖",
    chess.Piece(chess.QUEEN, chess.WHITE): "♕",
    chess.Piece(chess.KING, chess.WHITE): "♔",
    chess.Piece(chess.PAWN, chess.BLACK): "♟",
    chess.Piece(chess.KNIGHT, chess.BLACK): "♞",
    chess.Piece(chess.BISHOP, chess.BLACK): "♝",
    chess.Piece(chess.ROOK, chess.BLACK): "♜",
    chess.Piece(chess.QUEEN, chess.BLACK): "♛",
    chess.Piece(chess.KING, chess.BLACK): "♚",
}

PROMO_MAP = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}


class TimedKasparovNetGUI:
    def __init__(
        self,
        *,
        root: tk.Tk,
        gm_name: str,
        board: chess.Board,
        maia_runner: MaiaOncePerPly,
        timer_head: TimerHead,
        feat_dim: int,
        all_moves_dict: Dict[str, int],
        engine: chess.engine.SimpleEngine,
        timer_cfg: TimerConfig,
        elo_self: int = 2800,
        elo_oppo: int = 2800,
        topk: int = 10,
        square_size: int = 72,
        start_clock_ms: int = 5 * 60 * 1000,
    ):
        self.root = root
        self.gm_name = gm_name
        self.board = board

        self.maia_runner = maia_runner
        self.timer_head = timer_head
        self.feat_dim = feat_dim

        self.all_moves_dict = all_moves_dict
        # Build inverse vocab list for topk display (uci_eff)
        self.idx_to_uci_eff = [None] * len(all_moves_dict)
        for u, i in all_moves_dict.items():
            if 0 <= int(i) < len(self.idx_to_uci_eff):
                self.idx_to_uci_eff[int(i)] = u
        self.engine = engine
        self.cfg = timer_cfg

        self.elo_self = int(elo_self)
        self.elo_oppo = int(elo_oppo)
        self.topk = int(topk)

        # Clock state
        self.clock_w_ms = int(start_clock_ms)
        self.clock_b_ms = int(start_clock_ms)
        self.increment_ms = 0  # keep simple: 5+0
        self.turn_start_wall = time.time()
        self.last_ply_times_ms: List[int] = []  # used for prev5 feature (last 5 plies)

        # Choose side via a quick dialog (no CLI args)
        self.human_plays = self._ask_side()  # "white" or "black"

        # UI geometry
        self.square_size = square_size
        self.margin = 16
        self.canvas_size = self.margin * 2 + self.square_size * 8

        # Selection state
        self.selected_square: Optional[int] = None
        self.legal_targets: set[int] = set()
        self.model_busy = False

        # Orientation
        self.view_from_white = (self.human_plays != "black")

        self.root.title(f"KasparovNet Timed — {gm_name}")
        self.root.geometry(f"{self.canvas_size + 420}x{self.canvas_size + 70}")

        self.frame = tk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.frame,
            width=self.canvas_size,
            height=self.canvas_size,
            highlightthickness=0,
        )
        self.canvas.pack(side="left", padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        self.right = tk.Frame(self.frame)
        self.right.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.status = tk.Label(self.right, text="", anchor="w", justify="left", font=("Helvetica", 12))
        self.status.pack(fill="x")

        self.clocks = tk.Label(self.right, text="", anchor="w", justify="left", font=("Menlo", 12, "bold"))
        self.clocks.pack(fill="x", pady=(6, 0))

        self.pred = tk.Label(self.right, text="", anchor="w", justify="left", font=("Menlo", 11))
        self.pred.pack(fill="x", pady=(6, 0))

        self.cands_label = tk.Label(self.right, text="Policy top-k (prob):", anchor="w", font=("Helvetica", 11, "bold"))
        self.cands_label.pack(fill="x", pady=(10, 0))

        self.cands = tk.Text(self.right, height=18, width=48, font=("Menlo", 11))
        self.cands.pack(fill="both", expand=True)

        btn_row = tk.Frame(self.right)
        btn_row.pack(fill="x", pady=(10, 0))

        self.btn_new = tk.Button(btn_row, text="New Game", command=self.reset_game)
        self.btn_new.pack(side="left")

        self.btn_undo = tk.Button(btn_row, text="Undo (2 ply)", command=self.undo_2ply)
        self.btn_undo.pack(side="left", padx=(8, 0))

        self.btn_flip = tk.Button(btn_row, text="Flip Board", command=self.flip_view)
        self.btn_flip.pack(side="left", padx=(8, 0))

        self.btn_swap = tk.Button(btn_row, text="Swap Sides (New)", command=self.swap_sides_new)
        self.btn_swap.pack(side="left", padx=(8, 0))

        self.redraw()
        self.maybe_make_model_move()

    # ----------------------------
    # Side selection dialog
    # ----------------------------

    def _ask_side(self) -> str:
        # Simple modal
        dlg = tk.Toplevel(self.root)
        dlg.title("Choose side")
        dlg.geometry("320x140")
        dlg.grab_set()

        choice = {"val": "black"}

        tk.Label(dlg, text="Play as:", font=("Helvetica", 12)).pack(pady=12)

        row = tk.Frame(dlg)
        row.pack()

        def set_choice(v: str):
            choice["val"] = v
            dlg.destroy()

        tk.Button(row, text="White", width=10, command=lambda: set_choice("white")).pack(side="left", padx=8)
        tk.Button(row, text="Black", width=10, command=lambda: set_choice("black")).pack(side="left", padx=8)

        self.root.wait_window(dlg)
        return choice["val"]

    # ----------------------------
    # Board rendering
    # ----------------------------

    def square_to_xy(self, square: int) -> Tuple[int, int]:
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        if self.view_from_white:
            x_file = file
            y_rank = 7 - rank
        else:
            x_file = 7 - file
            y_rank = rank

        x0 = self.margin + x_file * self.square_size
        y0 = self.margin + y_rank * self.square_size
        return x0, y0

    def xy_to_square(self, x: int, y: int) -> Optional[int]:
        x -= self.margin
        y -= self.margin
        if x < 0 or y < 0:
            return None
        file = x // self.square_size
        rank = y // self.square_size
        if not (0 <= file <= 7 and 0 <= rank <= 7):
            return None

        if self.view_from_white:
            real_file = int(file)
            real_rank = 7 - int(rank)
        else:
            real_file = 7 - int(file)
            real_rank = int(rank)

        return chess.square(real_file, real_rank)

    def redraw(self) -> None:
        import tkinter.font as tkfont

        self.canvas.delete("all")
        light = "#EEEED2"
        dark = "#769656"
        sel = "#F6F669"
        target = "#BACA44"
        last = "#CDD26A"

        last_from = last_to = None
        if len(self.board.move_stack) > 0:
            mv = self.board.move_stack[-1]
            last_from, last_to = mv.from_square, mv.to_square

        for sq in chess.SQUARES:
            x0, y0 = self.square_to_xy(sq)
            x1, y1 = x0 + self.square_size, y0 + self.square_size

            is_dark = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            color = dark if is_dark else light

            if sq == self.selected_square:
                color = sel
            elif sq in self.legal_targets:
                color = target
            elif sq == last_from or sq == last_to:
                color = last

            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        # coords
        for i in range(8):
            file_char = "abcdefgh"[i]
            rank_char = str(i + 1)

            if self.view_from_white:
                x = self.margin + i * self.square_size + self.square_size // 2
                y = self.margin + 8 * self.square_size + 2
                self.canvas.create_text(x, y, text=file_char, anchor="n", font=("Helvetica", 10))

                x2 = self.margin - 4
                y2 = self.margin + (7 - i) * self.square_size + self.square_size // 2
                self.canvas.create_text(x2, y2, text=rank_char, anchor="e", font=("Helvetica", 10))
            else:
                x = self.margin + (7 - i) * self.square_size + self.square_size // 2
                y = self.margin + 8 * self.square_size + 2
                self.canvas.create_text(x, y, text=file_char, anchor="n", font=("Helvetica", 10))

                x2 = self.margin - 4
                y2 = self.margin + i * self.square_size + self.square_size // 2
                self.canvas.create_text(x2, y2, text=rank_char, anchor="e", font=("Helvetica", 10))

        # pieces
        try:
            fams = set(tkfont.families())
            font_family = next((f for f in PREFERRED_CHESS_FONTS if f in fams), "Helvetica")
        except Exception:
            font_family = "Helvetica"

        glyph_font = (font_family, int(self.square_size * 0.65))

        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if not piece:
                continue
            sym = UNICODE_PIECES.get(piece, piece.symbol())
            x0, y0 = self.square_to_xy(sq)
            cx = x0 + self.square_size // 2
            cy = y0 + self.square_size // 2

            if piece.color == chess.WHITE:
                fill = "#FFFFFF"
                outline = "#111111"
            else:
                fill = "#111111"
                outline = "#FFFFFF"

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                self.canvas.create_text(cx + dx, cy + dy, text=sym, font=glyph_font, fill=outline)
            self.canvas.create_text(cx, cy, text=sym, font=glyph_font, fill=fill)

        self.update_status()

    def update_status(self) -> None:
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        human_turn = self.is_human_turn()

        # Clock display
        self.clocks.config(
            text=f"White: {fmt_clock(self.clock_w_ms)}    Black: {fmt_clock(self.clock_b_ms)}"
        )

        if self.board.is_game_over():
            outcome = self.board.outcome()
            msg = f"Game over: {self.board.result()}"
            if outcome is not None and outcome.termination is not None:
                msg += f" ({outcome.termination.name})"
        else:
            msg = f"Turn: {turn} | Human to move: {'YES' if human_turn else 'NO'}"
            if self.model_busy:
                msg += " | Model thinking…"

        self.status.config(text=msg)

    def set_candidates_text(self, text: str) -> None:
        self.cands.delete("1.0", tk.END)
        self.cands.insert(tk.END, text)

    # ----------------------------
    # Interaction
    # ----------------------------

    def is_human_turn(self) -> bool:
        return (self.human_plays == "white" and self.board.turn == chess.WHITE) or (
            self.human_plays == "black" and self.board.turn == chess.BLACK
        )

    def on_click(self, event) -> None:
        if self.board.is_game_over() or self.model_busy:
            return

        sq = self.xy_to_square(event.x, event.y)
        if sq is None:
            return

        if not self.is_human_turn():
            return

        if self.selected_square is None:
            piece = self.board.piece_at(sq)
            if piece is None or piece.color != self.board.turn:
                return
            self.selected_square = sq
            self.legal_targets = {m.to_square for m in self.board.legal_moves if m.from_square == sq}
            self.redraw()
            return

        from_sq = self.selected_square
        to_sq = sq

        if to_sq == from_sq:
            self.clear_selection()
            return

        move = self.build_move_with_possible_promotion(from_sq, to_sq)
        if move is None:
            self.clear_selection()
            return

        if move not in self.board.legal_moves:
            piece = self.board.piece_at(to_sq)
            if piece is not None and piece.color == self.board.turn:
                self.selected_square = to_sq
                self.legal_targets = {m.to_square for m in self.board.legal_moves if m.from_square == to_sq}
                self.redraw()
                return
            messagebox.showinfo("Illegal move", "That move is not legal in this position.")
            self.clear_selection()
            return

        # --- apply human move + spend real wall time ---
        spent_ms = int(max(0.0, (time.time() - self.turn_start_wall)) * 1000.0)
        self._spend_time_for_side(self.board.turn, spent_ms)
        self.last_ply_times_ms.append(spent_ms)

        self.board.push(move)
        self.clear_selection()

        if self.board.is_game_over():
            self.redraw()
            return

        self.turn_start_wall = time.time()
        self.maybe_make_model_move()

    def clear_selection(self) -> None:
        self.selected_square = None
        self.legal_targets = set()
        self.redraw()

    def build_move_with_possible_promotion(self, from_sq: int, to_sq: int) -> Optional[chess.Move]:
        piece = self.board.piece_at(from_sq)
        if piece is None:
            return None

        if piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if (piece.color == chess.WHITE and to_rank == 7) or (piece.color == chess.BLACK and to_rank == 0):
                promo = self.ask_promotion()
                if promo is None:
                    return None
                return chess.Move(from_sq, to_sq, promotion=promo)

        return chess.Move(from_sq, to_sq)

    def ask_promotion(self) -> Optional[int]:
        dlg = tk.Toplevel(self.root)
        dlg.title("Choose promotion")
        dlg.geometry("300x110")
        dlg.grab_set()
        choice = {"val": None}

        def set_choice(letter: str):
            choice["val"] = PROMO_MAP[letter]
            dlg.destroy()

        tk.Label(dlg, text="Promote to:", font=("Helvetica", 12)).pack(pady=8)
        row = tk.Frame(dlg)
        row.pack()

        for letter, name in [("q", "Queen"), ("r", "Rook"), ("b", "Bishop"), ("n", "Knight")]:
            tk.Button(row, text=name, width=7, command=lambda l=letter: set_choice(l)).pack(side="left", padx=4)

        self.root.wait_window(dlg)
        return choice["val"]

    # ----------------------------
    # Clocks
    # ----------------------------

    def _spend_time_for_side(self, side_to_move: bool, spent_ms: int) -> None:
        # side_to_move is chess.WHITE / chess.BLACK before the move is pushed
        if side_to_move == chess.WHITE:
            self.clock_w_ms = max(0, self.clock_w_ms - spent_ms) + self.increment_ms
        else:
            self.clock_b_ms = max(0, self.clock_b_ms - spent_ms) + self.increment_ms

    def _clock_left_ms_for_side(self, side_to_move: bool) -> int:
        return self.clock_w_ms if side_to_move == chess.WHITE else self.clock_b_ms

    # ----------------------------
    # Model move pipeline (Maia once + timer + Stockfish time limit)
    # ----------------------------

    def maybe_make_model_move(self) -> None:
        if self.board.is_game_over():
            self.redraw()
            return
        if self.is_human_turn():
            self.redraw()
            return

        self.model_busy = True
        self.update_status()

        start_fen = self.board.fen()
        t = threading.Thread(target=self._model_move_worker, args=(start_fen,), daemon=True)
        t.start()

    def _build_prev5(self) -> List[float]:
        prev = self.last_ply_times_ms[-5:]
        prev = [float(x) for x in prev]
        while len(prev) < 5:
            prev.insert(0, 0.0)
        return prev

    @torch.no_grad()
    def _predict_think_ms(
        self,
        feats: torch.Tensor,             # [1, D]
        ply_idx: int,
        side_is_white: bool,
        prev5_ms: List[float],
        prev_clock_w_ms: int,
        prev_clock_b_ms: int,
    ) -> int:
        device = feats.device
        prev5 = torch.tensor([prev5_ms], device=device, dtype=torch.float32)  # [1, 5]
        prev5_feat = torch.log1p(torch.clamp(prev5, min=0.0))

        side = torch.tensor([1 if side_is_white else 0], device=device, dtype=torch.long)  # [1]
        cw = torch.tensor([float(prev_clock_w_ms)], device=device)  # [1]
        cb = torch.tensor([float(prev_clock_b_ms)], device=device)  # [1]
        clock_left_ms = torch.where(side == 1, cw, cb).clamp(min=0.0)  # [1]
        clock_feat = torch.log1p(clock_left_ms).unsqueeze(-1)          # [1,1]

        ply = torch.tensor([float(ply_idx)], device=device).unsqueeze(-1)  # [1,1]
        ply_feat = (ply / float(self.cfg.ply_norm)).clamp(min=0.0, max=10.0)

        x = torch.cat([feats, prev5_feat, clock_feat, ply_feat], dim=-1)  # [1, D+7]
        pred_log = self.timer_head(x).squeeze(0)  # scalar
        pred_ms = float(torch.expm1(pred_log).clamp(min=0.0).item())

        # Clamp to sane range and to remaining clock (minus safety)
        pred_ms_i = int(round(pred_ms))
        pred_ms_i = max(self.cfg.min_think_ms, min(self.cfg.max_think_ms, pred_ms_i))

        mover_clock = prev_clock_w_ms if side_is_white else prev_clock_b_ms
        pred_ms_i = min(pred_ms_i, max(self.cfg.min_think_ms, mover_clock - self.cfg.safety_ms))
        return max(self.cfg.min_think_ms, pred_ms_i)

    def _model_move_worker(self, start_fen: str) -> None:
        try:
            local_board = chess.Board(start_fen)
            side_is_white = (local_board.turn == chess.WHITE)

            # Clock snapshot (ms)
            prev_clock_w = int(self.clock_w_ms)
            prev_clock_b = int(self.clock_b_ms)

            if (side_is_white and prev_clock_w <= 0) or ((not side_is_white) and prev_clock_b <= 0):
                self.root.after(0, lambda: self._flag_loss(side_is_white))
                return

            # --- Maia forward ONCE (masked logits + features) ---
            feats, masked_logits = self.maia_runner.forward_once(
                start_fen, self.elo_self, self.elo_oppo
            )  # feats [1,D], logits [1,V]

            # Policy probs for display + sampling space
            probs = torch.softmax(masked_logits, dim=-1).squeeze(0)  # [V]
            topv, topi = torch.topk(probs, k=min(self.topk, probs.shape[0]))

            # Convert uci_eff -> actual uci for display
            side_token = start_fen.split(" ")[1]
            is_black_turn = (side_token == "b")

            def uci_eff_to_actual(uci_eff: str) -> str:
                return mirror_move(uci_eff) if is_black_turn else uci_eff

            cands_lines = []
            for p, idx in zip(topv.tolist(), topi.tolist()):
                uci_eff = self.idx_to_uci_eff[int(idx)] or "??"
                uci = uci_eff_to_actual(uci_eff)
                cands_lines.append(f"{uci:6s}  p={p:.4f}")
            cands_text = "\n".join(cands_lines)

            # --- Timer prediction using the SAME feats ---
            ply_idx = len(local_board.move_stack) + 1
            prev5 = self._build_prev5()
            think_ms = self._predict_think_ms(
                feats=feats,
                ply_idx=ply_idx,
                side_is_white=side_is_white,
                prev5_ms=prev5,
                prev_clock_w_ms=prev_clock_w,
                prev_clock_b_ms=prev_clock_b,
            )

            # Stockfish time limit (seconds)
            think_s = max(0.01, float(think_ms) / 1000.0)
            limit = chess.engine.Limit(time=think_s)

            # --- Choose final move with Stockfish "tree search" ---
            # Preferred: your biased-by-policy helper
            if batch_choose_moves_sf_topk_biased_by_policy is not None:
                # helper expects logits [B,V] (masked), and returns uci in *actual* coordinates
                # (Your helper likely already returns actual UCI strings consistent with board)
                res = batch_choose_moves_sf_topk_biased_by_policy(
                    fens=[start_fen],
                    logits_pi_masked=masked_logits,  # [1,V]
                    all_moves_dict=self.all_moves_dict,
                    engine=self.engine,
                    limit=limit,
                    sample=True,
                )
                # be tolerant to different tuple shapes
                tup = res[0]
                if isinstance(tup, (tuple, list)) and len(tup) >= 2:
                    # your prior code: _, uci_selected, _, _, _, _, _ = ...
                    uci_selected = tup[1]
                else:
                    raise RuntimeError(f"Unexpected SF helper return: {tup!r}")
            else:
                # Fallback: ask stockfish directly (no policy bias)
                mv = self.engine.play(local_board, limit).move
                uci_selected = mv.uci()

            # Spend predicted time for model move (clock bookkeeping happens on UI thread)
            self.root.after(
                0,
                lambda u=uci_selected, c=cands_text, fen=start_fen, tms=think_ms: self._apply_model_move(u, c, fen, tms),
            )

        except Exception as e:
            self.root.after(0, lambda err=e: self._handle_model_error(err))

    def _apply_model_move(self, uci_selected: str, cands_text: str, expected_fen: str, think_ms: int) -> None:
        self.set_candidates_text(cands_text)

        # Show prediction line
        self.pred.config(text=f"Model predicted think-time: {think_ms} ms  (SF limit={think_ms/1000.0:.2f}s)")

        # Discard stale if board changed
        if self.board.fen() != expected_fen:
            self.model_busy = False
            self.redraw()
            return

        mv = chess.Move.from_uci(uci_selected)
        if mv not in self.board.legal_moves:
            self.model_busy = False
            self.redraw()
            messagebox.showerror("Model error", f"Model produced illegal move: {uci_selected}")
            return

        # spend predicted time
        self._spend_time_for_side(self.board.turn, int(think_ms))
        self.last_ply_times_ms.append(int(think_ms))

        self.board.push(mv)
        self.model_busy = False
        self.turn_start_wall = time.time()
        self.redraw()

        if self.board.is_game_over():
            outcome = self.board.outcome()
            messagebox.showinfo("Game over", f"Result: {self.board.result()}\nOutcome: {outcome}")

    def _flag_loss(self, side_is_white_to_move: bool) -> None:
        self.model_busy = False
        self.redraw()
        loser = "White" if side_is_white_to_move else "Black"
        messagebox.showinfo("Flag", f"{loser} flagged (clock reached 0).")

    def _handle_model_error(self, e: Exception) -> None:
        self.model_busy = False
        self.redraw()
        messagebox.showerror("Inference error", str(e))

    # ----------------------------
    # Controls
    # ----------------------------

    def reset_game(self) -> None:
        if self.model_busy:
            return
        self.board.reset()
        self.clock_w_ms = 5 * 60 * 1000
        self.clock_b_ms = 5 * 60 * 1000
        self.last_ply_times_ms = []
        self.turn_start_wall = time.time()
        self.selected_square = None
        self.legal_targets = set()
        self.set_candidates_text("")
        self.pred.config(text="")
        self.redraw()
        self.maybe_make_model_move()

    def undo_2ply(self) -> None:
        if self.model_busy:
            return
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
            if len(self.last_ply_times_ms) >= 2:
                self.last_ply_times_ms = self.last_ply_times_ms[:-2]
        elif len(self.board.move_stack) == 1:
            self.board.pop()
            if len(self.last_ply_times_ms) >= 1:
                self.last_ply_times_ms = self.last_ply_times_ms[:-1]
        self.turn_start_wall = time.time()
        self.selected_square = None
        self.legal_targets = set()
        self.redraw()
        self.maybe_make_model_move()

    def flip_view(self) -> None:
        self.view_from_white = not self.view_from_white
        self.redraw()

    def swap_sides_new(self) -> None:
        if self.model_busy:
            return
        self.human_plays = "white" if self.human_plays == "black" else "black"
        self.view_from_white = (self.human_plays != "black")
        self.reset_game()


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", type=str, help="GM key used by your training scripts (e.g., magnus)")
    args = ap.parse_args()

    gm = args.gm_name.strip()

    # ---- paths (two PT files) ----
    # Prefer DPO checkpoint name; fallback to policy_best.pt
    policy_pt1 = Path(f"./processed/single_gm/train_val/{gm}/policy_dpo_best.pt")
    policy_pt2 = Path(f"./processed/single_gm/train_val/{gm}/policy_best.pt")
    if policy_pt1.exists():
        policy_pt = policy_pt1
    elif policy_pt2.exists():
        policy_pt = policy_pt2
    else:
        raise FileNotFoundError(
            f"Could not find policy weights at:\n"
            f"  {policy_pt1}\n  {policy_pt2}"
        )

    timer_pt = Path(f"./processed/single_gm/time_per_move/train_val/{gm}/timer_head_best.pt")
    if not timer_pt.exists():
        raise FileNotFoundError(f"Could not find timer head at: {timer_pt}")

    device = device_auto()
    print(f"[init] device={device}")
    print(f"[init] policy_pt={policy_pt}")
    print(f"[init] timer_pt={timer_pt}")

    # ---- Load Maia2 base (blitz) + GM weights ----
    maia = maia_model.from_pretrained(type="blitz", device=device)
    sd_policy = load_state_dict_fuzzy(policy_pt, map_location="cpu")
    missing, unexpected = maia.load_state_dict(sd_policy, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (showing 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (showing 10): {unexpected[:10]}")

    if device == "cuda":
        maia = maia.cuda()
    elif device == "mps":
        maia = maia.to("mps")
    else:
        maia = maia.cpu()
    maia.eval()

    # ---- prepare Maia vocab ----
    all_moves_dict, elo_dict, _ = inference.prepare()

    # ---- timer head ----
    timer_head, feat_dim = build_timer_head_from_ckpt(timer_pt, device=device)
    print(f"[timer] inferred feat_dim={feat_dim} (from timer head in_dim)")

    # ---- Maia runner (hook + logits fallback) ----
    timer_cfg = TimerConfig(
        hook_layer="last_ln",
        logits_feature="masked_logits",
        min_think_ms=50,
        max_think_ms=10_000,
        safety_ms=50,
    )
    maia_runner = MaiaOncePerPly(
        maia=maia,
        all_moves_dict=all_moves_dict,
        elo_dict=elo_dict,
        device=device,
        timer_cfg=timer_cfg,
    )

    # ---- Stockfish ----
    sf_path = find_stockfish()
    engine = make_stockfish_engine(sf_path, threads=16, hash_mb=2048)
    print(f"[sf] path={sf_path} threads=16 hash=2048")

    # ---- GUI ----
    board = chess.Board(chess.STARTING_FEN)
    root = tk.Tk()

    app = TimedKasparovNetGUI(
        root=root,
        gm_name=gm,
        board=board,
        maia_runner=maia_runner,
        timer_head=timer_head,
        feat_dim=feat_dim,
        all_moves_dict=all_moves_dict,
        engine=engine,
        timer_cfg=timer_cfg,
        elo_self=2800,
        elo_oppo=2800,
        topk=10,
        square_size=72,
        start_clock_ms=5 * 60 * 1000,
    )

    def on_close():
        try:
            engine.quit()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
