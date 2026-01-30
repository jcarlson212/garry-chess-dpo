# kasparovnet_gui.py
# A simple graphical Maia2 DPO “play vs single GM tuned policy” GUI using Tkinter.
#
# Features
# - Loads policy weights from: ./processed/single_gm/train_val/{gm_name}/policy_best.pt
# - Same Maia2 inference path as your scripts (inference.prepare + inference.inference_each)
# - Click-to-move (two clicks) with legal-move validation
# - Promotion chooser popup
# - Side panel shows top-k candidates + win_prob
# - Model can play White or Black
# - Keeps UI responsive by running model move in a background thread
#
# Usage:
#   python kasparovnet_gui.py --gm_name magnus --maia_type blitz --device mps --human_plays white
#   python kasparovnet_gui.py --gm_name magnus --human_plays black --elo_self 2800 --elo_oppo 2800
#
# Notes:
# - “Nice graphics”: this uses Unicode chess symbols on a colored board. No external assets required.
# - If your terminal / system font renders chess symbols poorly, try installing a font that supports them.
#
from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import chess
import torch
import tkinter as tk
from tkinter import messagebox

from maia2 import inference, model as maia_model
from grandmaster_dpo.tree_search.maia_beam_search_utilities import choose_move_depth_limited

#def choose_move_depth_limited(policy, prepared, board: chess.Board, elo_self: int, elo_oppo: int,
#                             depth: int = 4, beam: int = 12) -> tuple[str, str]:

PREFERRED_CHESS_FONTS = [
    "Noto Sans Symbols 2",
    "Noto Sans Symbols",
    "DejaVu Sans",
    "Segoe UI Symbol",      # Windows
    "Apple Symbols",        # macOS (sometimes helps)
    "Symbola",
    "Arial Unicode MS",
    "Helvetica",
]

def pick_glyph_font(size: int) -> tuple[str, int]:
    """
    Pick a font likely to render chess glyphs well.
    Tk doesn't give a perfect way to probe glyph coverage, so we try a list.
    """
    try:
        import tkinter.font as tkfont
        available = set(tkfont.families())
        for f in PREFERRED_CHESS_FONTS:
            if f in available:
                return (f, size)
    except Exception:
        pass
    return ("Helvetica", size)


# ----------------------------
# Model loading
# ----------------------------

def device_from_str(s: str) -> str:
    s = s.lower()
    if s in ("gpu",):
        return "cuda"
    return s

def load_policy(
    *,
    maia_type: str,
    device: str,
    policy_pt: Path,
) -> torch.nn.Module:
    """Load base Maia2 then load your DPO weights on top."""
    m = maia_model.from_pretrained(type=maia_type, device=device)

    sd = torch.load(policy_pt, map_location="cpu")
    # tolerate common checkpoint formats
    if isinstance(sd, dict) and "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
        sd = sd["model_state_dict"]
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported checkpoint object type: {type(sd)}")

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (showing 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (showing 10): {unexpected[:10]}")

    # Ensure on requested device
    if device == "cuda":
        m = m.cuda()
    elif device == "mps":
        m = m.to("mps")
    else:
        m = m.cpu()

    m.eval()
    return m


# ----------------------------
# Inference helpers
# ----------------------------

def topk_from_move_probs(move_probs: Dict[str, float], k: int) -> list[Tuple[str, float]]:
    # inference_each typically returns dict already sorted by prob desc, but don’t rely on it.
    return sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)[:k]

def pick_best_move(move_probs: Dict[str, float]) -> str:
    return max(move_probs.items(), key=lambda kv: kv[1])[0]


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

class KasparovNetGUI:
    def __init__(
        self,
        *,
        root: tk.Tk,
        board: chess.Board,
        policy: torch.nn.Module,
        prepared: object,
        elo_self: int,
        elo_oppo: int,
        human_plays: str,
        topk: int,
        square_size: int = 72,
    ):
        self.root = root
        self.board = board
        self.policy = policy
        self.prepared = prepared
        self.elo_self = elo_self
        self.elo_oppo = elo_oppo
        self.human_plays = human_plays  # "white" or "black"
        self.topk = topk

        self.square_size = square_size
        self.margin = 16
        self.canvas_size = self.margin * 2 + self.square_size * 8

        # Selection state
        self.selected_square: Optional[int] = None
        self.legal_targets: set[int] = set()

        # Busy flag for model thinking
        self.model_busy = False

        # Layout
        self.root.title("KasparovNet (Maia2 DPO) — Click-to-move")
        self.root.geometry(f"{self.canvas_size + 360}x{self.canvas_size + 40}")

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

        self.cands_label = tk.Label(self.right, text="Model candidates (top-k):", anchor="w", font=("Helvetica", 11, "bold"))
        self.cands_label.pack(fill="x", pady=(10, 0))

        self.cands = tk.Text(self.right, height=18, width=42, font=("Menlo", 11))
        self.cands.pack(fill="both", expand=True)

        btn_row = tk.Frame(self.right)
        btn_row.pack(fill="x", pady=(10, 0))

        self.btn_new = tk.Button(btn_row, text="New Game", command=self.reset_game)
        self.btn_new.pack(side="left")

        self.btn_undo = tk.Button(btn_row, text="Undo (2 ply)", command=self.undo_2ply)
        self.btn_undo.pack(side="left", padx=(8, 0))

        self.btn_flip = tk.Button(btn_row, text="Flip Board", command=self.flip_view)
        self.btn_flip.pack(side="left", padx=(8, 0))

        self.view_from_white = True  # board orientation
        if self.human_plays == "black":
            self.view_from_white = (self.human_plays != "black")  # white=True, black=False


        # Draw initial
        self.redraw()

        # If model goes first
        self.maybe_make_model_move()


    # ----------------------------
    # Board rendering
    # ----------------------------

    def square_to_xy(self, square: int) -> Tuple[int, int]:
        # square: 0=a1 ... 63=h8
        file = chess.square_file(square)  # 0..7 (a..h)
        rank = chess.square_rank(square)  # 0..7 (1..8)

        # determine render coordinates based on view
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
        light  = "#EEEED2"
        dark   = "#769656"
        sel    = "#F6F669"  # selection yellow
        target = "#BACA44"  # legal target
        last   = "#CDD26A"  # last move

        # last move highlighting
        last_from = last_to = None
        if len(self.board.move_stack) > 0:
            mv = self.board.move_stack[-1]
            last_from, last_to = mv.from_square, mv.to_square

        # draw squares
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

        # coordinates
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

        # ---- draw pieces (glyphs) with strong contrast ----

        # Pick a symbol-friendly font if available; otherwise fall back.
        preferred_fonts = [
            "Noto Sans Symbols 2",
            "Noto Sans Symbols",
            "DejaVu Sans",
            "Segoe UI Symbol",   # Windows
            "Apple Symbols",     # macOS
            "Symbola",
            "Arial Unicode MS",
            "Helvetica",
        ]
        try:
            fams = set(tkfont.families())
            font_family = next((f for f in preferred_fonts if f in fams), "Helvetica")
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

            # Explicit colors + outline so glyph style can't "invert" visually
            if piece.color == chess.WHITE:
                fill = "#FFFFFF"
                outline = "#111111"
            else:
                fill = "#111111"
                outline = "#FFFFFF"

            # stroke effect by drawing around the main glyph
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                self.canvas.create_text(cx + dx, cy + dy, text=sym, font=glyph_font, fill=outline)

            self.canvas.create_text(cx, cy, text=sym, font=glyph_font, fill=fill)

        self.update_status()


    def update_status(self) -> None:
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        human_turn = (self.human_plays == "white" and self.board.turn == chess.WHITE) or (
            self.human_plays == "black" and self.board.turn == chess.BLACK
        )
        over = self.board.is_game_over()
        if over:
            outcome = self.board.outcome()
            result = self.board.result()
            msg = f"Game over: {result}"
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

    def on_click(self, event) -> None:
        if self.board.is_game_over():
            return
        if self.model_busy:
            return

        sq = self.xy_to_square(event.x, event.y)
        if sq is None:
            return

        # Only allow selecting/moving on human's turn
        if not self.is_human_turn():
            return

        if self.selected_square is None:
            # select if piece belongs to side to move
            piece = self.board.piece_at(sq)
            if piece is None:
                return
            if piece.color != self.board.turn:
                return
            self.selected_square = sq
            self.legal_targets = {m.to_square for m in self.board.legal_moves if m.from_square == sq}
            self.redraw()
            return

        # second click: attempt move
        from_sq = self.selected_square
        to_sq = sq

        # deselect if clicking same square
        if to_sq == from_sq:
            self.clear_selection()
            return

        # build a move; handle promotion if needed
        move = self.build_move_with_possible_promotion(from_sq, to_sq)
        if move is None:
            # canceled promotion dialog
            self.clear_selection()
            return

        if move not in self.board.legal_moves:
            # illegal; keep selection if user clicked another own piece
            piece = self.board.piece_at(to_sq)
            if piece is not None and piece.color == self.board.turn:
                self.selected_square = to_sq
                self.legal_targets = {m.to_square for m in self.board.legal_moves if m.from_square == to_sq}
                self.redraw()
                return

            messagebox.showinfo("Illegal move", "That move is not legal in this position.")
            self.clear_selection()
            return

        self.board.push(move)
        self.clear_selection()

        if self.board.is_game_over():
            self.redraw()
            return

        self.maybe_make_model_move()

    def clear_selection(self) -> None:
        self.selected_square = None
        self.legal_targets = set()
        self.redraw()

    def is_human_turn(self) -> bool:
        return (self.human_plays == "white" and self.board.turn == chess.WHITE) or (
            self.human_plays == "black" and self.board.turn == chess.BLACK
        )

    def build_move_with_possible_promotion(self, from_sq: int, to_sq: int) -> Optional[chess.Move]:
        piece = self.board.piece_at(from_sq)
        if piece is None:
            return None

        # detect promotion scenario
        is_pawn = piece.piece_type == chess.PAWN
        if is_pawn:
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
    # Model move
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

        # snapshot state for the worker
        start_fen = self.board.fen()
        t = threading.Thread(target=self._model_move_worker, args=(start_fen,), daemon=True)
        t.start()

    def _model_move_worker(self, start_fen: str) -> None:
        try:
            # Work on a snapshot only (never touch self.board in this thread)
            local_board = chess.Board(start_fen)

            best, text = choose_move_depth_limited(
                self.policy, self.prepared, local_board,
                self.elo_self, self.elo_oppo,
                depth=5, beam=4,
                inference=inference,
            )

            # Apply only if we’re still on the same position
            self.root.after(
                0,
                lambda u=best, t=text, fen=start_fen: self._apply_model_move(u, t, fen),
            )
        except Exception as e:
            self.root.after(0, lambda err=e: self._handle_model_error(err))


    def _apply_model_move(self, best_uci: str, candidates_text: str, expected_fen: str) -> None:
        self.set_candidates_text(candidates_text)

        # Discard stale move if position changed since we started thinking
        if self.board.fen() != expected_fen:
            self.model_busy = False
            self.redraw()
            return

        mv = chess.Move.from_uci(best_uci)
        if mv not in self.board.legal_moves:
            self.model_busy = False
            self.redraw()
            messagebox.showerror("Model error", f"Model produced illegal move: {best_uci}")
            return

        self.board.push(mv)
        self.model_busy = False
        self.redraw()

        if self.board.is_game_over():
            outcome = self.board.outcome()
            messagebox.showinfo("Game over", f"Result: {self.board.result()}\nOutcome: {outcome}")

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
        self.clear_selection()
        self.set_candidates_text("")
        self.maybe_make_model_move()

    def undo_2ply(self) -> None:
        if self.model_busy:
            return
        # undo model + human (or vice versa) if possible
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
        elif len(self.board.move_stack) == 1:
            self.board.pop()
        self.clear_selection()
        self.set_candidates_text("")
        self.maybe_make_model_move()

    def flip_view(self) -> None:
        self.view_from_white = not self.view_from_white
        self.redraw()


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", required=True, help="Folder/key used by training script (e.g., magnus)")
    ap.add_argument("--maia_type", default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "gpu", "mps"])
    ap.add_argument("--elo_self", type=int, default=2800)
    ap.add_argument("--elo_oppo", type=int, default=2800)
    ap.add_argument("--human_plays", default="black", choices=["white", "black"])
    ap.add_argument("--start_fen", default=chess.STARTING_FEN)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--square_size", type=int, default=72)
    args = ap.parse_args()

    device = device_from_str(args.device)

    # Match your training/eval directory conventions
    policy_pt = Path(f"./processed/single_gm/train_val/{args.gm_name}/policy_best.pt")
    if not policy_pt.exists():
        raise FileNotFoundError(
            f"Could not find policy weights at: {policy_pt}\n"
            f"Expected training output at ./processed/single_gm/train_val/{args.gm_name}/policy_best.pt"
        )

    policy = load_policy(maia_type=args.maia_type, device=device, policy_pt=policy_pt)
    prepared = inference.prepare()

    board = chess.Board(args.start_fen)

    root = tk.Tk()
    app = KasparovNetGUI(
        root=root,
        board=board,
        policy=policy,
        prepared=prepared,
        elo_self=args.elo_self,
        elo_oppo=args.elo_oppo,
        human_plays=args.human_plays,
        topk=args.topk,
        square_size=args.square_size,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
