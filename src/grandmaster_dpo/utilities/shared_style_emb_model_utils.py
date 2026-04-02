from __future__ import annotations

import chess
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..train.style_embeddings_for_gms.dataset_schema import TrainConfig

# Keep behavior aligned with your training script.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ---------------------------------------------------------------------
# Canonical mappings
# ---------------------------------------------------------------------

PIECE_TO_ID: Dict[str, int] = {
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
}

GAME_TYPE_TO_ID: Dict[str, int] = {
    "blitz": 1,
    "rapid": 2,
    "classical": 3,
}

PHASE_TO_ID: Dict[str, int] = {
    "opening": 1,
    "middlegame": 2,
    "endgame": 3,
}

PROMO_TO_ID: Dict[str, int] = {
    "q": 1,
    "r": 2,
    "b": 3,
    "n": 4,
}

NUM_PIECE_CLASSES_WITH_EMPTY = 13
NUM_PIECE_PLANES = 12
NUM_BOARDS = 5
DEFAULT_OPPONENT_CONTEXT_DIM = 32


# ---------------------------------------------------------------------
# Repro / device
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Use both python random and numpy/torch so training/eval behavior
    stays aligned across scripts.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(device_arg: str = "auto") -> torch.device:
    """
    Training semantics should be the source of truth:
    prefer MPS, then CPU, but also allow explicit override and CUDA.
    """
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.backends.mps.is_available():
        print("[device] using MPS (Apple GPU)")
        return torch.device("mps")

    if torch.cuda.is_available():
        print("[device] using CUDA")
        return torch.device("cuda")

    print("[device] using CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# Variant helpers
# ---------------------------------------------------------------------

def model_variant_uses_game_type(model_variant_name: str) -> bool:
    return model_variant_name in {"phi1", "phi3"}


def model_variant_uses_opponent_context(model_variant_name: str) -> bool:
    return model_variant_name == "phi3"


# ---------------------------------------------------------------------
# Move encoding
# ---------------------------------------------------------------------

def encode_square(s: str) -> int:
    """
    Training-compatible square encoding.
    Returns 0 on malformed input instead of raising.
    """
    if len(s) != 2:
        return 0

    file_idx = ord(s[0]) - ord("a")
    rank_idx = ord(s[1]) - ord("1")

    if not (0 <= file_idx < 8 and 0 <= rank_idx < 8):
        return 0

    return 1 + rank_idx * 8 + file_idx


def encode_move_uci(move: str) -> np.ndarray:
    """
    Fixed-length move tokenization: [from, to, promo]
    """
    out = np.zeros(3, dtype=np.uint8)
    if len(move) < 4:
        return out

    out[0] = encode_square(move[:2])
    out[1] = encode_square(move[2:4])

    if len(move) >= 5:
        out[2] = PROMO_TO_ID.get(move[4].lower(), 0)

    return out


# ---------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------

def fen_to_piece_tokens_64(fen: str) -> np.ndarray:
    """
    Compact board representation: [64] with 0=empty, 1..12=piece ids.
    This is ideal for caching / memmap.
    """
    board = fen.split()[0]
    out = np.zeros(64, dtype=np.uint8)

    row = 0
    col = 0
    for ch in board:
        if ch == "/":
            row += 1
            col = 0
        elif ch.isdigit():
            col += int(ch)
        else:
            idx = row * 8 + col
            if not (0 <= idx < 64):
                raise ValueError(f"Bad FEN board indexing for fen={fen}")
            out[idx] = PIECE_TO_ID[ch]
            col += 1

    return out


def piece_tokens_64_to_planes(board_tokens_64: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Convert a single compact board [64] with ids 0..12 into training-style
    one-hot planes [12, 8, 8].
    """
    if not isinstance(board_tokens_64, torch.Tensor):
        board_tokens_64 = torch.as_tensor(board_tokens_64)

    x = board_tokens_64.long()  # [64]
    x = F.one_hot(x, num_classes=NUM_PIECE_CLASSES_WITH_EMPTY)[..., 1:]  # [64, 12]
    x = x.transpose(0, 1).reshape(NUM_PIECE_PLANES, 8, 8).float()
    return x


def board_tokens_5x64_to_planes(boards_5x64: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Convert compact cached boards [5, 64] into training-style model boards
    [5, 12, 8, 8].
    """
    if not isinstance(boards_5x64, torch.Tensor):
        boards_5x64 = torch.as_tensor(boards_5x64)

    x = boards_5x64.long()  # [5, 64]
    x = F.one_hot(x, num_classes=NUM_PIECE_CLASSES_WITH_EMPTY)[..., 1:]  # [5,64,12]
    x = x.permute(0, 2, 1).reshape(NUM_BOARDS, NUM_PIECE_PLANES, 8, 8).float()
    return x


def raw_example_to_cached_arrays(ex: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.uint8]:
    """
    Canonical raw -> compact cache representation.
    Returns:
      boards: [5,64] uint8
      move:   [3] uint8
      game_type: scalar uint8
    """
    default_fen = chess.Board().fen()

    boards = np.stack(
        [
            fen_to_piece_tokens_64(ex.get("board_t_minus_4", default_fen) or default_fen),
            fen_to_piece_tokens_64(ex.get("board_t_minus_3", default_fen) or default_fen),
            fen_to_piece_tokens_64(ex.get("board_t_minus_2", default_fen) or default_fen),
            fen_to_piece_tokens_64(ex.get("board_t_minus_1", default_fen) or default_fen),
            fen_to_piece_tokens_64(ex.get("board_t", default_fen) or default_fen),
        ],
        axis=0,
    ).astype(np.uint8, copy=False)

    move = encode_move_uci(ex["move_played"])
    game_type = np.uint8(GAME_TYPE_TO_ID.get(ex.get("game_type", ""), 0))
    return boards, move, game_type


def cached_arrays_to_model_features(
    *,
    boards_5x64: np.ndarray | torch.Tensor,
    move_3: np.ndarray | torch.Tensor,
    game_type: Optional[int | np.integer | torch.Tensor],
    variant_name: str,
    opponent_context_dim: int = DEFAULT_OPPONENT_CONTEXT_DIM,
) -> Dict[str, torch.Tensor]:
    """
    Canonical compact cache -> model-facing tensors, preserving training semantics.
    """
    feat: Dict[str, torch.Tensor] = {
        "boards": board_tokens_5x64_to_planes(boards_5x64),         # [5,12,8,8]
        "move": torch.as_tensor(move_3, dtype=torch.long),          # [3]
    }

    if model_variant_uses_game_type(variant_name):
        feat["game_type"] = torch.as_tensor(
            0 if game_type is None else int(game_type),
            dtype=torch.long,
        )

    if model_variant_uses_opponent_context(variant_name):
        feat["opponent_context"] = torch.zeros(opponent_context_dim, dtype=torch.float32)

    return feat


def raw_example_to_model_features(
    ex: Dict[str, Any],
    variant_name: str,
    opponent_context_dim: int = DEFAULT_OPPONENT_CONTEXT_DIM,
) -> Dict[str, torch.Tensor]:
    """
    Raw json example -> model-facing tensors, using the same semantics as
    training but routed through the compact representation so cache/eval
    stay identical.
    """
    boards, move, game_type = raw_example_to_cached_arrays(ex)
    return cached_arrays_to_model_features(
        boards_5x64=boards,
        move_3=move,
        game_type=game_type,
        variant_name=variant_name,
        opponent_context_dim=opponent_context_dim,
    )


# ---------------------------------------------------------------------
# Batch/device helpers
# ---------------------------------------------------------------------

def stack_feature_dicts(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, List[torch.Tensor]] = {}
    for item in items:
        for k, v in item.items():
            out.setdefault(k, []).append(v)
    return {k: torch.stack(v, dim=0) for k, v in out.items()}


def move_feature_dict_to_device(
    batch_part: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch_part.items()
    }


# ---------------------------------------------------------------------
# Checkpoint / model loading
# ---------------------------------------------------------------------


def resolve_checkpoint(model_dir: Path, checkpoint_name: str = "best") -> Path:
    candidate = model_dir / checkpoint_name
    if candidate.exists():
        return candidate

    if checkpoint_name == "best":
        best_pt = model_dir / "best.pt"
        if best_pt.exists():
            return best_pt

    if checkpoint_name == "last":
        epoch_pts = sorted(model_dir.glob("epoch_*.pt"))
        if epoch_pts:
            return epoch_pts[-1]

    raise FileNotFoundError(
        f"Could not resolve checkpoint '{checkpoint_name}' in {model_dir}"
    )


def assert_model_dir_matches_variant(model_dir: Path, cfg: TrainConfig) -> None:
    variant = cfg.model.variant_name
    if variant not in str(model_dir):
        raise ValueError(
            f"Checkpoint config says variant={variant}, "
            f"but model_dir does not contain that substring: {model_dir}"
        )

# Makes it so a single example can be passed to the model without needing to add a batch dimension manually.
def add_batch_dim(feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.unsqueeze(0) for k, v in feats.items()}

class BoardCNN(nn.Module):
    def __init__(self, board_embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, board_embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 12, 8, 8]
        return self.net(x)

class StyleEncoder(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        m = cfg.model

        self.num_boards = 5
        self.board_embed_dim = 128

        self.move_embed = nn.Embedding(70, m.token_embed_dim)
        self.game_type_embed = nn.Embedding(8, 16)

        self.board_cnn = BoardCNN(
            board_embed_dim=self.board_embed_dim,
            dropout=m.dropout,
        )

        board_in_dim = self.num_boards * self.board_embed_dim
        move_in_dim = 3 * m.token_embed_dim

        aux_dim = 0
        if m.variant_name in {"phi1", "phi3"}:
            aux_dim += 16
        if m.variant_name == "phi3":
            aux_dim += 32

        hidden_dim = m.hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(board_in_dim + move_in_dim + aux_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(m.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(m.dropout),
            nn.Linear(hidden_dim, m.embedding_dim),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        boards = feats["boards"]   # [B, 5, 12, 8, 8]
        move = feats["move"]       # [B, 3]

        bsz, num_boards, channels, h, w = boards.shape

        # flatten batch and board-time dimension so each board goes through the same CNN
        board_inputs = boards.reshape(bsz * num_boards, channels, h, w)   # [B*5, 12, 8, 8]

        # encode each board independently
        board_vecs = self.board_cnn(board_inputs)                         # [B*5, board_embed_dim]

        # concatenate the 5 board embeddings
        board_vecs = board_vecs.reshape(bsz, num_boards * self.board_embed_dim)

        # encode move
        m_emb = self.move_embed(move).reshape(bsz, -1)

        parts = [board_vecs, m_emb]

        if "game_type" in feats:
            gt = self.game_type_embed(feats["game_type"])
            parts.append(gt)

        if "opponent_context" in feats:
            parts.append(feats["opponent_context"])

        x = torch.cat(parts, dim=-1)
        z = self.mlp(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


