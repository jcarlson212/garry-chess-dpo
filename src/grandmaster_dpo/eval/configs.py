from dataclasses import dataclass, fields
from typing import Optional, Tuple


@dataclass
class OpeningLogitDistConfig:
    # Which early plies to track (absolute ply count from FEN):
    # 0 = white's first move, 1 = black's first reply, etc.
    plies: Tuple[int, ...] = (0, 1)

    # Use probabilities derived from logits/temperature
    temperature: float = 1.0

    # For efficiency + stability, only allocate probability mass over top-K moves
    # (renormalized within top-K). Set to 0 to use full distribution (expensive).
    topk: int = 50

    # If you set min_prob_mass < 1, you can grow K until cumulative prob reaches this mass.
    # (Requires sorting; only meaningful when topk==0 or you choose to implement dynamic K.)
    min_prob_mass: Optional[float] = None

@dataclass
class SfConfig:
    stockfish_path: str
    depth: int = 10
    multipv_topk: int = 10
    uci_elo: Optional[int] = None           # None => full strength
    threads: int = 1
    hash_mb: int = 128
    timeout_s: float = 30.0

    restrict_cp_window: Optional[int] = 60  # keep moves with cp >= best_cp - window
    temperature: float = 1.0
    sample: bool = False
    seed: int = 0
    eps: float = 1e-12

    use_gibbs: bool = False
    k: int = 40

    @classmethod
    def from_dict(cls, data):
        # Get names of all fields expected by the __init__ method
        field_names = {f.name for f in fields(cls) if f.init}
        # Filter data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)
