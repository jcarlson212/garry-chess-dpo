from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ExampleRow:
    example_id: str
    player_id: str
    opponent_id: str
    game_id: str
    ply_idx: int
    move_color: str
    game_type: str
    opening_bucket: str
    phase: str
    board_t_minus_5: str
    board_t_minus_4: str
    board_t_minus_3: str
    board_t_minus_2: str
    board_t_minus_1: str
    board_t: str
    move_played: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExampleRow":
        return cls(
            example_id=d["example_id"],
            player_id=d["player_id"],
            opponent_id=d.get("opponent_id", ""),
            game_id=d["game_id"],
            ply_idx=int(d["ply_idx"]),
            move_color=d["move_color"],
            game_type=d["game_type"],
            opening_bucket=d["opening_bucket"],
            phase=d["phase"],
            board_t_minus_5=d["board_t_minus_5"],
            board_t_minus_4=d["board_t_minus_4"],
            board_t_minus_3=d["board_t_minus_3"],
            board_t_minus_2=d["board_t_minus_2"],
            board_t_minus_1=d["board_t_minus_1"],
            board_t=d["board_t"],
            move_played=d["move_played"],
        )


@dataclass(frozen=True)
class PairRow:
    anchor: ExampleRow
    positives: List[ExampleRow]
    negatives: List[ExampleRow]
    meta: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PairRow":
        return cls(
            anchor=ExampleRow.from_dict(d["anchor"]),
            positives=[ExampleRow.from_dict(x) for x in d.get("positives", [])],
            negatives=[ExampleRow.from_dict(x) for x in d.get("negatives", [])],
            meta=d.get("meta", {}),
        )


@dataclass(frozen=True)
class ModelConfig:
    name: str
    embedding_dim: int = 256
    board_vocab_size: int = 32
    move_vocab_size: int = 5000
    token_embed_dim: int = 96
    hidden_dim: int = 256
    num_board_layers: int = 2
    dropout: float = 0.1
    variant_name: str = "phi0"  # phi0 / phi1 / phi3
    use_game_type: bool = False
    use_phase: bool = False
    use_opening: bool = False


@dataclass(frozen=True)
class TrainConfig:
    study_name: str
    train_dir: str
    eval_dir: str
    trained_models_root: str
    training_summary_root: str
    max_train_rows: int
    max_eval_rows: int

    pair_variant: str = "v1"

    seed: int = 42
    epochs: int = 5
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    tau: float = 0.07
    grad_clip_norm: float = 1.0

    num_workers: int = 8
    prefetch_factor: int = 16
    pin_memory: bool = False  # MPS generally does not benefit the same way as CUDA
    persistent_workers: bool = True

    max_steps_per_epoch: Optional[int] = None
    max_eval_batches: Optional[int] = 300

    save_every_epoch: bool = True
    keep_last_n_checkpoints: int = 2

    timeout_minutes: Optional[int] = None

    model: ModelConfig = field(default_factory=lambda: ModelConfig(name="default"))

    def run_name(self) -> str:
        run_name_str = (
            f"{self.study_name}"
            f"__pair-{self.pair_variant}"
            f"__phi-{self.model.variant_name}"
            f"__edim-{self.model.embedding_dim}"
            f"__bs-{self.batch_size}"
            f"__lr-{self.lr:g}"
            f"__tau-{self.tau:g}"
            f"__seed-{self.seed}"
        )
        if self.max_train_rows < 1_000_000_000:
            run_name_str += f"__debug"
        return run_name_str

    def checkpoint_dir(self) -> Path:
        return Path(self.trained_models_root) / self.run_name()

    def summary_path(self) -> Path:
        return Path(self.training_summary_root) / f"{self.run_name()}.jsonl"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["run_name"] = self.run_name()
        return d
