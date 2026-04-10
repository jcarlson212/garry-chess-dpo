from __future__ import annotations

import json
from typing import Any, Dict, List

from torch.utils.data import Dataset


class DpoPairs(Dataset):
    def __init__(self, jsonl_path: str, debug: bool = False):
        self.rows: List[Dict[str, Any]] = []
        self.debug = debug
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
            print(f"Loaded {len(self.rows)} rows from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        p = r.get("prompt", {}) or {}
        meta = r.get("meta", {}) or {}

        gh = meta.get("game_header_hash")
        game_id = str(gh)

        return {
            "fen": p["fen"],
            "elo_self": int(p.get("elo_self", 2800)),
            "elo_oppo": int(p.get("elo_oppo", 2800)),
            "chosen": r["chosen"],
            "rejected": r["rejected"],
            "game_id": game_id,
            "ply_idx": int(meta.get("ply_idx", -1)),
            "fullmove_number": int(meta.get("fullmove_number", -1)),
            "side_to_move": str(meta.get("side_to_move", "")),
            "opening_prefix_uci_20": meta.get("opening_prefix_uci_20") or [],
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {
        "fen": [],
        "elo_self": [],
        "elo_oppo": [],
        "chosen": [],
        "rejected": [],
        "game_id": [],
        "ply_idx": [],
        "fullmove_number": [],
        "side_to_move": [],
        "opening_prefix_uci_20": [],
    }
    for b in batch:
        for k in out:
            out[k].append(b.get(k))
    return out