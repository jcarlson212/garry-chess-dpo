from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from grandmaster_dpo.eval.types import _SFCandidate


class DpoPairs(Dataset):
    def __init__(self, jsonl_path: Optional[str] = None, sf_cached_pairs: Optional[SFCachedPairs] = None, debug: bool = False):
        self.rows: List[Dict[str, Any]] = []
        self.debug = debug

        if sf_cached_pairs:
            self.rows = sf_cached_pairs.rows 
        else:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.rows.append(json.loads(line))

        self.rows = sorted(self.rows, key=lambda r: (r["meta"]["game_header_hash"], r["meta"]["ply_idx"]))

        self.game_id_and_ply_to_prev_10_plys = {}
        self.game_id_and_ply_to_fut_10_plys = {}

        def create_window_item(rows, index, target_game):
            if index < 0:
                return None 
            elif index >= len(rows):
                return None
            else:
                if rows[index]["meta"]["game_header_hash"] != target_game:
                    return None
                return rows[index]

        for i, r in enumerate(self.rows):
            hash_key = f'{r["meta"]["game_header_hash"]}_{r["meta"]["ply_idx"]}'
            self.game_id_and_ply_to_prev_10_plys[hash_key] = [create_window_item(self.rows, i, r["meta"]["game_header_hash"]) for i in range(i-10, i)]
            self.game_id_and_ply_to_fut_10_plys[hash_key] = [create_window_item(self.rows, i, r["meta"]["game_header_hash"]) for i in range(i+1, i+11)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        print(f"current row is {r}")
        p = r.get("prompt", {}) or {}
        meta = r.get("meta", {}) or {}

        # Correct location: meta['game_header_hash'] (not top-level).
        gh = meta.get("game_header_hash")
        game_id = str(gh)

        if self.debug and idx < 3:
            print(f"r keys: {r.keys()}")
            print(f"meta keys: {meta.keys()}")
            print(f"p keys: {p.keys()}")
            print(f"meta.game_header_hash: {gh!r}")
            print(f"computed game_id: {game_id}")

        # Minimal required keys + metadata keys used by eval
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
            "meta": meta,
        }
    
    def shuffle(self, seed: int) -> None:
        import random 
        random.seed(seed)
        random.shuffle(self.rows)

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Collate *all* fields your eval loop may read.
    """
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
        "meta": [],
    }
    for b in batch:
        for k in out:
            out[k].append(b.get(k))
    return out

class SFCachedPairs(DpoPairs):
    def __init__(self, stockfish_metadata_path: str, debug: bool = False):
        self.rows: List[Dict[str, Any]] = []
        self.debug = debug
        with open(stockfish_metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
            print(f"Loaded {len(self.rows)} rows from {stockfish_metadata_path}")
        self.rows = sorted(self.rows, key=lambda r: (r["meta"]["game_header_hash"], r["meta"]["ply_idx"]))

        self.game_id_and_ply_to_prev_10_plys = {}
        self.game_id_and_ply_to_fut_10_plys = {}

        def create_window_item(rows, index, target_game):
            if index < 0:
                return None 
            elif index >= len(rows):
                return None
            else:
                if rows[index]["meta"]["game_header_hash"] != target_game:
                    return None
                return rows[index]

        for i, r in enumerate(self.rows):
            hash_key = f'{r["meta"]["game_header_hash"]}_{r["meta"]["ply_idx"]}'
            self.game_id_and_ply_to_prev_10_plys[hash_key] = [create_window_item(self.rows, i, r["meta"]["game_header_hash"]) for i in range(i-10, i)]
            self.game_id_and_ply_to_fut_10_plys[hash_key] = [create_window_item(self.rows, i, r["meta"]["game_header_hash"]) for i in range(i+1, i+11)]

        

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        return r

def merge_inf_sf_and_reference_sf(cached_inference_stockfish_dpo_pairs: DpoPairs, cached_reference_stockfish_dpo_pairs: DpoPairs) -> DpoPairs:
    ref_output_temp_index = {}

    def build_key(game_id: str, ply_idx: int) -> str:
        return f"{game_id}:{ply_idx}"
    
    def extract_sf_candidates(stockfish_data: Dict[Any, Any]) -> List[_SFCandidate]:
        return [_SFCandidate(l[0], l[1]) for l in stockfish_data["sf_moves_returned"]]
    
    for p in cached_reference_stockfish_dpo_pairs:
        game_id = p["game_id"]
        ply_idx = p["ply_idx"]
        ref_output_temp_index[build_key(game_id, ply_idx)] = p
    
    for p in cached_inference_stockfish_dpo_pairs:
        game_id = p["game_id"]
        ply_idx = p["ply_idx"]
        key = build_key(game_id, ply_idx)
      
        if key in ref_output_temp_index.keys():
            p["meta"]["stockfish_reference"] = extract_sf_candidates(ref_output_temp_index[key]["meta"]["stockfish"])

        p["meta"]["stockfish_inference"] = extract_sf_candidates(p["meta"]["stockfish"])

    return cached_inference_stockfish_dpo_pairs
