from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from maia2 import inference, model as maia_model
from maia2.utils import mirror_move


# ----------------------------
# Dataset
# ----------------------------

class SupervisedPairs(Dataset):
    """
    Same JSONL format as your DPO file, but for supervised fine-tuning we only use:
      - prompt.fen, prompt.elo_self, prompt.elo_oppo
      - chosen (the human move)
    We ignore rejected/label/meta (but keep parsing for compatibility).
    """
    def __init__(self, jsonl_path: str):
        self.path = jsonl_path
        self.rows: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        p = r["prompt"]
        return {
            "fen": p["fen"],
            "elo_self": int(p.get("elo_self", 2800)),
            "elo_oppo": int(p.get("elo_oppo", 2800)),
            "chosen": r["chosen"],  # UCI
            "meta": r.get("meta", {}),
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {"fen": [], "elo_self": [], "elo_oppo": [], "chosen": []}
    for b in batch:
        for k in out:
            out[k].append(b[k])
    return out


# ----------------------------
# Helpers
# ----------------------------

def kl_pi_ref_from_logits(
    logits_pi: torch.Tensor,   # [B, V] already legal-masked (illegal = -inf)
    logits_ref: torch.Tensor,  # [B, V] already legal-masked
) -> torch.Tensor:
    """
    Returns KL(pi || ref) per example: [B]
    """
    logp_pi = torch.log_softmax(logits_pi, dim=-1)     # [B, V]
    logp_ref = torch.log_softmax(logits_ref, dim=-1)   # [B, V]
    p_pi = logp_pi.exp()
    # KL(pi||ref) = sum_a pi(a) (log pi(a) - log ref(a))
    kl = (p_pi * (logp_pi - logp_ref)).sum(dim=-1)     # [B]
    return kl

def ply_from_fen(fen: str) -> int:
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    ply = 2 * (fullmove - 1)
    if side == "b":
        ply += 1
    return ply

def device_from_str(s: str) -> torch.device:
    s = s.lower()
    if s in ("cpu",):
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        return torch.device("cuda")
    if s in ("mps",):
        return torch.device("mps")
    return torch.device(s)


def max_elo_supported(elo_dict: dict) -> int:
    # find keys like ">=2000" and return 2000
    mx = None
    for k in elo_dict.keys():
        m = re.match(r"^>=\s*(\d+)$", k)
        if m:
            mx = max(mx or 0, int(m.group(1)))
    return mx if mx is not None else 3000


def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
    # legal_moves is 0/1 mask same shape as logits
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(legal_moves > 0, logits, torch.full_like(logits, neg_inf))


def batch_preprocess(
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    fens: List[str],
    elo_self: List[int],
    elo_oppo: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calls maia2.inference.preprocessing (repo version you pasted):
      preprocessing(fen, elo_self, elo_oppo, elo_dict, all_moves_dict)
        -> board_input, elo_self_cat, elo_oppo_cat, legal_moves_mask
    """
    board_inputs = []
    legal_moves = []
    elo_self_cats = []
    elo_oppo_cats = []

    mx = max_elo_supported(elo_dict)

    for fen, es, eo in zip(fens, elo_self, elo_oppo):
        es = min(int(es), mx)
        eo = min(int(eo), mx)

        bi, es_cat, eo_cat, lm = inference.preprocessing(
            fen, es, eo, elo_dict, all_moves_dict
        )

        board_inputs.append(bi)
        legal_moves.append(lm)
        elo_self_cats.append(int(es_cat))
        elo_oppo_cats.append(int(eo_cat))

    board_input = torch.stack(board_inputs, dim=0).to(device)         # [B, C, 8, 8]
    legal_moves_t = torch.stack(legal_moves, dim=0).to(device)        # [B, V]
    elo_self_t = torch.tensor(elo_self_cats, device=device).long()    # [B]
    elo_oppo_t = torch.tensor(elo_oppo_cats, device=device).long()    # [B]
    return board_input, legal_moves_t, elo_self_t, elo_oppo_t


def forward_logits(
    maia2_model: torch.nn.Module,
    board_input: torch.Tensor,
    elo_self_tensor: torch.Tensor,
    elo_oppo_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Maia2 repo inference uses:
      logits_maia, _, logits_value = model(boards, elos_self, elos_oppo)
    """
    logits_maia, _, _ = maia2_model(board_input, elo_self_tensor, elo_oppo_tensor)
    return logits_maia


def chosen_index_tensor(
    fens: List[str],
    all_moves_dict: Dict[str, int],
    moves_uci: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Convert UCI -> Maia vocab index (mirroring if fen is black-to-move).
    Returns idx_t with -1 for unknown moves (should be rare; those will be ignored).
    """
    idxs: List[int] = []
    for fen, uci in zip(fens, moves_uci):
        side = fen.split(" ")[1]
        uci_eff = mirror_move(uci) if side == "b" else uci
        idx = all_moves_dict.get(uci_eff, None)
        idxs.append(-1 if idx is None else int(idx))
    return torch.tensor(idxs, device=device, dtype=torch.long)


def supervised_nll_loss(
    logits_masked: torch.Tensor,
    idx_t: torch.Tensor,
) -> torch.Tensor:
    """
    Standard supervised fine-tuning objective:
      loss = -mean(log p(chosen_move))
    ignoring examples where chosen move isn't in vocab (idx == -1).
    """
    logp_all = torch.log_softmax(logits_masked, dim=-1)  # [B, V]

    valid = idx_t >= 0
    if valid.sum().item() == 0:
        # return a zero scalar that still has grad
        return logits_masked.sum() * 0.0

    safe_idx = idx_t.clamp(min=0)
    gathered = logp_all.gather(dim=1, index=safe_idx.view(-1, 1)).squeeze(1)  # [B]
    gathered = gathered[valid]
    return (-gathered).mean()


# ----------------------------
# Eval
# ----------------------------

@torch.no_grad()
def evaluate(
    policy: torch.nn.Module,
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    policy.eval()

    total_loss = 0.0
    n = 0

    for batch in loader:
        board_input, legal_moves, es_t, eo_t = batch_preprocess(
            all_moves_dict, elo_dict, batch["fen"], batch["elo_self"], batch["elo_oppo"], device
        )

        logits = forward_logits(policy, board_input, es_t, eo_t)
        logits = apply_legal_mask(logits, legal_moves)

        idx_t = chosen_index_tensor(batch["fen"], all_moves_dict, batch["chosen"], device)

        loss = supervised_nll_loss(logits, idx_t)

        bs = len(batch["fen"])
        total_loss += float(loss) * bs
        n += bs

    return {"nll_loss": total_loss / max(1, n)}


# ----------------------------
# Train
# ----------------------------

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/train/single_gm/train_sft_maia2.py --gm_name carlsen --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", type=str, required=True)

    ap.add_argument("--device", type=str, default="cpu")  # "mps" works too if your torch build supports it

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--train_val_folder", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--maia_type", type=str, default="blitz", choices=["blitz", "rapid"]) # this is just the base model from maia2 used.

    args = ap.parse_args()

    # Reuse your same train/val JSONLs (they contain chosen + rejected, but we only use chosen)
    train_jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_train_dpo.jsonl")
    val_jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_val_dpo.jsonl")

    out_dir = Path(f"{args.out_dir}/{args.gm_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_from_str(args.device)

    # Load Maia-2 base weights once (SFT updates these weights)
    policy = maia_model.from_pretrained(type=args.maia_type, device=str(device))
    policy.train()
    policy.to(device)

    # Repo version: prepare() returns [all_moves_dict, elo_dict, all_moves_dict_reversed]
    prep = inference.prepare()
    all_moves_dict, elo_dict, _ = prep

    train_ds = SupervisedPairs(str(train_jsonl))
    val_ds = SupervisedPairs(str(val_jsonl))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        policy.train()
        running = 0.0
        seen = 0

        for batch in train_loader:
            step += 1

            board_input, legal_moves, es_t, eo_t = batch_preprocess(
                all_moves_dict, elo_dict, batch["fen"], batch["elo_self"], batch["elo_oppo"], device
            )

            logits = forward_logits(policy, board_input, es_t, eo_t)
            logits = apply_legal_mask(logits, legal_moves)

            idx_t = chosen_index_tensor(batch["fen"], all_moves_dict, batch["chosen"], device)
            
            loss = supervised_nll_loss(logits, idx_t)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optim.step()

            bs = len(batch["fen"])
            running += float(loss.detach()) * bs
            seen += bs

            if step % 50 == 0:
                print(f"[epoch {epoch}] step={step} train_nll_loss={running/max(1,seen):.4f}")

        metrics = evaluate(policy, all_moves_dict, elo_dict, val_loader, device=device)
        val_loss = metrics["nll_loss"]
        print(f"[epoch {epoch}] val_nll_loss={val_loss:.4f}")

        ckpt_path = out_dir / f"policy_sft_epoch{epoch}.pt"
        torch.save(policy.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")

        if val_loss < best_val:
            best_val = val_loss
            best_path = out_dir / "policy_sft_best.pt"
            torch.save(policy.state_dict(), best_path)
            print(f"Saved best: {best_path} (val_nll_loss={best_val:.4f})")


if __name__ == "__main__":
    main()
