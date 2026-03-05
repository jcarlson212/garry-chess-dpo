from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math

import torch
from torch.utils.data import DataLoader, Dataset

from maia2 import inference, model as maia_model
from maia2.utils import mirror_move


# ----------------------------
# Dataset
# ----------------------------

class DpoPairs(Dataset):
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
            "chosen": r["chosen"],       # UCI
            "rejected": r["rejected"],   # UCI
            "label": int(r.get("preference", {}).get("label", 1)),
            "meta": r.get("meta", {}),
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {"fen": [], "elo_self": [], "elo_oppo": [], "chosen": [], "rejected": []}
    for b in batch:
        for k in out:
            out[k].append(b[k])
    return out


# ----------------------------
# Helpers
# ----------------------------

def ply_from_fen(fen: str) -> int:
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    return 2 * (fullmove - 1) + (1 if side == "b" else 0)

def kl_lambda_by_ply(ply: torch.Tensor, lam_open: float = 0.20, lam_mid: float = 0.02, lam_end: float = 0.08) -> torch.Tensor:
    # ply: [B] long

    # example cutoffs (in ply)
    # opening: <= 20 ply  (~10 moves each side)
    # middlegame: 21..80 ply
    # endgame: > 80 ply
    lam = torch.full_like(ply, lam_mid, dtype=torch.float32)

    lam = torch.where(ply <= 20, torch.tensor(lam_open, device=ply.device), lam)
    lam = torch.where(ply > 80,  torch.tensor(lam_end,  device=ply.device), lam)
    return lam

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


def move_logprob_from_logits(
    logits: torch.Tensor,
    fens: List[str],
    all_moves_dict: Dict[str, int],
    moves_uci: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Convert UCI -> index in Maia vocab (mirroring if fen is black-to-move),
    then gather logprob from logits.
    """
    logp_all = torch.log_softmax(logits, dim=-1)

    idxs: List[int] = []
    for fen, uci in zip(fens, moves_uci):
        side = fen.split(" ")[1]
        uci_eff = mirror_move(uci) if side == "b" else uci
        idx = all_moves_dict.get(uci_eff, None)
        idxs.append(-1 if idx is None else int(idx))

    idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
    safe_idx = idx_t.clamp(min=0)

    gathered = logp_all.gather(dim=1, index=safe_idx.view(-1, 1)).squeeze(1)
    gathered = torch.where(idx_t >= 0, gathered, torch.full_like(gathered, -1e9))
    return gathered


def dpo_loss(
    logp_pi_ch: torch.Tensor,
    logp_pi_rj: torch.Tensor,
    logp_ref_ch: torch.Tensor,
    logp_ref_rj: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    # DPO: -log σ( beta * ((π_ch-π_rj) - (ref_ch-ref_rj)) )
    pi_gap = logp_pi_ch - logp_pi_rj
    ref_gap = logp_ref_ch - logp_ref_rj
    x = beta * (pi_gap - ref_gap)
    return -torch.nn.functional.logsigmoid(x).mean()

# ----------------------------
# Eval
# ----------------------------

@torch.no_grad()
def evaluate(
    policy: torch.nn.Module,
    ref: torch.nn.Module,
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> Dict[str, float]:
    policy.eval()
    ref.eval()

    total_loss = 0.0
    n = 0

    for batch in loader:
        board_input, legal_moves, es_t, eo_t = batch_preprocess(
            all_moves_dict, elo_dict, batch["fen"], batch["elo_self"], batch["elo_oppo"], device
        )

        logits_pi = forward_logits(policy, board_input, es_t, eo_t)
        logits_ref = forward_logits(ref, board_input, es_t, eo_t)

        logits_pi = apply_legal_mask(logits_pi, legal_moves)
        logits_ref = apply_legal_mask(logits_ref, legal_moves)

        logp_pi_ch = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["chosen"], device)
        logp_pi_rj = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["rejected"], device)

        logp_ref_ch = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["chosen"], device)
        logp_ref_rj = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["rejected"], device)

        loss = dpo_loss(logp_pi_ch, logp_pi_rj, logp_ref_ch, logp_ref_rj, beta=beta)

        kl = kl_pi_ref_from_logits(logits_pi, logits_ref)        # [B]
        ply_t = torch.tensor([ply_from_fen(f) for f in batch["fen"]], device="cpu").long() # [B]

        logp_ref = torch.log_softmax(logits_ref, dim=-1)  # [B, V]
        p_ref = torch.exp(logp_ref)
        entropy = -(p_ref * logp_ref).sum(dim=-1)          # [B]
        num_legal = (legal_moves > 0).sum(dim=-1).clamp(min=2).float()  # [B]
        H_max = torch.log(num_legal).to(entropy.device)
        entropy_norm = (entropy / H_max).clamp(0.0, 1.0)
        k_entropy = 0.5
        lam_t = k_entropy * entropy_norm            # [B]
        lam_t = lam_t.clamp(min=0.05, max=0.40)   # e.g., 0.05 to 0.40

        kl_term = (lam_t * kl).mean()
        loss += kl_term

        bs = len(batch["fen"])
        total_loss += float(loss) * bs
        n += bs

    return {"dpo_loss": total_loss / max(1, n)}


# ----------------------------
# Train
# ----------------------------

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/train/single_gm/train_dpo_maia2.py --gm_name carlsen --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", type=str, required=True)

    ap.add_argument("--device", type=str, default="cpu")  # "mps" works too if your torch build supports it
    ap.add_argument("--beta", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--maia_type", type=str, default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--train_val_folder", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()

    train_jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_train_dpo.jsonl")
    val_jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_val_dpo.jsonl")
    out_dir = Path(f"{args.out_dir}/{args.gm_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_from_str(args.device)

    # Load Maia-2 base weights twice
    policy = maia_model.from_pretrained(type=args.maia_type, device=str(device))
    ref = maia_model.from_pretrained(type=args.maia_type, device=str(device))

    policy.to(device)
    ref.to(device)
    for p in ref.parameters():
        p.requires_grad_(False)

    # Repo version: prepare() returns [all_moves_dict, elo_dict, all_moves_dict_reversed]
    prep = inference.prepare()
    all_moves_dict, elo_dict, _ = prep

    train_ds = DpoPairs(train_jsonl)
    val_ds = DpoPairs(val_jsonl)

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

            logits_pi = forward_logits(policy, board_input, es_t, eo_t)
            with torch.no_grad():
                logits_ref = forward_logits(ref, board_input, es_t, eo_t)

            logits_pi = apply_legal_mask(logits_pi, legal_moves)
            logits_ref = apply_legal_mask(logits_ref, legal_moves)

            logp_pi_ch = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["chosen"], device)
            logp_pi_rj = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["rejected"], device)

            with torch.no_grad():
                logp_ref_ch = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["chosen"], device)
                logp_ref_rj = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["rejected"], device)

            loss = dpo_loss(logp_pi_ch, logp_pi_rj, logp_ref_ch, logp_ref_rj, beta=args.beta)

            kl = kl_pi_ref_from_logits(logits_pi, logits_ref)        # [B]
            ply_t = torch.tensor([ply_from_fen(f) for f in batch["fen"]], device="cpu").long() # [B]

            logp_ref = torch.log_softmax(logits_ref, dim=-1)  # [B, V]
            p_ref = torch.exp(logp_ref)
            entropy = -(p_ref * logp_ref).sum(dim=-1)          # [B]
            num_legal = (legal_moves > 0).sum(dim=-1).clamp(min=2).float()  # [B]
            H_max = torch.log(num_legal).to(entropy.device)
            entropy_norm = (entropy / H_max).clamp(0.0, 1.0)
            k_entropy = 0.5
            lam_t = k_entropy * entropy_norm            # [B]
            lam_t = lam_t.clamp(min=0.05, max=0.40)   # e.g., 0.05 to 0.40

            kl_term = (lam_t * kl).mean()
            loss += kl_term

            if step % 200 == 0:
                print("lam mean/min/max:", lam_t.mean().item(), lam_t.min().item(), lam_t.max().item())
                print("kl mean/p95:", kl.mean().item(), torch.quantile(kl.detach(), 0.95).item())

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optim.step()

            bs = len(batch["fen"])
            running += float(loss.detach()) * bs
            seen += bs

            if step % 50 == 0:
                print(f"[epoch {epoch}] step={step} train_dpo_loss={running/max(1,seen):.4f}")

        metrics = evaluate(policy, ref, all_moves_dict, elo_dict, val_loader, device=device, beta=args.beta)
        val_loss = metrics["dpo_loss"]
        print(f"[epoch {epoch}] val_dpo_loss={val_loss:.4f}")

        ckpt_path = out_dir / f"policy_epoch{epoch}_dpo.pt"
        torch.save(policy.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")

        if val_loss < best_val:
            best_val = val_loss
            best_path = out_dir / "policy_best_dpo.pt"
            torch.save(policy.state_dict(), best_path)
            print(f"Saved best: {best_path} (val_dpo_loss={best_val:.4f})")


if __name__ == "__main__":
    main()



