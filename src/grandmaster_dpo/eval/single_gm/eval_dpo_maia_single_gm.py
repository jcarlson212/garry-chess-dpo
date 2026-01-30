from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from maia2 import inference, model as maia_model
from maia2.utils import create_elo_dict, get_all_possible_moves, mirror_move


# ----------------------------
# Dataset
# ----------------------------

class DpoPairs(Dataset):
    def __init__(self, jsonl_path: str):
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
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {"fen": [], "elo_self": [], "elo_oppo": [], "chosen": [], "rejected": []}
    for b in batch:
        for k in out:
            out[k].append(b[k])
    return out


# ----------------------------
# Helpers (match training)
# ----------------------------

def device_from_str(s: str) -> torch.device:
    s = s.lower()
    if s in ("cpu",):
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        return torch.device("cuda")
    if s in ("mps",):
        return torch.device("mps")
    return torch.device(s)


def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
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
    board_inputs = []
    legal_moves = []
    elo_self_cats = []
    elo_oppo_cats = []

    for fen, es, eo in zip(fens, elo_self, elo_oppo):
        bi, es_cat, eo_cat, lm = inference.preprocessing(fen, int(es), int(eo), elo_dict, all_moves_dict)
        board_inputs.append(bi)
        legal_moves.append(lm)
        elo_self_cats.append(int(es_cat))
        elo_oppo_cats.append(int(eo_cat))

    board_input = torch.stack(board_inputs, dim=0).to(device)
    legal_moves_t = torch.stack(legal_moves, dim=0).to(device)
    elo_self_t = torch.tensor(elo_self_cats, device=device).long()
    elo_oppo_t = torch.tensor(elo_oppo_cats, device=device).long()
    return board_input, legal_moves_t, elo_self_t, elo_oppo_t


def forward_logits(m: torch.nn.Module, board_input: torch.Tensor, es: torch.Tensor, eo: torch.Tensor) -> torch.Tensor:
    logits_maia, _, _ = m(board_input, es, eo)
    return logits_maia


def uci_to_vocab_index(all_moves_dict: Dict[str, int], fen: str, uci: str) -> int:
    side = fen.split(" ")[1]
    uci_eff = mirror_move(uci) if side == "b" else uci
    return int(all_moves_dict.get(uci_eff, -1))


def gather_logprob(logits_masked: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
    # logits_masked already has illegal moves at -inf; safe to log_softmax
    logp_all = torch.log_softmax(logits_masked, dim=-1)
    safe_idx = idxs.clamp(min=0)
    gathered = logp_all.gather(dim=1, index=safe_idx.view(-1, 1)).squeeze(1)
    gathered = torch.where(idxs >= 0, gathered, torch.full_like(gathered, -1e9))
    return gathered


def dpo_loss(logp_pi_ch, logp_pi_rj, logp_ref_ch, logp_ref_rj, beta: float) -> torch.Tensor:
    x = beta * ((logp_pi_ch - logp_pi_rj) - (logp_ref_ch - logp_ref_rj))
    return -torch.nn.functional.logsigmoid(x).mean()


@torch.no_grad()
def kl_policy_base_from_logits(logits_pi_masked: torch.Tensor, logits_ref_masked: torch.Tensor) -> torch.Tensor:
    # KL( pi || ref ) over vocab
    p = torch.softmax(logits_pi_masked, dim=-1)
    logp = torch.log_softmax(logits_pi_masked, dim=-1)
    logq = torch.log_softmax(logits_ref_masked, dim=-1)
    kl = (p * (logp - logq)).sum(dim=-1)  # [B]
    return kl


@torch.no_grad()
def top1_accuracy(logits_masked: torch.Tensor, fens: List[str], all_moves_dict: Dict[str, int], chosen_uci: List[str]) -> torch.Tensor:
    # top1 index in vocab
    top_idx = logits_masked.argmax(dim=-1)  # [B]
    chosen_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, uci) for fen, uci in zip(fens, chosen_uci)],
                              device=logits_masked.device, dtype=torch.long)
    return (top_idx == chosen_idx).float()  # [B]


@torch.no_grad()
def chosen_probability(logits_masked: torch.Tensor, fens: List[str], all_moves_dict: Dict[str, int], chosen_uci: List[str]) -> torch.Tensor:
    probs = torch.softmax(logits_masked, dim=-1)
    chosen_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, uci) for fen, uci in zip(fens, chosen_uci)],
                              device=logits_masked.device, dtype=torch.long)
    safe_idx = chosen_idx.clamp(min=0)
    p = probs.gather(dim=1, index=safe_idx.view(-1, 1)).squeeze(1)
    p = torch.where(chosen_idx >= 0, p, torch.zeros_like(p))
    return p


# ----------------------------
# Main eval
# ----------------------------

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/eval/single_gm/eval_dpo_maia_single_gm.py --gm_name magnus
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", required=True, help="Name of the grandmaster.")
    ap.add_argument("--split_name", required=False, default="val", help="train or val")
    ap.add_argument("--maia_type", default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.1)
    
    args = ap.parse_args()
    jsonl = Path(f"./processed/single_gm/train_val/{args.gm_name}_{args.split_name}_dpo.jsonl")
    policy_pt = Path(f"./processed/single_gm/train_val/{args.gm_name}/policy_best.pt")
    out_dir = Path(f"./processed/single_gm/train_val/validation_results/{args.gm_name}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_str(args.device)

    # Build vocab + elo dict deterministically (avoid prepare() ordering issues)
    all_moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    # Load base twice; then load policy weights into one
    base = maia_model.from_pretrained(type=args.maia_type, device=str(device)).to(device)
    policy = maia_model.from_pretrained(type=args.maia_type, device=str(device)).to(device)

    sd = torch.load(policy_pt, map_location="cpu")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (showing 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (showing 10): {unexpected[:10]}")

    base.eval()
    policy.eval()

    ds = DpoPairs(jsonl)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    # Aggregate metrics
    n = 0
    sum_dpo = 0.0

    sum_pi_gap = 0.0
    sum_ref_gap = 0.0
    sum_gap_improvement = 0.0

    sum_top1_pi = 0.0
    sum_top1_ref = 0.0

    sum_p_chosen_pi = 0.0
    sum_p_chosen_ref = 0.0

    sum_kl = 0.0

    for batch in loader:
        fens = batch["fen"]
        es = batch["elo_self"]
        eo = batch["elo_oppo"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        bs = len(fens)

        board_input, legal_moves, es_t, eo_t = batch_preprocess(all_moves_dict, elo_dict, fens, es, eo, device)

        logits_pi = forward_logits(policy, board_input, es_t, eo_t)
        logits_ref = forward_logits(base, board_input, es_t, eo_t)

        logits_pi_m = apply_legal_mask(logits_pi, legal_moves)
        logits_ref_m = apply_legal_mask(logits_ref, legal_moves)

        # indices for chosen/rejected
        chosen_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, u) for fen, u in zip(fens, chosen)],
                                  device=device, dtype=torch.long)
        rejected_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, u) for fen, u in zip(fens, rejected)],
                                    device=device, dtype=torch.long)

        logp_pi_ch = gather_logprob(logits_pi_m, chosen_idx)
        logp_pi_rj = gather_logprob(logits_pi_m, rejected_idx)
        logp_ref_ch = gather_logprob(logits_ref_m, chosen_idx)
        logp_ref_rj = gather_logprob(logits_ref_m, rejected_idx)

        loss = dpo_loss(logp_pi_ch, logp_pi_rj, logp_ref_ch, logp_ref_rj, beta=args.beta)

        pi_gap = (logp_pi_ch - logp_pi_rj)          # [B]
        ref_gap = (logp_ref_ch - logp_ref_rj)       # [B]
        gap_improve = (pi_gap - ref_gap)            # [B]

        top1_pi = top1_accuracy(logits_pi_m, fens, all_moves_dict, chosen)
        top1_ref = top1_accuracy(logits_ref_m, fens, all_moves_dict, chosen)

        p_chosen_pi = chosen_probability(logits_pi_m, fens, all_moves_dict, chosen)
        p_chosen_ref = chosen_probability(logits_ref_m, fens, all_moves_dict, chosen)

        kl = kl_policy_base_from_logits(logits_pi_m, logits_ref_m)     # [B]

        n += bs
        sum_dpo += float(loss) * bs

        sum_pi_gap += float(pi_gap.mean()) * bs
        sum_ref_gap += float(ref_gap.mean()) * bs
        sum_gap_improvement += float(gap_improve.mean()) * bs

        sum_top1_pi += float(top1_pi.mean()) * bs
        sum_top1_ref += float(top1_ref.mean()) * bs

        sum_p_chosen_pi += float(p_chosen_pi.mean()) * bs
        sum_p_chosen_ref += float(p_chosen_ref.mean()) * bs

        sum_kl += float(kl.mean()) * bs

    def avg(x: float) -> float:
        return x / max(1, n)

    print("\n=== Eval summary ===")
    print(f"GM: {args.gm_name}")
    print(f"examples: {n}")
    print(f"dpo_loss (policy vs base ref): {avg(sum_dpo):.4f}")
    print("")
    print(f"mean logp_gap policy (chosen - rejected): {avg(sum_pi_gap):.4f}")
    print(f"mean logp_gap base   (chosen - rejected): {avg(sum_ref_gap):.4f}")
    print(f"mean gap improvement (policy - base):     {avg(sum_gap_improvement):.4f}")
    print("")
    print(f"top1 accuracy on chosen (policy): {avg(sum_top1_pi):.4f}")
    print(f"top1 accuracy on chosen (base):   {avg(sum_top1_ref):.4f}")
    print("")
    print(f"mean P(chosen) (policy): {avg(sum_p_chosen_pi):.4f}")
    print(f"mean P(chosen) (base):   {avg(sum_p_chosen_ref):.4f}")
    print("")
    print(f"mean KL(policy || base) over legal moves: {avg(sum_kl):.4f}")
    print("")

    out_dir.joinpath(f"eval_results_{args.split_name}.json").write_text(json.dumps({
        "dpo_loss": avg(sum_dpo),
        "mean_logp_gap_policy_chosen_rejected": avg(sum_pi_gap),
        "mean_logp_gap_base_chosen_rejected": avg(sum_ref_gap),
        "mean_gap_improvement": avg(sum_gap_improvement),
        "top1_accuracy_on_chosen_policy": avg(sum_top1_pi),
        "top1_accuracy_on_chosen_base": avg(sum_top1_ref),
        "mean_p_chosen_policy": avg(sum_p_chosen_pi),
        "mean_p_chosen_base": avg(sum_p_chosen_ref),
        "mean_kl": avg(sum_kl),
    }))
    print(f"Eval results saved to {out_dir.joinpath(f'eval_results_{args.split_name}.json')}")
    print(f"Eval results saved to {out_dir.joinpath(f'eval_results_{args.split_name}.json')}")
    # Now we write csv to out_dir.joinpath(f"eval_results_{args.split_name}.csv")
    import csv
    with open(out_dir.joinpath(f"eval_results_{args.split_name}.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["dpo_loss", "mean_logp_gap_policy_chosen_rejected", "mean_logp_gap_base_chosen_rejected", "mean_gap_improvement", "top1_accuracy_on_chosen_policy", "top1_accuracy_on_chosen_base", "mean_p_chosen_policy", "mean_p_chosen_base", "mean_kl"])
        writer.writerow([avg(sum_dpo), avg(sum_pi_gap), avg(sum_ref_gap), avg(sum_gap_improvement), avg(sum_top1_pi), avg(sum_top1_ref), avg(sum_p_chosen_pi), avg(sum_p_chosen_ref), avg(sum_kl)])
    print(f"CSV saved to {out_dir.joinpath(f'eval_results_{args.split_name}.csv')}")


if __name__ == "__main__":
    main()
