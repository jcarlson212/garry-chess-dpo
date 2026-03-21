from __future__ import annotations

import argparse
from dataclasses import dataclass
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
# Data Models
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
    out: Dict[str, List[Any]] = {"fen": [], "elo_self": [], "elo_oppo": [], "chosen": [], "rejected": [], "meta": []}
    for b in batch:
        for k in out:
            out[k].append(b[k])
    return out

@dataclass
class TrainConfig:
    gm_name: str
    device: str
    beta: float
    gamma: float
    cp_scale_open: float
    cp_scale_mid: float
    cp_scale_late: float
    alpha: float
    use_ref: bool
    lambda_safe: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    maia_type: str
    train_val_folder: str
    out_dir: str
    run_name: str

# ----------------------------
# Helpers
# ----------------------------

def extract_top_engine_cp(meta: dict) -> float:
    sf_moves = meta["stockfish"]["sf_moves_returned"]
    if not sf_moves:
        return 0.0
    return float(max(cp for _, cp in sf_moves))

def extract_chosen_cp(meta: dict, chosen_uci: str) -> float:
    sf_moves = meta["stockfish"]["sf_moves_returned"]

    # Try to find the chosen move in the Stockfish list
    for uci, cp in sf_moves:
        if uci == chosen_uci:
            return float(cp)

    # Fallback: chosen move not in Stockfish top moves
    cp_values = [cp for _, cp in sf_moves]
    fallback_cp = float(min(cp_values)) if cp_values else 0.0

    print(
        f"[WARN] chosen move {chosen_uci} not found in sf_moves_returned "
        f"(game={meta.get('game_header_hash')}, ply={meta.get('ply_idx')}). "
        f"Using fallback cp={fallback_cp}"
    )

    return fallback_cp

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


def kto_w_safety_loss(
    logp_pi_ch: torch.Tensor,
    logp_pi_rj: torch.Tensor,
    logp_ref_ch: torch.Tensor,
    logp_ref_rj: torch.Tensor,
    top_engine_cps: torch.Tensor,
    player_chosen_cps: torch.Tensor,
    ply_t: torch.Tensor,
    beta: float,
    gamma: float,
    cp_scale_open: float,
    cp_scale_mid: float,
    cp_scale_late: float,
    alpha: float,
    use_ref: bool,
    lambda_safe: float
) -> torch.Tensor:
    # KTO w cp gap
    pi_gap = beta*(logp_pi_ch - logp_pi_rj)
    ref_gap = beta*(logp_ref_ch - logp_ref_rj)

    preference_gap = pi_gap 

    if use_ref:
        preference_gap -= ref_gap 

    preference_gap -= gamma 

    # CP gap: how far chosen move is from engine average
    # Positive when player move is worse than engine avg.
    cp_gap = torch.clamp(top_engine_cps - player_chosen_cps, min=0.0)

    # Phase-dependent scaling
    # first 20 plies: early, next 20: midgame, rest: endgame
    phase_scale = torch.where(
        ply_t < 20,
        torch.full_like(cp_gap, cp_scale_open),
        torch.where(
            ply_t < 40,
            torch.full_like(cp_gap, cp_scale_mid),
            torch.full_like(cp_gap, cp_scale_late),
        ),
    )

    # Turn cp gap into a nonnegative weight.
    # Clamp at 0 so chosen moves better than engine avg do not produce weird negative weights.
    safe_score = torch.exp(- (cp_gap / phase_scale) * alpha)
    weight = 1.0 + lambda_safe * safe_score # range is [1, lambda_safe]

    x = preference_gap

    return (-torch.nn.functional.logsigmoid(x) * weight).mean()

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
    gamma: float,
    cp_scale_open: float,
    cp_scale_mid: float,
    cp_scale_late: float,
    alpha: float,
    use_ref: bool,
    lambda_safe: float,
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
        
        meta_list = batch["meta"]

        ply_t = torch.tensor(
            [int(m["ply_idx"]) for m in meta_list],
            dtype=torch.long,
            device=device,
        )

        player_chosen_cps = torch.tensor(
            [
                extract_chosen_cp(m, chosen_uci)
                for m, chosen_uci in zip(meta_list, batch["chosen"])
            ],
            dtype=torch.float32,
            device=device,
        )

        top_engine_cps = torch.tensor(
            [extract_top_engine_cp(m) for m in meta_list],
            dtype=torch.float32,
            device=device,
        )

        loss = kto_w_safety_loss(
            logp_pi_ch=logp_pi_ch,
            logp_pi_rj=logp_pi_rj,
            logp_ref_ch=logp_ref_ch,
            logp_ref_rj=logp_ref_rj,
            top_engine_cps=top_engine_cps,
            player_chosen_cps=player_chosen_cps,
            ply_t=ply_t,
            beta=beta,
            gamma=gamma,
            cp_scale_open=cp_scale_open,
            cp_scale_mid=cp_scale_mid,
            cp_scale_late=cp_scale_late,
            alpha=alpha,
            use_ref=use_ref,
            lambda_safe=lambda_safe,
        )

        bs = len(batch["fen"])
        total_loss += float(loss) * bs
        n += bs

    return {"loss": total_loss / max(1, n)}


# ----------------------------
# Train
# ----------------------------

def train_one_run(cfg: TrainConfig) -> None:
    train_jsonl = Path(f"{cfg.train_val_folder}/{cfg.gm_name}_train_dpo.jsonl")
    val_jsonl = Path(f"{cfg.train_val_folder}/{cfg.gm_name}_val_dpo.jsonl")
    out_dir = Path(f"{cfg.out_dir}/{cfg.gm_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_from_str(cfg.device)

    # Load Maia-2 base weights twice
    policy = maia_model.from_pretrained(type=cfg.maia_type, device=str(device))
    policy.train()
    ref = maia_model.from_pretrained(type=cfg.maia_type, device=str(device))
    ref.eval()

    policy.to(device)
    ref.to(device)
    for p in ref.parameters():
        p.requires_grad_(False)

    # Repo version: prepare() returns [all_moves_dict, elo_dict, all_moves_dict_reversed]
    prep = inference.prepare()
    all_moves_dict, elo_dict, _ = prep

    train_ds = DpoPairs(train_jsonl)
    val_ds = DpoPairs(val_jsonl)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    optim = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
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

            meta_list = batch["meta"]

            ply_t = torch.tensor(
                [int(m["ply_idx"]) for m in meta_list],
                dtype=torch.long,
                device=device,
            )

            player_chosen_cps = torch.tensor(
                [
                    extract_chosen_cp(m, chosen_uci)
                    for m, chosen_uci in zip(meta_list, batch["chosen"])
                ],
                dtype=torch.float32,
                device=device,
            )

            top_engine_cps = torch.tensor(
                [extract_top_engine_cp(m) for m in meta_list],
                dtype=torch.float32,
                device=device,
            )

            loss = kto_w_safety_loss(
                logp_pi_ch=logp_pi_ch,
                logp_pi_rj=logp_pi_rj,
                logp_ref_ch=logp_ref_ch,
                logp_ref_rj=logp_ref_rj,
                top_engine_cps=top_engine_cps,
                player_chosen_cps=player_chosen_cps,
                ply_t=ply_t,
                beta=cfg.beta,
                gamma=cfg.gamma,
                cp_scale_open=cfg.cp_scale_open,
                cp_scale_mid=cfg.cp_scale_mid,
                cp_scale_late=cfg.cp_scale_late,
                alpha=cfg.alpha,
                use_ref=cfg.use_ref,
                lambda_safe=cfg.lambda_safe,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            optim.step()

            bs = len(batch["fen"])
            running += float(loss.detach()) * bs
            seen += bs

            if step % 50 == 0:
                print(f"[epoch {epoch}] step={step} train_kto_w_safety_loss={running/max(1,seen):.4f}")

        metrics = evaluate(policy, ref, all_moves_dict, elo_dict, val_loader, device=device, 
                           beta=cfg.beta, gamma=cfg.gamma, cp_scale_open=cfg.cp_scale_open,
                           cp_scale_mid=cfg.cp_scale_mid,cp_scale_late=cfg.cp_scale_late,
                           alpha=cfg.alpha,use_ref=cfg.use_ref)
        
        val_loss = metrics["loss"]
        print(f"[epoch {epoch}] val_kto_w_safety_loss={val_loss:.4f}")

        ckpt_path = out_dir / f"policy_epoch{epoch}_kto_w_safety_{make_run_name(cfg)}.pt"
        torch.save(policy.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")

        if val_loss < best_val:
            best_val = val_loss
            best_path = out_dir / f"policy_best_kto_w_safety_{make_run_name(cfg)}.pt"
            torch.save(policy.state_dict(), best_path)
            print(f"Saved best: {best_path} (val_kto_w_safety_loss={best_val:.4f})")

def make_run_name(cfg: TrainConfig) -> str:
    return (
        f"beta={cfg.beta:.2f}_gamma={cfg.gamma:.2f}_"
        f"open={cfg.cp_scale_open:.0f}_mid={cfg.cp_scale_mid:.0f}_late={cfg.cp_scale_late:.0f}_"
        f"alpha={cfg.alpha:.2f}_use_ref={int(cfg.use_ref)}_lambda={cfg.lambda_safe:.2f}"
    )


def build_runs(args) -> list[TrainConfig]:
    base = dict(
        gm_name=args.gm_name,
        device=args.device,
        beta=args.beta,
        gamma=args.gamma,
        cp_scale_open=args.cp_scale_open,
        cp_scale_mid=args.cp_scale_mid,
        cp_scale_late=args.cp_scale_late,
        use_ref=args.use_ref,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        maia_type=args.maia_type,
        train_val_folder=args.train_val_folder,
        out_dir=args.out_dir,
    )

    runs = []

    if args.preset == "single":
        cfg = TrainConfig(
            **base,
            alpha=args.alpha,
            lambda_safe=args.lambda_safe,
            run_name="single_" + f"alpha={args.alpha:.2f}_lambda={args.lambda_safe:.2f}",
        )
        runs.append(cfg)

    elif args.preset == "overnight4":
        combos = [
            (1.0, 0.5),
            (1.0, 1.0),
            (2.0, 1.0),
            (2.0, 2.0),
        ]
        for alpha, lambda_safe in combos:
            cfg = TrainConfig(
                **base,
                alpha=alpha,
                lambda_safe=lambda_safe,
                run_name=f"alpha={alpha:.2f}_lambda={lambda_safe:.2f}",
            )
            runs.append(cfg)

    return runs


def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/train/single_gm/train_kto_w_safety_maia2.py --gm_name caruana --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic --beta 0.8 --gamma 0.0 --alpha 1.0 --lambda_safe 0.5 --use_ref --preset overnight4

    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", type=str, required=True)

    ap.add_argument("--device", type=str, default="cpu")  # "mps" works too if your torch build supports it
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--cp_scale_open", type=float, default=20)
    ap.add_argument("--cp_scale_mid", type=float, default=50)
    ap.add_argument("--cp_scale_late", type=float, default=100)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--use_ref", action="store_true", help="Whether to use the base maia2 model for regularizing KL divergence eff in SimPO-like loss")
    ap.add_argument("--lambda_safe", type=float, default=4.0)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--maia_type", type=str, default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--train_val_folder", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--preset", type=str, default="single", choices=["single", "overnight4"])

    args = ap.parse_args()
    
    runs = build_runs(args)

    print("Generated run configs:")
    for cfg in runs:
        print(make_run_name(cfg))

    for i, cfg in enumerate(runs, start=1):
        print("=" * 80)
        print(f"Starting run {i}/{len(runs)}: {cfg.run_name}")
        print(cfg)
        print("=" * 80)
        train_one_run(cfg)


if __name__ == "__main__":
    main()
