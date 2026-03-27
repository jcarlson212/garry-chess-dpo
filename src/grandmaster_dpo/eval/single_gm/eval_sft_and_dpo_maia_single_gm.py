from __future__ import annotations

import argparse
from pathlib import Path
import itertools

import torch

from grandmaster_dpo.eval.single_gm.shared_eval_metric_utilities import run_eval


# Loss Functions

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
    return (-torch.nn.functional.logsigmoid(x)).mean()

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
# Main eval
# ----------------------------

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/eval/single_gm/eval_sft_and_dpo_maia_single_gm.py --gm_name caruana --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/eval_results_twic --model_dir ./final_experiments_for_paper/experiment1/trained_models_twic
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", required=True, help="Name of the grandmaster.")
    ap.add_argument("--split_name", required=False, default="val", help="train or val")
    ap.add_argument("--maia_type", default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.6], help="List of beta values (e.g. --betas 0.1 0.2 0.4)")
    ap.add_argument("--dpo_loss_weights", type=float, nargs="+", default=[0.1, 0.2, 0.4], help="List of beta values (e.g. --dpo_loss_weight 0.1 0.2 0.4)")
    ap.add_argument("--n_boot", type=int, default=100, help="Number of bootstrap resamples for confidence intervals")
    ap.add_argument("--train_val_folder", required=True, help="Train/val folder.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--model_dir", required=True, help="Model directory.")

    args = ap.parse_args()

    for beta, dpo_loss_weight in itertools.product(args.betas, args.dpo_loss_weights):
        jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_{args.split_name}_dpo.jsonl")
            

        policy_pt = Path(f"{args.model_dir}/{args.gm_name}/policy_best_sft_and_dpo_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}.pt")

        def supplied_loss_function(logp_pi_ch, 
                                logp_pi_rj, 
                                logp_ref_ch, 
                                logp_ref_rj, 
                                logits_pi_m, 
                                logits_ref_m, 
                                idx_t, 
                                chosen_cps, 
                                rejected_cps, 
                                prev_fens_batch,
                                next_fens_chosen_batch,
                                next_fens_rejected_batch,
                                batch_meta_data):
            loss = dpo_loss_weight*dpo_loss(logp_pi_ch, logp_pi_rj, logp_ref_ch, logp_ref_rj, beta=beta) + supervised_nll_loss(logits_pi_m, idx_t)

            return loss

        run_eval(jsonl, 
            policy_pt,
            args.out_dir, 
            args.gm_name, 
            args.device, 
            args.maia_type, 
            f"opening_probe_policy_sft_and_dpo_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}.json",
            args.n_boot,
            args.batch_size,
            args.split_name,
            f"eval_results_sft_and_dpo_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_{args.split_name}.json",
            f"eval_results_extended_sft_and_dpo_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_{args.split_name}.json",
            f"eval_results_sft_and_dpo_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_{args.split_name}.csv",
            f"eval_per_row_metrics_sft_and_dpo_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_{args.split_name}.jsonl",
            supplied_loss_function
        )


if __name__ == "__main__":
    main()
