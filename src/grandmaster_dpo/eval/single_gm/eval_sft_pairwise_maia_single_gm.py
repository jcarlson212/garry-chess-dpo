from __future__ import annotations

import argparse
from pathlib import Path

import torch

from grandmaster_dpo.eval.single_gm.shared_eval_metric_utilities import run_eval


def sft_pairwise_loss(
    logp_pi_ch: torch.Tensor,
    logp_pi_rj: torch.Tensor,
) -> torch.Tensor:
    pi_gap = logp_pi_ch - logp_pi_rj
    return (-torch.nn.functional.logsigmoid(pi_gap)).mean()


# ----------------------------
# Main eval
# ----------------------------

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/eval/single_gm/eval_sft_pairwise_maia_single_gm.py --gm_name magnus
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", required=True, help="Name of the grandmaster.")
    ap.add_argument("--split_name", required=False, default="val", help="train or val")
    ap.add_argument("--maia_type", default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--n_boot", type=int, default=100, help="Number of bootstrap resamples for confidence intervals")
    ap.add_argument("--train_val_folder", required=True, help="Train/val folder.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--model_dir", required=True, help="Model directory.")
    
    args = ap.parse_args()
    jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_{args.split_name}_dpo.jsonl")
    policy_pt = Path(f"{args.model_dir}/{args.gm_name}/policy_pairwise_sft_best.pt")

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
                                batch_meta_data
    ):
        loss = sft_pairwise_loss(logp_pi_ch, logp_pi_rj)
        return loss
        
    run_eval(str(jsonl), 
                str(policy_pt), 
                args.out_dir, 
                args.gm_name, 
                args.device, 
                args.maia_type, 
                f"opening_probe_policy_sft_pairwise.json",
                args.n_boot,
                args.batch_size,
                args.split_name,
                f"eval_results_sft_pairwise_{args.split_name}.json",
                f"eval_results_sft_pairwise_extended_{args.split_name}.json",
                f"eval_results_sft_pairwise_{args.split_name}.csv",
                f"eval_per_row_metrics_sft_pairwise_{args.split_name}.jsonl",
                supplied_loss_function
    )

if __name__ == "__main__":
    main()
