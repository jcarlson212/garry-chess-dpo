from __future__ import annotations

import argparse
from pathlib import Path

import torch
from grandmaster_dpo.eval.single_gm.shared_eval_metric_utilities import run_eval


# DPO Loss function
def dpo_loss(logp_pi_ch, logp_pi_rj, logp_ref_ch, logp_ref_rj, beta: float) -> torch.Tensor:
    x = beta * ((logp_pi_ch - logp_pi_rj) - (logp_ref_ch - logp_ref_rj))
    return -torch.nn.functional.logsigmoid(x).mean()

# ----------------------------
# Main eval
# ----------------------------

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/eval/single_gm/eval_dpo_maia_single_gm.py --gm_name carlsen --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/eval_results_twic --model_dir ./final_experiments_for_paper/experiment1/trained_models_twic
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", required=True, help="Name of the grandmaster.")
    ap.add_argument("--split_name", required=False, default="val", help="train or val")
    ap.add_argument("--maia_type", default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.4, 0.6], help="List of beta values (e.g. --betas 0.1 0.2 0.4)")
    ap.add_argument("--n_boot", type=int, default=100, help="Number of bootstrap resamples for confidence intervals")
    ap.add_argument("--train_val_folder", required=True, help="Train/val folder.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--model_dir", required=True, help="Model directory.")

    args = ap.parse_args()
    jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_{args.split_name}_dpo.jsonl")

    

    for beta in args.betas:

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
            return dpo_loss(logp_pi_ch, logp_pi_rj, logp_ref_ch, logp_ref_rj, beta=beta)
            
        run_eval(jsonl, 
                 f"{args.model_dir}/{args.gm_name}/policy_best_dpo_beta={beta:.2f}.pt", 
                 args.out_dir, 
                 args.gm_name, 
                 args.device, 
                 args.maia_type, 
                 f"opening_probe_policy_dpo_beta={beta:.2f}.json",
                 args.n_boot,
                 args.batch_size,
                 args.split_name,
                 f"eval_results_dpo_beta={beta:.2f}_{args.split_name}.json",
                 f"eval_results_extended_dpo_beta={beta:.2f}_{args.split_name}.json",
                 f"eval_results_dpo_beta={beta:.2f}_{args.split_name}.csv",
                 f"eval_per_row_metrics_dpo_beta={beta:.2f}_{args.split_name}.jsonl",
                 supplied_loss_function
        )

if __name__ == "__main__":
    main()
