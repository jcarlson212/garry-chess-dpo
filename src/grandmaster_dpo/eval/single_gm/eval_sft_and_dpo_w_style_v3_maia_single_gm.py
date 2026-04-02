from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass
from grandmaster_dpo.train.style_embeddings_for_gms.train_configs import make_config
from grandmaster_dpo.utilities.shared_style_emb_model_utils import StyleEncoder, move_feature_dict_to_device
import chess
import itertools
import math
from grandmaster_dpo.utilities.shared_style_emb_model_utils import add_batch_dim, move_feature_dict_to_device, raw_example_to_model_features

import torch
from torch import nn

from grandmaster_dpo.eval.single_gm.shared_eval_metric_utilities import run_eval


def compute_style_score_v3(
    fen: str,
    chosen_uci: str,
    rejected_uci: str,
    style_embedding_model: nn.Module,
    event: str,
    style_tau: float,
    prev_fens: Optional[Sequence[Optional[str]]] = None,
) -> float:
    game_type = "classical"
    if "blitz" in event.lower():
        game_type = "blitz"
    elif "rapid" in event.lower():
        game_type = "rapid"

    device = next(style_embedding_model.parameters()).device

    style_model_features_chosen = add_batch_dim(move_feature_dict_to_device(
        raw_example_to_model_features(
            {
                "board_t_minus_4": prev_fens[-4] if prev_fens and len(prev_fens) >= 4 else None,
                "board_t_minus_3": prev_fens[-3] if prev_fens and len(prev_fens) >= 3 else None,
                "board_t_minus_2": prev_fens[-2] if prev_fens and len(prev_fens) >= 2 else None,
                "board_t_minus_1": prev_fens[-1] if prev_fens and len(prev_fens) >= 1 else None,
                "board_t": fen,
                "move_played": chosen_uci,
                "game_type": game_type, 
            },
            "phi1"
        ), 
        device))
    
    style_model_features_rejected = add_batch_dim(move_feature_dict_to_device(
        raw_example_to_model_features(
            {
                "board_t_minus_4": prev_fens[-4] if prev_fens and len(prev_fens) >= 4 else None,
                "board_t_minus_3": prev_fens[-3] if prev_fens and len(prev_fens) >= 3 else None,
                "board_t_minus_2": prev_fens[-2] if prev_fens and len(prev_fens) >= 2 else None,
                "board_t_minus_1": prev_fens[-1] if prev_fens and len(prev_fens) >= 1 else None,
                "board_t": fen,
                "move_played": rejected_uci,
                "game_type": game_type, 
            },
            "phi1"
        ), 
        device))

    style_embedding_chosen = style_embedding_model(style_model_features_chosen).squeeze(0)
    style_embedding_rejected = style_embedding_model(style_model_features_rejected).squeeze(0)

    style_score = torch.dot(style_embedding_chosen, style_embedding_rejected) / style_tau

    return style_score

def dpo_loss_style_weighted_v3(
    logp_pi_ch: torch.Tensor,
    logp_pi_rj: torch.Tensor,
    logp_ref_ch: torch.Tensor,
    logp_ref_rj: torch.Tensor,
    style_score: torch.Tensor,
    beta: float,
    tau: float,
) -> torch.Tensor:
    u = torch.exp(-(style_score) / tau)
    dpo = -torch.nn.functional.logsigmoid(beta * ((logp_pi_ch - logp_pi_rj) - (logp_ref_ch - logp_ref_rj)))
    loss = (u.detach() * dpo).mean()
    return loss

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
    # Example usage: python ./src/grandmaster_dpo/eval/single_gm/eval_sft_and_dpo_w_style_v3_maia_single_gm.py --gm_name caruana --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment2_style_model/eval_results_single_gm_twic --model_dir ./final_experiments_for_paper/experiment2_style_model/trained_models_single_gm_twic
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", required=True, help="Name of the grandmaster.")
    ap.add_argument("--split_name", required=False, default="val", help="train or val")
    ap.add_argument("--maia_type", default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.6], help="List of beta values (e.g. --betas 0.1 0.2 0.4)")
    ap.add_argument("--dpo_loss_weights", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6], help="List of beta values (e.g. --dpo_loss_weight 0.1 0.2 0.4)")

    ap.add_argument("--style_embedding_model_checkpoint", type=str, default=(
            "./final_experiments_for_paper/experiment2_style_model/trained_models/"
            "super_v2_phi1_tau0_25_if_winner__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42/"
            "best.pt"
    ))
    
    ap.add_argument("--style_taus", type=float, nargs="+", default=[0.25], help="List of style taus for tuning how much style similarity reweights loss")

    ap.add_argument("--n_boot", type=int, default=100, help="Number of bootstrap resamples for confidence intervals")
    ap.add_argument("--train_val_folder", required=True, help="Train/val folder.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--model_dir", required=True, help="Model directory.")

    args = ap.parse_args()
    
    style_encoder_training_cfg = make_config(
        study_name="super_v3_phi1_tau0_25_warm_from_v2final",
        train_dir="./final_experiments_for_paper/experiment2_style_model/pairs_v3_cached/train",
        eval_dir="./final_experiments_for_paper/experiment2_style_model/pairs_v2_cached/eval",
        pair_variant="v3",
        seed=42,
        embedding_dim=256,
        batch_size=4096,
        lr=3e-4,
        tau=0.25,
        phi_variant="phi1",
        epochs=8,
        max_steps_per_epoch=5000,
        max_eval_batches=150,
        num_workers=0,
        init_from_checkpoint=args.style_embedding_model_checkpoint,
        init_reset_optimizer=True,
        init_strict_load=True,
    )

    style_embedding_model = StyleEncoder(style_encoder_training_cfg)

    for beta, dpo_loss_weight, style_tau in itertools.product(args.betas, args.dpo_loss_weights, args.style_taus):
        full_name = f"sft_and_dpo_w_style_v3_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_tau={style_tau:.2f}_embedding_model={args.style_embedding_model_checkpoint.split('/')[-1]}"
        jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_{args.split_name}_dpo.jsonl")
        
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
            style_scores = torch.tensor(
                [
                    compute_style_score_v3(
                        fen=fen,
                        chosen_uci=ch,
                        rejected_uci=rj,
                        prev_fens=prev_fens,
                        style_embedding_model=style_embedding_model,
                        event=meta_list["event"],
                        style_tau=style_tau,
                    )
                    for fen, ch, rj, ch_cp, rj_cp, ply_idx, prev_fens, next_fens_chosen, next_fens_rejected, meta_list
                    in batch_meta_data
                ],
                dtype=torch.float32,
                device=args.device,
            )

            loss = (
                dpo_loss_weight
                * dpo_loss_style_weighted_v3(
                    logp_pi_ch=logp_pi_ch,
                    logp_pi_rj=logp_pi_rj,
                    logp_ref_ch=logp_ref_ch,
                    logp_ref_rj=logp_ref_rj,
                    style_score=style_scores,
                    beta=beta,
                    tau=style_tau,
                )
                + supervised_nll_loss(logits_pi_m, idx_t)
            )
            return loss
            
        run_eval(jsonl, 
                 f"{args.model_dir}/{args.gm_name}/policy_best_{full_name}.pt", 
                 args.out_dir, 
                 args.gm_name, 
                 args.device, 
                 args.maia_type, 
                 f"opening_probe_policy_{full_name}.json",
                 args.n_boot,
                 args.batch_size,
                 args.split_name,
                 f"eval_results_{full_name}_{args.split_name}.json",
                 f"eval_results_extended_{full_name}_{args.split_name}.json",
                 f"eval_results_{full_name}_{args.split_name}.csv",
                 f"eval_per_row_metrics_{full_name}_{args.split_name}.jsonl",
                 supplied_loss_function
        )

if __name__ == "__main__":
    main()
