from typing import Dict, List

from grandmaster_dpo.eval.stockfish_helpers import uci_to_vocab_index
import torch


@torch.inference_mode()
def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(legal_moves > 0, logits, torch.full_like(logits, neg_inf))

@torch.inference_mode()
def gather_logprob_from_masked_logits(logits_masked: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
    logp_all = torch.log_softmax(logits_masked, dim=-1)
    safe = idxs.clamp(min=0)
    out = logp_all.gather(1, safe.view(-1, 1)).squeeze(1)
    out = torch.where(idxs >= 0, out, torch.full_like(out, -1e9))
    return out

@torch.inference_mode()
def kl_policy_base_from_logits(logits_pi_masked: torch.Tensor, logits_ref_masked: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits_pi_masked, dim=-1)
    logp = torch.log_softmax(logits_pi_masked, dim=-1)
    logq = torch.log_softmax(logits_ref_masked, dim=-1)
    return (p * (logp - logq)).sum(dim=-1)

@torch.inference_mode()
def chosen_probability(logits_masked: torch.Tensor, fens: List[str], all_moves_dict: Dict[str, int], chosen_uci: List[str]) -> torch.Tensor:
    probs = torch.softmax(logits_masked, dim=-1)
    chosen_idx = torch.tensor(
        [uci_to_vocab_index(all_moves_dict, fen, uci) for fen, uci in zip(fens, chosen_uci)],
        device=logits_masked.device,
        dtype=torch.long,
    )
    safe_idx = chosen_idx.clamp(min=0)
    p = probs.gather(1, safe_idx.view(-1, 1)).squeeze(1)
    return torch.where(chosen_idx >= 0, p, torch.zeros_like(p))

@torch.inference_mode()
def hit_at_k(logits_masked: torch.Tensor, chosen_idx: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.zeros((logits_masked.size(0),), device=logits_masked.device)
    topk = torch.topk(logits_masked, k=min(k, logits_masked.size(-1)), dim=-1).indices
    chosen_safe = chosen_idx.clamp(min=0).view(-1, 1)
    hit = (topk == chosen_safe).any(dim=1).float()
    return torch.where(chosen_idx >= 0, hit, torch.zeros_like(hit))

@torch.inference_mode()
def chosen_rank(logits_masked: torch.Tensor, chosen_idx: torch.Tensor) -> torch.Tensor:
    chosen_safe = chosen_idx.clamp(min=0)
    chosen_logit = logits_masked.gather(1, chosen_safe.view(-1, 1)).squeeze(1)
    greater = (logits_masked > chosen_logit.unsqueeze(1)).sum(dim=1)
    rank = greater + 1
    return torch.where(chosen_idx >= 0, rank, torch.full_like(rank, 10**9))

