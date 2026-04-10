from grandmaster_dpo.eval.chess_utils import fen_to_ply_abs, vocab_index_to_uci
from grandmaster_dpo.eval.configs import OpeningLogitDistConfig
import torch
from typing import Counter, Dict, List

def update_opening_distributions_from_logits(
    *,
    opening_counts: Dict[str, Counter],
    fens: List[str],
    logits_masked: torch.Tensor,  # [B, V] or [V]
    all_moves: List[str],
    cfg: OpeningLogitDistConfig,
) -> None:
    """
    Accumulate *soft* opening distributions using only model logits.

    opening_counts:
      - dict[str, Counter] you maintain across the eval loop, e.g.:
          opening_counts = {"ply0_white": Counter(), "ply1_black": Counter()}
      - We add *float mass* to Counter values (Counter supports float increments).

    Strategy:
      - For each position whose ply_abs is in cfg.plies:
          probs = softmax(logits / T)
          take top-K (cfg.topk) for efficiency
          renormalize within top-K
          add prob mass to the UCI move key

    Notes:
      - logits_masked should already have illegal moves at -inf (your apply_legal_mask),
        so softmax puts ~0 mass on illegal moves automatically.
      - Using soft counts avoids brittle argmax and captures uncertainty.
    """
    if logits_masked.dim() == 1:
        logits_masked = logits_masked.unsqueeze(0)

    assert logits_masked.dim() == 2, f"expected [B,V], got {tuple(logits_masked.shape)}"
    B, V = logits_masked.shape
    assert len(fens) == B, f"len(fens)={len(fens)} must match B={B}"
    assert len(all_moves) == V, f"len(all_moves)={len(all_moves)} must match V={V}"

    T = max(float(cfg.temperature), 1e-6)

    # Build a quick mapping: ply_abs -> bucket name
    # (You can customize naming; these two are usually what you want.)
    ply_to_bucket: Dict[int, str] = {}
    for p in cfg.plies:
        if p == 0:
            ply_to_bucket[p] = "ply0_white"
        elif p == 1:
            ply_to_bucket[p] = "ply1_black"
        else:
            ply_to_bucket[p] = f"ply{p}"

    # Ensure all buckets exist
    for b in ply_to_bucket.values():
        opening_counts.setdefault(b, Counter())

    with torch.no_grad():
        # We operate per-row because each row needs fen-dependent mirroring for idx->uci.
        for i in range(B):
            fen = fens[i]
            ply_abs = fen_to_ply_abs(fen)
            if ply_abs not in ply_to_bucket:
                continue

            bucket = ply_to_bucket[ply_abs]
            row_logits = logits_masked[i] / T

            if cfg.topk and cfg.topk > 0 and cfg.topk < V:
                vals, idxs = torch.topk(row_logits, k=int(cfg.topk), dim=-1)
                probs = torch.softmax(vals, dim=-1)  # renormalized within top-K
                idxs_list = idxs.tolist()
                probs_list = probs.detach().cpu().tolist()
                for j, p in zip(idxs_list, probs_list):
                    uci = vocab_index_to_uci(all_moves, fen, int(j))
                    if uci:
                        opening_counts[bucket][uci] += float(p)
            else:
                # Full distribution (can be slower)
                probs_full = torch.softmax(row_logits, dim=-1).detach().cpu()
                # If you want to be safe about numeric noise, you can skip tiny probs
                # but iterating V can be expensive.
                for j in range(V):
                    p = float(probs_full[j].item())
                    if p <= 0.0:
                        continue
                    uci = vocab_index_to_uci(all_moves, fen, j)
                    if uci:
                        opening_counts[bucket][uci] += p


def summarize_opening_distribution(
    opening_counts: Dict[str, Counter],
    *,
    topn: int = 30,
    normalize: bool = True,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Convert Counters into a JSON-friendly summary.
    If normalize=True, convert masses to probabilities per bucket.
    """
    out: Dict[str, List[Dict[str, float]]] = {}
    for bucket, ctr in opening_counts.items():
        if not ctr:
            out[bucket] = []
            continue
        items = ctr.most_common(topn)
        if normalize:
            total = float(sum(ctr.values()))
            out[bucket] = [{"uci": u, "p": float(c) / max(1e-12, total)} for (u, c) in items]
        else:
            out[bucket] = [{"uci": u, "mass": float(c)} for (u, c) in items]
    return out

