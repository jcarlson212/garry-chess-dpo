# scripts/eval_gm_family.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from grandmaster_dpo.eval.eval_abstractions import (
    build_sf_models_for_gm,
)

from grandmaster_dpo.eval.eval_abstractions import (
    SfConfig,
)

from grandmaster_dpo.eval.single_gm.shared_eval_metric_utilities import DpoPairs
from grandmaster_dpo.utilities.shared_style_emb_model_utils import pick_device

def parse_int_list(s: str):
    try:
        return [int(x) for x in s.split(",")]
    except Exception:
        raise argparse.ArgumentTypeError(f"Expected comma-separated ints, got: {s}")

def main() -> None:
    # Example: python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py --gm_name carlsen --split val --sf_depth 5 --sf_tops 10 
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", type=str, required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--maia_type", default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--batch_size", type=int, default=64)

    # dataset + checkpoints layout (match your repo)
    ap.add_argument("--jsonl_template", default="./final_experiments_for_paper/experiment1/train_val_pgns_twic/{gm}_{split}_dpo.jsonl")
    ap.add_argument("--gm_ckpt_dir_template", default="./final_experiments_for_paper/experiment1/trained_models_twic/{gm}/")
    ap.add_argument("--exp2_gm_ckpt_dir_template", default="./final_experiments_for_paper/experiment2_style_model/trained_models_single_gm_twic/{gm}/")
    ap.add_argument("--out_root", default="./final_experiments_for_paper/experiment3/depth_study_validation_results/{gm}/family_eval_{split}")

    # SF-helper options
    ap.add_argument("--sf_path", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--sf_depth", type=parse_int_list, default=[5], help="Stockfish depth (human-likeness sweep target).")
    ap.add_argument("--sf_tops", type=parse_int_list, default=[10], help="MultiPV candidates.")
    ap.add_argument("--sf_uci_elo", default="none", help="none or integer (e.g. 1600).")
    ap.add_argument("--restrict_cp_window", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disable_initial_model_types", action="store_true")

    args = ap.parse_args()

    device = pick_device("auto")

    jsonl_path = args.jsonl_template.format(gm=args.gm_name, split=args.split)
    ds = DpoPairs(jsonl_path)
    print(f"dpo pairs has length: {len(ds)}")
    
    import random
    rng = random.Random(0) 
    rng.shuffle(ds.rows)
    #ds.rows = ds.rows[:500]

    gm_ckpt_dir = Path(args.gm_ckpt_dir_template.format(gm=args.gm_name))
    experiment2_gm_ckpt_dir = Path(args.exp2_gm_ckpt_dir_template.format(gm=args.gm_name))
    out_dir = Path(args.out_root.format(gm=args.gm_name, split=args.split))
    out_dir.mkdir(parents=True, exist_ok=True)

    sf_cfgs = []
    uci_elo = None if args.sf_uci_elo.lower() in ("none", "null", "full", "max") else int(args.sf_uci_elo)
    for depth in args.sf_depth:
        for topk in args.sf_tops:
            sf_cfg = SfConfig(
                stockfish_path=args.sf_path,
                depth=depth,
                multipv_topk=topk,
                uci_elo=uci_elo,
                restrict_cp_window=int(args.restrict_cp_window),
                temperature=float(args.temperature),
                sample=bool(args.sample),
                seed=int(args.seed),
                threads=32,
                use_gibbs=True,
            )
            sf_cfgs.append(sf_cfg)

    models = build_sf_models_for_gm(
        maia_type=args.maia_type,
        device=device,
        experiment1_gm_dir=gm_ckpt_dir,
        experiment2_gm_dir=experiment2_gm_ckpt_dir,
        sf_cfgs=sf_cfgs,
    )

    results = []
    try:
        for m in models:
            m_out = out_dir / m.tag
            m_out.mkdir(parents=True, exist_ok=True)
            res = m.run_eval(ds=ds, batch_size=int(args.batch_size), out_dir=m_out, gm_name=args.gm_name, n_boot=1000)
            results.append(res)
            print(f"[done] {m.tag} -> {m_out}")
    finally:
        for m in models:
            m.close()

    # one combined summary file
    (out_dir / "summary_all.json").write_text(json.dumps(results, indent=2))
    print(f"[saved] {out_dir / 'summary_all.json'}")


if __name__ == "__main__":
    main()
