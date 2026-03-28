from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from .train_configs import STUDIES
from .train_style_encoder import train_one_run


def rank_key(result: dict) -> float:
    # lower is better
    return float(result.get("best_eval_loss", 1e18))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--studies",
        nargs="+",
        required=True,
        help="Study names from train_configs.STUDIES",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
    )
    args = ap.parse_args()

    selected = []
    for name in args.studies:
        if name not in STUDIES:
            raise ValueError(f"Unknown study: {name}")
        selected.append(STUDIES[name])

    results: List[dict] = []
    failures: List[dict] = []

    for idx, cfg in enumerate(selected, start=1):
        print(f"[runner] ({idx}/{len(selected)}) starting {cfg.run_name()}")
        t0 = time.time()
        try:
            res = train_one_run(cfg)
            res["elapsed_sec"] = time.time() - t0
            results.append(res)
            print(f"[runner] finished {cfg.run_name()} best_eval_loss={res['best_eval_loss']:.6f}")
        except Exception as e:
            fail = {
                "study": cfg.run_name(),
                "error": repr(e),
                "elapsed_sec": time.time() - t0,
            }
            failures.append(fail)
            print(f"[runner] FAILED {cfg.run_name()} {repr(e)}")
            if args.stop_on_error:
                break

    ranked = sorted(results, key=rank_key)

    print("\n=== Ranked Results ===")
    for i, r in enumerate(ranked, start=1):
        print(
            f"{i}. {r['run_name']} "
            f"best_eval_loss={r['best_eval_loss']:.6f} "
            f"summary={r['summary_path']}"
        )

    if failures:
        print("\n=== Failures ===")
        for f in failures:
            print(json.dumps(f, indent=2))


if __name__ == "__main__":
    main()