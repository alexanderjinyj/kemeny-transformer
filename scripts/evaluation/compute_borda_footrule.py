#!/usr/bin/env python3
"""
Score the Borda and Spearman-footrule consensus methods on the jiggling / random
/ repeat test sets, using the same Kemeny distance as everything else.

Base .npy files are (N, voters, items) with values = item->position rank, which is
exactly the input format both methods expect. Writes per-config CSVs with
sample_id, borda_kemeny, footrule_kemeny under results/borda_footrule/.

Usage:
    PYTHONPATH=. python scripts/evaluation/compute_borda_footrule.py
    PYTHONPATH=. python scripts/evaluation/compute_borda_footrule.py --items 200 250
"""

import argparse
import os
import numpy as np
import pandas as pd

from kemeny_transformer.data.synthesis import kemeny_distance_batch
from kemeny_transformer.baselines.borda import borda_from_ranking_position_batch
from kemeny_transformer.baselines.spearman import footrule_consensus_batch

SRC_FMT = "data/test/{t}/test_dataset_{t}_nvoters_{v}_nitems_{i}.npy"
OUT_ROOT = "results/borda_footrule"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--types", nargs="+", default=["jiggling", "random", "repeat"])
    parser.add_argument("--voters", nargs="+", type=int, default=[6, 8, 10])
    parser.add_argument("--items", nargs="+", type=int, default=[100, 125, 150])
    args = parser.parse_args()

    summary = []
    for t in args.types:
        out_dir = os.path.join(OUT_ROOT, t)
        os.makedirs(out_dir, exist_ok=True)
        for v in args.voters:
            for i in args.items:
                src = SRC_FMT.format(t=t, v=v, i=i)
                if not os.path.exists(src):
                    print(f"[skip] missing {src}")
                    continue
                base = np.load(src, allow_pickle=True).astype(np.int64)  # (N, voters, items)
                inst = [base[b] for b in range(base.shape[0])]

                borda = borda_from_ranking_position_batch(base)          # (N, items)
                footrule = footrule_consensus_batch(inst)                # (N, items)

                borda_kem = kemeny_distance_batch(inst, [borda[b] for b in range(len(inst))])
                foot_kem = kemeny_distance_batch(inst, [footrule[b] for b in range(len(inst))])

                df = pd.DataFrame({"sample_id": np.arange(len(inst)),
                                   "borda_kemeny": borda_kem,
                                   "footrule_kemeny": foot_kem})
                df.to_csv(os.path.join(out_dir,
                          f"test_dataset_{t}_nvoters_{v}_nitems_{i}_borda_footrule.csv"),
                          index=False)
                summary.append({"dataset": t, "n_voters": v, "n_items": i,
                                "borda_mean": float(borda_kem.mean()),
                                "footrule_mean": float(foot_kem.mean())})
                print(f"{t} v{v} i{i}: borda={borda_kem.mean():8.1f}  "
                      f"footrule={foot_kem.mean():8.1f}")
    pd.DataFrame(summary).to_csv(os.path.join(OUT_ROOT, "summary_means.csv"), index=False)
    print(f"\nSaved summary -> {os.path.join(OUT_ROOT, 'summary_means.csv')}")


if __name__ == "__main__":
    main()
