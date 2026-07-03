#!/usr/bin/env python3
"""Borda + footrule scoring for the 200/250-item ablation extension.
Identical logic to compute_borda_footrule.py but ITEMS = [200, 250]."""

import os
import numpy as np
import pandas as pd

from data_synthesis import kemeny_distance_batch
from Borda import borda_from_ranking_position_batch
from Spearsman import footrule_consensus_batch

SRC_FMT = "test_dataset/test_dataset_{t}/test_dataset_{t}_nvoters_{v}_nitems_{i}.npy"
OUT_ROOT = "test_results/borda_footrule"
DATASET_TYPES = ["jiggling", "random", "repeat"]
VOTERS = [6, 8, 10]
ITEMS = [200, 250]


def main():
    for t in DATASET_TYPES:
        out_dir = os.path.join(OUT_ROOT, t)
        os.makedirs(out_dir, exist_ok=True)
        for v in VOTERS:
            for i in ITEMS:
                src = SRC_FMT.format(t=t, v=v, i=i)
                if not os.path.exists(src):
                    print(f"[skip] missing {src}")
                    continue
                base = np.load(src, allow_pickle=True).astype(np.int64)
                inst = [base[b] for b in range(base.shape[0])]
                borda = borda_from_ranking_position_batch(base)
                footrule = footrule_consensus_batch(inst)
                borda_kem = kemeny_distance_batch(inst, [borda[b] for b in range(len(inst))])
                foot_kem = kemeny_distance_batch(inst, [footrule[b] for b in range(len(inst))])
                df = pd.DataFrame({"sample_id": np.arange(len(inst)),
                                   "borda_kemeny": borda_kem,
                                   "footrule_kemeny": foot_kem})
                df.to_csv(os.path.join(out_dir,
                          f"test_dataset_{t}_nvoters_{v}_nitems_{i}_borda_footrule.csv"),
                          index=False)
                print(f"{t} v{v} i{i}: borda={borda_kem.mean():8.1f}  footrule={foot_kem.mean():8.1f}")


if __name__ == "__main__":
    main()
