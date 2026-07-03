#!/usr/bin/env python3
"""
Compute Kemeny distances for DECoR consensus rankings (produced by the R
scripts) against the base test datasets.

Reads results/decor/{type}/..._decor_rankings.csv and writes
..._decor_distances.csv next to each input.

Usage:
    PYTHONPATH=. python scripts/evaluation/compute_decor_distances.py
    PYTHONPATH=. python scripts/evaluation/compute_decor_distances.py --items 200 250
"""

import argparse
import os
import numpy as np
import pandas as pd

from kemeny_transformer.utils.kemeny_distance import kemeny_dist_paralle

DECOR_FMT = "results/decor/{dt}/test_dataset_{dt}_v{v}_i{i}_decor_rankings.csv"
BASE_FMT = "data/test/{dt}/test_dataset_{dt}_nvoters_{v}_nitems_{i}.npy"
OUT_FMT = "results/decor/{dt}/test_dataset_{dt}_v{v}_i{i}_decor_distances.csv"


def compute_kemeny_distance_parallel(batch_base_rankings, final_rankings, batch_indices):
    kemeny_distances = []
    for i, batch_idx in enumerate(batch_indices):
        if batch_idx >= batch_base_rankings.shape[0]:
            print(f"Warning: batch_idx {batch_idx} out of range for base_rankings")
            continue
        base_rankings = batch_base_rankings[batch_idx]
        final_ranking = final_rankings[i]
        kemeny_distance = kemeny_dist_paralle(base_rankings, final_ranking)
        kemeny_distances.append({'batch_idx': batch_idx, 'kemeny_distance': kemeny_distance})
    return kemeny_distances


def process_decor_results(dataset_types, voters, items):
    for dt in dataset_types:
        for v in voters:
            for i in items:
                decor_csv = DECOR_FMT.format(dt=dt, v=v, i=i)
                base_npy = BASE_FMT.format(dt=dt, v=v, i=i)
                output_csv = OUT_FMT.format(dt=dt, v=v, i=i)

                if not os.path.exists(decor_csv) or not os.path.exists(base_npy):
                    continue

                print(f"Processing: {os.path.basename(decor_csv)}")

                try:
                    # Using on_bad_lines='skip' to handle the corruption observed earlier
                    df_decor = pd.read_csv(decor_csv, on_bad_lines='skip')

                    rank_cols = [f"rank_{j}" for j in range(1, i + 1)]
                    if 'iteration_id' not in df_decor.columns:
                        print(f"  Error: iteration_id missing in {decor_csv}")
                        continue

                    # iteration_id is 1-indexed in R
                    batch_indices = df_decor['iteration_id'].values - 1
                    decor_rankings = df_decor[rank_cols].values - 1

                    base_dataset = np.load(base_npy)

                    results = compute_kemeny_distance_parallel(base_dataset, decor_rankings, batch_indices)

                    df_results = pd.DataFrame(results)
                    df_results.to_csv(output_csv, index=False)
                    print(f"  Saved {len(df_results)} results to {output_csv}")

                except Exception as e:
                    print(f"  Error processing {decor_csv}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score DECoR rankings with Kemeny distance.")
    parser.add_argument("--types", nargs="+", default=["random", "repeat", "jiggling"])
    parser.add_argument("--voters", nargs="+", type=int, default=[6, 8, 10])
    parser.add_argument("--items", nargs="+", type=int, default=[90, 100, 110, 125, 150])
    args = parser.parse_args()
    process_decor_results(args.types, args.voters, args.items)
