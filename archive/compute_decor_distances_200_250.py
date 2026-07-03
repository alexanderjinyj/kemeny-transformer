import os
import numpy as np
import pandas as pd
from numba import njit

@njit()
def kemeny_dist_paralle(base_rankings, candidate_ranking):
    nb_voters = base_rankings.shape[0]
    nb_candidates = base_rankings.shape[1]
    kemeny_dist = 0
    for i in range(nb_voters):
        tau = 0
        base_ranking = base_rankings[i]
        for j in range(nb_candidates):
            for k in range(j+1, nb_candidates):
                if (np.sign(candidate_ranking[j] - candidate_ranking[k]) == -np.sign(base_ranking[j] - base_ranking[k])):
                    tau += 1
        kemeny_dist += tau

    kemeny_dist = kemeny_dist / nb_voters
    return kemeny_dist

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

def process_decor_results():
    voters = [6, 8, 10]
    items = [200, 250]
    dataset_types = ['random', 'repeat', 'jiggling']
    
    for dt in dataset_types:
        for v in voters:
            for i in items:
                decor_csv = f"test_dataset/test_dataset_{dt}/decor_result/test_dataset_{dt}_v{v}_i{i}_decor_rankings.csv"
                base_npy = f"test_dataset/test_dataset_{dt}/test_dataset_{dt}_nvoters_{v}_nitems_{i}.npy"
                output_csv = f"test_dataset/test_dataset_{dt}/decor_result/test_dataset_{dt}_v{v}_i{i}_decor_distances.csv"
                
                if not os.path.exists(decor_csv) or not os.path.exists(base_npy):
                    continue
                
                print(f"Processing: {os.path.basename(decor_csv)}")
                
                try:
                    # Using on_bad_lines='skip' to handle the corruption observed earlier
                    df_decor = pd.read_csv(decor_csv, on_bad_lines='skip')
                    
                    # Ensure we have the right columns
                    rank_cols = [f"rank_{j}" for j in range(1, i + 1)]
                    if 'iteration_id' not in df_decor.columns:
                        print(f"  Error: iteration_id missing in {decor_csv}")
                        continue
                        
                    # Extract rankings and batch indices (iteration_id is 1-indexed in R)
                    batch_indices = df_decor['iteration_id'].values - 1
                    decor_rankings = df_decor[rank_cols].values - 1
                    
                    # Load base dataset
                    base_dataset = np.load(base_npy)
                    
                    # Compute distances
                    results = compute_kemeny_distance_parallel(base_dataset, decor_rankings, batch_indices)
                    
                    # Save to CSV
                    df_results = pd.DataFrame(results)
                    df_results.to_csv(output_csv, index=False)
                    print(f"  Saved {len(df_results)} results to {output_csv}")
                    
                except Exception as e:
                    print(f"  Error processing {decor_csv}: {e}")

if __name__ == "__main__":
    process_decor_results()
