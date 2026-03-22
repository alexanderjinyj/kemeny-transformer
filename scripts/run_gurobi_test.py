"""
Unified Gurobi test script. Replaces:
  gurobi_test_random.py, gurobi_test_repeat.py, gurobi_test_jiggling.py,
  gurobi_test_50_random.py, gurobi_test_50_repeat.py, gurobi_test_50_jiggling.py,
  gurobi_test_100_25.py, gurobi_test_75_50.py, gurobi_scalarity_test.py

Usage examples:
  # Standard mode (replaces gurobi_test_random.py):
  python scripts/run_gurobi_test.py --mode standard --data-types random

  # Ablation mode (replaces gurobi_test_50_random.py):
  python scripts/run_gurobi_test.py --mode ablation --data-types random --num-candidates 50

  # Ablation with multiple (replaces gurobi_test_100_25.py):
  python scripts/run_gurobi_test.py --mode ablation --data-types random repeat jiggling --num-candidates 25

  # Ablation with multiple candidates (replaces gurobi_test_75_50.py):
  python scripts/run_gurobi_test.py --mode ablation --data-types random repeat jiggling --num-candidates 75 50

  # Scalarity mode (replaces gurobi_scalarity_test.py):
  python scripts/run_gurobi_test.py --mode scalarity --data-types repeat --num-candidates 150 --max-samples 25
"""
import argparse
import os
import os.path
from time import time

import numpy as np

from kemeny_transformer.utils.gurobi_solver import aggregate_kemeny
from kemeny_transformer.utils.kemeny_distance import compute_kemeny_distance_parallel


def parse_args():
    parser = argparse.ArgumentParser(description='Run Gurobi Kemeny optimal ranking test')
    parser.add_argument('--mode', choices=['standard', 'ablation', 'scalarity'], required=True,
                        help='Test mode: standard (100 items), ablation (variable items), scalarity (scalability)')
    parser.add_argument('--data-types', nargs='+', default=['random', 'repeat', 'jiggling'],
                        help='Data types to test')
    parser.add_argument('--num-candidates', nargs='+', type=int, default=[100],
                        help='Number of candidates to test')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of samples (e.g., 25 for scalarity)')
    parser.add_argument('--data-dir', default='test_dataset',
                        help='Base data directory')
    return parser.parse_args()


def run_gurobi_test(test_dataset, output_path, numb_candidates=None):
    """Run Gurobi solver on a test dataset with checkpoint/resume support."""
    bsz = test_dataset.shape[0]
    numb_voters = test_dataset.shape[1]
    if numb_candidates is None:
        numb_candidates = test_dataset.shape[2]

    kemeny_rankings = np.empty(shape=(bsz, numb_candidates), dtype=np.float32)
    kemeny_rankings_running_time = np.empty(shape=(bsz))
    start_batch = 0

    result = {
        'kemeny_rankings': kemeny_rankings,
        'kemeny_rankings_running_time': kemeny_rankings_running_time,
        'idx_batch': start_batch,
    }

    # Resume from checkpoint if exists
    if os.path.isfile(output_path):
        result_load = np.load(output_path, allow_pickle=True)
        result_dict = result_load.item() if hasattr(result_load, 'item') else result_load
        # Try to find kemeny_rankings under various key patterns
        for key in result_dict:
            if 'kemeny_rankings' in key and 'running_time' not in key and 'distance' not in key:
                kemeny_rankings = result_dict[key]
                break
        for key in result_dict:
            if 'running_time' in key:
                kemeny_rankings_running_time = result_dict[key]
                break
        start_batch = result_dict.get('idx_batch', 0) + 1
        print(f'Resumed from batch {start_batch}')

    start = time()
    for i in range(start_batch, bsz):
        base_rankings = test_dataset[i]
        kemeny_ranking, running_time = aggregate_kemeny(numb_voters, numb_candidates, base_rankings)
        kemeny_rankings_running_time[i] = running_time

        print(f'gurobi running time : {running_time}, index batch: {i}')
        kemeny_rankings[i] = np.array(kemeny_ranking).astype(np.float32)
        result = {
            'kemeny_rankings': kemeny_rankings,
            'kemeny_rankings_running_time': kemeny_rankings_running_time,
            'idx_batch': i,
        }
        np.save(output_path, result)

    kemeny_rankings_distance = compute_kemeny_distance_parallel(test_dataset, kemeny_rankings)
    result["kemeny_distances"] = kemeny_rankings_distance
    print(f'gurobi time: {time()-start}')
    np.save(output_path, result)

    return kemeny_rankings_distance


def main():
    args = parse_args()

    for num in args.num_candidates:
        for data_type in args.data_types:
            print(f'\n{"="*60}')
            print(f'Mode: {args.mode}, Data type: {data_type}, Candidates: {num}')
            print(f'{"="*60}')

            # Determine input/output paths based on mode
            if args.mode == 'standard':
                data_file = f'{args.data_dir}/test_dataset_{data_type}.npy'
                output_dir = f'{args.data_dir}/gurobi_result'
                output_path = f'{output_dir}/test_dataset_{data_type}_kemeny_optimal_ranking_gurobi.npy'
                distance_path = f'{output_dir}/test_dataset_{data_type}_kemeny_optimal_ranking_distance_gurobi.npy'
            elif args.mode == 'ablation':
                data_file = f'{args.data_dir}/ablation_test_dataset/test_dataset_{data_type}_{num}.npy'
                output_dir = f'{args.data_dir}/gurobi_result/ablation_gurobi_test'
                output_path = f'{output_dir}/test_dataset_{data_type}_{num}_kemeny_optimal_ranking_gurobi.npy'
                distance_path = f'{output_dir}/test_dataset_{data_type}_{num}_kemeny_optimal_ranking_distance_gurobi.npy'
            elif args.mode == 'scalarity':
                data_file = f'{args.data_dir}/test_dataset_{num}_{data_type}.npy'
                output_dir = f'{args.data_dir}/gurobi_result/scalarity'
                output_path = f'{output_dir}/test_dataset_{num}_{data_type}_kemeny_optimal_ranking_gurobi.npy'
                distance_path = None  # Scalarity saves everything in one file

            os.makedirs(output_dir, exist_ok=True)

            # Load dataset
            test_dataset = np.load(data_file)
            if args.max_samples is not None:
                test_dataset = test_dataset[0:args.max_samples]

            print(f'Dataset shape: {test_dataset.shape}')

            # Run test
            kemeny_rankings_distance = run_gurobi_test(
                test_dataset, output_path, numb_candidates=num
            )

            # Save distance separately for standard/ablation modes
            if distance_path is not None:
                np.save(distance_path, kemeny_rankings_distance)


if __name__ == '__main__':
    main()
