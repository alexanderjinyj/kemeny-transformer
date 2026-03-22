"""
Unified result analysis script. Replaces:
  result_analysis_8_100_base_rankings_random.py
  result_analysis_8_100_base_rankings_repeat.py
  result_analysis_8_100_base_rankings_jiggling.py
  result_analysis_fine_tuning_8_100_base_rankings_random.py
  result_analysis_fine_tuning_8_100_base_rankings_repeat.py
  result_analysis_fine_tuning_8_100_base_rankings_jiggling.py
  scalarity_result_ananlysis.py

Usage examples:
  python scripts/run_result_analysis.py --data-type random
  python scripts/run_result_analysis.py --data-type repeat --fine-tuning
  python scripts/run_result_analysis.py --scalarity --data-types random jiggling repeat --num-candidates 20 50 100 150
"""
import argparse
import os

import numpy as np
from numba import njit

from kemeny_transformer.utils.kemeny_distance import (
    compute_kemeny_distance_parallel_greedy,
    compute_kemeny_distance_parallel_beam_search,
    permutation_to_ranking_greedy,
    permutation_to_ranking_beam_search,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run result analysis')
    parser.add_argument('--data-type', default=None, choices=['random', 'repeat', 'jiggling'],
                        help='Data type for standard/fine-tuning analysis')
    parser.add_argument('--data-types', nargs='+', default=['random', 'jiggling', 'repeat'],
                        help='Data types for scalarity analysis')
    parser.add_argument('--fine-tuning', action='store_true',
                        help='Use fine-tuning results')
    parser.add_argument('--scalarity', action='store_true',
                        help='Run scalarity analysis mode')
    parser.add_argument('--num-candidates', nargs='+', type=int, default=[20, 50, 100, 150],
                        help='Number of candidates for scalarity analysis')
    parser.add_argument('--data-dir', default='test_dataset',
                        help='Base data directory')
    return parser.parse_args()


def compute_gap_metrics(kemeny_distances, optimal_distances):
    """Compute gap and percentage gap metrics."""
    gap = kemeny_distances - optimal_distances
    gap_mean = np.mean(gap)
    gap_percent = np.divide(gap, optimal_distances)
    gap_percent = np.where(np.isnan(gap_percent), 0, gap_percent)
    gap_percent = np.where(gap_percent == np.inf, np.nan, gap_percent)
    gap_percent_mean = np.nanmean(gap_percent)
    return gap, gap_mean, gap_percent_mean


def load_transformer_data(npy_data, data_type):
    """Load transformer results, handling both .item().get() and .get() patterns."""
    if hasattr(npy_data, 'item'):
        try:
            data_dict = npy_data.item()
            if isinstance(data_dict, dict):
                return data_dict
        except (ValueError, AttributeError):
            pass
    # Try direct dict access (for non-0d arrays)
    if hasattr(npy_data, '__getitem__') and not isinstance(npy_data, np.ndarray):
        return npy_data
    # If it's a 0-d array wrapping a dict
    return npy_data


def run_standard_analysis(data_type, fine_tuning, data_dir):
    """Run analysis for a single data type (standard or fine-tuning)."""
    dt = data_type

    # Load Gurobi results
    gurobi_data = np.load(
        f'{data_dir}/gurobi_result/test_dataset_{dt}_kemeny_optimal_ranking_gurobi.npy',
        allow_pickle=True
    )
    optimal_distances = np.load(
        f'{data_dir}/gurobi_result/test_dataset_{dt}_kemeny_optimal_ranking_distance_gurobi.npy',
        allow_pickle=True
    )
    gurobi_running_time = gurobi_data.item().get('kemeny_rankings_running_time')
    gurobi_running_time_total = np.sum(gurobi_running_time)
    print(f'gurobi running time mean:{gurobi_running_time_total}')

    # Load base rankings
    base_rankings = np.load(f'{data_dir}/test_dataset_{dt}.npy')

    # Load transformer results
    if fine_tuning:
        transformer_path = f'{data_dir}/transformer_test_result/test_data_fine_tuning/test_dataset_{dt}_transformer.npy'
    else:
        transformer_path = f'{data_dir}/transformer_test_result/test_data/test_dataset_{dt}_transformer.npy'

    transformer_data_raw = np.load(transformer_path, allow_pickle=True)
    transformer_data = load_transformer_data(transformer_data_raw, dt)

    # Extract greedy results
    if isinstance(transformer_data, dict):
        td = transformer_data
    else:
        td = transformer_data.item() if hasattr(transformer_data, 'item') else transformer_data

    greedy_permutations = np.array(td.get(f'final ranking greedy {dt} permutation'))
    greedy_running_time = td.get(f'greedy time {dt}')
    beam_permutations = np.array(td.get(f'final ranking beam search {dt} permutation'))
    print(f'shape:{beam_permutations.shape}')
    beam_running_time = td.get(f'beam search time {dt}')

    # Compute rankings from permutations
    greedy_rankings = permutation_to_ranking_greedy(greedy_permutations)
    beam_rankings = permutation_to_ranking_beam_search(beam_permutations)

    # Compute Kemeny distances
    greedy_distances = compute_kemeny_distance_parallel_greedy(base_rankings, greedy_rankings)
    beam_distances, beam_optimal_rankings = compute_kemeny_distance_parallel_beam_search(base_rankings, beam_rankings)

    # Compute metrics
    greedy_mean = np.nanmean(greedy_distances)
    greedy_gap, greedy_gap_mean, greedy_gap_percent = compute_gap_metrics(greedy_distances, optimal_distances)
    beam_mean = np.nanmean(beam_distances)
    beam_gap, beam_gap_mean, beam_gap_percent = compute_gap_metrics(beam_distances, optimal_distances)

    # Build analysis data
    analysis_data = {
        'gurobi running time ': gurobi_running_time_total,
        'transformer greedy running time': greedy_running_time,
        'transformer beam search running time': beam_running_time,
        'transformer greedy kemeny distance gap': greedy_gap,
        'transformer beam search kemeny distance gap': beam_gap,
        'transformer greedy kemeny distance mean': greedy_mean,
        'transformer beam search kemeny distance mean': beam_mean,
        'transformer greedy kemeny distance gap mean': greedy_gap_mean,
        'transformer beam search kemeny distance gap mean': beam_gap_mean,
        'transformer greedy kemeny distance gap mean percent': greedy_gap_percent,
        'transformer beam search kemeny distance gap mean percent': beam_gap_percent,
        'transformer beam search final optimal rankings ': beam_optimal_rankings,
    }

    # Print results
    print(f'gurobi running time:{gurobi_running_time_total},'
          f'greedy running time:{greedy_running_time},'
          f'beam search running time:{beam_running_time},'
          f'greedy kemeny distance mean:{greedy_mean},'
          f'beam search kemeny distance mean:{beam_mean},'
          f'greedy gap mean:{greedy_gap_mean},'
          f'beam search gap mean:{beam_gap_mean},'
          f'transformer greedy kemeny distance gap mean percent:{greedy_gap_percent},'
          f'transformer beam search kemeny distance gap mean percent:{beam_gap_percent},')

    # Save results
    if fine_tuning:
        output_dir = f'{data_dir}/analysis_data/fine_tuning'
        output_file = f'{output_dir}/analysis_data_2000_{dt}_fine_tuning.npy'
    else:
        output_dir = f'{data_dir}/analysis_data'
        output_file = f'{output_dir}/analysis_data_2000_{dt}.npy'

    os.makedirs(output_dir, exist_ok=True)
    np.save(output_file, analysis_data)
    print(f'Saved to: {output_file}')


def run_scalarity_analysis(data_types, num_candidates_list, data_dir):
    """Run scalarity analysis across multiple data types and candidate counts."""
    for data_type in data_types:
        for nb_candidates in num_candidates_list:
            print(f"\n{'='*60}")
            print(f'{data_type=} {nb_candidates=}')
            print(f"{'='*60}")

            result = {}
            gurobi_data = np.load(
                f"{data_dir}/gurobi_result/scalarity/test_dataset_{nb_candidates}_{data_type}_kemeny_optimal_ranking_gurobi.npy",
                allow_pickle=True
            )
            result_dict = gurobi_data.item()
            optimal_distances = result_dict.get("kemeny_distances")
            gurobi_running_time = result_dict.get('kemeny_rankings_running_time')
            gurobi_running_time_mean = np.mean(gurobi_running_time)
            gurobi_size = gurobi_running_time.shape[0]

            print(f'gurobi running time mean:{gurobi_running_time_mean}')

            base_rankings = np.load(f"{data_dir}/test_dataset_{nb_candidates}_{data_type}.npy")[0:gurobi_size]

            transformer_data = np.load(
                f"{data_dir}/transformer_test_result/fine_tuning/test_dataset_{nb_candidates}_{data_type}_transformer.npy",
                allow_pickle=True
            )

            greedy_permutations = np.array(transformer_data[f'final ranking greedy {data_type} permutation'][0:gurobi_size])
            greedy_running_time = transformer_data[f'greedy time {data_type}']

            greedy_rankings = permutation_to_ranking_greedy(greedy_permutations)
            greedy_distances = compute_kemeny_distance_parallel_greedy(base_rankings, greedy_rankings)

            greedy_mean = np.nanmean(greedy_distances)
            greedy_gap, greedy_gap_mean, greedy_gap_percent = compute_gap_metrics(greedy_distances, optimal_distances)

            analysis_data = {
                'gurobi running time': gurobi_running_time_mean,
                'transformer greedy running time': greedy_running_time / gurobi_size,
                'transformer greedy kemeny distance mean': greedy_mean,
                'transformer greedy kemeny distance gap': greedy_gap,
                'transformer greedy kemeny distance gap mean': greedy_gap_mean,
                'transformer greedy kemeny distance gap mean percent': greedy_gap_percent,
                'transformer greedy search final optimal rankings': greedy_rankings,
            }

            print(f"gurobi running time:{analysis_data['gurobi running time']},"
                  f"greedy running time:{analysis_data['transformer greedy running time']},"
                  f"greedy kemeny distance mean:{greedy_mean},"
                  f"greedy gap mean:{greedy_gap_mean},"
                  f"transformer greedy kemeny distance gap mean percent:{greedy_gap_percent},")

            output_dir = f"{data_dir}/analysis_data/fine_tuning/sclarity"
            os.makedirs(output_dir, exist_ok=True)
            np.save(f'{output_dir}/analysis_data_{nb_candidates}_{data_type}_fine_tuning.npy', analysis_data)


def main():
    args = parse_args()

    if args.scalarity:
        run_scalarity_analysis(args.data_types, args.num_candidates, args.data_dir)
    else:
        if args.data_type is None:
            print("Error: --data-type is required for standard/fine-tuning mode")
            return
        run_standard_analysis(args.data_type, args.fine_tuning, args.data_dir)


if __name__ == '__main__':
    main()
