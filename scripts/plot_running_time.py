"""
Plot running time comparison: Kemeny Transformer (greedy) vs Gurobi.

Reads scalarity analysis results and produces log-scale running time plots
for each dataset type (jiggling, random, repeat).

Usage:
  PYTHONPATH=. python scripts/plot_running_time.py

  # Custom paths:
  PYTHONPATH=. python scripts/plot_running_time.py \
    --data-dir test_dataset \
    --output-dir image \
    --num-candidates 20 50 100 150 \
    --data-types jiggling random repeat
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Plot running time comparison')
    parser.add_argument('--data-dir', default='test_dataset',
                        help='Base data directory containing analysis results')
    parser.add_argument('--output-dir', default='image',
                        help='Directory to save plots')
    parser.add_argument('--num-candidates', nargs='+', type=int, default=[20, 50, 100, 150],
                        help='Number of candidates to plot on x-axis')
    parser.add_argument('--data-types', nargs='+', default=['jiggling', 'random', 'repeat'],
                        help='Dataset types to generate plots for')
    return parser.parse_args()


def load_running_times(data_dir, data_type, num_candidates_list):
    """Load Gurobi and Transformer running times from scalarity analysis results."""
    gurobi_times = []
    transformer_times = []
    valid_candidates = []

    for nb_candidates in num_candidates_list:
        # Try loading from scalarity analysis results first
        analysis_path = f"{data_dir}/analysis_data/fine_tuning/sclarity/analysis_data_{nb_candidates}_{data_type}_fine_tuning.npy"
        if os.path.exists(analysis_path):
            analysis = np.load(analysis_path, allow_pickle=True).item()
            gurobi_time = analysis.get('gurobi running time', None)
            transformer_time = analysis.get('transformer greedy running time', None)
            if gurobi_time is not None and transformer_time is not None:
                gurobi_times.append(float(gurobi_time))
                transformer_times.append(float(transformer_time))
                valid_candidates.append(nb_candidates)
                continue

        # Fallback: load directly from gurobi result + transformer result
        gurobi_path = f"{data_dir}/gurobi_result/scalarity/test_dataset_{nb_candidates}_{data_type}_kemeny_optimal_ranking_gurobi.npy"
        transformer_path = f"{data_dir}/transformer_test_result/fine_tuning/test_dataset_{nb_candidates}_{data_type}_transformer.npy"

        if os.path.exists(gurobi_path) and os.path.exists(transformer_path):
            gurobi_data = np.load(gurobi_path, allow_pickle=True).item()
            gurobi_running_time = gurobi_data.get('kemeny_rankings_running_time')
            gurobi_size = gurobi_running_time.shape[0]
            gurobi_time_mean = np.mean(gurobi_running_time)

            transformer_data = np.load(transformer_path, allow_pickle=True)
            if hasattr(transformer_data, 'item'):
                transformer_data = transformer_data.item()
            greedy_time = transformer_data.get(f'greedy time {data_type}', None)
            if greedy_time is None:
                # Try alternate key patterns
                for key in transformer_data:
                    if 'greedy time' in key:
                        greedy_time = transformer_data[key]
                        break
            if greedy_time is not None:
                transformer_time_mean = float(greedy_time) / gurobi_size
                gurobi_times.append(gurobi_time_mean)
                transformer_times.append(transformer_time_mean)
                valid_candidates.append(nb_candidates)
                continue

        print(f"  Warning: No data found for {data_type} with {nb_candidates} candidates")

    return valid_candidates, gurobi_times, transformer_times


def plot_running_time(data_type, candidates, gurobi_times, transformer_times, output_path):
    """Create a log-scale running time comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    log_gurobi = np.log(gurobi_times)
    log_transformer = np.log(transformer_times)

    ax.plot(candidates, log_gurobi, 'o-', color='tab:blue',
            linewidth=2.5, markersize=10, label='Gurobi')
    ax.plot(candidates, log_transformer, 'o-', color='tab:orange',
            linewidth=2.5, markersize=10, label='Transformer Greedy')

    ax.set_xlabel('Number of Items', fontsize=16, fontweight='bold')
    ax.set_ylabel('Log Mean Running Time (seconds)', fontsize=16, fontweight='bold')
    ax.set_title(f'Kemeny Transformer vs Gurobi  ({data_type.capitalize()})',
                 fontsize=18, fontweight='bold')
    ax.set_xticks(candidates)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=14)
    ax.grid(False)

    # Thicker spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print("Generating running time plots")
    print(f"{'='*60}")

    for data_type in args.data_types:
        print(f"\n--- {data_type.capitalize()} ---")
        candidates, gurobi_times, transformer_times = load_running_times(
            args.data_dir, data_type, args.num_candidates
        )

        if len(candidates) == 0:
            print(f"  No data found for {data_type}. Skipping.")
            continue

        print(f"  Candidates: {candidates}")
        print(f"  Gurobi times (mean/sample): {[f'{t:.4f}' for t in gurobi_times]}")
        print(f"  Transformer times (mean/sample): {[f'{t:.6f}' for t in transformer_times]}")

        output_path = os.path.join(args.output_dir, f"running time {data_type}.png")
        plot_running_time(data_type, candidates, gurobi_times, transformer_times, output_path)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
