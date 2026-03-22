"""
Plot normalized gap-from-best comparison across all methods.

Methods: Max Agreement (H1), Min Regret (H2), KwikSort, Markov Chain, DECOR, Kemeny Transformer.

Reads per-sample Kemeny distances from:
  - traditional_methods_result/{dataset_type}/ (H1, H2, KwikSort, MC)
  - test_dataset_{type}/decor_result/ (DECOR)
  - Transformer inference results (from scripts/inference.py output)

Produces plots like: image/comparison_repeat_items_100_125_150.png

Usage:
  PYTHONPATH=. python scripts/plot_results.py

  # Custom configs:
  PYTHONPATH=. python scripts/plot_results.py \
    --data-dir test_dataset \
    --output-dir image \
    --voter-configs 6 8 10 \
    --item-configs 100 125 150 \
    --data-types jiggling random repeat
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Plot gap-from-best comparison')
    parser.add_argument('--data-dir', default='test_dataset',
                        help='Base data directory')
    parser.add_argument('--output-dir', default='image',
                        help='Directory to save plots')
    parser.add_argument('--voter-configs', nargs='+', type=int, default=[6, 8, 10],
                        help='Number of voters')
    parser.add_argument('--item-configs', nargs='+', type=int, default=[100, 125, 150],
                        help='Number of items')
    parser.add_argument('--data-types', nargs='+', default=['jiggling', 'random', 'repeat'],
                        help='Dataset types')
    parser.add_argument('--transformer-result-dir', default=None,
                        help='Directory containing transformer .npy results (default: auto-detect)')
    return parser.parse_args()


def load_traditional_distances(data_dir, dataset_type, n_voters, n_items):
    """Load H1, H2, KwikSort, MC distances from traditional methods CSV."""
    csv_path = os.path.join(
        data_dir, 'traditional_methods_result', dataset_type,
        f'test_dataset_{dataset_type}_nvoters_{n_voters}_nitems_{n_items}_traditional.csv'
    )
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return {
        'Max Agreement': df['h1_distance'].values,
        'Min Regret': df['h2_distance'].values,
        'KwikSort': df['kwiksort_distance'].values,
        'Markov Chain': df['mc_distance'].values,
    }


def load_decor_distances(data_dir, dataset_type, n_voters, n_items):
    """Load DECOR distances from decor_result CSV."""
    csv_path = os.path.join(
        data_dir, f'test_dataset_{dataset_type}', 'decor_result',
        f'test_dataset_{dataset_type}_v{n_voters}_i{n_items}_decor_distances.csv'
    )
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df['kemeny_distance'].values


def load_transformer_distances(data_dir, dataset_type, n_voters, n_items, transformer_result_dir=None):
    """Load Kemeny Transformer distances from inference results."""
    csv_name = f'v{n_voters}_i{n_items}_with_kemeny.csv'

    # Build search paths for CSV results
    search_paths = []

    if transformer_result_dir:
        search_paths.append(os.path.join(transformer_result_dir, dataset_type, csv_name))

    # Default location: test_dataset/individual_configs_epoch_520/{type}/
    search_paths.append(os.path.join(data_dir, 'individual_configs_epoch_520', dataset_type, csv_name))

    # Other possible locations
    for subdir in ['test_results/individual_configs_epoch_520',
                    'test_results/700k_various_voters_various_items_epoch_500',
                    'test_results']:
        search_paths.append(os.path.join(subdir, dataset_type, csv_name))
        search_paths.append(os.path.join(data_dir, subdir, dataset_type, csv_name))

    for csv_path in search_paths:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'kemeny_distance' in df.columns:
                return df['kemeny_distance'].values

    # Fallback: try .npy format from inference.py output
    for subdir in ['transformer_test_result/test_data', 'transformer_test_result/test_data_fine_tuning',
                    'transformer_test_result/fine_tuning']:
        npy_path = os.path.join(
            data_dir, subdir,
            f'test_dataset_{dataset_type}_v{n_voters}_i{n_items}_transformer.npy'
        )
        if not os.path.exists(npy_path):
            npy_path = os.path.join(
                data_dir, subdir,
                f'test_dataset_{dataset_type}_transformer.npy'
            )
        if os.path.exists(npy_path):
            data = np.load(npy_path, allow_pickle=True)
            if hasattr(data, 'item'):
                data = data.item()
            for key in data:
                if 'greedy kemeny distance' in key and 'mean' not in key:
                    return data[key]

    return None


def plot_gap_from_best(dataset_type, voter_configs, item_configs, all_data, output_path):
    """Create normalized gap-from-best plot."""
    methods = ['Max Agreement', 'Min Regret', 'KwikSort', 'Markov Chain', 'DECOR', 'Kemeny Transformer']
    colors = {
        'Max Agreement': 'tab:orange',
        'Min Regret': 'tab:blue',
        'KwikSort': 'tab:green',
        'Markov Chain': 'tab:red',
        'DECOR': 'tab:purple',
        'Kemeny Transformer': 'brown',
    }
    markers = {
        'Max Agreement': 's',
        'Min Regret': 'o',
        'KwikSort': '^',
        'Markov Chain': 'D',
        'DECOR': '*',
        'Kemeny Transformer': 'P',
    }

    # Build x-axis labels and data
    x_labels = []
    method_gaps = {m: [] for m in methods}

    for n_items in item_configs:
        for n_voters in voter_configs:
            key = (n_voters, n_items)
            if key not in all_data:
                continue

            x_labels.append(f'({n_voters}, {n_items})')
            config_data = all_data[key]

            # Find the best (minimum) mean distance across all methods for this config
            means = {}
            for method in methods:
                if method in config_data and config_data[method] is not None:
                    means[method] = np.nanmean(config_data[method])

            if not means:
                for method in methods:
                    method_gaps[method].append(np.nan)
                continue

            best_mean = min(means.values())

            # Compute gap from best for each method
            for method in methods:
                if method in means:
                    method_gaps[method].append(means[method] - best_mean)
                else:
                    method_gaps[method].append(np.nan)

    if not x_labels:
        print(f"  No data to plot for {dataset_type}")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(x_labels))

    for method in methods:
        gaps = method_gaps[method]
        if all(np.isnan(g) for g in gaps):
            continue
        ax.plot(x_pos, gaps, marker=markers[method], color=colors[method],
                linewidth=2, markersize=8, label=method)

    # Shade alternating item groups
    items_per_group = len(voter_configs)
    for i in range(0, len(x_labels), items_per_group):
        group_idx = i // items_per_group
        if group_idx % 2 == 0:
            ax.axvspan(i - 0.5, min(i + items_per_group - 0.5, len(x_labels) - 0.5),
                       alpha=0.1, color='gray')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=9)
    ax.set_xlabel('(Voters, Items)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap from Best Method (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title(f'Normalized Gap from Best - {dataset_type.capitalize()}\n(0 = Winner)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=3, loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print("Generating gap-from-best comparison plots")
    print(f"{'='*60}")

    for dataset_type in args.data_types:
        print(f"\n--- {dataset_type.capitalize()} ---")

        all_data = {}
        for n_items in args.item_configs:
            for n_voters in args.voter_configs:
                config_data = {}

                # Load traditional methods
                trad = load_traditional_distances(args.data_dir, dataset_type, n_voters, n_items)
                if trad:
                    config_data.update(trad)
                    print(f"  ({n_voters}, {n_items}) traditional: "
                          f"H1={np.mean(trad['Max Agreement']):.1f}, "
                          f"H2={np.mean(trad['Min Regret']):.1f}, "
                          f"KS={np.mean(trad['KwikSort']):.1f}, "
                          f"MC={np.mean(trad['Markov Chain']):.1f}")

                # Load DECOR
                decor = load_decor_distances(args.data_dir, dataset_type, n_voters, n_items)
                if decor is not None:
                    config_data['DECOR'] = decor
                    print(f"  ({n_voters}, {n_items}) DECOR: {np.mean(decor):.1f}")

                # Load Transformer
                transformer = load_transformer_distances(
                    args.data_dir, dataset_type, n_voters, n_items,
                    transformer_result_dir=args.transformer_result_dir
                )
                if transformer is not None:
                    config_data['Kemeny Transformer'] = transformer
                    print(f"  ({n_voters}, {n_items}) Transformer: {np.mean(transformer):.1f}")

                if config_data:
                    all_data[(n_voters, n_items)] = config_data

        if not all_data:
            print(f"  No data found for {dataset_type}. Skipping.")
            continue

        items_str = '_'.join(str(i) for i in args.item_configs)
        output_path = os.path.join(
            args.output_dir,
            f'comparison_{dataset_type}_items_{items_str}.png'
        )
        plot_gap_from_best(dataset_type, args.voter_configs, args.item_configs, all_data, output_path)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
