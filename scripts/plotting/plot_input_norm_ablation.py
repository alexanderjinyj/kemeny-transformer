#!/usr/bin/env python3
"""
Input-normalization ablation plots (epoch 900), in the same style as
comparison_jiggling.png: "Normalized Gap from Best" over voters {6,8,10} x
items {90,100,110}, for jiggling / random / repeat.

Compared methods: Max Agreement, Min Repeat, KwikSort, Markov Chain, DECOR,
and the two Kemeny Transformer variants that differ only in input normalization
(with_input_norm vs without_input_norm).

Reads the transformer kemeny distances produced by run_input_norm_ablation.py.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

matplotlib.rc("font", size=22)
plt.rcParams["figure.figsize"] = (14, 8)

OUT_ROOT = Path("results/input_norm_ablation_epoch_900")
VOTERS = [6, 8, 10]
ITEMS = [100, 125, 150, 200]
DATASET_TYPES = ["jiggling", "random", "repeat"]


def load_traditional(dtype, v, i):
    p = Path(f"results/traditional_methods/{dtype}/"
             f"test_dataset_{dtype}_nvoters_{v}_nitems_{i}_traditional.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return {"max_agreement": df["h1_distance"].mean(),
            "min_repeat": df["h2_distance"].mean(),
            "kwiksort": df["kwiksort_distance"].mean(),
            "markov_chain": df["mc_distance"].mean()}


def load_decor(dtype, v, i):
    p = Path(f"results/decor/{dtype}/"
             f"test_dataset_{dtype}_v{v}_i{i}_decor_distances.csv")
    if not p.exists():
        return None
    return pd.read_csv(p)["kemeny_distance"].mean()


def load_transformer(variant, dtype, v, i):
    p = (OUT_ROOT / variant / f"test_dataset_{dtype}" / "csv_results" /
         f"test_dataset_{dtype}_nvoters_{v}_nitems_{i}_kemeny_distances.csv")
    if not p.exists():
        return None
    return pd.read_csv(p)["kemeny_distance"].mean()


def load_all(dtype):
    results = {}
    for v in VOTERS:
        for i in ITEMS:
            trad = load_traditional(dtype, v, i)
            results[f"({v}, {i})"] = {
                "max_agreement": trad["max_agreement"] if trad else None,
                "min_repeat": trad["min_repeat"] if trad else None,
                "kwiksort": trad["kwiksort"] if trad else None,
                "markov_chain": trad["markov_chain"] if trad else None,
                "decor": load_decor(dtype, v, i),
                "transformer_with_norm": load_transformer("with_input_norm", dtype, v, i),
                "transformer_without_norm": load_transformer("without_input_norm", dtype, v, i),
                "n_voters": v, "n_items": i,
            }
    return results


METHODS = {
    "max_agreement": {"color": "#ff7f0e", "marker": "s", "label": "Max Agreement", "linewidth": 2},
    "min_repeat": {"color": "#1f77b4", "marker": "o", "label": "Min Repeat", "linewidth": 2},
    "kwiksort": {"color": "#2ca02c", "marker": "^", "label": "KwikSort", "linewidth": 2},
    "markov_chain": {"color": "#d62728", "marker": "D", "label": "Markov Chain", "linewidth": 2},
    "decor": {"color": "#9467bd", "marker": "v", "label": "DECOR", "linewidth": 2},
    "transformer_with_norm": {"color": "#8c564b", "marker": "*", "label": "Transformer (input norm)",
                              "linewidth": 3, "markersize": 14},
    "transformer_without_norm": {"color": "#e377c2", "marker": "P", "label": "Transformer (no input norm)",
                                 "linewidth": 3, "markersize": 12},
}


def normalized_gaps(results):
    norm = {}
    for cfg, methods in results.items():
        vals = {k: v for k, v in methods.items()
                if k not in ("n_voters", "n_items") and v is not None}
        mn = min(vals.values()) if vals else 0.0
        norm[cfg] = {k: (v - mn if (k not in ("n_voters", "n_items") and v is not None)
                         else v) for k, v in methods.items()}
    return norm


def plot(dtype, results, out_file):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    norm = normalized_gaps(results)
    cfgs = sorted(norm.keys(), key=lambda c: (results[c]["n_items"], results[c]["n_voters"]))
    x = np.arange(len(cfgs))
    labels = [f"({results[c]['n_voters']}, {results[c]['n_items']})" for c in cfgs]

    for m, style in METHODS.items():
        y = [norm[c][m] for c in cfgs]
        if any(v is not None for v in y):
            y = [v if v is not None else np.nan for v in y]
            ax.plot(x, y, marker=style["marker"], color=style["color"], label=style["label"],
                    linewidth=style["linewidth"], markersize=style.get("markersize", 8))

    plt.axvspan(-0.1, 2.1, color="grey", alpha=0.2)
    ax.set_xlabel("(Voters, Items)", fontweight="bold")
    ax.set_ylabel("Gap from Best Method (Lower is Better)", fontweight="bold")
    ax.set_title(f"Normalized Gap from Best - {dtype.capitalize()}\n"
                 f"Input-Norm Ablation (0 = Winner)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(ncol=2, loc="best", fontsize=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")


def plot_absolute(dtype, results, out_file):
    """Absolute mean Kemeny distance for DECOR + the two transformer variants only."""
    only = ["decor", "transformer_with_norm", "transformer_without_norm"]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    cfgs = sorted(results.keys(), key=lambda c: (results[c]["n_items"], results[c]["n_voters"]))
    x = np.arange(len(cfgs))
    labels = [f"({results[c]['n_voters']}, {results[c]['n_items']})" for c in cfgs]

    for m in only:
        style = METHODS[m]
        y = [results[c][m] for c in cfgs]
        if any(v is not None for v in y):
            y = [np.log(v) if v is not None else np.nan for v in y]
            ax.plot(x, y, marker=style["marker"], color=style["color"], label=style["label"],
                    linewidth=style["linewidth"], markersize=style.get("markersize", 8))

    plt.axvspan(-0.1, 2.1, color="grey", alpha=0.2)
    ax.set_xlabel("(Voters, Items)", fontweight="bold")
    ax.set_ylabel("log(Average Kemeny Distance) (Lower is Better)", fontweight="bold")
    ax.set_title(f"log(Average Kemeny Distance) - {dtype.capitalize()}\n"
                 f"Input-Norm Ablation vs DECOR", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(loc="best", fontsize=16)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")


def print_table(dtype, results):
    cfgs = sorted(results.keys(), key=lambda c: (results[c]["n_voters"], results[c]["n_items"]))
    print(f"\n=== {dtype.upper()} - mean Kemeny distance ===")
    print(f"{'Config':<11}{'MaxAgr':>9}{'MinRep':>9}{'KwikS':>9}{'Markov':>9}"
          f"{'DECOR':>9}{'TF+norm':>10}{'TF-norm':>10}")
    def f(x, w=9):
        return f"{x:>{w}.1f}" if x is not None else f"{'NA':>{w}}"
    for c in cfgs:
        r = results[c]
        print(f"{c:<11}{f(r['max_agreement'])}{f(r['min_repeat'])}{f(r['kwiksort'])}"
              f"{f(r['markov_chain'])}{f(r['decor'])}"
              f"{f(r['transformer_with_norm'], 10)}{f(r['transformer_without_norm'], 10)}")


DTYPE_COLORS = {"jiggling": "#1f77b4", "random": "#2ca02c", "repeat": "#d62728"}


def plot_combined_gap(all_results, out_file):
    """All three datasets on one figure: gap-from-best for the two transformer
    variants. Color = dataset, solid = no input norm, dashed = input norm."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    any_dtype = next(iter(all_results))
    cfgs = sorted(all_results[any_dtype].keys(),
                  key=lambda c: (all_results[any_dtype][c]["n_items"],
                                 all_results[any_dtype][c]["n_voters"]))
    x = np.arange(len(cfgs))
    labels = [f"({all_results[any_dtype][c]['n_voters']}, "
              f"{all_results[any_dtype][c]['n_items']})" for c in cfgs]

    for dtype in DATASET_TYPES:
        norm = normalized_gaps(all_results[dtype])
        color = DTYPE_COLORS[dtype]
        ax.plot(x, [norm[c]["transformer_without_norm"] for c in cfgs],
                marker="P", color=color, linewidth=3, markersize=11,
                linestyle="-", label=f"{dtype.capitalize()} - no input norm")
        ax.plot(x, [norm[c]["transformer_with_norm"] for c in cfgs],
                marker="*", color=color, linewidth=2, markersize=12,
                linestyle="--", label=f"{dtype.capitalize()} - input norm")

    plt.axvspan(-0.1, 2.1, color="grey", alpha=0.15)
    ax.set_xlabel("(Voters, Items)", fontweight="bold")
    ax.set_ylabel("Gap from Best Method (Lower is Better)", fontweight="bold")
    ax.set_title("Transformer Gap from Best - Input-Norm Ablation\n"
                 "Jiggling / Random / Repeat", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(ncol=3, loc="best", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")


def plot_gap_mean_std_per_config(all_results, out_file, csv_file):
    """7 lines (methods) across 9 configs. At each config, each method's gap-to-best
    is collected over the 3 datasets (jiggling/random/repeat); y = mean, error bar = std."""
    any_dtype = next(iter(all_results))
    cfgs = sorted(all_results[any_dtype].keys(),
                  key=lambda c: (all_results[any_dtype][c]["n_items"],
                                 all_results[any_dtype][c]["n_voters"]))
    x = np.arange(len(cfgs))
    labels = [f"({all_results[any_dtype][c]['n_voters']}, "
              f"{all_results[any_dtype][c]['n_items']})" for c in cfgs]
    # Fold in Borda / Footrule (from results/borda_footrule/) so they
    # compete for the per-config "best", without touching the other plots.
    def load_bf(dtype, v, i):
        p = Path(f"results/borda_footrule/{dtype}/"
                 f"test_dataset_{dtype}_nvoters_{v}_nitems_{i}_borda_footrule.csv")
        if not p.exists():
            return None, None
        d = pd.read_csv(p)
        return d["borda_kemeny"].mean(), d["footrule_kemeny"].mean()

    aug = {}
    for d in DATASET_TYPES:
        aug[d] = {}
        for c, r in all_results[d].items():
            rr = dict(r)
            rr["borda"], rr["footrule"] = load_bf(d, r["n_voters"], r["n_items"])
            aug[d][c] = rr
    norms = {d: normalized_gaps(aug[d]) for d in DATASET_TYPES}

    extra_styles = {
        "borda": {"color": "#17becf", "marker": "X", "label": "Borda", "linewidth": 2},
        "footrule": {"color": "#bcbd22", "marker": "p", "label": "Footrule", "linewidth": 2},
    }
    style_map = {**METHODS, **extra_styles}
    only = ["markov_chain", "kwiksort", "min_repeat", "borda", "footrule", "decor",
            "transformer_with_norm", "transformer_without_norm"]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    csv_rows = []
    for m in only:
        style = style_map[m]
        means = []
        for c in cfgs:
            vals = [norms[d][c][m] for d in DATASET_TYPES if norms[d][c][m] is not None]
            means.append(np.mean(vals) if vals else np.nan)
            csv_rows.append({"config": c, "method": style["label"], "mean_gap": means[-1]})
        ax.plot(x, means, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=style["linewidth"],
                markersize=style.get("markersize", 8))

    plt.axvspan(-0.1, 2.1, color="grey", alpha=0.15)
    ax.set_xlabel("(Voters, Items)", fontweight="bold")
    ax.set_ylabel("Gap from Best (lower is better)", fontweight="bold")
    ax.set_title("Normalized Gap from Best (0 = Winner)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(ncol=2, loc="best", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")
    pd.DataFrame(csv_rows).to_csv(csv_file, index=False)
    print(f"Saved {csv_file}")


def plot_ablation_emphasis(all_results, out_file, csv_file):
    """Clean ablation image: input-norm TF vs no-input-norm TF, with DECOR and
    Borda as baselines. The no-input-norm line is emphasized (bold red) to make
    its bad performance at the hard (few voters, many items) configs stand out.
    Gap-from-best is averaged over the 3 datasets per config."""
    any_d = next(iter(all_results))
    cfgs = sorted(all_results[any_d].keys(),
                  key=lambda c: (all_results[any_d][c]["n_items"],
                                 all_results[any_d][c]["n_voters"]))
    x = np.arange(len(cfgs))
    labels = [f"({all_results[any_d][c]['n_voters']}, {all_results[any_d][c]['n_items']})"
              for c in cfgs]

    def load_borda(dtype, v, i):
        p = Path(f"results/borda_footrule/{dtype}/"
                 f"test_dataset_{dtype}_nvoters_{v}_nitems_{i}_borda_footrule.csv")
        return pd.read_csv(p)["borda_kemeny"].mean() if p.exists() else None

    aug = {}
    for d in DATASET_TYPES:
        aug[d] = {}
        for c, r in all_results[d].items():
            rr = dict(r)
            rr["borda"] = load_borda(d, r["n_voters"], r["n_items"])
            aug[d][c] = rr
    norms = {d: normalized_gaps(aug[d]) for d in DATASET_TYPES}

    # Baselines muted (thin, dashed, grey-ish); the two transformers prominent;
    # the no-input-norm transformer is the bold red focus line.
    styles = [
        ("borda", {"color": "#9e9e9e", "marker": "X", "label": "Borda", "lw": 1.8, "ls": "--", "ms": 7, "z": 1}),
        ("decor", {"color": "#7e57c2", "marker": "v", "label": "DECOR", "lw": 1.8, "ls": "--", "ms": 7, "z": 1}),
        ("transformer_with_norm", {"color": "#2e7d32", "marker": "*", "label": "Transformer (input norm)", "lw": 3, "ls": "-", "ms": 16, "z": 3}),
        ("transformer_without_norm", {"color": "#d62728", "marker": "P", "label": "Transformer (NO input norm)", "lw": 4.5, "ls": "-", "ms": 14, "z": 4}),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    rows = []
    all_means = []
    for m, s in styles:
        means = [np.nanmean([norms[d][c][m] for d in DATASET_TYPES
                             if norms[d][c].get(m) is not None]) for c in cfgs]
        all_means += means
        ax.plot(x, means, marker=s["marker"], color=s["color"], label=s["label"],
                linewidth=s["lw"], linestyle=s["ls"], markersize=s["ms"], zorder=s["z"])
        for c, mv in zip(cfgs, means):
            rows.append({"config": c, "method": s["label"], "mean_gap": mv})

    items = [all_results[any_d][c]["n_items"] for c in cfgs]
    train_idx = [i for i, it in enumerate(items) if it == 100]   # in-distribution (trained on 90-110)
    bad_idx = [i for i, it in enumerate(items) if it in (150, 200)]  # far out-of-distribution
    top = max(all_means) * 1.18
    ax.set_ylim(-5, top)

    # Gray band = training item range (in-distribution); red band = the 150- and
    # 200-item configs where the no-input-norm transformer degrades / collapses.
    if train_idx:
        ax.axvspan(min(train_idx) - 0.5, max(train_idx) + 0.5, color="grey", alpha=0.18, zorder=0)
        ax.text((min(train_idx) + max(train_idx)) / 2, top * 0.96, "training\nitem range",
                ha="center", va="top", fontsize=14, fontweight="bold", color="#555555")
    if bad_idx:
        ax.axvspan(min(bad_idx) - 0.5, max(bad_idx) + 0.5, color="#d62728", alpha=0.10, zorder=0)
        ax.text((min(bad_idx) + max(bad_idx)) / 2, top * 0.96,
                "no-input-norm collapses\n(150 & 200 items)",
                ha="center", va="top", fontsize=14, fontweight="bold", color="#d62728")

    ax.set_xlabel("(Voters, Items)", fontweight="bold")
    ax.set_ylabel("Gap from Best (lower is better)", fontweight="bold")
    ax.set_title("Input-Normalization Ablation\nGap from Best Method (0 = Winner)",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    # Legend at mid-top, shifted slightly left of center so it stays clear of the
    # red-zone "no-input-norm degrades" text at the top-right.
    ax.legend(loc="upper center", bbox_to_anchor=(0.40, 0.99), fontsize=14, framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")
    pd.DataFrame(rows).to_csv(csv_file, index=False)


def plot_method_gap_summary(all_results, out_file, csv_file):
    """For each method, pool its gap-to-best over all 3 datasets x all configs,
    then bar-plot mean +/- std."""
    pooled = {m: [] for m in METHODS}
    for dtype in DATASET_TYPES:
        norm = normalized_gaps(all_results[dtype])
        for c in norm:
            for m in METHODS:
                v = norm[c][m]
                if v is not None:
                    pooled[m].append(v)

    order = list(METHODS.keys())
    means = [np.mean(pooled[m]) for m in order]
    stds = [np.std(pooled[m]) for m in order]
    colors = [METHODS[m]["color"] for m in order]
    names = [METHODS[m]["label"] for m in order]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    x = np.arange(len(order))
    ax.bar(x, means, yerr=stds, color=colors, capsize=6,
           error_kw={"elinewidth": 2})
    for xi, mn in zip(x, means):
        ax.text(xi, mn, f"{mn:.1f}", ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_ylabel("Gap from Best Method (Lower is Better)", fontweight="bold")
    ax.set_title("Mean +/- Std Gap from Best\n"
                 "pooled over Jiggling / Random / Repeat (all configs)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")

    pd.DataFrame({"method": names, "mean_gap": means, "std_gap": stds,
                  "n_values": [len(pooled[m]) for m in order]}).to_csv(csv_file, index=False)
    print(f"Saved {csv_file}")


def main():
    rows = []
    all_results = {}
    for dtype in DATASET_TYPES:
        results = load_all(dtype)
        all_results[dtype] = results
        print_table(dtype, results)
        plot(dtype, results, OUT_ROOT / f"comparison_{dtype}.png")
        plot_absolute(dtype, results, OUT_ROOT / f"comparison_{dtype}_absolute.png")
        for c, r in results.items():
            row = {"dataset": dtype, "n_voters": r["n_voters"], "n_items": r["n_items"]}
            row.update({k: r[k] for k in METHODS})
            rows.append(row)
    plot_combined_gap(all_results, OUT_ROOT / "comparison_all_datasets_gap.png")
    plot_method_gap_summary(all_results, OUT_ROOT / "comparison_method_gap_mean_std.png",
                            OUT_ROOT / "method_gap_mean_std.csv")
    plot_gap_mean_std_per_config(all_results,
                                 OUT_ROOT / "comparison_gap_mean_std_per_config_1.png",
                                 OUT_ROOT / "gap_mean_std_per_config_1.csv")
    plot_ablation_emphasis(all_results,
                           OUT_ROOT / "ablation_emphasis.png",
                           OUT_ROOT / "ablation_emphasis.csv")
    pd.DataFrame(rows).to_csv(OUT_ROOT / "comparison_table.csv", index=False)
    print(f"\nSaved table -> {OUT_ROOT / 'comparison_table.csv'}")


if __name__ == "__main__":
    main()
