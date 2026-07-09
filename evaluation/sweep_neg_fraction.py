#!/usr/bin/env python3
"""Sweep over different numbers of training negatives for the GP alignment
scorer and plot mean AUROC vs n_train_neg.

The kernel is computed once per family; only the GP fit is repeated per sweep
point, so the total cost is approximately one kernel build per family.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scorers.gp_alignment_benchmark import FamilyData, _instance_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances-dir", default="data/benchmark/instances")
    parser.add_argument("--n-families", type=int, default=20)
    parser.add_argument("--n-train-pos", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--neg-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 8, 10, 15, 20, 30, 50],
        help="Absolute numbers of training negatives to sweep over",
    )
    parser.add_argument("--output-csv", default="evaluation/results/gp_neg_sweep.csv")
    parser.add_argument("--output-plot", default="evaluation/results/gp_neg_sweep.png")
    args = parser.parse_args()

    instances_dir = Path(args.instances_dir)
    selected_dirs = list(_instance_dirs(instances_dir))[: args.n_families]
    if not selected_dirs:
        raise FileNotFoundError(f"No instances in {instances_dir}")

    neg_counts = sorted(args.neg_counts)

    all_rows = []

    for idx, instance_dir in enumerate(selected_dirs, start=1):
        family_id = instance_dir.name
        print(f"[{idx}/{len(selected_dirs)}] {family_id}")
        try:
            fam = FamilyData(instance_dir, n_train_pos=args.n_train_pos, seed=args.seed)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        for n_neg in neg_counts:
            if n_neg > fam.n_total_neg:
                break  # skip counts larger than available negatives
            metrics = fam.score_with_n_neg(n_neg, seed=args.seed)
            if metrics is None:
                continue
            all_rows.append({
                "family_id": family_id,
                "n_train_neg_requested": n_neg,
                "n_train_neg_actual": metrics["n_train_neg"],
                "n_eval": metrics["n_samples"],
                "raw_auroc": metrics["raw_auroc"],
                "adjusted_auroc": metrics["adjusted_auroc"],
                "raw_paired_ranking_accuracy": metrics["raw_paired_ranking_accuracy"],
            })
        print(f"  done ({fam.n_total_neg} total negatives)")

    if not all_rows:
        print("No results.")
        return

    # Write CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_csv}")

    # ---- Aggregate and plot ------------------------------------------------
    # Group by requested n_train_neg
    count_to_raw = {}
    count_to_adj = {}
    for row in all_rows:
        k = row["n_train_neg_requested"]
        count_to_raw.setdefault(k, []).append(row["raw_auroc"])
        count_to_adj.setdefault(k, []).append(row["adjusted_auroc"])

    plot_n = []
    plot_raw_mean, plot_raw_se = [], []
    plot_adj_mean, plot_adj_se = [], []
    plot_n_families = []

    for k in sorted(count_to_raw.keys()):
        raw = np.array(count_to_raw[k])
        adj = np.array(count_to_adj[k])
        plot_n.append(k)
        plot_raw_mean.append(np.mean(raw))
        plot_raw_se.append(np.std(raw) / np.sqrt(len(raw)))
        plot_adj_mean.append(np.mean(adj))
        plot_adj_se.append(np.std(adj) / np.sqrt(len(adj)))
        plot_n_families.append(len(raw))

    plot_n = np.array(plot_n)
    plot_raw_mean = np.array(plot_raw_mean)
    plot_raw_se = np.array(plot_raw_se)
    plot_adj_mean = np.array(plot_adj_mean)
    plot_adj_se = np.array(plot_adj_se)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        plot_n, plot_raw_mean, yerr=plot_raw_se,
        fmt="o-", capsize=4, label="Raw AUROC",
    )
    ax.errorbar(
        plot_n, plot_adj_mean, yerr=plot_adj_se,
        fmt="s--", capsize=4, label="Similarity-adjusted AUROC",
    )

    # Annotate each point with number of families contributing
    for x, n_fam in zip(plot_n, plot_n_families):
        ax.annotate(
            f"n={n_fam}", (x, 0.05), fontsize=7, ha="center", color="gray",
        )

    ax.set_xlabel("Number of training negatives")
    ax.set_ylabel("AUROC (mean ± SE across families)")
    ax.set_title("GP alignment-kernel classifier: effect of training set size")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_plot = Path(args.output_plot)
    fig.savefig(out_plot, dpi=150)
    print(f"Saved plot to {out_plot}")


if __name__ == "__main__":
    main()
