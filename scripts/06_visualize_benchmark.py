#!/usr/bin/env python3
"""Visualise a constructed benchmark: similarity matching, gaps, tiers, baseline.

Produces a multi-panel dashboard plus a few standalone figures under
``<data_dir>/<subdir>/figures/``. When the original ``benchmark`` is also present
it adds a before/after panel showing the positive-minus-negative identity bias.

Usage:
    python scripts/06_visualize_benchmark.py --output-subdir benchmark_v2
"""
import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score


def load_pairs(subdir_path):
    """Return list of pair dicts and the flat per-candidate rows per instance.

    Each pair: pos_pid, neg_pid, sim_bin, tier.
    """
    pairs = []
    instance_rows = {}  # ec_dir -> list of (label, pid)
    for f in sorted(glob.glob(str(subdir_path / "instances" / "*" / "labels.tsv"))):
        rows = list(csv.DictReader(open(f), delimiter="\t"))
        by_id = {r["seq_id"]: r for r in rows}
        instance_rows[Path(f).parent.name] = [
            (int(r["label"]), float(r["max_pident_to_conditioning"])) for r in rows
        ]
        seen = set()
        for r in rows:
            if r["seq_id"] in seen:
                continue
            if r["label"] == "1":
                p, n = r, by_id[r["paired_with"]]
            else:
                n, p = r, by_id[r["paired_with"]]
            seen.add(p["seq_id"])
            seen.add(n["seq_id"])
            pairs.append({
                "pos_pid": float(p["max_pident_to_conditioning"]),
                "neg_pid": float(n["max_pident_to_conditioning"]),
                "sim_bin": p["sim_bin"],
                "tier": int(p["tier"]),
            })
    return pairs, instance_rows


def instance_baseline_aurocs(instance_rows):
    aurocs = []
    for rows in instance_rows.values():
        labels = [l for l, _ in rows]
        scores = [s for _, s in rows]
        if len(set(labels)) == 2:
            aurocs.append(roc_auc_score(labels, scores))
    return aurocs


# Consistent colours
C_POS = "#2c7fb8"
C_NEG = "#d95f0e"
C_ACC = "#31a354"


def main():
    ap = argparse.ArgumentParser(description="Visualise a benchmark")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output-subdir", default="benchmark_v2")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_dir = Path(config["data"]["output_dir"])
    sub = data_dir / args.output_subdir
    fig_dir = sub / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pairs, instance_rows = load_pairs(sub)
    pos = np.array([p["pos_pid"] for p in pairs])
    neg = np.array([p["neg_pid"] for p in pairs])
    gap = neg - pos
    tiers = np.array([p["tier"] for p in pairs])
    bins_lbl = [p["sim_bin"] for p in pairs]
    aurocs = instance_baseline_aurocs(instance_rows)
    n_inst = len(instance_rows)

    print(f"{args.output_subdir}: {n_inst} instances, {len(pairs)} pairs")

    # Optional comparison with the original closest-in-bin benchmark
    orig = data_dir / "benchmark"
    orig_gap = None
    if orig.exists() and orig.resolve() != sub.resolve():
        opairs, _ = load_pairs(orig)
        if opairs:
            orig_gap = np.array([p["neg_pid"] - p["pos_pid"] for p in opairs])

    # ---------------- Dashboard ----------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Benchmark '{args.output_subdir}': {n_inst} instances, {len(pairs)} pairs",
        fontsize=15, fontweight="bold",
    )

    # (1) Overlaid identity distributions
    ax = axes[0, 0]
    edges = np.arange(20, 102, 2)
    ax.hist(pos, bins=edges, alpha=0.6, label="Positive (in-family)", color=C_POS)
    ax.hist(neg, bins=edges, alpha=0.6, label="Negative (out-of-family)", color=C_NEG)
    ax.set_xlabel("Max % identity to conditioning set")
    ax.set_ylabel("Candidates")
    ax.set_title("Identity distributions are matched")
    ax.legend()

    # (2) Per-pair gap histogram
    ax = axes[0, 1]
    ax.hist(gap, bins=np.arange(0, 10.5, 0.5), color=C_ACC, alpha=0.85)
    ax.axvline(np.median(gap), color="k", ls="--",
               label=f"median = {np.median(gap):.2f} pp")
    ax.set_xlabel("Negative − Positive identity (pp)")
    ax.set_ylabel("Pairs")
    ax.set_title("Every negative is slightly MORE similar")
    ax.legend()

    # (3) Scatter pos vs neg (subsample for clarity)
    ax = axes[0, 2]
    rng = np.random.default_rng(0)
    idx = np.arange(len(pos))
    if len(idx) > 3000:
        idx = rng.choice(idx, 3000, replace=False)
    tier_colors = {1: "#762a83", 2: "#1b7837", 3: "#e08214", 4: "#999999"}
    for t in [4, 3, 2, 1]:
        m = idx[tiers[idx] == t]
        ax.scatter(pos[m], neg[m], s=8, alpha=0.5,
                   color=tier_colors[t], label=f"Tier {t}")
    lim = [min(pos.min(), neg.min()) - 2, max(pos.max(), neg.max()) + 2]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Positive identity (%)")
    ax.set_ylabel("Negative identity (%)")
    ax.set_title("Negatives sit just above the diagonal")
    ax.legend(markerscale=2, fontsize=8)

    # (4) Pairs per similarity bin
    ax = axes[1, 0]
    bin_order = [f"{b}-{b+10}" for b in range(20, 100, 10)]
    counts = defaultdict(int)
    for b in bins_lbl:
        counts[b] += 1
    vals = [counts.get(b, 0) for b in bin_order]
    ax.bar(bin_order, vals, color=C_POS, alpha=0.85)
    ax.set_xlabel("Similarity bin (positive % identity)")
    ax.set_ylabel("Pairs")
    ax.set_title("Pairs per similarity bin")
    ax.tick_params(axis="x", rotation=45)

    # (5) Tier distribution
    ax = axes[1, 1]
    tcounts = [int((tiers == t).sum()) for t in [1, 2, 3, 4]]
    tlabels = ["1\nsame EC3", "2\nsame EC2", "3\nsame EC1", "4\ndiff EC1"]
    ax.bar(tlabels, tcounts, color=[tier_colors[t] for t in [1, 2, 3, 4]], alpha=0.85)
    for i, v in enumerate(tcounts):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Pairs")
    ax.set_title("Decoy tier distribution (harder = lower)")

    # (6) NN baseline AUROC per instance
    ax = axes[1, 2]
    ax.hist(aurocs, bins=np.arange(0, 1.02, 0.05), color="#756bb1", alpha=0.85)
    ax.axvline(0.5, color="k", ls="--", label="chance = 0.5")
    ax.axvline(np.mean(aurocs), color=C_NEG, ls="-",
               label=f"mean = {np.mean(aurocs):.2f}")
    ax.set_xlabel("Nearest-neighbour baseline AUROC")
    ax.set_ylabel("Instances")
    ax.set_title("Homology baseline is at/below chance")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    dash = fig_dir / "dashboard.png"
    fig.savefig(dash, dpi=150)
    plt.close(fig)

    # ---------------- Standalone: bias before/after ----------------
    if orig_gap is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        rng_edges = np.arange(-12, 12.5, 1.0)
        ax.hist(-orig_gap, bins=rng_edges, alpha=0.6,
                label=f"Original (closest-in-bin)\nmean pos−neg = {(-orig_gap).mean():+.2f} pp",
                color=C_NEG)
        ax.hist(-gap, bins=rng_edges, alpha=0.6,
                label=f"v2 (negative-higher)\nmean pos−neg = {(-gap).mean():+.2f} pp",
                color=C_POS)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("Positive − Negative identity (pp)   "
                      "(> 0 favours a homology shortcut)")
        ax.set_ylabel("Pairs")
        ax.set_title("Removing the pro-homology bias")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "bias_before_after.png", dpi=150)
        plt.close(fig)

    # ---------------- Standalone: gap ECDF ----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sg = np.sort(gap)
    ecdf = np.arange(1, len(sg) + 1) / len(sg)
    ax.plot(sg, ecdf, color=C_ACC, lw=2)
    for q in (2, 5):
        frac = (gap <= q).mean()
        ax.axvline(q, color="grey", ls=":", lw=1)
        ax.text(q, 0.05, f"{frac*100:.0f}% ≤ {q}pp", rotation=90,
                va="bottom", ha="right", fontsize=9)
    ax.set_xlabel("Negative − Positive identity gap (pp)")
    ax.set_ylabel("Cumulative fraction of pairs")
    ax.set_title("How close are the matched pairs?")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(fig_dir / "gap_ecdf.png", dpi=150)
    plt.close(fig)

    print(f"Figures written to {fig_dir}/")
    for p in sorted(fig_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
