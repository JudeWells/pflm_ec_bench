#!/usr/bin/env python3
"""Validate benchmark quality: no leakage, similarity matching, baseline performance."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from utils.io_utils import read_fasta, read_tsv, read_json


def check_no_leakage(instance_dir):
    """Verify no candidate appears in the conditioning set."""
    cond = read_fasta(instance_dir / "conditioning.fasta")
    cand = read_fasta(instance_dir / "candidates.fasta")
    overlap = set(cond.keys()) & set(cand.keys())
    return overlap


def check_similarity_matching(labels):
    """Verify positives and negatives have matched similarity distributions."""
    pos_sims = []
    neg_sims = []
    for row in labels:
        pid = float(row["max_pident_to_conditioning"])
        if int(row["label"]) == 1:
            pos_sims.append(pid)
        else:
            neg_sims.append(pid)

    if not pos_sims or not neg_sims:
        return None, None, None

    pos_mean = np.mean(pos_sims)
    neg_mean = np.mean(neg_sims)
    diff = abs(pos_mean - neg_mean)
    return pos_mean, neg_mean, diff


def nearest_neighbor_baseline(labels):
    """Compute AUROC using max similarity to conditioning set as the score.

    If this is high, the similarity matching has failed.
    """
    scores = []
    true_labels = []
    for row in labels:
        scores.append(float(row["max_pident_to_conditioning"]))
        true_labels.append(int(row["label"]))

    if len(set(true_labels)) < 2:
        return None

    return roc_auc_score(true_labels, scores)


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["output_dir"])
    benchmark_dir = data_dir / "benchmark"
    manifest = read_json(benchmark_dir / "manifest.json")

    print(f"Validating {manifest['n_instances']} benchmark instances...\n")

    issues = []
    baseline_aurocs = []
    sim_diffs = []
    all_n_pairs = []

    for entry in manifest["instances"]:
        ec = entry["ec_number"]
        ec_dir = ec.replace(".", "_")
        instance_dir = benchmark_dir / "instances" / ec_dir

        # Check existence
        if not instance_dir.exists():
            issues.append(f"{ec}: instance directory missing")
            continue

        # Check no leakage
        overlap = check_no_leakage(instance_dir)
        if overlap:
            issues.append(f"{ec}: {len(overlap)} sequences in both conditioning "
                         f"and candidate sets: {list(overlap)[:3]}")

        # Load labels
        labels = read_tsv(instance_dir / "labels.tsv")

        # Check label balance
        n_pos = sum(1 for r in labels if int(r["label"]) == 1)
        n_neg = sum(1 for r in labels if int(r["label"]) == 0)
        if n_pos != n_neg:
            issues.append(f"{ec}: label imbalance: {n_pos} pos, {n_neg} neg")

        all_n_pairs.append(n_pos)

        # Check similarity matching
        pos_mean, neg_mean, diff = check_similarity_matching(labels)
        if diff is not None:
            sim_diffs.append(diff)
            if diff > 10:
                issues.append(
                    f"{ec}: large similarity mismatch: "
                    f"pos_mean={pos_mean:.1f}, neg_mean={neg_mean:.1f}"
                )

        # Baseline AUROC
        auroc = nearest_neighbor_baseline(labels)
        if auroc is not None:
            baseline_aurocs.append(auroc)
            if auroc > 0.9:
                issues.append(
                    f"{ec}: nearest-neighbor baseline AUROC={auroc:.3f} "
                    f"(>0.9 suggests poor similarity matching)"
                )

    # Report
    print("=== Validation Results ===\n")

    if issues:
        print(f"ISSUES FOUND: {len(issues)}")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("No issues found.")

    print(f"\n=== Statistics ===")
    print(f"Total instances: {manifest['n_instances']}")
    print(f"Total candidate pairs: {manifest['total_pairs']:,}")

    if all_n_pairs:
        print(f"\nPairs per instance:")
        print(f"  Min: {min(all_n_pairs)}, Max: {max(all_n_pairs)}, "
              f"Median: {int(np.median(all_n_pairs))}, "
              f"Mean: {np.mean(all_n_pairs):.1f}")

    if sim_diffs:
        print(f"\nSimilarity matching (|pos_mean - neg_mean|):")
        print(f"  Mean diff: {np.mean(sim_diffs):.2f}%")
        print(f"  Max diff: {max(sim_diffs):.2f}%")
        print(f"  Instances with diff > 5%: "
              f"{sum(1 for d in sim_diffs if d > 5)}/{len(sim_diffs)}")

    if baseline_aurocs:
        print(f"\nNearest-neighbor baseline AUROC:")
        print(f"  Mean: {np.mean(baseline_aurocs):.3f}")
        print(f"  Median: {np.median(baseline_aurocs):.3f}")
        print(f"  Instances with AUROC > 0.7: "
              f"{sum(1 for a in baseline_aurocs if a > 0.7)}/{len(baseline_aurocs)}")
        print(f"  Instances with AUROC > 0.9: "
              f"{sum(1 for a in baseline_aurocs if a > 0.9)}/{len(baseline_aurocs)}")

    print(f"\nSkipped families: {manifest.get('skipped', {})}")

    # Tier distribution across all instances
    tier_totals = defaultdict(int)
    for entry in manifest["instances"]:
        for tier, count in entry.get("tier_distribution", {}).items():
            tier_totals[tier] += count

    if tier_totals:
        print(f"\nDecoy tier distribution (all instances):")
        for tier in sorted(tier_totals.keys()):
            print(f"  {tier}: {tier_totals[tier]:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
