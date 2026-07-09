#!/usr/bin/env python3
"""Score benchmark families with a 1-NN classifier using pairwise alignment
similarity. This serves as a baseline to check whether the GP alignment-kernel
classifier is doing anything beyond nearest-neighbour lookup.

For each candidate, the score is the alignment similarity to the nearest
positive training example minus the similarity to the nearest negative
training example. This gives a continuous score suitable for AUROC.
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gp_alignment_benchmark import (
    FamilyData,
    _instance_dirs,
    _read_labels,
    compute_metrics,
)


def knn_score_family(fam: FamilyData, n_train_neg: int, seed: int = 42) -> Optional[Dict]:
    """Score candidates with 1-NN using the precomputed alignment kernel.

    Score = max similarity to any positive training seq
          − max similarity to any negative training seq.
    """
    rng = np.random.RandomState(seed)

    # Cap negatives to keep ≥2 pos + ≥2 neg for eval
    max_neg = min(fam.n_total_neg - 2, fam.n_total_pos - 2)
    actual_n_neg = min(n_train_neg, max(0, max_neg))

    if actual_n_neg > 0:
        idx = rng.choice(len(fam.neg_cands), actual_n_neg, replace=False)
        train_neg = [fam.neg_cands[i] for i in idx]
    else:
        train_neg = []

    if not train_neg:
        return None

    # Indices into gram matrix
    pos_idx = np.array([fam.id_to_idx[sid] for sid, _ in fam.train_pos])
    neg_idx = np.array([fam.id_to_idx[sid] for sid, _ in train_neg])
    cand_idx = np.array([fam.id_to_idx[sid] for sid, _ in fam.cand_pairs])

    # For each candidate: max kernel similarity to pos training, to neg training
    sim_to_pos = fam.gram[np.ix_(cand_idx, pos_idx)].max(axis=1)  # (n_cand,)
    sim_to_neg = fam.gram[np.ix_(cand_idx, neg_idx)].max(axis=1)  # (n_cand,)

    raw_scores = sim_to_pos - sim_to_neg  # positive → more like positives

    all_scores = {sid: float(s) for (sid, _), s in zip(fam.cand_pairs, raw_scores)}

    # Exclude training negatives and their paired positives from eval
    train_neg_ids = {sid for sid, _ in train_neg}
    excluded = set(train_neg_ids)
    for sid in train_neg_ids:
        partner = fam.pair_map.get(sid)
        if partner:
            excluded.add(partner)

    eval_scores = {sid: s for sid, s in all_scores.items() if sid not in excluded}

    try:
        metrics = compute_metrics(fam.labels, eval_scores)
    except ValueError:
        return None

    metrics["n_train_pos"] = len(fam.train_pos)
    metrics["n_train_neg"] = actual_n_neg
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run 1-NN alignment-similarity baseline on benchmark families"
    )
    parser.add_argument("--instances-dir", default="data/benchmark/instances")
    parser.add_argument("--n-families", type=int, default=20)
    parser.add_argument("--n-train-pos", type=int, default=10)
    parser.add_argument("--neg-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv", default="evaluation/results/knn_alignment_metrics.csv",
    )
    args = parser.parse_args()

    instances_dir = Path(args.instances_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    selected_dirs = list(_instance_dirs(instances_dir))[: args.n_families]
    if not selected_dirs:
        raise FileNotFoundError(f"No valid benchmark instances in {instances_dir}")

    rows = []
    print(f"Scoring {len(selected_dirs)} families with 1-NN alignment baseline "
          f"(neg_fraction={args.neg_fraction})...")

    for idx, instance_dir in enumerate(selected_dirs, start=1):
        family_id = instance_dir.name
        print(f"  [{idx}/{len(selected_dirs)}] {family_id}")
        try:
            fam = FamilyData(instance_dir, n_train_pos=args.n_train_pos, seed=args.seed)
            n_neg = max(1, int(round(fam.n_total_neg * args.neg_fraction)))
            metrics = knn_score_family(fam, n_neg, seed=args.seed)
            if metrics is None:
                print("    SKIPPED: not enough data")
                continue
            rows.append({"family_id": family_id, **metrics})
            print(f"    n_train_neg={metrics['n_train_neg']}  "
                  f"raw_auroc={metrics['raw_auroc']:.3f}  "
                  f"adjusted_auroc={metrics['adjusted_auroc']:.3f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

    if not rows:
        print("No families scored successfully.")
        return

    mean_row = {"family_id": "MEAN"}
    metric_keys = [k for k in rows[0] if k != "family_id"]
    for key in metric_keys:
        vals = [float(r[key]) for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
        mean_row[key] = float(np.mean(vals)) if vals else float("nan")
    rows.append(mean_row)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows) - 1} family rows + mean row to {output_csv}")


if __name__ == "__main__":
    main()
