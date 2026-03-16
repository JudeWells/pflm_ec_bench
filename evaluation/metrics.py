#!/usr/bin/env python3
"""Evaluation metrics for benchmark scoring."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils.io_utils import read_tsv, read_json


def compute_metrics(labels, scores):
    """Compute overall and per-bin metrics.

    Args:
        labels: list of dicts from labels.tsv
        scores: dict of {seq_id: float score} from model
    """
    # Merge
    true_labels = []
    pred_scores = []
    per_bin = defaultdict(lambda: {"labels": [], "scores": []})
    per_tier = defaultdict(lambda: {"labels": [], "scores": []})

    for row in labels:
        sid = row["seq_id"]
        if sid not in scores:
            continue
        y = int(row["label"])
        s = scores[sid]
        true_labels.append(y)
        pred_scores.append(s)
        per_bin[row["sim_bin"]]["labels"].append(y)
        per_bin[row["sim_bin"]]["scores"].append(s)
        per_tier[f"tier{row['tier']}"]["labels"].append(y)
        per_tier[f"tier{row['tier']}"]["scores"].append(s)

    if len(set(true_labels)) < 2:
        return None

    results = {
        "overall": {
            "auroc": roc_auc_score(true_labels, pred_scores),
            "auprc": average_precision_score(true_labels, pred_scores),
            "n_samples": len(true_labels),
        },
        "per_bin": {},
        "per_tier": {},
    }

    for bin_name, data in sorted(per_bin.items()):
        if len(set(data["labels"])) < 2:
            continue
        results["per_bin"][bin_name] = {
            "auroc": roc_auc_score(data["labels"], data["scores"]),
            "n_samples": len(data["labels"]),
        }

    for tier_name, data in sorted(per_tier.items()):
        if len(set(data["labels"])) < 2:
            continue
        results["per_tier"][tier_name] = {
            "auroc": roc_auc_score(data["labels"], data["scores"]),
            "n_samples": len(data["labels"]),
        }

    return results


def load_scores(scores_path):
    """Load model scores from TSV (seq_id, score)."""
    scores = {}
    rows = read_tsv(scores_path)
    for row in rows:
        scores[row["seq_id"]] = float(row["score"])
    return scores


def evaluate_instance(instance_dir, scores_path):
    """Evaluate a single benchmark instance."""
    labels = read_tsv(instance_dir / "labels.tsv")
    scores = load_scores(scores_path)
    return compute_metrics(labels, scores)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model scores")
    parser.add_argument("--benchmark-dir", required=True,
                        help="Path to benchmark directory")
    parser.add_argument("--scores-dir", required=True,
                        help="Directory containing score TSV files per instance")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    scores_dir = Path(args.scores_dir)
    manifest = read_json(benchmark_dir / "manifest.json")

    all_results = {}
    aurocs = []

    for entry in manifest["instances"]:
        ec = entry["ec_number"]
        ec_dir = ec.replace(".", "_")
        instance_dir = benchmark_dir / "instances" / ec_dir
        scores_path = scores_dir / ec_dir / "scores.tsv"

        if not scores_path.exists():
            continue

        result = evaluate_instance(instance_dir, scores_path)
        if result:
            all_results[ec] = result
            aurocs.append(result["overall"]["auroc"])

    if aurocs:
        print(f"Evaluated {len(aurocs)} instances")
        print(f"Mean AUROC: {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}")
        print(f"Median AUROC: {np.median(aurocs):.3f}")

    if args.output:
        from utils.io_utils import write_json
        write_json(all_results, args.output)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
