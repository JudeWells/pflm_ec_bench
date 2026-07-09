#!/usr/bin/env python3
"""Score benchmark families with ProFam and write CSV metrics."""
import argparse
import csv
import importlib
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from Bio import SeqIO
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from profam import ProFam



def _parse_seq_id(raw_id: str) -> str:
    """Extract accession from FASTA ID if needed."""
    if "|" in raw_id:
        parts = raw_id.split("|")
        if len(parts) >= 2:
            return parts[1]
    return raw_id


def _read_fasta_pairs(path: Path) -> List[Tuple[str, str]]:
    """Read FASTA and return (seq_id, sequence) pairs."""
    pairs: List[Tuple[str, str]] = []
    for rec in SeqIO.parse(path, "fasta"):
        pairs.append((_parse_seq_id(rec.id), str(rec.seq)))
    return pairs


def _build_prompt(conditioning_seqs: Sequence[str], max_residues: int) -> List[str]:
    """Build a prompt list with total length <= max_residues."""
    prompt: List[str] = []
    residues = 0

    for seq in conditioning_seqs:
        seq_len = len(seq)
        if residues + seq_len > max_residues:
            if not prompt:
                # Keep a non-empty prompt while respecting the residue cap.
                prompt.append(seq[:max_residues])
            break
        prompt.append(seq)
        residues += seq_len

    return prompt



def score(
    model,
    conditioning_fasta: Path,
    candidates_fasta: Path,
    max_prompt_residues: int = 6000,
) -> Tuple[Dict[str, float], int, int]:
    """Score candidates conditioned on capped-length prompt sequences.

    Returns:
        (scores_by_seq_id, n_prompt_sequences, n_prompt_residues)
    """
    cond_pairs = _read_fasta_pairs(Path(conditioning_fasta))
    cand_pairs = _read_fasta_pairs(Path(candidates_fasta))

    prompt = _build_prompt(
        conditioning_seqs=[seq for _, seq in cond_pairs],
        max_residues=max_prompt_residues,
    )
    prompt_residues = sum(len(s) for s in prompt)

    candidate_ids = [sid for sid, _ in cand_pairs]
    candidate_seqs = [seq for _, seq in cand_pairs]

    result = model.score(sequences=candidate_seqs, prompt=prompt, use_diversity_weights=False, ensemble_size=1)
    score_values = result.scores
    return dict(zip(candidate_ids, score_values)), len(prompt), prompt_residues


def _read_labels(labels_tsv: Path) -> List[Dict[str, str]]:
    with open(labels_tsv, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _paired_ranking_accuracy(labels: List[Dict[str, str]], scores: Dict[str, float]) -> float:
    """How often positive score > paired negative score."""
    seq_to_label = {row["seq_id"]: int(row["label"]) for row in labels}
    seq_to_pair = {row["seq_id"]: row["paired_with"] for row in labels}

    seen = set()
    wins = 0.0
    total = 0
    for seq_id, label in seq_to_label.items():
        if label != 1:
            continue
        pair_id = seq_to_pair.get(seq_id)
        if pair_id is None:
            continue
        pair_key = tuple(sorted((seq_id, pair_id)))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        if seq_id not in scores or pair_id not in scores:
            continue
        total += 1
        if scores[seq_id] > scores[pair_id]:
            wins += 1.0
        elif scores[seq_id] == scores[pair_id]:
            wins += 0.5
    return (wins / total) if total else float("nan")


def _best_threshold_accuracy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Best in-sample thresholded accuracy via Youden's J."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    best_thr = thresholds[best_idx]
    y_pred = (y_score >= best_thr).astype(int)
    return float(np.mean(y_pred == y_true))


def _binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: List[Dict[str, str]],
    scores_by_seq_id: Dict[str, float],
) -> Dict[str, float]:
    """Compute binary-classification metrics for one score vector."""
    if len(set(y_true.tolist())) < 2:
        raise ValueError("Need both positive and negative labels to compute metrics.")

    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "paired_ranking_accuracy": float(_paired_ranking_accuracy(labels, scores_by_seq_id)),
        "best_threshold_accuracy": _best_threshold_accuracy(y_true, y_score),
        "mean_pos_score": float(np.mean(y_score[y_true == 1])),
        "mean_neg_score": float(np.mean(y_score[y_true == 0])),
    }


def _fit_similarity_adjustment(
    raw_scores: np.ndarray, max_pident: np.ndarray
) -> Tuple[np.ndarray, float, float, float]:
    """Remove linear score component explained by max prompt similarity."""
    design = np.column_stack([np.ones_like(max_pident), max_pident])
    coeffs, *_ = np.linalg.lstsq(design, raw_scores, rcond=None)
    intercept = float(coeffs[0])
    slope = float(coeffs[1])

    expected = intercept + slope * max_pident
    adjusted = raw_scores - expected

    corr = np.corrcoef(raw_scores, max_pident)[0, 1]
    corr = float(corr) if np.isfinite(corr) else float("nan")
    return adjusted, intercept, slope, corr


def compute_metrics(labels: List[Dict[str, str]], scores: Dict[str, float]) -> Dict[str, float]:
    """Compute raw and similarity-adjusted metrics for one family."""
    seq_ids: List[str] = []
    y_true: List[int] = []
    y_score_raw: List[float] = []
    max_pident: List[float] = []
    for row in labels:
        sid = row["seq_id"]
        if sid not in scores:
            continue
        if "max_pident_to_conditioning" not in row:
            raise KeyError("labels.tsv is missing `max_pident_to_conditioning` column.")
        seq_ids.append(sid)
        y_true.append(int(row["label"]))
        y_score_raw.append(float(scores[sid]))
        max_pident.append(float(row["max_pident_to_conditioning"]))

    if len(set(y_true)) < 2:
        raise ValueError("Need both positive and negative labels to compute metrics.")

    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_raw_arr = np.asarray(y_score_raw, dtype=float)
    max_pident_arr = np.asarray(max_pident, dtype=float)

    adjusted_arr, intercept, slope, corr = _fit_similarity_adjustment(
        y_score_raw_arr, max_pident_arr
    )

    raw_scores_by_seq_id = {
        sid: score for sid, score in zip(seq_ids, y_score_raw_arr.tolist())
    }
    adjusted_scores_by_seq_id = {
        sid: score for sid, score in zip(seq_ids, adjusted_arr.tolist())
    }
    raw_metrics = _binary_metrics(
        y_true=y_true_arr,
        y_score=y_score_raw_arr,
        labels=labels,
        scores_by_seq_id=raw_scores_by_seq_id,
    )
    adjusted_metrics = _binary_metrics(
        y_true=y_true_arr,
        y_score=adjusted_arr,
        labels=labels,
        scores_by_seq_id=adjusted_scores_by_seq_id,
    )

    return {
        "n_samples": int(len(y_true_arr)),
        "n_pos": int(np.sum(y_true_arr == 1)),
        "n_neg": int(np.sum(y_true_arr == 0)),
        "similarity_adjust_intercept": intercept,
        "similarity_adjust_slope": slope,
        "raw_similarity_corr": corr,
        **{f"raw_{k}": v for k, v in raw_metrics.items()},
        **{f"adjusted_{k}": v for k, v in adjusted_metrics.items()},
    }


def _instance_dirs(instances_dir: Path) -> Iterable[Path]:
    for p in sorted(instances_dir.iterdir()):
        if not p.is_dir():
            continue
        if (
            (p / "conditioning.fasta").exists()
            and (p / "candidates.fasta").exists()
            and (p / "labels.tsv").exists()
        ):
            yield p


def main():
    parser = argparse.ArgumentParser(description="Run ProFam scorer on benchmark families")
    parser.add_argument(
        "--instances-dir",
        default="data/benchmark/instances",
        help="Directory containing benchmark family instance folders",
    )
    parser.add_argument(
        "--n-families",
        type=int,
        default=20,
        help="Number of families to score (sorted deterministic order)",
    )
    parser.add_argument(
        "--max-prompt-residues",
        type=int,
        default=6000,
        help="Maximum total conditioning residues passed as prompt",
    )
    parser.add_argument(
        "--output-csv",
        default="evaluation/results/profam_20_families_metrics.csv",
        help="Where to write per-family metrics CSV",
    )
    args = parser.parse_args()

    instances_dir = Path(args.instances_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    selected_dirs = list(_instance_dirs(instances_dir))[: args.n_families]
    if not selected_dirs:
        raise FileNotFoundError(f"No valid benchmark instances found in {instances_dir}")

    rows = []
    print(
        f"Scoring {len(selected_dirs)} families with prompt cap "
        f"{args.max_prompt_residues} residues..."
    )
    for idx, instance_dir in enumerate(selected_dirs, start=1):
        family_id = instance_dir.name
        print(f"  [{idx}/{len(selected_dirs)}] {family_id}")
        model = ProFam()
        scores, prompt_n, prompt_res = score(
            model,
            conditioning_fasta=instance_dir / "conditioning.fasta",
            candidates_fasta=instance_dir / "candidates.fasta",
            max_prompt_residues=args.max_prompt_residues,
        )
        labels = _read_labels(instance_dir / "labels.tsv")
        metrics = compute_metrics(labels, scores)
        for k,v in metrics.items():
            print(k, v)
        rows.append(
            {
                "family_id": family_id,
                "n_prompt_sequences": prompt_n,
                "prompt_residues": prompt_res,
                **metrics,
            }
        )

    mean_row = {"family_id": "MEAN", "n_prompt_sequences": "", "prompt_residues": ""}
    metric_keys = [k for k in rows[0].keys() if k not in {"family_id", "n_prompt_sequences", "prompt_residues"}]
    for key in metric_keys:
        mean_row[key] = float(np.nanmean([float(r[key]) for r in rows]))
    rows.append(mean_row)

    with open(output_csv, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows) - 1} family rows + mean row to {output_csv}")


if __name__ == "__main__":
    main()
