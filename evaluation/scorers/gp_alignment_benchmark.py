#!/usr/bin/env python3
"""Score benchmark families with a GP classifier using a pairwise alignment kernel.

For each EC family instance the scorer:
1. Selects training positives from the conditioning set and training negatives
   from the candidate set (using ground-truth labels).
2. Computes a normalized Smith-Waterman local-alignment kernel over all
   training + candidate sequences (BLOSUM62, gap-open -11, gap-extend -1).
3. Fits a Gaussian-process classifier (Laplace approximation) on the training
   split and predicts class probabilities for every candidate.
4. Reports the same raw + similarity-adjusted metrics as the ProFam scorer.
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Kernel
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# I/O helpers (duplicated from profam_benchmark to avoid importing profam)
# ---------------------------------------------------------------------------

def _parse_seq_id(raw_id: str) -> str:
    if "|" in raw_id:
        parts = raw_id.split("|")
        if len(parts) >= 2:
            return parts[1]
    return raw_id


def _read_fasta_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for rec in SeqIO.parse(path, "fasta"):
        pairs.append((_parse_seq_id(rec.id), str(rec.seq)))
    return pairs


def _read_labels(labels_tsv: Path) -> List[Dict[str, str]]:
    with open(labels_tsv, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _instance_dirs(instances_dir: Path):
    for p in sorted(instances_dir.iterdir()):
        if not p.is_dir():
            continue
        if (
            (p / "conditioning.fasta").exists()
            and (p / "candidates.fasta").exists()
            and (p / "labels.tsv").exists()
        ):
            yield p


# ---------------------------------------------------------------------------
# Metrics (same as profam_benchmark)
# ---------------------------------------------------------------------------

def _paired_ranking_accuracy(labels: List[Dict[str, str]], scores: Dict[str, float]) -> float:
    seq_to_label = {row["seq_id"]: int(row["label"]) for row in labels}
    seq_to_pair = {row["seq_id"]: row["paired_with"] for row in labels}
    seen = set()
    wins = total = 0.0
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
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = int(np.argmax(tpr - fpr))
    best_thr = thresholds[best_idx]
    y_pred = (y_score >= best_thr).astype(int)
    return float(np.mean(y_pred == y_true))


def _binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: List[Dict[str, str]],
    scores_by_seq_id: Dict[str, float],
) -> Dict[str, float]:
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
    design = np.column_stack([np.ones_like(max_pident), max_pident])
    coeffs, *_ = np.linalg.lstsq(design, raw_scores, rcond=None)
    intercept, slope = float(coeffs[0]), float(coeffs[1])
    adjusted = raw_scores - (intercept + slope * max_pident)
    corr = np.corrcoef(raw_scores, max_pident)[0, 1]
    corr = float(corr) if np.isfinite(corr) else float("nan")
    return adjusted, intercept, slope, corr


def compute_metrics(labels: List[Dict[str, str]], scores: Dict[str, float]) -> Dict[str, float]:
    seq_ids, y_true, y_score_raw, max_pident = [], [], [], []
    for row in labels:
        sid = row["seq_id"]
        if sid not in scores:
            continue
        seq_ids.append(sid)
        y_true.append(int(row["label"]))
        y_score_raw.append(float(scores[sid]))
        max_pident.append(float(row["max_pident_to_conditioning"]))

    y_true_arr = np.asarray(y_true, dtype=int)
    y_raw_arr = np.asarray(y_score_raw)
    pident_arr = np.asarray(max_pident)

    adjusted_arr, intercept, slope, corr = _fit_similarity_adjustment(y_raw_arr, pident_arr)

    raw_by_id = dict(zip(seq_ids, y_raw_arr.tolist()))
    adj_by_id = dict(zip(seq_ids, adjusted_arr.tolist()))

    raw_m = _binary_metrics(y_true_arr, y_raw_arr, labels, raw_by_id)
    adj_m = _binary_metrics(y_true_arr, adjusted_arr, labels, adj_by_id)

    return {
        "n_samples": len(y_true_arr),
        "n_pos": int(np.sum(y_true_arr == 1)),
        "n_neg": int(np.sum(y_true_arr == 0)),
        "similarity_adjust_intercept": intercept,
        "similarity_adjust_slope": slope,
        "raw_similarity_corr": corr,
        **{f"raw_{k}": v for k, v in raw_m.items()},
        **{f"adjusted_{k}": v for k, v in adj_m.items()},
    }


# ---------------------------------------------------------------------------
# Precomputed alignment kernel for sklearn GP
# ---------------------------------------------------------------------------

class PrecomputedAlignmentKernel(Kernel):
    """Wraps a precomputed Gram matrix so sklearn's GPC can use it."""

    def __init__(self, gram_matrix: np.ndarray):
        self.gram_matrix = gram_matrix

    def __call__(self, X, Y=None, eval_gradient=False):
        xi = X[:, 0].astype(int)
        if Y is None:
            K = self.gram_matrix[np.ix_(xi, xi)]
        else:
            yi = Y[:, 0].astype(int)
            K = self.gram_matrix[np.ix_(xi, yi)]
        if eval_gradient:
            return K, np.empty((K.shape[0], K.shape[1], 0))
        return K

    def diag(self, X):
        xi = X[:, 0].astype(int)
        return np.array(self.gram_matrix[xi, xi]).ravel()

    def is_stationary(self):
        return False

    @property
    def hyperparameters(self):
        return []

    @property
    def theta(self):
        return np.array([])

    @theta.setter
    def theta(self, value):
        pass

    @property
    def bounds(self):
        return np.empty((0, 2))

    def get_params(self, deep=True):
        return {"gram_matrix": self.gram_matrix}

    def clone_with_theta(self, theta):
        return PrecomputedAlignmentKernel(self.gram_matrix)


# ---------------------------------------------------------------------------
# Alignment kernel computation
# ---------------------------------------------------------------------------

def _sanitize_sequence(seq: str) -> str:
    """Replace non-standard amino acids with closest BLOSUM62-compatible residue."""
    table = str.maketrans({"U": "C", "X": "A", "B": "D", "Z": "E", "J": "L", "O": "K"})
    return seq.translate(table)


def compute_alignment_kernel(sequences: List[str], verbose: bool = False) -> np.ndarray:
    """Normalised Smith-Waterman local-alignment kernel (BLOSUM62)."""
    sequences = [_sanitize_sequence(s) for s in sequences]
    n = len(sequences)
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1

    raw = np.zeros((n, n))
    total_pairs = n * (n + 1) // 2
    done = 0
    for i in range(n):
        for j in range(i, n):
            s = aligner.score(sequences[i], sequences[j])
            raw[i, j] = s
            raw[j, i] = s
            done += 1
        if verbose and (i + 1) % 50 == 0:
            print(f"      aligned {done}/{total_pairs} pairs")

    diag = np.sqrt(np.diag(raw).copy())
    diag[diag == 0] = 1.0
    K = raw / np.outer(diag, diag)
    K += 1e-6 * np.eye(n)
    return K


# ---------------------------------------------------------------------------
# Family data container — loads data and kernel once, supports repeated scoring
# ---------------------------------------------------------------------------

class FamilyData:
    """Preloads sequences, labels, and kernel for one benchmark family."""

    def __init__(self, instance_dir: Path, n_train_pos: int = 10, seed: int = 42):
        self.instance_dir = instance_dir
        self.family_id = instance_dir.name
        self.seed = seed

        self.cond_pairs = _read_fasta_pairs(instance_dir / "conditioning.fasta")
        self.cand_pairs = _read_fasta_pairs(instance_dir / "candidates.fasta")
        self.labels = _read_labels(instance_dir / "labels.tsv")
        self.label_map = {row["seq_id"]: int(row["label"]) for row in self.labels}
        self.pair_map = {row["seq_id"]: row["paired_with"] for row in self.labels}

        self.neg_cands = [
            (sid, seq) for sid, seq in self.cand_pairs if self.label_map.get(sid) == 0
        ]
        self.n_total_neg = len(self.neg_cands)
        self.n_total_pos = sum(1 for sid, _ in self.cand_pairs if self.label_map.get(sid) == 1)

        # Sample training positives from conditioning (fixed across sweeps)
        rng = np.random.RandomState(seed)
        if len(self.cond_pairs) > n_train_pos:
            idx = rng.choice(len(self.cond_pairs), n_train_pos, replace=False)
            self.train_pos = [self.cond_pairs[i] for i in idx]
        else:
            self.train_pos = list(self.cond_pairs)

        # Build combined sequence list: train_pos + all candidates (deduplicated)
        self.all_ids: List[str] = []
        self.all_seqs: List[str] = []
        self.id_to_idx: Dict[str, int] = {}

        for sid, seq in self.train_pos:
            self._add(sid, seq)
        for sid, seq in self.cand_pairs:
            self._add(sid, seq)

        # Compute kernel once
        n = len(self.all_seqs)
        print(f"    Computing alignment kernel for {n} sequences "
              f"({n * (n + 1) // 2} pairs)...")
        self.gram = compute_alignment_kernel(self.all_seqs, verbose=(n > 100))

    def _add(self, sid: str, seq: str):
        if sid not in self.id_to_idx:
            self.id_to_idx[sid] = len(self.all_ids)
            self.all_ids.append(sid)
            self.all_seqs.append(seq)

    def score_with_n_neg(self, n_train_neg: int, seed: int = 42) -> Optional[Dict]:
        """Fit GP with n_train_neg negatives, return metrics dict or None."""
        rng = np.random.RandomState(seed)

        # Cap to keep ≥2 pos + ≥2 neg for eval
        max_neg = min(self.n_total_neg - 2, self.n_total_pos - 2)
        actual_n_neg = min(n_train_neg, max(0, max_neg))

        if actual_n_neg > 0:
            idx = rng.choice(len(self.neg_cands), actual_n_neg, replace=False)
            train_neg = [self.neg_cands[i] for i in idx]
        else:
            train_neg = []

        # Prepare training arrays
        train_indices = np.array(
            [self.id_to_idx[sid] for sid, _ in self.train_pos]
            + [self.id_to_idx[sid] for sid, _ in train_neg]
        ).reshape(-1, 1)
        train_labels = np.array([1] * len(self.train_pos) + [0] * len(train_neg))

        if len(set(train_labels)) < 2:
            return None

        test_indices = np.array(
            [self.id_to_idx[sid] for sid, _ in self.cand_pairs]
        ).reshape(-1, 1)

        # Fit and predict
        kernel = PrecomputedAlignmentKernel(self.gram)
        gpc = GaussianProcessClassifier(kernel=kernel, optimizer=None, random_state=seed)
        gpc.fit(train_indices, train_labels)
        probs = gpc.predict_proba(test_indices)[:, 1]

        all_scores = {sid: float(p) for (sid, _), p in zip(self.cand_pairs, probs)}

        # Exclude training negatives and their paired positives from eval
        train_neg_ids = {sid for sid, _ in train_neg}
        excluded = set(train_neg_ids)
        for sid in train_neg_ids:
            partner = self.pair_map.get(sid)
            if partner:
                excluded.add(partner)

        eval_scores = {sid: s for sid, s in all_scores.items() if sid not in excluded}

        try:
            metrics = compute_metrics(self.labels, eval_scores)
        except ValueError:
            return None

        metrics["n_train_pos"] = len(self.train_pos)
        metrics["n_train_neg"] = actual_n_neg
        return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run GP alignment-kernel scorer on benchmark families"
    )
    parser.add_argument(
        "--instances-dir", default="data/benchmark/instances",
    )
    parser.add_argument(
        "--n-families", type=int, default=20,
    )
    parser.add_argument(
        "--n-train-pos", type=int, default=10,
    )
    parser.add_argument(
        "--neg-fraction", type=float, default=0.5,
        help="Fraction of candidate negatives to use for training (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--output-csv",
        default="evaluation/results/gp_alignment_metrics.csv",
    )
    args = parser.parse_args()

    instances_dir = Path(args.instances_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    selected_dirs = list(_instance_dirs(instances_dir))[: args.n_families]
    if not selected_dirs:
        raise FileNotFoundError(f"No valid benchmark instances in {instances_dir}")

    rows = []
    print(f"Scoring {len(selected_dirs)} families  "
          f"(neg_fraction={args.neg_fraction}, n_train_pos={args.n_train_pos})...")

    for idx, instance_dir in enumerate(selected_dirs, start=1):
        family_id = instance_dir.name
        print(f"  [{idx}/{len(selected_dirs)}] {family_id}")
        try:
            fam = FamilyData(instance_dir, n_train_pos=args.n_train_pos, seed=args.seed)
            n_neg = max(1, int(round(fam.n_total_neg * args.neg_fraction)))
            metrics = fam.score_with_n_neg(n_neg, seed=args.seed)
            if metrics is None:
                print("    SKIPPED: not enough data after split")
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

    # mean row
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
