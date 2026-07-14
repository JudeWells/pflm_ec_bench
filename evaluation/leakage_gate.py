#!/usr/bin/env python3
"""Leakage gate: can a cheap homology statistic solve the benchmark?

A conditional-PLM benchmark that claims to control for homology is sound only if
statistics the construction pipeline never optimised against sit at chance. This
script measures that directly.

For every candidate it recomputes, against the instance's conditioning set:

    pident      max local percent identity              (the variable v2 matched on)
    coverage    max alnlen / max(qlen, tlen)
    covadj      max pident * coverage
    neglog_e    -log10 of the best E-value
    log_alnlen  log10 alignment length of the best covadj hit
    log_tlen    log10 length of the candidate

plus, optionally, max cosine similarity to the conditioning set in an embedding
space supplied via --embeddings.

It then reports, for each statistic, the **within-instance** AUROC averaged over
instances. Within-instance is the only meaningful unit: models score each
instance's candidates against that instance's prompt, so pooling across families
mixes in between-family variation and understates the leak.

Finally it fits a *per-instance* learned adversary (logistic regression over all
statistics, cross-validated inside the instance). A single global model is a
weak adversary because its coefficients cannot adapt to per-family scale; fitting
per instance is what a model exploiting the leak would effectively do.

Exit status is non-zero if any statistic exceeds the gate threshold.
"""
import argparse
import csv
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["pident", "coverage", "covadj", "neglog_e", "log_alnlen", "log_tlen"]
NEG_INF_E = 300.0


def neglog10(evalue):
    return min(NEG_INF_E, -math.log10(max(evalue, 1e-300)))


def read_fasta_ids(path):
    return {line.split()[0][1:] for line in open(path) if line.startswith(">")}


def load_instances(benchmark_dir):
    """Return {ec: {'cond': set, 'labels': [(seq_id, label), ...]}}."""
    inst_dir = Path(benchmark_dir) / "instances"
    instances = {}
    for d in sorted(inst_dir.iterdir()):
        if not (d / "labels.tsv").exists():
            continue
        labels = [
            (r["seq_id"], int(r["label"]))
            for r in csv.DictReader(open(d / "labels.tsv"), delimiter="\t")
        ]
        instances[d.name] = {
            "cond": read_fasta_ids(d / "conditioning.fasta"),
            "labels": labels,
        }
    return instances


def harvest_features(instances, search_tsvs, cache_path):
    """Stream mmseqs results once; collect per (instance, candidate) features.

    Each statistic is maximised (or minimised, for E-value) independently over
    the conditioning set, mirroring how ``compute_max_sim_to_conditioning`` in
    the construction pipeline aggregates.
    """
    tsvs = [Path(p) for p in search_tsvs if Path(p).exists()]
    if cache_path.exists():
        newest = max(p.stat().st_mtime for p in tsvs) if tsvs else 0
        if cache_path.stat().st_mtime >= newest:
            print(f"  loading cached features: {cache_path.name}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    cond_map = defaultdict(list)
    cand_of = {}
    for ec, d in instances.items():
        for cid in d["cond"]:
            cond_map[cid].append(ec)
        cand_of[ec] = {s for s, _ in d["labels"]}

    # feats[ec][seq] = [pident, coverage, covadj, neglog_e, alnlen, tlen]
    feats = {ec: {} for ec in instances}

    for path in tsvs:
        print(f"  streaming {path.name} ...", flush=True)
        with open(path) as f:
            for line in f:
                p = line.split("\t")
                if len(p) < 7:
                    continue
                q, t = p[0], p[1]
                if q not in cond_map and t not in cond_map:
                    continue
                for cid, other in ((q, t), (t, q)):
                    ecs = cond_map.get(cid)
                    if not ecs or other == cid:
                        continue
                    try:
                        pid = float(p[2])
                        aln = int(p[3])
                        ev = float(p[4])
                        qlen, tlen = int(p[5]), int(p[6])
                    except ValueError:
                        continue
                    # `other` is the candidate; its own length is qlen if it was
                    # the query, else tlen.
                    olen = qlen if other == q else tlen
                    cov = aln / max(qlen, tlen)
                    cadj = pid * cov
                    nle = neglog10(ev)
                    for ec in ecs:
                        if other not in cand_of[ec]:
                            continue
                        row = feats[ec].get(other)
                        if row is None:
                            feats[ec][other] = [pid, cov, cadj, nle, aln, olen]
                        else:
                            row[0] = max(row[0], pid)
                            row[1] = max(row[1], cov)
                            if cadj > row[2]:
                                row[2], row[4] = cadj, aln
                            row[3] = max(row[3], nle)
                            row[5] = olen

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  cached to {cache_path.name}")
    return feats


def harvest_from_retrieval(instances, retrieval_tsv, cov_floor):
    """Features from the complete conditioning-vs-all search (08_retrieval_search.py).

    Prefer this over --search-tsv. The step-03 MMseqs2 output is all-vs-all only
    within an EC3 group, with cross-EC3 pairs restricted to 50% cluster
    representatives, so a candidate's absence from it is not "no similarity" but
    "never measured" — and absence is itself class-correlated (on v4, 11.2% of
    negatives are absent versus 3.7% of positives, giving the bare presence
    indicator an AUROC of 0.62). Imputing those to 0.0 hands the adversary a
    signal that is an artifact of the *search plan*, not of the benchmark.

    Here every (conditioning member, database sequence) pair is measured
    uniformly, so a zero means a genuine absence of detectable homology.
    """
    cond_of = defaultdict(list)
    for ec, d in instances.items():
        for c in d["cond"]:
            cond_of[c].append(ec)
    want = {ec: {s for s, _ in d["labels"]} for ec, d in instances.items()}

    feats = {ec: {} for ec in instances}
    for line in open(retrieval_tsv):
        f = line.rstrip("\n").split("\t")
        if len(f) < 10:
            continue
        q, t = f[0], f[1]
        ecs = cond_of.get(q)
        if not ecs:
            continue
        pid, aln, ev, bits = float(f[2]), int(f[3]), float(f[4]), float(f[5])
        qcov, tcov, tlen = float(f[6]), float(f[7]), int(f[9])
        qualifies = qcov >= cov_floor and tcov >= cov_floor
        for ec in ecs:
            if t not in want[ec]:
                continue
            row = feats[ec].get(t)
            if row is None:
                # [S(retrieval score), coverage, covadj, neglog_e, alnlen, tlen]
                row = feats[ec][t] = [0.0, 0.0, 0.0, 0.0, 0.0, float(tlen)]
            if qualifies and pid > row[0]:
                row[0] = pid                    # S: max qualifying pident
            row[1] = max(row[1], tcov)
            row[2] = max(row[2], pid * tcov)
            row[3] = max(row[3], neglog10(ev))
            row[4] = max(row[4], float(aln))
    return feats


def add_profile_features(instances, feats, hits_path, n_base):
    """Append (hmm_bitscore, hmm_neglog_e) from 06_profile_bitscores.py output.

    A candidate with no reported hit gets 0 bits / 0 -log10E, which is what a
    practitioner ranking by bitscore would effectively assign it.
    """
    hits = defaultdict(dict)
    for r in csv.DictReader(open(hits_path), delimiter="\t"):
        hits[r["ec"]][r["seq_id"]] = (float(r["bitscore"]), neglog10(float(r["evalue"])))
    for ec, d in instances.items():
        for sid, _ in d["labels"]:
            row = feats[ec].setdefault(sid, [0.0] * n_base)
            bits, nle = hits.get(ec, {}).get(sid, (0.0, 0.0))
            row.extend([bits, nle])
    return feats


def add_embedding_feature(instances, feats, emb_path):
    """Max cosine similarity to any conditioning sequence, from a .npz of vectors."""
    z = np.load(emb_path)
    emb = {k: z[k] for k in z.files}
    for ec, d in instances.items():
        cond = np.stack([emb[c] for c in d["cond"] if c in emb]) if d["cond"] else None
        if cond is None or not len(cond):
            continue
        cond = cond / np.linalg.norm(cond, axis=1, keepdims=True)
        for sid, _ in d["labels"]:
            if sid not in emb or sid not in feats[ec]:
                continue
            v = emb[sid] / np.linalg.norm(emb[sid])
            feats[ec][sid].append(float((cond @ v).max()))
    return feats


def build_matrix(instances, feats, ec, n_feat):
    X, y = [], []
    for sid, lab in instances[ec]["labels"]:
        row = feats[ec].get(sid)
        if row is None:
            # No detectable alignment to the conditioning set: this is exactly
            # what the construction pipeline coerced to 0.0, so mirror it.
            row = [0.0] * n_feat
        else:
            row = list(row)
            # Indices 4/5 are alignment length and sequence length in both feature
            # layouts; compress their dynamic range for the linear adversary.
            row[4] = math.log10(row[4] + 1.0)
            row[5] = math.log10(row[5] + 1.0)
        X.append(row[:n_feat])
        y.append(lab)
    return np.asarray(X, dtype=float), np.asarray(y)


def per_instance_adversary(X, y, seed=0):
    """Cross-validated logistic regression fitted *inside* the instance."""
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    n_splits = min(5, n_pos, n_neg)
    if n_splits < 2:
        return None
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000)
        ).fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", required=True)
    ap.add_argument("--search-tsv", nargs="+", default=[
        "data/processed/mmseqs/within_ec3_results.tsv",
        "data/processed/mmseqs/cross_ec3_results.tsv",
    ])
    ap.add_argument("--retrieval-tsv", default=None,
                    help="cond_vs_all.tsv from 08_retrieval_search.py. Preferred "
                         "feature source: measures every candidate uniformly, so a "
                         "zero means no homology rather than 'not searched'.")
    ap.add_argument("--cov-floor", type=float, default=0.6,
                    help="coverage floor defining the retrieval score S")
    ap.add_argument("--profile-hits", default=None,
                    help="profile_hits.tsv from 06_profile_bitscores.py; adds "
                         "hmm_bitscore + hmm_neglog_e to the gate")
    ap.add_argument("--embeddings", default=None,
                    help=".npz mapping seq_id -> embedding vector (optional)")
    ap.add_argument("--min-pairs", type=int, default=5,
                    help="skip instances with fewer pairs; their AUROC is quantised")
    ap.add_argument("--max-auroc", type=float, default=0.55,
                    help="gate threshold on mean within-instance AUROC")
    ap.add_argument("--per-instance-out", default=None)
    args = ap.parse_args()

    bdir = Path(args.benchmark_dir)
    instances = load_instances(bdir)
    print(f"Loaded {len(instances)} instances from {bdir}")

    if args.retrieval_tsv:
        print(f"  features from {args.retrieval_tsv} (cov floor {args.cov_floor:.0%})")
        feats = harvest_from_retrieval(instances, args.retrieval_tsv, args.cov_floor)
        names = ["S_retrieval", "coverage", "covadj", "neglog_e",
                 "alnlen", "tlen"]
    else:
        feats = harvest_features(instances, args.search_tsv,
                                 bdir / "leakage_features.pkl")
        names = list(FEATURES)
    if args.profile_hits:
        feats = add_profile_features(instances, feats, args.profile_hits, len(names))
        names += ["hmm_bitscore", "hmm_neglog_e"]
    if args.embeddings:
        feats = add_embedding_feature(instances, feats, args.embeddings)
        names.append("emb_cosine")
    n_feat = len(names)

    rows = []
    for ec in instances:
        y = np.array([lab for _, lab in instances[ec]["labels"]])
        if y.sum() < args.min_pairs or (1 - y).sum() < args.min_pairs:
            continue
        X, y = build_matrix(instances, feats, ec, n_feat)
        if len(set(y)) < 2:
            continue
        rec = {"ec": ec, "n_pairs": int(y.sum())}
        for i, nm in enumerate(names):
            rec[nm] = roc_auc_score(y, X[:, i]) if X[:, i].std() > 0 else 0.5
        adv = per_instance_adversary(X, y)
        rec["adversary"] = adv if adv is not None else float("nan")
        rec["best_single"] = max(max(rec[nm], 1 - rec[nm]) for nm in names)
        rows.append(rec)

    if not rows:
        print("No instances with enough pairs to evaluate.")
        return 1

    print(f"\nEvaluated {len(rows)} instances with >= {args.min_pairs} pairs "
          f"({sum(r['n_pairs'] for r in rows):,} pairs)\n")

    hdr = f"{'statistic':>14s} {'mean':>8s} {'median':>8s} {'>0.60':>7s} {'>0.70':>7s}"
    print(hdr)
    print("-" * len(hdr))
    failures = []
    for nm in names + ["adversary"]:
        v = np.array([r[nm] for r in rows if not math.isnan(r[nm])])
        mean = v.mean()
        print(f"{nm:>14s} {mean:8.3f} {np.median(v):8.3f} "
              f"{(v > 0.60).mean():7.1%} {(v > 0.70).mean():7.1%}")
        # Deviation in either direction is a design failure: a below-chance
        # homology statistic means negatives were made *more* homologous than
        # positives, which penalises any model that legitimately correlates
        # with homology.
        if abs(mean - 0.5) > (args.max_auroc - 0.5):
            failures.append((nm, mean))

    bs = np.array([r["best_single"] for r in rows])
    print(f"\n{'best single stat per instance (optimistic):':<46s} "
          f"mean {bs.mean():.3f}, median {np.median(bs):.3f}")

    if args.per_instance_out:
        with open(args.per_instance_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"per-instance results -> {args.per_instance_out}")

    print()
    if failures:
        print(f"GATE FAILED (threshold |AUROC - 0.5| <= {args.max_auroc - 0.5:.2f}):")
        for nm, mean in failures:
            print(f"  {nm}: mean within-instance AUROC {mean:.3f}")
        print("\nA cheap homology statistic solves this benchmark. Results on it "
              "cannot distinguish functional signal from homology.")
        return 1

    print(f"GATE PASSED: no statistic exceeds |AUROC - 0.5| = {args.max_auroc - 0.5:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
