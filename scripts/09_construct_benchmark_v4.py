#!/usr/bin/env python3
"""Construct v4: matched on the naive sequence-only retrieval score.

The design compromise, after v2 and v3 both failed the leakage gate.

Homology cannot be fully removed from this task (see README). So instead of
chasing a set on which *every* homology statistic is at chance — which is either
impossible or empties the benchmark — we:

  1. Define, precisely, the naive sequence-only method a practitioner would use.
  2. Restrict the candidate pool to what that method actually *retrieves*.
  3. Match positives to negatives on that method's own score.
  4. Report the method as the headline baseline, and report honestly what still
     leaks. Other methods have to beat it.

**The naive retriever.** Search the known family members (the conditioning set)
against the database; discard hits that do not align over most of both sequences;
rank the survivors by percent identity:

    S(c) = max { pident(q, c) : q in conditioning, qcov >= T, tcov >= T }

**The pool.** Candidates are exactly the sequences with at least one qualifying
alignment. A sequence with no such alignment was never retrieved, so it is not a
candidate — it is not a "hard negative", it is a non-result. This is the step that
kills the coverage leak that sank v2: the short spurious local alignments that
made negatives look identity-matched are *removed from the pool by construction*
rather than balanced against.

**Why a coverage floor of 0.6.** Swept empirically (README). At T=0 the design
degenerates to v2 and the learned adversary sits at 0.90. T=0.5 drops it to 0.71,
T=0.6 to 0.65, with bitscore, alignment length and sequence length all falling to
chance. Higher floors buy nothing and cost instances. An additional coverage
caliper on the pair was tested and rejected: it bought ~0.03 of adversary AUROC
for a real loss of pairs.

**Positives must be experimentally evidenced.** Tested and load-bearing: allowing
homology-propagated positives raises the adversary from 0.65 to 0.76 and degrades
the matching itself (S drifts from 0.55 to 0.61), because propagated labels were
assigned *by* homology and therefore sit at high identity where no decoy exists to
match them. This is the circularity of README §7, visible in the numbers.

Verify with:
    python evaluation/leakage_gate.py --benchmark-dir data/benchmark_v4
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.io_utils import read_fasta, write_fasta, write_tsv, write_json
from utils.family import classify_decoy_tier, is_complete_ec4


def load_ec_annotations(path):
    """(seq -> complete EC4s, seq -> partial EC prefixes)."""
    full, partial = defaultdict(set), defaultdict(set)
    for r in csv.DictReader(open(path), delimiter="\t"):
        acc = r.get("Entry") or r.get("accession")
        for e in (r.get("EC number") or "").split(";"):
            e = e.strip()
            if not e:
                continue
            if is_complete_ec4(e):
                full[acc].add(e)
            elif "-" in e:
                partial[acc].add(e)
    return dict(full), dict(partial)


def partial_conflicts(partials, target_ec):
    """A decoy annotated `1.1.1.-` cannot be used as a negative for `1.1.1.5`."""
    tp = target_ec.split(".")
    for e in partials:
        pp = e.split(".")
        if len(pp) == 4 and all(a == "-" or a == b for a, b in zip(pp, tp)):
            return True
    return False


def load_retrieval(path, cond_of, cond_sets, cov_floor):
    """Apply the coverage floor and reduce to per-(instance, candidate) features.

    Returns {ec: {seq_id: (S, coverage, bits, alnlen, tlen)}} over the retrieved
    pool only — candidates with no qualifying alignment are simply absent.
    """
    acc = {ec: {} for ec in cond_sets}
    n_rows = n_kept = 0
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 10:
            continue
        n_rows += 1
        q, t = f[0], f[1]
        ecs = cond_of.get(q)
        if not ecs:
            continue
        qcov, tcov = float(f[6]), float(f[7])
        if qcov < cov_floor or tcov < cov_floor:
            continue
        pid, alnlen, bits = float(f[2]), int(f[3]), float(f[5])
        tlen = int(f[9])
        n_kept += 1
        for ec in ecs:
            if t in cond_sets[ec]:
                continue
            cur = acc[ec].get(t)
            # S is the max qualifying pident; the other features are taken from
            # whichever qualifying alignment is best by identity, except coverage
            # and bits which take their own max (a candidate is as "retrievable"
            # as its best evidence).
            if cur is None or pid > cur[0]:
                acc[ec][t] = [pid, tcov, bits, alnlen, tlen]
            else:
                cur[1] = max(cur[1], tcov)
                cur[2] = max(cur[2], bits)
    print(f"  {n_rows:,} alignments, {n_kept:,} pass the {cov_floor:.0%} coverage floor")
    return acc


def match_nearest(positives, decoys, caliper):
    """1:1 nearest-neighbour matching on the retrieval score S.

    Positives are consumed in descending S: high-S decoys are the scarce resource,
    so allocating them to the positives that need them first maximises retention.
    Matching is *nearest*, not nearest-above — the target is a baseline AUROC of
    0.5, not below it (README §9).
    """
    pool = sorted(decoys, key=lambda d: d[1][0])
    svals = [d[1][0] for d in pool]
    used, pairs = set(), []
    for pid, pf in sorted(positives, key=lambda p: -p[1][0]):
        lo = np.searchsorted(svals, pf[0] - caliper, "left")
        hi = np.searchsorted(svals, pf[0] + caliper, "right")
        best, best_d = None, float("inf")
        for j in range(lo, hi):
            did, df = pool[j]
            if did in used:
                continue
            d = abs(df[0] - pf[0])
            if d < best_d:
                best, best_d = pool[j], d
        if best is not None:
            used.add(best[0])
            pairs.append((pid, pf, best[0], best[1]))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-benchmark", default="data/benchmark_v2",
                    help="supplies conditioning sets (prompts)")
    ap.add_argument("--retrieval", default="data/processed/retrieval/cond_vs_all.tsv")
    ap.add_argument("--metadata", default="data/raw/swissprot_ec_metadata.tsv")
    ap.add_argument("--experimental", default="data/raw/swissprot_ec_experimental.tsv")
    ap.add_argument("--sequences", default="data/raw/swissprot_ec.fasta")
    ap.add_argument("--out-dir", default="data/benchmark_v4")
    ap.add_argument("--cov-floor", type=float, default=0.6,
                    help="min qcov AND tcov for an alignment to count as retrieved")
    ap.add_argument("--caliper", type=float, default=2.0,
                    help="max |ΔS| within a matched pair, in percentage points")
    ap.add_argument("--fn-identity", type=float, default=70.0,
                    help="exclude decoys at or above this identity to the prompt")
    ap.add_argument("--min-pairs", type=int, default=5)
    ap.add_argument("--max-pairs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.source_benchmark)

    print("Loading annotations...")
    seq_ec, seq_partial = load_ec_annotations(args.metadata)
    exp_ec, _ = load_ec_annotations(args.experimental)
    experimental = set(exp_ec)
    print(f"  {len(seq_ec):,} sequences with a complete EC4; "
          f"{len(experimental):,} with experimental catalytic-activity evidence")

    cond_sets, cond_of = {}, defaultdict(list)
    for d in sorted((src / "instances").iterdir()):
        fa = d / "conditioning.fasta"
        if not fa.exists():
            continue
        cond_sets[d.name] = set(read_fasta(fa))
        for c in cond_sets[d.name]:
            cond_of[c].append(d.name)
    print(f"  {len(cond_sets)} conditioning sets")

    print(f"Loading retrieval hits (coverage floor {args.cov_floor:.0%})...")
    retrieved = load_retrieval(args.retrieval, cond_of, cond_sets, args.cov_floor)

    all_seqs = read_fasta(args.sequences)
    out_root = Path(args.out_dir)
    manifest, skipped = [], defaultdict(int)

    for ec, pool in retrieved.items():
        target_ec = ec.replace("_", ".")

        positives, decoys = [], []
        for sid, f in pool.items():
            ecs = seq_ec.get(sid, set())
            if target_ec in ecs:
                # Positive: an experimentally-evidenced member of the target family.
                if sid in experimental:
                    positives.append((sid, f))
                continue
            if sid not in experimental or not ecs:
                continue
            # Negative: experimentally evidenced for a *different* reaction.
            if partial_conflicts(seq_partial.get(sid, ()), target_ec):
                continue
            # False-negative protection: at this identity to a known member, shared
            # function is near-certain regardless of what the annotation says.
            if f[0] >= args.fn_identity:
                continue
            decoys.append((sid, f + [min(classify_decoy_tier(target_ec, o) for o in ecs)]))

        if len(positives) < args.min_pairs:
            skipped["too_few_positives"] += 1
            continue
        if len(decoys) < args.min_pairs:
            skipped["too_few_decoys"] += 1
            continue
        if len(positives) > args.max_pairs:
            positives = random.sample(positives, args.max_pairs)

        pairs = match_nearest(positives, decoys, args.caliper)
        if len(pairs) < args.min_pairs:
            skipped["too_few_matched_pairs"] += 1
            continue

        inst = out_root / "instances" / ec
        inst.mkdir(parents=True, exist_ok=True)
        write_fasta({s: all_seqs[s] for s in cond_sets[ec] if s in all_seqs},
                    inst / "conditioning.fasta")

        cand, labels, gaps = {}, [], []
        for pid, pf, nid, nf in pairs:
            if pid not in all_seqs or nid not in all_seqs:
                continue
            cand[pid], cand[nid] = all_seqs[pid], all_seqs[nid]
            gaps.append(nf[0] - pf[0])
            tier = nf[5]
            for sid, lab, f in ((pid, 1, pf), (nid, 0, nf)):
                labels.append({
                    "seq_id": sid,
                    "label": lab,
                    "tier": tier,
                    "paired_with": nid if lab else pid,
                    "retrieval_score": round(f[0], 1),
                    "coverage": round(f[1], 3),
                    "score_gap": round(nf[0] - pf[0], 2),
                })
        if len(labels) < 2 * args.min_pairs:
            skipped["missing_sequences"] += 1
            continue

        write_fasta(cand, inst / "candidates.fasta")
        write_tsv(labels, inst / "labels.tsv")

        tiers = defaultdict(int)
        for _, _, _, nf in pairs:
            tiers[f"tier{nf[5]}"] += 1
        manifest.append({
            "ec_number": target_ec,
            "conditioning_size": len(cond_sets[ec]),
            "n_pairs": len(gaps),
            "n_positives_retrieved": len(positives),
            "n_decoys_retrieved": len(decoys),
            "mean_abs_score_gap": float(np.mean(np.abs(gaps))),
            "median_score_gap": float(np.median(gaps)),
            "tier_distribution": dict(tiers),
        })

    write_json({
        "instances": manifest,
        "n_instances": len(manifest),
        "total_pairs": sum(m["n_pairs"] for m in manifest),
        "skipped": dict(skipped),
        "baseline": "retrieval_score = max pident to conditioning set over "
                    f"alignments with qcov,tcov >= {args.cov_floor}",
        "config": vars(args),
    }, out_root / "manifest.json")

    print(f"\n=== v4 summary ===")
    print(f"Instances: {len(manifest)}   pairs: {sum(m['n_pairs'] for m in manifest):,}")
    print(f"Skipped: {dict(skipped)}")
    if manifest:
        g = np.array([m["mean_abs_score_gap"] for m in manifest])
        s = np.array([m["median_score_gap"] for m in manifest])
        print(f"Mean |ΔS| within a pair: median across instances {np.median(g):.2f} pp")
        print(f"Median signed gap (neg-pos): {np.median(s):+.2f} pp")
        tot = defaultdict(int)
        for m in manifest:
            for k, v in m["tier_distribution"].items():
                tot[k] += v
        print(f"Tiers: {dict(sorted(tot.items()))}")
    print(f"\nNow run: python evaluation/leakage_gate.py --benchmark-dir {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
