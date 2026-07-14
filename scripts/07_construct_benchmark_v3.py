#!/usr/bin/env python3
"""Construct v3 benchmark instances: profile-bitscore-matched, experimentally labelled.

Three changes from v2, each addressing a measured failure (see README):

1. **Match on the baseline you intend to nullify.** Negatives are matched to
   positives on profile-HMM bitscore against an HMM built from the conditioning
   set — the score a practitioner would actually rank a database by. Equalising
   any *other* scalar is whack-a-mole: the signal just moves into the neighbouring
   alignment statistics (README §1).

2. **Negatives carry contrary evidence.** A negative is a sequence experimentally
   shown to catalyse a *different* reaction, not merely one lacking the target
   annotation. Absence of annotation is not evidence of absence (README §7).

3. **Match to parity, not past it.** v2's `negative_higher` drove the homology
   baseline to ~0.32, which demands anti-correlation with homology and penalises
   any model that legitimately encodes it. Here negatives are matched *nearest*
   in bitscore, targeting a baseline AUROC of 0.5 (README §9).

Conditioning sets are reused verbatim from the source benchmark so that the HMMs
built by `06_profile_bitscores.py` remain valid. This inherits v2's conditioning
selection artifact (README §4); rebuilding prompts by holding out whole 70%
clusters is left as future work.

Verify the output with:
    python evaluation/leakage_gate.py --benchmark-dir data/benchmark_v3
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


def load_ec_map(path):
    """Return (seq -> complete EC4s, seq -> partial EC prefixes, ec -> seqs)."""
    seq_ec, seq_partial, ec_seq = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in csv.DictReader(open(path), delimiter="\t"):
        acc = r.get("Entry") or r.get("accession")
        for e in (r.get("EC number") or "").split(";"):
            e = e.strip()
            if not e:
                continue
            if is_complete_ec4(e):
                seq_ec[acc].add(e)
                ec_seq[e].add(acc)
            elif "-" in e:
                # e.g. `1.1.1.-`: cannot be a family label, but it *can* rule a
                # sequence out as a negative for any target under that prefix.
                seq_partial[acc].add(e)
    return dict(seq_ec), dict(seq_partial), dict(ec_seq)


def partial_conflicts(partials, target_ec):
    """True if any partial EC of this sequence is consistent with target_ec."""
    tp = target_ec.split(".")
    for e in partials:
        pp = e.split(".")
        if len(pp) == 4 and all(a == "-" or a == b for a, b in zip(pp, tp)):
            return True
    return False


def load_profile_hits(path):
    """{ec: {seq_id: (bitscore, evalue)}}, conditioning members excluded."""
    hits = defaultdict(dict)
    for r in csv.DictReader(open(path), delimiter="\t"):
        if int(r["is_conditioning"]):
            continue
        hits[r["ec"]][r["seq_id"]] = (float(r["bitscore"]), float(r["evalue"]))
    return hits


def match_nearest(positives, decoys, caliper_frac, caliper_abs):
    """1:1 nearest-neighbour matching on bitscore, with a caliper.

    Positives are consumed in descending bitscore order: high-bitscore decoys are
    scarce, so allocating them first maximises retention. Matching is *nearest*,
    not nearest-above, so the baseline is centred on 0.5 rather than pushed below.

    positives: [(seq_id, bits)]   decoys: [(seq_id, bits, tier)]
    returns:   [(pos_id, neg_id, tier, pos_bits, neg_bits)]
    """
    pool = sorted(decoys, key=lambda d: d[1])
    bits = [d[1] for d in pool]
    used = set()
    pairs = []

    for pid, pbits in sorted(positives, key=lambda p: -p[1]):
        cal = max(caliper_abs, caliper_frac * abs(pbits))
        lo = np.searchsorted(bits, pbits - cal, "left")
        hi = np.searchsorted(bits, pbits + cal, "right")
        best, best_d = None, float("inf")
        for j in range(lo, hi):
            did, dbits, tier = pool[j]
            if did in used:
                continue
            d = abs(dbits - pbits)
            # Tie-break toward the harder (lower-tier) decoy.
            if d < best_d or (d == best_d and best and tier < best[2]):
                best, best_d = pool[j], d
        if best is not None:
            used.add(best[0])
            pairs.append((pid, best[0], best[2], pbits, best[1]))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-benchmark", default="data/benchmark_v2",
                    help="supplies conditioning sets (prompts) and instance list")
    ap.add_argument("--profile-hits", default=None,
                    help="default: <source-benchmark>/profile_hits.tsv")
    ap.add_argument("--metadata", default="data/raw/swissprot_ec_metadata.tsv")
    ap.add_argument("--experimental", default="data/raw/swissprot_ec_experimental.tsv")
    ap.add_argument("--sequences", default="data/raw/swissprot_ec.fasta")
    ap.add_argument("--out-dir", default="data/benchmark_v3")
    ap.add_argument("--caliper-frac", type=float, default=0.02,
                    help="max |Δbitscore| / bitscore between matched pair")
    ap.add_argument("--caliper-abs", type=float, default=2.0,
                    help="floor on the caliper, in bits")
    ap.add_argument("--min-pairs", type=int, default=5)
    ap.add_argument("--max-pairs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.source_benchmark)
    hits_path = Path(args.profile_hits) if args.profile_hits else src / "profile_hits.tsv"

    print("Loading annotations...")
    seq_ec, seq_partial, ec_seq = load_ec_map(args.metadata)
    exp_seq_ec, _, _ = load_ec_map(args.experimental)
    experimental = set(exp_seq_ec)
    print(f"  {len(seq_ec):,} sequences with a complete EC4; "
          f"{len(experimental):,} with experimental catalytic-activity evidence")

    all_seqs = read_fasta(args.sequences)
    hits = load_profile_hits(hits_path)
    print(f"  profile hits for {len(hits)} instances")

    out_root = Path(args.out_dir)
    manifest, skipped = [], defaultdict(int)

    for ec_dir in sorted((src / "instances").iterdir()):
        ec = ec_dir.name
        target_ec = ec.replace("_", ".")
        if ec not in hits:
            skipped["no_profile"] += 1
            continue

        cond = set(read_fasta(ec_dir / "conditioning.fasta"))
        family = ec_seq.get(target_ec, set())
        inst_hits = hits[ec]

        # Positives: experimentally-evidenced family members, held out of the prompt.
        positives = [
            (s, b) for s, (b, _) in inst_hits.items()
            if s in family and s not in cond and s in experimental
        ]

        # Negatives: hit by the profile, experimentally evidenced for a *different*
        # reaction, with no partial EC that could cover the target.
        decoys = []
        for s, (b, _) in inst_hits.items():
            if s in family or s in cond or s not in experimental:
                continue
            other = seq_ec.get(s, set())
            if not other or target_ec in other:
                continue
            if partial_conflicts(seq_partial.get(s, ()), target_ec):
                continue
            tier = min(classify_decoy_tier(target_ec, o) for o in other)
            decoys.append((s, b, tier))

        if len(positives) < args.min_pairs:
            skipped["too_few_positives"] += 1
            continue
        if len(decoys) < args.min_pairs:
            skipped["too_few_decoys"] += 1
            continue

        if len(positives) > args.max_pairs:
            positives = random.sample(positives, args.max_pairs)

        pairs = match_nearest(positives, decoys, args.caliper_frac, args.caliper_abs)
        if len(pairs) < args.min_pairs:
            skipped["too_few_matched_pairs"] += 1
            continue

        inst_out = out_root / "instances" / ec
        inst_out.mkdir(parents=True, exist_ok=True)
        write_fasta({s: all_seqs[s] for s in cond if s in all_seqs},
                    inst_out / "conditioning.fasta")

        cand, labels = {}, []
        gaps = []
        for pid, nid, tier, pb, nb in pairs:
            if pid not in all_seqs or nid not in all_seqs:
                continue
            cand[pid] = all_seqs[pid]
            cand[nid] = all_seqs[nid]
            gaps.append(nb - pb)
            for sid, lab, bits in ((pid, 1, pb), (nid, 0, nb)):
                labels.append({
                    "seq_id": sid, "label": lab, "tier": tier,
                    "paired_with": nid if lab else pid,
                    "hmm_bitscore": round(bits, 2),
                    "bitscore_gap": round(nb - pb, 2),
                })
        if len(labels) < 2 * args.min_pairs:
            skipped["missing_sequences"] += 1
            continue

        write_fasta(cand, inst_out / "candidates.fasta")
        write_tsv(labels, inst_out / "labels.tsv")

        tiers = defaultdict(int)
        for _, _, t, _, _ in pairs:
            tiers[f"tier{t}"] += 1
        manifest.append({
            "ec_number": target_ec,
            "conditioning_size": len(cond),
            "n_pairs": len(gaps),
            "n_positives_available": len(positives),
            "n_decoys_available": len(decoys),
            "median_bitscore_gap": float(np.median(gaps)),
            "mean_abs_bitscore_gap": float(np.mean(np.abs(gaps))),
            "tier_distribution": dict(tiers),
        })
        if len(manifest) % 20 == 0:
            print(f"  {len(manifest)} instances built", flush=True)

    write_json({
        "instances": manifest,
        "n_instances": len(manifest),
        "total_pairs": sum(m["n_pairs"] for m in manifest),
        "skipped": dict(skipped),
        "config": vars(args),
    }, out_root / "manifest.json")

    print(f"\n=== v3 summary ===")
    print(f"Instances: {len(manifest)}   pairs: {sum(m['n_pairs'] for m in manifest):,}")
    print(f"Skipped: {dict(skipped)}")
    if manifest:
        g = np.array([m["mean_abs_bitscore_gap"] for m in manifest])
        med = np.array([m["median_bitscore_gap"] for m in manifest])
        print(f"Mean |Δbitscore| within a pair: median across instances {np.median(g):.2f} bits")
        print(f"Median signed gap (neg - pos):  median across instances {np.median(med):+.2f} bits")
        tot = defaultdict(int)
        for m in manifest:
            for k, v in m["tier_distribution"].items():
                tot[k] += v
        print(f"Tiers: {dict(sorted(tot.items()))}")
    print(f"\nNow run: python evaluation/leakage_gate.py --benchmark-dir {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
