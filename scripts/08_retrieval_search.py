#!/usr/bin/env python3
"""Search every conditioning set against all of Swiss-Prot, with coverage fields.

This is the *naive sequence-only retrieval* a practitioner would run to screen a
database for members of a known family: take the known members, search them
against the database, keep hits that align over most of their length, rank by
percent identity.

The existing MMseqs2 output (`03_build_mmseqs_db.py`) cannot support this. It is
all-vs-all only *within* an EC3 group, and cross-EC3 similarities exist only
between 50%-identity cluster representatives, so identity to the conditioning set
is systematically under-measured for distant sequences and silently absent for
most (README §6). This script measures every (conditioning member, database
sequence) pair uniformly, and emits `qcov`/`tcov` so that a coverage floor can be
applied — the fields `convertalis` was never asked for in step 03.

Output: data/processed/retrieval/cond_vs_all.tsv
    query target pident alnlen evalue bits qcov tcov qlen tlen
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).parent))
from utils.io_utils import parse_uniprot_id


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"FAILED: {' '.join(map(str, cmd))}\n{r.stderr[:2000]}\n")
        raise RuntimeError(cmd[0])
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", default="data/benchmark_v2",
                    help="source of conditioning sets")
    ap.add_argument("--target-db", default="data/processed/mmseqs/swissprot_db")
    ap.add_argument("--out", default="data/processed/retrieval/cond_vs_all.tsv")
    ap.add_argument("--sensitivity", default="7.5")
    ap.add_argument("--max-seqs", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"Already exists: {out}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)

    # Collect the union of all conditioning sequences (they repeat across instances
    # only rarely, but dedupe anyway — one search row serves every instance that
    # contains the query).
    recs = {}
    for d in sorted((Path(args.benchmark_dir) / "instances").iterdir()):
        fa = d / "conditioning.fasta"
        if not fa.exists():
            continue
        for r in SeqIO.parse(fa, "fasta"):
            recs[parse_uniprot_id(r.id)] = r
    print(f"{len(recs):,} unique conditioning sequences")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        qfa = tmp / "cond.fasta"
        with open(qfa, "w") as f:
            for acc, r in recs.items():
                f.write(f">{acc}\n{r.seq}\n")

        qdb, res = tmp / "qdb", tmp / "res"
        run(["mmseqs", "createdb", qfa, qdb])
        # -c 0.0: do not filter by coverage here. The floor is applied downstream so
        # that it can be swept without re-running this search.
        print("mmseqs search (this is the slow step)...", flush=True)
        run(["mmseqs", "search", qdb, args.target_db, res, tmp / "t",
             "-s", args.sensitivity, "--max-seqs", args.max_seqs,
             "-c", "0.0", "-e", "10", "--threads", args.threads])
        run(["mmseqs", "convertalis", qdb, args.target_db, res, out,
             "--format-output",
             "query,target,pident,alnlen,evalue,bits,qcov,tcov,qlen,tlen",
             "--threads", args.threads])

    n = sum(1 for _ in open(out))
    print(f"Wrote {n:,} alignments -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
