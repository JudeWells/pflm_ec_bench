#!/usr/bin/env python3
"""Build a profile HMM per benchmark instance and score a target database with it.

This produces the homology baseline the benchmark should actually be measured
against, and the statistic v3 should match negatives to positives on.

Why a profile rather than pairwise identity: a profile is defined against the
conditioning set *as a whole* rather than a single best hit, and its bitscore is
coverage-sensitive by construction. Local percent identity is neither, which is
why matching on it leaves a coverage leak (see README §1).

Pipeline, per instance:

    conditioning.fasta --mafft--> MSA --hmmbuild--> HMM

All instance HMMs are concatenated into one file and the target database is
scanned once with `hmmsearch`, which is far cheaper than one process per family.

Output is a long TSV: ec, seq_id, bitscore, evalue, dom_bitscore, dom_evalue.
Conditioning-set members are retained in the output (marked via `is_conditioning`)
so downstream code can decide whether to exclude them.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).parent))
from utils.io_utils import read_fasta, parse_uniprot_id


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.stderr.write(f"FAILED: {' '.join(map(str, cmd))}\n{r.stderr[:2000]}\n")
        raise RuntimeError(cmd[0])
    return r


def normalise_target_fasta(src, dst):
    """Rewrite `sp|ACC|NAME ...` headers to bare accessions so hmmsearch target
    names match the accessions used in labels.tsv."""
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dst, "w") as out:
        for rec in SeqIO.parse(src, "fasta"):
            out.write(f">{parse_uniprot_id(rec.id)}\n{rec.seq}\n")
            n += 1
    print(f"  normalised {n:,} target sequences -> {dst.name}")
    return dst


def build_hmm(ec, cond_fasta, workdir, threads):
    """mafft (FFT-NS-2) | hmmbuild. Returns path to the HMM, or None if unusable.

    FFT-NS-2 rather than `--auto`: `--auto` switches to iterative refinement for
    some instances, which on a 100-sequence prompt of long helicases (`3.6.4.13`)
    consumed >5 GB and did not converge in minutes. It also means different
    instances would be aligned by different algorithms, making their bitscores
    incomparable. Progressive alignment is ample for building a profile.
    """
    seqs = read_fasta(cond_fasta)
    if len(seqs) < 2:
        return None  # hmmbuild needs an alignment; single-seq prompts need phmmer

    aln = workdir / f"{ec}.afa"
    hmm = workdir / f"{ec}.hmm"
    if hmm.exists() and hmm.stat().st_size > 0:
        return hmm
    with open(aln, "w") as f:
        r = run(["mafft", "--retree", "2", "--maxiterate", "0",
                 "--anysymbol", "--quiet",
                 "--thread", str(threads), str(cond_fasta)])
        f.write(r.stdout)
    # `-n` names the HMM, which becomes the query name in the tblout, letting a
    # single hmmsearch over a concatenated HMM file be demultiplexed by instance.
    run(["hmmbuild", "--amino", "--informat", "afa",
         "-n", ec, "--cpu", str(threads), str(hmm), str(aln)])
    return hmm


def parse_tblout(path):
    """Yield (query_hmm, target_seq, bitscore, evalue, dom_bits, dom_evalue)."""
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 9:
            continue
        # target, tacc, query, qacc, full_E, full_score, full_bias, dom_E, dom_score
        yield f[2], f[0], float(f[5]), float(f[4]), float(f[8]), float(f[7])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", required=True)
    ap.add_argument("--target-fasta", default="data/raw/swissprot_ec.fasta",
                    help="database to score; defaults to the full Swiss-Prot EC set "
                         "so the same run supplies both matched-pair scores and a "
                         "full decoy pool for v3 construction")
    ap.add_argument("--out", default=None, help="output TSV (default: <bench>/profile_hits.tsv)")
    ap.add_argument("--evalue", type=float, default=10.0,
                    help="hmmsearch reporting threshold; keep permissive so that "
                         "weak hits are recorded rather than silently coerced to 0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only N instances (prototyping)")
    ap.add_argument("--keep-hmms", default=None, help="directory to retain built HMMs")
    args = ap.parse_args()

    bdir = Path(args.benchmark_dir)
    out_path = Path(args.out) if args.out else bdir / "profile_hits.tsv"
    inst_dirs = sorted((bdir / "instances").iterdir())
    if args.limit:
        inst_dirs = inst_dirs[:args.limit]

    target = normalise_target_fasta(Path(args.target_fasta),
                                    Path("data/processed/target_acc.fasta"))

    workroot = Path(args.keep_hmms) if args.keep_hmms else None
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(workroot) if workroot else Path(tmp)
        wd.mkdir(parents=True, exist_ok=True)

        hmms, cond_members, skipped = [], {}, []
        for i, d in enumerate(inst_dirs):
            ec = d.name
            cond_fa = d / "conditioning.fasta"
            if not cond_fa.exists():
                continue
            cond_members[ec] = set(read_fasta(cond_fa))
            hmm = build_hmm(ec, cond_fa, wd, args.threads)
            if hmm is None:
                skipped.append(ec)
                continue
            hmms.append(hmm)
            if (i + 1) % 20 == 0:
                print(f"  built {len(hmms)} HMMs ({i+1}/{len(inst_dirs)})", flush=True)

        if not hmms:
            print("No HMMs built.")
            return 1
        print(f"Built {len(hmms)} HMMs" + (f", skipped {len(skipped)} (<2 prompt seqs)"
                                           if skipped else ""))

        combined = wd / "all.hmm"
        with open(combined, "w") as out:
            for h in hmms:
                out.write(open(h).read())

        tbl = wd / "hits.tbl"
        print(f"hmmsearch: {len(hmms)} profiles vs {target} (E<={args.evalue})...", flush=True)
        run(["hmmsearch", "--tblout", str(tbl), "-E", str(args.evalue),
             "--cpu", str(args.threads), "-o", "/dev/null",
             str(combined), str(target)])

        n = 0
        with open(out_path, "w") as f:
            f.write("ec\tseq_id\tbitscore\tevalue\tdom_bitscore\tdom_evalue\tis_conditioning\n")
            for ec, sid, bits, ev, dbits, dev in parse_tblout(tbl):
                isc = int(sid in cond_members.get(ec, ()))
                f.write(f"{ec}\t{sid}\t{bits}\t{ev}\t{dbits}\t{dev}\t{isc}\n")
                n += 1

    print(f"Wrote {n:,} hits -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
