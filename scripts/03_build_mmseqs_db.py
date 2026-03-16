#!/usr/bin/env python3
"""Build mmseqs2 databases and run similarity searches.

Two-stage approach:
1. Within-EC3 all-vs-all: precise similarities for hard negative selection
2. Cross-EC3 representative search: approximate distant similarities
"""
import argparse
import csv
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml
from Bio import SeqIO

from utils.io_utils import read_tsv
from utils.family import parse_ec_annotations


def run_mmseqs(args_list, desc=""):
    """Run an mmseqs2 command."""
    cmd = ["mmseqs"] + args_list
    print(f"  Running: {' '.join(cmd[:6])}... {desc}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
        raise RuntimeError(f"mmseqs failed: {' '.join(cmd[:4])}")
    return result


def parse_uniprot_id(raw_id):
    """Extract UniProt accession from sp|ACC|NAME format."""
    if "|" in raw_id:
        parts = raw_id.split("|")
        if len(parts) >= 2:
            return parts[1]
    return raw_id


def write_subset_fasta(all_seqs, ids, output_path):
    """Write a subset of sequences to FASTA, matching by UniProt accession."""
    records = []
    id_set = set(ids)
    for record in all_seqs:
        acc = parse_uniprot_id(record.id)
        if acc in id_set:
            records.append(record)
    SeqIO.write(records, output_path, "fasta")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Build mmseqs2 databases")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stage", choices=["all", "within_ec3", "cross_ec3", "cluster"],
        default="all",
        help="Which stage to run"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["output_dir"])
    mmseqs_dir = data_dir / "processed" / "mmseqs"
    mmseqs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    fasta_path = raw_dir / "swissprot_ec.fasta"
    threads = str(config["mmseqs"]["threads"])
    sensitivity = str(config["mmseqs"]["sensitivity"])

    # Load metadata for EC grouping
    print("Loading metadata...")
    metadata = read_tsv(raw_dir / "swissprot_ec_metadata.tsv")
    seq_to_ecs, ec_to_seqs, _ = parse_ec_annotations(metadata)

    # Group by EC3
    ec3_groups = defaultdict(set)
    for ec, members in ec_to_seqs.items():
        ec3 = ".".join(ec.split(".")[:3])
        ec3_groups[ec3].update(members)

    print(f"  {len(ec3_groups)} EC3 groups")

    # Create main database
    main_db = mmseqs_dir / "swissprot_db"
    if not main_db.with_suffix(".dbtype").exists():
        print("Creating main mmseqs2 database...")
        run_mmseqs(["createdb", str(fasta_path), str(main_db)])
    else:
        print("Main database already exists.")

    # --- Stage: Cluster at conditioning identity for representative selection ---
    if args.stage in ("all", "cluster"):
        cluster_id = config["conditioning"]["cluster_identity"]
        cluster_db = mmseqs_dir / f"cluster_{int(cluster_id * 100)}"
        cluster_tsv = mmseqs_dir / f"cluster_{int(cluster_id * 100)}.tsv"

        if not cluster_tsv.exists():
            print(f"\nClustering at {cluster_id * 100:.0f}% identity...")
            with tempfile.TemporaryDirectory() as tmpdir:
                run_mmseqs([
                    "cluster", str(main_db), str(cluster_db), tmpdir,
                    "--min-seq-id", str(cluster_id),
                    "--threads", threads,
                    "-c", "0.8",
                    "--cov-mode", "0",
                ], f"({cluster_id*100:.0f}% identity)")

                run_mmseqs([
                    "createtsv", str(main_db), str(main_db),
                    str(cluster_db), str(cluster_tsv),
                ])
            print(f"  Cluster assignments: {cluster_tsv}")
        else:
            print(f"Cluster file already exists: {cluster_tsv}")

    # --- Stage: Within-EC3 all-vs-all ---
    if args.stage in ("all", "within_ec3"):
        print("\n--- Within-EC3 all-vs-all search ---")

        # Load all sequences for subsetting
        all_records = list(SeqIO.parse(fasta_path, "fasta"))
        all_ids = {parse_uniprot_id(r.id) for r in all_records}

        # Only process EC3 groups with multiple EC4 families
        # (otherwise there are no hard negatives to find)
        ec3_with_multiple_ec4 = {}
        for ec3, members in ec3_groups.items():
            ec4s_in_group = [ec for ec in ec_to_seqs if ec.startswith(ec3 + ".")]
            if len(ec4s_in_group) >= 2:
                ec3_with_multiple_ec4[ec3] = members

        print(f"  {len(ec3_with_multiple_ec4)} EC3 groups with >=2 EC4 families")

        results_dir = mmseqs_dir / "within_ec3"
        results_dir.mkdir(exist_ok=True)
        combined_results = mmseqs_dir / "within_ec3_results.tsv"

        if combined_results.exists():
            print(f"  Within-EC3 results already exist: {combined_results}")
        else:
            all_results = []
            for i, (ec3, members) in enumerate(sorted(ec3_with_multiple_ec4.items())):
                valid_members = members & all_ids
                if len(valid_members) < 2:
                    continue

                result_file = results_dir / f"{ec3.replace('.', '_')}.tsv"
                if result_file.exists():
                    continue

                print(f"  [{i+1}/{len(ec3_with_multiple_ec4)}] {ec3} "
                      f"({len(valid_members)} seqs)")

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir = Path(tmpdir)
                    subset_fasta = tmpdir / "subset.fasta"
                    n_written = write_subset_fasta(
                        all_records, valid_members, subset_fasta
                    )
                    if n_written < 2:
                        continue

                    subset_db = tmpdir / "subset_db"
                    result_db = tmpdir / "result_db"

                    run_mmseqs(["createdb", str(subset_fasta), str(subset_db)])
                    run_mmseqs([
                        "search", str(subset_db), str(subset_db),
                        str(result_db), str(tmpdir / "tmp"),
                        "-s", sensitivity,
                        "--max-seqs", str(config["mmseqs"]["max_seqs"]),
                        "--threads", threads,
                    ])
                    run_mmseqs([
                        "convertalis", str(subset_db), str(subset_db),
                        str(result_db), str(result_file),
                        "--format-output",
                        "query,target,pident,alnlen,evalue,qlen,tlen",
                    ])

            # Combine all within-EC3 results
            print("  Combining within-EC3 results...")
            with open(combined_results, "w") as out:
                out.write("query\ttarget\tpident\talnlen\tevalue\tqlen\ttlen\n")
                for rf in sorted(results_dir.glob("*.tsv")):
                    with open(rf) as f:
                        for line in f:
                            out.write(line)
            print(f"  Combined results: {combined_results}")

    # --- Stage: Cross-EC3 representative search ---
    if args.stage in ("all", "cross_ec3"):
        print("\n--- Cross-EC3 representative search ---")
        cross_id = config["mmseqs"]["cluster_identity_for_cross_ec3"]
        cross_cluster_db = mmseqs_dir / f"cluster_{int(cross_id * 100)}"
        cross_results = mmseqs_dir / "cross_ec3_results.tsv"

        if cross_results.exists():
            print(f"  Cross-EC3 results already exist: {cross_results}")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Cluster at 50% for representatives
                if not cross_cluster_db.with_suffix(".dbtype").exists():
                    print(f"  Clustering at {cross_id*100:.0f}% for representatives...")
                    run_mmseqs([
                        "cluster", str(main_db), str(cross_cluster_db), tmpdir,
                        "--min-seq-id", str(cross_id),
                        "--threads", threads,
                        "-c", "0.8",
                    ])

                # Extract representative sequences
                rep_db = mmseqs_dir / "rep_db"
                run_mmseqs([
                    "createsubdb", str(cross_cluster_db), str(main_db),
                    str(rep_db),
                ])

                # All-vs-all on representatives
                rep_result_db = mmseqs_dir / "rep_result_db"
                run_mmseqs([
                    "search", str(rep_db), str(rep_db),
                    str(rep_result_db), tmpdir,
                    "-s", sensitivity,
                    "--max-seqs", str(config["mmseqs"]["max_seqs"]),
                    "--threads", threads,
                ])

                run_mmseqs([
                    "convertalis", str(rep_db), str(rep_db),
                    str(rep_result_db), str(cross_results),
                    "--format-output",
                    "query,target,pident,alnlen,evalue,qlen,tlen",
                ])

            print(f"  Cross-EC3 results: {cross_results}")

    print("\nDone.")


if __name__ == "__main__":
    main()
