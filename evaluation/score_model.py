#!/usr/bin/env python3
"""Generic scorer interface for benchmark evaluation.

Usage:
    python score_model.py --model poet --benchmark-dir data/benchmark --output-dir results/poet

Each model scorer should implement:
    score(conditioning_fasta, candidates_fasta) -> dict of {seq_id: float}
"""
import argparse
import importlib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils.io_utils import read_json, write_tsv


def main():
    parser = argparse.ArgumentParser(description="Score benchmark with a model")
    parser.add_argument("--model", required=True,
                        choices=["poet", "profam", "msa_transformer"],
                        help="Model to use for scoring")
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    manifest = read_json(benchmark_dir / "manifest.json")

    # Import the scorer
    scorer_module = importlib.import_module(f"scorers.{args.model}")

    print(f"Scoring {manifest['n_instances']} instances with {args.model}...")

    for i, entry in enumerate(manifest["instances"]):
        ec = entry["ec_number"]
        ec_dir = ec.replace(".", "_")
        instance_dir = benchmark_dir / "instances" / ec_dir

        cond_fasta = instance_dir / "conditioning.fasta"
        cand_fasta = instance_dir / "candidates.fasta"
        out_dir = output_dir / ec_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        scores_path = out_dir / "scores.tsv"

        if scores_path.exists():
            continue

        print(f"  [{i+1}/{manifest['n_instances']}] {ec}")
        scores = scorer_module.score(cond_fasta, cand_fasta)

        rows = [{"seq_id": sid, "score": f"{s:.6f}"} for sid, s in scores.items()]
        write_tsv(rows, scores_path)

    print("Done.")


if __name__ == "__main__":
    main()
