#!/usr/bin/env python3
"""Filter and select EC4 families meeting size and quality criteria."""
import argparse
from pathlib import Path

import yaml

from utils.io_utils import read_tsv, write_tsv
from utils.family import parse_ec_annotations, filter_families


def main():
    parser = argparse.ArgumentParser(description="Filter EC4 families")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["output_dir"])
    metadata_path = data_dir / "raw" / "swissprot_ec_metadata.tsv"

    print("Loading metadata...")
    metadata = read_tsv(metadata_path)
    print(f"  {len(metadata):,} protein entries loaded")

    print("Parsing EC annotations...")
    seq_to_ecs, ec_to_seqs, promiscuous = parse_ec_annotations(metadata)
    print(f"  {len(ec_to_seqs):,} unique EC4 numbers found")
    print(f"  {len(promiscuous):,} promiscuous proteins (multiple EC annotations)")

    print("Filtering families...")
    selected = filter_families(ec_to_seqs, promiscuous, config)
    print(f"  {len(selected):,} families pass filters")

    # Write summary
    out_path = data_dir / "processed" / "ec4_families.tsv"
    rows = []
    for ec in sorted(selected.keys()):
        members = selected[ec]
        n_prom = sum(1 for m in members if m in promiscuous)
        rows.append({
            "ec_number": ec,
            "family_size": len(members),
            "n_promiscuous": n_prom,
            "promiscuous_fraction": round(n_prom / len(members), 3),
        })

    write_tsv(rows, out_path)
    print(f"  Written to {out_path}")

    # Stats
    sizes = [r["family_size"] for r in rows]
    if sizes:
        sizes_int = [int(s) for s in sizes]
        print(f"\nFamily size stats:")
        print(f"  Min: {min(sizes_int)}, Max: {max(sizes_int)}, "
              f"Median: {sorted(sizes_int)[len(sizes_int)//2]}")

    print("Done.")


if __name__ == "__main__":
    main()
