#!/usr/bin/env python3
"""Download EC-annotated Swiss-Prot sequences from UniProt REST API."""
import argparse
import gzip
import shutil
import time
from pathlib import Path

import requests
import yaml


UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/stream"


def download_stream(url, params, output_path, max_retries=3):
    """Download a streaming response to a file with retries."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gz_path = output_path.with_suffix(output_path.suffix + ".gz")

    for attempt in range(max_retries):
        try:
            print(f"  Downloading (attempt {attempt + 1})...")
            params_with_compress = {**params, "compressed": "true"}
            resp = requests.get(
                url, params=params_with_compress, stream=True, timeout=600
            )
            resp.raise_for_status()

            with open(gz_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

            # Decompress
            print("  Decompressing...")
            with gzip.open(gz_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
            return True

        except (requests.RequestException, gzip.BadGzipFile) as e:
            print(f"  Error: {e}")
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description="Download Swiss-Prot EC sequences")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["data"]["output_dir"]) / "raw"
    query = config["data"]["swissprot_query"]

    # Download FASTA
    fasta_path = output_dir / "swissprot_ec.fasta"
    if fasta_path.exists():
        print(f"FASTA already exists: {fasta_path}")
    else:
        print("Downloading Swiss-Prot EC-annotated sequences (FASTA)...")
        download_stream(
            UNIPROT_BASE,
            {"query": query, "format": "fasta"},
            fasta_path,
        )

    # Download TSV metadata
    tsv_path = output_dir / "swissprot_ec_metadata.tsv"
    if tsv_path.exists():
        print(f"TSV already exists: {tsv_path}")
    else:
        print("Downloading metadata (TSV)...")
        download_stream(
            UNIPROT_BASE,
            {
                "query": query,
                "format": "tsv",
                "fields": "accession,ec,organism_name,length,sequence",
            },
            tsv_path,
        )

    # Quick stats
    if fasta_path.exists():
        n_seqs = sum(1 for line in open(fasta_path) if line.startswith(">"))
        print(f"\nTotal sequences downloaded: {n_seqs:,}")

    print("Done.")


if __name__ == "__main__":
    main()
