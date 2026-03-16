"""I/O helpers for FASTA, TSV, and JSON files."""
import json
import csv
from pathlib import Path
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


def parse_uniprot_id(raw_id):
    """Extract UniProt accession from FASTA header ID.

    Handles both 'sp|P12345|NAME' format and bare 'P12345' format.
    """
    if "|" in raw_id:
        parts = raw_id.split("|")
        if len(parts) >= 2:
            return parts[1]
    return raw_id


def read_fasta(path):
    """Read FASTA file, return dict of {accession: str(sequence)}.

    Extracts UniProt accession from sp|ACC|NAME format headers.
    """
    records = {}
    for record in SeqIO.parse(path, "fasta"):
        acc = parse_uniprot_id(record.id)
        records[acc] = str(record.seq)
    return records


def write_fasta(records, path):
    """Write dict of {id: sequence} or list of (id, sequence) to FASTA."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seq_records = []
    items = records.items() if isinstance(records, dict) else records
    for seq_id, seq in items:
        seq_records.append(SeqRecord(Seq(seq), id=seq_id, description=""))
    SeqIO.write(seq_records, path, "fasta")


def read_tsv(path, has_header=True):
    """Read TSV file, return list of dicts (if header) or list of lists."""
    rows = []
    with open(path) as f:
        if has_header:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(dict(row))
        else:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                rows.append(row)
    return rows


def write_tsv(rows, path, fieldnames=None):
    """Write list of dicts to TSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(data, path):
    """Write data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_json(path):
    """Read JSON file."""
    with open(path) as f:
        return json.load(f)
