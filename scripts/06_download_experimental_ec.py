#!/usr/bin/env python3
"""Download the subset of Swiss-Prot EC entries with *experimental* evidence.

README §7: most Swiss-Prot EC numbers are propagated by sequence similarity
(ECO:0000250 / ECO:0000256), not measured. Using them as ground truth for a
benchmark that tests whether a model exceeds homology is circular — the labels
were themselves assigned by homology.

`cc_catalytic_activity_exp` restricts to entries whose catalytic-activity
annotation carries experimental evidence. As of writing:

    (reviewed:true) AND (ec:*)                                    280,036
    (reviewed:true) AND (ec:*) AND (cc_catalytic_activity:*)      254,396
    (reviewed:true) AND (ec:*) AND (cc_catalytic_activity_exp:*)   48,496

Only *candidates* need experimental labels. The conditioning set is a prompt,
not a test item, so exemplars may still be drawn from the full reviewed set.
"""
import argparse
import csv
import io
import re
import sys
from pathlib import Path

import requests

SEARCH = "https://rest.uniprot.org/uniprotkb/search"
QUERY = "(reviewed:true) AND (ec:*) AND (cc_catalytic_activity_exp:*)"


def paginate(query, fields, page_size=500):
    """The /stream endpoint intermittently 500s on this query; paginate /search."""
    url, params, rows, page = SEARCH, {
        "query": query, "format": "tsv", "fields": fields, "size": page_size
    }, [], 0
    while url:
        r = requests.get(url, params=params if page == 0 else None, timeout=180)
        r.raise_for_status()
        rows.extend(csv.DictReader(io.StringIO(r.text), delimiter="\t"))
        page += 1
        m = re.search(r'<(.+?)>; rel="next"', r.headers.get("Link", ""))
        url = m.group(1) if m else None
        if page % 20 == 0:
            print(f"  page {page}, {len(rows):,} rows", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/swissprot_ec_experimental.tsv")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"Already exists: {out}")
        return 0

    print(f"Querying UniProt: {QUERY}")
    rows = paginate(QUERY, "accession,ec,xref_pfam")
    print(f"  {len(rows):,} entries")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Entry", "EC number", "Pfam"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
