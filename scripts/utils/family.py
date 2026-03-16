"""EC family selection and conditioning set construction."""
import random
from collections import defaultdict


def parse_ec_annotations(metadata_rows):
    """Parse metadata TSV rows into family mappings.

    Returns:
        seq_to_ecs: {accession: set of EC numbers}
        ec_to_seqs: {ec_number: set of accessions}
        promiscuous: {accession: set of EC numbers} (only multi-EC proteins)
    """
    seq_to_ecs = defaultdict(set)
    ec_to_seqs = defaultdict(set)

    for row in metadata_rows:
        acc = row.get("accession", row.get("Entry", ""))
        ec_field = row.get("ec", row.get("EC number", ""))
        if not ec_field:
            continue
        # EC field may contain multiple EC numbers separated by '; '
        ecs = [e.strip() for e in ec_field.split(";") if e.strip()]
        for ec in ecs:
            # Only keep fully specified EC4 numbers (no wildcards)
            parts = ec.split(".")
            if len(parts) == 4 and all(p != "-" and p != "n" for p in parts):
                seq_to_ecs[acc].add(ec)
                ec_to_seqs[ec].add(acc)

    promiscuous = {acc: ecs for acc, ecs in seq_to_ecs.items() if len(ecs) > 1}

    return dict(seq_to_ecs), dict(ec_to_seqs), promiscuous


def filter_families(ec_to_seqs, promiscuous, config):
    """Filter EC4 families by size and promiscuity criteria.

    Returns dict of {ec_number: set of accessions} for selected families.
    """
    min_size = config["families"]["min_family_size"]
    max_size = config["families"]["max_family_size"]
    max_prom_frac = config["families"]["max_promiscuous_fraction"]

    selected = {}
    for ec, members in ec_to_seqs.items():
        size = len(members)
        if size < min_size or size > max_size:
            continue
        n_prom = sum(1 for m in members if m in promiscuous)
        if n_prom / size > max_prom_frac:
            continue
        selected[ec] = members

    return selected


def get_ec_hierarchy(ec_number):
    """Return (ec1, ec2, ec3, ec4) from a full EC number string."""
    parts = ec_number.split(".")
    return (
        parts[0],
        ".".join(parts[:2]),
        ".".join(parts[:3]),
        ec_number,
    )


def classify_decoy_tier(target_ec, decoy_ec):
    """Classify the relationship tier between two EC numbers.

    Returns:
        1: same EC3, different EC4 (hardest)
        2: same EC2, different EC3
        3: same EC1, different EC2
        4: different EC1 (easiest)
    """
    t1, t2, t3, t4 = get_ec_hierarchy(target_ec)
    d1, d2, d3, d4 = get_ec_hierarchy(decoy_ec)

    if t3 == d3 and t4 != d4:
        return 1
    if t2 == d2 and t3 != d3:
        return 2
    if t1 == d1 and t2 != d2:
        return 3
    return 4


def select_conditioning_set(family_members, cluster_assignments, max_size):
    """Select conditioning set from cluster representatives.

    Args:
        family_members: set of accessions in the family
        cluster_assignments: dict {member_id: representative_id}
        max_size: maximum conditioning set size

    Returns:
        conditioning_ids: set of selected accession IDs
        remaining_ids: set of accessions not in conditioning set
    """
    # Find unique cluster representatives within this family
    reps = set()
    for member in family_members:
        rep = cluster_assignments.get(member, member)
        if rep in family_members:
            reps.add(rep)

    # If we have more representatives than max_size, subsample
    if len(reps) > max_size:
        reps = set(random.sample(sorted(reps), max_size))

    # If very few clusters, add some non-representative members
    if len(reps) < 3 and len(family_members) >= 3:
        remaining = family_members - reps
        extra = min(max_size - len(reps), len(remaining))
        reps = reps | set(random.sample(sorted(remaining), extra))

    remaining = family_members - reps
    return reps, remaining
