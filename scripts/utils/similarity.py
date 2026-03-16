"""Similarity binning, matching, and false-negative filtering logic."""
import random
from collections import defaultdict


def get_sim_bin(pident, bin_edges):
    """Return the bin index for a given percent identity.

    bin_edges = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    A pident of 35.2 with these edges falls in bin index 1 (range [30, 40)).
    Values below the first edge go to bin -1 (out of range).
    Values in [90, 100] go to the last bin.
    """
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= pident < bin_edges[i + 1]:
            return i
    if pident >= bin_edges[-1]:
        return len(bin_edges) - 2
    return -1


def bin_label(bin_idx, bin_edges):
    """Return human-readable label for a bin index."""
    if bin_idx < 0:
        return f"<{bin_edges[0]}"
    lo = bin_edges[bin_idx]
    hi = bin_edges[bin_idx + 1]
    return f"{lo}-{hi}"


def compute_max_pident_to_set(query_id, target_set_ids, sim_matrix):
    """Compute max percent identity of query to any member of target_set.

    sim_matrix: dict of {query_id: {target_id: pident}}
    """
    if query_id not in sim_matrix:
        return 0.0
    hits = sim_matrix[query_id]
    max_pid = 0.0
    for tid in target_set_ids:
        if tid in hits:
            max_pid = max(max_pid, hits[tid])
    return max_pid


def build_sim_lookup(sim_rows, target_ids=None):
    """Build similarity lookup dict from mmseqs result rows.

    Returns: {query_id: {target_id: pident}}
    If target_ids is provided, only store hits to those targets.
    """
    lookup = defaultdict(dict)
    target_set = set(target_ids) if target_ids else None
    for row in sim_rows:
        q, t = row[0], row[1]
        pid = float(row[2])
        if target_set is None or t in target_set:
            if t not in lookup[q] or pid > lookup[q][t]:
                lookup[q][t] = pid
        if target_set is None or q in target_set:
            if q not in lookup[t] or pid > lookup[t][q]:
                lookup[t][q] = pid
    return dict(lookup)


def select_matched_decoys(
    positives_with_sim,
    decoy_pool_with_sim,
    bin_edges,
    exclude_ids=None,
):
    """Select decoys matched by similarity bin to positives.

    Args:
        positives_with_sim: list of (seq_id, max_pident_to_conditioning)
        decoy_pool_with_sim: list of (seq_id, max_pident_to_conditioning, tier)
            tier: 1=same EC3, 2=same EC2, 3=same EC1, 4=different EC1
        bin_edges: similarity bin edges
        exclude_ids: set of seq_ids to exclude (false negative protection)

    Returns:
        list of (positive_id, negative_id, sim_bin, tier, pos_pident, neg_pident)
    """
    exclude_ids = exclude_ids or set()

    # Group decoys by (bin, tier)
    decoy_by_bin_tier = defaultdict(list)
    for seq_id, pid, tier in decoy_pool_with_sim:
        if seq_id in exclude_ids:
            continue
        b = get_sim_bin(pid, bin_edges)
        if b < 0:
            continue
        decoy_by_bin_tier[(b, tier)].append((seq_id, pid))

    # Shuffle decoy pools for random selection
    for key in decoy_by_bin_tier:
        random.shuffle(decoy_by_bin_tier[key])

    # Track used decoys
    used_decoys = set()
    pairs = []

    for pos_id, pos_pid in positives_with_sim:
        pos_bin = get_sim_bin(pos_pid, bin_edges)
        if pos_bin < 0:
            continue

        matched = False
        # Try tiers in order of preference (hardest first)
        for tier in [1, 2, 3, 4]:
            pool = decoy_by_bin_tier.get((pos_bin, tier), [])
            best_decoy = None
            best_dist = float("inf")
            best_idx = -1

            for idx, (dec_id, dec_pid) in enumerate(pool):
                if dec_id in used_decoys:
                    continue
                dist = abs(dec_pid - pos_pid)
                if dist < best_dist:
                    best_dist = dist
                    best_decoy = (dec_id, dec_pid)
                    best_idx = idx

            if best_decoy is not None:
                dec_id, dec_pid = best_decoy
                used_decoys.add(dec_id)
                bl = bin_label(pos_bin, bin_edges)
                pairs.append((pos_id, dec_id, bl, tier, pos_pid, dec_pid))
                matched = True
                break

        if not matched:
            # Skip this positive if no matching decoy found
            pass

    return pairs


def find_false_negative_ids(
    decoy_candidate_ids,
    target_family_ids,
    sim_matrix,
    pident_threshold,
    promiscuous_map=None,
    target_ec=None,
):
    """Identify decoy candidates that are likely false negatives.

    Returns set of seq_ids to exclude from decoy pool.
    """
    exclude = set()
    target_set = set(target_family_ids)

    for dec_id in decoy_candidate_ids:
        # Check sequence similarity
        if dec_id in sim_matrix:
            for tid in target_set:
                if tid in sim_matrix[dec_id]:
                    if sim_matrix[dec_id][tid] >= pident_threshold * 100:
                        exclude.add(dec_id)
                        break

        # Check promiscuous annotations
        if promiscuous_map and target_ec and dec_id in promiscuous_map:
            if target_ec in promiscuous_map[dec_id]:
                exclude.add(dec_id)

    return exclude
