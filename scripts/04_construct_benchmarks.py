#!/usr/bin/env python3
"""Construct benchmark instances: conditioning sets + similarity-matched candidate pairs.

For each selected EC4 family:
1. Build conditioning set from cluster representatives
2. Select positive candidates (held-out family members)
3. Select negative candidates matched by max sequence identity to conditioning set
4. Apply false-negative protection filters
5. Write benchmark instance files
"""
import argparse
import csv
import pickle
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from utils.io_utils import read_fasta, write_fasta, read_tsv, write_tsv, write_json
from utils.family import (
    parse_ec_annotations,
    filter_families,
    select_conditioning_set,
    classify_decoy_tier,
)
from utils.similarity import (
    get_sim_bin,
    bin_label,
    select_matched_decoys,
    find_false_negative_ids,
)


def load_cluster_assignments(cluster_tsv):
    """Load mmseqs cluster assignments: {member: representative}."""
    assignments = {}
    with open(cluster_tsv) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                rep, member = row[0], row[1]
                assignments[member] = rep
    return assignments


def load_similarity_results(results_path):
    """Load mmseqs similarity results into lookup dict.

    Uses pandas C parser for fast I/O and caches the parsed dict as pickle
    so subsequent runs load in seconds instead of re-parsing the TSV.

    Returns: {query: {target: pident}}
    """
    results_path = Path(results_path)
    cache_path = results_path.with_suffix(".sim.pkl")

    if (cache_path.exists()
            and cache_path.stat().st_mtime >= results_path.stat().st_mtime):
        print(f"    Loading cached: {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with open(results_path) as f:
        has_header = f.readline().startswith("query")

    df = pd.read_csv(
        results_path, sep="\t", usecols=[0, 1, 2],
        header=0 if has_header else None, engine="c",
    )
    df.columns = ["query", "target", "pident"]

    sim = defaultdict(dict)
    qs = df["query"].values
    ts = df["target"].values
    pids = df["pident"].values
    del df

    for i in range(len(qs)):
        q, t, pid = qs[i], ts[i], pids[i]
        qd = sim[q]
        if t not in qd or pid > qd[t]:
            qd[t] = pid
        td = sim[t]
        if q not in td or pid > td[q]:
            td[q] = pid

    result = dict(sim)
    del sim

    print(f"    Caching to {cache_path.name}")
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    return result


def compute_max_sim_to_conditioning(seq_id, conditioning_ids, sim_matrix):
    """Get max percent identity of seq_id to any conditioning set member."""
    if seq_id not in sim_matrix:
        return 0.0
    hits = sim_matrix[seq_id]
    max_pid = 0.0
    for cid in conditioning_ids:
        if cid in hits:
            max_pid = max(max_pid, hits[cid])
    return max_pid


def main():
    parser = argparse.ArgumentParser(description="Construct benchmark instances")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-subdir", default="benchmark",
        help="Subdirectory under the data dir to write instances/manifest into "
             "(e.g. 'benchmark_v2' to build a new version without overwriting).",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["output_dir"])
    bin_edges = config["candidates"]["sim_bins"]
    max_pairs = config["candidates"]["max_pairs_per_instance"]
    min_pairs = config["candidates"]["min_pairs_per_instance"]
    max_per_bin = config["candidates"]["max_per_bin"]
    prefer_higher = config["candidates"].get("negative_higher", False)
    max_pair_gap = config["candidates"].get("max_pair_gap", None)
    fn_threshold = config["false_negative_protection"]["exclude_pident_threshold"]

    if prefer_higher:
        print(f"  Matching mode: negative slightly HIGHER than positive "
              f"(max_pair_gap={max_pair_gap})")
    else:
        print("  Matching mode: closest identity within same bin")

    # Load data
    print("Loading data...")
    metadata = read_tsv(data_dir / "raw" / "swissprot_ec_metadata.tsv")
    seq_to_ecs, ec_to_seqs, promiscuous = parse_ec_annotations(metadata)

    all_seqs = read_fasta(data_dir / "raw" / "swissprot_ec.fasta")
    print(f"  {len(all_seqs):,} sequences loaded")

    selected_families = filter_families(ec_to_seqs, promiscuous, config)
    print(f"  {len(selected_families):,} selected families")

    # Load cluster assignments
    cluster_id = int(config["conditioning"]["cluster_identity"] * 100)
    cluster_tsv = data_dir / "processed" / "mmseqs" / f"cluster_{cluster_id}.tsv"
    print(f"Loading cluster assignments from {cluster_tsv}...")
    cluster_assignments = load_cluster_assignments(cluster_tsv)

    # Load similarity matrices
    within_ec3_path = data_dir / "processed" / "mmseqs" / "within_ec3_results.tsv"
    cross_ec3_path = data_dir / "processed" / "mmseqs" / "cross_ec3_results.tsv"

    print("Loading similarity results...")
    sim_matrix = {}
    if within_ec3_path.exists():
        sim_matrix = load_similarity_results(within_ec3_path)
        print(f"  Within-EC3: {len(sim_matrix):,} query entries")

    # Merge cross-EC3 results (lower priority, fill gaps)
    if cross_ec3_path.exists():
        cross_sim = load_similarity_results(cross_ec3_path)
        for q, targets in cross_sim.items():
            if q not in sim_matrix:
                sim_matrix[q] = targets
            else:
                for t, pid in targets.items():
                    if t not in sim_matrix[q] or pid > sim_matrix[q][t]:
                        sim_matrix[q][t] = pid
        print(f"  After merging cross-EC3: {len(sim_matrix):,} query entries")

    # Build promiscuous map for false-negative filtering
    promiscuous_map = {}
    if config["false_negative_protection"]["exclude_promiscuous"]:
        promiscuous_map = promiscuous

    # Process each family
    output_root = data_dir / args.output_subdir
    benchmark_dir = output_root / "instances"
    manifest_entries = []
    skipped = {"too_few_positives": 0, "too_few_pairs": 0, "no_decoys": 0}

    print(f"\nConstructing benchmark instances...")
    for family_idx, (target_ec, family_members) in enumerate(
        sorted(selected_families.items())
    ):
        # Only keep members that are in our sequence database
        family_members = family_members & set(all_seqs.keys())
        if len(family_members) < config["families"]["min_family_size"]:
            continue

        # 1. Build conditioning set
        conditioning_ids, remaining_ids = select_conditioning_set(
            family_members,
            cluster_assignments,
            config["conditioning"]["max_size"],
        )

        if len(remaining_ids) < min_pairs:
            skipped["too_few_positives"] += 1
            continue

        # 2. Compute max similarity of remaining members to conditioning set
        positives_with_sim = []
        for sid in remaining_ids:
            max_sim = compute_max_sim_to_conditioning(
                sid, conditioning_ids, sim_matrix
            )
            positives_with_sim.append((sid, max_sim))

        # Sample positives per bin
        pos_by_bin = defaultdict(list)
        for sid, sim in positives_with_sim:
            b = get_sim_bin(sim, bin_edges)
            if b >= 0:
                pos_by_bin[b].append((sid, sim))

        sampled_positives = []
        for b in sorted(pos_by_bin.keys()):
            pool = pos_by_bin[b]
            random.shuffle(pool)
            sampled_positives.extend(pool[:max_per_bin])

        if len(sampled_positives) < min_pairs:
            skipped["too_few_positives"] += 1
            continue

        # Cap total positives
        if len(sampled_positives) > max_pairs:
            random.shuffle(sampled_positives)
            sampled_positives = sampled_positives[:max_pairs]

        # 3. Collect decoy candidates from other EC families
        ec3_prefix = ".".join(target_ec.split(".")[:3])
        decoy_pool_with_sim = []

        for other_ec, other_members in ec_to_seqs.items():
            if other_ec == target_ec:
                continue
            tier = classify_decoy_tier(target_ec, other_ec)

            for sid in other_members:
                if sid not in all_seqs:
                    continue
                max_sim = compute_max_sim_to_conditioning(
                    sid, conditioning_ids, sim_matrix
                )
                decoy_pool_with_sim.append((sid, max_sim, tier))

        if not decoy_pool_with_sim:
            skipped["no_decoys"] += 1
            continue

        # 4. False-negative protection
        decoy_ids = {d[0] for d in decoy_pool_with_sim}
        false_neg_ids = find_false_negative_ids(
            decoy_ids,
            family_members,
            sim_matrix,
            fn_threshold,
            promiscuous_map,
            target_ec,
        )

        # 5. Select matched pairs
        pairs = select_matched_decoys(
            sampled_positives,
            decoy_pool_with_sim,
            bin_edges,
            exclude_ids=false_neg_ids,
            prefer_higher=prefer_higher,
            max_gap=max_pair_gap,
        )

        if len(pairs) < min_pairs:
            skipped["too_few_pairs"] += 1
            continue

        # 6. Write benchmark instance
        ec_dir_name = target_ec.replace(".", "_")
        instance_dir = benchmark_dir / ec_dir_name
        instance_dir.mkdir(parents=True, exist_ok=True)

        # Conditioning FASTA
        cond_seqs = {sid: all_seqs[sid] for sid in conditioning_ids}
        write_fasta(cond_seqs, instance_dir / "conditioning.fasta")

        # Candidates FASTA and labels
        candidate_seqs = {}
        labels = []
        for pos_id, neg_id, sim_bin_label, tier, pos_pid, neg_pid in pairs:
            candidate_seqs[pos_id] = all_seqs[pos_id]
            candidate_seqs[neg_id] = all_seqs[neg_id]
            labels.append({
                "seq_id": pos_id,
                "label": 1,
                "sim_bin": sim_bin_label,
                "tier": tier,
                "paired_with": neg_id,
                "max_pident_to_conditioning": round(pos_pid, 1),
            })
            labels.append({
                "seq_id": neg_id,
                "label": 0,
                "sim_bin": sim_bin_label,
                "tier": tier,
                "paired_with": pos_id,
                "max_pident_to_conditioning": round(neg_pid, 1),
            })

        write_fasta(candidate_seqs, instance_dir / "candidates.fasta")
        write_tsv(labels, instance_dir / "labels.tsv")

        # Collect tier distribution
        tier_dist = defaultdict(int)
        bin_dist = defaultdict(int)
        for _, _, sb, t, _, _ in pairs:
            tier_dist[f"tier{t}"] += 1
            bin_dist[sb] += 1

        manifest_entries.append({
            "ec_number": target_ec,
            "conditioning_size": len(conditioning_ids),
            "n_pairs": len(pairs),
            "n_candidates": len(candidate_seqs),
            "sim_bins": dict(bin_dist),
            "tier_distribution": dict(tier_dist),
            "family_size": len(family_members),
            "n_excluded_false_neg": len(false_neg_ids & decoy_ids),
        })

        if (family_idx + 1) % 50 == 0:
            print(f"  Processed {family_idx + 1} families, "
                  f"{len(manifest_entries)} instances created")

    # Write manifest
    manifest = {
        "instances": manifest_entries,
        "config": config,
        "n_instances": len(manifest_entries),
        "total_pairs": sum(e["n_pairs"] for e in manifest_entries),
        "skipped": skipped,
    }
    write_json(manifest, output_root / "manifest.json")

    print(f"\n=== Summary ===")
    print(f"Benchmark instances created: {len(manifest_entries)}")
    print(f"Total candidate pairs: {manifest['total_pairs']:,}")
    print(f"Skipped families: {skipped}")
    print("Done.")


if __name__ == "__main__":
    main()
