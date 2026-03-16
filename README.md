# pflm_ec_bench

A benchmark for evaluating **conditional protein family language models** (PoET, ProFam, MSA Transformer, etc.) on enzyme function classification, with controls for sequence similarity.

## Motivation

Conditional protein language models assign likelihood scores to a query protein *conditioned on* a set of related sequences (a "prompt"). A natural application is predicting whether a candidate enzyme belongs to a given functional family: condition the model on known family members, then check whether the candidate receives a high likelihood.

The problem is that conditional likelihood correlates strongly with raw sequence identity between the candidate and the prompt. A model that merely measures homology—without learning anything about function—could appear to perform well on a naive benchmark. **This benchmark is designed to make that shortcut impossible.**

We construct a binary classification task—*in-family vs. out-of-family*—where the distribution of maximum sequence identity between candidates and the conditioning set is **identical** for positive and negative examples. Any model that outperforms chance must be capturing something beyond raw homology.

## Task definition

For each benchmark instance (one per EC4 enzyme family):

| Component | Description | Size |
|---|---|---|
| **Conditioning set** | Known members of the target family, used as the model's "prompt" | 1–100 sequences |
| **Positive candidates** | Held-out members of the same family | 1–500 sequences |
| **Negative candidates** | Sequences from *other* EC4 families, matched by sequence similarity to the conditioning set | 1–500 sequences (equal to positives) |

Candidates are added in **matched pairs**: for every positive at *X*% maximum sequence identity to the conditioning set, there is a negative also at *X*% identity (within the same 10-percentage-point bin). A model that simply thresholds on sequence identity will score at chance.

## Design decisions

### Why EC4 families?

EC (Enzyme Commission) numbers have four levels of specificity. The fourth level (e.g., `1.1.1.1` = alcohol dehydrogenase with NAD+) specifies the exact reaction catalysed, including substrate. This is the natural granularity for "does this enzyme do the same thing?"—coarser levels (EC3, EC2) group enzymes that share mechanism but act on different substrates.

### How families are selected

From the full set of reviewed Swiss-Prot proteins with EC annotations (~280k sequences, ~5,700 EC4 families), we apply three filters:

- **Minimum size ≥ 50** — enough sequences to form a conditioning set plus held-out positives.
- **Maximum size ≤ 10,000** — avoids a few giant families dominating compute.
- **Promiscuity filter ≤ 20%** — families where more than 20% of members carry multiple EC annotations are excluded, because ambiguous ground-truth labels undermine the binary classification task.

This yields ~650 benchmark-eligible families.

### How conditioning sets are built

Rather than randomly sampling family members (which would over-represent dense clusters of near-identical orthologs), we:

1. Cluster the family at 70% sequence identity using MMseqs2.
2. Select one representative per cluster, up to 100.
3. If there are more than 100 clusters, randomly subsample 100 representatives.

This ensures the conditioning set covers the family's sequence diversity without redundancy—analogous to how a well-curated MSA would look in practice.

### How similarity matching works

The key innovation of this benchmark. For each positive candidate *p*, we:

1. Compute *p*'s maximum percent identity to any member of the conditioning set.
2. Assign *p* to a similarity bin: [20–30%), [30–40%), ..., [90–100%].
3. Find a negative candidate whose maximum percent identity to the conditioning set falls in the **same bin**.
4. Prefer the negative whose identity is **closest** to *p*'s within the bin.

Up to 50 positives are sampled per bin, and each is matched to exactly one negative, producing balanced pairs. Instances with fewer than 2 viable pairs are discarded.

### Decoy tier system

Negatives are drawn from a hierarchy of EC distance to the target family, preferring harder (more biologically confusable) decoys:

| Tier | Relationship to target | Example |
|------|----------------------|---------|
| **1 (hardest)** | Same EC3, different EC4 | Target `1.1.1.1`, decoy from `1.1.1.2` — same mechanism, different substrate |
| **2** | Same EC2, different EC3 | Target `1.1.1.1`, decoy from `1.1.2.*` |
| **3** | Same EC1, different EC2 | Target `1.1.1.1`, decoy from `1.2.*.*` |
| **4 (easiest)** | Different EC1 entirely | Target `1.1.1.1`, decoy from `2.*.*.*` |

For each positive, we attempt to find a similarity-matched Tier 1 decoy first. If none exists, we fall back through Tiers 2–4. The tier is recorded in the labels file, enabling per-tier evaluation.

### False-negative protection

A sequence annotated under a different EC4 number may still possess the target function—EC annotations are incomplete, and many enzymes are promiscuous. Using such a sequence as a "negative" would be a false negative, poisoning the benchmark. Three safeguards are applied:

1. **Sequence identity filter**: Any candidate decoy with >70% identity to *any* member of the full target family (not just the conditioning set) is excluded. At 70% identity, functional conservation is near-certain.

2. **Promiscuous enzyme filter**: If a candidate decoy is annotated with multiple EC numbers and one of them is the target EC4, it is excluded.

3. **Family overlap flag**: If any member of the decoy's own EC4 family shares >90% identity with any member of the target family, that entire decoy family is deprioritised—its annotations may be inconsistent with the target family's.

### Why MMseqs2 for similarity computation

A full all-vs-all comparison of ~280k sequences would require ~31 billion pairwise alignments. We use a two-stage strategy:

1. **Within-EC3 all-vs-all**: For each EC3 group, run `mmseqs search` on just the sequences sharing that EC3 prefix. These are the biologically meaningful comparisons (potential Tier 1 decoys). Most EC3 groups contain hundreds to low thousands of sequences, making this tractable.

2. **Cross-EC3 representative search**: Cluster the full database at 50% identity, then run all-vs-all on cluster representatives. This provides approximate cross-group similarities for Tier 2–4 decoy selection and distant false-negative detection, without the quadratic cost.

## Repository structure

```
pflm_ec_bench/
├── config.yaml                      # All tunable parameters
├── environment.yml                  # Conda environment specification
│
├── scripts/                         # Data pipeline (run in order)
│   ├── 01_download_swissprot.py     # Download EC-annotated Swiss-Prot
│   ├── 02_filter_families.py        # Select eligible EC4 families
│   ├── 03_build_mmseqs_db.py        # MMseqs2 databases + similarity search
│   ├── 04_construct_benchmarks.py   # Build conditioning sets + matched pairs
│   ├── 05_validate_benchmark.py     # QC checks and baseline evaluation
│   └── utils/
│       ├── io_utils.py              # FASTA / TSV / JSON I/O
│       ├── similarity.py            # Bin matching, pair selection, FN filters
│       └── family.py                # EC parsing, family filtering, conditioning
│
├── evaluation/                      # Model evaluation
│   ├── score_model.py               # Generic scorer interface
│   ├── metrics.py                   # AUROC, AUPRC, per-bin/per-tier metrics
│   └── scorers/                     # Model-specific scorers (stubs)
│       ├── poet.py
│       ├── profam.py
│       └── msa_transformer.py
│
└── data/                            # Generated (gitignored)
    ├── raw/                         # Downloaded Swiss-Prot FASTA + metadata
    ├── processed/                   # Family TSV, MMseqs2 databases + results
    └── benchmark/
        ├── manifest.json            # Index of all benchmark instances
        └── instances/{ec_number}/
            ├── conditioning.fasta   # Family prompt sequences
            ├── candidates.fasta     # Positive + negative candidates
            └── labels.tsv           # Ground truth + metadata per candidate
```

## Setup

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate pflm_ec_bench
```

MMseqs2 is included in the conda environment. If you already have it installed system-wide, either version will work.

## Running the pipeline

All scripts are run from the repository root and read parameters from `config.yaml`.

```bash
# 1. Download Swiss-Prot EC-annotated sequences (~280k seqs, ~5 min)
python scripts/01_download_swissprot.py

# 2. Filter EC4 families by size and quality (~1 min)
python scripts/02_filter_families.py

# 3. Build MMseqs2 databases and compute similarities (~30–60 min)
python scripts/03_build_mmseqs_db.py

# 4. Construct benchmark instances with matched pairs (~10–30 min)
python scripts/04_construct_benchmarks.py

# 5. Validate: check for leakage, similarity matching, baseline AUROC (~5 min)
python scripts/05_validate_benchmark.py
```

Steps 1–3 only need to be run once. Step 4 can be re-run with different parameters (e.g., different bin sizes or pair counts) by editing `config.yaml`.

## Output format

Each benchmark instance produces three files:

### `conditioning.fasta`
Standard FASTA containing the 1–100 conditioning sequences for the target family. This is the input "prompt" for conditional models.

### `candidates.fasta`
Standard FASTA containing all candidate sequences (positives interleaved with negatives).

### `labels.tsv`

| Column | Description |
|--------|-------------|
| `seq_id` | Sequence identifier (UniProt accession) |
| `label` | `1` = in-family (positive), `0` = out-of-family (negative) |
| `sim_bin` | Similarity bin, e.g. `30-40` |
| `tier` | Decoy tier (1–4, see above); same value for both members of a pair |
| `paired_with` | The seq_id this candidate is paired with |
| `max_pident_to_conditioning` | Maximum percent identity to any conditioning set member |

### `manifest.json`
Top-level index listing all instances with summary statistics: family size, conditioning set size, number of pairs, similarity bin distribution, and tier distribution.

## Evaluating a model

### 1. Score candidates

For each benchmark instance, compute a conditional log-likelihood (or pseudo-likelihood) for each sequence in `candidates.fasta`, conditioned on `conditioning.fasta`. Write results to a TSV:

```
seq_id	score
P12345	-45.23
Q67890	-52.17
...
```

Implement your scorer in `evaluation/scorers/` following the interface in `evaluation/score_model.py`, then run:

```bash
python evaluation/score_model.py \
    --model poet \
    --benchmark-dir data/benchmark \
    --output-dir results/poet
```

### 2. Compute metrics

```bash
python evaluation/metrics.py \
    --benchmark-dir data/benchmark \
    --scores-dir results/poet \
    --output results/poet_metrics.json
```

This reports:
- **Overall AUROC and AUPRC** across all instances.
- **Per-similarity-bin AUROC** — the key diagnostic. A model that learns function-specific features should maintain performance even in the low-similarity bins (20–40%), where raw homology provides little signal.
- **Per-tier AUROC** — performance on hard (Tier 1: same EC3) vs. easy (Tier 4: different EC1) negatives.

## Validation checks

`05_validate_benchmark.py` runs several quality-control checks:

1. **No leakage**: No candidate sequence appears in any conditioning set.
2. **Similarity matching**: For each pair, positive and negative have `max_pident` in the same bin. Reports the mean absolute difference.
3. **Label balance**: Equal positives and negatives per instance.
4. **Nearest-neighbour baseline**: Uses `max_pident_to_conditioning` as the score. If this achieves AUROC > 0.9, similarity matching has failed for that instance. A well-constructed benchmark should yield baseline AUROC near 0.5.

## Configuration reference

All parameters are in `config.yaml`:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `families` | `min_family_size` | 50 | Minimum sequences per EC4 family |
| | `max_family_size` | 10,000 | Maximum sequences per EC4 family |
| | `max_promiscuous_fraction` | 0.2 | Max fraction of multi-EC members |
| `conditioning` | `max_size` | 100 | Max sequences in conditioning set |
| | `cluster_identity` | 0.7 | Clustering threshold for representative selection |
| `candidates` | `max_pairs_per_instance` | 500 | Max positive/negative pairs per instance |
| | `min_pairs_per_instance` | 2 | Min pairs required to keep an instance |
| | `sim_bins` | [20,30,...,100] | Similarity bin edges (percent identity) |
| | `max_per_bin` | 50 | Max positives sampled per bin |
| `false_negative_protection` | `exclude_pident_threshold` | 0.7 | Exclude decoys above this identity to target family |
| | `exclude_promiscuous` | true | Exclude decoys annotated with target EC |
| | `family_overlap_pident` | 0.9 | Flag families with cross-identity above this |
| `mmseqs` | `sensitivity` | 7.5 | MMseqs2 search sensitivity |
| | `threads` | 8 | MMseqs2 threads |

## Related work

This benchmark fills a gap between existing enzyme function prediction benchmarks and the evaluation needs of conditional generative protein models:

- **CARE** — EC classification with test splits stratified by similarity to training data. Provides the right evaluation culture but is framed as classification, not conditional likelihood scoring.
- **EC-Bench** — Unified evaluation of EC prediction methods across multiple tasks. Does not test conditional models.
- **CLEAN** — Evaluates EC prediction under low-homology regimes (≤50% identity splits). Relevant difficulty calibration, different task formulation.
- **Price-149 / New-392** — Hard external test sets of experimentally validated enzymes, widely reused.
- **FunFams (CATH)** — Functionally coherent subfamilies within homologous superfamilies. A potential alternative family definition for a structural-similarity-controlled variant of this benchmark.

None of these benchmarks test whether a conditional PLM's likelihood score, given a family prompt, can discriminate in-family from out-of-family sequences under matched sequence identity.

## Citation

If you use this benchmark, please cite the repository and the underlying data source (UniProt/Swiss-Prot).
