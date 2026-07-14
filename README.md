# pflm_ec_bench

A benchmark for evaluating **conditional protein family language models** (PoET, ProFam, MSA Transformer, etc.) on enzyme function classification, with controls for sequence similarity.

> ### ⚠️ Status: the similarity control does not work
>
> The intended guarantee — that a pure-homology model scores at chance — **does not hold**. The benchmark matches positives and negatives on MMseqs2 *local* percent identity, which ignores alignment coverage. Alignment coverage alone reaches a mean within-instance AUROC of **0.90 (v1) / 0.91 (v2)**, and a cross-validated logistic regression over six cheap alignment statistics reaches **0.94 in both**, with a median of **1.000**. Run `python evaluation/leakage_gate.py --benchmark-dir data/benchmark_v2` to reproduce; it exits non-zero. See [Known limitations and confounds](#known-limitations-and-confounds).
>
> Results produced with `data/benchmark/` or `data/benchmark_v2/` should not be interpreted as evidence that a model uses function rather than homology.
>
> **Use `data/benchmark_v4/` instead** (`scripts/08_retrieval_search.py` → `scripts/09_construct_benchmark_v4.py`). It matches candidates on the score of an explicit *naive sequence-only retriever* — coverage-floored maximum percent identity to the prompt — and restricts the candidate pool to what that retriever actually returns. On v4 every pairwise sequence statistic sits at or near chance (matched score 0.543, E-value 0.523, coverage 0.612) versus 0.914 for coverage on v2. A profile HMM still scores 0.698, which is the *point*: it is a better homology method than the naive baseline, and it is the bar a conditional PLM must clear. 19 instances, 520 pairs. See [v4](#v4-match-on-the-naive-retriever-then-report-what-beats-it).

## Motivation

Conditional protein language models assign likelihood scores to a query protein *conditioned on* a set of related sequences (a "prompt"). A natural application is predicting whether a candidate enzyme belongs to a given functional family: condition the model on known family members, then check whether the candidate receives a high likelihood.

The problem is that conditional likelihood correlates strongly with raw sequence identity between the candidate and the prompt. A model that merely measures homology—without learning anything about function—could appear to perform well on a naive benchmark. **This benchmark is designed to make that shortcut impossible.**

We construct a binary classification task—*in-family vs. out-of-family*—where the distribution of maximum *local* percent identity between candidates and the conditioning set is matched between positive and negative examples (in `negative_higher` mode, negatives are made marginally *more* similar).

The intent was that any model outperforming chance must be capturing something beyond raw homology. **This does not follow**, because local percent identity is not a sufficient statistic for homology: it says nothing about how much of the sequence was aligned. Matching it leaves alignment coverage, E-value and bitscore free to vary, and they vary enormously between positives and negatives. See [Known limitations and confounds](#known-limitations-and-confounds).

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

From the full set of reviewed Swiss-Prot proteins with EC annotations (279,501 sequences), we apply three filters:

- **Minimum size** (`min_family_size`, currently **10**) — enough sequences to form a conditioning set plus held-out positives.
- **Maximum size** (`max_family_size`, currently effectively unlimited) — intended to stop a few giant families dominating compute.
- **Promiscuity filter ≤ 20%** — families where more than 20% of members carry multiple EC annotations are excluded, because ambiguous ground-truth labels undermine the binary classification task.

Around 1,400 families pass. Only **152** survive pair construction (see [Silent truncation](#3-the-identity-range-is-silently-truncated)).

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

#### "Negative slightly higher" mode (recommended)

With pure closest-in-bin matching, the available decoys within a bin tend to cluster toward its lower edge (out-of-family sequences are, by construction, less similar to the conditioning set), so the matched negative is on average *slightly less* similar than the positive — a residual bias that mildly **inflates** the apparent performance of any model that scores by raw homology.

Set `candidates.negative_higher: true` (the default in `config.yaml`) to remove this. Instead of the absolute-closest decoy in the same bin, each positive is paired with the closest decoy whose maximum identity is **strictly greater** than the positive's, drawn from the full decoy pool (harder tiers still preferred). `candidates.max_pair_gap` (percentage points) caps the negative-minus-positive gap to keep pairs close. This guarantees that in **every** pair the negative is marginally *more* similar to the conditioning set, so a pure-homology model scores **at or below chance** — any model beating it is genuinely using function, not similarity.

On the reference data (`max_pair_gap: 5.0`) this yields **152 instances / 3,445 pairs** in `data/benchmark_v2/`, versus 201 instances / 4,810 pairs for closest-in-bin in `data/benchmark/`. In 97.2% of v2 pairs the negative's recorded identity exceeds the positive's, by 1.21 pp on average.

**This mode does not do what it claims.** It removes a ~1 pp bias in local percent identity, a statistic that was never the source of the homology shortcut. Alignment coverage — the actual leak — is unchanged between v1 and v2 (mean coverage of the best-identity hit: 0.285 → 0.278 for negatives), and the learned-adversary AUROC is identical (0.940 → 0.936). The reported "nearest-neighbour baseline AUROC of ~0.32" is circular; see [The validation baseline cannot fail](#2-the-validation-baseline-cannot-fail). Targeting a *below-chance* baseline is itself a design error; see [§9](#9-negative_higher-targets-a-below-chance-baseline-which-is-the-wrong-target).

The mode also has an unadvertised side effect: because no decoy may exceed 70% identity to the target family (false-negative protection), no positive above ~65% identity can ever find a more-similar decoy, so all high-identity positives are discarded.

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
│   ├── 05_validate_benchmark.py     # QC checks (superseded by leakage_gate.py)
│   ├── 06_download_experimental_ec.py  # Entries with experimental EC evidence
│   ├── 06_profile_bitscores.py      # MAFFT + hmmbuild + hmmsearch profiles
│   ├── 07_construct_benchmark_v3.py # Bitscore-matched v3 (experiment; fails gate)
│   ├── 08_retrieval_search.py       # Conditioning vs all Swiss-Prot, with coverage
│   ├── 09_construct_benchmark_v4.py # v4: matched on naive retrieval score (USE THIS)
│   └── utils/
│       ├── io_utils.py              # FASTA / TSV / JSON I/O
│       ├── similarity.py            # Bin matching, pair selection, FN filters
│       └── family.py                # EC parsing, family filtering, conditioning
│
├── evaluation/                      # Model evaluation
│   ├── score_model.py               # Generic scorer interface
│   ├── metrics.py                   # AUROC, AUPRC, per-bin/per-tier metrics
│   ├── leakage_gate.py              # Homology-leak gate (run this first)
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

# 5b. The check that matters: can a cheap homology statistic solve it? (~15 min,
#     then seconds on the cached features). Exits non-zero on failure.
python evaluation/leakage_gate.py --benchmark-dir data/benchmark_v2
```

### Profile-HMM baseline and the v3 experiment

Requires HMMER (`hmmbuild`, `hmmsearch`) and MAFFT.

```bash
# Entries whose catalytic activity carries experimental evidence (~2 min)
python scripts/06_download_experimental_ec.py

# One profile HMM per instance, searched against Swiss-Prot (~20 min)
python scripts/06_profile_bitscores.py --benchmark-dir data/benchmark_v2 --threads 24

# Rebuild candidates, matched on profile bitscore (~2 min)
python scripts/07_construct_benchmark_v3.py

# Gate it, with the HMM bitscore as an additional gated statistic
python evaluation/leakage_gate.py --benchmark-dir data/benchmark_v3 \
    --profile-hits data/benchmark_v2/profile_hits.tsv
```

v3 nullifies the profile HMM (0.518) but still fails the gate on `pident` (0.884) and the learned adversary (0.898). See [The matched-pairs premise does not survive contact with the data](#the-matched-pairs-premise-does-not-survive-contact-with-the-data).

### v4 — the benchmark to actually use

```bash
# Uniform conditioning-vs-all search with coverage fields (~30 min, 9.7M alignments)
python scripts/08_retrieval_search.py --benchmark-dir data/benchmark_v2 --threads 24

# Build candidates matched on the naive retrieval score (~2 min)
python scripts/09_construct_benchmark_v4.py --cov-floor 0.6

# Gate it. --retrieval-tsv is the correct feature source: it measures every
# candidate uniformly, so a zero means "no homology", not "never searched".
python evaluation/leakage_gate.py --benchmark-dir data/benchmark_v4 \
    --retrieval-tsv data/processed/retrieval/cond_vs_all.tsv --cov-floor 0.6 \
    --profile-hits data/benchmark_v2/profile_hits.tsv
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
2. **Similarity matching**: Compares the *instance-level mean* `max_pident` of positives against that of negatives. (It does **not** check that each pair falls in the same bin — under `negative_higher` the negative frequently does not, by design.)
3. **Label balance**: Equal positives and negatives per instance.
4. **Nearest-neighbour baseline**: Uses `max_pident_to_conditioning` as the score. ⚠️ **This check is circular and cannot fail** — see [below](#2-the-validation-baseline-cannot-fail). It is retained only as a regression test that the matching code ran.

None of these checks test the property the benchmark exists to guarantee. A validator that would is described in [Toward a benchmark that actually controls homology](#toward-a-benchmark-that-actually-controls-homology).

## Configuration reference

All parameters are in `config.yaml`:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `families` | `min_family_size` | 10 | Minimum sequences per EC4 family |
| | `max_family_size` | 99999999999 | Maximum sequences per EC4 family (effectively disabled) |
| | `max_promiscuous_fraction` | 0.2 | Max fraction of multi-EC members |
| `conditioning` | `max_size` | 100 | Max sequences in conditioning set |
| | `cluster_identity` | 0.7 | Clustering threshold for representative selection |
| `candidates` | `max_pairs_per_instance` | 500 | Max positive/negative pairs per instance |
| | `min_pairs_per_instance` | 2 | Min pairs required to keep an instance |
| | `sim_bins` | [20,30,...,100] | Similarity bin edges (percent identity) |
| | `max_per_bin` | 50 | Max positives sampled per bin |
| | `negative_higher` | true | Pair each positive with the closest decoy of *strictly greater* local identity. Does **not** remove the homology shortcut. |
| | `max_pair_gap` | 5.0 | Max negative−positive identity gap (pp) when `negative_higher` is on |
| `false_negative_protection` | `exclude_pident_threshold` | 0.7 | Exclude decoys above this identity to target family |
| | `exclude_promiscuous` | true | Exclude decoys annotated with target EC |
| | `family_overlap_pident` | 0.9 | Flag families with cross-identity above this |
| `mmseqs` | `sensitivity` | 7.5 | MMseqs2 search sensitivity |
| | `threads` | 8 | MMseqs2 threads |

## Known limitations and confounds

All figures below were measured directly on the shipped data (`data/benchmark/` = v1, `data/benchmark_v2/` = v2, 152 instances / 3,445 pairs).

### 1. The similarity control matches the wrong variable

`pident` as reported by `mmseqs convertalis` is percent identity **over the aligned region of a local alignment**. `03_build_mmseqs_db.py` runs `mmseqs search` without a `-c` coverage flag, so the default coverage requirement is 0. A 70-residue local hit at 47% identity and a 400-residue full-length hit at 46% identity are treated as equivalent matches. `04_construct_benchmarks.py` reads only columns `query, target, pident` and discards `alnlen`, `qlen`, `tlen`, `evalue`.

The two populations are not remotely comparable. For the best-identity hit of each candidate to its conditioning set:

| | median alignment length | median coverage (`alnlen / max(qlen,tlen)`) | mean coverage |
|---|---|---|---|
| Positives | 272 aa | 0.91 | 0.72 |
| Negatives | 74 aa | 0.16 | 0.28 |

Consequently, homology statistics that the pipeline never optimised against separate the classes easily. The table reports **mean within-instance AUROC** — the only meaningful unit, since a model ranks each instance's candidates against that instance's prompt, and pooling across families mixes in between-family variation. Produced by `evaluation/leakage_gate.py`; instances with ≥5 pairs (v1: 137, v2: 109).

| Statistic (max over conditioning set) | v1 | v2 |
|---|---|---|
| `pident` — *the matched variable* | 0.609 | 0.397 |
| Alignment coverage | **0.902** | **0.914** |
| Coverage-adjusted identity (`pident × alnlen / max(qlen,tlen)`) | 0.893 | 0.873 |
| MMseqs2 E-value | 0.809 | 0.789 |
| Alignment length | 0.729 | 0.742 |
| Candidate length | 0.435 | 0.473 |
| **Learned adversary** (per-instance logistic regression, cross-validated) | **0.940** | **0.936** |

The adversary's *median* is **1.000** in both versions: for most instances a six-feature logistic regression fitted inside that instance separates positives from negatives perfectly. This is not small-sample overfitting — the AUROC is out-of-fold, and it is *higher* on the largest instances (0.955 for v1 instances with ≥50 pairs; Spearman correlation between instance size and adversary AUROC is +0.05).

**v1 and v2 are indistinguishable** on this measure (0.940 vs 0.936). The `negative_higher` change had no effect on the leak.

A model reporting AUROC 0.75 on this benchmark would be presented as beating homology while in fact performing *worse than sorting by MMseqs2 E-value*. The residual bias is in the pro-model direction.

For reference, one confound that was checked and is **not** present: raw sequence length (mean within-instance AUROC 0.435 / 0.473).

#### Single-statistic checks are not sufficient

Instance `2_7_13_3` has coverage AUROC 0.320 — a leak in the *opposite* direction, negatives more covered than positives — and would pass any "is coverage balanced?" test. Its learned adversary scores 0.995. Only a check that considers combinations of statistics detects it.

Nor can the problem be fixed by matching on a better scalar. Rebuilding the pairs under different selection rules (30 largest instances, decoy pool reconstructed from the same MMseqs2 output) shows that **equalising any one statistic pushes the signal into the others**:

| Negative selection rule | `pident` | coverage | cov-adj id | −log10 E | adversary |
|---|---|---|---|---|---|
| match on `pident` (v2's choice) | 0.605 | **0.892** | 0.891 | 0.842 | 0.887 |
| match on coverage-adjusted identity | **0.738** | 0.727 | 0.759 | 0.657 | 0.664 |
| match on E-value | 0.738 | 0.655 | 0.632 | 0.491 | 0.747 |

Two further negative results from the same experiment, both of which contradict plausible hypotheses:

- **The greedy argmax in `select_higher_decoys` is not the cause.** Replacing it with stratified random sampling within bins of the matching statistic changes the adversary by less than 0.02 (0.747 → 0.741 matching on E-value; 0.700 → 0.713 matching on coverage-adjusted identity). The leak is a property of *which variable is matched*, not of *how* the decoy is picked within a matched stratum.
- **Restricting the decoy pool to well-covered alignments helps but does not fix it.** Requiring decoys to have ≥50% coverage to the conditioning set shrinks the pool from 35,064 to 14,368 and leaves the adversary at 0.672.

### 2. The validation baseline cannot fail

`nearest_neighbor_baseline()` in `05_validate_benchmark.py` scores candidates by `max_pident_to_conditioning` read from `labels.tsv` — precisely the quantity `select_higher_decoys()` maximised for negatives. AUROC below 0.5 is an algebraic identity, not evidence of a controlled benchmark. Any meaningful QC baseline must use a statistic construction did not see.

Use `evaluation/leakage_gate.py` instead:

```bash
python evaluation/leakage_gate.py --benchmark-dir data/benchmark_v2 \
    --per-instance-out evaluation/results/leakage_gate_v2.csv
```

It recomputes six alignment statistics per candidate from the raw MMseqs2 output, reports mean within-instance AUROC for each, fits a per-instance cross-validated learned adversary, and exits non-zero if any statistic deviates from chance by more than `--max-auroc`. It fails on both shipped datasets. An `--embeddings` flag adds max cosine similarity to the conditioning set in a supplied embedding space (e.g. ESM-2), which should be part of the gate once available.

Note that the gate treats deviation in *either* direction as failure. A below-chance homology statistic is also a design flaw — see §9.

The existing alignment-feature baselines in `evaluation/results/` corroborate the leak: `gp_alignment_metrics.csv` reports raw AUROC of 0.92–1.00 on small families. The `_fit_similarity_adjustment()` used to produce the "adjusted" columns regresses the score on `max_pident`, which is the wrong covariate, and leaves coverage untouched. It is also fitted on the evaluation data itself.

### 3. The identity range is silently truncated

False-negative protection excludes any decoy above 70% identity to the target family. The conditioning set is a subset of that family, so no decoy exceeds ~70% identity to it either. Under `negative_higher`, every positive needs a *strictly more similar* decoy — therefore **no positive above ~65% identity can ever be paired**.

The shipped v2 bins confirm it: `20-30`: 316, `30-40`: 765, `40-50`: 1,088, `50-60`: 914, `60-70`: 362, and nothing above. 1,174 of ~1,400 eligible families were dropped as `too_few_pairs` (plus 98 as `too_few_positives`).

This is arguably a reasonable difficulty regime, but it is an emergent consequence of two interacting parameters rather than a stated design choice, and it is not documented anywhere in the output.

### 4. Positive selection is a subsampling side-effect

Conditioning sets are 70%-identity cluster representatives, capped at 100 by `random.sample()`. Positives are the leftover family members. But only **52 of 3,445 positives** have their own 70%-cluster representative in the conditioning set, and **134 of 152 instances** have exactly 100 conditioning sequences.

So positives are almost entirely the *orphans of clusters whose representative was randomly discarded by the size cap*. The population of positives is defined by a subsampling accident, not by a stated criterion, and this is why only large, diverse families survive. Holding out whole 70% clusters would be the principled equivalent.

### 5. Candidates are not independent

Within an instance only ~71% of positives occupy distinct 70% clusters (mean; e.g. `2_7_4_25` has 106 positives spanning 53 clusters, one contributing 12 members). Negatives are ~76% unique and are additionally reused across instances. The effective sample size is well below 3,445, so pooled confidence intervals are too narrow.

### 6. Decoy tier confounds with measurement quality

Tier-1 (same EC3) similarities come from a full within-EC3 all-vs-all search. Tier 2–4 similarities come *only* from the 50%-identity cluster-representative search (`cross_ec3_results.tsv`). A tier-2–4 decoy has a recorded similarity to a conditioning member only if **both** happen to be 50% cluster representatives; otherwise the pair is absent from `sim_matrix` and `compute_max_sim_to_conditioning()` returns `0.0`, dropping the sequence from the pool.

Three consequences: `max_pident_to_conditioning` is systematically under-measured for distant tiers; the tier-2–4 decoy pool is silently restricted to cluster representatives; and `find_false_negative_ids()` is nearly inert there, since it can only fire on pairs present in the matrix. Across all 152 families, only **3,647** decoys were excluded as false negatives, out of pools of order 10⁵.

Observed tier distribution (v2): tier 1: 2,237, tier 2: 278, tier 3: 256, tier 4: 674.

### 7. EC labels are not a reliable target

This is the deepest problem, and unlike the others it cannot be fixed by better pair matching.

**No evidence codes.** `data/raw/swissprot_ec_metadata.tsv` has columns `Entry, EC number, Organism, Length, Sequence`. Swiss-Prot is manually reviewed, but for most entries the EC number is *propagated by sequence similarity* (`ECO:0000250`, `ECO:0000256`) rather than experimentally determined. A large fraction of positive labels were therefore assigned **because the protein looked homologous to a family member**. The ground truth is partly a function of the confound the benchmark is trying to remove. Any homology-controlled functional benchmark built on Swiss-Prot must at minimum stratify by evidence code and report the experimentally-annotated (`ECO:0000269`, or RHEA-cross-referenced) subset separately.

**Negatives are unverified.** "Negative" means *not annotated with the target EC*, which is not the same as *does not catalyse the target reaction*. Under-annotation is the normal state of Swiss-Prot. The promiscuity filter only catches proteins already annotated with both ECs — exactly the subset that was never a problem.

**Two parser leaks:**

- `parse_ec_annotations()` rejects EC components equal to the literal string `"n"`, but preliminary EC numbers are formatted `3.6.5.n1`. The test `p != "n"` passes them. **1,305 entries** carry preliminary ECs and are treated as legitimate EC4 families and as tier-1 decoys (`3.6.5.n1` alone has 707 members).
- Proteins annotated with a partial EC alongside a complete one (`1.1.1.-; 2.7.1.5`) enter the decoy pool via the complete EC, while the partial annotation may be consistent with the target family. **41 of 3,445 v2 negatives** are of this form; `find_false_negative_ids()` does not check partial ECs.

**EC4 is a reaction class, not a homology class.** The same EC4 is catalysed by non-homologous proteins (convergent evolution), and a single EC3 spans multiple superfamilies. "Same EC3, different EC4" therefore does not reliably mean *biologically confusable*; it frequently means *no detectable homology at all*, which is exactly what the coverage figures in §1 show. The tier hierarchy is a proxy for difficulty that does not measure difficulty.

### 8. Metrics are dominated by tiny instances

53 of 152 instances have ≤ 5 pairs. At 2 pairs, a per-instance AUROC can only take the values {0, 0.25, 0.5, 0.75, 1}. `evaluation/metrics.py` reports an unweighted mean across instances, so much of the reported signal is quantisation noise, while the 10 largest instances hold 33.2% of all pairs.

Because the design is paired, the appropriate statistic is **paired ranking accuracy** with a sign test, bootstrapped **by family** (not by pair) to respect the cluster structure described in §5.

### 9. `negative_higher` targets a below-chance baseline, which is the wrong target

The mode is documented as a virtue: negatives are made *strictly more similar* than positives so that "a pure-homology model scores at or below chance (~0.32)".

Driving the homology baseline below 0.5 does not make the benchmark harder in a useful way — it makes it **anti-correlated with homology**. A good conditional PLM's log-likelihood is legitimately homology-correlated: it *should* assign higher likelihood to family members. On a set where the homology baseline sits at 0.32, any model that partly encodes homology is dragged below chance, while a functionally-blind, homology-inverted scorer looks excellent. The benchmark penalises the model class it exists to reward.

The correct target is a homology baseline at **0.5**, i.e. matched distributions, not reversed ones. It is sufficient that some negatives are at least as similar as some positives — the strict per-pair inequality is not required and is actively harmful.

---

## Toward a benchmark that actually controls homology

### The matched-pairs premise does not survive contact with the data

The natural fix for §1 is: *pick the homology baseline you claim to beat, and match positives to negatives on that baseline's own score.* Then its within-instance AUROC is 0.5 by construction, and — the argument runs — there is no residual variable to leak into.

**This was implemented and tested. It does not work.**

`scripts/06_profile_bitscores.py` builds a profile HMM per instance (MAFFT → `hmmbuild`) and scores all of Swiss-Prot with it (`hmmsearch`). This is the baseline a practitioner would actually run, and it is coverage-sensitive by construction, so the §1 confound cannot arise from it. On the **v2** pairs it is devastating:

| | mean within-instance AUROC | median |
|---|---|---|
| profile-HMM bitscore on `benchmark_v2` | **0.842** | **0.992** |

`scripts/07_construct_benchmark_v3.py` then rebuilds the candidate sets, matching each positive to the nearest-bitscore negative (caliper: max(2 bits, 2%)), with positives restricted to experimentally-evidenced family members and negatives required to carry experimental evidence of a *different* reaction. Result: 22 instances, 729 pairs, median |Δbitscore| within a pair of 0.96 bits.

The matching works exactly as intended — and the benchmark is still solvable:

| Statistic | v3 mean within-instance AUROC |
|---|---|
| **`hmm_bitscore`** — *the matched variable* | **0.518** |
| `pident` | 0.884 |
| coverage-adjusted identity | 0.888 |
| coverage | 0.856 |
| E-value | 0.803 |
| **Learned adversary** | **0.898** |

Bitscore is a scalar summary of, roughly, *per-residue identity × aligned length*. Equalising it leaves the decomposition free. The matched negatives turn out to be long proteins matching weakly across many positions; the positives are shorter with high per-residue identity. Same bitscore, entirely different alignments — and `pident` separates them at 0.884.

**No scalar homology summary is sufficient.** Equalising one always frees the others, and the profile HMM is no exception.

### Nor can it be repaired by matching on more covariates

Requiring a matched negative to agree with its positive on bitscore *and* `pident` *and* coverage simultaneously (calipers: 2%/2 bits, 5 pp, 0.10) empties the benchmark:

| Joint caliper | matchable positives | instances retaining ≥5 pairs |
|---|---|---|
| bitscore only | 53.0% | 19 / 22 |
| bitscore + `pident` | 34.3% | 9 / 22 |
| bitscore + coverage | 40.3% | 12 / 22 |
| **bitscore + `pident` + coverage** | **25.3%** | **5 / 22** |

(These counts permit decoy reuse; under 1:1 matching retention is strictly lower.)

Even bitscore alone truncates severely, and for a structural reason. Across the 152 instances, experimental positives have median bitscore 219 against their family HMM; decoys have median 58. Only 34.5% of positives have *any* decoy within the bitscore caliper, and retention collapses in the upper deciles:

| positive bitscore decile | fraction with a decoy in caliper |
|---|---|
| 9–52 | 56.1% |
| 103–252 | 44–57% |
| 252–302 | 37.6% |
| **302–380** | **8.0%** |
| **380–449** | **1.4%** |

This is the §3 truncation reappearing in bitscore space, and it is not an artifact. **Out-of-family sequences that are homologically indistinguishable from family members largely do not exist**, because a sequence that close to an enzyme family usually *is* a member of it. That fact is precisely why homology search works so well for enzyme function prediction — and it is what makes a fully homology-matched benchmark unbuildable at useful scale.

### v4: match on the naive retriever, then report what beats it

The compromise that works. Rather than trying to nullify *homology*, nullify **one precisely specified, honestly reported baseline** — the naive sequence-only method a practitioner would actually use — and let better methods try to beat it.

**The naive retriever.** Search the known family members against the database, discard hits that do not align over most of both sequences, rank the survivors by percent identity:

```
S(c) = max { pident(q, c) : q ∈ conditioning set, qcov ≥ T, tcov ≥ T }
```

**The pool is what it retrieves.** Candidates are exactly the sequences with at least one qualifying alignment. A sequence with no qualifying alignment was never retrieved, so it is not a hard negative — it is a non-result, and it is not a candidate.

This is the step that fixes §1. The short spurious local alignments that let v2's negatives look identity-matched are *removed from the pool by construction* rather than balanced against. It also required re-measuring similarity properly: `scripts/08_retrieval_search.py` searches all 14,058 conditioning sequences against all of Swiss-Prot with `qcov`/`tcov` emitted (9.7M alignments), because the step-03 output cannot support this (§6).

**Coverage floor T = 0.6**, chosen by sweep, not by taste:

| T | instances | pairs | learned adversary |
|---|---|---|---|
| 0.0 (≡ v2) | 70 | 3,049 | 0.897 |
| 0.5 | 27 | 912 | 0.705 |
| **0.6** | **25** | **673** | **0.651** |
| 0.7 | 23 | 484 | 0.751 |
| 0.9 | 15 | 160 | 0.741 |

An additional coverage caliper on the matched pair was tested and rejected — it bought ~0.03 of adversary AUROC for a real loss of pairs.

**Positives must be experimentally evidenced**, and this is load-bearing rather than fastidious. Allowing homology-propagated positives raises the adversary from 0.65 to 0.76 *and degrades the matching itself* (S drifts from 0.55 to 0.61), because propagated labels were assigned **by** homology and therefore sit at high identity where no decoy exists to match them. The circularity of §7, visible in the numbers.

#### Result: `data/benchmark_v4/` — 19 instances, 520 pairs

Gated with uniform, complete features (`--retrieval-tsv`), every pairwise sequence statistic is at or near chance:

| Statistic | mean within-instance AUROC |
|---|---|
| **`S_retrieval`** — *the baseline, matched* | **0.543** |
| coverage-adjusted identity | 0.557 |
| E-value | 0.523 |
| alignment length | 0.532 |
| sequence length | 0.483 |
| coverage | 0.612 |
| Learned adversary, **sequence features only** | 0.740 |
| **`hmm_bitscore` — profile HMM** | **0.698** |
| Learned adversary, **incl. profile HMM** | 0.827 |

Compare v2, where coverage was 0.914 and the adversary 0.954.

**The profile HMM at 0.698 is the point of the benchmark, not a bug.** It is a *different and better* homology method than the naive retriever, it was never matched on, and it beats the nullified baseline decisively. That is exactly the finding the benchmark is built to produce — and it sets the bar a conditional PLM has to clear. Beating `S_retrieval` is table stakes; the real question is whether a PLM beats the profile HMM.

#### How to report a model on v4

1. **Headline:** within-instance AUROC. `S_retrieval` ≈ 0.5 by construction, so any lift is lift over the naive retriever.
2. **The bar that matters:** ΔAUROC against `hmm_bitscore` (0.698). A model that fails to beat this is not doing anything a profile HMM cannot.
3. **Always report the gate table alongside.** Coverage still leaks at 0.612 and the supervised adversary reaches 0.740 on sequence features alone. The adversary is a *supervised ceiling* — it sees the labels, so it is not a fair competitor for a zero-shot model — but it is the honest statement of how much of this set is explicable by cheap alignment features. Do not quote a model's AUROC without it.
4. **Bootstrap by family**, not by pair (§5). With 19 instances, confidence intervals will be wide; say so.

#### Honest limitations of v4

- **19 instances is small.** The attrition is structural (§3): matchable negatives simply do not exist for high-identity positives.
- **Coverage is not fully neutralised** (0.612), and the sequence-only adversary reaches 0.740. v4 is *far* better controlled than v2, not perfectly controlled.
- **Conditioning sets are inherited from v2**, so the §4 subsampling artifact carries over.
- **False-negative protection is measured against the prompt only.** A decoy ≥70% identical to a conditioning member is excluded, but non-conditioning family members are not checked directly. Since conditioning sequences are 70%-cluster representatives of the family, this catches the dangerous cases, but it is weaker than a full family-vs-all check.

`data/benchmark_v3/` is retained as the artifact of the bitscore-matching experiment, not as a shipping benchmark. It fails the gate.

### Negatives need contrary evidence, not absent annotation

Restrict candidates to the **experimentally-annotated subset**: `(reviewed:true) AND (ec:*) AND (cc_catalytic_activity_exp:*)` returns **48,496** entries (of 280,036), of which 45,181 carry a complete EC4. A negative is then a sequence *experimentally shown to catalyse a different reaction*, not merely one that lacks the target annotation. This is the only definition of a negative that a functional benchmark can honestly use (§7).

Scale is not the obstacle. Among these, **1,033 EC4 families have ≥10 experimentally-evidenced members**, and taking as hard negatives the candidates that share a Pfam domain with the family but carry a different EC4:

| | Families |
|---|---|
| ≥1 homologous different-reaction negative | 965 / 1,033 |
| ≥10 | 876 |
| ≥50 | **653** |

Median hard-negative pool: 83 sequences per family, of which a median of 23 share the target's EC3 (same mechanism, different substrate). That is 653 candidate instances against today's 152. (Pfam-sharing is a loose homology proxy that over-counts multidomain proteins sharing a regulatory domain; in the real pipeline, define homology by the family HMM and use Pfam only as a sanity check.)

Note also that **only candidates need experimental labels**. The conditioning set is a prompt, not a test item, so exemplars may be drawn from the full 280k reviewed set. Family sizes for prompt construction are unaffected.

### Gate on a learned adversary, not a list of scalars

Run `evaluation/leakage_gate.py` and require every statistic *and* the per-instance learned adversary to fall within 0.5 ± 0.05. §1 shows why the adversary is the load-bearing check: an instance can have every scalar near chance and still be perfectly separable by a linear combination of them.

Add ESM-2 embedding cosine to the gate's feature set (`--embeddings`) before shipping.

### Report two numbers, not one

A set on which the HMM sits at 0.5 is, by construction, not the set encountered in deployment. So:

- **Mechanistic claim** — within-instance ΔAUROC of the model over the profile HMM on matched pairs, bootstrapped by family. Answers: *does the model carry information the HMM lacks?*
- **Deployment claim** — per-bitscore-stratum ΔAUROC on the unmatched hard-negative pool. Answers: *would swapping HMMER for this model improve a real screen?*

The bitscores needed for the second are already computed for the matching in the first.

### Remaining fixes

- **Drop `negative_higher`.** Target a homology baseline at 0.5, not 0.32 (§9).
- **Decouple the false-negative threshold from the identity ceiling** so the high-identity bins return. Use a coverage-aware exclusion criterion (e.g. ≥70% identity *and* ≥80% coverage) rather than local `pident` (§3).
- **Hold out whole 70% clusters** as positives instead of the orphans of subsampled clusters (§4).
- **Fix `parse_ec_annotations()`**: reject any component matching `^n\d*$`, and check partial ECs against the target prefix in `find_false_negative_ids()` (§7).
- **Report paired ranking accuracy** with family-level bootstrap; drop or down-weight instances with ≤5 pairs (§8).
- **Compute cross-EC3 similarities for all sequences**, not just 50% cluster representatives, or record which similarities are missing rather than silently coercing them to `0.0` (§6).

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
