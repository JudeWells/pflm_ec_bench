# pflm_ec_bench

A benchmark for evaluating **conditional protein family language models** (PoET, ProFam, MSA Transformer, …) on enzyme function classification, with a homology control that has been tested rather than assumed.

**Use `data/benchmark_v4/`.** Earlier versions (`data/benchmark/`, `data/benchmark_v2/`) are retained for reference but their homology control does not work — a cross-validated logistic regression over six cheap alignment statistics separates their positives from negatives at AUROC 0.94, median 1.00. See [Design history](#design-history) for what went wrong and how it was found.

---

## The task

Conditional protein language models score a query sequence *given* a set of related sequences (a "prompt"). The natural application is database screening: **given some known members of an enzyme family, which other sequences in a database are likely to have the same function?**

The obvious baseline is a sequence homology search. So the only interesting question is whether a model **beats homology** — and to ask that honestly, the benchmark must nullify a homology baseline that is actually specified, actually strong, and actually reported.

v4 does this by nullifying one precisely defined method and reporting everything else.

### The naive retriever

The sequence-only method a practitioner would really run: search the known family members against the database, throw away hits that only align over a fragment, rank what survives by percent identity.

```
S(c) = max { pident(q, c) : q ∈ conditioning set, qcov ≥ 0.6, tcov ≥ 0.6 }
```

### The pool is what that retriever returns

Candidates are **exactly** the sequences with at least one qualifying alignment. A sequence whose only similarity to the prompt is a 70-residue fragment was never retrieved: it is not a hard negative, it is a non-result, and it is not a candidate.

This is the main idea, and it is what earlier versions got wrong. v1/v2 tried to *balance* such sequences against the positives; v4 removes them from the pool by construction. Balancing cannot work — see [§1](#1-matching-local-identity-controls-the-wrong-variable).

### Matching

Each positive is paired 1:1 with the nearest-`S` negative (2 pp caliper). So `S` — the naive retriever — sits at chance by construction, and any lift a model shows is lift over it.

| Component | Definition |
|---|---|
| **Conditioning set** | 100 known family members (70%-identity cluster representatives) — the prompt |
| **Positives** | Held-out family members with **experimental** catalytic-activity evidence |
| **Negatives** | Sequences with **experimental** evidence of a *different* reaction, matched on `S` |

Both positives and negatives require experimental evidence (`cc_catalytic_activity_exp`, 48,496 of 280,036 reviewed EC entries). This is not fastidiousness — see [§4](#4-ec-labels-are-not-a-reliable-target).

---

## What v4 contains

**19 EC4 families, 520 matched pairs.** The families span 18 EC3 groups, 12 EC2 groups, and 6 of the 7 top-level EC classes (no lyases).

| EC4 | Pairs | Class | | EC4 | Pairs | Class |
|---|---|---|---|---|---|---|
| `2.7.11.1` | 229 | transferase | | `3.1.1.4` | 12 | hydrolase |
| `2.7.13.3` | 36 | transferase | | `2.7.10.1` | 11 | transferase |
| `7.1.1.2` | 34 | translocase | | `2.7.4.3` | 11 | transferase |
| `3.2.1.4` | 32 | hydrolase | | `3.2.1.21` | 11 | hydrolase |
| `3.1.26.4` | 20 | hydrolase | | `5.3.1.24` | 10 | isomerase |
| `5.6.2.4` | 20 | isomerase | | `3.1.3.5` | 9 | hydrolase |
| `3.6.4.13` | 18 | hydrolase | | `6.1.1.16` | 9 | ligase |
| `3.4.19.12` | 17 | hydrolase | | `2.3.1.225` | 7 | transferase |
| `1.11.1.7` | 15 | oxidoreductase | | `3.6.1.23` | 6 | hydrolase |
| `2.5.1.18` | 13 | transferase | | | | |

⚠️ **`2.7.11.1` (protein kinase) holds 229 of 520 pairs — 44%.** A pooled AUROC over v4 is close to *"how well does the model do on kinases"*. Always report **mean within-instance** AUROC and bootstrap **by family**.

Decoy tiers (EC distance from the target): tier 1 (same EC3) 274, tier 2 103, tier 3 23, tier 4 120.

### It is homology-controlled — measured, not assumed

`evaluation/leakage_gate.py` recomputes cheap homology statistics from the raw search output and reports **mean within-instance AUROC** for each, plus a per-instance cross-validated logistic regression over all of them (the "learned adversary").

| Statistic | v2 | **v4** |
|---|---|---|
| the matched score | 0.397 | **0.543** |
| alignment coverage | 0.914 | 0.612 |
| coverage-adjusted identity | 0.873 | 0.557 |
| E-value | 0.789 | 0.523 |
| alignment length | 0.742 | 0.532 |
| sequence length | 0.473 | 0.483 |
| **learned adversary** (sequence features) | **0.954** | **0.740** |

Every pairwise sequence statistic is at or near chance. Coverage at 0.612 is the largest residual.

### The bar to beat: a profile HMM scores 0.698

A profile HMM built from the prompt (`hmmbuild` → `hmmsearch`) scores **0.698** on v4. It was never matched on.

**This is the point of the benchmark, not a defect.** It says a *better homology method* extracts substantially more signal than coverage-floored identity does. So:

- Beating `S_retrieval` (≈0.5) is **table stakes** — it only means you beat a naive BLAST-and-sort.
- The real question is whether a conditional PLM beats **0.698**. A model that does not is doing nothing a profile HMM cannot already do.

On v2 you could not have asked this: a model there could score 0.75 and still be *worse* than sorting by E-value.

---

## Evaluating a model on v4

For each instance, score every sequence in `candidates.fasta` conditioned on `conditioning.fasta`, and write `seq_id<TAB>score`.

```bash
python evaluation/score_model.py --model poet \
    --benchmark-dir data/benchmark_v4 --output-dir results/poet
python evaluation/metrics.py \
    --benchmark-dir data/benchmark_v4 --scores-dir results/poet \
    --output results/poet_metrics.json
```

**Report all four of these. A bare AUROC is not interpretable.**

1. **Mean within-instance AUROC**, bootstrapped by family (not by pair — candidates within a family are not independent, [§6](#6-candidates-are-not-independent)). With 19 families, intervals will be wide. Say so.
2. **ΔAUROC vs. the profile HMM** (0.698) — the number that decides whether the model is worth using.
3. **The gate table** for the benchmark version you ran on. Coverage still leaks at 0.612 and the supervised adversary reaches 0.740; a reader needs those to calibrate your headline.
4. **Per-tier AUROC.** Tier-1 negatives (same EC3, different substrate) are the biologically hard ones.

The learned adversary is a **supervised ceiling**: it sees the labels, so it is not a fair competitor for a zero-shot model. It answers "how much of this set is explicable by cheap alignment features?", not "what should my model beat".

### `labels.tsv`

| Column | Description |
|---|---|
| `seq_id` | UniProt accession |
| `label` | 1 = in-family, 0 = out-of-family |
| `tier` | Decoy tier 1–4 (1 = same EC3, hardest) |
| `paired_with` | The `seq_id` this candidate is matched to |
| `retrieval_score` | `S` — max coverage-floored identity to the prompt |
| `coverage` | Alignment coverage of the best qualifying hit |
| `score_gap` | Negative − positive `S` within the pair |

---

## Running the pipeline

```bash
conda env create -f environment.yml && conda activate pflm_ec_bench   # + HMMER, MAFFT
```

```bash
# Steps 1-3: Swiss-Prot download, family filter, MMseqs2 databases  (run once)
python scripts/01_download_swissprot.py
python scripts/02_filter_families.py
python scripts/03_build_mmseqs_db.py

# Step 4: conditioning sets. v4 reuses v2's prompts, so this is still needed.
python scripts/04_construct_benchmarks.py --output-subdir benchmark_v2

# Experimentally-evidenced EC entries (~2 min)
python scripts/06_download_experimental_ec.py

# Uniform conditioning-vs-all search with coverage fields (~30 min, 9.7M alignments)
python scripts/08_retrieval_search.py --benchmark-dir data/benchmark_v2 --threads 24

# Build v4 (~2 min)
python scripts/09_construct_benchmark_v4.py --cov-floor 0.6

# Gate it. --retrieval-tsv is the correct feature source: it measures every
# candidate uniformly, so a zero means "no homology", not "never searched".
python evaluation/leakage_gate.py --benchmark-dir data/benchmark_v4 \
    --retrieval-tsv data/processed/retrieval/cond_vs_all.tsv --cov-floor 0.6 \
    --profile-hits data/benchmark_v2/profile_hits.tsv
```

The profile-HMM baseline (needed for the 0.698 number):

```bash
python scripts/06_profile_bitscores.py --benchmark-dir data/benchmark_v2 --threads 24
```

### Why the coverage floor is 0.6

Swept, not chosen by taste. At `T=0` the design degenerates to v2:

| Coverage floor | Instances | Pairs | Learned adversary |
|---|---|---|---|
| 0.0 (≡ v2) | 70 | 3,049 | 0.897 |
| 0.5 | 27 | 912 | 0.705 |
| **0.6** | **25** | **673** | **0.651** |
| 0.7 | 23 | 484 | 0.751 |
| 0.9 | 15 | 160 | 0.741 |

(Instance counts here are pre-false-negative-filter; the shipped v4 has 19.) An additional coverage caliper on the matched pair was tested and rejected: ~0.03 of adversary AUROC for a real loss of pairs.

---

## Honest limitations of v4

- **19 instances is small, and the attrition is structural.** Out-of-family sequences that are homologically indistinguishable from family members largely *do not exist* — a sequence that close to an enzyme family usually **is** a member of it. That is exactly why homology search works so well here, and it is what caps the size of any homology-matched benchmark. Only 34.5% of experimental positives have any decoy within a matching caliper at all.
- **One family is 44% of the pairs.** See the warning above.
- **Homology is not fully removed.** Coverage leaks at 0.612; the sequence-only supervised adversary reaches 0.740. v4 is *far* better controlled than v2, not perfectly controlled.
- **Selection effect on which families survive.** Families that yield many pairs are the large, diverse ones where matchable decoys are plentiful — which is a mild bias in what v4 measures.
- **Conditioning sets are inherited from v2**, so the subsampling artifact of [§5](#5-positive-selection-is-a-subsampling-side-effect) carries over.
- **False-negative protection is measured against the prompt only.** Decoys ≥70% identical to a conditioning member are excluded; non-conditioning family members are not checked directly. Since conditioning sequences are 70%-cluster representatives, this catches the dangerous cases, but it is weaker than a full family-vs-all check.

---

# Design history

Three earlier designs, each of which fixed the previous one's stated problem and failed for a new reason. All numbers below were measured on the shipped data, not assumed.

## v1 — bin-matched pairs

`data/benchmark/` · 201 instances / 4,810 pairs · `scripts/04_construct_benchmarks.py`

Each positive was matched to a negative whose maximum local percent identity to the prompt fell in the same 10-point bin. The intent: a model that merely thresholds on sequence identity scores at chance.

**Observed problem.** Within a bin, decoys clustered toward the lower edge, so matched negatives were on average *slightly less* similar than their positives — a residual bias flattering any homology-based model.

## v2 — "negative slightly higher"

`data/benchmark_v2/` · 152 instances / 3,445 pairs · `candidates.negative_higher: true`

Fixed v1's bias by pairing each positive with the closest decoy of *strictly greater* identity. In 97.2% of v2 pairs the negative's recorded identity exceeds the positive's, by 1.21 pp on average. The stated homology baseline sits at ~0.32 — below chance.

**This looked airtight and was not.** Two independent problems:

### 1. Matching local identity controls the wrong variable

`pident` from `mmseqs convertalis` is percent identity **over the aligned region of a local alignment**. Step 03 ran `mmseqs search` with no `-c` flag, so the coverage requirement was 0, and step 04 read only `query, target, pident` — discarding `alnlen`, `qlen`, `tlen`, `evalue`.

A 70-residue hit at 47% identity and a 400-residue hit at 46% identity were therefore treated as equivalent. They are not:

| Best-identity hit to the prompt | median alignment length | median coverage |
|---|---|---|
| Positives | 272 aa | 0.91 |
| Negatives | 74 aa | 0.16 |

So statistics the pipeline never optimised against separate the classes easily (mean within-instance AUROC):

| Statistic | v1 | v2 |
|---|---|---|
| `pident` — *the matched variable* | 0.609 | 0.397 |
| **alignment coverage** | **0.902** | **0.914** |
| coverage-adjusted identity | 0.893 | 0.873 |
| E-value | 0.809 | 0.789 |
| **learned adversary** | **0.940** | **0.936** |

The adversary's *median* is **1.000** in both: for most instances a six-feature logistic regression fitted inside that instance separates the classes perfectly, out-of-fold. It is *stronger* on the largest instances (0.955 for v1 instances with ≥50 pairs), so this is not small-sample overfitting.

**v2 did not improve on v1** (0.940 → 0.936). It removed a 1 pp bias in a variable that was never the leak.

A model reporting AUROC 0.75 on v2 would be presented as beating homology while performing *worse than sorting by MMseqs2 E-value*.

### 2. The validation baseline could not fail

`nearest_neighbor_baseline()` in `05_validate_benchmark.py` scored candidates by `max_pident_to_conditioning` — read from `labels.tsv`, i.e. precisely the quantity `select_higher_decoys()` had maximised for negatives. AUROC below 0.5 was an algebraic identity, not evidence.

This is the general lesson, and it is why `evaluation/leakage_gate.py` exists: **a QC baseline must use a statistic that construction never saw.**

### Three things that also had to be true, and were not

- **Single-statistic checks are insufficient.** Instance `2_7_13_3` has coverage AUROC 0.320 — a leak in the *opposite* direction — and would pass any "is coverage balanced?" test. Its learned adversary scores 0.995. Only a check over *combinations* catches it.
- **No scalar can be matched safely.** Equalising one frees the others: match `pident` → coverage leaks at 0.892; match coverage-adjusted identity → `pident` leaks at 0.738; match E-value → `pident` leaks at 0.738.
- **The greedy argmax was not the cause.** Replacing `select_higher_decoys`'s argmax with stratified random sampling inside matched bins changes the adversary by <0.02. The leak is a property of *which variable is matched*, not of how the decoy is picked.

### 3. `negative_higher` targeted a below-chance baseline, which is wrong

Driving the homology baseline to 0.32 does not make the benchmark harder — it makes it **anti-correlated with homology**. A good conditional PLM's likelihood is legitimately homology-correlated; it *should* score family members higher. On a set where homology sits at 0.32, such a model is dragged below chance while a functionally-blind, homology-*inverted* scorer looks excellent. v4 targets 0.5.

## v3 — matched on profile-HMM bitscore

`data/benchmark_v3/` · 22 instances / 729 pairs · `scripts/06_profile_bitscores.py`, `scripts/07_construct_benchmark_v3.py`

The principled response to §1: *match on the score of the baseline you intend to nullify.* Build a profile HMM from the prompt, search Swiss-Prot, match each positive to the nearest-bitscore negative. Profile bitscore is coverage-sensitive by construction, so the v2 confound cannot arise from it. Positives restricted to experimental evidence; negatives required to carry experimental evidence of a *different* reaction.

The matching worked exactly as designed — and the benchmark was still solvable:

| Statistic | v3 |
|---|---|
| **`hmm_bitscore`** — *the matched variable* | **0.518** |
| `pident` | 0.884 |
| coverage-adjusted identity | 0.888 |
| coverage | 0.856 |
| **learned adversary** | **0.898** |

**Why.** Bitscore is a scalar summary of roughly *per-residue identity × aligned length*. Equalise the product and the factors stay free. The matched negatives turned out to be long proteins matching weakly across many positions; the positives were shorter with high per-residue identity. Identical bitscore, entirely different alignments.

**And it cannot be repaired by matching on more covariates**, because that empties the benchmark:

| Joint caliper | Matchable positives | Instances with ≥5 pairs |
|---|---|---|
| bitscore only | 53.0% | 19 / 22 |
| bitscore + `pident` | 34.3% | 9 / 22 |
| bitscore + coverage | 40.3% | 12 / 22 |
| **bitscore + `pident` + coverage** | **25.3%** | **5 / 22** |

## What v3 taught, and how v4 used it

v3 established that **no scalar homology summary can be matched safely** — and that the failure is structural, not a bug in the choice of statistic.

v4's insight was to stop matching harder and instead **change the pool**. The leak in v1–v3 came from a population of candidates whose similarity to the prompt is a short spurious local alignment. Those sequences cannot be balanced against, but they *also would never be retrieved* by any real screening method. Excluding them by a coverage floor — rather than trying to match them — is what finally brought the sequence statistics to chance.

Two further findings from building v4:

- **Homology-propagated positives make things worse.** Allowing non-experimental positives raises the adversary from 0.65 to 0.76 *and degrades the matching itself* (`S` drifts from 0.55 to 0.61). Propagated EC labels are assigned **by** homology, so they sit at high identity where no decoy exists to match them. [§4](#4-ec-labels-are-not-a-reliable-target)'s circularity, visible as a number.
- **The gate's own data source can leak.** The first v4 gate run reported an adversary of 0.872 — but it was reading the step-03 MMseqs output, where cross-EC3 similarities exist only between 50% cluster representatives, so 11.2% of negatives were absent versus 3.7% of positives. Bare *presence* in that file gives AUROC 0.62. Imputing absences to 0.0 handed the adversary a signal about the *search plan*, not the data. With the uniform search (`--retrieval-tsv`) the honest number is 0.740.

---

# Known confounds (all versions)

Numbered issues referenced above. Items 1–3 are fixed in v4; 4–8 are not.

### 4. EC labels are not a reliable target

The deepest problem, and unlike the others it cannot be fixed by better matching.

**Most EC numbers are propagated by homology.** Swiss-Prot is manually reviewed, but for most entries the EC number is assigned by sequence similarity (`ECO:0000250`, `ECO:0000256`), not measured. A large fraction of naive positive labels were therefore assigned **because the protein looked homologous to a family member** — the ground truth is partly a function of the confound being controlled for.

v4 addresses this by requiring `cc_catalytic_activity_exp` on both positives and negatives (48,496 of 280,036 reviewed EC entries; 1,033 EC4 families retain ≥10 experimentally-evidenced members, so scale is not the obstacle).

**Negatives need contrary evidence, not absent annotation.** "Not annotated with the target EC" ≠ "does not catalyse the target reaction"; under-annotation is Swiss-Prot's normal state. v4 negatives are *experimentally shown to catalyse something else*.

**Two parser leaks (fixed in `utils/family.py`).** `parse_ec_annotations()` rejected EC components equal to `"n"`, but preliminary EC numbers are formatted `3.6.5.n1` — so 1,305 entries with preliminary ECs passed as legitimate families and tier-1 decoys (`3.6.5.n1` alone has 707 members). Separately, proteins with a partial EC alongside a complete one (`1.1.1.-; 2.7.1.5`) entered the decoy pool via the complete EC while their partial annotation could cover the target; 41 of 3,445 v2 negatives are of this form.

**EC4 is a reaction class, not a homology class.** The same EC4 is catalysed by non-homologous proteins (convergent evolution), and a single EC3 spans multiple superfamilies. "Same EC3, different EC4" therefore does not reliably mean *biologically confusable* — often it means *no detectable homology at all*, which is exactly what the v2 coverage figures showed. The tier hierarchy is a proxy for difficulty that does not measure difficulty.

### 5. Positive selection is a subsampling side-effect

Conditioning sets are 70%-identity cluster representatives, capped at 100 by `random.sample()`; positives are the leftovers. But only **52 of 3,445** v2 positives have their own cluster representative in the conditioning set, and 134 of 152 instances have exactly 100 conditioning sequences.

So positives are almost entirely the *orphans of clusters whose representative was randomly discarded by the size cap* — a subsampling accident, not a stated criterion. Holding out whole 70% clusters would be the principled equivalent. **v4 inherits this.**

### 6. Candidates are not independent

Within a v2 instance only ~71% of positives occupy distinct 70% clusters (e.g. `2_7_4_25`: 106 positives spanning 53 clusters, one contributing 12). Negatives are ~76% unique and are reused across instances. Effective sample size is well below the pair count, so **bootstrap by family, not by pair**.

### 7. Decoy tier confounds with measurement quality (v1–v3)

Tier-1 similarities came from a full within-EC3 all-vs-all; tier 2–4 similarities came *only* from the 50%-identity cluster-representative search. A distant decoy had a recorded similarity only if **both** it and the conditioning member happened to be 50% representatives; otherwise `compute_max_sim_to_conditioning()` silently returned `0.0` and the sequence dropped out.

Consequences: identity was systematically under-measured for distant tiers; the tier-2–4 decoy pool was silently restricted to cluster representatives; and `find_false_negative_ids()` was nearly inert there (only 3,647 decoys excluded across all 152 families, out of pools of order 10⁵).

**Fixed in v4** by `08_retrieval_search.py`, which measures every (prompt, database) pair uniformly.

### 8. Metrics dominated by tiny instances

53 of 152 v2 instances have ≤5 pairs. At 2 pairs, a per-instance AUROC can only take values in {0, 0.25, 0.5, 0.75, 1}. `evaluation/metrics.py` reports an unweighted mean across instances, so much of the reported signal is quantisation noise while the 10 largest instances hold 33.2% of the pairs. v4 requires ≥5 pairs per instance, but see the 44%-kinase warning above.

---

## Repository structure

```
pflm_ec_bench/
├── config.yaml
├── scripts/
│   ├── 01_download_swissprot.py         # Reviewed Swiss-Prot with EC annotations
│   ├── 02_filter_families.py            # Select eligible EC4 families
│   ├── 03_build_mmseqs_db.py            # MMseqs2 databases (see §7 for its limits)
│   ├── 04_construct_benchmarks.py       # v1/v2 pair matching  [superseded]
│   ├── 05_validate_benchmark.py         # v1/v2 QC             [superseded by the gate]
│   ├── 06_download_experimental_ec.py   # Entries with experimental EC evidence
│   ├── 06_profile_bitscores.py          # MAFFT + hmmbuild + hmmsearch -> profile_hits.tsv
│   ├── 07_construct_benchmark_v3.py     # Bitscore-matched v3   [experiment; fails gate]
│   ├── 08_retrieval_search.py           # Prompt vs all Swiss-Prot, with coverage
│   ├── 09_construct_benchmark_v4.py     # v4 — USE THIS
│   └── utils/
├── evaluation/
│   ├── leakage_gate.py                  # Homology-leak gate — run before trusting anything
│   ├── metrics.py
│   ├── score_model.py
│   └── scorers/
└── data/
    ├── raw/
    ├── processed/retrieval/cond_vs_all.tsv
    ├── benchmark/       # v1  [fails gate]
    ├── benchmark_v2/    # v2  [fails gate]
    ├── benchmark_v3/    # v3  [fails gate]
    └── benchmark_v4/    # v4  ← use this
        ├── manifest.json
        └── instances/{ec}/{conditioning,candidates}.fasta, labels.tsv
```

## Related work

- **CARE** — EC classification with similarity-stratified test splits. Right evaluation culture; framed as classification, not conditional likelihood scoring.
- **CLEAN** — EC prediction under low-homology regimes (≤50% identity splits).
- **EC-Bench** — Unified evaluation of EC prediction methods. Does not test conditional models.
- **Price-149 / New-392** — Hard external test sets of experimentally validated enzymes.
- **FunFams (CATH)** — Functionally coherent subfamilies within homologous superfamilies; a possible alternative family definition for a structure-controlled variant.

None of these test whether a conditional PLM's likelihood, given a family prompt, discriminates in-family from out-of-family sequences under a controlled and *reported* homology baseline.

## Citation

If you use this benchmark, please cite the repository and UniProt/Swiss-Prot.
