# SGD Yeast GRN Inheritance Probability Model

Implementation of the mathematical framework from:

> **Counting Subnetworks Under Gene Duplication in Genetic Regulatory Networks**
> Ashley Scruse, Jonathan Arnold, Robert Robinson
> arXiv:2405.03148v1 · University of Georgia · May 2024

Applied to *Saccharomyces cerevisiae* data from the [Saccharomyces Genome Database (SGD)](https://www.yeastgenome.org/) using 127 curated transcription factors from [YEASTRACT](https://www.yeastract.com/).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the interactive frontend
streamlit run app.py

# 3. Or use the CLI
python main.py summary
python main.py tfs
python main.py significance --k 2 --strategy largest
```

---

## What This Project Does

Gene duplication is how genomes grow in complexity. When a gene duplicates, its regulatory relationships — which transcription factors bind to it and control its expression — may or may not be inherited by the new copy. The probability of that inheritance is **π** (pi).

This project:

1. **Identifies transcription factors (TFs)** in the yeast genome and characterises their binding sites and regulatory targets
2. **Groups genes into families** based on shared GO Biological Process terms (operationalising the Scruse et al. duplication model)
3. **Estimates π⃗ = (π₁, …, πₖ)** — the per-family inheritance probability vector — using three data-driven methods
4. **Tests whether subnetwork motifs are significant** relative to the gene duplication null model

---

## Biological Questions Answered

### What are transcription factor binding sites binding?
TF binding sites (also called *cis-regulatory elements* or *TFBS*) are short DNA sequences in **gene promoter regions**. A TF protein binds its TFBS via a DNA-binding domain (GO:0003677, GO:0000981) and either activates or represses transcription of the downstream gene. When a TF gene duplicates, the question is whether the new copy retains the same TFBS in its targets' promoters — that retention probability is **πᵢ**.

### Which transcription factor regulates which gene?
Inferred from shared GO Biological Process terms: if a TF and a gene share a process annotation, the TF is a candidate regulator. The 127 YEASTRACT-curated TFs are mapped to 6,446 target genes across 120,000+ GO annotations.

### What is the inheritance probability vector?
Estimated three ways — see [Estimation Methods](#estimation-methods) below.

---

## Mathematical Framework

All functions are implemented in [model/scruse_math.py](model/scruse_math.py).

| Result | Equation | What it gives you |
|--------|----------|-------------------|
| **Theorem 1** | `E[│M(n)│] = Γ(n+k)Γ(m) / [Γ(n)Γ(m+k)]` | Expected motif count under **Full Duplication** (π⃗ = 1) |
| **Theorem 4** | `E[│M(n)│] = Γ(π̂+n)Γ(m) / [Γ(π̂+m)Γ(n)]` | Expected count under **Partial Duplication**; depends only on π̂ = Σπᵢ |
| **Lemma 4** | `f(p,s) = Γ(p+s) / [Γ(s)Γ(p+1)]` | Expected single-gene motif instances given family size s and π = p |
| **Theorem 6** | Inclusion-exclusion over all 2^k subsets | Second moment under **Binary Inheritance** (maximises variance, Theorem 5) |
| **Corollary 2** | `Var = E[│M│²] − E[│M│]²` | Variance under Full Duplication |
| **Corollary 16** | Variance under Binary Inheritance | Used to construct the significance test |
| **Theorem 8** | Pólya urn probability | Exact distribution of gene family sizes |

**Growth rates** (from Theorem 4):
- Full Duplication: `Θ(nᵏ)` — degree equals motif size k
- Partial Duplication: `Θ(n^π̂)` — exponent equals total inheritance probability

---

## Estimation Methods

### Method 1 — Evidence-based
Maps SGD evidence codes to a π prior. Experimental evidence → high π; automated annotation → low π.

| Evidence Code | Meaning | π prior |
|---------------|---------|---------|
| IDA | Inferred from Direct Assay | 0.90 |
| IMP | Inferred from Mutant Phenotype | 0.82 |
| IGI | Inferred from Genetic Interaction | 0.72 |
| IBA | Biological Aspect of Ancestor | 0.58 |
| IEA | Inferred from Electronic Annotation | 0.30 |
| ND  | No Data | 0.10 |

### Method 2 — MLE (Theorem 4 inversion)
Given an observed motif count |M(n)|, numerically solves:

```
Γ(π̂+n)Γ(m) / [Γ(π̂+m)Γ(n)] = observed_count
```

for π̂, then distributes π̂ across families proportional to evidence weights.

### Method 3 — SNP divergence proxy
Uses `sgd_YFL039C_inheritance_vectors.csv` as a demonstration. Higher sequence divergence from the reference strain → older regulatory link → less likely to be inherited:

```
πᵢ ≈ 1 − (pct_alt / 100)
```

This mirrors the paper's MITE/rice genome application (Section 9.2): greater MITE sequence divergence dates older regulatory links.

---

## Data Files

All data lives in `sgd_yeast_data/sgd_yeast_data/`:

| File | Rows | Contents |
|------|------|----------|
| `sgd_transcription_factors.csv` | 732 (127 YEASTRACT-curated used) | TF names, GO terms, evidence codes, activator/repressor flags |
| `sgd_go_annotations_full.csv` | ~120,000 | GO annotations for all 6,446 yeast genes |
| `sgd_tf_go_annotations.csv` | ~1,942 | GO annotations specifically for TFs |
| `sgd_chromosome_lengths.csv` | 16 | Chromosome lengths (12.07 Mb total genome) |
| `sgd_YFL039C_inheritance_vectors.csv` | 11 | Per-strain SNP divergence for gene YFL039C |
| `sgd_YFL039C_snps.csv` | 9 | Individual SNP positions for YFL039C |

---

## Project Structure

```
sgd_yeast_data/
├── model/
│   ├── scruse_math.py          Core math: Theorems 1-8, Lemmas, Corollaries
│   ├── data_loader.py          CSV loading with evidence score mapping
│   ├── tf_network.py           TF identification, binding sites, TF→gene network
│   ├── gene_families.py        GO-based family grouping, m/n/d parameters
│   ├── inheritance_estimator.py  Three π estimation methods + significance test
│   ├── consensus_loader.py     YEASTRACT TF↔consensus sequences (127 TFs)
│   └── __init__.py
├── sgd_yeast_data/             Raw CSV data files
├── app.py                      Streamlit frontend (5 tabs)
├── main.py                     Command-line interface
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `streamlit`, `pandas`, `numpy`, `scipy`, `plotly`

---

## Running the Frontend

```bash
python -m streamlit run app.py
```

Opens in your browser at `http://localhost:8501`. Five tabs:

| Tab | What it shows |
|-----|---------------|
| **Overview** | Dataset summary, key theorems, binding site explanation, chromosome lengths |
| **TF Explorer** | Browse and filter all 127 YEASTRACT TFs; binding site info; regulatory targets per TF |
| **Gene Families** | Family size distribution, Pólya urn parameters (m, n, d), Dirichlet simulation |
| **π Estimator** | Select k families, estimate π⃗ with any of the three methods, sensitivity plot |
| **Motif Significance** | Full significance test: Z-scores, p-values, null model distributions |

---

## Command-Line Interface

```bash
python main.py <command> [options]
```

### Commands

**`summary`** — Dataset overview
```bash
python main.py summary
```

**`math-demo`** — Verify the mathematical functions
```bash
python main.py math-demo
```

**`tfs`** — List transcription factors
```bash
python main.py tfs --limit 20 --dna-binding --min-evidence 0.5
```

**`families`** — Show gene families (GO process clusters)
```bash
python main.py families --min-size 3 --limit 10
```

**`binding`** — Describe what a TF's binding sites bind to
```bash
python main.py binding --tf ABF1
python main.py binding --tf GAL4
```

**`estimate`** — Estimate π⃗ for a set of gene families
```bash
python main.py estimate --genes GO:0006355 GO:0045944 --method evidence
python main.py estimate --genes GO:0006355 GO:0045944 --method mle --observed 500
python main.py estimate --genes GO:0006355 GO:0045944 --method snp
```

**`significance`** — Run a full subnetwork motif significance test
```bash
# k=2 motif, 2 largest families, evidence-based π
python main.py significance --k 2 --strategy largest

# k=3 motif with a specific observed count
python main.py significance --k 3 --strategy highest_ev --observed 1000
```

Options for `--strategy`: `largest` | `highest_ev` | `balanced` | `random`

---

## Example Output

```
python main.py significance --k 2 --strategy largest

  Motif size k = 2,  m = 8,  n = 856
  Selected families: ['GO:0045944', 'GO:0006357']

  Observed count:              41829
  Expected (Full Dup):         10188.78      [Theorem 1]
  Expected (Partial Dup):      16.75         [Theorem 4, π̂=0.60]
  Variance (Full Dup):         165,769,603   [Corollary 2]
  Variance (Binary Inherit.):  540.56        [Corollary 16]
  Z-score  (Full Dup):         2.46          p = 0.014  *
  Z-score  (Partial Dup):      1798.38       p < 0.001  ***

  Interpretation:
  The observed motif count is significantly over-represented relative to
  the Partial Duplication null model (fold change 2497x, p ≈ 0).
  This pattern is consistent with positive selection for this regulatory wiring.
```

---

## Key Concepts from the Paper

**Subnetwork motif** — A gene-family-specific regulatory substructure. Unlike network motifs (defined up to graph isomorphism), subnetwork motifs label each node with a specific gene family. Two topologically identical patterns involving different families are counted as *distinct* subnetwork motifs.

**Full Duplication (π⃗ = 1)** — Every duplication event deterministically inherits all regulatory links. `|M(n)| = c₁ × c₂ × … × cₖ` (Cartesian product of family sizes).

**Partial Duplication (0 ≤ π⃗ ≤ 1)** — Each regulatory link is independently inherited with probability πᵢ per family. `|M(n)|` is a random subset of the Cartesian product.

**Binary Inheritance** — A refinement of Partial Duplication in which all instances sharing a common gene either all inherit or all fail together. Theorem 5 proves this maximises `E[|M(n)|²]` over all refinements, making it the canonical choice for the variance calculation and significance test.

**Pólya urn connection** — The duplication process is a multi-colour Pólya urn (Theorem 8). Family proportions cᵢ/n converge almost surely to a Dirichlet(1,…,1) distribution.

---

## Reference

Scruse, A., Arnold, J., & Robinson, R. (2024).
*Counting Subnetworks Under Gene Duplication in Genetic Regulatory Networks.*
arXiv:2405.03148v1.
