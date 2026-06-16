# SGD Yeast GRN Inheritance Probability Model

Implementation of the mathematical framework from:

> **Counting Subnetworks Under Gene Duplication in Genetic Regulatory Networks**
> Ashley Scruse, Jonathan Arnold, Robert Robinson
> arXiv:2405.03148v1 · University of Georgia · May 2024

Applied to *Saccharomyces cerevisiae* using curated data from [SGD](https://www.yeastgenome.org/), [YEASTRACT](https://www.yeastract.com/), and [JASPAR 2024](https://jaspar.elixir.no/), and extended with cross-species analysis across **1,154 yeast genomes** from the [Y1000+ dataset](https://y1000plus.wei.wisc.edu/) (Opulente et al. 2024, *Science*).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the interactive frontend
python -m streamlit run app.py

# 3. Or use the CLI
python main.py summary
python main.py tfs
python main.py significance --k 2 --strategy largest
```

The app opens at `http://localhost:8501`. Cross-species data (π₂, π₃, π₄) is generated automatically in the background on first launch — see [Y1000+ Pipeline](#y1000-pipeline) for details.

> **Note:** Utility scripts have moved to `scripts/`. Use `python scripts/utils/check_progress.py`, `python scripts/utils/reset_y1000_generation.py`, and `python scripts/data/download_y1000plus.py` instead of the root-level paths shown in older docs.

---

## What This Project Does

Gene duplication is how genomes grow in complexity. When a gene duplicates, its regulatory relationships — which transcription factors bind to it, and control its expression — may or may not be inherited by the new copy. The probability of that inheritance is **π** (pi).

This project estimates **π⃗ = (π₁, …, πₖ)**, the per-family inheritance probability vector, using seven complementary methods across two data tiers:

**SGD-based methods** (run instantly, no extra data needed):
1. Evidence-based: maps SGD experimental evidence codes to a π prior
2. MLE: numerically inverts Theorem 4 given an observed motif count
3. SNP divergence: uses sequence divergence at gene YFL039C as a proxy for link age
4. Consensus-adjusted: adjusts π by YEASTRACT binding-sequence flexibility

**Y1000+ cross-species methods** (generated automatically in the background):
5. **π₂ — Sequence homology**: pairwise protein identity between TF paralogs
6. **π₃ — TFBS conservation**: fraction of 1,154 yeast genomes that retain a significant PWM hit upstream of the orthologous gene
7. **π₄ — SNP at binding sites**: IC-weighted polymorphism rate at the exact binding site positions

---

## Mathematical Framework

All functions are in [model/scruse_math.py](model/scruse_math.py).

| Result | Equation | What it gives you |
|--------|----------|-------------------|
| **Theorem 1** | E[&#124;M(n)&#124;] = Γ(n+k)Γ(m) / [Γ(n)Γ(m+k)] | Expected motif count under **Full Duplication** (π⃗ = 1) |
| **Theorem 4** | E[&#124;M(n)&#124;] = Γ(π̂+n)Γ(m) / [Γ(π̂+m)Γ(n)] | Expected count under **Partial Duplication**; depends only on π̂ = Σπᵢ |
| **Corollary 2** | Var = E[&#124;M&#124;²] − E[&#124;M&#124;]² | Variance under Full Duplication |
| **Corollary 16** | Binary Inheritance variance | Drives the significance z-score |
| **Theorem 8** | Pólya urn probability | Exact distribution of gene family sizes |

**Growth rates** (Theorem 4): Full Duplication → Θ(n^k) ; Partial → Θ(n^π̂).

---

## The Seven Estimation Methods

### Methods 1–4: SGD-based (instant)

| # | Name | Data source | Key idea |
|---|------|-------------|----------|
| 1 | Evidence-based | SGD evidence codes | IDA (direct assay) → π = 0.90; IEA (automated) → π = 0.30 |
| 2 | MLE — Theorem 4 | Observed motif count | Numerically inverts `Γ(π̂+n)Γ(m)/[Γ(π̂+m)Γ(n)] = count` for π̂ |
| 3 | SNP divergence | `sgd_YFL039C_snps.csv` | `πᵢ ≈ 1 − (pct_alt / 100)` across strains of gene YFL039C |
| 4 | Consensus-adjusted | YEASTRACT consensus | More flexible IUPAC binding sequences → binding tolerates divergence → higher π |

### Methods 5–7: Y1000+ cross-species (auto-generated)

| # | Name | Family definition | Formula |
|---|------|-------------------|---------|
| 5 | π₂ — Sequence homology | Protein identity clusters (30/50/80%) | `π₂ = pct_identity / 100` |
| 6 | π₃ — TFBS conservation | Genes sharing a TF regulator | `π₃ = (genomes with significant PWM hit) / (genomes with ortholog)` |
| 7 | π₄ — SNP at binding sites | Same as π₃ | `π₄ = 1 − Σ IC_weight[pos] × polymorphism_rate[pos]` |

π₃ and π₄ use JASPAR 2024 position weight matrices (PWMs) to score **1,000 bp upstream of the translational start** for each gene across the Y1000+ species panel. IC-weighting in π₄ means that mutations at high-information-content (critical) positions count more heavily.

The Y1000+ π Estimators tab also includes a **gene-centric upstream retention figure** that aggregates π₃ by target gene rather than by TF→gene edge: for each gene in the *S. cerevisiae* network, it plots the mean retention fraction across all TFs whose binding sites were scanned in that gene's 1,000 bp upstream window, with ±1 SD error bars across TFs. This lets you identify which genes consistently retain upstream binding sites across yeasts regardless of which TF is doing the binding.

---

## Y1000+ Pipeline

### Data

Download the Y1000+ archives from [Figshare](https://figshare.com/articles/dataset/22802147) (or run `download_y1000plus.py`) and place them in `y1000plus_data/`:

| Archive | Size | Contents |
|---------|------|----------|
| `y1000p_gff3_files.tar.gz` | 360 MB | GFF3 gene annotations for all 1,154 species |
| `y1000p_gtf_files.tar.gz` | — | GTF gene annotations (alternative annotation format) |
| `y1000p_pep_files.tar.gz` | 1.9 GB | Protein (peptide) sequences |
| `y1000p_cds_files.tar.gz` | 2.7 GB | CDS nucleotide sequences |
| `y1000p_genome_files.zip` | 4.4 GB | Genome FASTA assemblies (`.fas.gz` inside `.zip`) |

The GFF3 archive is bulk-extracted to `y1000plus_data/processed/y1000p_gff3_files/` on first use. All other species files are extracted on-demand.

### Auto-generation of π₂/π₃/π₄

When you open the Streamlit app, a background thread automatically generates the three cross-species π CSV files if they do not yet exist. Progress is shown live in the **Y1000+ π Estimators** tab with per-step status pills, a progress bar, and a per-species counter that updates as each genome is processed. The app auto-refreshes every 5 seconds while generation is running.

Generation runs in this order:

| Step | Time (first run) | Time (subsequent) | Bottleneck |
|------|-----------------|-------------------|------------|
| **π₂** sequence homology | ~2 seconds | ~2 seconds | k-mer alignment of 8,001 TF pairs |
| **π₃** TFBS conservation | 15–30 minutes | 3–5 minutes | First-run genome extraction from 4.4 GB ZIP |
| **π₄** SNP at binding sites | 15–30 minutes | 3–5 minutes | Same genome extraction |

The large range for π₃/π₄ on first run is because each species' genome FASTA must be decompressed from the 4.4 GB ZIP (doubly compressed: DEFLATE + gzip). Once extracted to `y1000plus_data/processed/genomes/`, all subsequent runs read from disk and are much faster.

Results are cached as CSVs in `y1000plus_data/processed/`. Restarting the app resumes from the last completed step.

### Monitoring and resetting generation

**Check progress** from a second terminal at any time while the app is running:

```bash
python scripts/utils/check_progress.py
```

Output:
```
=======================================================
  Y1000+ Generation Status
=======================================================
  [##########--------------------] 35%
  Status  : running_pi3
  Message : TFBS conservation: species 12/48 — lachancea_thermotolerans
  Updated : 2026-06-08T14:35:02

  Output CSVs:
    pi2: done   (7,875 rows, 750 KB)
    pi3: running
    pi4: pending

  Genomes cached : 12 / 48
=======================================================
```

**Clean restart** — if generation stalls or you want to start over:

```bash
# 1. Stop the Streamlit app (Ctrl+C in its terminal)

# 2a. Reset state, keep cached genomes (faster next run):
python scripts/utils/reset_y1000_generation.py

# 2b. Full reset including genome FASTAs (re-extracts everything from ZIP):
python scripts/utils/reset_y1000_generation.py --full

# 3. Restart the app
python -m streamlit run app.py
```

The reset script removes the progress/lock files and all three output CSVs, but keeps the already-extracted genome FASTAs by default so the next run is significantly faster.

### S. cerevisiae reference

The Y1000+ dataset includes two annotation files for *S. cerevisiae* S288C:
- `.sgd.gff3 / .sgd.pep` — curated SGD reference annotation (used by this pipeline)
- `.final.gff3 / .final.pep` — BRAKER ab initio annotation (used for all other species)

The SGD GFF3 uses `chrI/chrII/…` chromosome names while the genome FASTA uses NCBI accessions (`NC_001133/NC_001134/…`). The mapping is hardcoded in `model/promoter_extractor.SGD_CHR_MAP`.

### Phylogenetic species subset

π₃ and π₄ scan a curated 48-species subset (`REPRESENTATIVE_SUBSET` in `model/y1000plus_loader.py`) spanning all major Saccharomycotina clades: *Saccharomyces* sensu stricto, Lachancea, Kluyveromyces/Eremothecium, Candida/CTG clade, Pichia/Komagataella, Debaryomycetaceae, and deep outgroups. This balances phylogenetic coverage with compute time.

Note: the Y1000+ archive uses strain-prefixed IDs for many species (e.g. `yHMPu5000034678_lachancea_thermotolerans_180604` rather than `lachancea_thermotolerans`). `REPRESENTATIVE_SUBSET` uses the exact archive IDs as they appear in the manifest.

---

## Gene Family Grouping

Five methods available via the sidebar:

| Method | Basis |
|--------|-------|
| GO Biological Process | Shared biological pathway (default; paper Section 2) |
| GO Molecular Function | Shared molecular activity / binding-domain type |
| GO Cellular Component | Shared subcellular location or complex |
| JASPAR TF Class | DNA-binding domain architecture (12 classes, 177 TFs) |
| JASPAR TF Family | Finer binding-domain subtype (17 families) |

Each family starts as a singleton (minimum size = 1), consistent with Proposition 1. Model parameters: **m** = number of families, **n** = total genes, **d = n − m** = duplication events.

For the Y1000+ estimators, families are defined differently:
- **π₂**: genes clustered by protein identity at three thresholds (30%, 50%, 80%)
- **π₃/π₄**: genes sharing the same TF regulator (= TF regulon = one family)

---

## Data Files

### SGD / JASPAR / YEASTRACT (`sgd_yeast_data/sgd_yeast_data/`)

| File | Rows | Contents |
|------|------|----------|
| `sgd_transcription_factors.csv` | 732 (179 used) | TF names, GO terms, evidence codes, activator/repressor flags |
| `sgd_go_annotations_full.csv` | ~120,000 | GO annotations for 6,446 genes (all three GO aspects) |
| `sgd_tf_go_annotations.csv` | ~1,942 | GO annotations specifically for TFs |
| `sgd_chromosome_lengths.csv` | 16 | Chromosome lengths (12.07 Mb total) |
| `sgd_YFL039C_inheritance_vectors.csv` | 11 | Per-strain SNP divergence for YFL039C |
| `sgd_YFL039C_snps.csv` | 9 | Individual SNP positions for YFL039C |
| `jaspar_yeast_tfbs_2024.csv` | 177 | JASPAR 2024 yeast TFs: PFMs, IC scores, consensus sequences, TF class/family |
| `jaspar_yeast_pfm_long.csv` | 5,708 | Long-format PFM table: one row per (TF, position, nucleotide) |
| `jaspar_yeastract_crossref.csv` | 127 | JASPAR × YEASTRACT name cross-reference (115 exact, 6 fuzzy, 6 no-match) |
| `yeastract_consensus.csv` | — | IUPAC consensus binding sequences for 127 YEASTRACT TFs |

### Y1000+ processed outputs (`y1000plus_data/processed/`)

| File | Contents |
|------|----------|
| `y1000plus_manifest.csv` | One row per (species, annotation_type): archive paths, species names, reference flag |
| `promoters_Scerevisiae_S288C.fasta` | 1,000 bp upstream sequences for 6,579 S. cerevisiae genes |
| `pi2_sequence_homology.csv` | Pairwise protein identity for all TF pairs; cluster labels at 30/50/80% |
| `pi3_tfbs_conservation.csv` | Per TF→gene retention fraction across the species subset |
| `pi3_pairwise_histogram.csv` | Per-family pairwise sharing statistics (π̂₃ family-level estimate) |
| `pi4_snp_binding_sites.csv` | IC-weighted polymorphism rate and π₄ per TF→gene edge |

---

## Project Structure

```
sgd_yeast_data/
│
├── model/
│   ├── scruse_math.py            Core math: Theorems 1–8, Lemmas, Corollaries
│   ├── data_loader.py            SGD CSV loading, evidence score mapping, TF sets
│   ├── tf_network.py             TF identification, binding sites, TF→gene network
│   ├── gene_families.py          Family grouping (5 methods), m/n/d parameters
│   ├── inheritance_estimator.py  All 7 π estimation methods + significance tests
│   ├── consensus_loader.py       YEASTRACT TF↔consensus sequences (127 TFs)
│   ├── jaspar_loader.py          JASPAR 2024 PFMs, IC factors, YEASTRACT cross-ref
│   │
│   ├── y1000plus_loader.py       Manifest builder, on-demand archive extraction
│   ├── promoter_extractor.py     1,000 bp upstream extraction (strand-aware, GFF3 parser)
│   ├── pi2_sequence_homology.py  π₂: pairwise protein sequence identity
│   ├── pi3_tfbs_conservation.py  π₃: PWM conservation scan across Y1000+ species
│   ├── pi4_snp_binding.py        π₄: IC-weighted SNP rate at binding site positions
│   └── y1000plus_generator.py    Background thread manager for auto-generating CSVs
│
├── assets/
│   ├── subnetwork_motifs.png           Sidebar figure: four subnetwork motifs A–D
│   ├── sgd_logo.png                    SGD logo (local fallback)
│   ├── y1000plus_species_labeled.png   Phylogenetic species panel
│   ├── y1000plus_phylogeny.png         ML phylogeny with branch lengths
│   └── generate_figure1.py             Script to regenerate subnetwork_motifs.png
│
├── scripts/
│   ├── analysis/
│   │   ├── _check_consensus.py         Dev: inspect YEASTRACT/JASPAR consensus agreement
│   │   ├── _generate_doc.py            Dev: auto-generate documentation snippets
│   │   └── _get_estimates.py           Dev: batch π estimation across TF families
│   ├── data/
│   │   ├── download_y1000plus.py       Download Y1000+ archives from Figshare
│   │   └── fetch_jaspar_yeast.py       Fetch/refresh JASPAR 2024 yeast PFMs
│   └── utils/
│       ├── check_progress.py           Check Y1000+ generation status from terminal
│       ├── reset_y1000_generation.py   Reset generation state for a clean restart
│       └── _inspect_yeastract.py       Dev: inspect YEASTRACT binding-sequence data
│
├── sgd_yeast_data/               SGD, JASPAR, YEASTRACT CSV data files
├── y1000plus_data/               Y1000+ archives + processed outputs
│   └── processed/                Extracted files and generated π CSVs
│       ├── gff3/                 Extracted per-species GFF3 annotation files
│       └── pep/                  Extracted per-species peptide FASTA files
│
├── app.py                        Streamlit frontend (9 tabs)
├── main.py                       Command-line interface
├── requirements.txt
└── README.md
```

---

## App — Nine Tabs

```bash
python -m streamlit run app.py
```

| Tab | Contents |
|-----|----------|
| **Overview** | Card-by-card map of all tabs — what each does and what data it uses |
| **Introduction** | Plain-language biology primer: what duplication is, what π means |
| **Methodology** | Mathematical background: Theorems 1–8, Pólya urn, Full vs Partial Duplication |
| **TF Explorer** | Browse 179 TFs; evidence codes, GO terms, JASPAR PFMs, YEASTRACT consensus sequences, sequence logos, regulatory targets |
| **Gene Families** | Family-size distributions, m/n/d parameters, Dirichlet simulation — five grouping methods |
| **π Estimator** | Select k families; estimate π⃗ by any of the four SGD-based methods; sensitivity plot; expandable methodology guide explaining the Gamma-ratio formula and the biological rationale behind each estimation method |
| **Motif Significance** | Z-scores, p-values, null distributions; predictive forward forecast of motif growth; expandable methodology guide explaining the significance framework, null model hierarchy, and the role of Binary Inheritance variance |
| **Y1000+ π Estimators** | π₂/π₃/π₄ results with live generation status; per-TF retention histogram; per-gene upstream retention figure (1,000 bp from translational start, averaged across all targeting TFs); pairwise shared binding-site bar chart (y-axis fixed to [0,1]); π₃ vs π₁ scatter; phylogenetic context; expandable methodology guide on cross-species π inference |
| **Method Estimation Test** | Synthetic validation of all four SGD-based estimation methods against known π profiles. Select a method (Method 1 – Evidence-based, Method 2 – Moment Estimation, Method 3 – SNP Divergence, or Method 4 – Consensus-adjusted), a motif size k (3 or 4), and — for Methods 1 and 4 — a set of k TFs. Each method estimates π from its real data source and the result is compared against Linear and Quadratic true-π profiles via per-family MSE. Every method includes a written interpretation of how the test works in context, what the MSE reveals about accuracy, and what structural limitations constrain recovery. The π Estimator tab's methodology expander contains a **clickable button** (JavaScript tab-jump) that navigates here directly; works both locally and on Streamlit Cloud. |
| **Glossary & References** | Definitions of all model terms and full citations |

### Sidebar settings

| Setting | Default | Effect |
|---------|---------|--------|
| Min. evidence score | 0.0 | Filter TFs below this quality threshold |
| Max TFs for network | 100 | Cap on network construction (performance) |
| Min family size | 1 | Families start as singletons; 1 is the paper-consistent minimum |
| Family grouping | GO Biological Process | Switches all family-based calculations simultaneously |

---

## Utility Scripts

| Script | Usage | Description |
|--------|-------|-------------|
| `scripts/utils/check_progress.py` | `python scripts/utils/check_progress.py` | Show live Y1000+ generation status: which CSVs are done, how many genomes are cached, current species being processed |
| `scripts/utils/reset_y1000_generation.py` | `python scripts/utils/reset_y1000_generation.py` | Wipe generation state for a clean restart (keeps cached genomes); use `--full` to also delete genome FASTAs |
| `scripts/data/download_y1000plus.py` | `python scripts/data/download_y1000plus.py` | Download all Y1000+ archives from Figshare into `y1000plus_data/` |
| `scripts/data/fetch_jaspar_yeast.py` | `python scripts/data/fetch_jaspar_yeast.py` | Fetch/refresh JASPAR 2024 yeast TF PFMs and write to `sgd_yeast_data/` |

---

## Command-Line Interface

```bash
python main.py <command> [options]
```

| Command | Example | Description |
|---------|---------|-------------|
| `summary` | `python main.py summary` | Dataset overview: TF counts, GO annotations, genome size |
| `math-demo` | `python main.py math-demo` | Verify Theorems 1–8 with test values |
| `tfs` | `python main.py tfs --dna-binding --min-evidence 0.7` | List TFs with filters |
| `families` | `python main.py families --min-size 2 --limit 10` | Show gene families |
| `binding` | `python main.py binding --tf GAL4` | Describe what a TF's binding sites regulate |
| `estimate` | `python main.py estimate --genes GAL4 GCN4 --method evidence` | Estimate π⃗ |
| `significance` | `python main.py significance --k 2 --strategy largest` | Full significance test |

Options for `--method`: `evidence` | `mle` | `snp` | `consensus`
Options for `--strategy`: `largest` | `highest_ev` | `balanced` | `random`

---

## Installation

```bash
pip install -r requirements.txt
```

Core requirements: `streamlit`, `pandas`, `numpy`, `scipy`, `plotly`, `pillow`, `biopython`, `requests`, `tqdm`

---

## Key Concepts

**Subnetwork motif** — A labelled regulatory substructure. Two topologically identical patterns involving different gene families are distinct subnetwork motifs (unlike classical network motifs which are defined up to isomorphism).

**Full Duplication (π⃗ = 1)** — All regulatory links are deterministically inherited. `|M(n)| = c₁ × c₂ × … × cₖ`.

**Partial Duplication (0 ≤ π⃗ ≤ 1)** — Each link is inherited independently with probability πᵢ. Expected count follows Theorem 4.

**Binary Inheritance** — A refinement where all instances sharing a common gene inherit or fail together. Theorem 5 proves this maximises `E[|M(n)|²]`, making it the canonical variance model for the significance test.

**Pólya urn** — The duplication process is a multi-colour Pólya urn (Theorem 8). Family proportions converge almost surely to a Dirichlet(1,…,1) distribution.

**Retention fraction (π₃)** — For a TF→gene edge, the proportion of Y1000+ species that have a significant JASPAR PWM hit in the 1,000 bp upstream of the translational start of the orthologous gene. Interpreted as an empirical estimate of the per-edge inheritance probability.

**Gene-centric upstream retention** — Retention fraction averaged across all TFs that target a given gene. Shown in the Y1000+ tab as a sorted bar chart, this reveals which genes consistently retain any upstream binding signal across species, independent of which specific TF is driving it.

**IC-weighted polymorphism (π₄)** — Positions in a binding site are weighted by their PWM information content. A mutation at a highly conserved, high-IC position contributes more to the polymorphism score than one at a degenerate position.

---

## References

**[1]** Scruse, A., Arnold, J., & Robinson, R. (2024). *Counting Subnetworks Under Gene Duplication in Genetic Regulatory Networks.* arXiv:2405.03148v1. https://arxiv.org/abs/2405.03148

**[2]** Opulente, D. A., Langdon, Q. K., Buh, K. V., et al. (2024). *Genomic survey of 1,154 Saccharomycotina yeasts.* Science, 384, eadq2116. https://doi.org/10.1126/science.adq2116
— *Y1000+ dataset: 1,154 yeast genome assemblies, GFF3 annotations, and protein sequences used for cross-species π estimation.*

**[3]** Castro-Mondragon, J. A., Riudavets-Puig, R., Rauluseviciute, I., et al. (2022). *JASPAR 2022: the 9th release of the open-access database of transcription factor binding profiles.* Nucleic Acids Research, 50(D1), D165–D173.
— *JASPAR 2024 CORE yeast TF collection: 177 position frequency matrices used for PWM scanning in π₃ and π₄.*

**[4]** Harbison, C. T., Gordon, D. B., Lee, T. I., et al. (2004). *Transcriptional regulatory code of a eukaryotic genome.* Nature, 431, 99–104. https://doi.org/10.1038/nature02800

**[5]** Ren, B., Robert, F., Wyrick, J. J., et al. (2000). *Genome-Wide Location and Function of DNA Binding Proteins.* Science, 290, 2306–2309. https://doi.org/10.1126/science.290.5500.2306
