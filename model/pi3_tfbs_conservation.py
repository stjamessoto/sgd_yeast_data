"""
pi3_tfbs_conservation.py
Estimates π₃: regulatory inheritance probability from TFBS conservation
across the Y1000+ 1,154-species panel (Opulente et al. 2024).

Algorithm (per TF, v2 — ortholog-anchored):
  1. Scan S. cerevisiae S288C promoters with the TF's JASPAR PWM to identify
     target genes (significant hit at p < 1e-4, SCORE_THRESHOLD = 6.0).
  2. For each species in the representative subset:
     a. Extract all promoters (GFF3 + genome FASTA).
     b. Load the species proteome and find the best-matching protein for
        each S. cerevisiae target gene via k-mer vote counting
        (see model/ortholog_finder.py).
     c. "Retained" for that TF→gene edge = the ortholog's promoter has a
        significant PWM hit.  If no confident ortholog is found the species
        is excluded from that gene's denominator (neither counted as
        retained nor not-retained).
  3. retention_fraction = (species with hit) / (species with confident ortholog)
  4. π̂₃ for the TF = mean retention_fraction across all its target genes.

Why this matters:
  The previous (v1) approach asked "does ANY promoter in the species have a
  hit?" and credited ALL target genes if yes.  Because TF binding motifs are
  short and degenerate, nearly every genome has at least one random match,
  inflating retention fractions to near 1.0.  Anchoring to the specific
  ortholog promoter removes this bias.

PWM scoring (FIMO-style log-odds):
  score(seq, pwm, pos) = Σ pwm[nuc][pos] over motif width positions
  Thresholded at SCORE_THRESHOLD = 6.0 (≈ p-value 1e-4 for typical motifs).

Key outputs (saved to y1000plus_data/processed/):
  pi3_tfbs_conservation.csv — one row per (TF, target_gene)
  pi3_pairwise_histogram.csv — pairwise shared binding site counts per family

Reference: Scruse, Arnold & Robinson (2024) arXiv:2405.03148v1, Section 4–5.
"""

from __future__ import annotations

import ast
import csv
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .jaspar_loader import load_jaspar_yeastract_crossref, load_jaspar_pfm
from .promoter_extractor import (
    extract_promoters,
    extract_scerevisiae_promoters,
    SGD_CHR_MAP,
)
from .y1000plus_loader import (
    PROCESSED_DIR,
    get_gff3_path,
    get_genome_fasta_path,
    get_pep_path,
    get_representative_subset,
    load_manifest,
    scerevisiae_pep_path,
)
from .ortholog_finder import find_orthologs, load_species_proteins

PI3_CSV = PROCESSED_DIR / "pi3_tfbs_conservation.csv"
PI3_PAIRWISE_CSV = PROCESSED_DIR / "pi3_pairwise_histogram.csv"

# PWM significance threshold.
# Sequences with score >= SCORE_THRESHOLD are called significant hits.
# This is equivalent to a p-value cutoff of ~1e-4 for typical yeast motifs.
SCORE_THRESHOLD = 6.0
PSEUDOCOUNT = 0.1      # added to each PFM cell before log2 transform


# ---------------------------------------------------------------------------
# PWM construction from JASPAR PFM
# ---------------------------------------------------------------------------

def build_pwm_from_crossref(tf_name: str) -> Optional[np.ndarray]:
    """
    Build a log-odds PWM (shape: 4 × motif_width) for a TF from JASPAR data.

    Nucleotide order: A=0, C=1, G=2, T=3.
    Returns None if TF has no JASPAR profile.
    PWM values are log2(freq / 0.25) — assuming uniform background.
    """
    xref = load_jaspar_yeastract_crossref()
    row = xref[xref["yeastract_tf_name"].str.upper() == tf_name.upper()]
    if row.empty or row.iloc[0]["match_type"] == "no_match":
        return None

    r = row.iloc[0]
    try:
        pfm = {
            nuc: np.array(ast.literal_eval(str(r[f"pfm_{nuc}"])), dtype=float)
            for nuc in "ACGT"
        }
    except Exception:
        return None

    width = len(pfm["A"])
    if width == 0:
        return None

    pwm = np.zeros((4, width))
    for i, nuc in enumerate("ACGT"):
        col_sums = sum(pfm[n] for n in "ACGT")
        col_sums = np.where(col_sums == 0, 1, col_sums)
        freq = (pfm[nuc] + PSEUDOCOUNT) / (col_sums + 4 * PSEUDOCOUNT)
        pwm[i] = np.log2(freq / 0.25)

    return pwm


# ---------------------------------------------------------------------------
# PWM scanning — vectorized numpy implementation
# ---------------------------------------------------------------------------

# ASCII lookup table: maps byte value of A/C/G/T/N to 0-3
_ASCII_TO_NUC = np.zeros(256, dtype=np.int8)
_ASCII_TO_NUC[ord("A")] = _ASCII_TO_NUC[ord("a")] = 0
_ASCII_TO_NUC[ord("C")] = _ASCII_TO_NUC[ord("c")] = 1
_ASCII_TO_NUC[ord("G")] = _ASCII_TO_NUC[ord("g")] = 2
_ASCII_TO_NUC[ord("T")] = _ASCII_TO_NUC[ord("t")] = 3
# N / anything else maps to 0 (treated as A — contributes least to score)

_RC_NUC = np.array([3, 2, 1, 0], dtype=np.int8)  # A↔T, C↔G

# Keep a dict-based fallback for single-char use elsewhere
_NUC_IDX = {n: i for i, n in enumerate("ACGTacgt")}
_NUC_IDX.update({"N": 0, "n": 0})


def _seq_to_idx(seq: str) -> np.ndarray:
    """Convert a DNA string to a numpy array of nucleotide indices (0–3)."""
    return _ASCII_TO_NUC[np.frombuffer(seq.upper().encode(), dtype=np.uint8)]


def _score_all_positions(idx: np.ndarray, pwm: np.ndarray) -> np.ndarray:
    """
    Vectorized PWM scoring via numpy sliding_window_view.
    Returns max(forward, reverse-complement) score at every position.
    Shape: (len(idx) - width + 1,)
    """
    width = pwm.shape[1]
    n = len(idx)
    if n < width:
        return np.empty(0)

    # Forward: sliding window → shape (n-width+1, width)
    windows = np.lib.stride_tricks.sliding_window_view(idx, width)
    col_idx = np.arange(width)
    fwd = pwm[windows, col_idx].sum(axis=1)

    # Reverse complement
    rc_idx = _RC_NUC[idx[::-1]]
    rc_windows = np.lib.stride_tricks.sliding_window_view(rc_idx, width)
    rev = pwm[rc_windows, col_idx].sum(axis=1)

    return np.maximum(fwd, rev)


def score_sequence(seq: str, pwm: np.ndarray) -> list[tuple[int, float]]:
    """
    Slide PWM across seq (both strands). Returns (position, score) pairs
    where score >= SCORE_THRESHOLD. Position is 0-based in the forward seq.
    """
    idx = _seq_to_idx(seq)
    scores = _score_all_positions(idx, pwm)
    if len(scores) == 0:
        return []
    mask = scores >= SCORE_THRESHOLD
    positions = np.where(mask)[0]
    return [(int(p), float(scores[p])) for p in positions]


def has_significant_hit(seq: str, pwm: np.ndarray) -> bool:
    """Return True if any position scores >= SCORE_THRESHOLD (vectorized)."""
    idx = _seq_to_idx(seq)
    scores = _score_all_positions(idx, pwm)
    return len(scores) > 0 and bool(scores.max() >= SCORE_THRESHOLD)


def best_score(seq: str, pwm: np.ndarray) -> float:
    """Return the maximum PWM score across both strands."""
    idx = _seq_to_idx(seq)
    scores = _score_all_positions(idx, pwm)
    return float(scores.max()) if len(scores) > 0 else float("-inf")


# ---------------------------------------------------------------------------
# Step 1: Scan S. cerevisiae promoters to build binding map
# ---------------------------------------------------------------------------

def scan_scerevisiae(
    tf_name: str,
    pwm: np.ndarray,
    promoter_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Scan S. cerevisiae S288C promoters for a given TF PWM.

    Returns DataFrame of significant hits:
      gene_id, gene_name, best_score, n_hits
    """
    if promoter_df is None:
        promoter_df = extract_scerevisiae_promoters(save_fasta=False)

    rows = []
    for _, row in promoter_df.iterrows():
        hits = score_sequence(row["sequence"], pwm)
        if hits:
            rows.append(
                {
                    "gene_id": row["gene_id"],
                    "gene_name": row.get("gene_name", ""),
                    "best_score": max(s for _, s in hits),
                    "n_hits": len(hits),
                }
            )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["gene_id", "gene_name", "best_score", "n_hits"]
    )


# ---------------------------------------------------------------------------
# Main estimator — single TF
# ---------------------------------------------------------------------------

def estimate_pi3_for_tf(
    tf_name: str,
    species_subset: Optional[list[str]] = None,
    promoter_df: Optional[pd.DataFrame] = None,
    sc_pep: Optional[dict[str, str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Estimate π₃ for all target genes of one TF.

    Ortholog-anchored: for each species, only the promoter of the gene with
    the highest protein-sequence similarity to the S. cerevisiae target gene
    is checked.  Species where no confident ortholog is found are excluded
    from the denominator.

    Parameters
    ----------
    tf_name       : YEASTRACT TF name (e.g. 'GAL4', 'GCN4').
    species_subset: List of assembly_ids to scan.  Defaults to
                    REPRESENTATIVE_SUBSET.
    promoter_df   : Pre-computed S. cerevisiae promoter DataFrame (reuse).
    sc_pep        : Pre-loaded S. cerevisiae proteins {gene_id → seq}.
    verbose       : Print progress.

    Returns DataFrame with columns:
      tf_name, jaspar_matrix_id, target_gene_id, target_gene_name,
      n_genomes_scanned, n_genomes_with_hit, n_orthologs_found,
      retention_fraction, pi3_estimate
    """
    pwm = build_pwm_from_crossref(tf_name)
    if pwm is None:
        if verbose:
            print(f"  {tf_name}: no JASPAR PWM — skipped")
        return pd.DataFrame()

    xref = load_jaspar_yeastract_crossref()
    xr = xref[xref["yeastract_tf_name"].str.upper() == tf_name.upper()].iloc[0]
    matrix_id = xr["jaspar_matrix_id"]

    # Step 1: Find S. cerevisiae targets
    sc_hits = scan_scerevisiae(tf_name, pwm, promoter_df)
    if sc_hits.empty:
        if verbose:
            print(f"  {tf_name}: no S. cerevisiae hits — skipped")
        return pd.DataFrame()

    target_ids = set(sc_hits["gene_id"].tolist())
    if verbose:
        print(f"  {tf_name} ({matrix_id}): {len(target_ids)} S.cer. targets")

    # Load S. cerevisiae proteins if not supplied
    if sc_pep is None:
        sc_pep = load_species_proteins(scerevisiae_pep_path())

    sc_seqs = {gid: sc_pep[gid] for gid in target_ids if gid in sc_pep}

    manifest = load_manifest()
    if species_subset is None:
        species_subset = get_representative_subset()

    # Accumulators
    hit_counts: dict[str, int] = {gid: 0 for gid in target_ids}
    scan_counts: dict[str, int] = {gid: 0 for gid in target_ids}

    for aid in species_subset:
        rows = manifest[manifest["assembly_id"] == aid]
        if rows.empty:
            continue
        preferred = rows[rows["annotation_type"] == "sgd"]
        if preferred.empty:
            preferred = rows[rows["annotation_type"] == "final"]
        if preferred.empty:
            continue
        ann_type = preferred.iloc[0]["annotation_type"]

        # Extract promoters for this species
        try:
            gff3 = get_gff3_path(aid, ann_type)
            fasta = get_genome_fasta_path(aid)
            sp_prom = extract_promoters(gff3_path=gff3, fasta_path=fasta)
        except Exception as e:
            warnings.warn(f"Failed to extract promoters for {aid}: {e}")
            continue

        if sp_prom.empty:
            continue

        # Load species proteins and find orthologs
        try:
            pep_path = get_pep_path(aid, ann_type)
            sp_pep = load_species_proteins(pep_path)
        except Exception as e:
            warnings.warn(f"Failed to load proteins for {aid}: {e}")
            sp_pep = {}

        ortholog_map = find_orthologs(sc_seqs, sp_pep, target_ids)

        # Build gene_id → promoter sequence index for this species
        sp_prom_seqs: dict[str, str] = {
            row["gene_id"]: row["sequence"]
            for _, row in sp_prom.iterrows()
        }

        # Check ortholog's promoter
        for gid in target_ids:
            orth_id = ortholog_map.get(gid)
            if orth_id is None:
                continue   # no ortholog — exclude this species from denominator
            orth_seq = sp_prom_seqs.get(orth_id)
            if orth_seq is None:
                continue   # ortholog found but no promoter extracted — skip
            scan_counts[gid] += 1
            if has_significant_hit(orth_seq, pwm):
                hit_counts[gid] += 1

    # Build output rows
    gene_meta = dict(zip(sc_hits["gene_id"], sc_hits["gene_name"]))
    output_rows = []
    for gid in sorted(target_ids):
        n_scan = scan_counts[gid]
        n_hit = hit_counts[gid]
        ret = n_hit / n_scan if n_scan > 0 else float("nan")
        output_rows.append(
            {
                "tf_name": tf_name,
                "jaspar_matrix_id": matrix_id,
                "target_gene_id": gid,
                "target_gene_name": gene_meta.get(gid, ""),
                "n_genomes_scanned": n_scan,
                "n_genomes_with_hit": n_hit,
                "n_orthologs_found": n_scan,   # == species where ortholog was located
                "retention_fraction": round(ret, 4) if not np.isnan(ret) else None,
                "pi3_estimate": round(ret, 4) if not np.isnan(ret) else None,
            }
        )

    return pd.DataFrame(output_rows)


# ---------------------------------------------------------------------------
# Main estimator — all TFs (species-first loop for efficiency)
# ---------------------------------------------------------------------------

def estimate_pi3_all_tfs(
    tf_list: Optional[list[str]] = None,
    species_subset: Optional[list[str]] = None,
    save: bool = True,
    verbose: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Run π₃ estimation for all (or specified) YEASTRACT TFs.

    Uses a species-first loop: each species' promoters and proteins are
    loaded exactly once, then all TF PWMs are scanned against that batch.
    This reduces genome reads from (n_tfs × n_species) to n_species.

    Retention is computed per target gene relative to its best-matching
    ortholog in each species (see ortholog_finder.py).  Species where no
    confident ortholog is found for a given gene are excluded from that
    gene's denominator.

    Parameters
    ----------
    tf_list       : TFs to process (default: all YEASTRACT TFs with JASPAR PWMs).
    species_subset: Species to scan (default: REPRESENTATIVE_SUBSET ~50 species).
    save          : Whether to write pi3_tfbs_conservation.csv.
    verbose       : Print per-TF and per-species progress.
    progress_callback: Optional callable(phase, done, total, label) for
                       UI progress bars.
    """
    from .data_loader import YEASTRACT_TFS_SGD

    if tf_list is None:
        tf_list = sorted(YEASTRACT_TFS_SGD)

    # ── Step 1: S. cerevisiae promoters ──────────────────────────────
    if verbose:
        print("Pre-computing S. cerevisiae promoters...")
    sc_prom = extract_scerevisiae_promoters(save_fasta=False)

    # ── Step 2: Build all PWMs, scan S. cerevisiae, load S.cer. proteins ─
    if verbose:
        print(f"Building PWMs and scanning S. cerevisiae for {len(tf_list)} TFs...")

    xref = load_jaspar_yeastract_crossref()
    n_tfs_total = len(tf_list)

    pwm_cache: dict[str, np.ndarray] = {}
    sc_targets: dict[str, dict] = {}   # tf → {gene_id: gene_name}
    matrix_ids: dict[str, str] = {}

    for tf_i, tf in enumerate(tf_list):
        pwm = build_pwm_from_crossref(tf)
        if pwm is None:
            continue
        hits = scan_scerevisiae(tf, pwm, sc_prom)
        if hits.empty:
            continue
        pwm_cache[tf] = pwm
        sc_targets[tf] = dict(zip(hits["gene_id"], hits["gene_name"]))
        xr = xref[xref["yeastract_tf_name"].str.upper() == tf.upper()]
        matrix_ids[tf] = xr.iloc[0]["jaspar_matrix_id"] if not xr.empty else ""

        if progress_callback and (tf_i + 1) % 10 == 0:
            progress_callback("prescan", tf_i + 1, n_tfs_total, tf)

    active_tfs = list(pwm_cache.keys())
    if verbose:
        print(f"  {len(active_tfs)} TFs have PWMs and S. cerevisiae hits")

    # Collect every unique target gene ID across all TFs
    all_target_ids: set[str] = set()
    for tf in active_tfs:
        all_target_ids.update(sc_targets[tf].keys())

    # Load S. cerevisiae proteins (needed for ortholog mapping)
    if verbose:
        print("Loading S. cerevisiae reference proteins...")
    sc_pep_all = load_species_proteins(scerevisiae_pep_path())
    sc_seqs = {gid: sc_pep_all[gid] for gid in all_target_ids if gid in sc_pep_all}
    if verbose:
        print(f"  {len(sc_seqs)}/{len(all_target_ids)} target genes have protein sequences")

    # Accumulators: hit_counts[tf][gene_id], scan_counts[tf][gene_id]
    hit_counts: dict[str, dict[str, int]] = {
        tf: {gid: 0 for gid in sc_targets[tf]} for tf in active_tfs
    }
    scan_counts: dict[str, dict[str, int]] = {
        tf: {gid: 0 for gid in sc_targets[tf]} for tf in active_tfs
    }

    # ── Step 3: Species-first loop ────────────────────────────────────
    if species_subset is None:
        species_subset = get_representative_subset()
    manifest = load_manifest()

    if verbose:
        print(f"Scanning {len(species_subset)} species (ortholog-anchored)...")

    for sp_i, aid in enumerate(species_subset):
        rows = manifest[manifest["assembly_id"] == aid]
        if rows.empty:
            continue
        preferred = rows[rows["annotation_type"] == "sgd"]
        if preferred.empty:
            preferred = rows[rows["annotation_type"] == "final"]
        if preferred.empty:
            continue
        ann_type = preferred.iloc[0]["annotation_type"]

        # Extract promoters for this species (one disk read)
        try:
            gff3 = get_gff3_path(aid, ann_type)
            fasta = get_genome_fasta_path(aid)
            sp_prom = extract_promoters(gff3_path=gff3, fasta_path=fasta)
        except Exception as e:
            warnings.warn(f"Skipping {aid}: {e}")
            continue

        if sp_prom.empty:
            warnings.warn(
                f"No promoters extracted for {aid} "
                "(FASTA/GFF3 name mismatch?) — skipping"
            )
            continue

        if verbose:
            print(
                f"  [{sp_i+1}/{len(species_subset)}] {aid}: "
                f"{len(sp_prom)} promoters"
            )
        if progress_callback:
            progress_callback("species", sp_i + 1, len(species_subset), aid)

        # Load species proteins and find orthologs for all target genes
        try:
            pep_path = get_pep_path(aid, ann_type)
            sp_pep = load_species_proteins(pep_path)
        except Exception as e:
            warnings.warn(f"Could not load proteins for {aid}: {e}")
            sp_pep = {}

        # ortholog_map: {sc_gene_id → sp_gene_id | None}
        ortholog_map = find_orthologs(sc_seqs, sp_pep, all_target_ids)

        # Build gene_id → pre-converted index array for fast PWM scanning
        sp_prom_idx: dict[str, np.ndarray] = {
            row["gene_id"]: _seq_to_idx(row["sequence"])
            for _, row in sp_prom.iterrows()
        }

        # For each TF, check only the ortholog's promoter
        for tf in active_tfs:
            pwm = pwm_cache[tf]
            for gid in sc_targets[tf]:
                orth_id = ortholog_map.get(gid)
                if orth_id is None:
                    continue   # no ortholog → exclude from denominator
                orth_idx = sp_prom_idx.get(orth_id)
                if orth_idx is None:
                    continue   # ortholog found, but promoter not extracted → skip
                scan_counts[tf][gid] += 1
                scores = _score_all_positions(orth_idx, pwm)
                if len(scores) > 0 and bool(scores.max() >= SCORE_THRESHOLD):
                    hit_counts[tf][gid] += 1

    # ── Step 4: Build result DataFrame ───────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for tf in active_tfs:
        for gid, gname in sc_targets[tf].items():
            n_scan = scan_counts[tf][gid]
            n_hit = hit_counts[tf][gid]
            ret = n_hit / n_scan if n_scan > 0 else float("nan")
            output_rows.append({
                "tf_name": tf,
                "jaspar_matrix_id": matrix_ids[tf],
                "target_gene_id": gid,
                "target_gene_name": gname,
                "n_genomes_scanned": n_scan,
                "n_genomes_with_hit": n_hit,
                "n_orthologs_found": n_scan,   # species where ortholog was located
                "retention_fraction": round(ret, 4) if not np.isnan(ret) else None,
                "pi3_estimate":       round(ret, 4) if not np.isnan(ret) else None,
            })

    if not output_rows:
        return pd.DataFrame()

    result = pd.DataFrame(output_rows)
    if save:
        result.to_csv(PI3_CSV, index=False)
        if verbose:
            print(f"Saved -> {PI3_CSV}")
    return result


# ---------------------------------------------------------------------------
# Pairwise shared binding site histogram
# ---------------------------------------------------------------------------

def build_pairwise_histogram(
    pi3_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    For each TF family (= set of S. cerevisiae genes targeted by the same TF),
    compute pairwise shared binding-site counts across all species pairs.

    A 'shared' site between species A and species B means both have a
    retention_fraction > 0 for the same TF→gene edge.

    Returns DataFrame with columns:
      tf_name, n_target_genes, mean_retention, std_retention,
      pairwise_sharing_mean, pairwise_sharing_std, pi3_family_estimate

    The pairwise_sharing_mean is the key quantity connecting to Scruse et al.:
    it estimates π̂₃ for the family under the binary inheritance model.
    """
    if pi3_df is None and PI3_PAIRWISE_CSV.exists():
        return pd.read_csv(PI3_PAIRWISE_CSV)

    if pi3_df is None:
        if not PI3_CSV.exists():
            raise FileNotFoundError(f"Run estimate_pi3_all_tfs() first: {PI3_CSV}")
        pi3_df = pd.read_csv(PI3_CSV)

    rows = []
    for tf_name, grp in pi3_df.groupby("tf_name"):
        ret = grp["retention_fraction"].dropna().values
        if len(ret) == 0:
            continue

        if len(ret) > 1:
            outer = np.outer(ret, ret)
            triu = outer[np.triu_indices(len(ret), k=1)]
            pw_mean = float(triu.mean())
            pw_std = float(triu.std())
        else:
            pw_mean = float(ret[0]) ** 2
            pw_std = 0.0

        rows.append(
            {
                "tf_name": tf_name,
                "n_target_genes": int(len(ret)),
                "mean_retention": round(float(np.mean(ret)), 4),
                "std_retention": round(float(np.std(ret)), 4),
                "pairwise_sharing_mean": round(pw_mean, 4),
                "pairwise_sharing_std": round(pw_std, 4),
                "pi3_family_estimate": round(float(np.mean(ret)), 4),
            }
        )

    result = pd.DataFrame(rows)
    if save and not result.empty:
        result.to_csv(PI3_PAIRWISE_CSV, index=False)
    return result


# ---------------------------------------------------------------------------
# Load cached results
# ---------------------------------------------------------------------------

def load_pi3_results() -> pd.DataFrame:
    """Load cached pi3_tfbs_conservation.csv, running the scan if missing."""
    if not PI3_CSV.exists():
        raise FileNotFoundError(
            f"pi3_tfbs_conservation.csv not found. "
            f"Run estimate_pi3_all_tfs() to generate it."
        )
    return pd.read_csv(PI3_CSV)


def pi3_summary() -> dict:
    """Summary statistics over all π₃ estimates."""
    df = load_pi3_results()
    return {
        "n_tf_gene_edges": len(df),
        "n_tfs": df["tf_name"].nunique(),
        "n_target_genes": df["target_gene_id"].nunique(),
        "mean_pi3": round(float(df["pi3_estimate"].mean()), 4),
        "median_pi3": round(float(df["pi3_estimate"].median()), 4),
        "n_high_conservation": int((df["retention_fraction"] >= 0.8).sum()),
        "n_low_conservation": int((df["retention_fraction"] <= 0.2).sum()),
    }
