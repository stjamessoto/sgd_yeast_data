"""
pi4_snp_binding.py
Estimates π₄: regulatory inheritance probability from SNP polymorphism
rates at JASPAR-identified binding site positions.

This extends the existing YFL039C SNP method (Method 3 in inheritance_estimator.py)
genome-wide to all YEASTRACT TF binding sites identified in S. cerevisiae S288C.

Algorithm (per TF → gene edge):
  1. Identify binding site positions in the S. cerevisiae promoter (from π₃ scan).
  2. For each position in the PWM motif, retrieve the local sequence from each
     Y1000+ species' orthologous promoter region (extracted via genome + GFF3).
  3. At each position i, compute polymorphism_rate[i] =
       (# species with a different nucleotide) / (# species with the gene).
  4. Weight polymorphism_rate[i] by the PWM information content at position i:
       ic_weight[i] = IC[i] / Σ IC[j]   (normalized per-position IC)
  5. weighted_poly_rate = Σ ic_weight[i] × polymorphism_rate[i]
  6. π₄ = 1 − weighted_poly_rate

Biological rationale:
  A binding site that is perfectly conserved across all 1,154 genomes has
  polymorphism_rate = 0, giving π₄ = 1.0 (link likely inherited).
  A binding site with mutations at high-IC (critical) positions has
  weighted_poly_rate → 1, giving π₄ → 0 (link likely lost after duplication).

Limitation:
  True one-to-one ortholog mapping requires BLASTP / OrthoFinder.  Here we
  approximate using the best-scoring PWM hit in each species' promoters as
  the "orthologous site".  For highly-specific TFs (high IC) this is
  reliable; for broad binders it may introduce noise.

Output: pi4_snp_binding_sites.csv
  tf_name, target_gene_id, target_gene_name, binding_site_start,
  binding_site_seq, n_genomes_aligned, weighted_polymorphism_rate, pi4_estimate
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .pi3_tfbs_conservation import (
    build_pwm_from_crossref,
    scan_scerevisiae,
    score_sequence,
    SCORE_THRESHOLD,
)
from .promoter_extractor import (
    extract_promoters,
    extract_scerevisiae_promoters,
    SGD_CHR_MAP,
)
from .y1000plus_loader import (
    PROCESSED_DIR,
    get_gff3_path,
    get_genome_fasta_path,
    get_representative_subset,
    load_manifest,
)

PI4_CSV = PROCESSED_DIR / "pi4_snp_binding_sites.csv"

PSEUDOCOUNT = 0.1


# ---------------------------------------------------------------------------
# Per-position information content from PWM
# ---------------------------------------------------------------------------

def pwm_information_content(pwm: np.ndarray) -> np.ndarray:
    """
    Compute per-position information content (IC) from a log-odds PWM.

    IC[pos] = Σ freq[nuc][pos] × log2(freq[nuc][pos] / 0.25)
            = Σ freq × (pwm / ln2) where pwm is in log2 units

    Since pwm[nuc][pos] = log2(freq[nuc][pos] / 0.25), and
    IC = Σ freq × log2(freq / 0.25) = Σ freq × pwm,
    we estimate freq ≈ 0.25 × 2^pwm.

    Returns IC array of shape (motif_width,).
    """
    freq = 0.25 * np.power(2.0, pwm.clip(-10, 10))  # shape (4, width)
    freq = freq / freq.sum(axis=0, keepdims=True)     # normalize columns
    ic = np.sum(freq * pwm, axis=0)                   # (width,)
    return np.maximum(ic, 0.0)


# ---------------------------------------------------------------------------
# Find best binding site in a promoter sequence
# ---------------------------------------------------------------------------

def find_best_binding_site(
    seq: str,
    pwm: np.ndarray,
) -> Optional[tuple[int, float, str, bool]]:
    """
    Find the best (highest-scoring) binding site using the vectorized scanner
    from pi3_tfbs_conservation. Returns (position, score, site_seq, is_rc)
    or None if no position scores >= SCORE_THRESHOLD.
    """
    from .pi3_tfbs_conservation import _seq_to_idx, _RC_NUC, SCORE_THRESHOLD as _THRESH
    width = pwm.shape[1]
    n = len(seq)
    if n < width:
        return None

    idx = _seq_to_idx(seq)
    col_idx = np.arange(width)

    fwd_windows = np.lib.stride_tricks.sliding_window_view(idx, width)
    fwd_scores = pwm[fwd_windows, col_idx].sum(axis=1)

    rc_idx = _RC_NUC[idx[::-1]]
    rc_windows = np.lib.stride_tricks.sliding_window_view(rc_idx, width)
    rc_scores = pwm[rc_windows, col_idx].sum(axis=1)

    fwd_best_pos = int(fwd_scores.argmax())
    rc_best_pos  = int(rc_scores.argmax())

    if fwd_scores[fwd_best_pos] >= rc_scores[rc_best_pos]:
        best_pos, best_score_, is_rc = fwd_best_pos, float(fwd_scores[fwd_best_pos]), False
    else:
        best_pos, best_score_, is_rc = rc_best_pos, float(rc_scores[rc_best_pos]), True

    if best_score_ < _THRESH:
        return None

    if is_rc:
        site_seq = seq[n - best_pos - width : n - best_pos]
    else:
        site_seq = seq[best_pos : best_pos + width]

    return best_pos, best_score_, site_seq, is_rc


# ---------------------------------------------------------------------------
# Cross-species polymorphism at binding site positions
# ---------------------------------------------------------------------------

def compute_site_polymorphism(
    ref_site: str,
    pwm: np.ndarray,
    species_subset: list[str],
    manifest: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    """
    For each position in a binding site, compute the fraction of species
    where that position differs from the S. cerevisiae reference.

    ref_site       : Reference binding site sequence (length = motif width).
    pwm            : Log-odds PWM (4 × width).
    species_subset : Assembly IDs to check.
    manifest       : Y1000+ manifest DataFrame.

    Returns:
      poly_rates : np.ndarray of shape (width,) — per-position polymorphism rates.
      n_aligned  : number of species where a binding site was found.
    """
    width = len(ref_site)
    mismatch_counts = np.zeros(width, dtype=float)
    n_found = 0

    for aid in species_subset:
        if aid == "saccharomyces_cerevisiae":
            continue  # skip reference

        rows = manifest[manifest["assembly_id"] == aid]
        if rows.empty:
            continue
        preferred = rows[rows["annotation_type"] != "sgd"]
        if preferred.empty:
            preferred = rows
        ann_type = preferred.iloc[0]["annotation_type"]

        try:
            gff3 = get_gff3_path(aid, ann_type)
            fasta = get_genome_fasta_path(aid)
            prom_df = extract_promoters(gff3_path=gff3, fasta_path=fasta)
        except Exception as e:
            warnings.warn(f"Cannot extract promoters for {aid}: {e}")
            continue

        if prom_df.empty:
            continue

        # Find the best binding site across all promoters in this species
        best: Optional[tuple[int, float, str, bool]] = None
        for _, row in prom_df.iterrows():
            hit = find_best_binding_site(row["sequence"], pwm)
            if hit is not None and (best is None or hit[1] > best[1]):
                best = hit

        if best is None:
            continue

        _, _, site_seq, _ = best
        n_found += 1
        for pos in range(min(width, len(site_seq))):
            if site_seq[pos].upper() != ref_site[pos].upper():
                mismatch_counts[pos] += 1

    poly_rates = mismatch_counts / n_found if n_found > 0 else np.zeros(width)
    return poly_rates, n_found


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def estimate_pi4_for_tf(
    tf_name: str,
    species_subset: Optional[list[str]] = None,
    promoter_df: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Estimate π₄ for all target genes of one TF.

    π₄ = 1 − weighted_polymorphism_rate, where weights are the per-position
    PWM information content (IC).

    Returns DataFrame with columns:
      tf_name, target_gene_id, target_gene_name, binding_site_start,
      binding_site_seq, n_genomes_aligned, weighted_polymorphism_rate, pi4_estimate
    """
    pwm = build_pwm_from_crossref(tf_name)
    if pwm is None:
        return pd.DataFrame()

    if promoter_df is None:
        promoter_df = extract_scerevisiae_promoters(save_fasta=False)

    # Find S. cerevisiae target genes with binding sites
    sc_hits = scan_scerevisiae(tf_name, pwm, promoter_df)
    if sc_hits.empty:
        return pd.DataFrame()

    # Compute per-position IC weights
    ic = pwm_information_content(pwm)
    ic_weights = ic / ic.sum() if ic.sum() > 0 else np.ones(len(ic)) / len(ic)

    if species_subset is None:
        species_subset = get_representative_subset()
    manifest = load_manifest()

    # For each target gene, find its binding site in S. cerevisiae and compute
    # cross-species polymorphism at those positions
    output_rows = []

    # Merge promoter sequences with hit gene IDs
    prom_by_gene = {row["gene_id"]: row["sequence"]
                    for _, row in promoter_df.iterrows()}
    gene_name_by_id = {row["gene_id"]: row.get("gene_name", "")
                       for _, row in promoter_df.iterrows()}

    # Sample at most 100 target genes per TF for efficiency
    target_ids = sc_hits["gene_id"].tolist()
    if len(target_ids) > 100:
        import random
        random.seed(42)
        target_ids = random.sample(target_ids, 100)

    for gene_id in target_ids:
        seq = prom_by_gene.get(gene_id)
        if seq is None:
            continue

        site_hit = find_best_binding_site(seq, pwm)
        if site_hit is None:
            continue

        site_pos, site_score, site_seq, _ = site_hit

        # Compute cross-species polymorphism
        poly_rates, n_aligned = compute_site_polymorphism(
            ref_site=site_seq,
            pwm=pwm,
            species_subset=species_subset,
            manifest=manifest,
        )

        weighted_poly = float(np.dot(ic_weights, poly_rates))
        pi4 = max(0.0, 1.0 - weighted_poly)

        output_rows.append(
            {
                "tf_name": tf_name,
                "target_gene_id": gene_id,
                "target_gene_name": gene_name_by_id.get(gene_id, ""),
                "binding_site_start": site_pos,
                "binding_site_seq": site_seq,
                "n_genomes_aligned": n_aligned,
                "weighted_polymorphism_rate": round(weighted_poly, 4),
                "pi4_estimate": round(pi4, 4),
            }
        )

    return pd.DataFrame(output_rows) if output_rows else pd.DataFrame()


def estimate_pi4_all_tfs(
    tf_list: Optional[list[str]] = None,
    species_subset: Optional[list[str]] = None,
    save: bool = True,
    verbose: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Run π₄ estimation using a species-first loop identical in structure to
    estimate_pi3_all_tfs. Each species' promoters are extracted exactly once;
    all TF PWMs are scanned against that batch before moving to the next species.

    The old TF-first approach re-extracted 48 species × 100 genes × 121 TFs =
    580,800 times and would have taken days. This version takes ~5–10 minutes.
    """
    from .data_loader import YEASTRACT_TFS_SGD
    from .pi3_tfbs_conservation import (
        _seq_to_idx, _score_all_positions, _RC_NUC, SCORE_THRESHOLD,
    )

    if tf_list is None:
        tf_list = sorted(YEASTRACT_TFS_SGD)
    if species_subset is None:
        species_subset = get_representative_subset()

    # ── Step 1: S. cerevisiae pre-scan ───────────────────────────────
    if verbose:
        print("Pre-computing S. cerevisiae promoters...")
    sc_prom = extract_scerevisiae_promoters(save_fasta=False)
    prom_by_gene = {r["gene_id"]: r["sequence"] for _, r in sc_prom.iterrows()}
    gene_name_map = {r["gene_id"]: r.get("gene_name", "") for _, r in sc_prom.iterrows()}

    pwm_cache: dict[str, np.ndarray] = {}
    ic_cache: dict[str, np.ndarray] = {}
    sc_sites: dict[str, dict[str, str]] = {}   # tf -> {gene_id: site_seq}
    sc_site_pos: dict[str, dict[str, int]] = {} # tf -> {gene_id: position}

    n_tfs_total = len(tf_list)
    for tf_i, tf in enumerate(tf_list):
        pwm = build_pwm_from_crossref(tf)
        if pwm is None:
            continue
        sc_hits = scan_scerevisiae(tf, pwm, sc_prom)
        if sc_hits.empty:
            continue

        target_ids = sc_hits["gene_id"].tolist()
        if len(target_ids) > 100:
            import random as _rnd
            _rnd.seed(42)
            target_ids = _rnd.sample(target_ids, 100)

        gene_sites: dict[str, str] = {}
        gene_pos: dict[str, int] = {}
        for gid in target_ids:
            seq = prom_by_gene.get(gid)
            if seq is None:
                continue
            hit = find_best_binding_site(seq, pwm)
            if hit is None:
                continue
            pos, _, site_seq, _ = hit
            gene_sites[gid] = site_seq
            gene_pos[gid] = pos

        if not gene_sites:
            continue

        pwm_cache[tf] = pwm
        ic = pwm_information_content(pwm)
        ic_cache[tf] = ic / ic.sum() if ic.sum() > 0 else np.ones(len(ic)) / len(ic)
        sc_sites[tf] = gene_sites
        sc_site_pos[tf] = gene_pos

        if progress_callback and (tf_i + 1) % 10 == 0:
            progress_callback("prescan", tf_i + 1, n_tfs_total, tf)

    active_tfs = list(sc_sites.keys())
    if verbose:
        print(f"  {len(active_tfs)} TFs have S. cerevisiae binding sites")

    # ── Accumulators ─────────────────────────────────────────────────
    mismatch_acc: dict[str, dict[str, np.ndarray]] = {
        tf: {gid: np.zeros(pwm_cache[tf].shape[1]) for gid in sc_sites[tf]}
        for tf in active_tfs
    }
    n_aligned: dict[str, dict[str, int]] = {
        tf: {gid: 0 for gid in sc_sites[tf]}
        for tf in active_tfs
    }

    # ── Step 2: Species-first loop ────────────────────────────────────
    manifest = load_manifest()
    if verbose:
        print(f"Scanning {len(species_subset)} species (extract once, all TFs)...")

    for sp_i, aid in enumerate(species_subset):
        rows = manifest[manifest["assembly_id"] == aid]
        if rows.empty:
            continue
        preferred = rows[rows["annotation_type"] != "sgd"]
        if preferred.empty:
            preferred = rows
        ann_type = preferred.iloc[0]["annotation_type"]

        try:
            gff3 = get_gff3_path(aid, ann_type)
            fasta = get_genome_fasta_path(aid)
            sp_prom = extract_promoters(gff3_path=gff3, fasta_path=fasta)
        except Exception as e:
            warnings.warn(f"Skipping {aid}: {e}")
            continue

        if sp_prom.empty:
            continue

        sp_seqs = sp_prom["sequence"].tolist()
        sp_idx = [_seq_to_idx(s) for s in sp_seqs]

        if verbose:
            print(f"  [{sp_i+1}/{len(species_subset)}] {aid}: {len(sp_seqs)} promoters")
        if progress_callback:
            progress_callback("species", sp_i + 1, len(species_subset), aid)

        for tf in active_tfs:
            pwm = pwm_cache[tf]
            width = pwm.shape[1]

            # Find the globally best binding site for this TF in this species
            best_score_sp = float("-inf")
            best_site_seq_sp: Optional[str] = None
            for seq, idx_arr in zip(sp_seqs, sp_idx):
                scores = _score_all_positions(idx_arr, pwm)
                if len(scores) == 0:
                    continue
                max_s = float(scores.max())
                if max_s > best_score_sp and max_s >= SCORE_THRESHOLD:
                    best_score_sp = max_s
                    # Retrieve the actual sequence at that position
                    hit = find_best_binding_site(seq, pwm)
                    if hit is not None:
                        best_site_seq_sp = hit[2]

            if best_site_seq_sp is None:
                continue

            # Compare to each S. cerevisiae gene's reference site
            for gid, ref_site in sc_sites[tf].items():
                n_aligned[tf][gid] += 1
                for pos in range(min(width, len(best_site_seq_sp), len(ref_site))):
                    if best_site_seq_sp[pos].upper() != ref_site[pos].upper():
                        mismatch_acc[tf][gid][pos] += 1

    # ── Step 3: Build output ──────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for tf in active_tfs:
        for gid, ref_site in sc_sites[tf].items():
            n = n_aligned[tf][gid]
            if n == 0:
                continue
            poly_per_pos = mismatch_acc[tf][gid] / n
            weighted_poly = float(np.dot(ic_cache[tf], poly_per_pos))
            pi4 = max(0.0, 1.0 - weighted_poly)
            output_rows.append({
                "tf_name": tf,
                "target_gene_id": gid,
                "target_gene_name": gene_name_map.get(gid, ""),
                "binding_site_start": sc_site_pos[tf].get(gid, -1),
                "binding_site_seq": ref_site,
                "n_genomes_aligned": n,
                "weighted_polymorphism_rate": round(weighted_poly, 4),
                "pi4_estimate": round(pi4, 4),
            })

    if not output_rows:
        return pd.DataFrame()

    result = pd.DataFrame(output_rows)
    if save:
        result.to_csv(PI4_CSV, index=False)
        if verbose:
            print(f"Saved -> {PI4_CSV}")
    return result


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pi4_results() -> pd.DataFrame:
    """Load cached pi4_snp_binding_sites.csv."""
    if not PI4_CSV.exists():
        raise FileNotFoundError(
            f"pi4_snp_binding_sites.csv not found. Run estimate_pi4_all_tfs() first."
        )
    return pd.read_csv(PI4_CSV)


def pi4_summary() -> dict:
    """Summary statistics over all π₄ estimates."""
    df = load_pi4_results()
    return {
        "n_tf_gene_edges": len(df),
        "n_tfs": df["tf_name"].nunique(),
        "mean_pi4": round(float(df["pi4_estimate"].mean()), 4),
        "mean_weighted_poly_rate": round(float(df["weighted_polymorphism_rate"].mean()), 4),
        "n_perfectly_conserved": int((df["pi4_estimate"] == 1.0).sum()),
        "mean_genomes_aligned": round(float(df["n_genomes_aligned"].mean()), 1),
    }
