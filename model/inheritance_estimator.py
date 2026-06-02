"""
inheritance_estimator.py
Estimates the inheritance probability vector π⃗ = (π₁, ..., πₖ) for gene families.

Four estimation methods, corresponding to different data sources:

  Method 1 — Evidence-based (from sgd_transcription_factors.csv)
    π_i = f(evidence_quality_i) using the EVIDENCE_QUALITY map.
    Higher-quality experimental support → links more likely to truly persist
    through duplication → higher πᵢ.

  Method 2 — MLE from observed counts (Theorem 4 inversion)
    Given observed |M(n)| and model parameters m, n, solve numerically for π̂.
    Then distribute π̂ across families proportional to evidence weights.

  Method 3 — SNP-divergence proxy (from sgd_YFL039C_inheritance_vectors.csv)
    π_i ≈ 1 − (pct_alt_i / 100).
    Higher divergence from reference → regulatory links less likely inherited.
    This is a per-strain/per-gene proxy demonstrated on gene YFL039C.

  Method 4 — Consensus-adjusted (from YEASTRACT consensuslist.php)
    Applies a pi_consensus_factor derived from the number and ambiguity of a
    TF's YEASTRACT consensus sequences.  Promiscuous binders (many/flexible
    consensus sequences) are more likely to maintain regulatory links after
    duplication because their binding sites tolerate sequence divergence.
    Factor = 1 + variant_boost + ambiguity_boost  (see consensus_loader.py).

All methods produce:
  - pi_vec:  List[float] of per-family πᵢ values  (one per family)
  - pi_hat:  float = Σπᵢ  (used directly in Theorem 4)
  - method:  str label
  - weights: List[float] allocation weights used

The result dict feeds directly into scruse_math functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from .scruse_math import (
    expected_partial,
    estimate_pi_hat,
    allocate_pi,
    expected_full,
    variance_full,
    variance_binary,
    second_moment_binary,
    z_score,
    p_value_from_z,
    significance_label,
    expected_bounds_partial,
)
from .data_loader import load_transcription_factors, load_inheritance_vectors, EVIDENCE_QUALITY
from .gene_families import estimate_model_parameters
from .consensus_loader import get_pi_consensus_factor, load_tf_consensus_stats


# ---------------------------------------------------------------------------
# Method 1: Evidence-based π estimation
# ---------------------------------------------------------------------------

def estimate_pi_from_evidence(
    gene_names: List[str],
) -> Dict:
    """
    Method 1: Estimate per-family π from evidence code quality scores.

    For each gene family represented by one TF, πᵢ = evidence_score of that TF.
    If a gene is not in the TF list, a default of EVIDENCE_QUALITY['IEA'] is used.

    Returns:
      pi_vec, pi_hat, weights, method, gene_names, evidence_scores
    """
    tfs = load_transcription_factors()
    tf_ev = dict(zip(tfs["gene_name"].str.upper(), tfs["pi_prior"]))

    default_ev = EVIDENCE_QUALITY["IEA"]
    pi_vec = [tf_ev.get(g.upper(), default_ev) for g in gene_names]
    pi_hat = sum(pi_vec)

    return {
        "method": "evidence_based",
        "gene_names": gene_names,
        "pi_vec": [round(p, 4) for p in pi_vec],
        "pi_hat": round(pi_hat, 4),
        "weights": pi_vec,
        "evidence_scores": pi_vec,
        "description": (
            "π estimated from SGD evidence code quality. "
            "IDA/IMP (experimental) → high π; IEA (automated) → low π."
        ),
    }


# ---------------------------------------------------------------------------
# Method 2: MLE inversion of Theorem 4
# ---------------------------------------------------------------------------

def estimate_pi_from_mle(
    observed_count: float,
    m: int,
    n: int,
    gene_names: List[str],
    weights: Optional[List[float]] = None,
) -> Dict:
    """
    Method 2: MLE — invert Theorem 4 to find π̂, then allocate across families.

    Solves: Γ(π̂+n)Γ(m) / [Γ(π̂+m)Γ(n)] = observed_count
    for π̂, then sets πᵢ = π̂ · wᵢ / Σwᵢ (weights default to uniform).

    Parameters
    ----------
    observed_count : float
        Observed number of subnetwork motif instances |M(n)|.
    m : int
        Number of gene families (initial gene count).
    n : int
        Total gene count at stage n.
    gene_names : List[str]
        Names of the k gene families in the motif.
    weights : optional List[float]
        Allocation weights; defaults to evidence scores if available, else uniform.
    """
    k = len(gene_names)
    if weights is None:
        # Fall back to evidence-based weights
        ev_result = estimate_pi_from_evidence(gene_names)
        weights = ev_result["evidence_scores"]

    pi_hat_est = estimate_pi_hat(observed_count, m, n)

    if pi_hat_est is None:
        pi_hat_est = float(k) * 0.5  # fallback
        note = "MLE did not converge; using uniform fallback π̂ = k/2."
    else:
        note = f"MLE converged: π̂ = {pi_hat_est:.4f}"

    pi_vec = allocate_pi(pi_hat_est, weights)

    return {
        "method": "mle_theorem4",
        "gene_names": gene_names,
        "pi_vec": [round(p, 4) for p in pi_vec],
        "pi_hat": round(pi_hat_est, 4),
        "weights": weights,
        "observed_count": observed_count,
        "m": m,
        "n": n,
        "note": note,
        "description": (
            "π̂ estimated by inverting Theorem 4 (MLE). "
            "Solves Γ(π̂+n)Γ(m)/[Γ(π̂+m)Γ(n)] = observed count. "
            "Individual πᵢ allocated proportional to evidence weights."
        ),
    }


# ---------------------------------------------------------------------------
# Method 3: SNP-divergence proxy
# ---------------------------------------------------------------------------

def estimate_pi_from_snp(
    gene_names: Optional[List[str]] = None,
    snp_data: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Method 3: Estimate π from per-strain SNP divergence.

    Uses sgd_YFL039C_inheritance_vectors.csv as the demonstration dataset.
    π_i ≈ 1 − (pct_alt / 100)  per strain.

    If gene_names is provided, returns a single π estimate per gene
    (mean over strains).  Otherwise returns the full per-strain table.

    This method demonstrates the paper's connection between sequence
    divergence and regulatory inheritance (Section 9.2, MITEs discussion):
    greater divergence ⇒ older link ⇒ lower inheritance probability.
    """
    ivec = load_inheritance_vectors() if snp_data is None else snp_data

    # Per-strain π estimates
    per_strain = ivec[["strain", "gene", "pct_alt", "pi_snp"]].copy()

    # Aggregate across strains: mean π per gene
    gene_pi = (
        ivec.groupby("gene")["pi_snp"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "pi_mean", "std": "pi_std"})
    )

    if gene_names:
        # Map to requested families; default to gene_pi if available
        pi_vec = []
        for g in gene_names:
            row = gene_pi[gene_pi["gene"].str.upper() == g.upper()]
            if not row.empty:
                pi_vec.append(float(row["pi_mean"].iloc[0]))
            else:
                pi_vec.append(float(ivec["pi_snp"].mean()))  # dataset-wide mean

        pi_hat = sum(pi_vec)
        return {
            "method": "snp_divergence",
            "gene_names": gene_names,
            "pi_vec": [round(p, 4) for p in pi_vec],
            "pi_hat": round(pi_hat, 4),
            "weights": pi_vec,
            "per_strain_df": per_strain,
            "gene_pi_df": gene_pi,
            "description": (
                "π estimated from SNP divergence relative to reference strain. "
                "πᵢ = 1 − (% alternative alleles / 100). "
                "Higher divergence → older link → less likely inherited."
            ),
        }

    return {
        "method": "snp_divergence",
        "gene_names": list(gene_pi["gene"]),
        "pi_vec": gene_pi["pi_mean"].round(4).tolist(),
        "pi_hat": round(gene_pi["pi_mean"].sum(), 4),
        "per_strain_df": per_strain,
        "gene_pi_df": gene_pi,
        "description": "Full SNP-divergence π table for all genes in dataset.",
    }


# ---------------------------------------------------------------------------
# Method 4: Consensus-adjusted (YEASTRACT binding sequences)
# ---------------------------------------------------------------------------

def estimate_pi_consensus_adjusted(
    gene_names: List[str],
    base_method: str = "evidence",
) -> Dict:
    """
    Method 4: Adjust evidence-based π using YEASTRACT consensus sequence data.

    For each TF family, the pi_consensus_factor (from consensus_loader.py) is
    applied as a multiplier to the base evidence-based π:

        πᵢ_adjusted = min(1.0, πᵢ_base × pi_consensus_factor_i)

    Biological rationale (paper Section 9.2 / MITE analogy):
      A TF with many distinct, ambiguity-rich consensus sequences is a
      promiscuous binder — its regulatory links are more likely to be
      maintained after duplication because the duplicate can still bind
      even if the target promoter sequence drifts slightly.
      → More sequences / higher ambiguity fraction → higher πᵢ.

    Returns the same dict structure as other estimators.
    """
    # Get base π from evidence scores
    base_result = estimate_pi_from_evidence(gene_names)
    base_pi = base_result["pi_vec"]

    # Apply consensus factor for each TF
    adjusted_pi = []
    factors = []
    consensus_stats = []
    for gene in gene_names:
        factor = get_pi_consensus_factor(gene)
        factors.append(factor)
        adj = min(1.0, base_pi[gene_names.index(gene)] * factor)
        adjusted_pi.append(round(adj, 4))

        # Collect per-TF consensus stats for display
        stats = load_tf_consensus_stats()
        row = stats[stats["tf_sgd"].str.upper() == gene.upper().rstrip("P")]
        if not row.empty:
            consensus_stats.append({
                "gene": gene,
                "n_consensuses": int(row["n_consensuses"].iloc[0]),
                "mean_length": float(row["mean_length"].iloc[0]),
                "mean_ambiguity": float(row["mean_ambiguity"].iloc[0]),
                "pi_factor": factor,
            })
        else:
            consensus_stats.append({
                "gene": gene,
                "n_consensuses": 0,
                "mean_length": float("nan"),
                "mean_ambiguity": float("nan"),
                "pi_factor": 1.0,
            })

    pi_hat = sum(adjusted_pi)
    return {
        "method": "consensus_adjusted",
        "gene_names": gene_names,
        "pi_vec": adjusted_pi,
        "pi_hat": round(pi_hat, 4),
        "weights": adjusted_pi,
        "base_pi": base_pi,
        "consensus_factors": factors,
        "consensus_stats": consensus_stats,
        "description": (
            "π adjusted using YEASTRACT binding consensus sequences. "
            "Promiscuous binders (many/flexible consensuses) get a higher pi_factor "
            "because their regulatory links tolerate sequence divergence after duplication."
        ),
    }


# ---------------------------------------------------------------------------
# Combined estimator: run all four methods and compare
# ---------------------------------------------------------------------------

def estimate_pi_all_methods(
    gene_names: List[str],
    m: int,
    n: int,
    observed_count: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run all four estimation methods and return a comparison DataFrame.

    If observed_count is None, MLE method uses the Full Duplication expected
    count as a substitute (treats the network as if it were at the Full
    Duplication expectation).
    """
    k = len(gene_names)

    # Method 1
    ev_result = estimate_pi_from_evidence(gene_names)

    # Method 2 (MLE)
    if observed_count is None:
        observed_count = expected_full(k, m, n)
    mle_result = estimate_pi_from_mle(observed_count, m, n, gene_names)

    # Method 3 (SNP)
    snp_result = estimate_pi_from_snp(gene_names)

    # Method 4 (Consensus-adjusted)
    cons_result = estimate_pi_consensus_adjusted(gene_names)

    rows = []
    for i, gene in enumerate(gene_names):
        rows.append({
            "gene_family": gene,
            "pi_evidence": ev_result["pi_vec"][i],
            "pi_mle": mle_result["pi_vec"][i],
            "pi_snp": snp_result["pi_vec"][i],
            "pi_consensus": cons_result["pi_vec"][i],
        })

    df = pd.DataFrame(rows)
    df["pi_ensemble"] = df[["pi_evidence", "pi_mle", "pi_snp", "pi_consensus"]].mean(axis=1).round(4)
    return df


# ---------------------------------------------------------------------------
# Full significance analysis
# ---------------------------------------------------------------------------

def full_significance_analysis(
    pi_vec: List[float],
    m: int,
    n: int,
    observed_count: float,
    k: Optional[int] = None,
) -> Dict:
    """
    Run the complete Scruse et al. significance test for a subnetwork motif.

    Computes:
      - Expected count under Full Duplication (Theorem 1)
      - Expected count under Partial Duplication with π⃗ (Theorem 4)
      - Variance under Full Duplication (Corollary 2)
      - Variance under Binary Inheritance (Corollary 16)
      - Z-scores and p-values for both models
      - Significance labels
      - Bounds on expected count (Corollary 9)

    Returns a dict suitable for display in the frontend.
    """
    if k is None:
        k = len(pi_vec)
    pi_hat = sum(pi_vec)

    # Expected counts
    e_full = expected_full(k, m, n)
    e_partial = expected_partial(pi_hat, m, n)

    # Variances
    var_full = variance_full(k, m, n)
    var_binary = variance_binary(pi_vec, m, n)

    # Bounds
    lo, hi = expected_bounds_partial(pi_hat, m, n, k)

    # Z-scores
    z_full = z_score(observed_count, e_full, var_full)
    z_partial = z_score(observed_count, e_partial, var_binary)

    # P-values
    p_full = p_value_from_z(z_full)
    p_partial = p_value_from_z(z_partial)

    # Second moments
    e2_binary = second_moment_binary(pi_vec, m, n)

    return {
        # Model parameters
        "k": k,
        "m": m,
        "n": n,
        "pi_vec": [round(p, 4) for p in pi_vec],
        "pi_hat": round(pi_hat, 4),
        "observed_count": observed_count,
        # Full Duplication (Theorem 1 / Corollary 2)
        "expected_full": round(e_full, 4),
        "variance_full": round(var_full, 6),
        "std_full": round(np.sqrt(max(var_full, 0)), 4),
        "z_full": round(z_full, 4) if not np.isnan(z_full) else None,
        "p_full": round(p_full, 6) if not np.isnan(p_full) else None,
        "sig_full": significance_label(p_full),
        # Partial Duplication (Theorem 4 / Corollary 16)
        "expected_partial": round(e_partial, 4),
        "variance_binary": round(var_binary, 6),
        "std_binary": round(np.sqrt(max(var_binary, 0)), 4),
        "second_moment_binary": round(e2_binary, 4),
        "z_partial": round(z_partial, 4) if not np.isnan(z_partial) else None,
        "p_partial": round(p_partial, 6) if not np.isnan(p_partial) else None,
        "sig_partial": significance_label(p_partial),
        # Bounds (Corollary 9)
        "expected_partial_lower_bound": round(lo, 4),
        "expected_partial_upper_bound": round(hi, 4),
        # Growth rates (from paper: Θ(nᵏ) for Full, Θ(n^π̂) for Partial)
        "growth_exponent_full": k,
        "growth_exponent_partial": round(pi_hat, 4),
        "interpretation": _interpret(observed_count, e_partial, p_partial),
    }


def _interpret(observed: float, expected: float, p: float) -> str:
    """Plain-language interpretation of the significance result."""
    if np.isnan(p):
        return "Insufficient data to assess significance."
    fold = observed / max(expected, 1e-9)
    direction = "over-represented" if observed > expected else "under-represented"
    sig = "significantly" if p < 0.05 else "not significantly"
    return (
        f"The observed motif count ({observed:.1f}) is {sig} {direction} "
        f"relative to the Partial Duplication null model (expected ~{expected:.2f}, "
        f"fold change {fold:.2f}x, p = {p:.4f}). "
        + (
            "This pattern is consistent with positive selection for this regulatory wiring."
            if (observed > expected and p < 0.05)
            else "This is consistent with neutral evolution under gene duplication."
            if p >= 0.05
            else "This pattern is consistent with negative selection against this wiring."
        )
    )


# ---------------------------------------------------------------------------
# π sensitivity analysis
# ---------------------------------------------------------------------------

def pi_sensitivity(
    m: int,
    n: int,
    k: int,
    pi_hat_range: Tuple = (0.0, float("inf")),
    steps: int = 50,
) -> pd.DataFrame:
    """
    Compute expected count and variance over a range of π̂ values.
    Useful for showing how sensitive significance is to the π estimate.
    """
    max_pi_hat = min(pi_hat_range[1], float(k))
    pi_hats = np.linspace(pi_hat_range[0] + 1e-4, max_pi_hat, steps)
    rows = []
    for ph in pi_hats:
        e = expected_partial(ph, m, n)
        pi_vec_uniform = [ph / k] * k
        var = variance_binary(pi_vec_uniform, m, n)
        lo, hi = expected_bounds_partial(ph, m, n, k)
        rows.append({
            "pi_hat": round(ph, 4),
            "expected": round(e, 4),
            "std": round(np.sqrt(max(var, 0)), 4),
            "lower_bound": round(lo, 4),
            "upper_bound": round(hi, 4),
        })
    return pd.DataFrame(rows)
