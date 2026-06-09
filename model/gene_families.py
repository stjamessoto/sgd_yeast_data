"""
gene_families.py
Gene-family construction and parameter estimation for the Scruse et al. model.

In the duplication model (Section 2, Proposition 1):
  - m  = initial number of gene families (each with 1 ancestral gene)
  - n  = total gene count after d = n − m duplications
  - cᵢ = size of family i  (Σcᵢ = n)
  - ⃗c  = (c₁, ..., cₘ) uniformly distributed over all compositions of n into m parts

Gene families here are operationally defined by shared GO Biological Process
terms among TFs.  Each distinct GO process cluster is one "family"; TFs within
a cluster are the duplicated members of that family.

This is biologically motivated: paralogs that arose by duplication tend to
retain overlapping functional annotation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from .data_loader import load_transcription_factors, load_go_annotations_full


# ---------------------------------------------------------------------------
# Build gene families from GO process clustering
# ---------------------------------------------------------------------------

def build_go_process_families(
    go_aspect: str = "P",
    min_family_size: int = 1,
    gene_subset: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """
    Group genes into families based on shared GO Biological Process terms.

    Each GO process term defines a family.  A gene may belong to multiple
    families (multi-membership is biologically realistic).

    Returns a DataFrame with columns:
      go_id, go_family_size, member_genes (list), mean_evidence_score
    """
    go_full = load_go_annotations_full()
    df = go_full[go_full["go_aspect"] == go_aspect].copy()
    if gene_subset is not None:
        df = df[df["gene_name"].isin(gene_subset)]

    families = (
        df.groupby("go_id")
        .agg(
            go_family_size=("gene_name", "nunique"),
            member_genes=("gene_name", lambda x: sorted(x.unique().tolist())),
            mean_evidence_score=("evidence_score", "mean"),
        )
        .reset_index()
    )
    families = families[families["go_family_size"] >= min_family_size]
    return families.sort_values("go_family_size", ascending=False).reset_index(drop=True)


def _build_go_tf_families(go_list_col: str, min_family_size: int) -> pd.DataFrame:
    """Group TFs by a parsed GO list column (tf_process_go_list or tf_function_go_list)."""
    tfs = load_transcription_factors()
    rows = []
    for _, row in tfs.iterrows():
        for go_id in row[go_list_col]:
            rows.append({
                "gene_name":      row["gene_name"],
                "go_id":          go_id,
                "evidence_score": row["evidence_score"],
                "is_activator":   row["is_activator"],
                "is_repressor":   row["is_repressor"],
            })
    if not rows:
        return pd.DataFrame()
    exploded = pd.DataFrame(rows)
    families = (
        exploded.groupby("go_id")
        .agg(
            family_size=("gene_name", "nunique"),
            member_tfs=("gene_name", lambda x: sorted(x.unique().tolist())),
            mean_evidence_score=("evidence_score", "mean"),
            n_activators=("is_activator", "sum"),
            n_repressors=("is_repressor", "sum"),
        )
        .reset_index()
    )
    families = families[families["family_size"] >= min_family_size]
    return families.sort_values("family_size", ascending=False).reset_index(drop=True)


def _build_component_tf_families(min_family_size: int) -> pd.DataFrame:
    """Group TFs by GO Cellular Component annotations (from the full GO file)."""
    tfs = load_transcription_factors()
    tf_names = set(tfs["gene_name"].str.upper())
    go_full = load_go_annotations_full()
    # Keep only the GO ID and gene name from the full GO slice (drop its evidence_score
    # to avoid a column conflict with the TF-level evidence_score added below).
    df = go_full.loc[
        (go_full["go_aspect"] == "C") &
        (go_full["gene_name"].str.upper().isin(tf_names)),
        ["go_id", "gene_name"],
    ].copy()
    df = df.merge(
        tfs[["gene_name", "evidence_score", "is_activator", "is_repressor"]],
        on="gene_name", how="left",
    )
    df["evidence_score"] = df["evidence_score"].fillna(0.10)
    df["is_activator"]   = df["is_activator"].fillna(False)
    df["is_repressor"]   = df["is_repressor"].fillna(False)
    families = (
        df.groupby("go_id")
        .agg(
            family_size=("gene_name", "nunique"),
            member_tfs=("gene_name", lambda x: sorted(x.unique().tolist())),
            mean_evidence_score=("evidence_score", "mean"),
            n_activators=("is_activator", "sum"),
            n_repressors=("is_repressor", "sum"),
        )
        .reset_index()
    )
    families = families[families["family_size"] >= min_family_size]
    return families.sort_values("family_size", ascending=False).reset_index(drop=True)


def _build_jaspar_tf_families(jaspar_col: str, min_family_size: int) -> pd.DataFrame:
    """Group TFs by JASPAR binding-domain class or family (protein-sequence proxy)."""
    from .jaspar_loader import load_jaspar_tfbs
    tfs = load_transcription_factors()
    jaspar = load_jaspar_tfbs()
    merged = tfs.merge(
        jaspar[["name", jaspar_col]].rename(columns={"name": "gene_name"}),
        on="gene_name", how="inner",
    )
    merged = merged[merged[jaspar_col].str.strip() != ""]
    if merged.empty:
        return pd.DataFrame()
    families = (
        merged.groupby(jaspar_col)
        .agg(
            family_size=("gene_name", "nunique"),
            member_tfs=("gene_name", lambda x: sorted(x.unique().tolist())),
            mean_evidence_score=("evidence_score", "mean"),
            n_activators=("is_activator", "sum"),
            n_repressors=("is_repressor", "sum"),
        )
        .reset_index()
        .rename(columns={jaspar_col: "go_id"})
    )
    families = families[families["family_size"] >= min_family_size]
    return families.sort_values("family_size", ascending=False).reset_index(drop=True)


def build_tf_families(min_family_size: int = 1, grouping: str = "GO_Process") -> pd.DataFrame:
    """
    Build gene families restricted to transcription factors.

    grouping options:
      'GO_Process'    — shared GO Biological Process term (default; paper Section 2)
      'GO_Function'   — shared GO Molecular Function term (binding-type proxy)
      'GO_Component'  — shared GO Cellular Component term (subcellular location)
      'JASPAR_Class'  — JASPAR binding-domain class (protein-sequence similarity proxy)
      'JASPAR_Family' — JASPAR binding-domain family (finer sequence grouping)

    Returns DataFrame with columns:
      go_id, family_size (= cᵢ), member_tfs, mean_evidence_score,
      n_activators, n_repressors
    """
    if grouping == "GO_Process":
        return _build_go_tf_families("tf_process_go_list", min_family_size)
    elif grouping == "GO_Function":
        return _build_go_tf_families("tf_function_go_list", min_family_size)
    elif grouping == "GO_Component":
        return _build_component_tf_families(min_family_size)
    elif grouping == "JASPAR_Class":
        return _build_jaspar_tf_families("tf_class", min_family_size)
    elif grouping == "JASPAR_Family":
        return _build_jaspar_tf_families("tf_family", min_family_size)
    else:
        raise ValueError(f"Unknown grouping method: {grouping!r}")


# ---------------------------------------------------------------------------
# Duplication model parameters  m, n, and composition ⃗c
# ---------------------------------------------------------------------------

def estimate_model_parameters(
    family_df: pd.DataFrame,
    size_col: str = "family_size",
) -> Dict[str, int | List[int]]:
    """
    Derive Scruse et al. model parameters from a family DataFrame.

    m = number of gene families (rows)
    n = total genes across all families = Σcᵢ
    c_vec = vector of family sizes (c₁, ..., cₘ)
    d = n − m = estimated number of duplication events

    Proposition 1: after d = n − m random duplications from m singletons,
    all compositions of n into m parts are equally likely.
    """
    c_vec = family_df[size_col].tolist()
    m = len(c_vec)
    n = int(np.sum(c_vec))
    d = n - m
    return {
        "m": m,
        "n": n,
        "d": d,
        "c_vec": c_vec,
        "mean_family_size": round(n / max(m, 1), 3),
        "max_family_size": int(max(c_vec)) if c_vec else 0,
        "min_family_size": int(min(c_vec)) if c_vec else 0,
    }


def composition_probability(c_vec: List[int]) -> float:
    """
    Proposition 1: Probability of any single composition ⃗c under uniform
    distribution = 1 / C(n−1, m−1).
    """
    from math import comb
    n = sum(c_vec)
    m = len(c_vec)
    if n < m or m < 1:
        return float("nan")
    denom = comb(n - 1, m - 1)
    return 1.0 / denom if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Family-size distribution (for plotting)
# ---------------------------------------------------------------------------

def family_size_distribution(family_df: pd.DataFrame, size_col: str = "family_size") -> pd.Series:
    """Return value counts of family sizes (for histogram)."""
    return family_df[size_col].value_counts().sort_index()


# ---------------------------------------------------------------------------
# Subnetwork motif family selection
# ---------------------------------------------------------------------------

def select_motif_families(
    tf_family_df: pd.DataFrame,
    k: int,
    strategy: str = "largest",
) -> List[Dict]:
    """
    Select k gene families to form a subnetwork motif M of size k.

    strategy:
      'largest'    — families with most members (maximises expected count)
      'highest_ev' — families with highest mean evidence score (most reliable)
      'random'     — random sample
      'balanced'   — mix of large and high-evidence families

    Returns list of k dicts: {go_id, family_size, member_tfs, mean_evidence_score}
    """
    if len(tf_family_df) < k:
        raise ValueError(f"Only {len(tf_family_df)} families available; cannot select k={k}")

    if strategy == "largest":
        selected = tf_family_df.nlargest(k, "family_size")
    elif strategy == "highest_ev":
        selected = tf_family_df.nlargest(k, "mean_evidence_score")
    elif strategy == "random":
        selected = tf_family_df.sample(k, random_state=42)
    elif strategy == "balanced":
        half = k // 2
        large = tf_family_df.nlargest(half, "family_size")
        rest = tf_family_df.drop(large.index)
        high_ev = rest.nlargest(k - half, "mean_evidence_score")
        selected = pd.concat([large, high_ev])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return selected.to_dict("records")


# ---------------------------------------------------------------------------
# Observed motif instance count
# ---------------------------------------------------------------------------

def count_observed_motif_instances(
    family_records: List[Dict],
    mode: str = "full_duplication",
) -> int:
    """
    Count observed instances |M(n)| for a k-tuple of gene families.

    Under Full Duplication (Section 3.1):
      |M(n)| = c₁ × c₂ × ... × cₖ   (Cartesian product of family members)

    Under Partial Duplication:
      |M(n)| ≤ c₁ × ... × cₖ (a random subset; exact count requires inheritance data)
      Here we use the Full Duplication count as an upper bound / approximation.

    Returns the Cartesian-product count (Full Duplication upper bound).
    """
    product = 1
    for rec in family_records:
        product *= int(rec.get("family_size", 1))
    return product
