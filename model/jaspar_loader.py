"""
jaspar_loader.py
Loads the JASPAR 2024 CORE yeast TF dataset.

Two source files (sgd_yeast_data/):
  jaspar_yeast_tfbs_2024.csv  — one row per TF: matrix metadata, PFM arrays,
                                consensus sequence, information content
  jaspar_yeast_pfm_long.csv   — long format: one row per (TF, position, nucleotide)

Integration with the Scruse et al. model:
  Information content (IC) measures how specific a TF's binding site is.
  High IC → narrow, specific binding sequence → regulatory links fragile to
  sequence change after duplication → lower π prior.
  Low IC / wide motifs → flexible binding → links survive divergence → higher π.

  pi_jaspar_factor multiplies the base evidence π, analogous to pi_consensus_factor
  from consensus_loader but grounded in quantitative PFM information content.
"""

from __future__ import annotations

import math
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent.parent
DATA_DIR = _HERE / "sgd_yeast_data"
TFBS_CSV = DATA_DIR / "jaspar_yeast_tfbs_2024.csv"
PFM_CSV  = DATA_DIR / "jaspar_yeast_pfm_long.csv"

_TFBS_DF: Optional[pd.DataFrame] = None
_PFM_DF:  Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# π factor from JASPAR information content
# ---------------------------------------------------------------------------

def _pi_factor_from_ic(ic_bits: float, motif_width: int) -> float:
    """
    Derive a multiplicative π adjustment from JASPAR IC and motif width.

    Rationale:
      - Higher IC → more conserved, specific binding site → lower π
        (the duplicate is less likely to inherit an exact high-IC site).
      - Wider motifs give more surface area for partial match → slightly higher π.
    Scale: factor in [0.80, 1.10], centred at 1.0 for average IC (~8 bits).
    """
    ic_norm   = min(1.0, ic_bits / 20.0)          # normalise to [0,1]; max ~20 bits
    width_norm = min(1.0, motif_width / 20.0)
    penalty    = ic_norm * 0.20                    # up to -0.20 for very specific sites
    bonus      = width_norm * 0.10                 # up to +0.10 for wide motifs
    return round(1.0 - penalty + bonus, 4)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_jaspar_tfbs(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load jaspar_yeast_tfbs_2024.csv.
    Returns one row per TF with cleaned numeric columns and pi_jaspar_factor.
    """
    global _TFBS_DF
    if _TFBS_DF is not None and not force_refresh:
        return _TFBS_DF

    df = pd.read_csv(TFBS_CSV, dtype=str).fillna("")

    for col in ("motif_width", "information_content_bits"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["pi_jaspar_factor"] = df.apply(
        lambda r: _pi_factor_from_ic(r["information_content_bits"], r["motif_width"]),
        axis=1,
    )
    _TFBS_DF = df.reset_index(drop=True)
    return _TFBS_DF


def load_jaspar_pfm(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load jaspar_yeast_pfm_long.csv.
    Returns long-format PFM table: (matrix_id, tf_name, position, nucleotide, count, pwm_score).
    """
    global _PFM_DF
    if _PFM_DF is not None and not force_refresh:
        return _PFM_DF

    df = pd.read_csv(PFM_CSV, dtype=str).fillna("")
    for col in ("position", "count", "pwm_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    _PFM_DF = df.reset_index(drop=True)
    return _PFM_DF


# ---------------------------------------------------------------------------
# Per-TF accessors
# ---------------------------------------------------------------------------

def get_jaspar_info(tf_name: str) -> dict:
    """
    Return JASPAR metadata for a single TF (case-insensitive SGD name).
    Returns an empty dict if the TF is not in JASPAR.
    """
    df = load_jaspar_tfbs()
    row = df[df["name"].str.upper() == tf_name.upper()]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "matrix_id":           r["matrix_id"],
        "name":                r["name"],
        "tf_class":            r["tf_class"],
        "tf_family":           r["tf_family"],
        "data_type":           r["data_type"],
        "motif_width":         int(r["motif_width"]),
        "information_content": float(r["information_content_bits"]),
        "consensus_sequence":  r["consensus_sequence"],
        "pi_jaspar_factor":    float(r["pi_jaspar_factor"]),
        "jaspar_url":          r["jaspar_url"],
    }


def get_jaspar_pfm(tf_name: str) -> pd.DataFrame:
    """Return PFM rows for a given TF (long format). Empty DataFrame if not found."""
    df = load_jaspar_pfm()
    return df[df["tf_name"].str.upper() == tf_name.upper()].reset_index(drop=True)


def get_pi_jaspar_factor(tf_name: str) -> float:
    """
    Return the pi_jaspar_factor for a TF.
    Values < 1 indicate high-IC (specific) binding → lower inheritance probability.
    Returns 1.0 if TF not in JASPAR.
    """
    info = get_jaspar_info(tf_name)
    return info.get("pi_jaspar_factor", 1.0)


def list_jaspar_tfs() -> list[str]:
    """Return sorted list of TF names in the JASPAR dataset (SGD notation)."""
    return sorted(load_jaspar_tfbs()["name"].str.upper().tolist())


def jaspar_tf_set() -> set[str]:
    """Return the set of JASPAR TF names in uppercase."""
    return set(load_jaspar_tfbs()["name"].str.upper().tolist())


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------

def jaspar_summary() -> dict:
    df = load_jaspar_tfbs()
    return {
        "n_jaspar_tfs":        len(df),
        "mean_ic_bits":        round(float(df["information_content_bits"].mean()), 2),
        "mean_motif_width":    round(float(df["motif_width"].mean()), 2),
        "n_chip_based":        int(df["data_type"].str.contains("ChIP", case=False, na=False).sum()),
        "n_pbm_based":         int(df["data_type"].str.contains("PBM",  case=False, na=False).sum()),
        "unique_tf_classes":   df["tf_class"].nunique(),
        "unique_tf_families":  df["tf_family"].nunique(),
    }


# ---------------------------------------------------------------------------
# JASPAR x YEASTRACT cross-reference
# ---------------------------------------------------------------------------

def build_jaspar_yeastract_crossref(save: bool = True) -> pd.DataFrame:
    """
    Cross-reference JASPAR 2024 CORE profiles against the 127 YEASTRACT TFs.

    Match types:
      'exact'  — case-insensitive name match
      'fuzzy'  — SequenceMatcher ratio >= 0.75 (handles minor name variants)
      'no_match' — no JASPAR profile found

    Output columns:
      yeastract_tf_name, jaspar_matrix_id, jaspar_tf_name, match_type,
      motif_width, information_content_bits, pfm_A, pfm_C, pfm_G, pfm_T
    """
    import difflib
    from .data_loader import YEASTRACT_TFS_SGD

    jaspar = load_jaspar_tfbs()
    jaspar_upper = {row["name"].upper(): row for _, row in jaspar.iterrows()}
    jaspar_names = list(jaspar_upper.keys())

    rows = []
    for tf in sorted(YEASTRACT_TFS_SGD):
        tf_up = tf.upper()

        # 1. Exact match (case-insensitive)
        if tf_up in jaspar_upper:
            j = jaspar_upper[tf_up]
            rows.append(_crossref_row(tf, j, "exact"))
            continue

        # 2. Fuzzy match
        matches = difflib.get_close_matches(tf_up, jaspar_names, n=1, cutoff=0.75)
        if matches:
            j = jaspar_upper[matches[0]]
            rows.append(_crossref_row(tf, j, "fuzzy"))
            continue

        # 3. No match
        rows.append({
            "yeastract_tf_name": tf,
            "jaspar_matrix_id": "",
            "jaspar_tf_name": "",
            "match_type": "no_match",
            "motif_width": None,
            "information_content_bits": None,
            "pfm_A": "",
            "pfm_C": "",
            "pfm_G": "",
            "pfm_T": "",
        })

    df = pd.DataFrame(rows)
    if save:
        out = DATA_DIR / "jaspar_yeastract_crossref.csv"
        df.to_csv(out, index=False)
    return df


def _crossref_row(yeastract_name: str, jaspar_row: pd.Series, match_type: str) -> dict:
    return {
        "yeastract_tf_name":       yeastract_name,
        "jaspar_matrix_id":        jaspar_row["matrix_id"],
        "jaspar_tf_name":          jaspar_row["name"],
        "match_type":              match_type,
        "motif_width":             jaspar_row["motif_width"],
        "information_content_bits": jaspar_row["information_content_bits"],
        "pfm_A":                   jaspar_row["pfm_A"],
        "pfm_C":                   jaspar_row["pfm_C"],
        "pfm_G":                   jaspar_row["pfm_G"],
        "pfm_T":                   jaspar_row["pfm_T"],
    }


def load_jaspar_yeastract_crossref(force_rebuild: bool = False) -> pd.DataFrame:
    """Load (or build) the JASPAR x YEASTRACT cross-reference table."""
    out = DATA_DIR / "jaspar_yeastract_crossref.csv"
    if not out.exists() or force_rebuild:
        return build_jaspar_yeastract_crossref(save=True)
    return pd.read_csv(out, dtype=str)
