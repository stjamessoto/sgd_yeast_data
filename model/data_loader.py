"""
data_loader.py
Loads and caches all SGD yeast CSV files. Provides clean DataFrames
with typed columns and normalised evidence codes for downstream analysis.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional

# Locate the data directory relative to this file
_HERE = Path(__file__).parent.parent  # sgd_yeast_data/
DATA_DIR = _HERE / "sgd_yeast_data"


# ---------------------------------------------------------------------------
# Evidence-code quality map (used across multiple modules)
# Higher value → more reliable experimental evidence → higher π prior
# ---------------------------------------------------------------------------

EVIDENCE_QUALITY: dict[str, float] = {
    "IDA": 0.90,   # Inferred from Direct Assay
    "IMP": 0.82,   # Inferred from Mutant Phenotype
    "IGI": 0.72,   # Inferred from Genetic Interaction
    "IPI": 0.68,   # Inferred from Physical Interaction
    "IBA": 0.58,   # Inferred from Biological Aspect of Ancestor
    "HDA": 0.55,   # High-throughput Direct Assay
    "IC":  0.50,   # Inferred by Curator
    "RCA": 0.45,   # Inferred from Reviewed Computational Analysis
    "IEA": 0.30,   # Inferred from Electronic Annotation (automated)
    "ND":  0.10,   # No biological Data
    "NAS": 0.20,   # Non-traceable Author Statement
    "TAS": 0.35,   # Traceable Author Statement
}


def _evidence_score(codes_str: str) -> float:
    """
    Compute mean evidence quality score from a pipe-separated code string.
    E.g. 'IDA|IMP|IEA' → mean of their quality scores.
    """
    if not isinstance(codes_str, str) or codes_str.strip() == "":
        return EVIDENCE_QUALITY["ND"]
    codes = [c.strip() for c in codes_str.split("|") if c.strip()]
    scores = [EVIDENCE_QUALITY.get(c, EVIDENCE_QUALITY["ND"]) for c in codes]
    return float(np.mean(scores)) if scores else EVIDENCE_QUALITY["ND"]


def _parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)


# ---------------------------------------------------------------------------
# TF filter — union of JASPAR 2024 and YEASTRACT curated lists
# Using the union maximises coverage: 179 TFs present in the SGD CSV.
# ---------------------------------------------------------------------------

# YEASTRACT curated list (127 TFs)
YEASTRACT_TFS_SGD: set[str] = {
    "SMP1","ERT1","KAR4","HCM1","RDS1","UGA3","PHO2","MBP1",
    "RPN4","LYS14","NRG1","GIS1","INO2","SWI5","UME6","UPC2",
    "ADR1","MET32","SUM1","CAD1","STP1","GCN4","MIG3","GLN3",
    "ACA1","MOT2","FLO8","SWI4","COM2","RPH1","ZNF1","HAC1",
    "GAT1","PHO4","FZF1","HAP2","MIG2","CUP2","SUT1","HSF1",
    "AFT1","MIG1","PDR1","RME1","RIM101","YAP3","STP2","STE12",
    "NDT80","ROF1","STB5","SKN7","FKH1","XBP1","CST6","YAP5",
    "DAL81","GZF3","GSM1","CBF1","IME1","ASH1","ABF1","HAP4",
    "MSN4","RGT1","IXR1","PUT3","DAL80","BAS1","PPR1","CHA4",
    "ACE2","RFX1","ECM22","HAP1","PDR8","LEU3","ARG81","TDA9",
    "WAR1","YAP1","STB4","MAC1","MSN2","ARG80","MCM1","MOT3",
    "MSS11","HOT1","CAT8","DAL82","RAP1","GCR2","SKO1","MET4",
    "FKH2","CRZ1","INO4","RTG1","CIN5","AZF1","SFL1","YRR1",
    "TEA1","TYE7","HAP5","PIP2","GAL4","USV1","AFT2","RDS2",
    "RLM1","GCR1","MET31","HAA1","ROX1","ARR1","OAF1","RTG3",
    "HAP3","PDR3","REB1","NRG2","TEC1","SIP4","ZAP1",
}

def _load_jaspar_tf_set() -> set[str]:
    try:
        p = DATA_DIR / "jaspar_yeast_tfbs_2024.csv"
        df = pd.read_csv(p, usecols=["name"], dtype=str)
        return set(df["name"].str.upper().dropna().tolist())
    except Exception:
        return set()

# JASPAR 2024 (177 TFs) — loaded from CSV
JASPAR_TFS_SGD: set[str] = _load_jaspar_tf_set()

# Union: 189 unique TFs; 179 present in the SGD CSV
COMBINED_TFS_SGD: set[str] = JASPAR_TFS_SGD | YEASTRACT_TFS_SGD


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_transcription_factors() -> pd.DataFrame:
    """
    Load sgd_transcription_factors.csv.
    Adds:
      - evidence_score: float [0,1] mean quality of evidence codes
      - pi_prior: estimated inheritance probability from evidence quality
    """
    path = DATA_DIR / "sgd_transcription_factors.csv"
    df = pd.read_csv(path, dtype=str).fillna("")

    bool_cols = ["has_dna_binding", "is_activator", "is_repressor"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(_parse_bool)

    int_cols = ["n_function_terms", "n_process_terms"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["evidence_score"] = df["evidence_codes"].apply(_evidence_score)

    # π prior from evidence quality; DNA-binding TFs with activation role weighted up
    def _pi_prior(row) -> float:
        base = row["evidence_score"]
        if row.get("has_dna_binding", False):
            base = min(1.0, base + 0.05)
        if row.get("is_activator", False):
            base = min(1.0, base + 0.03)
        return round(base, 4)

    df["pi_prior"] = df.apply(_pi_prior, axis=1)

    # Parse synonyms into list
    df["synonym_list"] = df["synonyms"].apply(
        lambda s: [x.strip() for x in s.split("|") if x.strip()] if s else []
    )

    # GO term lists
    for col in ["tf_function_go", "tf_process_go"]:
        list_col = col + "_list"
        df[list_col] = df[col].apply(
            lambda s: [x.strip() for x in s.split("|") if x.strip()] if s else []
        )

    # Filter to union of JASPAR 2024 + YEASTRACT (179 TFs present in SGD CSV)
    df = df[df["gene_name"].str.upper().isin(COMBINED_TFS_SGD)]

    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_tf_go_annotations() -> pd.DataFrame:
    """
    Load sgd_tf_go_annotations.csv.
    Covers GO annotations specifically for transcription factors.
    Adds:
      - evidence_score: float quality of the evidence code
    """
    path = DATA_DIR / "sgd_tf_go_annotations.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    df["evidence_score"] = df["evidence"].apply(
        lambda c: EVIDENCE_QUALITY.get(c.strip(), EVIDENCE_QUALITY["ND"])
    )
    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_go_annotations_full() -> pd.DataFrame:
    """
    Load sgd_go_annotations_full.csv (~120K rows, all genes).
    Subsets only Biological Process (go_aspect == 'P') for efficiency in
    regulatory-network construction. Full dataset retained as attribute.
    Adds:
      - evidence_score: float
    """
    path = DATA_DIR / "sgd_go_annotations_full.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    df["evidence_score"] = df["evidence_code"].apply(
        lambda c: EVIDENCE_QUALITY.get(c.strip(), EVIDENCE_QUALITY["ND"])
    )
    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_chromosome_lengths() -> pd.DataFrame:
    """
    Load sgd_chromosome_lengths.csv.
    Adds length_mb column for display convenience.
    """
    path = DATA_DIR / "sgd_chromosome_lengths.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    df["length_bp"] = pd.to_numeric(df["length_bp"], errors="coerce")
    df["length_mb"] = (df["length_bp"] / 1e6).round(3)
    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_inheritance_vectors() -> pd.DataFrame:
    """
    Load sgd_YFL039C_inheritance_vectors.csv.
    pct_alt ≈ percentage of alternative alleles relative to reference strain.
    Adds:
      - pi_snp: per-strain inheritance probability proxy = 1 − (pct_alt / 100)
    """
    path = DATA_DIR / "sgd_YFL039C_inheritance_vectors.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in ["n_alt_alleles", "n_snp_positions", "pct_alt", "pos_354", "pos_945"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["pi_snp"] = (1.0 - df["pct_alt"] / 100.0).clip(0.0, 1.0).round(4)
    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_snps() -> pd.DataFrame:
    """Load sgd_YFL039C_snps.csv — SNP positions across strains."""
    path = DATA_DIR / "sgd_YFL039C_snps.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    df["position_bp"] = pd.to_numeric(df["position_bp"], errors="coerce")
    return df.reset_index(drop=True)


def load_all() -> dict[str, pd.DataFrame]:
    """Convenience: load all datasets and return as a dict."""
    return {
        "tfs": load_transcription_factors(),
        "tf_go": load_tf_go_annotations(),
        "go_full": load_go_annotations_full(),
        "chromosomes": load_chromosome_lengths(),
        "inheritance_vectors": load_inheritance_vectors(),
        "snps": load_snps(),
    }


# ---------------------------------------------------------------------------
# Summary statistics (for frontend dashboard)
# ---------------------------------------------------------------------------

def dataset_summary() -> dict:
    """Return high-level counts for display."""
    tfs = load_transcription_factors()
    go = load_go_annotations_full()
    ivec = load_inheritance_vectors()
    chrom = load_chromosome_lengths()

    return {
        "n_tfs": len(tfs),
        "n_dna_binding_tfs": int(tfs["has_dna_binding"].sum()),
        "n_activators": int(tfs["is_activator"].sum()),
        "n_repressors": int(tfs["is_repressor"].sum()),
        "n_go_annotations": len(go),
        "n_unique_genes_go": go["gene_name"].nunique(),
        "n_chromosomes": len(chrom),
        "genome_size_mb": round(chrom["length_bp"].sum() / 1e6, 2),
        "n_strains_snp": int(ivec["strain"].nunique()),
        "example_gene_snp": str(ivec["gene"].iloc[0]) if len(ivec) else "N/A",
    }
