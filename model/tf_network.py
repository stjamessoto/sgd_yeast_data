"""
tf_network.py
Transcription-factor network construction from SGD data.

Answers three biological questions from the Scruse et al. framework:
  1. Which genes are transcription factors?
  2. What do TF binding sites bind to?
  3. Which transcription factor regulates which gene?

Approach:
  - TFs are identified from sgd_transcription_factors.csv.
  - Binding sites bind to promoter DNA of target genes (cis-regulatory elements);
    confirmed by GO:0003677 (DNA binding), GO:0000981 (RNAP-II TF activity),
    GO:0001228 (TF activator activity).
  - TF→gene regulation is inferred by shared GO Biological Process terms:
    genes whose GO process annotations overlap with a TF's process GO terms
    are treated as candidate regulatory targets.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
from .data_loader import (
    load_transcription_factors,
    load_tf_go_annotations,
    load_go_annotations_full,
    EVIDENCE_QUALITY,
)
from .consensus_loader import (
    get_consensus_for_tf,
    get_pi_consensus_factor,
    load_tf_consensus_stats,
)
from .jaspar_loader import (
    get_jaspar_info,
    get_pi_jaspar_factor,
)

# GO terms that confirm DNA-binding / TF activity
BINDING_GO_TERMS = {
    "GO:0003677": "DNA binding",
    "GO:0000981": "RNA pol II-specific DNA-binding TF activity",
    "GO:0001228": "DNA-binding TF activity (activator)",
    "GO:0001227": "DNA-binding TF activity (repressor)",
    "GO:0003700": "DNA-binding TF activity (general)",
}

# GO terms confirming transcriptional regulation (process side)
REGULATION_GO_TERMS = {
    "GO:0006355": "regulation of transcription, DNA-templated",
    "GO:0045944": "positive regulation of transcription (RNAP II)",
    "GO:0000122": "negative regulation of transcription (RNAP II)",
    "GO:0006357": "regulation of transcription by RNAP II",
}


# ---------------------------------------------------------------------------
# 1. TF identification
# ---------------------------------------------------------------------------

def get_transcription_factors(
    require_dna_binding: bool = False,
    only_activators: bool = False,
    only_repressors: bool = False,
    min_evidence_score: float = 0.0,
) -> pd.DataFrame:
    """
    Return TFs from sgd_transcription_factors.csv with optional filters.

    Columns added:
      - role:        'activator', 'repressor', 'dual', or 'unknown'
      - binding_go:  list of binding-related GO function terms
      - regulation_go: list of regulation-related GO process terms
    """
    df = load_transcription_factors().copy()

    if require_dna_binding:
        df = df[df["has_dna_binding"]]
    if only_activators:
        df = df[df["is_activator"]]
    if only_repressors:
        df = df[df["is_repressor"]]
    if min_evidence_score > 0:
        df = df[df["evidence_score"] >= min_evidence_score]

    def _role(row) -> str:
        a, r = row["is_activator"], row["is_repressor"]
        if a and r:
            return "dual"
        if a:
            return "activator"
        if r:
            return "repressor"
        return "unknown"

    df["role"] = df.apply(_role, axis=1)

    df["binding_go"] = df["tf_function_go_list"].apply(
        lambda terms: [t for t in terms if t in BINDING_GO_TERMS]
    )
    df["regulation_go"] = df["tf_process_go_list"].apply(
        lambda terms: [t for t in terms if t in REGULATION_GO_TERMS]
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Binding site characterisation
# ---------------------------------------------------------------------------

def describe_binding_sites(tf_name: str) -> dict:
    """
    For a given TF, describe what its binding sites bind to.

    Returns a dict with:
      - binding_go_terms: GO terms confirming DNA binding
      - binding_description: human-readable description
      - target_dna_type: 'promoter DNA', 'mitochondrial DNA', etc.
      - is_sequence_specific: True if RNA pol II-specific binding
    """
    tfs = load_transcription_factors()
    row = tfs[tfs["gene_name"].str.upper() == tf_name.upper()]
    if row.empty:
        return {"error": f"TF '{tf_name}' not found"}

    row = row.iloc[0]
    func_terms = row["tf_function_go_list"]
    binding_hits = {t: BINDING_GO_TERMS[t] for t in func_terms if t in BINDING_GO_TERMS}

    # Infer DNA target type from full name and GO terms
    full = row["full_name"].lower()
    if "mitochondrial" in full:
        dna_type = "mitochondrial DNA (promoter)"
    elif "rna pol ii" in full or "GO:0000981" in func_terms:
        dna_type = "nuclear promoter DNA (RNAP-II target)"
    elif "GO:0003677" in func_terms:
        dna_type = "genomic DNA (promoter / regulatory element)"
    else:
        dna_type = "DNA (regulatory element)"

    is_specific = "GO:0000981" in func_terms or "GO:0003700" in func_terms

    # JASPAR PFM data (primary binding site source)
    jaspar = get_jaspar_info(row["gene_name"])
    has_jaspar = bool(jaspar)
    pi_jaspar = get_pi_jaspar_factor(row["gene_name"])

    # YEASTRACT consensus sequences (secondary / fallback)
    consensus_df = get_consensus_for_tf(row["gene_name"])
    has_consensus = not consensus_df.empty
    consensus_sequences = consensus_df["consensus"].tolist() if has_consensus else []
    pi_consensus = get_pi_consensus_factor(row["gene_name"])

    # Combined π factor: JASPAR takes priority; fall back to YEASTRACT
    pi_factor = pi_jaspar if has_jaspar else pi_consensus

    # Binding description: use JASPAR consensus if available
    jaspar_consensus = jaspar.get("consensus_sequence", "")
    binding_note = (
        f"JASPAR PFM: consensus {jaspar_consensus!r}, "
        f"motif width {jaspar.get('motif_width')} bp, "
        f"IC = {jaspar.get('information_content', 0):.2f} bits "
        f"({jaspar.get('data_type', 'unknown')} data)."
        if has_jaspar
        else (
            f"YEASTRACT: {len(consensus_sequences)} consensus sequence(s)"
            f" (e.g. {consensus_sequences[0]!r})."
            if consensus_sequences
            else "No binding sequence data found."
        )
    )

    return {
        "gene_name":            row["gene_name"],
        "full_name":            row["full_name"],
        "binding_go_terms":     binding_hits,
        "binding_description": (
            "Binds sequence-specifically to promoter DNA of target genes via "
            "transcription-factor binding sites (TFBS / cis-regulatory elements)."
            if is_specific else "Binds DNA through general DNA-binding domain."
        ),
        "target_dna_type":      dna_type,
        "is_sequence_specific": is_specific,
        "is_activator":         bool(row["is_activator"]),
        "is_repressor":         bool(row["is_repressor"]),
        "evidence_score":       float(row["evidence_score"]),
        # JASPAR PFM data
        "has_jaspar":               has_jaspar,
        "jaspar_matrix_id":         jaspar.get("matrix_id", ""),
        "jaspar_tf_class":          jaspar.get("tf_class", ""),
        "jaspar_tf_family":         jaspar.get("tf_family", ""),
        "jaspar_data_type":         jaspar.get("data_type", ""),
        "jaspar_consensus":         jaspar_consensus,
        "jaspar_motif_width":       jaspar.get("motif_width", 0),
        "jaspar_ic_bits":           jaspar.get("information_content", 0.0),
        "jaspar_url":               jaspar.get("jaspar_url", ""),
        "pi_jaspar_factor":         pi_jaspar,
        # YEASTRACT consensus (supplementary)
        "has_yeastract_consensus":  has_consensus,
        "n_consensus_sequences":    len(consensus_sequences),
        "consensus_sequences":      consensus_sequences[:20],
        "consensus_sequences_all":  consensus_sequences,
        "pi_consensus_factor":      pi_consensus,
        # Combined
        "pi_factor":                pi_factor,
        "binding_note":             binding_note,
    }


# ---------------------------------------------------------------------------
# 3. TF → target gene regulation map
# ---------------------------------------------------------------------------

def build_tf_target_map(
    max_tfs: Optional[int] = None,
    min_evidence_score: float = 0.0,
) -> pd.DataFrame:
    """
    Build a long-format DataFrame of (tf_name, target_gene, shared_go_process, evidence_score).

    Method: a gene is a candidate target of TF X if both share at least one
    Biological Process GO term (go_aspect == 'P').  Evidence quality is the
    mean of the TF's evidence score and the target gene's evidence score for
    that term.

    This is consistent with the Scruse et al. framework in which a
    subnetwork motif instance I = (a₁, ..., aₖ) represents one TF gene aᵢ
    from each participating family regulating a shared target.
    """
    tfs = get_transcription_factors(min_evidence_score=min_evidence_score)
    if max_tfs:
        tfs = tfs.head(max_tfs)

    go_full = load_go_annotations_full()
    go_process = go_full[go_full["go_aspect"] == "P"].copy()

    # Build lookup: go_id → {gene_name: evidence_score}
    go_gene_map: dict[str, dict[str, float]] = {}
    for _, row in go_process.iterrows():
        gid = row["go_id"]
        if gid not in go_gene_map:
            go_gene_map[gid] = {}
        go_gene_map[gid][row["gene_name"]] = max(
            go_gene_map[gid].get(row["gene_name"], 0.0),
            float(row["evidence_score"]),
        )

    records = []
    for _, tf_row in tfs.iterrows():
        tf_name = tf_row["gene_name"]
        tf_ev = float(tf_row["evidence_score"])
        process_terms = tf_row["tf_process_go_list"]

        seen: dict[str, tuple[str, float]] = {}  # target → (go_id, score)
        for go_id in process_terms:
            if go_id not in go_gene_map:
                continue
            for target, target_ev in go_gene_map[go_id].items():
                if target == tf_name:
                    continue  # skip self-regulation (can be added separately)
                combined_ev = (tf_ev + target_ev) / 2.0
                if target not in seen or seen[target][1] < combined_ev:
                    seen[target] = (go_id, combined_ev)

        for target, (go_id, score) in seen.items():
            records.append(
                {
                    "tf_name": tf_name,
                    "target_gene": target,
                    "shared_go_process": go_id,
                    "evidence_score": round(score, 4),
                    "tf_role": tf_row["role"],
                    "tf_is_dna_binding": bool(tf_row["has_dna_binding"]),
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values(["tf_name", "evidence_score"], ascending=[True, False]).reset_index(drop=True)


def get_targets_of(tf_name: str, tf_target_map: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return all targets regulated by a specific TF (case-insensitive)."""
    if tf_target_map is None:
        tf_target_map = build_tf_target_map()
    mask = tf_target_map["tf_name"].str.upper() == tf_name.upper()
    return tf_target_map[mask].reset_index(drop=True)


def get_regulators_of(gene_name: str, tf_target_map: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return all TFs that regulate a specific gene."""
    if tf_target_map is None:
        tf_target_map = build_tf_target_map()
    mask = tf_target_map["target_gene"].str.upper() == gene_name.upper()
    return tf_target_map[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Network statistics
# ---------------------------------------------------------------------------

def network_statistics(tf_target_map: pd.DataFrame) -> dict:
    """Compute summary statistics for the regulatory network."""
    if tf_target_map.empty:
        return {}
    return {
        "n_tfs": int(tf_target_map["tf_name"].nunique()),
        "n_targets": int(tf_target_map["target_gene"].nunique()),
        "n_edges": len(tf_target_map),
        "mean_targets_per_tf": round(
            tf_target_map.groupby("tf_name")["target_gene"].count().mean(), 2
        ),
        "mean_tfs_per_target": round(
            tf_target_map.groupby("target_gene")["tf_name"].count().mean(), 2
        ),
        "median_evidence_score": round(float(tf_target_map["evidence_score"].median()), 4),
        "n_activator_edges": int(
            tf_target_map[tf_target_map["tf_role"] == "activator"].shape[0]
        ),
        "n_repressor_edges": int(
            tf_target_map[tf_target_map["tf_role"] == "repressor"].shape[0]
        ),
        "n_dual_edges": int(
            tf_target_map[tf_target_map["tf_role"] == "dual"].shape[0]
        ),
    }


# ---------------------------------------------------------------------------
# GO term label helper
# ---------------------------------------------------------------------------

def go_term_label(go_id: str) -> str:
    """Return human-readable label for known GO terms, else return go_id."""
    all_labels = {**BINDING_GO_TERMS, **REGULATION_GO_TERMS}
    return all_labels.get(go_id, go_id)
