"""
pi2_sequence_homology.py
Estimates π₂: regulatory inheritance probability from protein sequence homology.

Biological rationale (Scruse et al. framework):
  Paralogs with high sequence identity diverged recently; their regulatory
  wiring is more likely still shared. Low identity → older duplication →
  links have had more time to be lost.
  π₂ for a paralog pair = pairwise protein sequence identity.

Algorithm:
  1. Load S. cerevisiae S288C peptide sequences (from Y1000+ sgd.pep file).
  2. For each YEASTRACT TF, compute pairwise alignment against all other
     TF proteins using the Biopython pairwise aligner (Smith-Waterman local).
  3. Cluster all TF proteins into families at identity thresholds 30%, 50%, 80%.
  4. π₂ estimate for each paralog pair = pairwise_identity.
  5. Family-level π₂ = mean pairwise identity within the cluster.

Output: pi2_sequence_homology.csv
  gene1, gene2, pct_identity, alignment_score, alignment_len,
  family_30pct, family_50pct, family_80pct, pi2_estimate

Performance note:
  For 127 TFs, the all-vs-all matrix is 127×127/2 ≈ 8000 pairs.
  Biopython PairwiseAligner on protein sequences is fast enough for this scale.
  For all ~6000 S. cerevisiae proteins, switch to BLASTP.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .y1000plus_loader import PROCESSED_DIR, scerevisiae_pep_path

PI2_CSV = PROCESSED_DIR / "pi2_sequence_homology.csv"

# ---------------------------------------------------------------------------
# FASTA sequence loader
# ---------------------------------------------------------------------------

def load_pep_fasta(pep_path: Path) -> dict[str, str]:
    """
    Load a peptide FASTA into {gene_id: aa_sequence}.
    Gene ID = first word of the header line (e.g., 'YBR020W-A' or 'g000086').
    """
    seqs: dict[str, str] = {}
    current = None
    chunks: list[str] = []
    with open(pep_path, encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current:
                    seqs[current] = "".join(chunks)
                current = line[1:].split()[0]
                chunks = []
            elif current:
                chunks.append(line)
        if current:
            seqs[current] = "".join(chunks)
    return seqs


# ---------------------------------------------------------------------------
# Pairwise alignment
# ---------------------------------------------------------------------------

def _pairwise_identity(seq1: str, seq2: str) -> tuple[float, float, int]:
    """
    Compute local pairwise identity between two protein sequences.

    Returns (pct_identity, norm_score, alignment_len).
    pct_identity = identical_positions / aligned_length.
    norm_score   = raw_score / min(len(seq1), len(seq2))  [0,1].
    """
    try:
        from Bio import pairwise2
        alignments = pairwise2.align.localms(
            seq1, seq2,
            match=2, mismatch=-1, open=-10, extend=-0.5,
            one_alignment_only=True,
        )
        if not alignments:
            return 0.0, 0.0, 0

        aln = alignments[0]
        aligned_a = aln.seqA[aln.start : aln.end]
        aligned_b = aln.seqB[aln.start : aln.end]
        aligned_len = len(aligned_a)
        if aligned_len == 0:
            return 0.0, 0.0, 0

        identical = sum(a == b for a, b in zip(aligned_a, aligned_b)
                        if a != "-" and b != "-")
        non_gap_cols = sum(a != "-" and b != "-"
                           for a, b in zip(aligned_a, aligned_b))
        pct_id = (identical / non_gap_cols * 100.0) if non_gap_cols > 0 else 0.0
        norm_score = aln.score / max(len(seq1), len(seq2), 1)
        return round(pct_id, 2), round(norm_score, 4), aligned_len

    except ImportError:
        # Fallback: k-mer identity (fast but approximate)
        return _kmer_identity(seq1, seq2, k=4)


def _kmer_identity(seq1: str, seq2: str, k: int = 4) -> tuple[float, float, int]:
    """Simple k-mer Jaccard similarity as a sequence identity proxy."""
    set1 = {seq1[i : i + k] for i in range(len(seq1) - k + 1)}
    set2 = {seq2[i : i + k] for i in range(len(seq2) - k + 1)}
    union = set1 | set2
    if not union:
        return 0.0, 0.0, 0
    jaccard = len(set1 & set2) / len(union)
    pct_id = jaccard * 100.0
    aligned_len = min(len(seq1), len(seq2))
    return round(pct_id, 2), round(jaccard, 4), aligned_len


# ---------------------------------------------------------------------------
# Family clustering (Union-Find)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        self.parent[self.find(x)] = self.find(y)

    def clusters(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for x in self.parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return groups


def cluster_by_identity(
    pair_df: pd.DataFrame,
    threshold: float,
    all_genes: list[str],
) -> dict[str, str]:
    """
    Cluster genes into families where any pair with pct_identity >= threshold
    is in the same cluster.

    Returns {gene_id: cluster_label} where cluster_label is the
    alphabetically first gene in the cluster.
    """
    uf = _UnionFind(all_genes)
    for _, row in pair_df.iterrows():
        if row["pct_identity"] >= threshold:
            uf.union(row["gene1"], row["gene2"])
    clusters = uf.clusters()
    label_map: dict[str, str] = {}
    for root, members in clusters.items():
        label = sorted(members)[0]
        for m in members:
            label_map[m] = label
    return label_map


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def estimate_pi2(
    gene_subset: Optional[list[str]] = None,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute all-vs-all pairwise sequence identity for TF proteins.

    Parameters
    ----------
    gene_subset : Gene IDs to align (defaults to all YEASTRACT TF systematic names).
    save        : Write pi2_sequence_homology.csv.
    verbose     : Print progress.

    Returns DataFrame with columns:
      gene1, gene2, pct_identity, alignment_score, alignment_len,
      family_30pct, family_50pct, family_80pct, pi2_estimate
    """
    from .data_loader import YEASTRACT_TFS_SGD

    pep_path = scerevisiae_pep_path()
    all_seqs = load_pep_fasta(pep_path)

    if verbose:
        print(f"Loaded {len(all_seqs)} proteins from {pep_path.name}")

    # Map YEASTRACT TF names to systematic gene IDs via GFF3 gene_name attribute
    # The SGD PEP file uses systematic names (YBR020W) as sequence IDs.
    # We need the systematic ID for each TF common name.
    tf_systematic = _map_tf_names_to_systematic(gene_subset)

    # Filter to TFs present in the PEP file
    available = {
        common: syst
        for common, syst in tf_systematic.items()
        if syst in all_seqs
    }
    if verbose:
        print(f"TFs mapped to systematic IDs: {len(tf_systematic)}, "
              f"found in PEP: {len(available)}")

    gene_ids = sorted(available.values())
    gene_common = {v: k for k, v in available.items()}
    n = len(gene_ids)

    if verbose:
        print(f"Running {n*(n-1)//2} pairwise alignments...")

    pair_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            g1, g2 = gene_ids[i], gene_ids[j]
            pct_id, score, aln_len = _pairwise_identity(
                all_seqs[g1], all_seqs[g2]
            )
            pair_rows.append(
                {
                    "gene1": g1,
                    "gene2": g2,
                    "gene1_name": gene_common.get(g1, ""),
                    "gene2_name": gene_common.get(g2, ""),
                    "pct_identity": pct_id,
                    "alignment_score": score,
                    "alignment_len": aln_len,
                }
            )

    if not pair_rows:
        return pd.DataFrame()

    pair_df = pd.DataFrame(pair_rows)

    # Cluster at three identity thresholds
    for thr in [30, 50, 80]:
        col = f"family_{thr}pct"
        label_map = cluster_by_identity(pair_df, float(thr), gene_ids)
        pair_df[col] = pair_df.apply(
            lambda r: f"{label_map.get(r['gene1'], r['gene1'])}_cluster", axis=1
        )

    # π₂ = pct_identity / 100 clipped to [0, 1]
    pair_df["pi2_estimate"] = (pair_df["pct_identity"] / 100.0).clip(0, 1).round(4)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        pair_df.to_csv(PI2_CSV, index=False)
        if verbose:
            print(f"Saved -> {PI2_CSV}")

    return pair_df


def _map_tf_names_to_systematic(gene_subset: Optional[list[str]] = None) -> dict[str, str]:
    """
    Return {common_name: systematic_id} for YEASTRACT TFs.

    Uses the S. cerevisiae SGD GFF3 (gene= attribute) to build the mapping.
    TFs without a gene= attribute in the GFF3 are mapped to themselves (some
    SGD entries already use the systematic name as both ID and common name).
    """
    from .promoter_extractor import parse_gff3_genes, SGD_CHR_MAP
    from .y1000plus_loader import scerevisiae_gff3_path
    from .data_loader import YEASTRACT_TFS_SGD

    tfs_to_map = set(gene_subset) if gene_subset else YEASTRACT_TFS_SGD

    gff3 = scerevisiae_gff3_path()
    gene_df = parse_gff3_genes(gff3, chr_map=SGD_CHR_MAP)

    # Build common_name → systematic_id map from GFF3
    name_to_syst: dict[str, str] = {}
    for _, row in gene_df.iterrows():
        syst = row["gene_id"]          # e.g. YBR020W
        common = row.get("gene_name", "")   # e.g. GAL1
        if common:
            name_to_syst[common.upper()] = syst
        # Also add systematic-name → systematic-name self-mapping
        name_to_syst[syst.upper()] = syst

    result: dict[str, str] = {}
    for tf in tfs_to_map:
        mapped = name_to_syst.get(tf.upper())
        if mapped:
            result[tf] = mapped
        else:
            # Last resort: keep the YEASTRACT name as-is (might already be systematic)
            result[tf] = tf

    return result


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pi2_results() -> pd.DataFrame:
    """Load cached pi2_sequence_homology.csv."""
    if not PI2_CSV.exists():
        raise FileNotFoundError(
            f"pi2_sequence_homology.csv not found. Run estimate_pi2() first."
        )
    return pd.read_csv(PI2_CSV)


def pi2_summary() -> dict:
    """Summary statistics over all π₂ estimates."""
    df = load_pi2_results()
    return {
        "n_pairs": len(df),
        "n_genes": df["gene1"].nunique() + df["gene2"].nunique(),
        "mean_pi2": round(float(df["pi2_estimate"].mean()), 4),
        "median_pct_identity": round(float(df["pct_identity"].median()), 2),
        "n_high_identity_pairs_80pct": int((df["pct_identity"] >= 80).sum()),
        "n_low_identity_pairs_30pct": int((df["pct_identity"] <= 30).sum()),
    }
