"""
ortholog_finder.py
Lightweight k-mer–based ortholog identification for cross-species TFBS analysis.

For each S. cerevisiae target gene, finds the best-matching protein in a
non-cerevisiae species using amino-acid 5-mer vote counting.  This anchors
TFBS conservation checks to the promoter of the true ortholog rather than
scanning every promoter in the genome.

Algorithm
---------
1. Build an inverted k-mer index for the species proteome: each unique k-mer
   maps to the set of gene IDs that contain it (presence only, not frequency).
2. For each S. cerevisiae query protein, count how many of its unique k-mers
   appear in each species gene (vote counting).
3. The species gene with the highest vote count is the "best hit".
4. The normalised score  (votes / n_unique_query_kmers)  must reach
   MIN_KMER_SCORE to accept the hit as an ortholog.  Below this threshold
   the assignment is rejected (gene absent / too divergent).

ID normalisation
----------------
BRAKER GFF3 gene IDs (e.g. g000086) and the matching PEP file transcript IDs
(e.g. g000086.t1) differ by a ".tN" suffix.  load_species_proteins() strips
this suffix so that PEP IDs match GFF3 gene IDs.  When multiple transcripts
map to the same gene, the longest protein is kept.

Limitations
-----------
- Single-direction best hit only (not reciprocal best hit).  For paralogue-rich
  TF families in divergent species the assignment may pick the wrong paralogue.
- For very deep outgroups (e.g. Yarrowia lipolytica vs S. cerevisiae) even
  the best k-mer hit may be low-confidence.  MIN_KMER_SCORE filters out the
  worst cases but does not guarantee true orthology.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

KMER_K: int = 5
MIN_KMER_SCORE: float = 0.05   # fraction of query k-mers that must match best hit


# ---------------------------------------------------------------------------
# Protein FASTA loader with ID normalisation
# ---------------------------------------------------------------------------

def _strip_transcript_suffix(gene_id: str) -> str:
    """g000086.t1 → g000086; g000086.m1 → g000086; YBR020W-A unchanged."""
    return re.sub(r"\.[tTmM]\d+$", "", gene_id)


def load_species_proteins(pep_path: Path) -> dict[str, str]:
    """
    Load a species peptide FASTA, normalising IDs by stripping BRAKER
    transcript suffixes (.t1, .t2, …) to get gene-level IDs.

    When multiple transcripts share the same gene ID, the longest protein
    sequence is kept.

    Returns {gene_id: aa_sequence}.
    """
    seqs: dict[str, str] = {}
    current: Optional[str] = None
    chunks: list[str] = []

    with open(pep_path, encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current is not None:
                    seq = "".join(chunks)
                    if current not in seqs or len(seq) > len(seqs[current]):
                        seqs[current] = seq
                raw_id = line[1:].split()[0]
                current = _strip_transcript_suffix(raw_id)
                chunks = []
            elif current is not None:
                chunks.append(line)

    if current is not None:
        seq = "".join(chunks)
        if current not in seqs or len(seq) > len(seqs[current]):
            seqs[current] = seq

    return seqs


# ---------------------------------------------------------------------------
# k-mer index and vote-counting search
# ---------------------------------------------------------------------------

def _build_kmer_index(
    pep_seqs: dict[str, str],
    k: int = KMER_K,
) -> dict[str, list[str]]:
    """
    Inverted k-mer index: {k-mer → [gene_id, ...]}.

    Each gene contributes each unique k-mer at most once, so the vote count
    from a query reflects shared unique sequence rather than repetition.
    """
    index: dict[str, list[str]] = defaultdict(list)
    for gene_id, seq in pep_seqs.items():
        seen: set[str] = set()
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            if kmer not in seen:
                index[kmer].append(gene_id)
                seen.add(kmer)
    return dict(index)


def _find_best_hit(
    query_seq: str,
    sp_index: dict[str, list[str]],
    k: int = KMER_K,
) -> tuple[Optional[str], float]:
    """
    Vote-count best matching species gene for query_seq.

    score = votes / n_unique_query_kmers  (∈ [0, 1]).
    Returns (None, 0.0) when query has no k-mers or no votes.
    """
    query_kmers: set[str] = set()
    for i in range(len(query_seq) - k + 1):
        query_kmers.add(query_seq[i : i + k])

    if not query_kmers:
        return None, 0.0

    votes: dict[str, int] = defaultdict(int)
    for kmer in query_kmers:
        for gid in sp_index.get(kmer, []):
            votes[gid] += 1

    if not votes:
        return None, 0.0

    best = max(votes, key=votes.__getitem__)
    return best, votes[best] / len(query_kmers)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_orthologs(
    sc_seqs: dict[str, str],
    sp_seqs: dict[str, str],
    query_ids: set[str],
    k: int = KMER_K,
    min_score: float = MIN_KMER_SCORE,
) -> dict[str, Optional[str]]:
    """
    For each S. cerevisiae gene in query_ids, find its best-matching protein
    in sp_seqs via k-mer vote counting.

    Parameters
    ----------
    sc_seqs   : {sc_gene_id → protein_seq} — S. cerevisiae proteome (or subset).
    sp_seqs   : {sp_gene_id → protein_seq} — target species proteome.
    query_ids : Subset of sc_seqs keys to resolve (e.g. TF target gene IDs).
    k         : k-mer length in amino acids (default 5).
    min_score : Minimum normalised score to accept a hit.  Hits below this
                threshold return None (ortholog absent or too divergent).

    Returns
    -------
    {sc_gene_id → sp_gene_id | None}
    None means no confident ortholog was found in this species.
    """
    if not sp_seqs:
        return {gid: None for gid in query_ids}

    sp_index = _build_kmer_index(sp_seqs, k)

    result: dict[str, Optional[str]] = {}
    for sc_gid in query_ids:
        seq = sc_seqs.get(sc_gid)
        if not seq:
            result[sc_gid] = None
            continue
        best, score = _find_best_hit(seq, sp_index, k)
        result[sc_gid] = best if (best is not None and score >= min_score) else None

    return result
