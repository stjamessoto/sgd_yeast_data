"""
promoter_extractor.py
Extract upstream promoter sequences for TFBS scanning.

For each gene, the promoter is defined as the 1000bp immediately upstream
of the annotated transcription start site (TSS proxy = gene start coordinate).

Strand handling:
  + strand gene: TSS = GFF3 'start'; promoter = [start-1000, start-1] (1-based)
  - strand gene: TSS = GFF3 'end';   promoter = [end+1,    end+1000] (1-based),
                 then reverse-complemented so the sequence reads 5'->3' toward TSS.

Chromosome clamping:
  Promoter length is truncated to chromosome boundary rather than padded.
  Genes within 1000bp of the chromosome start/end still get extracted; the
  'promoter_len' column records the actual length.

GFF3 format notes:
  SGD GFF3 (saccharomyces_cerevisiae.sgd.gff3): uses chrI/chrII/... names.
  The genome FASTA uses NC_001133/NC_001134/... NCBI accessions.
  SGD_CHR_MAP provides the name translation.

  BRAKER GFF3 (*.final.gff3): chromosome names in GFF3 match the FASTA
  headers directly, so no mapping is needed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Chromosome name mapping: SGD GFF3 roman-numeral names → NCBI accessions
# in the Y1000+ genome FASTA for S. cerevisiae S288C.
# ---------------------------------------------------------------------------

SGD_CHR_MAP: dict[str, str] = {
    "chrI":    "NC_001133",
    "chrII":   "NC_001134",
    "chrIII":  "NC_001135",
    "chrIV":   "NC_001136",
    "chrV":    "NC_001137",
    "chrVI":   "NC_001138",
    "chrVII":  "NC_001139",
    "chrVIII": "NC_001140",
    "chrIX":   "NC_001141",
    "chrX":    "NC_001142",
    "chrXI":   "NC_001143",
    "chrXII":  "NC_001144",
    "chrXIII": "NC_001145",
    "chrXIV":  "NC_001146",
    "chrXV":   "NC_001147",
    "chrXVI":  "NC_001148",
    "chrmt":   "NC_001224",
}

_RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


# ---------------------------------------------------------------------------
# FASTA loader
# ---------------------------------------------------------------------------

def load_fasta(fasta_path: Path, uppercase: bool = True) -> dict[str, str]:
    """
    Load a FASTA file into a dict {header_word: sequence}.
    The key is the first word of the header line (after '>'), e.g. 'NC_001133'.
    """
    sequences: dict[str, str] = {}
    current_key: Optional[str] = None
    chunks: list[str] = []

    with open(fasta_path, encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_key is not None:
                    sequences[current_key] = "".join(chunks)
                current_key = line[1:].split()[0]
                chunks = []
            elif current_key is not None:
                chunks.append(line.upper() if uppercase else line)
        if current_key is not None:
            sequences[current_key] = "".join(chunks)

    return sequences


# ---------------------------------------------------------------------------
# GFF3 parser
# ---------------------------------------------------------------------------

def parse_gff3_genes(
    gff3_path: Path,
    chr_map: Optional[dict[str, str]] = None,
    feature_types: tuple[str, ...] = ("gene",),
) -> pd.DataFrame:
    """
    Parse a GFF3 file and return a DataFrame of gene features.

    Columns: gene_id, seqname, start (1-based), end (1-based), strand
    Only rows whose 'type' column is in feature_types are kept.
    chr_map: optional dict to rename seqname values (e.g. SGD_CHR_MAP).
    """
    records = []
    with open(gff3_path, encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqname, source, ftype, start, end, score, strand, phase, attrs = parts
            if ftype not in feature_types:
                continue

            # Extract gene ID and optional common name from attributes
            gene_id = _parse_attr(attrs, "ID") or _parse_attr(attrs, "Name") or "unknown"
            gene_id = re.sub(r"_(mRNA|CDS|exon|id\d+)$", "", gene_id)
            gene_name = _parse_attr(attrs, "gene") or ""

            mapped_seq = (chr_map or {}).get(seqname, seqname)
            records.append(
                {
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "seqname": mapped_seq,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No {feature_types} features found in {gff3_path}")
    return df.drop_duplicates("gene_id").reset_index(drop=True)


def _parse_attr(attrs: str, key: str) -> Optional[str]:
    """Extract value for key= from a GFF3 attributes string."""
    for token in attrs.split(";"):
        token = token.strip()
        if token.startswith(key + "="):
            return token[len(key) + 1 :]
    return None


# ---------------------------------------------------------------------------
# Promoter extraction
# ---------------------------------------------------------------------------

def extract_promoters(
    gff3_path: Path,
    fasta_path: Path,
    upstream_bp: int = 1000,
    chr_map: Optional[dict[str, str]] = None,
    gene_subset: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Extract upstream promoter sequences for all genes in a GFF3 file.

    Parameters
    ----------
    gff3_path    : Path to GFF3 annotation file.
    fasta_path   : Path to genome FASTA.
    upstream_bp  : Bases upstream of TSS to extract (default 1000).
    chr_map      : Optional seqname→FASTA-key mapping (use SGD_CHR_MAP for
                   the S. cerevisiae reference; None for BRAKER assemblies).
    gene_subset  : If provided, only extract promoters for these gene IDs.

    Returns
    -------
    DataFrame with columns:
      gene_id, seqname, gene_start, gene_end, strand,
      promoter_start, promoter_end, promoter_len, sequence
    """
    genome = load_fasta(fasta_path)
    genes = parse_gff3_genes(gff3_path, chr_map=chr_map)

    if gene_subset is not None:
        gene_subset_up = {g.upper() for g in gene_subset}
        genes = genes[genes["gene_id"].str.upper().isin(gene_subset_up)]

    rows = []
    for _, g in genes.iterrows():
        chrom_seq = genome.get(g["seqname"])
        if chrom_seq is None:
            continue

        chrom_len = len(chrom_seq)
        start = int(g["start"])   # 1-based inclusive
        end = int(g["end"])       # 1-based inclusive
        strand = g["strand"]

        if strand == "+":
            # TSS at 'start'; promoter is [start-upstream_bp, start-1] (1-based)
            p_start_1 = max(1, start - upstream_bp)
            p_end_1 = start - 1
            if p_end_1 < p_start_1:
                continue  # gene at very start of chromosome
            # Python slice (0-based): seq[p_start_1-1 : p_end_1]
            seq = chrom_seq[p_start_1 - 1 : p_end_1]
        else:
            # TSS at 'end'; promoter is [end+1, end+upstream_bp] (1-based)
            p_start_1 = end + 1
            p_end_1 = min(chrom_len, end + upstream_bp)
            if p_start_1 > p_end_1:
                continue
            # Python slice: seq[p_start_1-1 : p_end_1]
            raw = chrom_seq[p_start_1 - 1 : p_end_1]
            seq = _reverse_complement(raw)
            # After RC, the sequence reads 5'->3' toward the TSS

        if not seq:
            continue

        rows.append(
            {
                "gene_id": g["gene_id"],
                "gene_name": g.get("gene_name", ""),
                "seqname": g["seqname"],
                "gene_start": start,
                "gene_end": end,
                "strand": strand,
                "promoter_start": p_start_1,
                "promoter_end": p_end_1,
                "promoter_len": len(seq),
                "sequence": seq,
            }
        )

    _COLS = ["gene_id", "gene_name", "seqname", "gene_start", "gene_end",
             "strand", "promoter_start", "promoter_end", "promoter_len", "sequence"]
    if not rows:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame(rows, columns=_COLS).reset_index(drop=True)


def write_promoter_fasta(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write promoter sequences to a FASTA file.
    Header format: >{gene_id}  strand={strand}  len={promoter_len}
    """
    with open(out_path, "w") as fh:
        for _, row in df.iterrows():
            fh.write(
                f">{row['gene_id']}  strand={row['strand']}  "
                f"len={row['promoter_len']}\n"
            )
            seq = row["sequence"]
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")


# ---------------------------------------------------------------------------
# Convenience: S. cerevisiae S288C reference
# ---------------------------------------------------------------------------

def extract_scerevisiae_promoters(
    upstream_bp: int = 1000,
    gene_subset: Optional[set[str]] = None,
    save_fasta: bool = True,
) -> pd.DataFrame:
    """
    Extract 1000bp upstream promoters for S. cerevisiae S288C (SGD reference).

    Uses the extracted files in y1000plus_data/processed/ (built by y1000plus_loader).
    Applies SGD_CHR_MAP to reconcile chrI/chrII names (GFF3) with NC_001133/...
    names in the genome FASTA.

    If save_fasta=True, writes y1000plus_data/processed/promoters_Scerevisiae_S288C.fasta.
    """
    from .y1000plus_loader import scerevisiae_gff3_path, scerevisiae_genome_path, PROCESSED_DIR

    gff3 = scerevisiae_gff3_path()
    fasta = scerevisiae_genome_path()

    df = extract_promoters(
        gff3_path=gff3,
        fasta_path=fasta,
        upstream_bp=upstream_bp,
        chr_map=SGD_CHR_MAP,
        gene_subset=gene_subset,
    )

    if save_fasta:
        out = PROCESSED_DIR / "promoters_Scerevisiae_S288C.fasta"
        write_promoter_fasta(df, out)

    return df


# ---------------------------------------------------------------------------
# Validation: GAL4 UAS_GAL motif check
# ---------------------------------------------------------------------------

# UAS_GAL consensus: CGG(N11)CCG (Giniger & Ptashne 1988)
# Allow both orientations. We use a simple regex.
UAS_GAL_PATTERN = re.compile(r"CGG.{9,13}CCG|CGG.{9,13}CCG", re.IGNORECASE)


def validate_gal4_targets(promoter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check whether known GAL4 target gene promoters contain the UAS_GAL motif.

    Returns a summary DataFrame with columns:
      gene_id, has_uas_gal, n_hits, promoter_len

    GAL4 canonical targets: GAL1, GAL2, GAL7, GAL10, GAL80, MEL1.
    The UAS_GAL consensus is CGG(N11)CCG ± 2bp variation.
    """
    gal_targets = {"GAL1", "GAL2", "GAL7", "GAL10", "GAL80", "MEL1"}
    # Match on common gene_name first; fall back to gene_id
    name_col = "gene_name" if "gene_name" in promoter_df.columns else "gene_id"
    subset = promoter_df[
        promoter_df[name_col].str.upper().isin(gal_targets)
    ].copy()
    if subset.empty:
        # Try gene_id too (for non-SGD assemblies using common names)
        subset = promoter_df[
            promoter_df["gene_id"].str.upper().isin(gal_targets)
        ].copy()

    rows = []
    for _, r in subset.iterrows():
        hits = UAS_GAL_PATTERN.findall(r["sequence"])
        rows.append(
            {
                "gene_id": r["gene_id"],
                "gene_name": r.get("gene_name", ""),
                "has_uas_gal": len(hits) > 0,
                "n_hits": len(hits),
                "promoter_len": r["promoter_len"],
                "first_hit": hits[0] if hits else "",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["gene_id", "gene_name", "has_uas_gal", "n_hits", "promoter_len", "first_hit"]
        )
    return pd.DataFrame(rows)
