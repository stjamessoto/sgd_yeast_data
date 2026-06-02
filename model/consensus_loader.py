"""
consensus_loader.py
Loads the YEASTRACT Transcription Factor <-> Consensus list from:
  https://www.yeastract.com/consensuslist.php

Data provides the actual DNA consensus binding sequences for each TF —
i.e., the sequences that TF binding sites recognise in target gene promoters.

Integration with the Scruse et al. model:
  Consensus sequence properties inform the inheritance probability πᵢ:
  - A TF with many distinct consensus variants is a promiscuous binder;
    its regulatory links are more likely to be maintained after duplication
    even when the target promoter sequence drifts → higher πᵢ.
  - A TF with a short, ambiguity-rich consensus (many IUPAC codes) binds
    flexibly → links survive sequence divergence → higher πᵢ.
  - A TF with a single long, specific consensus requires an exact match →
    links are fragile to sequence change → lower πᵢ.

This is the genomic equivalent of the paper's MITE application (Section 9.2):
greater sequence flexibility ⇒ regulatory links more likely inherited.

Name normalisation:
  YEASTRACT uses protein notation: Gcn4p, Ino2p, …
  SGD uses gene notation:          GCN4,  INO2,  …
  Conversion: strip trailing 'p', uppercase → tf_sgd column.
"""

from __future__ import annotations

import re
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Module-level cache (avoids lru_cache issues with mutable DataFrames)
_CONSENSUS_DF: Optional[pd.DataFrame] = None
_STATS_DF: Optional[pd.DataFrame] = None

_HERE = Path(__file__).parent.parent
DATA_DIR = _HERE / "sgd_yeast_data"
CONSENSUS_CSV = DATA_DIR / "yeastract_consensus.csv"
YEASTRACT_URL = "https://www.yeastract.com/consensuslist.php"

# IUPAC ambiguity codes (any character that is not a plain A/C/G/T)
IUPAC_AMBIGUOUS = set("WSMKRYBDHVN")

# Complete dataset scraped 2026-06-02 — used as fallback when network is unavailable
_HARDCODED_DATA = [
    ("Smp1p","ACTACTAWWWWTAG"),
    ("Ert1p","CTCCGGG"),("Ert1p","GCCCCGCGGG"),("Ert1p","GTCCGCA"),
    ("Ert1p","TCCGGTT"),("Ert1p","ACTCCGA"),("Ert1p","ACTCCGAG"),
    ("Ert1p","ACTCCGATT"),("Ert1p","CCCCGCGG"),("Ert1p","CCGAGAA"),
    ("Ert1p","CCGCTCG"),("Ert1p","CCGGGCT"),("Ert1p","CCTCCGG"),
    ("Ert1p","CTCCGCC"),("Ert1p","CTCCGCG"),("Ert1p","CTCCGCGGAA"),
    ("Ert1p","CTCCGGGC"),("Ert1p","GCCCCGC"),("Ert1p","GCCCCGCGG"),
    ("Ert1p","GCCGGGT"),("Ert1p","TCCGATA"),("Ert1p","TCCGATG"),
    ("Ert1p","TCCGCGGA"),("Ert1p","TCCTCCGAGAAA"),("Ert1p","TCCTCCGGTT"),
    ("Ert1p","TCTCCGGG"),("Ert1p","TTCCGCGGAA"),("Ert1p","TTCCGCGTCCGCA"),
    ("Ert1p","TTCCGTG"),("Ert1p","TTCCGTGG"),("Ert1p","TTCTCCGCC"),
    ("Ert1p","TTTCTCCGG"),("Ert1p","TTTCTCNG"),("Ert1p","ACTCCGATA"),
    ("Ert1p","CTCCGCGGG"),("Ert1p","TCCGCTC"),("Ert1p","TCCGGTG"),
    ("Ert1p","TCCNCGGA"),("Ert1p","TCCTCCG"),("Ert1p","TCCTCCGG"),
    ("Ert1p","TTCCGAGAA"),("Ert1p","TTCCGCGG"),("Ert1p","TTCCGCGGA"),
    ("Ert1p","TTCCGCGGG"),("Ert1p","TTCCTCCGA"),("Ert1p","TTCCTCCGG"),
    ("Ert1p","TTNTCCGG"),("Ert1p","TTNTCCGGGCT"),("Ert1p","TTNTCCGGTG"),
    ("Ert1p","TTTCTCCGAGG"),("Ert1p","TTTCTCCGCGGAA"),("Ert1p","TCCGGGCT"),
    ("Kar4p","YAANNCAAANNCNGNYT"),("Kar4p","CTAANNCAAANNCNGNYT"),
    ("Kar4p","TTAANNCAAANNCNGNYT"),
    ("Hcm1p","WAAYAAACAAW"),
    ("Rds1p","BCGGCCG"),
    ("Uga3p","SGCGGNWTTT"),
    ("Pho2p","WTAWTW"),
    ("Mbp1p","ACGCGT"),
    ("Rpn4p","GGTGGCAAA"),
    ("Lys14p","TCCRNYGGA"),
    ("Nrg1p","CCCCT"),("Nrg1p","CCCTC"),("Nrg1p","GGACCCT"),
    ("Gis1p","TWAGGGAT"),("Gis1p","AGGGG"),
    ("Ino2p","ATGTGAAAT"),("Ino2p","CATGTGAAAT"),("Ino2p","WYTTCAYRTGS"),
    ("Ino2p","CACATGC"),("Ino2p","ACACATGCA"),("Ino2p","ACACATGCTG"),
    ("Ino2p","ACATGCC"),("Ino2p","ACATGCTG"),("Ino2p","AGCACATGCAA"),
    ("Ino2p","AGCACATGCCA"),("Ino2p","AGCACATGCG"),("Ino2p","AGCACATGCGC"),
    ("Ino2p","AGCACATGCTA"),("Ino2p","AGCACATGCTT"),("Ino2p","AGCACATGGG"),
    ("Ino2p","AGCACGTGC"),("Ino2p","ATCACATG"),("Ino2p","ATCACATGCA"),
    ("Ino2p","ATCACATGCTG"),("Ino2p","CACGTTC"),("Ino2p","CATATGCAA"),
    ("Ino2p","CCACGTGC"),("Ino2p","CCACGTGCT"),("Ino2p","CCTCATGC"),
    ("Ino2p","CGCACANG"),("Ino2p","CGCACATG"),("Ino2p","CGCACNTGG"),
    ("Ino2p","CTCATGCCG"),("Ino2p","GCACGTTC"),("Ino2p","GCNCATGC"),
    ("Ino2p","GCTCATGCT"),("Ino2p","GGCACATGCAA"),("Ino2p","GGCACATGG"),
    ("Ino2p","GGCATATGCT"),("Ino2p","TCACATGCA"),("Ino2p","TCACATGCAA"),
    ("Ino2p","TCACATGCCA"),("Ino2p","TCACGTGCT"),("Ino2p","TCCACATGCAC"),
    ("Ino2p","TCCACATGCCA"),("Ino2p","TCCACATGCTT"),("Ino2p","TCCACATGGG"),
    ("Ino2p","TGCACATGCGC"),("Ino2p","TGCACATGCTA"),("Ino2p","TGCACATGCTG"),
    ("Ino2p","TGCACGTGC"),("Ino2p","TGCATATGCAA"),("Ino2p","TTCACAAGC"),
    ("Ino2p","TTCACAAGCAG"),("Ino2p","TTCACATG"),("Ino2p","TTCACATGCA"),
    ("Ino2p","TTCACATGCAC"),("Ino2p","TTCACATGCC"),("Ino2p","TTCACATGCCA"),
    ("Ino2p","TTCACATGCCG"),("Ino2p","TTCACATGCG"),("Ino2p","TTCACATGCGC"),
    ("Ino2p","TTCACATGCT"),("Ino2p","TTCACATGCTA"),("Ino2p","TTCACATGCTG"),
    ("Ino2p","TTCACATGGA"),("Ino2p","TTCACATGGC"),("Ino2p","TTCACATGNT"),
    ("Ino2p","TTCACATNC"),("Ino2p","TTCACATNCA"),("Ino2p","TTCACGTGC"),
    ("Ino2p","TTCACGTGCTG"),("Ino2p","TTCATATGC"),("Ino2p","TTCATATGCA"),
    ("Ino2p","TTCATATGCAA"),("Ino2p","TTCATATGCC"),("Ino2p","TTCATATGCTA"),
    ("Ino2p","TTCNCGTG"),("Ino2p","TTCATATGCT"),("Ino2p","TTCATATGCGG"),
    ("Ino2p","TTCATATG"),("Ino2p","TTCACGTG"),("Ino2p","TTCACATGG"),
    ("Ino2p","TGCATATGCT"),("Ino2p","TGCATATGCAG"),("Ino2p","TGCACGTTCT"),
    ("Ino2p","TGCACGTTC"),("Ino2p","TGCACGTGCT"),("Ino2p","TGCACGT"),
    ("Ino2p","TGCACATGCA"),("Ino2p","TCTCATGCT"),("Ino2p","TCCACATGG"),
    ("Ino2p","TCCACATGCG"),("Ino2p","TCCACATG"),("Ino2p","TCATATGCT"),
    ("Ino2p","TCATATGCGG"),("Ino2p","TCATATGCAA"),("Ino2p","TCACGTGCTG"),
    ("Ino2p","TCACGTGCGG"),("Ino2p","TCACGTGC"),("Ino2p","TCACGTG"),
    ("Ino2p","TCACATNC"),("Ino2p","TCACATGGG"),("Ino2p","TCACATGGA"),
    ("Ino2p","TCACATGG"),("Ino2p","TCACATGCTT"),("Ino2p","TCACATG"),
    ("Ino2p","GGNACNTGC"),("Ino2p","GGNACGTGCT"),("Ino2p","GGCATATGCTA"),
    ("Ino2p","GGCATATGCAG"),("Ino2p","GGCANATG"),("Ino2p","GGCACATGGG"),
    ("Ino2p","GGCACATGCTG"),("Ino2p","GGCACATGCG"),("Ino2p","GGCACATG"),
    ("Ino2p","GCTCATGCCG"),("Ino2p","GCTCATGCC"),("Ino2p","GCTCATGC"),
    ("Ino2p","GCCACATGGG"),("Ino2p","GCCACATGCGG"),("Ino2p","GCCACATGCG"),
    ("Ino2p","GCCACATGCCG"),("Ino2p","GCCACATGCCA"),("Ino2p","GCCACATGCAA"),
    ("Ino2p","GCCACATG"),("Ino2p","GCATATGCTA"),("Ino2p","GCACGTTCT"),
    ("Ino2p","GCACGTTCC"),("Ino2p","GCACATGGG"),("Ino2p","GCACATGGC"),
    ("Ino2p","GCACATGGA"),("Ino2p","GCACANGC"),("Ino2p","CTCATGCT"),
    ("Ino2p","CTCATGCC"),("Ino2p","CTCATGC"),("Ino2p","CGCTCATGCCG"),
    ("Ino2p","CGCTCATGCC"),("Ino2p","CGCNCATG"),("Ino2p","CGCACGTGCTC"),
    ("Ino2p","CGCACGTGCGG"),("Ino2p","CGCACATGGC"),("Ino2p","CGCACATGGA"),
    ("Ino2p","CGCACATGCTT"),("Ino2p","CGCACATGCTA"),("Ino2p","CGCACATGCC"),
    ("Ino2p","CGCACANGC"),("Ino2p","CCTCATGCCG"),("Ino2p","CCTCATGCC"),
    ("Ino2p","CCACGTGCTG"),("Ino2p","CCACGTGCTC"),("Ino2p","CCACGTGCGG"),
    ("Ino2p","CCACATGG"),("Ino2p","CCACATGCTA"),("Ino2p","CATATGCGG"),
    ("Ino2p","CATATGCC"),("Ino2p","CACGTTCT"),("Ino2p","CACGTGCT"),
    ("Ino2p","CACGTGCGG"),("Ino2p","CACATGGG"),("Ino2p","CACATGGC"),
    ("Ino2p","CACANNCAC"),("Ino2p","CACAAGCAG"),("Ino2p","CACAAGCA"),
    ("Ino2p","CACAAGC"),("Ino2p","ATCACATGNT"),("Ino2p","ATCACATGGA"),
    ("Ino2p","ATCACATGG"),("Ino2p","ATCACATGCTT"),("Ino2p","ATCACATGCG"),
    ("Ino2p","ATCACATGCAA"),("Ino2p","ATATGCAA"),("Ino2p","ATATGCA"),
    ("Ino2p","ANATGCTA"),("Ino2p","ANATGCGG"),("Ino2p","AGCACNTG"),
    ("Ino2p","AGCACGTGCT"),("Ino2p","AGCACGTGCGG"),("Ino2p","AGCACATGGC"),
    ("Ino2p","AGCACATGGA"),("Ino2p","AGCACATGCCG"),("Ino2p","AGCACATGCA"),
    ("Ino2p","AGCACATG"),("Ino2p","ACNTGCGG"),("Ino2p","ACGTGCTG"),
    ("Ino2p","ACATGGC"),("Ino2p","ACATGCTA"),("Ino2p","ACATGCGG"),
    ("Ino2p","ACATGCGC"),("Ino2p","ACATGCG"),("Ino2p","ACATGCCG"),
    ("Ino2p","ACATGCAC"),("Ino2p","ACATGCAA"),("Ino2p","ACATGCA"),
    ("Ino2p","ACATATGCC"),("Ino2p","ACATATGCAG"),("Ino2p","ACATATGCA"),
    ("Ino2p","ACACAAGCAG"),("Ino2p","ACACAAGCA"),("Ino2p","ACACAAGC"),
    ("Ino2p","TTCACGTGCT"),("Ino2p","TCACATNCA"),("Ino2p","TCACATGCGG"),
    ("Ino2p","TTCATATGCAG"),("Ino2p","GCACATGG"),("Ino2p","CGCACATGCCG"),
    ("Ino2p","CCACATGCTG"),("Ino2p","CATATGCTA"),("Ino2p","ACGTGCT"),
    ("Swi5p","ACCAGC"),
    ("Ume6p","TAGCCGCCGA"),("Ume6p","TAGGCGGC"),("Ume6p","TCGGCGGCT"),
    ("Ume6p","TGGGTGGCTA"),
    ("Upc2p","TCGTATA"),("Upc2p","TCGTTYAG"),("Upc2p","TCGTWCC"),
    ("Upc2p","TATACGA"),("Upc2p","TAAACGA"),
    ("Adr1p","TTGGRGN{6,38}CYCCAA"),("Adr1p","TTGGRG"),
    ("Met32p","AAACTGTGG"),("Met32p","AAACTGTG"),("Met32p","CTGTGGC"),
    ("Sum1p","GNCRCAAAW"),
    ("Cad1p","TTACTAA"),
    ("Stp1p","CGGCTC"),("Stp1p","CGGCN{6}CGGC"),("Stp1p","RYRCGGCRC"),
    ("Gcn4p","TTACGTAA"),("Gcn4p","TGATTCA"),("Gcn4p","TGACTGA"),
    ("Gcn4p","TGACTMT"),("Gcn4p","RRTGACTC"),("Gcn4p","TGACTC"),
    ("Gcn4p","TGASTCA"),("Gcn4p","TTGCGCAA"),("Gcn4p","TTGCGTGA"),
    ("Gcn4p","GCACGTAG"),("Gcn4p","CACGTG"),("Gcn4p","AAATGACTCATCC"),
    ("Gcn4p","AAGTGAGTCACCT"),("Gcn4p","AAGTGAGTCACG"),("Gcn4p","AATGANTCACT"),
    ("Gcn4p","AGATGAGTCAGC"),("Gcn4p","AGGTGAGTCATGG"),("Gcn4p","AGTGAGTCAGC"),
    ("Gcn4p","AGTGAGTCATCC"),("Gcn4p","ATATGACTCATCC"),("Gcn4p","CATGAGTCACTA"),
    ("Gcn4p","CATTAGTCAC"),("Gcn4p","CGTGAGTCACTG"),("Gcn4p","CTGACTCATCA"),
    ("Gcn4p","CTGAGTCATAC"),("Gcn4p","CTGAGTCATCA"),("Gcn4p","GAATGAGTCAGC"),
    ("Gcn4p","GGTGAGTCATAC"),("Gcn4p","GGTTAGTCATAT"),("Gcn4p","GGTTAGTCATCT"),
    ("Gcn4p","GTGACTCATCC"),("Gcn4p","GTGANTCATAT"),("Gcn4p","TAAGAGTCATTCA"),
    ("Gcn4p","TCTGAGTCATCA"),("Gcn4p","TCTGAGTCATTC"),("Gcn4p","TGTTAGTCATCT"),
    ("Gcn4p","TTACTCATT"),("Gcn4p","TTATGAGTCACG"),("Gcn4p","TTGTGAGTCACTG"),
    ("Gcn4p","AAAAGAGTCACA"),("Gcn4p","AAAAGAGTCACCG"),("Gcn4p","AAAAGAGTCAGA"),
    ("Gcn4p","AAAAGAGTCATCC"),("Gcn4p","AAAAGAGTCATCT"),("Gcn4p","AAAAGAGTCATTA"),
    ("Gcn4p","AAAAGAGTCATTC"),("Gcn4p","AAAAGAGTCATTG"),("Gcn4p","AAAGAGTCACAG"),
    ("Gcn4p","AAAGAGTCACT"),("Gcn4p","AAAGAGTCACTA"),("Gcn4p","AAAGAGTCAG"),
    ("Gcn4p","AAAGAGTCATAA"),("Gcn4p","AAAGAGTCATCT"),("Gcn4p","AAATGAATCACA"),
    ("Gcn4p","AAATGAATCATC"),("Gcn4p","AAATGAGTC"),("Gcn4p","AAATGAGTCA"),
    ("Gcn4p","AAATGAGTCACG"),("Gcn4p","AAATGAGTNATT"),("Gcn4p","AAATGANTCA"),
    ("Gcn4p","AAATGANTCATAT"),("Gcn4p","AAATTAGTCA"),("Gcn4p","AAATTAGTCACT"),
    ("Gcn4p","AAATTAGTCATAT"),("Gcn4p","AAATTAGTCATGT"),("Gcn4p","AAATTAGTCATTC"),
    ("Gcn4p","AAATTAGTCATTT"),("Gcn4p","AAGAGTCAGA"),("Gcn4p","AAGAGTCATCAT"),
    ("Gcn4p","AAGAGTCATTT"),
]

# Complete list of 127 YEASTRACT transcription factors
YEASTRACT_TFS = [
    "Smp1p","Ert1p","Kar4p","Hcm1p","Rds1p","Uga3p","Pho2p","Mbp1p",
    "Rpn4p","Lys14p","Nrg1p","Gis1p","Ino2p","Swi5p","Ume6p","Upc2p",
    "Adr1p","Met32p","Sum1p","Cad1p","Stp1p","Gcn4p","Mig3p","Gln3p",
    "Aca1p","Mot2p","Flo8p","Swi4p","Com2p","Rph1p","Znf1p","Hac1p",
    "Gat1p","Pho4p","Fzf1p","Hap2p","Mig2p","Cup2p","Sut1p","Hsf1p",
    "Aft1p","Mig1p","Pdr1p","Rme1p","Rim101p","Yap3p","Stp2p","Ste12p",
    "Ndt80p","Rof1p","Stb5p","Skn7p","Fkh1p","Xbp1p","Cst6p","Yap5p",
    "Dal81p","Gzf3p","Gsm1p","Cbf1p","Ime1p","Ash1p","Abf1p","Hap4p",
    "Msn4p","Rgt1p","Ixr1p","Put3p","Dal80p","Bas1p","Ppr1p","Cha4p",
    "Ace2p","Rfx1p","Ecm22p","Hap1p","Pdr8p","Leu3p","Arg81p","Tda9p",
    "War1p","Yap1p","Stb4p","Mac1p","Msn2p","Arg80p","Mcm1p","Mot3p",
    "Mss11p","Hot1p","Cat8p","Dal82p","Rap1p","Gcr2p","Sko1p","Met4p",
    "Fkh2p","Crz1p","Ino4p","Rtg1p","Cin5p","Azf1p","Sfl1p","Yrr1p",
    "Tea1p","Tye7p","Hap5p","Pip2p","Gal4p","Usv1p","Aft2p","Rds2p",
    "Rlm1p","Gcr1p","Met31p","Haa1p","Rox1p","Arr1p","Oaf1p","Rtg3p",
    "Hap3p","Pdr3p","Reb1p","Nrg2p","Tec1p","Sip4p","Zap1p",
]

# SGD-normalised equivalents (strip trailing 'p', uppercase)
YEASTRACT_TFS_SGD = [t[:-1].upper() if t.lower().endswith("p") else t.upper() for t in YEASTRACT_TFS]


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def normalize_tf_name(yeastract_name: str) -> str:
    """
    Convert YEASTRACT protein notation to SGD gene notation.
    Examples: 'Gcn4p' -> 'GCN4', 'Ino2p' -> 'INO2', 'Stp1p' -> 'STP1'
    Rule: strip the trailing lowercase 'p' (protein suffix), then uppercase.
    """
    name = yeastract_name.strip()
    if name.endswith("p") or name.endswith("P"):
        name = name[:-1]
    return name.upper()


# ---------------------------------------------------------------------------
# Sequence analysis helpers
# ---------------------------------------------------------------------------

def _clean_consensus(seq: str) -> str:
    """Strip spacing and curly-brace spacer annotations like N{6,38}."""
    return re.sub(r"\{[^}]+\}", "", seq).strip().upper()


def _ambiguity_fraction(seq: str) -> float:
    """Fraction of characters in seq that are IUPAC ambiguity codes (not A/C/G/T)."""
    seq = _clean_consensus(seq)
    if not seq:
        return 0.0
    ambiguous = sum(1 for c in seq if c in IUPAC_AMBIGUOUS)
    return ambiguous / len(seq)


def _sequence_length(seq: str) -> int:
    return len(_clean_consensus(seq))


# ---------------------------------------------------------------------------
# Live fetcher (requires requests + beautifulsoup4)
# ---------------------------------------------------------------------------

def _fetch_live(url: str = YEASTRACT_URL, timeout: int = 30) -> list[tuple[str, str]]:
    """
    Scrape the YEASTRACT consensus table directly.
    The data lives in the innermost table (class='center') within the page's
    body table — not the first table, which is a navigation header.
    Returns list of (tf_yeastract, consensus) tuples.
    Raises ImportError if requests/bs4 not installed.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError(
            "requests and beautifulsoup4 are required for live fetching. "
            "Run: pip install requests beautifulsoup4"
        ) from e

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    # The data table is class="center" — the innermost table with TF/consensus rows.
    # Fallback: use the last table on the page (also works).
    data_table = soup.find("table", {"class": "center"})
    if data_table is None:
        all_tables = soup.find_all("table")
        data_table = all_tables[-1] if all_tables else None
    if data_table is None:
        return []

    records = []
    for row in data_table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 2:
            tf = cells[0].get_text(strip=True)
            consensus = cells[1].get_text(strip=True)
            # Skip header rows and empty cells
            if tf and consensus and not tf.lower().startswith("transcription"):
                records.append((tf, consensus))

    return records


# ---------------------------------------------------------------------------
# Build DataFrame from raw records
# ---------------------------------------------------------------------------

def _build_dataframe(records: list[tuple[str, str]]) -> pd.DataFrame:
    """Convert raw (tf_yeastract, consensus) records to an enriched DataFrame."""
    rows = []
    for tf_y, seq in records:
        clean_seq = _clean_consensus(seq)
        if not clean_seq:
            continue
        rows.append({
            "tf_yeastract": tf_y.strip(),
            "tf_sgd": normalize_tf_name(tf_y),
            "consensus": seq.strip(),
            "consensus_clean": clean_seq,
            "length": _sequence_length(seq),
            "ambiguity_fraction": round(_ambiguity_fraction(seq), 4),
            "specificity_score": round(1.0 - _ambiguity_fraction(seq), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-TF summary statistics
# ---------------------------------------------------------------------------

def _build_tf_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-TF binding statistics used for pi adjustment.

    Columns:
      tf_sgd, tf_yeastract, n_consensuses, mean_length, mean_ambiguity,
      mean_specificity, pi_consensus_factor
    """
    stats = (
        df.groupby(["tf_sgd", "tf_yeastract"])
        .agg(
            n_consensuses=("consensus", "count"),
            mean_length=("length", "mean"),
            mean_ambiguity=("ambiguity_fraction", "mean"),
            mean_specificity=("specificity_score", "mean"),
        )
        .reset_index()
    )
    stats["mean_length"] = stats["mean_length"].round(2)
    stats["mean_ambiguity"] = stats["mean_ambiguity"].round(4)
    stats["mean_specificity"] = stats["mean_specificity"].round(4)

    # pi_consensus_factor: multiplicative adjustment to base pi
    # More variants (promiscuous binding) → boost pi
    # More ambiguous consensus (flexible binding) → boost pi
    # Both effects are capped to avoid pi > 1 downstream.
    def _pi_factor(row) -> float:
        variant_boost = min(0.12, math.log(row["n_consensuses"] + 1) / math.log(50) * 0.12)
        ambiguity_boost = row["mean_ambiguity"] * 0.10
        return round(1.0 + variant_boost + ambiguity_boost, 4)

    stats["pi_consensus_factor"] = stats.apply(_pi_factor, axis=1)
    return stats.sort_values("n_consensuses", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_consensus_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load the full TF <-> consensus sequence table.

    Priority:
      1. Module-level cache (same process run)
      2. Cached CSV on disk (fast, works offline)
      3. Live fetch from YEASTRACT (requires internet + requests + bs4)
      4. Hardcoded fallback (always available)

    Returns a DataFrame with one row per (TF, consensus) pair plus
    sequence analysis columns.
    """
    global _CONSENSUS_DF
    if _CONSENSUS_DF is not None and not force_refresh:
        return _CONSENSUS_DF

    if not force_refresh and CONSENSUS_CSV.exists() and CONSENSUS_CSV.stat().st_size > 100:
        try:
            df = pd.read_csv(CONSENSUS_CSV, dtype=str)
            for col in ("length", "ambiguity_fraction", "specificity_score"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if "tf_sgd" in df.columns and len(df) > 0:
                _CONSENSUS_DF = df
                return _CONSENSUS_DF
        except Exception:
            pass  # fall through to rebuild

    # Try live fetch first, fall back to hardcoded dataset
    try:
        records = _fetch_live()
    except Exception:
        records = list(_HARDCODED_DATA)

    df = _build_dataframe(records)
    # Persist to disk for offline use
    try:
        df.to_csv(CONSENSUS_CSV, index=False)
    except OSError:
        pass

    _CONSENSUS_DF = df
    return _CONSENSUS_DF


def load_tf_consensus_stats(force_refresh: bool = False) -> pd.DataFrame:
    """Return per-TF aggregated binding statistics (used for pi adjustment)."""
    global _STATS_DF
    if _STATS_DF is not None and not force_refresh:
        return _STATS_DF
    _STATS_DF = _build_tf_stats(load_consensus_data(force_refresh))
    return _STATS_DF


def get_consensus_for_tf(tf_name: str) -> pd.DataFrame:
    """
    Return all consensus sequences for a given TF (SGD or YEASTRACT name).
    Case-insensitive. Returns empty DataFrame if TF not in dataset.
    """
    df = load_consensus_data()
    name_upper = tf_name.upper().rstrip("P")  # handle both GCN4 and GCN4p
    mask = df["tf_sgd"].str.upper() == name_upper
    return df[mask].reset_index(drop=True)


def get_pi_consensus_factor(tf_name: str) -> float:
    """
    Return the pi_consensus_factor for a TF.
    Factor > 1 means consensus evidence suggests higher inheritance probability.
    Returns 1.0 (no adjustment) if TF not in YEASTRACT dataset.
    """
    stats = load_tf_consensus_stats()
    name_upper = tf_name.upper().rstrip("P")
    row = stats[stats["tf_sgd"].str.upper() == name_upper]
    if row.empty:
        return 1.0
    return float(row["pi_consensus_factor"].iloc[0])


def list_yeastract_tfs() -> list[str]:
    """Return sorted list of SGD-normalised TF names in the YEASTRACT dataset."""
    stats = load_tf_consensus_stats()
    return sorted(stats["tf_sgd"].tolist())
