"""
y1000plus_loader.py
Manifest builder and on-demand file accessor for the Y1000+ dataset
(Opulente et al. 2024, Science; https://y1000plus.wei.wisc.edu/).

Archives (all in y1000plus_data/):
  y1000p_gff3_files.tar.gz  (360 MB)  — GFF3 gene annotations
  y1000p_pep_files.tar.gz   (1.9 GB)  — protein sequences
  y1000p_cds_files.tar.gz   (2.7 GB)  — CDS nucleotide sequences
  y1000p_genome_files.zip   (4.4 GB)  — genome FASTA assemblies (.fas.gz inside)

Naming convention:
  {assembly_id}[.haplomerger2].{annotation_type}.{ext}
  annotation_type: 'sgd'   — curated SGD reference (S. cerevisiae S288C only)
                   'final' — BRAKER ab initio predicted genes (all 1,154 species)

The S. cerevisiae S288C reference is assembly_id='saccharomyces_cerevisiae',
annotation_type='sgd', and is_reference=True in the manifest.

The genome ZIP contains doubly-compressed files (.fas.gz inside .zip).
extract_genome_fasta() decompresses both layers to a plain .fasta file.
"""

from __future__ import annotations

import gzip
import re
import tarfile
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

_HERE = Path(__file__).parent.parent
Y1000_DIR = _HERE / "y1000plus_data"
PROCESSED_DIR = Y1000_DIR / "processed"

GFF3_TAR = Y1000_DIR / "y1000p_gff3_files.tar.gz"
PEP_TAR = Y1000_DIR / "y1000p_pep_files.tar.gz"
CDS_TAR = Y1000_DIR / "y1000p_cds_files.tar.gz"
GENOME_ZIP = Y1000_DIR / "y1000p_genome_files.zip"
MANIFEST_CSV = PROCESSED_DIR / "y1000plus_manifest.csv"

REF_ASSEMBLY_ID = "saccharomyces_cerevisiae"
REF_ANN_TYPE = "sgd"


# ---------------------------------------------------------------------------
# Filename parsing helpers
# ---------------------------------------------------------------------------

def _parse_member_name(basename: str, ext: str) -> tuple[str, str]:
    """
    Parse archive member basename into (assembly_id, annotation_type).

    Examples:
      'saccharomyces_cerevisiae.sgd.gff3'            → ('saccharomyces_cerevisiae', 'sgd')
      'saccharomyces_cerevisiae.final.gff3'           → ('saccharomyces_cerevisiae', 'final')
      'yHDO603_zygosaccharomyces_sapae.haplomerger2.final.gff3'
                                                      → ('yHDO603_zygosaccharomyces_sapae', 'final')
    """
    stem = basename[: -len(ext)]           # strip extension (.gff3 / .pep / .cds)
    stem = stem.replace(".haplomerger2", "")  # normalize haplomerger2 assemblies
    if "." in stem:
        idx = stem.rfind(".")
        return stem[:idx], stem[idx + 1 :]
    return stem, "unknown"


def _species_name(assembly_id: str) -> str:
    """
    Convert assembly_id to a human-readable species name.

    'saccharomyces_cerevisiae'                → 'Saccharomyces cerevisiae'
    'yHAB133_kazachstania_unispora_160519'    → 'Kazachstania unispora'
    'yHDO603_zygosaccharomyces_sapae_190924' → 'Zygosaccharomyces sapae'
    """
    name = assembly_id
    # Strip lab-strain prefix: yHAB133_, yHMPu5000034864_, yHDO572_
    name = re.sub(r"^y[A-Za-z]+\d+_", "", name)
    # Strip trailing 6-digit date: _160519, _201018
    name = re.sub(r"_\d{6}$", "", name)
    words = name.replace("_", " ").split()
    if words:
        words[0] = words[0].capitalize()
    return " ".join(words)


# ---------------------------------------------------------------------------
# Archive member inventory
# ---------------------------------------------------------------------------

def _read_tar_members(tar_path: Path, ext: str) -> dict[tuple[str, str], str]:
    """
    Iterate a tar.gz archive without extracting content.
    Returns {(assembly_id, annotation_type): archive_member_path}.
    """
    members: dict[tuple[str, str], str] = {}
    with tarfile.open(tar_path, "r:gz") as t:
        for name in t.getnames():
            if name.endswith(ext):
                basename = Path(name).name
                key = _parse_member_name(basename, ext)
                members[key] = name
    return members


def _read_genome_members() -> dict[str, str]:
    """
    Iterate the genome ZIP without extracting.
    Returns {assembly_id: archive_member_path}.
    Files inside the zip are {assembly_id}[.haplomerger2].fas.gz.
    """
    members: dict[str, str] = {}
    with zipfile.ZipFile(GENOME_ZIP, "r") as z:
        for name in z.namelist():
            if name.endswith(".fas.gz"):
                basename = Path(name).name
                # Strip .haplomerger2 and .fas.gz to get assembly_id
                aid = basename.replace(".haplomerger2", "").replace(".fas.gz", "")
                members[aid] = name
    return members


# ---------------------------------------------------------------------------
# Manifest building and loading
# ---------------------------------------------------------------------------

def build_manifest(save: bool = True) -> pd.DataFrame:
    """
    Build a manifest CSV from archive member lists.

    Reads member names from all four archives (no extraction).
    One row per (assembly_id, annotation_type) combination.
    Most species have annotation_type='final'. S. cerevisiae S288C
    additionally has annotation_type='sgd' (is_reference=True).

    Columns:
      assembly_id, annotation_type, species_name,
      gff3_archive_path, pep_archive_path, cds_archive_path,
      genome_archive_path, is_reference
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for archive in (GFF3_TAR, PEP_TAR, CDS_TAR):
        if not archive.exists():
            raise FileNotFoundError(
                f"Required data archive not found: {archive.name}\n"
                "Download the Y1000+ dataset archives from https://y1000plus.wei.wisc.edu/ "
                "and place them in the y1000plus_data/ directory."
            )

    print("Reading GFF3 archive members...")
    gff3_m = _read_tar_members(GFF3_TAR, ".gff3")
    print(f"  {len(gff3_m)} GFF3 files")

    print("Reading PEP archive members...")
    pep_m = _read_tar_members(PEP_TAR, ".pep")
    print(f"  {len(pep_m)} PEP files")

    print("Reading CDS archive members...")
    # CDS files are nested: .cds.tar.gz inside the outer tar
    cds_m = _read_tar_members(CDS_TAR, ".cds.tar.gz")
    print(f"  {len(cds_m)} CDS files")

    print("Reading genome ZIP members...")
    genome_m = _read_genome_members()
    print(f"  {len(genome_m)} genome files")

    all_keys = sorted(set(gff3_m) | set(pep_m) | set(cds_m))

    rows = []
    for aid, ann in all_keys:
        rows.append(
            {
                "assembly_id": aid,
                "annotation_type": ann,
                "species_name": _species_name(aid),
                "gff3_archive_path": gff3_m.get((aid, ann), ""),
                "pep_archive_path": pep_m.get((aid, ann), ""),
                "cds_archive_path": cds_m.get((aid, ann), ""),
                "genome_archive_path": genome_m.get(aid, ""),
                "is_reference": (aid == REF_ASSEMBLY_ID and ann == REF_ANN_TYPE),
            }
        )

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(MANIFEST_CSV, index=False)
        print(f"Manifest saved -> {MANIFEST_CSV}")
    return df


@lru_cache(maxsize=1)
def load_manifest() -> pd.DataFrame:
    """Load manifest CSV, building it from archives if not yet present."""
    if not MANIFEST_CSV.exists():
        return build_manifest()
    df = pd.read_csv(MANIFEST_CSV, dtype=str)
    df["is_reference"] = df["is_reference"].str.strip().str.lower() == "true"
    return df


def get_reference_row() -> pd.Series:
    """Return the manifest row for S. cerevisiae S288C (SGD annotation)."""
    df = load_manifest()
    ref = df[df["is_reference"]]
    if ref.empty:
        raise ValueError("S. cerevisiae SGD reference row not found in manifest.")
    return ref.iloc[0]


def manifest_summary() -> dict:
    """High-level counts for the dataset."""
    df = load_manifest()
    return {
        "total_rows": len(df),
        "n_final": int((df["annotation_type"] == "final").sum()),
        "n_sgd": int((df["annotation_type"] == "sgd").sum()),
        "n_with_genome": int((df["genome_archive_path"] != "").sum()),
        "n_unique_assemblies": df["assembly_id"].nunique(),
    }


# ---------------------------------------------------------------------------
# On-demand file extraction
# ---------------------------------------------------------------------------

def _extract_from_tar(tar_path: Path, member_name: str) -> bytes:
    """Extract one member from a tar.gz archive and return its raw bytes."""
    if not tar_path.exists():
        raise FileNotFoundError(
            f"Required data archive not found: {tar_path.name}\n"
            "Download the Y1000+ dataset archives from https://y1000plus.wei.wisc.edu/ "
            "and place them in the y1000plus_data/ directory."
        )
    with tarfile.open(tar_path, "r:gz") as t:
        f = t.extractfile(member_name)
        if f is None:
            raise ValueError(f"Member {member_name!r} not found in {tar_path.name}")
        return f.read()


def _local_gff3_path(assembly_id: str, annotation_type: str, archive_basename: str) -> Path:
    """Return the expected local path for an extracted GFF3 file."""
    # Check bulk-extraction location first (from tarfile.extractall to processed/)
    bulk = PROCESSED_DIR / "y1000p_gff3_files" / archive_basename
    if bulk.exists():
        return bulk
    # On-demand extraction location
    return PROCESSED_DIR / "gff3" / archive_basename


def get_gff3_path(
    assembly_id: str,
    annotation_type: str = "final",
) -> Path:
    """
    Return the local path to a species' GFF3 file, extracting from the
    tar archive on demand if the file is not already present.

    Checks bulk-extraction location (y1000plus_data/processed/y1000p_gff3_files/)
    before falling back to on-demand extraction into y1000plus_data/processed/gff3/.
    """
    df = load_manifest()
    row = df[
        (df["assembly_id"] == assembly_id)
        & (df["annotation_type"] == annotation_type)
    ]
    if row.empty:
        raise ValueError(
            f"No manifest entry for assembly_id={assembly_id!r}, "
            f"annotation_type={annotation_type!r}"
        )
    archive_path = row.iloc[0]["gff3_archive_path"]
    if not archive_path:
        raise ValueError(f"No GFF3 archive path for {assembly_id!r}")

    basename = Path(archive_path).name
    local = _local_gff3_path(assembly_id, annotation_type, basename)
    if local.exists():
        return local

    local.parent.mkdir(parents=True, exist_ok=True)
    data = _extract_from_tar(GFF3_TAR, archive_path)
    local.write_bytes(data)
    return local


def get_pep_path(
    assembly_id: str,
    annotation_type: str = "final",
) -> Path:
    """
    Return the local path to a species' peptide FASTA, extracting on demand.
    Files are written to y1000plus_data/processed/pep/.
    """
    df = load_manifest()
    row = df[
        (df["assembly_id"] == assembly_id)
        & (df["annotation_type"] == annotation_type)
    ]
    if row.empty:
        raise ValueError(
            f"No manifest entry for assembly_id={assembly_id!r}, "
            f"annotation_type={annotation_type!r}"
        )
    archive_path = row.iloc[0]["pep_archive_path"]
    if not archive_path:
        raise ValueError(f"No PEP archive path for {assembly_id!r}")

    basename = Path(archive_path).name
    local = PROCESSED_DIR / "pep" / basename
    if local.exists():
        return local

    local.parent.mkdir(parents=True, exist_ok=True)
    data = _extract_from_tar(PEP_TAR, archive_path)
    local.write_bytes(data)
    return local


def get_genome_fasta_path(assembly_id: str) -> Path:
    """
    Return the local path to a species' genome FASTA.

    Genome files inside the ZIP are doubly compressed (.fas.gz inside .zip).
    This function decompresses both layers and writes a plain .fasta file
    to y1000plus_data/processed/genomes/{assembly_id}.fasta.
    """
    df = load_manifest()
    rows = df[df["assembly_id"] == assembly_id]
    if rows.empty:
        raise ValueError(f"assembly_id={assembly_id!r} not found in manifest")
    archive_path = rows.iloc[0]["genome_archive_path"]
    if not archive_path:
        raise ValueError(f"No genome archive path for {assembly_id!r}")

    dest_dir = PROCESSED_DIR / "genomes"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{assembly_id}.fasta"
    if out_path.exists():
        return out_path

    with zipfile.ZipFile(GENOME_ZIP, "r") as z:
        with z.open(archive_path) as compressed_gz:
            with gzip.open(compressed_gz, "rb") as fasta_stream:
                out_path.write_bytes(fasta_stream.read())
    return out_path


# ---------------------------------------------------------------------------
# Convenience accessors for S. cerevisiae S288C reference
# ---------------------------------------------------------------------------

def scerevisiae_gff3_path() -> Path:
    """Local path to S. cerevisiae S288C SGD GFF3 (extracted on demand)."""
    return get_gff3_path(REF_ASSEMBLY_ID, REF_ANN_TYPE)


def scerevisiae_pep_path() -> Path:
    """Local path to S. cerevisiae S288C SGD peptide FASTA (extracted on demand)."""
    return get_pep_path(REF_ASSEMBLY_ID, REF_ANN_TYPE)


def scerevisiae_genome_path() -> Path:
    """Local path to S. cerevisiae S288C genome FASTA (extracted on demand)."""
    return get_genome_fasta_path(REF_ASSEMBLY_ID)


# ---------------------------------------------------------------------------
# Representative species subset (for Tasks 4/6 — TFBS conservation scan)
# ---------------------------------------------------------------------------

# 50 species spanning Saccharomycotina phylogeny, chosen to maximise
# phylogenetic diversity while keeping compute tractable.
# Names are the exact assembly_id values from the Y1000+ manifest.
# Strain-prefixed IDs (yHMPu..., yHAB...) are used where the archive
# does not include a simple genus_species entry.
REPRESENTATIVE_SUBSET: list[str] = [
    # Saccharomyces sensu stricto (all have simple names in archive)
    "saccharomyces_cerevisiae",
    "saccharomyces_paradoxus",
    "saccharomyces_mikatae",
    "saccharomyces_kudriavzevii",
    "saccharomyces_uvarum",
    "saccharomyces_eubayanus",
    "saccharomyces_arboricola",
    "saccharomyces_jurei",
    # Lachancea clade
    "lachancea_fantastica",
    "yHMPu5000034678_lachancea_thermotolerans_180604",
    "yHMPu5000026219_lachancea_waltii_180604",
    # Kluyveromyces / Eremothecium
    "kluyveromyces_marxianus",
    "kluyveromyces_lactis",
    "eremothecium_gossypii",
    "eremothecium_cymbalariae",
    "eremothecium_coryli",
    # Naumovozyma / Tetrapisispora
    "yHMPu5000034871_naumovozyma_castellii_180604",
    "yHMPu5000034874_tetrapisispora_blattae_190924",
    # Zygosaccharomyces
    "zygosaccharomyces_bailii",
    "yHMPu5000034863_zygosaccharomyces_rouxii_180604",
    # Kazachstania group
    "kazachstania_naganishii",
    "yHAB127_kazachstania_bovina_160519",
    # Candida / CTG clade
    "candida_albicans",
    "candida_tropicalis",
    "candida_parapsilosis",
    "candida_dubliniensis",
    "candida_orthopsilosis",
    "clavispora_lusitaniae",
    # Pichia / Komagataella
    "komagataella_pastoris",
    "komagataella_phaffi",
    "ogataea_parapolymorpha",
    "hansenula_polymorpha",
    # Metschnikowia clade
    "metschnikowia_arizonensis",
    "metschnikowia_santaceciliae",
    # Hanseniaspora
    "hanseniaspora_uvarum",
    "yHMPu5000026147_hanseniaspora_vineae_190924",
    # Yarrowia (outgroup)
    "yarrowia_lipolytica",
    # Debaryomycetaceae
    "yHMPu5000034659_debaryomyces_hansenii_180604",
    "yHMPu5000035307_millerozyma_farinosa_180604",
    # Arxula / Scheffersomyces
    "arxula_adeninivorans",
    "scheffersomyces_stipitis",
    # Eremothecium / Ashbya outgroup
    "ashbya_aceri",
    # Deep outgroups
    "alloascoidea_hylecoeti",
    "saprochaete_fungicola",
    "saprochaete_suaveolens",
    # Brettanomyces
    "brettanomyces_anomalus",
    # Candida auris / versatilis
    "candida_auris",
    "candida_versatilis",
]


def get_representative_subset(validate_against_manifest: bool = True) -> list[str]:
    """
    Return the list of representative assembly_ids for TFBS conservation scans.
    Filters to species actually present in the manifest.
    """
    if not validate_against_manifest:
        return REPRESENTATIVE_SUBSET
    df = load_manifest()
    available = set(df["assembly_id"].tolist())
    present = [aid for aid in REPRESENTATIVE_SUBSET if aid in available]
    missing = [aid for aid in REPRESENTATIVE_SUBSET if aid not in available]
    if missing:
        import warnings
        warnings.warn(
            f"{len(missing)} species in REPRESENTATIVE_SUBSET not in manifest: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
            stacklevel=2,
        )
    return present
