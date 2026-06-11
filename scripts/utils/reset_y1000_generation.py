"""
reset_y1000_generation.py
Cleanly wipe Y1000+ generation state so the next app launch starts fresh.

IMPORTANT: Stop the Streamlit app (Ctrl+C) before running this script.
Running it while the app is open will conflict with the background thread.

Usage:
    python reset_y1000_generation.py           # reset state, keep cached genomes
    python reset_y1000_generation.py --full    # also delete all cached genome FASTAs
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.y1000plus_loader import PROCESSED_DIR
from model.y1000plus_generator import PROGRESS_FILE, LOCK_FILE
from model.pi3_tfbs_conservation import PI3_CSV
from model.pi2_sequence_homology import PI2_CSV
from model.pi4_snp_binding import PI4_CSV

FULL = "--full" in sys.argv

print()
print("=" * 55)
print("  Y1000+ Generation Reset")
print("=" * 55)

# ── Files that are always removed ────────────────────────────────────
always_remove = [
    (PROGRESS_FILE,  "progress file"),
    (LOCK_FILE,      "lock file"),
    (PI2_CSV,        "pi2_sequence_homology.csv"),
    (PI3_CSV,        "pi3_tfbs_conservation.csv"),
    (PI4_CSV,        "pi4_snp_binding_sites.csv"),
    (PROCESSED_DIR / "pi3_pairwise_histogram.csv", "pi3_pairwise_histogram.csv"),
]

# ── Genome FASTAs — only removed with --full ──────────────────────────
genome_dir = PROCESSED_DIR / "genomes"
genomes = list(genome_dir.glob("*.fasta")) if genome_dir.exists() else []

print()
if FULL:
    print(f"  Mode: FULL reset (includes {len(genomes)} cached genome FASTAs)")
    print("  Next run will re-extract all genomes from the ZIP (slow).")
else:
    print("  Mode: state reset only (cached genome FASTAs kept)")
    print("  Next run reuses cached genomes — much faster.")
    print("  Use --full to also delete genome FASTAs.")

print()
print("  Will delete:")
for path, label in always_remove:
    if path.exists():
        print(f"    {label}")
if FULL and genomes:
    print(f"    {len(genomes)} genome FASTAs in processed/genomes/")

print()
answer = input("  Proceed? [y/N] ").strip().lower()
if answer != "y":
    print("  Cancelled.")
    sys.exit(0)

print()

# Delete state and output files
for path, label in always_remove:
    if path.exists():
        path.unlink()
        print(f"  Deleted: {label}")
    else:
        print(f"  Skipped: {label} (not found)")

# Delete genome FASTAs if --full
if FULL and genome_dir.exists():
    shutil.rmtree(genome_dir)
    print(f"  Deleted: processed/genomes/ ({len(genomes)} files)")

print()
print("  Done. Start the Streamlit app to begin a fresh generation run.")
print("=" * 55)
print()
