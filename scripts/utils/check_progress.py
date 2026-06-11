"""
check_progress.py
Run from the project root to see the current Y1000+ generation status.

    python check_progress.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model.y1000plus_generator import get_status, csvs_ready
from model.y1000plus_loader import PROCESSED_DIR
from model.pi3_tfbs_conservation import PI3_CSV
from model.pi2_sequence_homology import PI2_CSV
from model.pi4_snp_binding import PI4_CSV

s = get_status()
r = csvs_ready()

genome_dir = PROCESSED_DIR / "genomes"
genomes = list(genome_dir.glob("*.fasta")) if genome_dir.exists() else []

BAR_WIDTH = 30
filled = int(BAR_WIDTH * s["pct"] / 100)
bar = "#" * filled + "-" * (BAR_WIDTH - filled)

print()
print("=" * 55)
print("  Y1000+ Generation Status")
print("=" * 55)
print(f"  [{bar}] {s['pct']}%")
print(f"  Status  : {s['status']}")
print(f"  Message : {s['message']}")
print(f"  Updated : {s['updated_at'] or 'never'}")
print()
print("  Output CSVs:")
for key, path in [("pi2", PI2_CSV), ("pi3", PI3_CSV), ("pi4", PI4_CSV)]:
    if path.exists():
        size_kb = path.stat().st_size // 1024
        rows = sum(1 for _ in open(path)) - 1
        print(f"    {key}: done   ({rows:,} rows, {size_kb} KB)")
    else:
        mark = "running" if s["status"].endswith(key) else "pending"
        print(f"    {key}: {mark}")
print()
print(f"  Genomes cached : {len(genomes)} / 48")
if genomes:
    print(f"  Last cached    : {sorted(genomes, key=lambda p: p.stat().st_mtime)[-1].stem}")
if s.get("error"):
    print(f"\n  ERROR: {s['error']}")
print("=" * 55)
print()
