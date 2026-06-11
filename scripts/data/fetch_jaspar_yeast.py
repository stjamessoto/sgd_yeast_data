"""
fetch_jaspar_yeast.py
=====================
Fetches all Saccharomyces cerevisiae (tax_id=4932) CORE TF binding profiles
from the JASPAR 2024 REST API and writes them to:

    jaspar_yeast_tfbs_2024.csv   — one row per TF, all metadata + PFM summary
    jaspar_yeast_pfm_long.csv    — one row per (TF × position × nucleotide), full PFM

Usage:
    pip install requests
    python fetch_jaspar_yeast.py

Outputs are placed in the project root directory.
"""

import requests
import json
import csv
import time
import math
import os
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_URL   = "https://jaspar.elixir.no/api/v1"
TAX_ID     = "4932"           # Saccharomyces cerevisiae
COLLECTION = "CORE"
PAGE_SIZE  = 200
HEADERS    = {"User-Agent": "jaspar-yeast-fetcher/1.0", "Accept": "application/json"}
DELAY      = 0.25             # seconds between detail requests (be polite)

_ROOT    = Path(__file__).parent.parent.parent
OUT_WIDE = str(_ROOT / "jaspar_yeast_tfbs_2024.csv")
OUT_LONG = str(_ROOT / "jaspar_yeast_pfm_long.csv")
# ───────────────────────────────────────────────────────────────────────────────


def fetch_json(url, params=None):
    """GET a JSON endpoint with simple retry logic."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 2:
                raise
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(2 ** attempt)


def pfm_to_consensus(pfm):
    """
    Derive a consensus sequence from a PFM dict  {A:[..], C:[..], G:[..], T:[..]}.
    At each position picks the nucleotide with the highest frequency.
    """
    if not pfm:
        return ""
    positions = len(pfm.get("A", []))
    consensus = []
    for i in range(positions):
        counts = {nuc: pfm[nuc][i] for nuc in "ACGT" if nuc in pfm}
        consensus.append(max(counts, key=counts.get))
    return "".join(consensus)


def pfm_to_pwm(pfm, pseudocount=0.8):
    """
    Convert PFM → PWM (log2 odds vs background 0.25 each base).
    Returns dict {A:[..], C:[..], G:[..], T:[..]} of floats.
    """
    if not pfm:
        return {}
    positions = len(pfm.get("A", []))
    pwm = {nuc: [] for nuc in "ACGT"}
    for i in range(positions):
        total = sum(pfm[nuc][i] for nuc in "ACGT" if nuc in pfm) + pseudocount * 4
        for nuc in "ACGT":
            freq = (pfm.get(nuc, [0] * positions)[i] + pseudocount) / total
            pwm[nuc].append(round(math.log2(freq / 0.25), 4))
    return pwm


def information_content(pfm, pseudocount=0.8):
    """Total information content of a PFM in bits."""
    if not pfm:
        return 0.0
    positions = len(pfm.get("A", []))
    total_ic = 0.0
    for i in range(positions):
        raw = [pfm.get(nuc, [0]*positions)[i] for nuc in "ACGT"]
        n = sum(raw) + pseudocount * 4
        freqs = [(v + pseudocount) / n for v in raw]
        h = -sum(f * math.log2(f) for f in freqs if f > 0)
        total_ic += 2 - h   # max entropy for 4 bases = 2 bits
    return round(total_ic, 4)


def fetch_all_profiles():
    """Page through the JASPAR API and collect all yeast CORE profile stubs."""
    profiles = []
    page = 1
    total = None
    while True:
        data = fetch_json(
            f"{BASE_URL}/matrix/",
            params={
                "tax_id":     TAX_ID,
                "collection": COLLECTION,
                "format":     "json",
                "page_size":  PAGE_SIZE,
                "version":    "latest",
                "page":       page,
            },
        )
        if total is None:
            total = data["count"]
            print(f"Found {total} yeast CORE profiles in JASPAR 2024.")
        profiles.extend(data["results"])
        print(f"  Page {page}: fetched {len(data['results'])} profiles "
              f"({len(profiles)}/{total})")
        if not data["next"]:
            break
        page += 1
        time.sleep(DELAY)
    return profiles


def fetch_profile_detail(matrix_id):
    """Fetch full detail (including pfm) for a single matrix ID."""
    return fetch_json(f"{BASE_URL}/matrix/{matrix_id}/")


def build_wide_row(detail):
    """Flatten a JASPAR detail response into one CSV row."""
    pfm = detail.get("pfm", {})
    pwm = pfm_to_pwm(pfm)
    consensus = pfm_to_consensus(pfm)
    ic = information_content(pfm)
    width = len(pfm.get("A", []))

    species_list = detail.get("species", [])
    species_names = "|".join(s.get("name", "") for s in species_list)
    tax_ids       = "|".join(str(s.get("tax_id", "")) for s in species_list)

    pubmed_ids = "|".join(
        str(p.get("pubmed_id", "")) for p in detail.get("pubmed", [])
    )

    # PFM serialised as JSON strings per nucleotide (easy to parse back in pandas)
    row = {
        "matrix_id":      detail.get("matrix_id", ""),
        "name":           detail.get("name", ""),
        "collection":     detail.get("collection", ""),
        "tf_class":       detail.get("class", [""])[0] if detail.get("class") else "",
        "tf_family":      detail.get("family", [""])[0] if detail.get("family") else "",
        "tax_group":      detail.get("tax_group", ""),
        "species_names":  species_names,
        "tax_ids":        tax_ids,
        "data_type":      detail.get("type", ""),
        "pubmed_ids":     pubmed_ids,
        "motif_width":    width,
        "information_content_bits": ic,
        "consensus_sequence": consensus,
        # PFM counts
        "pfm_A":          json.dumps(pfm.get("A", [])),
        "pfm_C":          json.dumps(pfm.get("C", [])),
        "pfm_G":          json.dumps(pfm.get("G", [])),
        "pfm_T":          json.dumps(pfm.get("T", [])),
        # PWM log-odds
        "pwm_A":          json.dumps(pwm.get("A", [])),
        "pwm_C":          json.dumps(pwm.get("C", [])),
        "pwm_G":          json.dumps(pwm.get("G", [])),
        "pwm_T":          json.dumps(pwm.get("T", [])),
        "jaspar_url":     f"https://jaspar.elixir.no/matrix/{detail.get('matrix_id', '')}/",
    }
    return row


def build_long_rows(detail):
    """One row per (TF, position, nucleotide) — useful for downstream scanning."""
    pfm = detail.get("pfm", {})
    pwm = pfm_to_pwm(pfm)
    width = len(pfm.get("A", []))
    rows = []
    for i in range(width):
        for nuc in "ACGT":
            count = pfm.get(nuc, [0]*width)[i]
            score = pwm.get(nuc, [0.0]*width)[i] if pwm else 0.0
            rows.append({
                "matrix_id":  detail.get("matrix_id", ""),
                "tf_name":    detail.get("name", ""),
                "position":   i + 1,  # 1-based
                "nucleotide": nuc,
                "count":      count,
                "pwm_score":  score,
            })
    return rows


def main():
    print("=" * 60)
    print("JASPAR 2024 — Yeast (S. cerevisiae) CORE Profile Fetcher")
    print("=" * 60)

    # ── 1. Get all profile stubs ───────────────────────────────────
    profiles = fetch_all_profiles()

    # ── 2. Fetch detail for each profile ──────────────────────────
    wide_rows = []
    long_rows = []
    failed    = []

    for i, stub in enumerate(profiles, 1):
        mid = stub.get("matrix_id") or stub.get("id", "")
        print(f"  [{i:>3}/{len(profiles)}] Fetching detail for {mid} "
              f"({stub.get('name', '?')})", end="  ", flush=True)
        try:
            detail = fetch_profile_detail(mid)
            wide_rows.append(build_wide_row(detail))
            long_rows.extend(build_long_rows(detail))
            print("OK")
        except Exception as e:
            print(f"FAILED — {e}")
            failed.append(mid)
        time.sleep(DELAY)

    # ── 3. Write wide CSV ──────────────────────────────────────────
    if wide_rows:
        wide_fields = list(wide_rows[0].keys())
        with open(OUT_WIDE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=wide_fields)
            writer.writeheader()
            writer.writerows(wide_rows)
        print(f"\n✓ Wrote {len(wide_rows)} rows → {OUT_WIDE}")
    
    # ── 4. Write long CSV ──────────────────────────────────────────
    if long_rows:
        long_fields = ["matrix_id", "tf_name", "position",
                       "nucleotide", "count", "pwm_score"]
        with open(OUT_LONG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=long_fields)
            writer.writeheader()
            writer.writerows(long_rows)
        print(f"✓ Wrote {len(long_rows)} rows  → {OUT_LONG}")

    if failed:
        print(f"\n⚠ Failed to fetch {len(failed)} profiles: {failed}")

    print("\nDone.")


if __name__ == "__main__":
    main()