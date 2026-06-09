"""
y1000plus_generator.py
Background generation of the three cross-species π CSV files.

Designed for use inside a Streamlit app:
  - start_generation_if_needed() is called once at server start via
    @st.cache_resource. It spawns a daemon thread only if any CSV is absent.
  - Progress is written to PROGRESS_FILE (a JSON file) so any Streamlit
    rerun can read current status without touching session_state.
  - get_status() returns a dict the UI can display.

The three CSVs are generated in dependency order:
  1. pi2_sequence_homology.csv  — protein pairwise identity (~minutes)
  2. pi3_tfbs_conservation.csv  — PWM conservation scan   (~minutes–hours)
  3. pi4_snp_binding_sites.csv  — SNP at binding sites    (~minutes–hours)

A lock file prevents a second thread from starting if the server is
restarted while a generation run is still in progress on disk.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from .y1000plus_loader import PROCESSED_DIR
from .pi3_tfbs_conservation import PI3_CSV
from .pi2_sequence_homology import PI2_CSV
from .pi4_snp_binding import PI4_CSV

PROGRESS_FILE = PROCESSED_DIR / ".y1000_generation_progress.json"
LOCK_FILE = PROCESSED_DIR / ".y1000_generation.lock"

_thread_lock = threading.Lock()

Status = Literal["idle", "running_pi2", "running_pi3", "running_pi4", "done", "error"]


# ---------------------------------------------------------------------------
# Progress file helpers
# ---------------------------------------------------------------------------

def _write_progress(
    status: Status,
    message: str = "",
    error: str = "",
    pct: int = 0,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(
        json.dumps(
            {
                "status": status,
                "message": message,
                "error": error,
                "pct": pct,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        ),
        encoding="utf-8",
    )


def get_status() -> dict:
    """
    Return the current generation status dict.

    Keys: status, message, error, pct, updated_at
    status values: idle | running_pi2 | running_pi3 | running_pi4 | done | error
    """
    if not PROGRESS_FILE.exists():
        if all(p.exists() for p in [PI2_CSV, PI3_CSV, PI4_CSV]):
            return {"status": "done", "message": "All CSVs present.", "pct": 100,
                    "error": "", "updated_at": ""}
        return {"status": "idle", "message": "Not yet started.", "pct": 0,
                "error": "", "updated_at": ""}
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "idle", "message": "Unreadable progress file.", "pct": 0,
                "error": "", "updated_at": ""}


def csvs_ready() -> dict[str, bool]:
    """Return which output CSVs already exist."""
    return {
        "pi2": PI2_CSV.exists(),
        "pi3": PI3_CSV.exists(),
        "pi4": PI4_CSV.exists(),
    }


def all_done() -> bool:
    return all(csvs_ready().values())


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _generation_worker() -> None:
    """
    Run inside a daemon thread. Generates the three π CSVs sequentially.
    Writes progress to PROGRESS_FILE after each step.
    """
    LOCK_FILE.touch()
    try:
        ready = csvs_ready()

        # ── π₂: protein sequence homology ────────────────────────────
        if not ready["pi2"]:
            _write_progress("running_pi2", "Computing pairwise protein sequence identity…", pct=5)
            from .pi2_sequence_homology import estimate_pi2
            estimate_pi2(verbose=False, save=True)
            _write_progress("running_pi2", "π₂ sequence homology done.", pct=33)
        else:
            _write_progress("running_pi2", "π₂ already cached — skipped.", pct=33)

        # ── π₃: TFBS conservation scan ───────────────────────────────
        if not PI3_CSV.exists():
            _write_progress(
                "running_pi3",
                "Extracting promoters and scanning TFBS conservation — "
                "first run extracts genomes from the archive (10–30 min).",
                pct=35,
            )

            def _pi3_species_callback(phase: str, done: int, total: int, label: str) -> None:
                if phase == "prescan":
                    # S. cerevisiae pre-scan: TF 10, 20, ... of ~127  →  35–48%
                    pct = 35 + int(13 * done / max(total, 1))
                    _write_progress(
                        "running_pi3",
                        f"Pre-scanning S. cerevisiae: TF {done}/{total} ({label})",
                        pct=pct,
                    )
                else:
                    # Species loop: 1..48  →  48–66%
                    pct = 48 + int(18 * done / max(total, 1))
                    short = label.split("_")[-3] + "_" + label.split("_")[-2] \
                        if label.count("_") >= 3 else label
                    _write_progress(
                        "running_pi3",
                        f"TFBS conservation: species {done}/{total} — {short}",
                        pct=pct,
                    )

            from .pi3_tfbs_conservation import estimate_pi3_all_tfs, build_pairwise_histogram
            pi3_df = estimate_pi3_all_tfs(
                verbose=False, save=True, progress_callback=_pi3_species_callback
            )
            build_pairwise_histogram(pi3_df, save=True)
            _write_progress("running_pi3", "π₃ TFBS conservation done.", pct=66)
        else:
            _write_progress("running_pi3", "π₃ already cached — skipped.", pct=66)

        # ── π₄: SNP at binding sites ──────────────────────────────────
        if not PI4_CSV.exists():
            _write_progress(
                "running_pi4",
                "Pre-scanning S. cerevisiae binding sites (~3 min)…",
                pct=68,
            )

            def _pi4_callback(phase: str, done: int, total: int, label: str) -> None:
                if phase == "prescan":
                    pct = 68 + int(10 * done / max(total, 1))
                    _write_progress(
                        "running_pi4",
                        f"Pre-scanning S. cerevisiae binding sites: TF {done}/{total}",
                        pct=pct,
                    )
                else:
                    pct = 78 + int(20 * done / max(total, 1))
                    short = label.split("_")[-3] + "_" + label.split("_")[-2] \
                        if label.count("_") >= 3 else label
                    _write_progress(
                        "running_pi4",
                        f"SNP at binding sites: species {done}/{total} — {short}",
                        pct=pct,
                    )

            from .pi4_snp_binding import estimate_pi4_all_tfs
            estimate_pi4_all_tfs(verbose=False, save=True, progress_callback=_pi4_callback)
            _write_progress("running_pi4", "π₄ SNP binding sites done.", pct=99)
        else:
            _write_progress("running_pi4", "π₄ already cached — skipped.", pct=99)

        _write_progress("done", "All three π estimators generated successfully.", pct=100)

    except Exception as exc:
        _write_progress("error", "", error=str(exc), pct=0)
    finally:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()


# ---------------------------------------------------------------------------
# Public entry point — call once at server start
# ---------------------------------------------------------------------------

def start_generation_if_needed() -> bool:
    """
    Spawn the background generation thread if any CSV is missing and no
    generation is already running.

    Returns True if a new thread was started, False if already done or
    already running.

    Safe to call multiple times — the _thread_lock ensures only one thread
    starts.  The LOCK_FILE guards against double-starts across server
    restarts.
    """
    if all_done():
        return False

    # If a lock file exists from a previous crashed run, clean it up
    # (it means the previous run never finished cleanly)
    current = get_status()
    if LOCK_FILE.exists() and current["status"] not in ("running_pi2", "running_pi3", "running_pi4"):
        LOCK_FILE.unlink(missing_ok=True)

    # Already running
    if current["status"] in ("running_pi2", "running_pi3", "running_pi4"):
        return False

    with _thread_lock:
        # Double-check after acquiring lock
        if all_done():
            return False
        if LOCK_FILE.exists():
            return False

        thread = threading.Thread(target=_generation_worker, daemon=True, name="y1000_generator")
        thread.start()
        return True
