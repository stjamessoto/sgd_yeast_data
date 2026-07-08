"""
_shared_family_data.py — common loader used by the three directional-motif
scripts in this folder (tf_regulates_tf_cascade.py, dominant_coregulation_flow.py,
duplication_inheritance_order.py). Not a standalone analysis itself.

Loads the same 7 GO-Process TF families and pi4 JASPAR binding-site table
used by scripts/analysis/motif_k5_recurrence_test.py and
scripts/analysis/motif_k5_topology_patterns.py, so all direction variants
are directly comparable to those undirected results.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from model.gene_families import build_tf_families, estimate_model_parameters

GO_NAMES = {
    "GO:0006355": "Regulation of transcription, DNA-templated",
    "GO:0045944": "Positive regulation of transcription by RNA Pol II",
    "GO:0006357": "Regulation of transcription by RNA Pol II",
    "GO:0000122": "Negative regulation of transcription by RNA Pol II",
    "GO:0006351": "Transcription by RNA Pol II",
    "GO:0045893": "Positive regulation of transcription, DNA-templated",
    "GO:0045892": "Negative regulation of transcription, DNA-templated",
}


def load_families():
    families = build_tf_families(min_family_size=1, grouping="GO_Process")
    params = estimate_model_parameters(families, "family_size")
    fam_list = []
    for i, (_, row) in enumerate(families.iterrows()):
        fam_list.append({
            "idx": i + 1,
            "go_id": row["go_id"],
            "size": int(row["family_size"]),
            "ev": float(row["mean_evidence_score"]),
            "name": GO_NAMES.get(row["go_id"], row["go_id"]),
            "member_tfs": [t.upper() for t in row["member_tfs"]],
        })
    assert len(fam_list) == 7
    return fam_list, params["m"], params["n"]


def load_pi4():
    """Return (df, tf->targetset dict, set of TF names with coverage)."""
    path = REPO_ROOT / "y1000plus_data" / "processed" / "pi4_snp_binding_sites.csv"
    df = pd.read_csv(path, usecols=["tf_name", "target_gene_id", "target_gene_name"])
    df["tf_name"] = df["tf_name"].str.upper()
    df["target_gene_name"] = df["target_gene_name"].str.upper()
    tf_targets = {
        tf: frozenset(grp["target_gene_id"])
        for tf, grp in df.groupby("tf_name")
    }
    return df, tf_targets, set(tf_targets)
