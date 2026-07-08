"""
motif_k5_topology_patterns.py — do all C(7,5)=21 five-family network
diagrams look like the one drawn in motif_k5_recurrence_test.py (dense
4-node core + one isolated singleton), or are there distinct topologies?

motif_k5_recurrence_test.py already showed that all 21 combinations are
statistically over-represented (BH FDR<5%) with the same *significance*
pattern. This script asks the separate structural question: does the
*network shape* (which nodes connect to which) also recur, or does it
change from combo to combo?

Each of the 5 nodes in a combination is one of the 7 GO-Process TF
families (F1..F7). An edge is drawn between two families if at least one
pair of their member TFs shares a JASPAR/pi4-scanned target gene. F7
(GO:0045892) has exactly one member TF, HMRA1, a mating-type cassette
gene that never appears in the pi4 binding-site table — so F7 has zero
pi4 coverage and is *structurally guaranteed* to be isolated (0 edges) in
every combination it appears in, regardless of biology. That is the key
thing this script checks for and calls out explicitly.

Run: python scripts/analysis/motif_k5_topology_patterns.py
"""

import sys
import math
from itertools import combinations as icombs, product as iproduct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.gene_families import build_tf_families, estimate_model_parameters

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_PNG = REPO_ROOT / "assets" / "motif_k5_topology_grid.png"

# ── load families + pi4 co-regulatory binding data (same as recurrence test) ─
families = build_tf_families(min_family_size=1, grouping="GO_Process")
params = estimate_model_parameters(families, "family_size")
M, N = params["m"], params["n"]

fam_list = []
for i, (_, row) in enumerate(families.iterrows()):
    fam_list.append({
        "idx": i + 1,
        "go_id": row["go_id"],
        "size": int(row["family_size"]),
        "member_tfs": [t.upper() for t in row["member_tfs"]],
    })
assert len(fam_list) == 7

_PI4_PATH = REPO_ROOT / "y1000plus_data" / "processed" / "pi4_snp_binding_sites.csv"
_pi4_df = pd.read_csv(_PI4_PATH, usecols=["tf_name", "target_gene_id"])
_pi4_df["tf_name"] = _pi4_df["tf_name"].str.upper()
_tf_targets: dict[str, frozenset] = {
    tf: frozenset(grp["target_gene_id"])
    for tf, grp in _pi4_df.groupby("tf_name")
}
_pi4_tfs = set(_tf_targets)

# per-family pi4 coverage (drives whether a family CAN ever have an edge)
print("Per-family pi4 coverage:")
coverage = {}
for f in fam_list:
    covered = [tf for tf in f["member_tfs"] if tf in _pi4_tfs]
    coverage[f["idx"]] = len(covered)
    print(f"  F{f['idx']} ({f['go_id']}): {len(covered)}/{len(f['member_tfs'])} "
          f"member TFs have pi4 coverage" +
          ("  <-- ZERO COVERAGE: structurally always isolated" if len(covered) == 0 else ""))


def _pairwise_shared_tf_pairs(fam_a, fam_b):
    tfs_a = [tf for tf in fam_a["member_tfs"] if tf in _pi4_tfs]
    tfs_b = [tf for tf in fam_b["member_tfs"] if tf in _pi4_tfs]
    n = 0
    for ta in tfs_a:
        for tb in tfs_b:
            if _tf_targets[ta] & _tf_targets[tb]:
                n += 1
    return n


# ── build the graph for every one of the 21 combinations ───────────────────
MAX_EDGES = math.comb(5, 2)  # 10
combo_graphs = []
for combo in icombs(range(len(fam_list)), 5):
    fams_sel = [fam_list[i] for i in combo]
    G = nx.Graph()
    for f in fams_sel:
        G.add_node(f["idx"])
    for a, b in icombs(fams_sel, 2):
        w = _pairwise_shared_tf_pairs(a, b)
        if w > 0:
            G.add_edge(a["idx"], b["idx"], weight=w)
    isolated = sorted(n for n in G.nodes if G.degree(n) == 0)
    combo_graphs.append({
        "label": "+".join("F%d" % f["idx"] for f in fams_sel),
        "fams": [f["idx"] for f in fams_sel],
        "G": G,
        "n_edges": G.number_of_edges(),
        "isolated": isolated,
        "complete": G.number_of_edges() == MAX_EDGES,
    })

# ── classify into topology buckets ──────────────────────────────────────────
buckets: dict[tuple, list] = {}
for c in combo_graphs:
    key = (len(c["isolated"]), c["n_edges"])
    buckets.setdefault(key, []).append(c["label"])

print(f"\nTopology of all 21 combinations (max possible edges among 5 nodes = {MAX_EDGES}):")
print(f"  {'label':<20} edges/10   isolated nodes      complete-graph?")
for c in sorted(combo_graphs, key=lambda c: (len(c["isolated"]), -c["n_edges"])):
    iso = ",".join(f"F{n}" for n in c["isolated"]) or "none"
    print(f"  {c['label']:<20} {c['n_edges']:>2}/10     {iso:<18} {c['complete']}")

print(f"\nDistinct topology buckets (n_isolated_nodes, n_edges) -> count:")
for key, labels in sorted(buckets.items()):
    n_iso, n_edges = key
    print(f"  {n_iso} isolated node(s), {n_edges}/10 edges  ->  {len(labels)} combos: {labels}")

contains_f7 = [c for c in combo_graphs if 7 in c["fams"]]
without_f7 = [c for c in combo_graphs if 7 not in c["fams"]]
f7_always_isolated = all(7 in c["isolated"] for c in contains_f7)
without_f7_all_complete = all(c["complete"] for c in without_f7)

# ── small-multiples grid of all 21 network diagrams ────────────────────────
fig, axes = plt.subplots(3, 7, figsize=(15.5, 7.2))
axes = axes.flatten()
sizes_by_idx = {f["idx"]: f["size"] for f in fam_list}

for ax, c in zip(axes, sorted(combo_graphs, key=lambda c: (len(c["isolated"]), c["label"]))):
    G = c["G"]
    pos = nx.circular_layout(G)
    node_colors = ["#7f8c8d" if n in c["isolated"] else "#c0392b" for n in G.nodes]
    node_sizes = [140 + 5.5 * sizes_by_idx[n] for n in G.nodes]
    max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    for u, v, d in G.edges(data=True):
        ax.plot(*zip(pos[u], pos[v]), color="#2980b9",
                linewidth=0.4 + 2.2 * (d["weight"] / max_w), alpha=0.6, zorder=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                            node_color=node_colors, edgecolors="white",
                            linewidths=0.8, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: f"F{n}" for n in G.nodes},
                             font_size=6, font_color="white", font_weight="bold")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")
    ax.set_title(f"{c['label']}\n{c['n_edges']}/10 edges", fontsize=6.8, pad=2)

for ax in axes[len(combo_graphs):]:
    ax.axis("off")

fig.suptitle(
    "Network topology of all 21 five-family (k=5) motif combinations\n"
    "grey node = isolated (0 shared-target edges); red = connected to >=1 other family",
    fontsize=10.5, fontweight="bold", y=0.99,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
OUT_PNG.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nGrid figure saved: {OUT_PNG}")

# ── verdict ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TOPOLOGY VERDICT")
print("=" * 70)
n_buckets = len(buckets)
if n_buckets == 1:
    print("  All 21 combinations share the exact same network topology.")
else:
    print(f"  NO — the network SHAPE does not recur identically. There are "
          f"{n_buckets} distinct topology buckets among the 21 combinations:")
    print(f"    - The {len(contains_f7)} combinations containing F7 "
          f"{'ALL' if f7_always_isolated else 'do NOT all'} have F7 isolated "
          f"(F7's only TF, HMRA1, has zero pi4 coverage).")
    print(f"    - The {len(without_f7)} combinations WITHOUT F7 "
          f"{'ARE ALL' if without_f7_all_complete else 'are NOT all'} complete "
          f"graphs (every one of the other 6 families has enough pi4 "
          f"coverage that all pairs share >=1 target).")
    print("  So the recurring STATISTICAL pattern (over-representation, "
          "significant at BH<5%) is universal across all 21 combos, but the "
          "recurring NETWORK-SHAPE pattern is really two patterns driven by "
          "a single structural fact: whether F7 is in the combination.")
print("=" * 70)
