"""
duplication_inheritance_order.py — direction = inheritance-probability
order from the Scruse et al. duplication model.

IMPORTANT CAVEAT (read before trusting this one): this repo does not
contain literal whole-genome-duplication / ohnolog ancestry data (no
"gene X is the ancestral copy of gene Y" table). So this script cannot
show true duplication ancestry between genes. What it CAN do, grounded in
data already in the pipeline, is order the 7 families by their per-family
inheritance probability pi_i (Method 1, evidence-based, normalized per
Scruse et al. Eq. 3 exactly as in scripts/analysis/_generate_doc.py) and
treat that as a proxy for "how much of the ancestral regulatory link this
family has retained since duplication." Arrows point from HIGH-pi
(strong retention, evidence-based) to LOW-pi (weaker retention) families,
for every pair that already has a shared-target edge (same undirected
edges as motif_k5_topology_patterns.py).

This is the weakest-grounded of the three directional scripts by design:
pi_i is a single scalar per family, so the resulting order is IDENTICAL
in every combination that shares two given families -- unlike the
TF-regulates-TF cascade (real pairwise data) or the co-regulation-breadth
version (a different real pairwise asymmetry). Treat this script's output
as illustrative of the model's ranking, not as literal duplication order.

Run: python assets/directional_motifs/duplication_inheritance_order.py
"""

import sys
from itertools import combinations as icombs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _shared_family_data import load_families, load_pi4

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
OUT_GRID_PNG = OUT_DIR / "duplication_inheritance_order_grid.png"
OUT_WHOLE_PNG = OUT_DIR / "duplication_inheritance_order_whole_network.png"

fam_list, M, N = load_families()
_, tf_targets, pi4_tfs = load_pi4()

# Method 1 (Scruse et al. Eq. 3): pi_i = evidence_i / sum(evidence)
_ev_total = sum(f["ev"] for f in fam_list)
fam_pi = {f["idx"]: f["ev"] / _ev_total for f in fam_list}
print("Per-family inheritance probability pi_i (Method 1, evidence-based):")
for f in sorted(fam_list, key=lambda f: -fam_pi[f["idx"]]):
    print(f"  F{f['idx']} ({f['go_id']}): pi_i={fam_pi[f['idx']]:.4f}  (ev={f['ev']:.3f})")


def shared_tf_pairs(fam_a, fam_b):
    tfs_a = [tf for tf in fam_a["member_tfs"] if tf in pi4_tfs]
    tfs_b = [tf for tf in fam_b["member_tfs"] if tf in pi4_tfs]
    n = 0
    for ta in tfs_a:
        for tb in tfs_b:
            if tf_targets[ta] & tf_targets[tb]:
                n += 1
    return n


def build_directed(fams_sel):
    G = nx.DiGraph()
    for f in fams_sel:
        G.add_node(f["idx"])
    for a, b in icombs(fams_sel, 2):
        w = shared_tf_pairs(a, b)
        if w == 0:
            continue
        hi, lo = (a, b) if fam_pi[a["idx"]] >= fam_pi[b["idx"]] else (b, a)
        G.add_edge(hi["idx"], lo["idx"], weight=w)
    return G


def draw_directed(ax, G, node_size_scale=5.5, base_node=140):
    pos = nx.circular_layout(G)
    sizes_by_idx = {f["idx"]: f["size"] for f in fam_list}
    node_sizes = [base_node + node_size_scale * sizes_by_idx[n] for n in G.nodes]
    isolated = [n for n in G.nodes if G.degree(n) == 0]
    node_colors = ["#7f8c8d" if n in isolated else "#c0392b" for n in G.nodes]
    max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    for u, v, d in G.edges(data=True):
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>", color="#e67e22", alpha=0.8,
                linewidth=0.5 + 2.6 * (d["weight"] / max_w),
                shrinkA=13, shrinkB=13, mutation_scale=11,
            ),
        )
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                            node_color=node_colors, edgecolors="white",
                            linewidths=0.8, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: f"F{n}" for n in G.nodes},
                             font_size=7, font_color="white", font_weight="bold")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")


# ── whole-network figure ────────────────────────────────────────────────────
WG = build_directed(fam_list)
fig1, ax1 = plt.subplots(figsize=(6.4, 6.4))
draw_directed(ax1, WG, node_size_scale=6.5, base_node=260)
ax1.set_title(
    "Duplication/inheritance-order proxy — all 7 GO-Process families\n"
    "arrow: higher pi_i (evidence-based retention) -> lower pi_i  ·  "
    "PROXY ONLY, not literal ohnolog ancestry (see module docstring)",
    fontsize=8.8,
)
fig1.tight_layout()
fig1.savefig(OUT_WHOLE_PNG, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"\nWhole-network figure saved: {OUT_WHOLE_PNG}")

# ── all 21 five-family combos ────────────────────────────────────────────
combo_graphs = []
for combo in icombs(range(len(fam_list)), 5):
    fams_sel = [fam_list[i] for i in combo]
    G = build_directed(fams_sel)
    label = "+".join("F%d" % f["idx"] for f in fams_sel)
    isolated = sorted(n for n in G.nodes if G.degree(n) == 0)
    combo_graphs.append({"label": label, "G": G, "isolated": isolated,
                          "n_edges": G.number_of_edges()})

fig2, axes = plt.subplots(3, 7, figsize=(15.5, 7.2))
axes = axes.flatten()
for ax, c in zip(axes, sorted(combo_graphs, key=lambda c: (len(c["isolated"]), c["label"]))):
    draw_directed(ax, c["G"])
    ax.set_title(f"{c['label']}\n{c['n_edges']}/10 directed edges", fontsize=6.5, pad=2)
for ax in axes[len(combo_graphs):]:
    ax.axis("off")
fig2.suptitle(
    "Duplication/inheritance-order proxy in all 21 five-family (k=5) motif combinations\n"
    "arrow: higher pi_i -> lower pi_i, on the same shared-target edges  ·  PROXY ONLY",
    fontsize=10.2, fontweight="bold", y=0.99,
)
fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig(OUT_GRID_PNG, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Grid figure saved: {OUT_GRID_PNG}")

pi_rank = [f["idx"] for f in sorted(fam_list, key=lambda f: -fam_pi[f["idx"]])]
print(f"\nFixed pi_i rank (highest to lowest): F{', F'.join(map(str, pi_rank))}")
print("Because pi_i is a single scalar per family, this fixed rank means every "
      "diagram's arrows always point in the same relative order regardless of "
      "which 5 families are selected -- the direction pattern is guaranteed to "
      "'recur' trivially, which is exactly why this proxy is weaker evidence "
      "than the TF-regulates-TF cascade script.")
