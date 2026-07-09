"""
dominant_coregulation_flow.py — direction = which family "dominates" a
shared co-regulatory edge.

The shared-target edges used in scripts/analysis/motif_k5_*.py (family A
and family B both regulate some common third gene C) are symmetric by
construction: sharing a target does not say who leads. This script adds a
direction on top of those same edges using an asymmetry that IS
well-defined: total regulatory breadth — the number of distinct target
genes reached, genome-wide, by all pi4-covered TFs in a family (a union
of target sets, from the same pi4_snp_binding_sites.csv table).

For every existing shared-target edge (A, B), the arrow points from the
family with the LARGER total regulatory breadth to the one with the
SMALLER breadth: "the more prolific regulator dominates the connection."
This is a heuristic ranking, not a causal claim — it never changes WHICH
family-pairs are connected (that's still the undirected shared-target
graph), only how the connection is drawn.

Produces the same two outputs as tf_regulates_tf_cascade.py: a grid of
all 21 five-family combinations, plus one whole-7-family diagram, so it
is directly comparable.

Run: python assets/directional_motifs/dominant_coregulation_flow.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LEGEND_HANDLES = [
    Line2D([0], [0], color="#27ae60", lw=2.5,
           label="higher-breadth family → lower-breadth family"),
    Patch(facecolor="#c0392b", edgecolor="white", label="connected (≥1 shared-target edge)"),
    Patch(facecolor="#7f8c8d", edgecolor="white", label="isolated (no shared-target edge)"),
]

OUT_DIR = Path(__file__).parent
OUT_GRID_PNG = OUT_DIR / "dominant_coregulation_flow_grid.png"
OUT_WHOLE_PNG = OUT_DIR / "dominant_coregulation_flow_whole_network.png"

fam_list, M, N = load_families()
_, tf_targets, pi4_tfs = load_pi4()


def shared_tf_pairs(fam_a, fam_b):
    tfs_a = [tf for tf in fam_a["member_tfs"] if tf in pi4_tfs]
    tfs_b = [tf for tf in fam_b["member_tfs"] if tf in pi4_tfs]
    n = 0
    for ta in tfs_a:
        for tb in tfs_b:
            if tf_targets[ta] & tf_targets[tb]:
                n += 1
    return n


def regulatory_breadth(fam):
    """# of distinct target genes reached by any pi4-covered TF in this family."""
    union = set()
    for tf in fam["member_tfs"]:
        if tf in pi4_tfs:
            union |= tf_targets[tf]
    return len(union)


breadth = {f["idx"]: regulatory_breadth(f) for f in fam_list}
print("Regulatory breadth per family (# distinct target genes, genome-wide):")
for f in sorted(fam_list, key=lambda f: -breadth[f["idx"]]):
    print(f"  F{f['idx']} ({f['go_id']}): breadth={breadth[f['idx']]}")


def build_directed(fams_sel):
    G = nx.DiGraph()
    for f in fams_sel:
        G.add_node(f["idx"])
    for a, b in icombs(fams_sel, 2):
        w = shared_tf_pairs(a, b)
        if w == 0:
            continue
        hi, lo = (a, b) if breadth[a["idx"]] >= breadth[b["idx"]] else (b, a)
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
                arrowstyle="-|>", color="#27ae60", alpha=0.75,
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


# ── whole-network figure (all 7 families) ──────────────────────────────────
WG = build_directed(fam_list)
fig1, ax1 = plt.subplots(figsize=(6.4, 6.4))
draw_directed(ax1, WG, node_size_scale=6.5, base_node=260)
ax1.set_title(
    "Dominant co-regulation flow — all 7 GO-Process families\n"
    "width = # shared-target TF pairs",
    fontsize=9.5,
)
ax1.legend(handles=LEGEND_HANDLES, loc="lower center", fontsize=8,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.1))
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
    "Dominant co-regulation flow in all 21 five-family (k=5) motif combinations",
    fontsize=10.5, fontweight="bold", y=0.99,
)
fig2.legend(handles=LEGEND_HANDLES, loc="lower center", ncol=3, fontsize=9,
            framealpha=0.9, bbox_to_anchor=(0.5, -0.015))
fig2.tight_layout(rect=[0, 0.03, 1, 0.94])
fig2.savefig(OUT_GRID_PNG, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Grid figure saved: {OUT_GRID_PNG}")

print("\nNote: breadth ranking is F" + ", F".join(
    str(f["idx"]) for f in sorted(fam_list, key=lambda f: -breadth[f["idx"]])
) + " (highest to lowest) — since this ranking never changes between combos, "
    "every diagram's arrows point 'inward' toward the same low-breadth families "
    "(F6, F7) whenever they're present, and otherwise follow that same fixed order. "
    "This is a much more repetitive/predictable pattern than the TF-regulates-TF "
    "cascade, precisely because it is derived from a single fixed per-family "
    "ranking rather than pairwise regulatory evidence.")
