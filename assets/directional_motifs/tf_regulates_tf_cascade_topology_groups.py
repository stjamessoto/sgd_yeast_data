"""
tf_regulates_tf_cascade_topology_groups.py -- numbers the 21 five-family
(k=5) combinations from tf_regulates_tf_cascade.py and groups them by
directed-graph topology: which combos end up with the same number of
directed edges AND the same number of mutual (A->B and B->A both present)
pairs.

WHY THIS IS A SEPARATE FILE
This is one of three sibling topology-group scripts (the others cover
dominant_coregulation_flow.py and duplication_inheritance_order.py). Each
method colors its arrows differently and means something different by
them, so each gets its own file and its own legend rather than one script
with a combined legend that would misrepresent the other two methods'
arrow colors -- this file's purple/blue legend matches
tf_regulates_tf_cascade.py's own ARROW_LEGEND exactly.

WHY THIS IS SEPARATE FROM THE EXISTING GRID
tf_regulates_tf_cascade.py already draws all 21 combos as a grid, sorted
by "how many isolated nodes" but otherwise in no particular numbered
order, and does not report which combos are structurally identical. This
script numbers the 21 combinations once, in a fixed canonical order (the
order itertools.combinations(range(7), 5) produces, 1-indexed), then
buckets the combos by the topology signature (n_isolated_nodes,
n_directed_edges, n_mutual_pairs) -- the same style of bucketing used by
scripts/analysis/motif_k5_topology_patterns.py for the undirected case,
extended with the mutual-pair count.

WHY EDGES RECUR IDENTICALLY ACROSS COMBOS
The direction and weight of an edge between two specific families is a
function of that PAIR alone (real TF-binds-TF binding-site evidence) -- it
does not depend on which other families are also in the 5-subset. So the
edge between, say, F1 and F2 is either present-and-pointing-the-same-way
in EVERY combo that contains both F1 and F2, or absent in all of them.
The only thing that changes combo to combo is WHICH 5 families (and
therefore which of the 21 possible pairs) are present. That is exactly
why grouping by topology is informative: this method has real reciprocal
(mutual) edges, so it is the richest of the three -- 13 distinct topology
groups among the 21 combos, versus only 2 for the other two (ranking-
based) methods.

Run: python assets/directional_motifs/tf_regulates_tf_cascade_topology_groups.py
"""

import sys
from itertools import combinations as icombs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _shared_family_data import load_families, load_pi4

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LEGEND_HANDLES = [
    Line2D([0], [0], color="#8e44ad", lw=2.5,
           label="mutual / feedback  (A→B and B→A both exist)"),
    Line2D([0], [0], color="#2980b9", lw=2.5,
           label="one-way  (A→B only, no edge back)"),
    Patch(facecolor="#c0392b", edgecolor="white", label="connected node (≥1 edge)"),
    Patch(facecolor="#7f8c8d", edgecolor="white", label="isolated node (0 edges)"),
    Patch(facecolor="#eaf2f8", edgecolor="#95a5a6", label="background shade = topology group"),
]

OUT_DIR = Path(__file__).parent
REPO_ROOT = OUT_DIR.parent.parent
OUT_PNG = OUT_DIR / "tf_regulates_tf_cascade_topology_groups.png"

fam_list, M, N = load_families()
_, tf_targets, pi4_tfs = load_pi4()

_PI4_PATH = REPO_ROOT / "y1000plus_data" / "processed" / "pi4_snp_binding_sites.csv"
_df = pd.read_csv(_PI4_PATH, usecols=["tf_name", "target_gene_name"])
_df["tf_name"] = _df["tf_name"].str.upper()
_df["target_gene_name"] = _df["target_gene_name"].str.upper()
_tf_to_target_names = {tf: set(grp["target_gene_name"]) for tf, grp in _df.groupby("tf_name")}


def directed_weight(fam_a, fam_b):
    n = 0
    for ta in fam_a["member_tfs"]:
        targets = _tf_to_target_names.get(ta, set())
        for tb in fam_b["member_tfs"]:
            if tb in targets:
                n += 1
    return n


def build_directed(fams_sel):
    G = nx.DiGraph()
    for f in fams_sel:
        G.add_node(f["idx"])
    for a, b in icombs(fams_sel, 2):
        w_ab, w_ba = directed_weight(a, b), directed_weight(b, a)
        if w_ab:
            G.add_edge(a["idx"], b["idx"], weight=w_ab)
        if w_ba:
            G.add_edge(b["idx"], a["idx"], weight=w_ba)
    return G


# ── number the 21 combinations (fixed canonical order) ─────────────────────
NUMBERED_COMBOS = list(enumerate(icombs(range(len(fam_list)), 5), start=1))
print("Canonical numbering of the 21 five-family combinations:")
for num, combo in NUMBERED_COMBOS:
    label = "+".join(f"F{fam_list[i]['idx']}" for i in combo)
    print(f"  #{num:<3} {label}")

combos = []
for num, combo in NUMBERED_COMBOS:
    fams_sel = [fam_list[i] for i in combo]
    G = build_directed(fams_sel)
    isolated = tuple(sorted(n for n in G.nodes if G.degree(n) == 0))
    mutual = tuple(sorted({tuple(sorted((u, v))) for u, v in G.edges if G.has_edge(v, u)}))
    combos.append({
        "num": num, "label": "+".join(f"F{f['idx']}" for f in fams_sel), "G": G,
        "n_isolated": len(isolated), "n_edges": G.number_of_edges(),
        "n_mutual": len(mutual), "mutual": mutual,
    })

# ── group by topology signature (n_isolated, n_edges, n_mutual) ────────────
buckets = {}
for c in combos:
    key = (c["n_isolated"], c["n_edges"], c["n_mutual"])
    buckets.setdefault(key, []).append(c)
group_ids = {key: gid for gid, key in enumerate(
    sorted(buckets, key=lambda k: (-len(buckets[k]), k)), start=1)}
for c in combos:
    c["group"] = group_ids[(c["n_isolated"], c["n_edges"], c["n_mutual"])]

print(f"\n{len(buckets)} distinct topology group(s) among the 21 combinations "
      f"(same # directed edges AND same # mutual pairs):")
for key, members in sorted(buckets.items(), key=lambda kv: group_ids[kv[0]]):
    n_iso, n_edges, n_mut = key
    gid = group_ids[key]
    nums = ", ".join(f"#{c['num']}" for c in sorted(members, key=lambda c: c["num"]))
    print(f"  Group {gid}: {n_edges} directed edges, {n_mut} mutual pair(s), "
          f"{n_iso} isolated node(s)  ->  {len(members)} combos: {nums}")

# ── grid figure, panels ordered/shaded by topology group ───────────────────
GROUP_PALETTE = [
    "#eaf2f8", "#fdebd0", "#eafaf1", "#f9ebea", "#f4ecf7",
    "#fef9e7", "#e8f8f5", "#fadbd8", "#e8eaf6", "#fff3e0",
]
combos_sorted = sorted(combos, key=lambda c: (c["group"], c["num"]))
fig, axes = plt.subplots(3, 7, figsize=(15.5, 7.6))
axes = axes.flatten()
sizes_by_idx = {f["idx"]: f["size"] for f in fam_list}

for ax, c in zip(axes, combos_sorted):
    G = c["G"]
    pos = nx.circular_layout(G)
    ax.set_facecolor(GROUP_PALETTE[(c["group"] - 1) % len(GROUP_PALETTE)])
    node_sizes = [140 + 5.5 * sizes_by_idx[n] for n in G.nodes]
    node_colors = ["#7f8c8d" if G.degree(n) == 0 else "#c0392b" for n in G.nodes]
    max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    for u, v, d in G.edges(data=True):
        bidir = G.has_edge(v, u)
        rad = 0.15 if bidir else 0.0
        ecolor = "#8e44ad" if bidir else "#2980b9"
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>", color=ecolor, alpha=0.8,
                linewidth=0.5 + 2.6 * (d["weight"] / max_w),
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=13, shrinkB=13, mutation_scale=10,
            ),
        )
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                            node_color=node_colors, edgecolors="white",
                            linewidths=0.8, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: f"F{n}" for n in G.nodes},
                             font_size=6, font_color="white", font_weight="bold")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#95a5a6")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"#{c['num']}  {c['label']}  (group {c['group']})\n"
        f"{c['n_edges']} edges" + (f", {c['n_mutual']} mutual" if c["n_mutual"] else ""),
        fontsize=6.5, pad=2,
    )
for ax in axes[len(combos_sorted):]:
    ax.axis("off")

fig.suptitle(
    "TF-regulates-TF cascade (real binding direction)\n"
    "all 21 combos, numbered and shaded by topology group "
    "(same # directed edges + same # mutual pairs = same shading)",
    fontsize=10, fontweight="bold", y=0.995,
)
fig.legend(handles=LEGEND_HANDLES, loc="lower center", ncol=3, fontsize=8.5,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.035, 1, 0.94])
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nGrid figure saved: {OUT_PNG}")
