"""
combo_topology_groups.py -- numbers the 21 five-family (k=5) combinations
used by tf_regulates_tf_cascade.py, dominant_coregulation_flow.py, and
duplication_inheritance_order.py, and groups them by directed-graph
topology: which combos end up with the same number of directed edges and
the same number of mutual (A->B and B->A both present) pairs.

WHY THIS IS SEPARATE FROM THE EXISTING GRIDS
Each of the three sibling scripts already draws all 21 combos as a grid,
sorted by "how many isolated nodes" but otherwise in no particular
numbered order, and none of them report which combos are structurally
identical. This script numbers the 21 combinations once, in a fixed
canonical order (the order itertools.combinations(range(7), 5) produces,
1-indexed), so "combo #7" means the same 5-family subset across every
method and every run. It then buckets the combos by the topology signature
(n_isolated_nodes, n_directed_edges, n_mutual_pairs) -- the same style of
bucketing used by scripts/analysis/motif_k5_topology_patterns.py for the
undirected case, extended with the mutual-pair count for the directed case.

WHY EDGES RECUR IDENTICALLY ACROSS COMBOS
For all three methods, the direction and weight of an edge between two
specific families is a function of that PAIR alone (real TF-binds-TF
evidence for the cascade method; a fixed genome-wide breadth or pi_i
ranking for the other two) -- it does not depend on which other families
are also in the 5-subset. So the edge between, say, F1 and F2 is either
present-and-pointing-the-same-way in EVERY combo that contains both F1 and
F2, or absent in all of them. The only thing that changes combo to combo
is WHICH 5 families (and therefore which of the 21 possible pairs) are
present. That is exactly why grouping by topology is informative here: it
reveals how much of the 21-combo grid is really just a handful of
recurring shapes, driven by which families happen to be included.

Produces 3 grid images (one per method), each with all 21 panels numbered
and shaded by topology group, plus a console report of the buckets.

Run: python assets/directional_motifs/combo_topology_groups.py
"""

import sys
from itertools import combinations as icombs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _shared_family_data import load_families, load_pi4

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
REPO_ROOT = OUT_DIR.parent.parent

fam_list, M, N = load_families()
_, tf_targets, pi4_tfs = load_pi4()
FAM_BY_IDX = {f["idx"]: f for f in fam_list}
TAXA = [f["idx"] for f in fam_list]

# fixed, canonical numbering of the 21 five-family combinations (1..21),
# shared by all three methods below
NUMBERED_COMBOS = list(enumerate(icombs(range(len(fam_list)), 5), start=1))

print("Canonical numbering of the 21 five-family combinations:")
for num, combo in NUMBERED_COMBOS:
    label = "+".join(f"F{fam_list[i]['idx']}" for i in combo)
    print(f"  #{num:<3} {label}")


# ── shared helper: undirected shared-target TF-pair count (used by two of
#    the three methods, matches _shared_family_data-adjacent sibling scripts) ──
def shared_tf_pairs(fam_a, fam_b):
    tfs_a = [tf for tf in fam_a["member_tfs"] if tf in pi4_tfs]
    tfs_b = [tf for tf in fam_b["member_tfs"] if tf in pi4_tfs]
    n = 0
    for ta in tfs_a:
        for tb in tfs_b:
            if tf_targets[ta] & tf_targets[tb]:
                n += 1
    return n


# ── method A: TF-regulates-TF cascade (real binding direction) ────────────
_PI4_PATH = REPO_ROOT / "y1000plus_data" / "processed" / "pi4_snp_binding_sites.csv"
_df = pd.read_csv(_PI4_PATH, usecols=["tf_name", "target_gene_name"])
_df["tf_name"] = _df["tf_name"].str.upper()
_df["target_gene_name"] = _df["target_gene_name"].str.upper()
_tf_to_target_names = {tf: set(grp["target_gene_name"]) for tf, grp in _df.groupby("tf_name")}


def _directed_weight(fam_a, fam_b):
    n = 0
    for ta in fam_a["member_tfs"]:
        targets = _tf_to_target_names.get(ta, set())
        for tb in fam_b["member_tfs"]:
            if tb in targets:
                n += 1
    return n


def build_cascade(fams_sel):
    G = nx.DiGraph()
    for f in fams_sel:
        G.add_node(f["idx"])
    for a, b in icombs(fams_sel, 2):
        w_ab, w_ba = _directed_weight(a, b), _directed_weight(b, a)
        if w_ab:
            G.add_edge(a["idx"], b["idx"], weight=w_ab)
        if w_ba:
            G.add_edge(b["idx"], a["idx"], weight=w_ba)
    return G


# ── method B: dominant co-regulation flow (breadth ranking) ───────────────
def _regulatory_breadth(fam):
    union = set()
    for tf in fam["member_tfs"]:
        if tf in pi4_tfs:
            union |= tf_targets[tf]
    return len(union)


_breadth = {f["idx"]: _regulatory_breadth(f) for f in fam_list}


def build_dominant_flow(fams_sel):
    G = nx.DiGraph()
    for f in fams_sel:
        G.add_node(f["idx"])
    for a, b in icombs(fams_sel, 2):
        w = shared_tf_pairs(a, b)
        if w == 0:
            continue
        hi, lo = (a, b) if _breadth[a["idx"]] >= _breadth[b["idx"]] else (b, a)
        G.add_edge(hi["idx"], lo["idx"], weight=w)
    return G


# ── method C: duplication/inheritance order (pi_i ranking) ────────────────
_ev_total = sum(f["ev"] for f in fam_list)
_fam_pi = {f["idx"]: f["ev"] / _ev_total for f in fam_list}


def build_duplication_order(fams_sel):
    G = nx.DiGraph()
    for f in fams_sel:
        G.add_node(f["idx"])
    for a, b in icombs(fams_sel, 2):
        w = shared_tf_pairs(a, b)
        if w == 0:
            continue
        hi, lo = (a, b) if _fam_pi[a["idx"]] >= _fam_pi[b["idx"]] else (b, a)
        G.add_edge(hi["idx"], lo["idx"], weight=w)
    return G


METHODS = [
    ("tf_regulates_tf_cascade", build_cascade,
     "TF-regulates-TF cascade (real binding direction)"),
    ("dominant_coregulation_flow", build_dominant_flow,
     "Dominant co-regulation flow (breadth ranking)"),
    ("duplication_inheritance_order", build_duplication_order,
     "Duplication/inheritance order (pi_i ranking)"),
]

GROUP_PALETTE = [
    "#eaf2f8", "#fdebd0", "#eafaf1", "#f9ebea", "#f4ecf7",
    "#fef9e7", "#e8f8f5", "#fadbd8", "#e8eaf6", "#fff3e0",
]


def analyze_method(method_name, build_fn, title):
    combos = []
    for num, combo in NUMBERED_COMBOS:
        fams_sel = [fam_list[i] for i in combo]
        G = build_fn(fams_sel)
        isolated = tuple(sorted(n for n in G.nodes if G.degree(n) == 0))
        mutual = tuple(sorted({tuple(sorted((u, v))) for u, v in G.edges if G.has_edge(v, u)}))
        n_edges = G.number_of_edges()
        label = "+".join(f"F{f['idx']}" for f in fams_sel)
        combos.append({
            "num": num, "label": label, "G": G,
            "n_isolated": len(isolated), "isolated": isolated,
            "n_edges": n_edges, "n_mutual": len(mutual), "mutual": mutual,
        })

    # ── group by topology signature (n_isolated, n_edges, n_mutual) ────────
    buckets = {}
    for c in combos:
        key = (c["n_isolated"], c["n_edges"], c["n_mutual"])
        buckets.setdefault(key, []).append(c)
    group_ids = {key: gid for gid, key in enumerate(
        sorted(buckets, key=lambda k: (-len(buckets[k]), k)), start=1)}
    for c in combos:
        c["group"] = group_ids[(c["n_isolated"], c["n_edges"], c["n_mutual"])]

    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print(f"  {len(buckets)} distinct topology group(s) among the 21 combinations "
          f"(same # directed edges AND same # mutual pairs):")
    for key, members in sorted(buckets.items(), key=lambda kv: group_ids[kv[0]]):
        n_iso, n_edges, n_mut = key
        gid = group_ids[key]
        nums = ", ".join(f"#{c['num']}" for c in sorted(members, key=lambda c: c["num"]))
        print(f"    Group {gid}: {n_edges} directed edges, {n_mut} mutual pair(s), "
              f"{n_iso} isolated node(s)  ->  {len(members)} combos: {nums}")

    # ── grid figure, panels ordered/shaded by topology group ───────────────
    combos_sorted = sorted(combos, key=lambda c: (c["group"], c["num"]))
    fig, axes = plt.subplots(3, 7, figsize=(15.5, 7.6))
    axes = axes.flatten()
    sizes_by_idx = {f["idx"]: f["size"] for f in fam_list}

    for ax, c in zip(axes, combos_sorted):
        G = c["G"]
        pos = nx.circular_layout(G)
        color = GROUP_PALETTE[(c["group"] - 1) % len(GROUP_PALETTE)]
        ax.set_facecolor(color)
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
        f"{title}\nall 21 combos, numbered and shaded by topology group "
        f"(same # directed edges + same # mutual pairs = same shading)",
        fontsize=10, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = OUT_DIR / f"{method_name}_topology_groups.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Grid figure saved: {out_png}")
    return combos, buckets, group_ids


all_results = {}
for method_name, build_fn, title in METHODS:
    all_results[method_name] = analyze_method(method_name, build_fn, title)

print("\n" + "=" * 78)
print("  CROSS-METHOD SUMMARY")
print("=" * 78)
for method_name, _, title in METHODS:
    _, buckets, _ = all_results[method_name]
    print(f"  {title}: {len(buckets)} distinct topology group(s)")
print("  Methods B and C (breadth ranking / pi_i ranking) direct every edge "
      "using a single fixed total order over the 7 families, so neither can "
      "ever produce a mutual pair -- their topology groups are driven purely "
      "by which families are isolated / how many pairs share a target, "
      "identical in shape to the undirected motif_k5_topology_patterns.py "
      "buckets. Method A (real TF-regulates-TF binding) is the only one "
      "where mutual pairs are structurally possible, so it is the only "
      "method where the mutual-pair count can separate combos that would "
      "otherwise land in the same bucket.")
print("=" * 78)
