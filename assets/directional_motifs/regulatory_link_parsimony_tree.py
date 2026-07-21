"""
regulatory_link_parsimony_tree.py -- maximum-parsimony phylogeny of the 7
GO-Process TF families, built from binary regulatory-link characters.

METHOD
Each character (column) is one pi4-covered target gene. A family scores 1
for that character if ANY of its member TFs binds that target (pi4 SNP
binding-site evidence), 0 otherwise -- exactly the discrete presence/
absence coding phylogenetics normally starts from (gene presence/absence,
morphological traits, etc.), here applied to "does this family regulate
this target." Columns are compressed to their unique 0/1 pattern across
the 7 families, weighted by how many targets share that pattern (site-
pattern compression -- 7 taxa admit at most 2**7 = 128 distinct patterns
no matter how many thousand raw target columns exist).

TREE SEARCH: exhaustive maximum parsimony. With only 7 taxa, all
(2*7-5)!! = 945 distinct unrooted binary trees can be built (by successive
taxon addition on every edge) and scored exactly with Fitch's algorithm
(Fitch 1971) -- no heuristic search, no external phylogenetics package.
The minimum-total-changes tree (ties reported if any) is the maximum-
parsimony estimate of which families' regulatory-link profiles are most
alike, and in what branching order.

ROOTING / "AGE": parsimony scoring itself is unrooted (change counts don't
depend on root placement). For a directional "what came first" reading,
the tree needs an outgroup, and one is available from the GO terms
themselves: of the 7 families, 6 are all "regulation of transcription"
(positive, negative, or generic control -- GO:0006355, :0045944, :0006357,
:0000122, :0045893, :0045892), and one, F5 (GO:0006351, "transcription by
RNA polymerase II"), is core transcription itself -- the actual synthesis
of RNA, not control over whether/how much of it happens. Under the RNA
World hypothesis (Robertson and Joyce 2012, "The Origins of the RNA
World," Cold Spring Harb Perspect Biol 4:a003608), RNA-based function --
replication and catalysis carried out by RNA itself -- is understood to
have preceded the elaborated protein-based regulatory machinery (TFs,
combinatorial control, activation/repression) built up around it later.
That gives a principled, non-arbitrary root: F5, the family representing
the core RNA-synthesis process, is used as the outgroup, and the tree is
rooted on F5's own branch (split at its midpoint, the standard convention
for a single-taxon outgroup). The other 6 (regulatory-control) families
are then read as diverging FROM that transcription-centered state, using
per-branch character-change counts (from a Fitch down-pass, i.e.
ACCTRAN-style state assignment) as branch lengths. This is a thematic
anchor, not literal ancestry -- F5 is an extant GO category over modern
S. cerevisiae genes, not a fossil or reconstructed ancestral sequence, and
"age" here means accumulated regulatory-link change relative to that
anchor, not a calibrated divergence time.

CAVEAT: like duplication_inheritance_order.py, this is model output on
7 pre-defined GO-Process TF groupings, not literal evolutionary history --
no fossil calibration, no gene-tree/species-tree reconciliation, no
molecular clock, no real ancestral sequence. It answers "given only which
targets each family's member TFs bind, what is the best-supported
branching pattern and how much has each branch diverged" -- nothing more.

TASKS still needed to harden this into a real phylogenetic result:
  1. Re-code characters at the individual-TF (or individual-gene) level
     instead of pooling by GO-Process family -- family-level OR-pooling
     can hide within-family conflict between member TFs.
  2. Add bootstrap support: resample characters with replacement, rerun
     the exhaustive MP search per replicate, report the % of replicates
     recovering each bipartition (branch confidence -- not computed here).
  3. If multiple equally-parsimonious trees are found, build their strict
     consensus tree instead of arbitrarily displaying the first one.
  4. Compare against a model-based tree (ML/Bayesian, e.g. a simple
     Camin-Sokal or Dollo model for irreversible link loss) once an
     explicit substitution-model assumption is chosen -- unweighted
     parsimony has no explicit model and can be misled by long-branch
     attraction.
  5. F5 (core RNA-Pol-II transcription) is used as a thematically-motivated
     root (see ROOTING above), not a literal ancestor. Bring in real
     WGD/ohnolog ancestry (Y1000plus or SGD paralog tables) to root against
     a genuine phylogenetic outgroup instead.
  6. Sensitivity check: does the MP tree change if characters are coded
     from a different pi4 evidence threshold, or from raw JASPAR motif
     hits instead of SNP-derived binding sites?

Run: python assets/directional_motifs/regulatory_link_parsimony_tree.py
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _shared_family_data import load_families, load_pi4

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
OUT_TREE_PNG = OUT_DIR / "regulatory_link_parsimony_tree.png"
OUT_MATRIX_PNG = OUT_DIR / "regulatory_link_parsimony_matrix.png"

fam_list, M, N = load_families()
_, tf_targets, pi4_tfs = load_pi4()

TAXA = [f["idx"] for f in fam_list]          # [1..7]
FAM_BY_IDX = {f["idx"]: f for f in fam_list}
N_TAXA = len(TAXA)


# ── 1. raw binary character matrix: family x target (1 = family has a
#      member TF, with pi4 coverage, that binds this target) ──────────────
all_targets = sorted(set().union(*(tf_targets[tf] for tf in pi4_tfs)))
target_index = {t: i for i, t in enumerate(all_targets)}

raw = np.zeros((N_TAXA, len(all_targets)), dtype=bool)
for row, f in enumerate(fam_list):
    member_targets = set()
    for tf in f["member_tfs"]:
        if tf in pi4_tfs:
            member_targets |= tf_targets[tf]
    for t in member_targets:
        raw[row, target_index[t]] = True

print(f"Raw character matrix: {raw.shape[0]} families x {raw.shape[1]} targets "
      f"(1 = family has a member TF regulating that target)")

for row, f in enumerate(fam_list):
    n_covered = sum(1 for tf in f["member_tfs"] if tf in pi4_tfs)
    if n_covered == 0:
        print(f"  WARNING: F{f['idx']} ({f['go_id']}) has 0 of its "
              f"{len(f['member_tfs'])} member TF(s) with pi4 coverage -> its "
              f"character vector is all-zero. Its branch placement below is "
              f"driven entirely by the OTHER families' patterns, not by any "
              f"F{f['idx']}-specific regulatory-link evidence -- treat its "
              f"position as unsupported, not a real divergence signal.")

# ── 2. compress to unique column patterns (site-pattern compression) ──────
pattern_counts = defaultdict(int)
for col in raw.T:
    pattern_counts[tuple(col.tolist())] += 1

pattern_list = []          # list of (bool array len N_TAXA, weight)
n_constant = 0
for pat, w in pattern_counts.items():
    ones = sum(pat)
    if ones == 0 or ones == N_TAXA:
        n_constant += w
        continue
    pattern_list.append((np.array(pat, dtype=bool), w))

n_informative = sum(w for p, w in pattern_list if 1 < int(p.sum()) < N_TAXA - 1)
n_autapomorphy = sum(w for p, w in pattern_list) - n_informative
print(f"{len(pattern_list)} unique variable patterns "
      f"({len(pattern_counts)} unique patterns total, {n_constant} constant "
      f"targets dropped as uninformative)")
print(f"  -> {n_informative} targets are parsimony-informative (>=2 families "
      f"differ on each side), {n_autapomorphy} are single-family autapomorphies")

pattern_matrix = np.array([p for p, _ in pattern_list])   # (n_patterns, N_TAXA)
weights = np.array([w for _, w in pattern_list], dtype=float)
leaf_vec = {TAXA[i]: pattern_matrix[:, i] for i in range(N_TAXA)}

# family x family weighted Hamming distance (diagnostic only, not used by
# the tree search itself)
dist_matrix = np.zeros((N_TAXA, N_TAXA))
for i in range(N_TAXA):
    for j in range(N_TAXA):
        dist_matrix[i, j] = sum(w for p, w in pattern_list if bool(p[i]) != bool(p[j]))
print("\nWeighted regulatory-link Hamming distance between families:")
header = "      " + "".join(f"F{TAXA[j]:<6}" for j in range(N_TAXA))
print(header)
for i in range(N_TAXA):
    print(f"  F{TAXA[i]} " + "".join(f"{dist_matrix[i, j]:<7.0f}" for j in range(N_TAXA)))


# ── 3. enumerate all unrooted binary trees over the 7 taxa (successive
#      taxon addition -- generates exactly (2n-5)!! trees, no duplicates) ──
def enumerate_trees(taxa):
    t0, t1, t2 = taxa[:3]
    trees = [([(t0, "I0"), (t1, "I0"), (t2, "I0")], 1)]
    for taxon in taxa[3:]:
        new_trees = []
        for edges, next_id in trees:
            for edge in edges:
                u, v = edge
                new_node = f"I{next_id}"
                new_edges = [e for e in edges if e != edge]
                new_edges += [(u, new_node), (v, new_node), (taxon, new_node)]
                new_trees.append((new_edges, next_id + 1))
        trees = new_trees
    return [edges for edges, _ in trees]


def build_adj(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def postorder(adj, root):
    order = []

    def dfs(node, parent):
        for nb in adj[node]:
            if nb != parent:
                dfs(nb, node)
        order.append((node, parent))

    dfs(root, None)
    return order


def fitch_score(adj, root, leaf_vec, weights):
    """Vectorized Fitch parsimony up-pass. Returns (total_cost, can0, can1, postorder)."""
    order = postorder(adj, root)
    can0, can1 = {}, {}
    total_cost = np.zeros(len(weights))
    for node, parent in order:
        if node in leaf_vec:
            v = leaf_vec[node]
            can0[node], can1[node] = ~v, v
            continue
        children = [nb for nb in adj[node] if nb != parent]
        c0, c1 = can0[children[0]].copy(), can1[children[0]].copy()
        for ch in children[1:]:
            i0, i1 = c0 & can0[ch], c1 & can1[ch]
            has = i0 | i1
            total_cost += np.where(has, 0.0, weights)
            c0 = np.where(has, i0, c0 | can0[ch])
            c1 = np.where(has, i1, c1 | can1[ch])
        can0[node], can1[node] = c0, c1
    return total_cost.sum(), can0, can1, order


ALL_TREES = enumerate_trees(TAXA)
# 'I0' is the internal node created by enumerate_trees' base case (the first
# 3-taxon star) and every subsequent taxon-insertion step preserves it, so
# it exists in all 945 generated trees. Rooting at an internal (free/
# unconstrained) node is required for Fitch's up-pass sum to equal the true
# minimum change count -- rooting at a leaf instead undercounts, because a
# leaf's OBSERVED state is fixed and must be reconciled against its
# neighbor's ambiguity set, which a simple pass-through never does.
ROOT_FOR_SEARCH = "I0"

all_scores = []
best_score = np.inf
best_trees = []
for edges in ALL_TREES:
    adj = build_adj(edges)
    score, *_ = fitch_score(adj, ROOT_FOR_SEARCH, leaf_vec, weights)
    all_scores.append(score)
    if score < best_score - 1e-9:
        best_score, best_trees = score, [edges]
    elif abs(score - best_score) < 1e-9:
        best_trees.append(edges)

all_scores = np.array(all_scores)
print(f"\nExhaustive search over {len(ALL_TREES)} unrooted binary trees complete.")
print(f"  score range across all trees: min={all_scores.min():.0f}  "
      f"mean={all_scores.mean():.1f}  max={all_scores.max():.0f}")
print(f"  minimum parsimony score (best tree): {best_score:.0f} character-state changes")
print(f"  equally-parsimonious (tied) trees found: {len(best_trees)}"
      + ("  (displaying the first; see TASKS item 3 re: strict consensus)"
         if len(best_trees) > 1 else ""))

best_edges = best_trees[0]
best_adj = build_adj(best_edges)
best_score_val, _, _, _ = fitch_score(best_adj, ROOT_FOR_SEARCH, leaf_vec, weights)

# For state RECONSTRUCTION (down-pass), root on an edge instead of on I0.
# I0 has degree 3 in every generated tree, so rooting straight at it makes
# the root a 3-way polytomy; the up-pass's pairwise-sequential reduction
# still sums to the correct total cost at a polytomy (verified above), but
# the can0/can1 SET it leaves at that node is not reliable for a down-pass
# reconstruction (it silently assumes a hidden binary sub-order among the
# 3 children). Splitting an arbitrary edge with a synthetic 2-child root
# keeps every node in the rooted computation strictly bifurcating, which a
# Fitch down-pass requires to actually realize the minimum change count.
_u0, _v0 = best_edges[0]
_split_edges = [e for e in best_edges if e != (_u0, _v0)] + [(_u0, "ROOTSPLIT"), ("ROOTSPLIT", _v0)]
_split_adj = build_adj(_split_edges)
_, can0, can1, order = fitch_score(_split_adj, "ROOTSPLIT", leaf_vec, weights)


# ── 4. Fitch down-pass: assign one state per node, derive per-branch
#      change counts (used as branch lengths) ─────────────────────────────
def assign_states(adj, can0, can1, order, leaf_vec):
    preorder = list(reversed(order))       # root comes first
    state = {}
    for node, parent in preorder:
        if node in leaf_vec:
            state[node] = leaf_vec[node].astype(np.int8)
            continue
        node_c0, node_c1 = can0[node], can1[node]
        own_choice = np.where(node_c1, 1, 0).astype(np.int8)
        if parent is None:
            # root has no parent state to reconcile against -- it's a free
            # (unobserved, internal) node, so just take any state from its
            # own Fitch set.
            state[node] = own_choice
            continue
        pstate = state[parent]
        matches_parent = np.where(pstate == 1, node_c1, node_c0)
        state[node] = np.where(matches_parent, pstate, own_choice)
    return state


state = assign_states(_split_adj, can0, can1, order, leaf_vec)

edge_changes = {}
for node, parent in reversed(order):
    if parent is None:
        continue
    diff = state[node] != state[parent]
    edge_changes[(parent, node)] = float((diff * weights).sum())

# merge the two synthetic ROOTSPLIT half-edges back into the one real edge
# (_u0, _v0) they were split from, so the reported tree matches best_edges.
w_u = edge_changes.pop(("ROOTSPLIT", _u0), edge_changes.pop((_u0, "ROOTSPLIT"), None))
w_v = edge_changes.pop(("ROOTSPLIT", _v0), edge_changes.pop((_v0, "ROOTSPLIT"), None))
edge_changes[(_u0, _v0)] = (w_u or 0.0) + (w_v or 0.0)

changes_total = sum(edge_changes.values())
print(f"\nDown-pass branch-length sum: {changes_total:.0f} "
      f"(sanity check vs. up-pass score {best_score_val:.0f})")


# ── 5. root on F5's own branch (RNA World-motivated outgroup rooting) ─────
Gw = nx.Graph()
for (u, v), w in edge_changes.items():
    Gw.add_edge(u, v, weight=w)

OUTGROUP = next(t for t in TAXA if FAM_BY_IDX[t]["go_id"] == "GO:0006351")
_og_neighbor = next(iter(Gw.neighbors(OUTGROUP)))
_og_branch_len = Gw[OUTGROUP][_og_neighbor]["weight"]

if _og_branch_len > 0:
    Gr = Gw.copy()
    Gr.remove_edge(OUTGROUP, _og_neighbor)
    Gr.add_edge(OUTGROUP, "ROOT", weight=_og_branch_len / 2)
    Gr.add_edge("ROOT", _og_neighbor, weight=_og_branch_len / 2)
    root_node = "ROOT"
else:
    Gr = Gw
    root_node = OUTGROUP

diam = max(nx.shortest_path_length(Gr, root_node, n, weight="weight") for n in TAXA)

print(f"Root placed on F{OUTGROUP}'s ({FAM_BY_IDX[OUTGROUP]['go_id']}, "
      f"\"{FAM_BY_IDX[OUTGROUP]['name']}\") own branch (split at its midpoint, "
      f"branch length {_og_branch_len:.0f} changes) -- an RNA World-motivated "
      f"outgroup choice, not a midpoint-of-diameter geometric one; see module "
      f"docstring.")


# ── 6. simple phylogram layout (depth = cumulative weighted changes from
#      the midpoint root; x-order = DFS leaf order) ────────────────────────
def layout_tree(G, root):
    adj = {n: list(G.neighbors(n)) for n in G.nodes}
    parent = {root: None}
    depth = {root: 0.0}
    order_ = []

    def dfs(node):
        for nb in adj[node]:
            if nb == parent.get(node):
                continue
            if nb in parent:
                continue
            parent[nb] = node
            depth[nb] = depth[node] + G[node][nb]["weight"]
            dfs(nb)
        order_.append(node)

    dfs(root)
    leaf_order = [n for n in order_ if n in TAXA]
    ypos = {n: i for i, n in enumerate(leaf_order)}
    for node in order_:
        if node not in ypos:
            children = [nb for nb in adj[node] if parent.get(nb) == node]
            ypos[node] = sum(ypos[c] for c in children) / len(children)
    return ypos, depth, parent, leaf_order


ypos, depth, parent, leaf_order = layout_tree(Gr, root_node)
root_y = ypos[root_node]

fig1, ax1 = plt.subplots(figsize=(9.5, 6))
for node, p in parent.items():
    if p is None:
        continue
    ax1.plot([depth[p], depth[node]], [ypos[p], ypos[node]],
             color="#2c3e50", lw=1.8, solid_capstyle="round")
for node in Gr.nodes:
    is_leaf = node in TAXA
    is_outgroup = node == OUTGROUP
    ax1.scatter([depth[node]], [ypos[node]],
                s=110 if is_outgroup else (90 if is_leaf else 28),
                color="#2980b9" if is_outgroup else ("#c0392b" if is_leaf else "#7f8c8d"),
                edgecolor="white", linewidths=0.8, zorder=3)
for leaf in leaf_order:
    fam = FAM_BY_IDX[leaf]
    tag = "  <- RNA World outgroup" if leaf == OUTGROUP else ""
    ax1.text(depth[leaf] + diam * 0.02, ypos[leaf],
              f"F{leaf}  {fam['name']}{tag}", va="center", fontsize=8)

# root marker: label placed in the empty space above the tree, connected to
# the actual root point with a thin line, so it never collides with the
# x-axis tick labels or the F5 leaf's own text below it.
ax1.axvline(0, color="#bdc3c7", lw=0.8, ls="--", zorder=0)
ax1.annotate(
    "root (F5 branch midpoint)",
    xy=(0, root_y), xycoords="data",
    xytext=(diam * 0.05, len(leaf_order) - 0.35), textcoords="data",
    fontsize=7.5, color="#7f8c8d", ha="left", va="top",
    arrowprops=dict(arrowstyle="-", color="#95a5a6", lw=0.7,
                     connectionstyle="arc3,rad=-0.2"),
)

ax1.set_xlim(-diam * 0.02, diam * 1.28)
ax1.set_ylim(-0.6, len(leaf_order) - 0.1)
ax1.set_xlabel("Divergence from the RNA-synthesis (F5) outgroup root "
               "(weighted regulatory-link character changes)")
ax1.set_yticks([])
for spine in ("top", "right", "left"):
    ax1.spines[spine].set_visible(False)
ax1.set_title(
    "Maximum-parsimony tree of 7 GO-Process TF families\n"
    f"binary regulatory-link characters (pi4 target coverage)  --  score {best_score:.0f} changes\n"
    "rooted on F5 (core RNA-Pol-II transcription) per the RNA World hypothesis  --  "
    "PROXY, not a dated phylogeny",
    fontsize=9.5, pad=14, linespacing=1.6,
)
source_text = (
    "Data source: pi4 SNP-derived TF binding-site predictions "
    "(y1000plus_data/processed/pi4_snp_binding_sites.csv)  |  "
    "TF families: GO-Process groupings (model/gene_families.py)"
)
fig1.text(0.5, 0.005, source_text, ha="center", va="bottom", fontsize=6.5,
           color="#555555",
           bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f7",
                      edgecolor="#bbbbbb", linewidth=0.6))
fig1.tight_layout(rect=[0, 0.035, 1, 1])
fig1.savefig(OUT_TREE_PNG, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"\nTree figure saved: {OUT_TREE_PNG}")


# ── 7. character-matrix heatmap (top patterns by weight) ──────────────────
TOP_N = 30
sorted_patterns = sorted(pattern_list, key=lambda pw: -pw[1])[:TOP_N]
mat = np.array([p for p, _ in sorted_patterns]).T          # (N_TAXA, n_shown)
wts = [w for _, w in sorted_patterns]

fig2, (ax_mat, ax_bar) = plt.subplots(
    2, 1, figsize=(11, 5.2), sharex=True, layout="constrained",
    gridspec_kw={"height_ratios": [N_TAXA, 2.2], "hspace": 0.06},
)
ax_mat.imshow(mat, aspect="auto", cmap="Greys", vmin=0, vmax=1)
ax_mat.set_yticks(range(N_TAXA))
ax_mat.set_yticklabels([f"F{t}" for t in TAXA], fontsize=8)
ax_mat.set_xticks([])
ax_mat.set_title(
    f"Top {len(sorted_patterns)} most common regulatory-link character "
    f"patterns (of {len(pattern_list)} unique) -- 1 = family regulates "
    f"targets with this pattern, 0 = not",
    fontsize=9.2,
)
ax_bar.bar(range(len(wts)), wts, color="#2c3e50", width=0.7)
ax_bar.set_ylabel("# targets\nsharing pattern", fontsize=7.5)
ax_bar.set_xlabel("character pattern (sorted by frequency)", fontsize=8)
fig2.savefig(OUT_MATRIX_PNG, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Matrix figure saved: {OUT_MATRIX_PNG}")
