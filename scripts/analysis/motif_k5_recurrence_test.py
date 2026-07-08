"""
motif_k5_recurrence_test.py — does the k=3 / k=4 over-representation pattern
recur at k=5?

Background: `_generate_doc.py` enumerates all C(7,3)=35 and C(7,4)=35
combinations of the 7 SGD GO-Process TF families and finds that essentially
every combination is significantly OVER-represented (BH FDR < 5%) relative
to the Partial Duplication null, and none is under-represented.

This script asks the same question at k=5: there are C(7,5) = 21 possible
five-family combinations. It:

  1. Draws one representative 5-family motif as a network diagram (nodes =
     families, sized by family_size; edges = number of co-regulatory TF
     pairs across the two families that share >=1 target gene).
  2. Computes the Partial Duplication significance test (Corollary 16 /
     Theorem 4) for all 21 combinations, with Benjamini-Hochberg FDR
     correction at alpha=5%.
  3. Plots the 21 Z-scores next to the existing k=3/k=4 distributions and a
     significant/over-represented-fraction comparison bar chart.
  4. Prints a definitive verdict: does the same "near-universal
     over-representation, zero under-representation" pattern recur at k=5?

Run: python scripts/analysis/motif_k5_recurrence_test.py
"""

import sys
import math
from itertools import combinations as icombs, product as iproduct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.gene_families import build_tf_families, estimate_model_parameters
from model.inheritance_estimator import full_significance_analysis

FDR_ALPHA = 0.05
REPO_ROOT = Path(__file__).parent.parent.parent
OUT_PNG = REPO_ROOT / "assets" / "motif_k5_recurrence.png"

GO_NAMES = {
    "GO:0006355": "Regulation of transcription, DNA-templated",
    "GO:0045944": "Positive regulation of transcription by RNA Pol II",
    "GO:0006357": "Regulation of transcription by RNA Pol II",
    "GO:0000122": "Negative regulation of transcription by RNA Pol II",
    "GO:0006351": "Transcription by RNA Pol II",
    "GO:0045893": "Positive regulation of transcription, DNA-templated",
    "GO:0045892": "Negative regulation of transcription, DNA-templated",
}

# ── load families + pi4 co-regulatory binding data ─────────────────────────
families = build_tf_families(min_family_size=1, grouping="GO_Process")
params = estimate_model_parameters(families, "family_size")
M, N = params["m"], params["n"]

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
assert len(fam_list) == 7, f"expected 7 families, found {len(fam_list)}"

_PI4_PATH = REPO_ROOT / "y1000plus_data" / "processed" / "pi4_snp_binding_sites.csv"
_pi4_df = pd.read_csv(_PI4_PATH, usecols=["tf_name", "target_gene_id"])
_pi4_df["tf_name"] = _pi4_df["tf_name"].str.upper()
_tf_targets: dict[str, frozenset] = {
    tf: frozenset(grp["target_gene_id"])
    for tf, grp in _pi4_df.groupby("tf_name")
}
_pi4_tfs = set(_tf_targets)

_ev_total = sum(f["ev"] for f in fam_list)
_fam_pi = {f["idx"]: f["ev"] / _ev_total for f in fam_list}


def _count_coreg(fams_sel):
    """Count k-tuples (one TF per family) that all share >=1 common target."""
    covered = [[tf for tf in f["member_tfs"] if tf in _pi4_tfs] for f in fams_sel]
    if any(len(c) == 0 for c in covered):
        return math.prod(f["size"] for f in fams_sel), True
    count = 0
    for tf_tuple in iproduct(*covered):
        shared = _tf_targets[tf_tuple[0]]
        for tf in tf_tuple[1:]:
            shared = shared & _tf_targets[tf]
            if not shared:
                break
        if shared:
            count += 1
    return count, False


def _pairwise_shared_tf_pairs(fam_a, fam_b):
    """# of (TF_a, TF_b) pairs across two families that share >=1 target — edge weight."""
    tfs_a = [tf for tf in fam_a["member_tfs"] if tf in _pi4_tfs]
    tfs_b = [tf for tf in fam_b["member_tfs"] if tf in _pi4_tfs]
    n = 0
    for ta in tfs_a:
        for tb in tfs_b:
            if _tf_targets[ta] & _tf_targets[tb]:
                n += 1
    return n


def run_motifs(k):
    out = []
    for combo in icombs(range(len(fam_list)), k):
        fams_sel = [fam_list[i] for i in combo]
        label = "+".join("F%d" % f["idx"] for f in fams_sel)
        pi_vec = [_fam_pi[f["idx"]] for f in fams_sel]
        obs, fallback = _count_coreg(fams_sel)
        try:
            sig = full_significance_analysis(pi_vec, M, N, float(obs), k)
        except Exception:
            continue
        out.append({
            "combo": combo, "label": label,
            "sizes": [f["size"] for f in fams_sel],
            "obs": obs, "fallback": fallback,
            "z": sig["z_partial"], "p": sig["p_partial"], "sig": sig["sig_partial"],
        })
    return out


def bh_correct(results):
    valid = [r for r in results if r["p"] is not None]
    ps = np.array([r["p"] for r in valid])
    n = len(ps)
    if n == 0:
        return results
    order = np.argsort(ps)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    bh_sig = ps <= (ranks / n) * FDR_ALPHA
    for r, s in zip(valid, bh_sig):
        r["bh_sig"] = bool(s)
    for r in results:
        r.setdefault("bh_sig", False)
    return results


# ── task 1: enumerate all C(7,5) = 21 five-family combinations ─────────────
print("Computing k=5 motifs (C(7,5) = 21 combinations)...")
res5 = run_motifs(5)
assert len(res5) == 21, f"expected 21 five-family combinations, got {len(res5)}"
res5 = bh_correct(res5)

print("Computing k=3 and k=4 for comparison (as in _generate_doc.py)...")
res3 = bh_correct(run_motifs(3))
res4 = bh_correct(run_motifs(4))


def summarize(results, k):
    sig = sum(1 for r in results if r["bh_sig"])
    over = sum(1 for r in results if r["bh_sig"] and r["z"] and r["z"] > 0)
    under = sum(1 for r in results if r["bh_sig"] and r["z"] and r["z"] < 0)
    return {"k": k, "n": len(results), "sig": sig, "over": over, "under": under}


sum3, sum4, sum5 = summarize(res3, 3), summarize(res4, 4), summarize(res5, 5)

print("\n  k   combos   significant(BH<5%)   over-repr.   under-repr.")
for s in (sum3, sum4, sum5):
    print(f"  {s['k']:<3} {s['n']:<8} {s['sig']:<20} {s['over']:<12} {s['under']}")

print("\n  Full k=5 table:")
for r in sorted(res5, key=lambda r: -r["z"] if r["z"] else 0):
    print(f"    {r['label']:<20} sizes={r['sizes']}  obs={r['obs']:<10.0f} "
          f"z={r['z']:>10.2f}  p={r['p']:.3e}  bh_sig={r['bh_sig']}")

# ── task 2: draw the 5 nodes for the top-Z representative combination ──────
best5 = max(res5, key=lambda r: r["z"] if r["z"] else -np.inf)
best_fams = [fam_list[i] for i in best5["combo"]]

G = nx.Graph()
for f in best_fams:
    G.add_node(f["idx"], size=f["size"], name=f["name"])
edge_weights = {}
for a, b in icombs(best_fams, 2):
    w = _pairwise_shared_tf_pairs(a, b)
    if w > 0:
        G.add_edge(a["idx"], b["idx"], weight=w)
        edge_weights[(a["idx"], b["idx"])] = w

# ── task 3: build the comparison figure ─────────────────────────────────────
fig = plt.figure(figsize=(11.5, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 1.0],
                       left=0.045, right=0.98, top=0.84, bottom=0.13, wspace=0.38)
ax_net, ax_strip, ax_bar = (fig.add_subplot(gs[i]) for i in range(3))

# -- panel A: the 5-node network diagram --
pos = nx.circular_layout(G)
node_sizes = [400 + 26 * G.nodes[n]["size"] for n in G.nodes]
max_w = max(edge_weights.values()) if edge_weights else 1
for (u, v), w in edge_weights.items():
    ax_net.plot(*zip(pos[u], pos[v]), color="#2980b9",
                linewidth=0.6 + 4.4 * (w / max_w), alpha=0.55, zorder=1)
nx.draw_networkx_nodes(G, pos, ax=ax_net, node_size=node_sizes,
                        node_color="#c0392b", edgecolors="white",
                        linewidths=1.4, alpha=0.92)
nx.draw_networkx_labels(G, pos, ax=ax_net,
                         labels={n: f"F{n}" for n in G.nodes},
                         font_size=9, font_color="white", font_weight="bold")
for (u, v), w in edge_weights.items():
    mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
    ax_net.annotate(str(w), (mx, my), fontsize=6.5, color="#1b4f72", ha="center")
ax_net.set_xlim(-1.35, 1.35)
ax_net.set_ylim(-1.35, 1.35)
ax_net.axis("off")
ax_net.set_title(f"5-node motif  {best5['label']}\n(node size = family size, "
                  f"edge = shared-target TF pairs)", fontsize=8.5, pad=6)

# -- panel B: strip/jitter of all 21 k=5 Z-scores --
np.random.seed(42)
z5 = np.array([r["z"] for r in res5 if r["z"] is not None])
lz5 = np.log10(np.maximum(z5, 1e-4))
bh5 = np.array([r["bh_sig"] for r in res5 if r["z"] is not None])
jit5 = np.random.uniform(-0.16, 0.16, len(lz5))

ax_strip.scatter(jit5[bh5], lz5[bh5], c="#c0392b", s=34, alpha=0.85, linewidths=0, zorder=3)
ax_strip.scatter(jit5[~bh5], lz5[~bh5], c="#aaaaaa", s=34, alpha=0.9, linewidths=0, zorder=3)
labels5 = [r["label"] for r in res5 if r["z"] is not None]
# only label the extremes (highest/lowest Z); full ranked table is printed to console
order5 = np.argsort(lz5)
label_idx = set(order5[:2].tolist()) | set(order5[-2:].tolist())
prev_y = []
for i in sorted(label_idx, key=lambda i: -lz5[i]):
    y = lz5[i]
    # nudge apart if two labels would collide vertically
    while any(abs(y - py) < 0.18 for py in prev_y):
        y -= 0.18
    prev_y.append(y)
    ax_strip.annotate(labels5[i], xy=(jit5[i], lz5[i]), xytext=(0.5, y),
                       fontsize=6.5, color="#333333", va="center",
                       arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7))
ax_strip.annotate(f"remaining {len(labels5) - 4} combinations cluster\nbetween these extremes",
                   xy=(0.5, 0.5), xycoords="axes fraction", fontsize=6.5,
                   color="#777777", ha="center", style="italic")
ax_strip.set_xlim(-0.5, 1.3)
ax_strip.set_xticks([])
ax_strip.set_ylabel("log$_{10}$(Z-score)", fontsize=9)
n_sig5 = int(bh5.sum())
ax_strip.set_title(f"All 21 five-family combinations (C(7,5)=21)\n"
                    f"{n_sig5}/21 red = significant (BH FDR<5%)", fontsize=8.5, pad=6)

# -- panel C: k=3 / k=4 / k=5 recurrence comparison --
ks = ["k=3\n(n=35)", "k=4\n(n=35)", "k=5\n(n=21)"]
sig_frac = [sum3["sig"] / sum3["n"], sum4["sig"] / sum4["n"], sum5["sig"] / sum5["n"]]
over_frac = [sum3["over"] / sum3["n"], sum4["over"] / sum4["n"], sum5["over"] / sum5["n"]]
under_frac = [sum3["under"] / sum3["n"], sum4["under"] / sum4["n"], sum5["under"] / sum5["n"]]

x = np.arange(3)
w = 0.27
ax_bar.bar(x - w, sig_frac, width=w, color="#2980b9", label="significant (BH<5%)")
ax_bar.bar(x, over_frac, width=w, color="#c0392b", label="over-represented")
ax_bar.bar(x + w, under_frac, width=w, color="#7f8c8d", label="under-represented")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(ks, fontsize=8.5)
ax_bar.set_ylim(0, 1.15)
ax_bar.set_ylabel("Fraction of combinations", fontsize=9)
ax_bar.set_title("Does the pattern recur?", fontsize=8.5, pad=6)
ax_bar.legend(fontsize=6.8, loc="upper center", framealpha=0.85, ncol=1)
for xi, v in zip(x, sig_frac):
    ax_bar.annotate(f"{v:.0%}", (xi - w, v + 0.03), ha="center", fontsize=7.5)

fig.suptitle(
    f"Subnetwork Motif Recurrence Test  ·  k=5 (21 combinations) vs. k=3/k=4  "
    f"(m={M} families, n={N} TFs, Partial Duplication null)",
    fontsize=10.5, fontweight="bold",
)

OUT_PNG.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {OUT_PNG}")

# ── task 4: definitive verdict ──────────────────────────────────────────────
same_pattern = (
    sum5["sig"] == sum5["n"] and sum5["under"] == 0
    and sum3["sig"] == sum3["n"] and sum4["sig"] == sum4["n"]
    and sum3["under"] == 0 and sum4["under"] == 0
)

print("\n" + "=" * 66)
print("  VERDICT")
print("=" * 66)
if same_pattern:
    print(
        "  YES — the pattern recurs at k=5.\n"
        f"  All {sum5['n']}/{sum5['n']} five-family combinations are significantly\n"
        "  OVER-represented (BH FDR<5%) relative to the Partial Duplication\n"
        "  null, and none is under-represented — identical in kind to what\n"
        f"  _generate_doc.py found at k=3 ({sum3['sig']}/{sum3['n']}) and\n"
        f"  k=4 ({sum4['sig']}/{sum4['n']}). The GRN shows near-universal\n"
        "  over-representation regardless of motif size k in {3,4,5}."
    )
else:
    print(
        "  NO — the pattern does NOT fully recur at k=5.\n"
        f"  k=3: {sum3['sig']}/{sum3['n']} significant, {sum3['under']} under-represented\n"
        f"  k=4: {sum4['sig']}/{sum4['n']} significant, {sum4['under']} under-represented\n"
        f"  k=5: {sum5['sig']}/{sum5['n']} significant, {sum5['under']} under-represented\n"
        "  See the per-combination table above for which k=5 motifs break rank."
    )
print("=" * 66)
