"""
app.py — Streamlit frontend for the Scruse et al. (2024) inheritance probability model.

Tabs (left → right as they appear in the UI):
  0. Overview               — app map; one-card summary of every tab
  1. Introduction           — plain-language guide: what the model does and how to navigate the app
  2. Methodology            — mathematical framework; Theorems 1–8, Pólya urn, Full vs Partial Duplication
  3. TF Explorer            — browse TFs, binding sites, consensus sequences, regulatory targets
  4. Gene Families          — family size distribution and Pólya urn parameters
  5. π Estimator            — estimate inheritance probability vector four ways
  6. Motif Significance     — test whether a k-motif is over/under-represented
  7. Y1000+ π Estimators   — cross-species π₂, π₃, π₄ from 1,154 yeast genomes
  8. Method Estimation Test — validate all 7 methods + ensemble against known π profiles via MSE
  9. Glossary & References  — term definitions and primary citations

Run:  streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import time
import io
import base64
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent))

from model.data_loader import (
    load_transcription_factors,
    load_tf_go_annotations,
    load_go_annotations_full,
    load_inheritance_vectors,
    load_snps,
    load_chromosome_lengths,
    dataset_summary,
    EVIDENCE_QUALITY,
)
from model.tf_network import (
    get_transcription_factors,
    build_tf_target_map,
    describe_binding_sites,
    network_statistics,
    go_term_label,
)
from model.consensus_loader import (
    load_consensus_data,
    load_tf_consensus_stats,
    get_consensus_for_tf,
    list_yeastract_tfs,
)
from model.jaspar_loader import (
    load_jaspar_tfbs,
    jaspar_summary,
    get_jaspar_info,
    get_jaspar_pfm,
)
from model.gene_families import (
    build_tf_families,
    estimate_model_parameters,
    family_size_distribution,
    select_motif_families,
    count_observed_motif_instances,
)
from model.inheritance_estimator import (
    estimate_pi_from_evidence,
    estimate_pi_from_mle,
    estimate_pi_from_snp,
    estimate_pi_consensus_adjusted,
    estimate_pi_all_methods,
    estimate_pi_per_family_ensemble,
    full_significance_analysis,
    pi_sensitivity,
)
from model.scruse_math import (
    expected_full,
    expected_partial,
    variance_full,
    variance_binary,
    f_func,
    g_func,
    expected_bounds_partial,
    second_moment_binary,
    estimate_pi_hat,
)

# ---------------------------------------------------------------------------
# Sequence logo renderer (logomaker + matplotlib → PNG bytes)
# ---------------------------------------------------------------------------

def _render_sequence_logo(pfm_df: "pd.DataFrame", tf_name: str, matrix_id: str) -> "io.BytesIO | None":
    """Convert a long-format PFM dataframe into an IC sequence logo PNG."""
    try:
        import logomaker
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pfm_wide = (
            pfm_df.pivot(index="position", columns="nucleotide", values="count")
            .fillna(0)
        )
        # Ensure A C G T column order
        for base in ["A", "C", "G", "T"]:
            if base not in pfm_wide.columns:
                pfm_wide[base] = 0.0
        pfm_wide = pfm_wide[["A", "C", "G", "T"]]

        freq = pfm_wide.div(pfm_wide.sum(axis=1), axis=0)
        ic_matrix = logomaker.transform_matrix(freq, from_type="probability", to_type="information")

        width = max(5, len(ic_matrix) * 0.55)
        fig, ax = plt.subplots(figsize=(width, 2.2))
        logomaker.Logo(ic_matrix, ax=ax, color_scheme="classic", show_spines=False)
        ax.set_xlabel("Position", fontsize=9)
        ax.set_ylabel("Bits", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.suptitle(f"{tf_name}  ·  {matrix_id}", fontsize=9, y=1.02)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Y1000+ background generation — fires once per server start
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _launch_y1000_generation():
    """
    Called once when the Streamlit server starts (cache_resource is server-wide).
    Spawns a daemon thread to generate the three cross-species π CSVs if any
    are missing. Does nothing if all CSVs already exist.
    """
    try:
        from model.y1000plus_generator import start_generation_if_needed
        return start_generation_if_needed()
    except Exception:
        return False

_launch_y1000_generation()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Yeast GRN Inheritance Model",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.metric-card {background:#f0f2f6;border-radius:8px;padding:12px 16px;margin:4px;}
.highlight {color:#d62728;font-weight:600;}

/* Theorem boxes — light card with dark equation area */
.theorem-box {
    background: #f5f8ff;
    border-left: 4px solid #2563eb;
    border: 1px solid #c7d9ff;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 10px 0;
    color: #1e293b;
}
.theorem-box b {
    color: #1d4ed8;
    font-size: 1.02em;
    display: block;
    margin-bottom: 6px;
}
.theorem-box pre, .formula {
    background: #1e293b;
    color: #93c5fd;
    padding: 8px 12px;
    border-radius: 4px;
    margin: 8px 0 4px 0;
    font-size: 0.83em;
    white-space: pre-wrap;
    word-break: break-all;
    line-height: 1.5;
}

.result-box {background:#e8f5e9;border-left:4px solid #2ca02c;
             padding:10px 14px;border-radius:4px;margin:8px 0;color:#1e293b;}
.warn-box   {background:#fff3e0;border-left:4px solid #ff7f0e;
             padding:10px 14px;border-radius:4px;margin:8px 0;color:#1e293b;}

/* Intro tab cards */
.intro-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
}
.intro-card h4 { margin: 0 0 6px 0; color: #1d4ed8; }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached heavy computations
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading SGD data…")
def _load_tfs():
    return get_transcription_factors()

@st.cache_data(show_spinner="Building regulatory network (may take ~30 s)…")
def _load_tf_target_map(max_tfs, min_ev):
    return build_tf_target_map(max_tfs=max_tfs, min_evidence_score=min_ev)

@st.cache_data(show_spinner="Building TF gene families…")
def _load_tf_families(min_size, grouping="GO_Process"):
    return build_tf_families(min_family_size=min_size, grouping=grouping)

@st.cache_data
def _dataset_summary():
    return dataset_summary()

@st.cache_data
def _load_inheritance():
    return load_inheritance_vectors()

@st.cache_data
def _load_snps():
    return load_snps()


# ---------------------------------------------------------------------------
# π interpretation helper
# ---------------------------------------------------------------------------

def _pi_interpretation(pi_hat: float, k: int, m: int, n: int) -> str:
    """Return a plain-language interpretation block for a π̂ estimate."""
    pi_hat = float(pi_hat) if pi_hat is not None else 0.0
    per_avg = pi_hat / k if k > 0 else pi_hat
    try:
        exp_p = expected_partial(pi_hat, m, n)
        exp_p = 0.0 if (exp_p is None or np.isnan(exp_p) or np.isinf(exp_p)) else exp_p
    except Exception:
        exp_p = 0.0
    try:
        exp_f = expected_full(k, m, n)
        exp_f = 0.0 if (exp_f is None or np.isnan(exp_f) or np.isinf(exp_f)) else exp_f
    except Exception:
        exp_f = 0.0
    ratio_pct = (exp_p / exp_f * 100) if exp_f > 0 else 0

    if per_avg >= 0.80:
        level = "very high"
        biology = (
            "regulatory links are strongly preserved after duplication. "
            "This is consistent with **positive selection** maintaining these TF binding "
            "sites in duplicated copies — likely because the binding motif is highly specific "
            "(high information-content positions) and any mutation disrupts binding."
        )
        why = (
            "High-IC, narrow binding motifs (e.g., zinc-finger TFs with 6-12 bp specific "
            "consensus) leave little room for neutral drift. Both copies of the TF are "
            "constrained to keep the same regulatory wiring, so π stays near 1."
        )
    elif per_avg >= 0.50:
        level = "moderately high"
        biology = (
            "most regulatory links survive duplication. "
            "This suggests **subfunctionalization** — both duplicated copies retain most "
            "of their original connections, with partial loss at some sites."
        )
        why = (
            "Intermediate π arises when TF binding specificity is moderate. Some promoter "
            "sites diverge after duplication (neutral drift or relaxed constraint) while "
            "conserved pathway-core sites are retained by purifying selection."
        )
    elif per_avg >= 0.25:
        level = "moderate"
        biology = (
            "roughly half of regulatory links are lost after duplication. "
            "This is a hallmark of **partial subfunctionalization**, where duplicated TFs "
            "divide up regulatory responsibilities between them."
        )
        why = (
            "Moderate π often reflects degeneracy in the binding motif (ambiguous IUPAC "
            "positions). Degenerate sites are easier to gain or lose by point mutation, "
            "so one paralog tends to drift away from a subset of targets."
        )
    else:
        level = "low"
        biology = (
            "most regulatory links are not inherited after duplication. "
            "This points toward **neofunctionalization or regulatory rewiring** — "
            "the duplicated copy has diverged substantially in its regulatory role."
        )
        why = (
            "Very short or degenerate binding sequences are nearly identical to background "
            "sequence, so new sites arise and old ones are lost by random mutation. "
            "Alternatively, the duplicated TF may have acquired a new DNA-binding "
            "specificity through mutations in its binding domain."
        )

    return (
        f"**Estimated π̂ = {pi_hat:.4f}** (mean per-family πᵢ ≈ {per_avg:.4f}, "
        f"~{per_avg*100:.0f}% of links inherited on average)\n\n"
        f"This is a **{level}** inheritance probability: {biology}\n\n"
        f"**Model output (m = {m}, n = {n}, k = {k}):** "
        f"Partial Duplication expects **{exp_p:.2f}** motif instances (Theorem 4) vs "
        f"Full Duplication upper bound of **{exp_f:.2f}** (Theorem 1) — "
        f"the data is ~{ratio_pct:.0f}% of the Full Duplication expectation.\n\n"
        f"**Why does the data look this way?** {why}"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("assets/subnetwork_motifs.png", use_container_width=True)
    st.title("GRN Inheritance Model")
    st.caption("Scruse, Arnold & Robinson (2026)")
    st.divider()

    st.subheader("⚙️ Global settings")
    min_ev = st.slider(
        "Min. evidence score filter", 0.0, 1.0, 0.0, 0.05,
        help="Filter out TFs whose evidence score is below this threshold.",
    )
    max_tfs_net = st.slider(
        "Max TFs for network build", 10, 200, 100, 10,
        help="Limits TFs used to build the regulatory network (performance).",
    )
    min_family_size = st.slider(
        "Min gene-family size", 1, 20, 1,
        help="Minimum is 1. The calculations assume the initial number in a family is 1 (each family starts as a singleton before duplication).",
    )
    family_grouping = st.selectbox(
        "Family grouping method",
        [
            "GO Biological Process",
            "GO Molecular Function",
            "GO Cellular Component",
            "JASPAR TF Class",
            "JASPAR TF Family",
        ],
        index=0,
        help=(
            "**GO Biological Process** (default) — groups TFs by shared biological pathway "
            "(paper Section 2). **GO Molecular Function** — groups by shared molecular "
            "activity / binding-domain type; closer to true sequence paralogy. "
            "**GO Cellular Component** — groups by subcellular location (nucleus, complex, …). "
            "**JASPAR TF Class / Family** — groups by DNA-binding domain architecture "
            "(protein-sequence similarity proxy; 12 classes, 177 TFs)."
        ),
    )
    _GROUPING_KEY = {
        "GO Biological Process": "GO_Process",
        "GO Molecular Function": "GO_Function",
        "GO Cellular Component": "GO_Component",
        "JASPAR TF Class":       "JASPAR_Class",
        "JASPAR TF Family":      "JASPAR_Family",
    }[family_grouping]

    st.divider()
    summary = _dataset_summary()
    st.metric("TFs loaded", summary["n_tfs"])
    st.metric(
        "GO annotations",
        f"{summary['n_go_annotations']:,}",
        help=(
            "**Why so many?**\n\n"
            "SGD stores one record per (gene, GO term, evidence code) triple — "
            "not just one record per gene. A single TF can have dozens of GO terms "
            "(biological process, molecular function, cellular component), and each term "
            "may be supported by several evidence codes (IDA, IMP, IEA, …), each counted "
            "separately. With ~6,000 yeast genes and ~120 GO annotations on average per "
            "well-studied TF, totals of 100,000+ records are normal and expected."
        ),
    )
    st.metric("Genome size", f"{summary['genome_size_mb']} Mb")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab0, tab2, tab3, tab4, tab5, tab6, tab7, tab9, tab8 = st.tabs([
    "📋 Overview",
    "📘 Introduction",
    "📊 Methodology",
    "🔬 TF Explorer",
    "👨‍👩‍👧 Gene Families",
    "🎲 π Estimator",
    "🧪 Motif Significance",
    "🌍 Y1000+ π Estimators",
    "🧮 Method Estimation Test",
    "📖 Glossary & References",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 0: Introduction
# ══════════════════════════════════════════════════════════════════════

with tab0:
    st.header("Welcome — What This App Does")
    st.markdown(
        "_Scruse, Arnold & Robinson (2026) · Bull. Math. Biol. 88, 31 · University of Georgia_"
    )
    st.markdown("""
    This app implements a mathematical model for studying **how gene regulatory networks
    (GRNs) evolve after gene duplication** in brewer's yeast (*Saccharomyces cerevisiae*).
    It draws on **179 transcription factors** from four curated sources:
    [JASPAR 2024](https://jaspar.elixir.no/) (177 TFs with experimentally validated
    position frequency matrices), [YEASTRACT](https://www.yeastract.com/) (127 curated TFs
    with consensus binding sequences), gene annotations from the
    [Saccharomyces Genome Database (SGD)](https://www.yeastgenome.org/), and cross-species
    conservation data from [Y1000+](https://doi.org/10.1016/j.cell.2023.11.016)
    (1,154 yeast genome assemblies used to estimate π₂, π₃, and π₄).
    """)

    st.divider()
    st.subheader("🧬 The Biological Question")
    st.markdown("""
    Genes duplicate. When they do, the new copy inherits DNA — but does it also inherit
    its **regulatory connections**? A transcription factor (TF) controls a gene by binding
    a short sequence in its promoter called a **binding site** (TFBS). After duplication:

    - The **original gene** keeps its TF binding sites.
    - The **new copy** may or may not retain those same binding sites.

    The probability that a regulatory link is retained is called **π (pi)** —
    the *inheritance probability*. This model estimates π from real yeast data and
    tests whether observed regulatory patterns are more (or less) common than expected
    by chance under a duplication model.
    """)

    st.divider()
    st.subheader("📐 The Mathematical Model")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Subnetwork motif** — a pattern of TFs from *k* different gene families
        all co-regulating the same target. The model counts how many such patterns
        exist in the GRN and asks: *is this count surprising?*

        **Full Duplication (π = 1)** — every duplication perfectly copies all
        regulatory links. Expected motif count grows as **Θ(nᵏ)**.

        **Partial Duplication (0 ≤ π ≤ 1)** — each link is independently
        inherited with probability π. Expected count grows as **Θ(n^π̂)**
        where π̂ = Σπᵢ is the total inheritance probability.
        """)
    with col_b:
        st.markdown("""
        **Why it matters** — if the observed motif count is much higher than the
        Partial Duplication expectation, it suggests **positive selection** for
        that regulatory wiring (it is being actively maintained). If lower, it
        suggests the pattern is being lost or avoided.

        **Pólya urn connection** — the gene duplication process is mathematically
        equivalent to drawing balls from a Pólya urn. Family size proportions
        converge to a Dirichlet distribution, giving the model its exact
        probabilistic grounding.
        """)

    st.divider()
    st.subheader("🗂️ How to Use This App")
    st.caption(
        "The tabs follow a natural analysis pipeline — start left, move right. "
        "Jump to any tab directly if you know what you need."
    )

    _htabs = [
        ("📋", "Overview",
         "A card-by-card map of every tab: what it does, what data it uses, and "
         "what question it answers. Good first stop if you are unsure where to look."),
        ("📊", "Methodology",
         "The mathematical backbone — Theorems 1–8, the Pólya urn model, Full vs "
         "Partial Duplication, and the significance-testing framework. No data required."),
        ("🔬", "TF Explorer",
         "Browse all 179 JASPAR + YEASTRACT transcription factors. Select any TF "
         "to see its JASPAR PFM, IUPAC consensus sequences, information content, "
         "GO annotations, and regulatory targets."),
        ("👨‍👩‍👧", "Gene Families",
         "Group genes into paralog families using five methods (GO Process/Function/"
         "Component, JASPAR TF Class/Family). Shows family-size distributions and "
         "derives the model parameters m, n, and d = n − m."),
        ("🎲", "π Estimator",
         "Select k families to form a subnetwork motif and estimate the inheritance "
         "probability vector $\\vec{\\pi}$ using seven complementary methods: four SGD-based "
         "(evidence codes, Moment Estimation via Theorem 4, SNP divergence at YFL039C, "
         "YEASTRACT binding flexibility) and three Y1000+ cross-species methods "
         "(π₂ sequence homology, π₃ TFBS conservation, π₄ IC-weighted SNP at binding sites)."),
        ("🧪", "Motif Significance",
         "Full significance test: compare the observed motif count against Full and "
         "Partial Duplication null models. Outputs Z-scores, p-values, and a "
         "predictive forward forecast of motif count growth."),
        ("🌍", "Y1000+ π Estimators",
         "Three cross-species estimators using 1,154 yeast genomes: "
         "π₂ (protein sequence identity), π₃ (TFBS conservation via PWM scanning), "
         "π₄ (IC-weighted SNP rate at binding site positions). "
         "Data is generated automatically in the background on first launch."),
        ("🧮", "Method Estimation Test",
         "Benchmark all seven estimation methods plus the multi-signal ensemble against "
         "known synthetic true-π profiles (Linear and Quadratic). Each method's MSE is "
         "computed analytically; Method 2 additionally shows a 1,000-replica Poisson "
         "simulation to capture count-observation noise. A **Compare All Methods** "
         "expander runs all methods side-by-side and ranks them by average MSE."),
        ("📖", "Glossary & References",
         "Definitions of all mathematical terms (π, m, n, d, k-motif, Pólya urn, …) "
         "and full citations for the paper, datasets, and databases. "
         "Good reference if you encounter unfamiliar notation anywhere in the app."),
    ]

    for i in range(0, len(_htabs), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(_htabs):
                icon, name, desc = _htabs[i + j]
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{icon} {name}**")
                        st.markdown(desc)

    st.divider()
    st.subheader("📚 Data Sources")
    _js = jaspar_summary()
    src_c1, src_c2, src_c3, src_c4 = st.columns(4)
    with src_c1:
        st.markdown(f"""
        **JASPAR 2024** *(jaspar.elixir.no)*
        - **{_js['n_jaspar_tfs']} TFs** with PFMs
        - {_js['n_chip_based']} ChIP · {_js['n_pbm_based']} PBM
        - Mean width: {_js['mean_motif_width']} bp
        - Mean IC: {_js['mean_ic_bits']} bits
        """)
    with src_c2:
        st.markdown("""
        **YEASTRACT** *(yeastract.com)*
        - 127 curated *S. cerevisiae* TFs
        - IUPAC consensus sequences
        - 115 TFs overlap with JASPAR
        """)
    with src_c3:
        st.markdown("""
        **SGD** *(yeastgenome.org)*
        - GO annotations, ~120,000 records
        - Evidence codes for π priors
        - Gene IDs, chromosome lengths
        """)
    with src_c4:
        st.markdown("""
        **Y1000+** *(Opulente et al. 2024)*
        - 1,154 yeast genome assemblies
        - GFF3 annotations + protein FASTA
        - Cross-species π₂, π₃, π₄ pipeline
        """)

    st.info(
        "The **Overview** tab has a full map of every tab. "
        "The **Y1000+ π Estimators** tab generates its data automatically in the background — "
        "results appear once the scan is complete.",
        icon="💡",
    )


# ══════════════════════════════════════════════════════════════════════
# TAB 1: Overview  (new — tab map)
# ══════════════════════════════════════════════════════════════════════

with tab1:
    st.header("App Overview")
    st.markdown(
        "A quick map of every tab in this app — what it contains, what data it uses, "
        "and what questions it answers."
    )

    tabs_info = [
        {
            "icon": "📘",
            "name": "Introduction",
            "what": (
                "Plain-language walkthrough of the model. Explains what gene duplication "
                "is, what a regulatory subnetwork motif is, and why inheritance probability "
                "π matters."
            ),
            "data": "No data required — narrative only.",
            "question": "What is this app doing and why?",
        },
        {
            "icon": "📊",
            "name": "Methodology",
            "what": (
                "Mathematical background from Scruse, Arnold & Robinson (2024). "
                "Covers Theorems 1–8, the Pólya urn model, Full vs Partial Duplication, "
                "and the significance-testing framework."
            ),
            "data": "Theoretical — no CSV data loaded.",
            "question": "How does the model work mathematically?",
        },
        {
            "icon": "🔬",
            "name": "TF Explorer",
            "what": (
                "Browse all 179 transcription factors in the combined JASPAR 2024 ∪ YEASTRACT "
                "curated set: evidence codes, GO annotations, binding consensus sequences, "
                "JASPAR PWM profiles, and regulatory targets."
            ),
            "data": (
                "sgd_transcription_factors.csv · sgd_tf_go_annotations.csv · "
                "jaspar_yeast_tfbs_2024.csv · yeastract_consensus.csv"
            ),
            "question": "What do we know about a specific TF's binding and regulation?",
        },
        {
            "icon": "👨‍👩‍👧",
            "name": "Gene Families",
            "what": (
                "Groups genes into families using five methods (GO Process, GO Function, "
                "GO Component, JASPAR TF Class, JASPAR TF Family). Shows family-size "
                "distributions and derives the model parameters m, n, and d."
            ),
            "data": "sgd_go_annotations_full.csv · sgd_transcription_factors.csv",
            "question": "How are genes grouped into paralog families and what are m, n, d?",
        },
        {
            "icon": "🎲",
            "name": "π Estimator",
            "what": (
                "Estimates the inheritance-probability vector $\\vec{\\pi}$ using seven complementary methods: "
                "four SGD-based — (1) evidence-code quality, (2) Moment Estimation via Theorem 4, "
                "(3) SNP divergence at YFL039C, (4) YEASTRACT consensus-sequence flexibility — "
                "and three Y1000+ cross-species methods: (5) π₃ TFBS conservation, "
                "(6) π₂ protein sequence homology, (7) π₄ IC-weighted SNP at binding sites."
            ),
            "data": (
                "sgd_transcription_factors.csv · sgd_YFL039C_inheritance_vectors.csv · "
                "yeastract_consensus.csv · Y1000+ π₂/π₃/π₄ CSVs (auto-generated)"
            ),
            "question": "What is the probability that a regulatory link survives duplication?",
        },
        {
            "icon": "🧪",
            "name": "Motif Significance",
            "what": (
                "Tests whether a k-tuple of gene families forms a significantly "
                "over- or under-represented subnetwork motif, using the z-score framework "
                "from Theorems 1 & 4. Also provides a predictive forecast of motif counts "
                "under future duplication."
            ),
            "data": "Computed from Gene Families + π Estimator outputs.",
            "question": "Is this regulatory wiring pattern more common than expected by chance?",
        },
        {
            "icon": "🌍",
            "name": "Y1000+ π Estimators",
            "what": (
                "Three cross-species estimators derived from 1,154 yeast genomes "
                "(Opulente et al. 2024): π₂ (protein sequence identity), "
                "π₃ (TFBS conservation via PWM scanning across a 48-species panel), "
                "and π₄ (IC-weighted SNP rate at binding site positions). "
                "The methodology expander includes a clickable link to the "
                "Method Estimation Test tab to benchmark π₃ against known profiles."
            ),
            "data": (
                "Y1000+ GFF3 + genome FASTAs · JASPAR 2024 PWMs · "
                "Pre-computed CSVs: pi2/pi3/pi4_*.csv (generate once via CLI)"
            ),
            "question": "How conserved are S. cerevisiae regulatory links across all yeasts?",
        },
        {
            "icon": "🧮",
            "name": "Method Estimation Test",
            "what": (
                "Benchmark all seven π estimation methods plus the multi-signal ensemble against "
                "known synthetic true-π profiles (Linear and Quadratic). "
                "Choose a method (M1–M7 or Ensemble), a motif size k = 3 or 4, and — for "
                "M1/M4/M5/M6/M7/Ensemble — a set of k TFs. The app computes per-family MSE "
                "analytically; Method 2 also shows a **1,000-replica Poisson simulation** "
                "to capture count-observation noise. "
                "The 📊 Compare All Methods expander runs all seven methods simultaneously, "
                "ranks them by average MSE, and explains why each method achieves its "
                "accuracy level given its underlying data source."
            ),
            "data": (
                "SGD evidence codes (M1/M4) · YFL039C SNP strains (M3) · "
                "YEASTRACT consensus sequences (M4) · "
                "Y1000+ π₃ CSV (M5) · Y1000+ π₂ CSV (M6) · Y1000+ π₄ CSV (M7, all require generation)"
            ),
            "question": "How accurately does each estimation method recover a known π value, and why?",
        },
        {
            "icon": "📖",
            "name": "Glossary & References",
            "what": (
                "Definitions of all mathematical terms (π, m, n, d, k-motif, Pólya urn, …) "
                "and full citations for the paper, datasets, and databases."
            ),
            "data": "No data — reference material only.",
            "question": "What does this term mean? Where does this data come from?",
        },
    ]

    for info in tabs_info:
        with st.expander(f"{info['icon']}  **{info['name']}**", expanded=False):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown(f"**What it does:** {info['what']}")
                st.markdown(f"**Key question:** _{info['question']}_")
            with col_r:
                st.markdown("**Data sources:**")
                st.caption(info["data"])


# ══════════════════════════════════════════════════════════════════════
# TAB 2: Methodology (was Overview)
# ══════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Counting Subnetworks Under Gene Duplication in GRNs")
    st.markdown(
        "_Scruse, Arnold & Robinson (2026) · Bull. Math. Biol. 88, 31 · University of Georgia_"
    )

    st.markdown("""
    This application implements the mathematical framework from the paper to:

    - **Identify** transcription factors (TFs) and their regulatory targets in the
      *S. cerevisiae* GRN from SGD data.
    - **Estimate** the per-family inheritance probability vector **$\\vec{\\pi}$ = (π₁, …, πₖ)**
      using seven data-driven methods (four SGD-based, three cross-species from Y1000+).
    - **Test significance** of subnetwork motifs against the Partial Duplication null model.
    """)

    st.divider()
    _js_sum = jaspar_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("JASPAR TFs", _js_sum["n_jaspar_tfs"])
    c2.metric("Mean IC (bits)", _js_sum["mean_ic_bits"])
    c3.metric("Mean motif width (bp)", _js_sum["mean_motif_width"])
    c4.metric("ChIP-based PFMs", _js_sum["n_chip_based"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("TFs in SGD (JASPAR ∪ YEASTRACT)", summary["n_tfs"])
    c6.metric("GO records", f"{summary['n_go_annotations']:,}")
    c7.metric("Chromosomes", summary["n_chromosomes"])
    c8.metric("Genome", f"{summary['genome_size_mb']} Mb")

    st.divider()
    st.subheader("📖 Key Mathematical Framework")

    col_l, col_r = st.columns(2)
    with col_l:
        with st.container(border=True):
            st.markdown("**Theorem 1 — Full Duplication ($\\vec{\\pi}$ = 1)**")
            st.code("E[|M(n)|; k, m, n] = Γ(n+k)Γ(m) / [Γ(n)Γ(m+k)]", language=None)
            st.caption("Growth rate: Θ(nᵏ) — degree equals motif size k.")

        with st.container(border=True):
            st.markdown("**Theorem 4 — Partial Duplication ($\\vec{\\pi}$ general)**")
            st.code("E[|M(n)|; m, n, π, k] = Γ(π̂+n)Γ(m) / [Γ(π̂+m)Γ(n)]", language=None)
            st.markdown("Depends on $\\vec{\\pi}$ only through π̂ = Σπᵢ.")
            st.caption("Growth rate: Θ(n^π̂) — exponent *equals* total inheritance probability.")

    with col_r:
        with st.container(border=True):
            st.markdown("**Lemma 4 — Single-gene motif**")
            st.code("f(p, s) = Γ(p+s) / [Γ(s) Γ(p+1)]", language=None)
            st.caption("Expected instances when family size is s and inheritance probability is p.")

        with st.container(border=True):
            st.markdown("**Theorem 6 — Binary Inheritance (max 2nd moment)**")
            st.code(
                "E[|M(n)|²] = Γ(m)/Γ(n) × Σ_{A⊆K} (-1)^|A| × 2^k ×\n"
                "              Γ(n+π̂+Σ_{i∉A}πᵢ) / [2^|A| × Γ(m+π̂+Σ_{i∉A}πᵢ)]",
                language=None,
            )
            st.caption("Binary Inheritance maximises E[|M(n)|²] over all Partial Duplication refinements (Theorem 5).")

    st.divider()
    st.subheader("🔗 Binding Sites: What Do They Bind?")
    st.markdown("""
Transcription factor **binding sites** (also called *cis-regulatory elements* or *TFBS*)
are short DNA sequences in **gene promoter regions**.
- The TF protein binds its TFBS sequence through a DNA-binding domain (confirmed by **GO:0003677** / **GO:0000981**).
- Binding **activates** or **represses** transcription of the downstream gene.
- In the Scruse et al. framework, a regulatory edge (TF → gene) is the "subnetwork motif link"
  whose inheritance probability πᵢ we estimate.
- When a TF gene duplicates, its TFBS in target promoters may or may not be inherited by the new copy → that probability is **πᵢ**.
""")

    # Chromosome lengths bar chart
    chrom = load_chromosome_lengths()
    fig_chrom = px.bar(
        chrom, x="chromosome", y="length_mb",
        title="S. cerevisiae Chromosome Lengths",
        labels={"length_mb": "Length (Mb)", "chromosome": "Chromosome"},
        color="length_mb", color_continuous_scale="Blues",
    )
    fig_chrom.update_layout(showlegend=False, height=300, margin=dict(t=40, b=20))
    st.plotly_chart(fig_chrom, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2: TF Explorer
# ══════════════════════════════════════════════════════════════════════

with tab3:
    st.header("🔬 Transcription Factor Explorer")
    st.caption("Source: sgd_transcription_factors.csv · sgd_tf_go_annotations.csv")

    tfs = _load_tfs()

    # --- Filter controls ---
    col1, col2, col3 = st.columns(3)
    with col1:
        role_filter = st.multiselect(
            "Role", ["activator", "dual", "repressor", "unknown"], default=[]
        )
    with col2:
        dna_bind_only = st.checkbox("DNA-binding only", value=False)
    with col3:
        search = st.text_input("Search TF name / synonym", "")

    filtered = tfs.copy()
    if min_ev > 0:
        filtered = filtered[filtered["evidence_score"] >= min_ev]
    if role_filter:
        # Compute role on filtered set
        def _role(r):
            if r["is_activator"] and r["is_repressor"]:
                return "dual"
            if r["is_activator"]:
                return "activator"
            if r["is_repressor"]:
                return "repressor"
            return "unknown"
        filtered["role"] = filtered.apply(_role, axis=1)
        filtered = filtered[filtered["role"].isin(role_filter)]
    if dna_bind_only:
        filtered = filtered[filtered["has_dna_binding"]]
    if search:
        mask = (
            filtered["gene_name"].str.contains(search, case=False, na=False)
            | filtered["synonyms"].str.contains(search, case=False, na=False)
            | filtered["full_name"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]

    st.markdown(f"**{len(filtered)} TFs** match current filters.")

    with st.expander("ℹ️ What is the Evidence Score?"):
        st.markdown("""
        The **evidence score** is a numeric quality weight (0–1) derived from the
        [SGD evidence codes](https://www.yeastgenome.org/help/analyze/go-slim-mapping) that
        describe how each TF's regulatory role was determined. It acts as the **prior for π**:
        a TF with strong experimental evidence is more likely to have stable, heritable
        regulatory links.

        | Code | Meaning | Score |
        |------|---------|-------|
        | **IDA** | Inferred from Direct Assay | 0.90 |
        | **IMP** | Inferred from Mutant Phenotype | 0.82 |
        | **IGI** | Inferred from Genetic Interaction | 0.72 |
        | **IPI** | Inferred from Physical Interaction | 0.68 |
        | **IBA** | Inferred from Biological Aspect of Ancestor | 0.58 |
        | **HDA** | High-throughput Direct Assay | 0.55 |
        | **IC** | Inferred by Curator | 0.50 |
        | **TAS** | Traceable Author Statement | 0.35 |
        | **IEA** | Inferred from Electronic Annotation (automated) | 0.30 |
        | **NAS** | Non-traceable Author Statement | 0.20 |
        | **ND** | No biological Data | 0.10 |

        When a TF has multiple evidence codes, the score is the **mean** across all codes.
        DNA-binding TFs receive a +0.05 bonus; activators a further +0.03, reflecting
        stronger evidence that their regulatory links are maintained after duplication.
        """)

    # Evidence score distribution
    fig_ev = px.histogram(
        filtered, x="evidence_score", nbins=30,
        title="Evidence Score Distribution (proxy for π prior)",
        labels={"evidence_score": "Evidence Score (0=ND, 1=IDA)"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig_ev.update_layout(height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig_ev, use_container_width=True)

    # Table
    display_cols = ["gene_name", "full_name", "has_dna_binding", "is_activator",
                    "is_repressor", "evidence_score", "pi_prior",
                    "n_function_terms", "n_process_terms"]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "gene_name": "TF", "full_name": "Description",
            "has_dna_binding": "DNA Binding", "is_activator": "Activator",
            "is_repressor": "Repressor", "evidence_score": "Ev. Score",
            "pi_prior": "π prior", "n_function_terms": "# Func GO",
            "n_process_terms": "# Proc GO",
        }),
        use_container_width=True, height=320,
    )

    st.divider()

    # --- Single TF deep-dive ---
    st.subheader("Single TF Deep-dive")
    tf_choice = st.selectbox(
        "Select a TF", sorted(tfs["gene_name"].tolist()), index=0
    )

    if tf_choice:
        binding_info = describe_binding_sites(tf_choice)
        if "error" not in binding_info:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{binding_info['gene_name']}** — {binding_info['full_name']}")
                role = "/".join(filter(None, [
                    "Activator" if binding_info["is_activator"] else "",
                    "Repressor" if binding_info["is_repressor"] else "",
                ])) or "Unknown"
                st.markdown(f"Role: `{role}`")
                st.markdown(f"Evidence score: **{binding_info['evidence_score']:.3f}**")
                st.markdown(f"Sequence-specific: **{'Yes' if binding_info['is_sequence_specific'] else 'No'}**")
                st.markdown(f"**π factor:** {binding_info['pi_factor']:.4f}")

                if binding_info["has_jaspar"]:
                    st.markdown(
                        f"**JASPAR** `{binding_info['jaspar_matrix_id']}` · "
                        f"consensus `{binding_info['jaspar_consensus']}` · "
                        f"{binding_info['jaspar_motif_width']} bp · "
                        f"IC {binding_info['jaspar_ic_bits']:.2f} bits  \n"
                        f"Class: {binding_info['jaspar_tf_class'] or '—'}  \n"
                        f"Family: {binding_info['jaspar_tf_family'] or '—'}  \n"
                        f"Data: {binding_info['jaspar_data_type'] or '—'}  \n"
                        f"[View on JASPAR]({binding_info['jaspar_url']})"
                    )
                if binding_info["has_yeastract_consensus"]:
                    st.caption(
                        f"YEASTRACT: {binding_info['n_consensus_sequences']} "
                        f"consensus sequences (π factor {binding_info['pi_consensus_factor']:.4f})"
                    )

            with col_b:
                st.markdown("**What do the binding sites bind?**")
                st.info(binding_info["binding_description"])
                st.markdown(f"**DNA target type:** {binding_info['target_dna_type']}")
                if binding_info["binding_go_terms"]:
                    for goid, label in binding_info["binding_go_terms"].items():
                        st.markdown(f"- `{goid}` — {label}")
                st.markdown(f"_{binding_info['binding_note']}_")

            # JASPAR PFM heatmap
            if binding_info["has_jaspar"]:
                st.markdown("**JASPAR Position Frequency Matrix**")
                pfm_df = get_jaspar_pfm(tf_choice)
                if not pfm_df.empty:
                    pfm_wide = pfm_df.pivot(index="nucleotide", columns="position", values="count").fillna(0)
                    fig_pfm = px.imshow(
                        pfm_wide,
                        labels={"x": "Position", "y": "Nucleotide", "color": "Count"},
                        color_continuous_scale="Blues",
                        title=f"{tf_choice} PFM — {binding_info['jaspar_matrix_id']}",
                        aspect="auto",
                    )
                    fig_pfm.update_layout(height=220, margin=dict(t=40, b=10))
                    st.plotly_chart(fig_pfm, use_container_width=True)

                    # Sequence logo (information-content representation)
                    logo_buf = _render_sequence_logo(
                        pfm_df,
                        tf_choice,
                        binding_info["jaspar_matrix_id"],
                    )
                    if logo_buf:
                        st.markdown("**Sequence Logo** — letter height = information content (bits)")
                        st.image(logo_buf, use_container_width=True)
                        st.caption(
                            "Classic colour scheme: A = green · C = blue · G = gold · T = red. "
                            "Total column height = IC at that position (max 2 bits). "
                            "Tall, single-letter columns are highly specific; short mixed columns are degenerate."
                        )

            # YEASTRACT consensus table (collapsed by default)
            if binding_info.get("has_yeastract_consensus"):
                with st.expander("YEASTRACT Consensus Sequences"):
                    cons_df = get_consensus_for_tf(tf_choice)
                    st.dataframe(
                        cons_df[["consensus", "length", "ambiguity_fraction", "specificity_score"]].rename(
                            columns={"ambiguity_fraction": "IUPAC ambiguity", "specificity_score": "specificity"}
                        ),
                        use_container_width=True,
                        height=min(300, 38 + 35 * len(cons_df)),
                    )

    st.divider()

    # --- Regulatory network preview ---
    st.subheader("TF → Target Regulatory Network")
    st.caption(
        "Targets are inferred from shared GO Biological Process terms, "
        "with TF binding characterised by JASPAR 2024 PFMs. "
        "This is the regulatory edge set underlying subnetwork motif instances."
    )

    with st.spinner("Building network…"):
        tf_map = _load_tf_target_map(max_tfs_net, min_ev)

    if not tf_map.empty:
        stats = network_statistics(tf_map)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TFs in network", stats["n_tfs"])
        c2.metric("Target genes", stats["n_targets"])
        c3.metric("Regulatory edges", f"{stats['n_edges']:,}")
        c4.metric("Avg targets / TF", stats["mean_targets_per_tf"])

        # Show targets of selected TF
        tf_in_net = tf_map[
            tf_map["tf_name"].str.upper() == tf_choice.upper()
        ].head(30)
        if not tf_in_net.empty:
            st.markdown(f"**Top 30 targets of {tf_choice}** (sorted by evidence score):")
            st.dataframe(
                tf_in_net[["target_gene", "shared_go_process", "evidence_score", "tf_role"]],
                use_container_width=True, height=280,
            )
        else:
            st.info(f"No targets found for {tf_choice} in this network slice.")

        # Edge evidence score distribution
        fig_net = px.histogram(
            tf_map, x="evidence_score", color="tf_role", nbins=25,
            title="Regulatory Edge Evidence Scores by TF Role",
            labels={"evidence_score": "Evidence Score", "tf_role": "TF Role"},
            barmode="overlay", opacity=0.7,
        )
        fig_net.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.warning("No network data — try adjusting filters.")


# ══════════════════════════════════════════════════════════════════════
# TAB 3: Gene Families
# ══════════════════════════════════════════════════════════════════════

with tab4:
    st.header("👨‍👩‍👧 Gene Family Analysis")

    _grouping_descriptions = {
        "GO_Process":    "Each **GO Biological Process** term defines a family — TFs that share the same biological pathway (paper Section 2, default).",
        "GO_Function":   "Each **GO Molecular Function** term defines a family — TFs sharing the same molecular activity or binding-domain type. Closer to protein-sequence paralogy than Process grouping.",
        "GO_Component":  "Each **GO Cellular Component** term defines a family — TFs sharing the same subcellular location or complex membership. Expect a dominant 'nucleus' family.",
        "JASPAR_Class":  "Each **JASPAR TF Class** (binding-domain architecture, e.g. 'C6 zinc cluster factors') defines a family. A protein-sequence similarity proxy using 177 JASPAR TFs.",
        "JASPAR_Family": "Each **JASPAR TF Family** (finer domain subtype, e.g. 'Myb/SANT domain factors') defines a family. Finer-grained protein-sequence proxy.",
    }
    st.markdown(
        _grouping_descriptions.get(_GROUPING_KEY, "") + """

This operationalises the Scruse et al. framework:

- **m** = number of gene families
- **n** = total TFs across families (Σcᵢ)
- **d = n − m** = estimated duplication events (Proposition 1)
"""
    )

    with st.spinner("Building TF families…"):
        fam_df = _load_tf_families(min_family_size, _GROUPING_KEY)

    if fam_df.empty:
        st.warning("No families found. Lower the minimum family size.")
    else:
        params = estimate_model_parameters(fam_df, size_col="family_size")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Families (m)", params["m"])
        c2.metric("Total TFs (n)", params["n"])
        c3.metric("Duplication events (d)", params["d"])
        c4.metric("Mean family size", params["mean_family_size"])

        st.markdown(
            f"Under **Proposition 1**, after d = {params['d']} duplications from m = {params['m']} "
            f"singletons, all compositions of n = {params['n']} into {params['m']} parts are equally likely."
        )

        # Family size distribution
        size_counts = family_size_distribution(fam_df, "family_size")
        fig_fam = px.bar(
            x=size_counts.index, y=size_counts.values,
            title=f"Gene Family Size Distribution ({family_grouping})",
            labels={"x": "Family Size (cᵢ)", "y": "Number of families"},
            color=size_counts.values, color_continuous_scale="Viridis",
        )
        fig_fam.update_layout(height=300, showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig_fam, use_container_width=True)

        # Expected count vs k under Full Duplication
        m_val, n_val = params["m"], params["n"]
        k_vals = list(range(1, min(11, m_val + 1)))
        exp_vals = [expected_full(k, m_val, n_val) for k in k_vals]

        fig_exp = go.Figure()
        fig_exp.add_trace(go.Scatter(
            x=k_vals, y=exp_vals, mode="lines+markers",
            name="E[|M(n)|]", line=dict(color="#1f77b4", width=2),
        ))
        fig_exp.update_layout(
            title=f"Theorem 1: Expected Motif Count vs Motif Size k (m={m_val}, n={n_val})",
            xaxis_title="Motif size k",
            yaxis_title="E[|M(n)|]",
            height=300, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_exp, use_container_width=True)

        _family_id_label = {
            "GO_Process":    "GO Process ID",
            "GO_Function":   "GO Function ID",
            "GO_Component":  "GO Component ID",
            "JASPAR_Class":  "JASPAR Binding Class",
            "JASPAR_Family": "JASPAR Binding Family",
        }.get(_GROUPING_KEY, "Family ID")
        st.subheader("Top Gene Families")
        st.dataframe(
            fam_df[["go_id", "family_size", "mean_evidence_score",
                    "n_activators", "n_repressors"]].head(30).rename(
                columns={"go_id": _family_id_label}
            ),
            use_container_width=True, height=300,
        )

        # Pólya urn connection
        st.subheader("Connection to the Pólya Urn Model (Section 7.1 / Theorem 8)")
        st.markdown("""
The gene duplication process is a multi-colour **Pólya urn**:
- Urn = genome, balls = genes, colours = gene families
- At each step a ball is drawn and a copy of the same colour is added

This gives the exact probability:

$$P[\\vec{X} = \\vec{t} \\mid \\vec{s}] = \\binom{n-m}{t_1,\\ldots,t_w} \\frac{(m-1)!}{(n-1)!} \\prod_{j=1}^{w} \\frac{(s_j+t_j-1)!}{(s_j-1)!}$$

Family proportions cᵢ/n converge almost surely to a **Dirichlet(1,…,1)** distribution.
""")

        # Simulate Dirichlet proportions
        alpha = [1.0] * min(params["m"], 10)
        sim_props = np.random.dirichlet(alpha, size=500)
        fig_dir = px.box(
            pd.DataFrame(sim_props, columns=[f"F{i+1}" for i in range(len(alpha))]),
            title=f"Simulated Family Proportion Distribution — Dirichlet({', '.join(['1']*len(alpha))})",
            labels={"value": "Family proportion (cᵢ / n)", "variable": "Family"},
        )
        fig_dir.update_layout(height=280, margin=dict(t=50, b=10))
        st.plotly_chart(fig_dir, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4: π Estimator
# ══════════════════════════════════════════════════════════════════════

with tab5:
    st.header("🎲 Inheritance Probability Estimator")
    st.markdown("""
Estimate **$\\vec{\\pi}$ = (π₁, …, πₖ)** — the per-family probability that regulatory
links are inherited through gene duplication.
""")

    st.markdown("Four estimation methods are available (Scruse et al. Sections 4, 6, 9.2):")

    with st.expander("📖 Methodology — how and why these estimates work", expanded=False):
        st.markdown("""
**What are we estimating, and why does this calculation make sense?**

The core quantity is **π̂ = Σπᵢ**, the total inheritance probability across all *k* gene families in a motif. Theorem 4 (Scruse et al.) shows that the *expected* number of motif instances in the GRN is:

```
E[|M(n)|] = Γ(π̂ + n)·Γ(m) / [Γ(π̂ + m)·Γ(n)]
```

This expression depends on $\\vec{\\pi}$ = (π₁, …, πₖ) **only through the scalar sum π̂** — a remarkable reduction that means all four methods below are estimating the same underlying quantity, just via different data sources.

**Why the ratio-of-Gamma-functions form?**
The gene duplication process is a multi-colour Pólya urn (Theorem 8). When the urn has *m* initial balls distributed among gene families and grows to *n* balls through *d = n − m* duplication steps, the exact probability that a specific k-tuple of genes all come from the right families involves rising factorials — which is exactly what the Gamma ratio computes. The inheritance probability π̂ "bends" the polynomial growth from Θ(nᵏ) (full duplication) to Θ(n^π̂) (partial inheritance).

**Why each method gives a valid estimate of π:**

| Method | Core insight | Why it works |
|--------|-------------|--------------|
| **Evidence-based (Method 1)** | SGD evidence codes grade how directly a regulatory relationship was verified | Higher experimental rigor → binding sites more likely to be real, stable, and conserved after duplication → higher π prior |
| **Moment Estimation (Method 2)** | Theorem 4 inverted: given the observed motif count, find the π̂ that makes the model's expected count match | This is the Method of Moments — the most data-driven estimate, anchoring π directly to the observed network structure |
| **SNP divergence (Method 3)** | Mutations at binding-site positions erode binding over evolutionary time | π ≈ 1 − (fraction of SNP positions) because a preserved binding site needs to be SNP-free at its specific positions; divergence accumulates proportionally to link age |
| **Consensus-adjusted (Method 4)** | IUPAC ambiguity in binding consensus reflects how many DNA sequences are tolerated | A TF with a precise, unambiguous consensus needs an exact sequence → sites are easy to lose by mutation → lower π; a degenerate TF tolerates more variation → sites survive more mutations → higher π |

All four methods produce values in [0, 1] per family and are compared on the same scale via the sensitivity analysis chart below.
""")

    with st.expander("📐 About the four estimation methods — click to expand"):
        st.markdown("""
**How we arrived at each method and what it measures:**

| # | Method | Data source | What it measures | How π is derived |
|---|--------|------------|-----------------|-----------------|
| 1 | **Evidence-based** | SGD evidence codes (IDA, IMP, IEA, …) | Quality of experimental evidence that a TF's regulatory role is real and stable | Each evidence code is assigned a weight (0.10–0.90) based on experimental rigor. The per-family π is the mean weight across all TFs in that family, adjusted for DNA-binding status and activator role. This is the *prior* — before looking at any network data. |
| 2 | **Moment Estimation (Theorem 4)** | Observed motif count &#124;M(n)&#124; | How much of the observed regulatory structure is explained by inherited duplication links | Theorem 4 gives E[&#124;M(n)&#124;] = Γ(π̂+n)Γ(m)/[Γ(π̂+m)Γ(n)]. We invert this numerically: find the π̂ that makes the expected count equal the observed count — this is Method of Moments, equating the theoretical first moment to the observed value — then distribute uniformly across k families. This is a *data-fitted* estimate. |
| 3 | **SNP divergence proxy** | YFL039C per-strain SNP data (SGD) | How much a specific well-studied TF target has diverged across yeast strains | π₃ ≈ 1 − (fraction of SNP positions in the TF binding site). The YFL039C locus is used as a calibrated example because it has dense strain-level variation data. This gives a *sequence-level* estimate grounded in observed genetic variation. |
| 4 | **Consensus-adjusted (YEASTRACT)** | YEASTRACT IUPAC consensus sequences | Binding specificity of each TF — how degenerate or precise its binding sequence is | TFs with many, highly ambiguous IUPAC consensus sequences have lower π (their sites are easy to lose or gain after duplication). TFs with few, specific consensus sequences have higher π. The IUPAC ambiguity fraction is inverted and calibrated to the 0–1 π range. |

**Which method should I use?**

- Start with **Method 1** for a biology-grounded prior based on experimental evidence quality.
- Use **Method 2** if you have a reliable observed motif count and want to back-calculate what π must have been.
- Use **Method 3** when you want a sequence-divergence view anchored in real genetic variation data.
- Use **Method 4** if you are specifically interested in how binding specificity constrains inheritance.
- Use **All four** to compare methods and compute an ensemble mean.
""")

        st.divider()
        st.markdown("**🧮 Method Estimation Test**")
        st.markdown(
            "Each method's accuracy and structural limitations are tested head-to-head "
            "using synthetic Linear and Quadratic π profiles (k = 3 or 4, m = 7). "
            "Click below to navigate there directly:"
        )
        # JavaScript tab-jump: queries Streamlit's rendered tab buttons by label text
        # and programmatically clicks the target. Works locally and on Streamlit Cloud
        # (same-origin iframe, so window.parent.document access is permitted).
        # Fragility note: depends on Streamlit's internal data-baseweb="tab" attribute;
        # may break if Streamlit changes its front-end component library.
        components.html("""
        <a href="#"
           onclick="
             (function() {
               var tabs = window.parent.document.querySelectorAll('button[data-baseweb=tab]');
               for (var i = 0; i < tabs.length; i++) {
                 if (tabs[i].innerText.indexOf('Method Estimation Test') !== -1) {
                   tabs[i].click();
                   window.parent.scrollTo(0, 0);
                   break;
                 }
               }
             })();
             return false;
           "
           style="display:inline-block; padding:7px 18px; background:#2563eb; color:white;
                  border-radius:6px; text-decoration:none; font-size:0.88em; font-weight:500;
                  font-family:-apple-system,BlinkMacSystemFont,sans-serif; cursor:pointer;"
           onmouseover="this.style.background='#1d4ed8'"
           onmouseout="this.style.background='#2563eb'">
          🧮 Open Method Estimation Test &rarr;
        </a>
        """, height=48)

        st.markdown("""
**Key findings from the test:**
- **Methods 1 & 4** are bounded by the SGD evidence-code range (~0.31–0.82) and cannot
  represent extreme inheritance probabilities near 0 or 1.
- **Method 2** recovers π̂ (the total) exactly via Theorem 4 inversion, but per-family
  MSE is non-zero because it distributes π̂ uniformly — it cannot resolve which
  individual families inherit more or less.
- **Method 3** applies a single cross-strain mean (μ ≈ 0.59) to all families equally;
  MSE is driven by how spread out the true profile is around that anchor point.
""")

    with st.spinner("Loading families…"):
        fam_df4 = _load_tf_families(min_family_size, _GROUPING_KEY)

    if fam_df4.empty:
        st.warning("No families found.")
        st.stop()

    params4 = estimate_model_parameters(fam_df4, "family_size")
    m4, n4 = params4["m"], params4["n"]

    # Select k families for the motif
    st.subheader("Step 1 — Define the subnetwork motif")
    col_k, col_strat = st.columns(2)
    with col_k:
        k4 = st.slider("Motif size k (number of gene families)", 1, min(8, m4), 2)
    with col_strat:
        strat4 = st.selectbox(
            "Family selection strategy",
            ["largest", "highest_ev", "balanced", "random"],
            help=(
                "largest: families with most TFs; "
                "highest_ev: best evidence; "
                "balanced: mix; "
                "random: random sample"
            ),
        )

    selected_fams = select_motif_families(fam_df4, k4, strategy=strat4)
    gene_names4 = [f["go_id"] for f in selected_fams]
    family_sizes4 = [f["family_size"] for f in selected_fams]

    st.markdown(f"**Selected {k4} families for motif M:**")
    fam_display = pd.DataFrame(selected_fams)[
        ["go_id", "family_size", "mean_evidence_score", "n_activators", "n_repressors"]
    ]
    st.dataframe(fam_display, use_container_width=True, height=180)

    obs_count4 = count_observed_motif_instances(selected_fams, mode="full_duplication")
    st.markdown(
        f"Cartesian-product observed count |M(n)| ≈ **{obs_count4:,}** "
        f"(Full Duplication upper bound: c₁×…×cₖ = {' × '.join(str(f['family_size']) for f in selected_fams)})"
    )

    st.divider()
    st.markdown("### Step 2 — Estimate $\\vec{\\pi}$")

    # Method selection
    method4 = st.radio(
        "Estimation method",
        [
            "Method 1: Evidence-based (Lemma 4 / evidence codes)",
            "Method 2: Moment Estimation (Theorem 4)",
            "Method 3: SNP divergence proxy (YFL039C example)",
            "Method 4: Consensus-adjusted (YEASTRACT)",
            "All four — comparison table",
        ],
        horizontal=False,
    )

    if "Method 1" in method4:
        result4 = estimate_pi_from_evidence(gene_names4)
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"**Method 1 — Evidence-based**\n\n{result4['description']}")
        st.markdown(f"pi_hat = {result4['pi_hat']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        pi_vec4 = result4["pi_vec"]
        with st.expander("🔍 What do these results mean?"):
            try:
                st.markdown(_pi_interpretation(result4["pi_hat"], k4, m4, n4))
            except Exception as _e:
                st.warning(f"Could not generate interpretation: {_e}")
            st.caption(
                "**Method 1 note:** This estimate is derived entirely from evidence-code "
                "quality weights and is independent of observed network structure. It is a "
                "prior — use Method 2 to check how this prior compares to what the observed "
                "motif count implies."
            )

        with st.expander("📖 How Method 1 works — step by step"):
            st.markdown("""
**Step 1 — Map evidence codes to reliability weights**

For every TF→gene regulatory relationship in SGD, the evidence code is mapped to a
numerical weight reflecting how directly that relationship was experimentally established:

| Code | Type | Weight |
|------|------|--------|
| IDA | Inferred from Direct Assay (ChIP, EMSA) | 0.90 |
| IMP | Inferred from Mutant Phenotype | 0.80 |
| IGI | Inferred from Genetic Interaction | 0.70 |
| IPI | Inferred from Physical Interaction | 0.65 |
| EXP | Inferred from Experiment (general) | 0.60 |
| IEP | Inferred from Expression Pattern | 0.50 |
| HTP | High-Throughput Experiment | 0.40 |
| IEA | Inferred from Electronic Annotation | 0.10 |

**Step 2 — Adjust for DNA-binding and activator status**

Each TF's weight is scaled by two biological factors from SGD annotations:
- **DNA-binding status**: TFs with confirmed DNA-binding domains get a multiplier > 1; TFs
  lacking a confirmed domain are downweighted (less likely to have stable, inheritable binding sites).
- **Activator role**: Transcriptional activators score slightly higher than repressors or
  dual-function TFs, because activator binding sites tend to be more experimentally characterised.

**Step 3 — Average within each gene family**

The k gene families in the selected motif each contain one or more TFs. For each family, πᵢ
is the mean adjusted weight across all TFs in that family. This gives a per-family inheritance
probability in [0, 1], reflecting the average experimental-evidence quality for the regulatory
relationships in that family.

**Step 4 — Sum to π̂**

π̂ = π₁ + π₂ + ··· + πₖ. This total feeds into Theorem 4 to give the model's expected motif count.

---

**What Method 1 is and is not**

Method 1 is a **prior** — it summarises the quality of evidence for each regulatory link *before*
looking at how many motif instances actually exist in the network. It cannot detect whether the
network is over- or under-represented relative to the duplication model's prediction. Use Method 2
to see how this prior compares to the network-anchored estimate.

The evidence-code range available in SGD (~0.10–0.90) means Method 1 cannot produce extreme
estimates near 0 or 1 — it is bounded by the available annotation quality.
""")

    elif "Method 2" in method4:
        obs_input = st.number_input(
            "Observed motif count |M(n)|",
            min_value=1.0, max_value=float(obs_count4),
            value=float(min(obs_count4, max(1, obs_count4 // 2))),
            step=1.0,
        )
        result4 = estimate_pi_from_mle(obs_input, m4, n4, gene_names4)
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"**Method 2 — Moment Estimation (Theorem 4)**")
        st.markdown(result4.get("note", ""))
        st.markdown(f"pi_hat = {result4['pi_hat']:.4f}   |   {result4['description'][:100]}...")
        st.markdown('</div>', unsafe_allow_html=True)
        pi_vec4 = result4["pi_vec"]
        with st.expander("🔍 What do these results mean?"):
            try:
                st.markdown(_pi_interpretation(result4["pi_hat"], k4, m4, n4))
            except Exception as _e:
                st.warning(f"Could not generate interpretation: {_e}")
            st.caption(
                "**Method 2 note:** Moment Estimation inverts Theorem 4 — it finds the π̂ that makes "
                "the model's expected count (the theoretical first moment) match the count you entered above. If the observed "
                "count is at or near the Full Duplication bound, π̂ will be close to k "
                "(nearly full inheritance). If well below, π̂ < k indicates partial loss."
            )

        with st.expander("📖 How Method 2 works — step by step"):
            st.markdown(f"""
**Step 1 — Estimating π̂ (the total inheritance probability)**

Theorem 4 (Scruse et al.) gives the *expected* motif count under Partial Duplication as a function of
the sum π̂ = π₁ + π₂ + ··· + π_k:

```
E[|M(n)|] = Γ(π̂ + n) · Γ(m) / [Γ(π̂ + m) · Γ(n)]
```

This equation is **monotone increasing in π̂**, so given the observed count `c`, there is at most one
π̂ ∈ [0, k] satisfying it. The code finds that root numerically (Brent's method) by solving:

```
f(π̂) = Γ(π̂+n)Γ(m)/[Γ(π̂+m)Γ(n)] − c = 0
```

This is **Method of Moments** — match the theoretical first moment to the observed count. It is not
MLE (which would require maximising the full likelihood of the count distribution).

**Compatibility with Theorem 1 (Full Duplication):** Theorem 4 *is* Theorem 1 when π̂ = k (all
links perfectly inherited). So if the observed count equals the Full Duplication expectation, the
inversion returns π̂ = k and every πᵢ = 1 — correctly recovering the Full Duplication boundary.

---

**Step 2 — Distributing π̂ to individual πᵢ values**

Theorem 4 depends on $\\vec{{\\pi}}$ **only through its sum π̂** — the individual πᵢ values are invisible to it.
This means the inversion in Step 1 cannot determine how π̂ is split across the k families; there are
infinitely many vectors $\\vec{{\\pi}}$ with the same sum.

To produce per-family values the code distributes π̂ proportionally to a weight vector:

```
πᵢ = π̂ · wᵢ / Σwᵢ,   clamped to [0, 1]
```

The weights wᵢ default to the **Method 1 evidence scores** (SGD experimental evidence codes). This
means the *shape* of the πᵢ distribution across families comes entirely from Method 1; Method 2
contributes only the *magnitude* of π̂, calibrated to the observed network count.

---

**Why n barely changes the accuracy**

Method 2 uses a **single observation** of |M(n)| — the count of motif instances in the current
network. By the delta method, the standard error of the recovered π̂ is approximately:

```
Std(π̂_hat) ≈ Std(|M(n)|) / |∂E/∂π̂|
            ≈ Θ(n^π̂) / [Θ(n^π̂) · ln(n/m)]
            = 1 / ln(n/m)
```

because both the noise (Std of the count, from Corollary 16) and the signal (∂E/∂π̂, the slope of
Theorem 4) grow at the same rate Θ(n^π̂), leaving only the logarithmic factor ln(n/m) in the
denominator. For yeast (n ≈ {n4:,}, m ≈ {m4:,}), ln(n/m) ≈ {np.log(max(n4, 2) / max(m4, 1)):.1f}.
Doubling the genome size would increase this by ln(2) ≈ 0.69 — a negligible gain.

The practical consequence is that **Method 2's uncertainty is set by the single-observation nature
of the problem, not by genome size**. Methods 5–7 (Y1000+ cross-species) escape this by averaging
across 1,154 independent genomes, which gives 1/√1154 ≈ 3% standard error on the retention fraction.
""")

    elif "Method 3" in method4:
        result4 = estimate_pi_from_snp(gene_names4)
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"**Method 3 — SNP Divergence Proxy**\n\n{result4['description']}")
        st.markdown(f"pi_hat = {result4['pi_hat']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        pi_vec4 = result4["pi_vec"]
        with st.expander("🔍 What do these results mean?"):
            try:
                st.markdown(_pi_interpretation(result4["pi_hat"], k4, m4, n4))
            except Exception as _e:
                st.warning(f"Could not generate interpretation: {_e}")
            st.caption(
                "**Method 3 note:** This estimate is calibrated on the YFL039C locus — "
                "a TF target with well-characterised strain-level SNP data in SGD. "
                "The chart below shows how π varies across yeast strains; the overall "
                "π̂ used here is the mean across strains. Strains with more alternative "
                "alleles at binding-site positions yield lower π (more divergence)."
            )

        with st.expander("📖 How Method 3 works — step by step"):
            st.markdown("""
**The core idea: binding-site divergence as a proxy for inheritance**

If a regulatory link (TF binding to a target gene) is truly inherited after gene duplication,
the binding site sequence must remain intact in both daughter copies. Single-nucleotide
polymorphisms (SNPs) at binding-site positions erode binding over time. A site with many
alternative alleles across strains is actively diverging — and therefore less likely to have
been faithfully inherited.

**Step 1 — Load per-strain SNP data for YFL039C**

SGD provides SNP calls across many *S. cerevisiae* strains for the YFL039C locus. This locus
was chosen because it has dense, well-curated per-strain variation data and is regulated by
multiple well-characterised TFs.

**Step 2 — Compute the alternative allele fraction per strain**

For each strain, `pct_alt` = (number of alternative alleles at TF binding positions) /
(total positions scored) × 100. A strain where every position matches the S288C reference
gets pct_alt = 0; a fully diverged strain gets pct_alt = 100.

**Step 3 — Convert to a per-strain π**

```
π₃(strain) = 1 − pct_alt / 100
```

This is the fraction of binding-site positions that are still reference-identical in that strain —
a direct measure of how much of the binding site has been preserved.

**Step 4 — Average across strains → π̂**

The mean of π₃(strain) over all sampled strains is the family-level estimate. This mean is
then applied uniformly to all k families in the selected motif.

---

**Important limitations**

- **Single locus:** All k families receive the same π̂ value because YFL039C is one calibration
  point — Method 3 cannot distinguish which family inherits more or less.
- **S. cerevisiae strains only:** This is within-species variation, not cross-species conservation.
  The Y1000+ π₃ estimator (tab 7) uses 1,154 species for a far stronger signal.
- **One TF's binding site:** The pct_alt is computed for a specific TF's binding-site positions
  at YFL039C. TFs with different binding-site lengths or sequence requirements may diverge at
  different rates.
""")

        st.subheader("YFL039C per-strain pi estimates")
        ivec = _load_inheritance()
        fig_snp = px.bar(
            ivec, x="strain", y="pi_snp", color="pct_alt",
            title="pi proxy by strain (pi = 1 - pct_alt/100)",
            labels={"pi_snp": "pi proxy", "pct_alt": "% Alt alleles"},
            color_continuous_scale="RdYlGn_r",
        )
        fig_snp.update_layout(height=300, xaxis_tickangle=-30, margin=dict(t=40, b=20))
        st.plotly_chart(fig_snp, use_container_width=True)

        snps = _load_snps()
        st.dataframe(snps, use_container_width=True, height=220)

    elif "Method 4" in method4:
        from model.inheritance_estimator import estimate_pi_consensus_adjusted
        result4 = estimate_pi_consensus_adjusted(gene_names4)
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"**Method 4 — Consensus-Adjusted (YEASTRACT)**\n\n{result4['description']}")
        st.markdown(f"pi_hat = {result4['pi_hat']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        pi_vec4 = result4["pi_vec"]
        with st.expander("🔍 What do these results mean?"):
            try:
                st.markdown(_pi_interpretation(result4["pi_hat"], k4, m4, n4))
            except Exception as _e:
                st.warning(f"Could not generate interpretation: {_e}")
            st.caption(
                "**Method 4 note:** π is lower for TFs with many, ambiguous IUPAC consensus "
                "sequences (degenerate binding) and higher for TFs with few, specific "
                "sequences. Degenerate sites are easier to gain or lose by point mutation, "
                "so a post-duplication copy is less likely to inherit every binding site. "
                "The scatter plot below illustrates the relationship: TFs in the upper-left "
                "(few sequences, low ambiguity) have higher π; those in the lower-right have lower π."
            )

        with st.expander("📖 How Method 4 works — step by step"):
            st.markdown("""
**The core idea: binding specificity constrains inheritance**

A TF with a highly precise, unambiguous binding consensus requires an exact DNA sequence. After
gene duplication, a single point mutation at a critical position can abolish binding entirely —
so inheritance probability is low. Conversely, a TF with a degenerate consensus (many tolerated
bases at each position) can still bind after mutations — so inheritance probability is higher.

YEASTRACT provides IUPAC consensus sequences for each TF. The IUPAC code encodes ambiguity:
`A/T/G/C` = fully specified; `R/Y/W/S/M/K` = two-fold ambiguous; `B/D/H/V` = three-fold;
`N` = fully degenerate. More ambiguous positions = less constraint on the binding site.

**Step 1 — Load IUPAC consensus sequences from YEASTRACT**

For each TF, YEASTRACT provides one or more consensus sequences. Multiple consensus sequences
indicate that the TF can bind several distinct sequence variants.

**Step 2 — Compute mean IUPAC ambiguity per TF**

For each consensus sequence, count the fraction of positions that are ambiguous IUPAC characters
(anything other than A, T, G, or C). Average across all positions and all consensus sequences
for that TF → `mean_ambiguity` ∈ [0, 1].

**Step 3 — Combine sequence count and ambiguity into a π factor**

Two signals are combined:
- **n_consensuses** (how many distinct sequence variants the TF recognises): more variants →
  easier to find a tolerated sequence after mutation → higher π
- **mean_ambiguity** (how degenerate each consensus is): more ambiguity → more mutation-tolerant
  → higher π

The π factor is a calibrated combination of both, normalised to the [0.1, 0.9] range of
observable YEASTRACT data.

**Step 4 — Average within each gene family**

Each gene family may contain multiple TFs. The family's πᵢ is the mean π factor across all TFs
in that family. Families dominated by highly specific TFs get lower πᵢ; families with degenerate
TFs get higher πᵢ.

---

**What this method captures and misses**

Method 4 captures the *mutational robustness* of the binding site — how many point mutations
the binding site can tolerate while remaining functional. It is independent of both experimental
evidence quality (Method 1) and the observed network structure (Method 2).

It does **not** capture whether binding sites are actually conserved across species (see π₃,
Y1000+ tab) or whether they have already diverged in existing strains (Method 3). A TF with
a degenerate consensus has high Method 4 π — but if its regulatory targets in *S. cerevisiae*
happen to have low sequence conservation, the actual inheritance probability could still be low.
""")

        # Show consensus stats per family
        st.markdown("**Per-family consensus binding statistics:**")
        cons_stats_df = pd.DataFrame(result4["consensus_stats"])
        if not cons_stats_df.empty:
            st.dataframe(cons_stats_df, use_container_width=True, height=200)

        # Top 20 TFs by consensus count from YEASTRACT
        st.subheader("YEASTRACT: Top TFs by consensus sequence count")
        yy_stats = load_tf_consensus_stats()
        fig_yy = px.bar(
            yy_stats.head(20), x="tf_sgd", y="n_consensuses",
            color="pi_consensus_factor",
            title="Top 20 TFs by number of consensus sequences (YEASTRACT)",
            labels={"tf_sgd": "TF (SGD name)", "n_consensuses": "# Consensus sequences",
                    "pi_consensus_factor": "pi factor"},
            color_continuous_scale="Blues",
        )
        fig_yy.update_layout(height=340, margin=dict(t=50, b=60), xaxis_tickangle=-40)
        st.plotly_chart(fig_yy, use_container_width=True)

        # Ambiguity vs count scatter
        fig_scatter = px.scatter(
            yy_stats, x="n_consensuses", y="mean_ambiguity",
            hover_name="tf_sgd", size="mean_length",
            title="Consensus count vs IUPAC ambiguity (bubble = mean length)",
            labels={"n_consensuses": "# Consensus seqs", "mean_ambiguity": "Mean IUPAC ambiguity"},
            color="pi_consensus_factor", color_continuous_scale="RdYlGn",
        )
        fig_scatter.update_layout(height=340, margin=dict(t=50, b=30))
        st.plotly_chart(fig_scatter, use_container_width=True)

    else:  # All four
        from model.inheritance_estimator import estimate_pi_consensus_adjusted
        comp_df = estimate_pi_all_methods(gene_names4, m4, n4, obs_count4)
        st.dataframe(comp_df, use_container_width=True, height=200)
        _ens_hat = comp_df["pi_ensemble"].sum() if "pi_ensemble" in comp_df.columns else 0
        with st.expander("🔍 What do these results mean?"):
            try:
                st.markdown(_pi_interpretation(_ens_hat, k4, m4, n4))
            except Exception as _e:
                st.warning(f"Could not generate interpretation: {_e}")
            st.markdown("""
**Reading the comparison table and chart:**

- **Agreement across methods** (bars at similar height) means the estimate is robust — the biological and sequence-level evidence are mutually consistent.
- **Disagreement** (e.g., Method 1 high but Method 3 low) suggests the evidence-code quality and the actual genetic divergence tell different stories. Method 2 (Moment Estimation) is the most data-driven; Method 1 is the most conservative prior.
- The **ensemble mean** (red bar) averages all four methods and is the recommended value to carry into the Motif Significance tab.
""")

        with st.expander("📖 How the ensemble comparison works — all four methods"):
            st.markdown("""
**Why compare four methods at all?**

Each method estimates π from a different data source and encodes different biological assumptions.
No single method is always correct — they are complementary:

| Method | Data source | What it encodes | Key limitation |
|--------|------------|----------------|----------------|
| **M1 — Evidence-based** | SGD evidence codes | Experimental confidence in each regulatory link | Bounded by annotation quality (range ~0.1–0.9); ignores network structure |
| **M2 — Moment Estimation** | Observed motif count | How much network structure is explained by duplication | Cannot distinguish per-family differences; high uncertainty from single observation |
| **M3 — SNP divergence** | YFL039C strain SNPs | Within-species sequence divergence at a calibration locus | All families get the same π; within-species variation only |
| **M4 — Consensus-adjusted** | YEASTRACT IUPAC sequences | Mutational robustness of binding sites | Does not measure whether sites are actually conserved across species |

**How the ensemble mean is computed**

For each gene family i, the ensemble mean is simply:
```
πᵢ_ensemble = (πᵢ_M1 + πᵢ_M2 + πᵢ_M3 + πᵢ_M4) / 4
```

and π̂_ensemble = Σ πᵢ_ensemble.

**How to read disagreement**

- **M1 high, M2 low:** SGD annotations suggest strong regulatory relationships, but the observed
  network has fewer motif instances than expected if π were that high. This could mean the evidence
  is biased toward well-studied TFs, or that some links are not preserved after duplication despite
  strong evidence.
- **M2 high, M1 low:** The network structure implies high inheritance, but evidence codes are weak.
  This suggests regulatory links that are functionally preserved but experimentally undercharacterised.
- **M3 diverges from M1/M2:** Within-species variation at YFL039C tells a different story than
  either the evidence quality or the network count. Could indicate that YFL039C is atypical, or
  that the motif families have different divergence histories.
- **M4 low, others high:** The TFs have highly specific, constrained binding sequences —
  so the binding sites are easily disrupted by mutation — but may still be conserved in practice
  (other methods). Specificity constrains maintenance but does not preclude it.

**The ensemble mean is a conservative starting point.** For publication-quality analysis, report
all four methods and note where they agree or disagree.
""")

        fig_comp = go.Figure()
        x = comp_df["gene_family"]
        for col, name, colour in [
            ("pi_evidence", "Evidence-based", "#1f77b4"),
            ("pi_mle", "Moment Estimation (Theorem 4)", "#ff7f0e"),
            ("pi_snp", "SNP divergence", "#2ca02c"),
            ("pi_consensus", "Consensus-adjusted", "#9467bd"),
            ("pi_ensemble", "Ensemble mean", "#d62728"),
        ]:
            fig_comp.add_trace(go.Bar(
                x=x, y=comp_df[col], name=name,
                marker_color=colour, opacity=0.8,
            ))
        fig_comp.update_layout(
            barmode="group",
            title="pi estimates by method and gene family",
            xaxis_title="Gene family (GO ID)",
            yaxis_title="pi_i",
            height=380, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        pi_vec4 = comp_df["pi_ensemble"].tolist()

    # π bar chart (single method)
    if "All four" not in method4:
        fig_pi = px.bar(
            x=[f"Family {i+1}\n({g[:12]})" for i, g in enumerate(gene_names4)],
            y=pi_vec4,
            title="Per-family Inheritance Probability (πᵢ)",
            labels={"x": "Gene family", "y": "πᵢ"},
            color=pi_vec4, color_continuous_scale="RdYlGn",
            range_color=[0, 1],
        )
        fig_pi.add_hline(y=1.0, line_dash="dash", line_color="grey",
                         annotation_text="Full Duplication (π=1)")
        fig_pi.update_layout(height=320, showlegend=False, margin=dict(t=50, b=20))
        st.plotly_chart(fig_pi, use_container_width=True)

    st.divider()
    st.subheader("Step 3 — π̂ Sensitivity Analysis (Corollary 9)")
    st.markdown(
        "Expected count E[|M(n)|] as a function of π̂ — shows how model output "
        "changes with the inheritance probability estimate."
    )
    sens_df = pi_sensitivity(m4, n4, k4, pi_hat_range=(0.01, float(k4)), steps=60)
    sens_df["upper_2sd"] = sens_df["expected"] + 2 * sens_df["std"]
    sens_df["lower_2sd"] = (sens_df["expected"] - 2 * sens_df["std"]).clip(lower=0)

    fig_sens = go.Figure()
    # ±2σ band (Corollary 9 / Theorem 6 variance) — drawn first so line sits on top
    fig_sens.add_trace(go.Scatter(
        x=pd.concat([sens_df["pi_hat"], sens_df["pi_hat"].iloc[::-1]]),
        y=pd.concat([sens_df["upper_2sd"], sens_df["lower_2sd"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(31,119,180,0.20)",
        line=dict(color="rgba(31,119,180,0.45)", width=1),
        name="±2σ prediction interval (Theorem 6)",
    ))
    fig_sens.add_trace(go.Scatter(
        x=sens_df["pi_hat"], y=sens_df["expected"],
        name="E[|M(n)|]", line=dict(color="#1f77b4", width=2),
    ))
    if "All four" not in method4:
        pi_hat4 = sum(pi_vec4)
        e4 = expected_partial(pi_hat4, m4, n4)
        fig_sens.add_trace(go.Scatter(
            x=[pi_hat4], y=[e4], mode="markers",
            marker=dict(size=12, color="red", symbol="star"),
            name=f"Current π̂ = {pi_hat4:.3f}",
        ))
    fig_sens.update_layout(
        xaxis_title="π̂ = Σπᵢ (total inheritance probability)",
        yaxis_title="E[|M(n)|]",
        height=320, margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    with st.expander("🔍 What do these sensitivity results mean?"):
        st.markdown("""
**Reading the sensitivity curve:**

The chart shows how the expected motif count E[|M(n)|] changes as the total inheritance
probability π̂ = Σπᵢ varies from near zero to its maximum (k). The red star marks your
current estimate.

**Key reference points:**

| π̂ value | Meaning |
|---------|---------|
| **π̂ → 0** | Near-zero inheritance — almost no regulatory links survive duplication. The motif count stays flat (constant) regardless of network size. |
| **0 < π̂ < k** | Partial Duplication — motif count grows as Θ(n^π̂). Each unit increase in π̂ raises the growth exponent by one. |
| **π̂ = k** | Full Duplication — all links are perfectly inherited. Maximum possible motif count; grows as Θ(nᵏ). |

**What the shaded band means:**

The shaded region is a **±2σ prediction interval** derived from the Binary Inheritance
variance (Theorem 6) — the approximate range within which an observed motif count should
fall ~95% of the time under the model at each π̂. If your observed count (plotted as the
red star) falls *outside* this band, the data is inconsistent with that π̂ value at the
5% level.

Note: this band is *not* the same as **Corollary 9**, which gives tight multiplicative
bounds on the *expected value* E[|M(n)|] itself (within a few percent of the mean when
n ≥ 2k²). Those bounds are mathematically precise but so narrow they are visually
indistinguishable from the curve. The ±2σ band shown here is wider and more useful for
judging whether an observed count is surprising — it reflects natural sampling variability
around the mean, not uncertainty in the asymptotic approximation.

**Biological significance:**

- A **steep curve** (large slope at your π̂) means small changes in inheritance probability
  produce large changes in motif count — the model is highly sensitive at this operating
  point. Even modest relaxation of selective pressure on binding sites would substantially
  reduce regulatory structure.
- A **flat curve** means the motif count is robust to uncertainty in π̂ — useful when your
  π̂ estimate carries some uncertainty.
- The **gap between Full and Partial Duplication** trajectories quantifies how much
  regulatory structure is *lost* relative to the maximum-inheritance scenario. A large gap
  indicates significant regulatory rewiring has occurred since duplication.
""")


# ══════════════════════════════════════════════════════════════════════
# TAB 5: Motif Significance
# ══════════════════════════════════════════════════════════════════════

with tab6:
    st.header("🧪 Subnetwork Motif Significance Testing")

    inf_tab, pred_tab = st.tabs(["📊 Inferential Test", "🔮 Predictive Forecast"])

    # ── Inferential Test ─────────────────────────────────────────────
    with inf_tab:
        st.markdown("""
Tests whether a subnetwork motif M of size k is **significantly over- or under-represented**
in the GRN relative to the gene duplication null model (Scruse et al. Sections 4–6).

**Z-score** = (observed − expected) / std dev  ·  **p-value** via normal approximation.
""")

        with st.expander("📖 Methodology — why this significance test makes sense", expanded=False):
            st.markdown("""
**The null hypothesis and why it is the right one:**

The null hypothesis is that the observed count of subnetwork motif instances |M(n)| is consistent with
the *gene duplication model* — that is, all regulatory links were inherited stochastically through
*d = n − m* duplication events, with per-family inheritance probability πᵢ.

This is a more biologically grounded null than a simple random-graph model, because it explicitly
models how GRNs grow: not randomly, but by copying genes and (stochastically) their regulatory connections.

**Two nested null models:**

| Null model | Assumption | Expected count |
|---|---|---|
| **Full Duplication (Theorem 1)** | Every regulatory link is perfectly copied at each duplication — π = 1 | E[&#124;M&#124;] = Γ(n+k)Γ(m)/[Γ(n)Γ(m+k)] — grows as Θ(nᵏ) |
| **Partial Duplication (Theorem 4)** | Each link survives with probability πᵢ; π̂ estimated from data | E[&#124;M&#124;] = Γ(π̂+n)Γ(m)/[Γ(π̂+m)Γ(n)] — grows as Θ(n^π̂) |

Testing against **both** nulls simultaneously tells a complete story:

- **Observed > Full Duplication expected**: The motif count exceeds even the maximum-inheritance null. This is biologically improbable under neutral duplication and strongly suggests *active positive selection* for this regulatory wiring pattern — perhaps because co-regulation by multiple TF families is essential for a specific cellular function.
- **Observed between the two nulls**: The motif count is consistent with partial inheritance but not with perfect copying. This is the expected neutral scenario — some links survive, some are lost.
- **Observed < Partial Duplication expected**: The motif count is *lower* than even a partial inheritance model predicts. This suggests *negative selection against* this wiring pattern, or that the families involved have undergone regulatory rewiring that actively dismantled these connections.

**Why the normal approximation is valid:**

The motif count |M(n)| is a sum of many correlated Bernoulli random variables (one per gene tuple). By the Central Limit Theorem, for reasonably large n and m, the sum converges to approximately Gaussian. The mean and variance are computed exactly from Theorems 1 and 4 (mean) and Corollaries 2 and 16 (variance), so the Z-score is:

```
Z = (observed − E[|M(n)|]) / σ
```

where σ is the standard deviation from the Binary Inheritance variance (Corollary 16).

**Why Binary Inheritance for the variance?**

Binary Inheritance is the refinement of Partial Duplication that *maximises* E[|M(n)|²] for any given π̂ (Theorem 5). This means its standard deviation provides a *conservative upper bound* — the widest possible ±2σ band consistent with the model. Using a conservative variance means the significance test is not artificially over-powered: if the Z-score is large even under this conservative variance, the departure from the null is real.
""")


        with st.spinner("Loading families…"):
            fam_df5 = _load_tf_families(min_family_size, _GROUPING_KEY)

        if fam_df5.empty:
            st.warning("No families found.")
            st.stop()

        params5 = estimate_model_parameters(fam_df5, "family_size")
        m5, n5 = params5["m"], params5["n"]

        col5a, col5b, col5c = st.columns(3)
        with col5a:
            k5 = st.slider("Motif size k", 1, min(8, m5), 2, key="k5")
        with col5b:
            strat5 = st.selectbox(
                "Family selection", ["largest", "highest_ev", "balanced", "random"],
                key="strat5",
            )
        with col5c:
            pi_method5 = st.selectbox(
                "π estimation method",
                ["Evidence-based", "Moment Estimation (Theorem 4)", "SNP proxy", "Manual"],
                key="pm5",
            )

        fams5 = select_motif_families(fam_df5, k5, strategy=strat5)
        gene_names5 = [f["go_id"] for f in fams5]
        family_sizes5 = [f["family_size"] for f in fams5]
        obs_full5 = count_observed_motif_instances(fams5)

        st.markdown(f"**Selected {k5} families:** {', '.join(gene_names5[:4])}{'…' if k5 > 4 else ''}")
        st.markdown(f"Family sizes: {family_sizes5}  |  Full-duplication count: **{obs_full5:,}**")

        # Get π
        if pi_method5 == "Evidence-based":
            pi_res5 = estimate_pi_from_evidence(gene_names5)
            pi_vec5 = pi_res5["pi_vec"]
        elif pi_method5 == "Moment Estimation (Theorem 4)":
            pi_res5 = estimate_pi_from_mle(float(obs_full5), m5, n5, gene_names5)
            pi_vec5 = pi_res5["pi_vec"]
        elif pi_method5 == "SNP proxy":
            pi_res5 = estimate_pi_from_snp(gene_names5)
            pi_vec5 = pi_res5["pi_vec"]
        else:
            st.markdown("**Manual $\\vec{\\pi}$ entry:**")
            pi_vec5 = []
            pi_cols = st.columns(k5)
            for i, col in enumerate(pi_cols):
                pi_i = col.slider(
                    f"π_{i+1}", 0.0, 1.0,
                    float(fams5[i].get("mean_evidence_score", 0.5)),
                    0.01, key=f"pi_manual_{i}",
                )
                pi_vec5.append(pi_i)

        # Observed count input
        obs5 = st.number_input(
            "Observed motif instance count |M(n)|",
            min_value=0, max_value=max(obs_full5 * 2, 10),
            value=obs_full5, step=1,
            help="Under Full Duplication this equals the Cartesian product of family sizes.",
        )

        # Run analysis
        if st.button("Run Significance Test", type="primary"):
            with st.spinner("Computing…"):
                result5 = full_significance_analysis(pi_vec5, m5, n5, float(obs5), k5)

            st.divider()
            st.subheader("Results")

            c1, c2, c3 = st.columns(3)
            c1.metric("Observed |M(n)|", f"{obs5:,}")
            c2.metric("Expected (Full Dup, Thm 1)",
                      f"{result5['expected_full']:,.2f}",
                      delta=f"{obs5 - result5['expected_full']:+.1f}")
            c3.metric(f"Expected (Partial Dup, Thm 4, π̂={result5['pi_hat']})",
                      f"{result5['expected_partial']:,.2f}",
                      delta=f"{obs5 - result5['expected_partial']:+.1f}")

            c4, c5, c6, c7 = st.columns(4)
            c4.metric("Z-score (Full Dup)", result5["z_full"])
            c5.metric("p-value (Full Dup)", result5["p_full"])
            c6.metric("Z-score (Partial Dup)", result5["z_partial"])
            c7.metric("p-value (Partial Dup)", result5["p_partial"])

            sig_col1, sig_col2 = st.columns(2)
            with sig_col1:
                sig = result5["sig_full"]
                colour = "🟢" if sig == "***" else "🟡" if sig in ("**", "*") else "⚪"
                st.markdown(f"**Full Duplication significance:** {colour} `{sig}`")
            with sig_col2:
                sig2 = result5["sig_partial"]
                colour2 = "🟢" if sig2 == "***" else "🟡" if sig2 in ("**", "*") else "⚪"
                st.markdown(f"**Partial Duplication significance:** {colour2} `{sig2}`")

            st.markdown(
                f'<div class="result-box">{result5["interpretation"]}</div>',
                unsafe_allow_html=True,
            )

            st.divider()
            st.subheader("Variance Decomposition")
            var_df = pd.DataFrame([
                {
                    "Model": "Full Duplication (Corollary 2)",
                    "Expected": result5["expected_full"],
                    "Std Dev": result5["std_full"],
                    "Variance": result5["variance_full"],
                    "Lower (−2σ)": result5["expected_full"] - 2 * result5["std_full"],
                    "Upper (+2σ)": result5["expected_full"] + 2 * result5["std_full"],
                },
                {
                    "Model": "Binary Inheritance (Corollary 16)",
                    "Expected": result5["expected_partial"],
                    "Std Dev": result5["std_binary"],
                    "Variance": result5["variance_binary"],
                    "Lower (−2σ)": result5["expected_partial"] - 2 * result5["std_binary"],
                    "Upper (+2σ)": result5["expected_partial"] + 2 * result5["std_binary"],
                },
            ])
            st.dataframe(var_df.set_index("Model"), use_container_width=True)

            fig5 = go.Figure()
            for label, mean_, std_, colour_ in [
                ("Full Dup null", result5["expected_full"], result5["std_full"], "#1f77b4"),
                ("Partial Dup null", result5["expected_partial"], result5["std_binary"], "#ff7f0e"),
            ]:
                x_range = np.linspace(max(0, mean_ - 4 * std_), mean_ + 4 * std_, 200)
                y_range = np.exp(-0.5 * ((x_range - mean_) / max(std_, 1e-6)) ** 2)
                y_range /= y_range.max()
                fig5.add_trace(go.Scatter(
                    x=x_range, y=y_range, name=label,
                    line=dict(color=colour_, width=2), fill="tozeroy",
                    fillcolor=colour_.replace(")", ",0.15)").replace("rgb(", "rgba("),
                ))
            fig5.add_vline(x=obs5, line_dash="dash", line_color="red", line_width=2,
                           annotation_text=f"Observed = {obs5}")
            fig5.update_layout(title="Observed Count vs Null Model Distributions",
                               xaxis_title="|M(n)| count", yaxis_title="Relative density",
                               height=320, margin=dict(t=50, b=30))
            st.plotly_chart(fig5, use_container_width=True)

            p_range = np.linspace(0.05, 1.0, 20)
            s_range = list(range(1, 21))
            f_matrix = np.array([[f_func(p, s) for s in s_range] for p in p_range])
            fig_f = px.imshow(f_matrix,
                              x=[str(s) for s in s_range], y=[f"{p:.2f}" for p in p_range],
                              labels={"x": "Family size s", "y": "Inheritance prob p", "color": "f(p,s)"},
                              title="f(p,s) = Γ(p+s) / [Γ(s)Γ(p+1)]  (Lemma 4)",
                              color_continuous_scale="Blues", aspect="auto")
            fig_f.update_layout(height=320, margin=dict(t=50, b=20))
            st.plotly_chart(fig_f, use_container_width=True)

            with st.expander("🔍 What do these results mean?"):
                st.markdown(f"""
**Interpreting the significance test output:**

**Z-scores** measure how many standard deviations the observed motif count
({obs5:,}) falls from the expected count under each null model.
A Z-score above +2 means the motif is significantly *over-represented*;
below −2 means it is significantly *under-represented*.

| Significance code | p-value threshold | Interpretation |
|---|---|---|
| `***` | p < 0.001 | Very strong evidence against the null model |
| `**` | p < 0.01 | Strong evidence |
| `*` | p < 0.05 | Moderate evidence |
| `ns` | p ≥ 0.05 | No significant departure from null |

**Full Duplication null (Theorem 1):**
Expected {result5['expected_full']:,.2f} motifs if π = 1 (every regulatory link
perfectly inherited). Z = {result5['z_full']} ({result5['sig_full']}).
{"The observed count is **above** the Full Duplication expectation — more motifs than even perfect inheritance predicts, suggesting this pattern is actively selected for." if obs5 > result5['expected_full'] else "The observed count is **below** the Full Duplication expectation — some regulatory links have been lost or the motif is under negative selection."}

**Partial Duplication null (Theorem 4, π̂ = {result5['pi_hat']}):**
Expected {result5['expected_partial']:,.2f} motifs under the estimated inheritance rate.
Z = {result5['z_partial']} ({result5['sig_partial']}).
{"The observed count is **consistent** with Partial Duplication at this π̂ — the model explains the data well." if abs(float(result5['z_partial'])) < 2 else "The observed count **departs significantly** from the Partial Duplication expectation — reconsider the π̂ estimate or check whether this motif class follows the duplication model."}

**Variance Decomposition:**
The Full Duplication standard deviation (σ = {result5['std_full']:.2f}) gives the
±2σ band shown in the chart. The Binary Inheritance standard deviation
(σ = {result5['std_binary']:.2f}) provides a conservative upper bound on variance under
any Partial Duplication refinement (Theorem 5).

**Bottom line:** {result5['interpretation']}

---

**Why are the Full Duplication, Partial Duplication, and Binary Inheritance numbers so different?**

Each model makes a fundamentally different assumption about what happens to regulatory links after a gene duplication event, and those assumptions lead to dramatically different expected motif counts — especially at the extremes.

- **Full Duplication (π = 1, Theorem 1)** assumes every single regulatory link is perfectly copied to the new daughter gene. Expected motif count grows as Θ(nᵏ) — that is, *polynomially in n with degree equal to the motif size k*. For a 2-family motif in a network of even moderate size, this produces enormous expected counts. It represents the theoretical maximum: a world of perfect regulatory memory where nothing is ever lost.

- **Partial Duplication (0 ≤ π̂ < k, Theorem 4)** is the realistic middle ground. Each link survives duplication independently with probability πᵢ, so the total inheritance π̂ = Σπᵢ controls the growth exponent. Expected count grows as Θ(n^π̂) — *sub-polynomial if π̂ < k*. Because π̂ is typically well below k (most links are not perfectly conserved), the expected count under Partial Duplication can be orders of magnitude smaller than Full Duplication. The gap between the two grows exponentially with network size n, which is why even a modest difference in π̂ produces enormous differences in expected motif count.

- **Binary Inheritance (Theorem 5 / Corollary 16)** is not a separate biological model — it is a mathematical tool. Among all Partial Duplication refinements that share the same π̂, Binary Inheritance (where each πᵢ is either 0 or 1) *maximises the second moment* E[|M(n)|²]. This means it provides the *most conservative* (widest) variance bound, so the ±2σ bands shown in the chart represent a worst-case spread rather than the actual distribution width. The Binary Inheritance variance can be much larger than the Full Duplication variance because variance in this model comes from the binary all-or-nothing nature of link retention.

In short: the three numbers diverge so starkly because (a) the growth exponents differ by the gap k − π̂, which magnifies with n, and (b) Binary Inheritance maximises spread while Full Duplication maximises the mean. Seeing the observed count far above the Partial Duplication expectation — as here — is a strong signal that selective pressure is actively maintaining this regulatory wiring beyond what neutral duplication-and-loss would produce.
""")

            with st.expander("📋 Full result JSON"):
                st.json(result5)

            st.divider()
            st.info(
                "**Is this predictive?**  \n"
                "The estimator is **inferential, not predictive**. It characterises the *current* "
                "state of the yeast GRN — how much of its regulatory structure is consistent with "
                "inherited duplication links — rather than forecasting the outcome of a specific "
                "future duplication event. The estimated $\\vec{\\pi}$ can be interpreted as: "
                "*given the observed network, what inheritance rate would a duplication model need "
                "to produce networks like this one?* "
                "To see forward projections under continued duplication, switch to the "
                "**🔮 Predictive Forecast** tab.",
                icon="ℹ️",
            )

    # ── Predictive Forecast ──────────────────────────────────────────
    with pred_tab:
        st.markdown("""
### 🔮 Predictive Forecast
**How is this different from the π Estimator tab?**

| | π Estimator | Predictive Forecast |
|---|---|---|
| **Question asked** | What is π now? | What will the network look like later? |
| **Direction** | Backward — fits π to the *current* observed network | Forward — projects *future* motif counts given π |
| **Output** | $\\vec{\\pi}$ values and sensitivity curves | Growth trajectories and target-count queries |
| **Use case** | Characterise the current GRN | Forecast evolutionary outcomes under duplication |

This tab takes the $\\vec{\\pi}$ you estimate from real data and uses it to project how the
expected number of regulatory motifs will grow as the yeast genome expands through
further gene duplication events.

> *Given the current inheritance rate, how many motif instances should we expect
> after N more duplications? And how large would the genome need to be to reach
> a target count?*
""")

        st.info(
            "**Forward projection** under the Partial Duplication model (Theorem 4). "
            "Assumes $\\vec{\\pi}$ remains stable and that each duplication event adds one TF "
            "to the network (n increases by 1 per event). "
            "The Inferential Test tab tells you *where you are*; this tab tells you *where you're going*.",
            icon="🔮",
        )

        with st.expander("📖 Methodology — how the predictive forecast is computed", expanded=False):
            st.markdown("""
**What this tab computes and why the calculation is valid:**

The Inferential Test tab works *backward* — it takes the observed motif count and fits π̂ to it.
This tab works *forward* — it takes a π̂ estimate and projects how motif counts will grow as the
genome expands through further duplication events.

**The growth equation (Theorem 4 applied forward):**

At any genome stage n, Theorem 4 gives:

```
E[|M(n)|] = Γ(π̂ + n) · Γ(m) / [Γ(π̂ + m) · Γ(n)]
```

This is evaluated at increasing values of n = n_current, n_current + 1, …, n_current + Δn to trace
the expected motif count trajectory. The formula is exact — not an approximation — and each point
on the curve is the theoretical mean under the Partial Duplication model at that genome size.

**Why the two curves diverge:**

- **Full Duplication** (blue dashed): grows as Θ(nᵏ) — a polynomial of degree k. This is the
  maximum possible trajectory, assuming every regulatory link is perfectly copied at every duplication.
- **Partial Duplication** (red): grows as Θ(n^π̂) — a polynomial of degree π̂. Because π̂ ≤ k,
  this curve always stays below (or equals) the Full Duplication curve. The gap between them widens
  exponentially with n — a small difference in growth exponent becomes enormous at large genome sizes.
  This divergence quantifies how much regulatory structure is lost relative to a hypothetical world
  of perfect inheritance.

**What the ±2σ band means (Full Duplication only):**

The shaded region around the Full Duplication curve is ±2 standard deviations computed from
Corollary 2 (Full Duplication variance). This represents the expected *natural sampling variability*
around the mean under Full Duplication — the range within which ~95% of observed counts would fall
if the network truly evolved under π = 1. It is not plotted for Partial Duplication because the
Partial Duplication variance (Binary Inheritance, Corollary 16) would dominate the chart at large n.

**The inverse query — why it makes sense:**

Given a target motif count T and a fixed π̂, the inverse query solves:

```
Γ(π̂ + n) · Γ(m) / [Γ(π̂ + m) · Γ(n)] = T    →    find n
```

This is solved numerically by scanning the forward trajectory and finding the first n where the
expected count exceeds T. The result answers: *how much larger would the genome need to be, under
continued duplication at the current inheritance rate, to generate this many motif instances on average?*
This is useful for calibrating what π̂ estimates imply about long-term regulatory network complexity.

**Key assumption — π̂ is stable over time:**

The model assumes the inheritance probability vector $\\vec{\\pi}$ does not change as the genome grows.
In reality, π may evolve: as TF paralogs diverge further in sequence, their binding specificities may
change, potentially lowering π over time. Treat the forecast as a projection under constant selective
pressure, not a precise biological prediction.
""")

        with st.spinner("Loading families…"):
            fam_df_p = _load_tf_families(min_family_size, _GROUPING_KEY)

        if fam_df_p.empty:
            st.warning("No families found.")
            st.stop()

        params_p = estimate_model_parameters(fam_df_p, "family_size")
        m_p, n_p = params_p["m"], params_p["n"]

        st.divider()
        st.subheader("Step 1 — Configure the motif")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            k_p = st.slider("Motif size k", 1, min(8, m_p), 2, key="k_pred")
        with pc2:
            strat_p = st.selectbox(
                "Family selection", ["largest", "highest_ev", "balanced", "random"],
                key="strat_pred",
            )
        with pc3:
            pi_method_p = st.selectbox(
                "π estimation method",
                ["Evidence-based", "Moment Estimation (Theorem 4)", "Manual"],
                key="pm_pred",
            )

        fams_p = select_motif_families(fam_df_p, k_p, strategy=strat_p)
        gene_names_p = [f["go_id"] for f in fams_p]
        obs_full_p = count_observed_motif_instances(fams_p)

        if pi_method_p == "Evidence-based":
            pi_res_p = estimate_pi_from_evidence(gene_names_p)
            pi_vec_p = pi_res_p["pi_vec"]
        elif pi_method_p == "Moment Estimation (Theorem 4)":
            pi_res_p = estimate_pi_from_mle(float(obs_full_p), m_p, n_p, gene_names_p)
            pi_vec_p = pi_res_p["pi_vec"]
        else:
            st.markdown("**Manual $\\vec{\\pi}$ entry:**")
            pi_vec_p = []
            pcols = st.columns(k_p)
            for i, col in enumerate(pcols):
                pi_i = col.slider(
                    f"π_{i+1}", 0.0, 1.0,
                    float(fams_p[i].get("mean_evidence_score", 0.5)),
                    0.01, key=f"pi_pred_{i}",
                )
                pi_vec_p.append(pi_i)

        pi_hat_p = sum(pi_vec_p)
        st.markdown(
            f"Selected **{k_p} families**: {', '.join(gene_names_p[:4])}{'…' if k_p > 4 else ''}  \n"
            f"Current n = **{n_p}** TFs · π̂ = **{pi_hat_p:.4f}**"
        )

        st.divider()
        st.subheader("Step 2 — Set the forecast horizon")
        ph1, ph2 = st.columns(2)
        with ph1:
            extra_dups = st.slider(
                "Additional duplication events (Δn)",
                min_value=10, max_value=2000, value=200, step=10,
                help="Each event adds one TF to the network (n increases by 1).",
            )
        with ph2:
            target_count = st.number_input(
                "Target motif count (optional inverse query)",
                min_value=0, value=0, step=100,
                help="If > 0, the chart will mark the n needed to reach this count.",
            )

        # Build trajectory
        n_values = list(range(n_p, n_p + extra_dups + 1, max(1, extra_dups // 100)))
        e_full_traj    = [expected_full(k_p, m_p, nv) for nv in n_values]
        e_partial_traj = [expected_partial(pi_hat_p, m_p, nv) for nv in n_values]
        std_full_traj  = [variance_full(k_p, m_p, nv) ** 0.5 for nv in n_values]

        traj_df = pd.DataFrame({
            "n (total TFs)":          n_values,
            "E[|M|] Full Dup":        e_full_traj,
            "E[|M|] Partial Dup":     e_partial_traj,
            "±2σ upper (Full Dup)":   [e + 2*s for e, s in zip(e_full_traj, std_full_traj)],
            "±2σ lower (Full Dup)":   [max(0, e - 2*s) for e, s in zip(e_full_traj, std_full_traj)],
        })

        # Growth trajectory chart
        fig_pred = go.Figure()

        # Full Dup ±2σ band
        fig_pred.add_trace(go.Scatter(
            x=n_values + n_values[::-1],
            y=traj_df["±2σ upper (Full Dup)"].tolist() + traj_df["±2σ lower (Full Dup)"].tolist()[::-1],
            fill="toself", fillcolor="rgba(31,119,180,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="±2σ Full Dup",
            showlegend=True,
        ))
        fig_pred.add_trace(go.Scatter(
            x=n_values, y=e_full_traj,
            name=f"Full Duplication (π=1)",
            line=dict(color="#1f77b4", width=2, dash="dash"),
        ))
        fig_pred.add_trace(go.Scatter(
            x=n_values, y=e_partial_traj,
            name=f"Partial Duplication (π̂={pi_hat_p:.3f})",
            line=dict(color="#d62728", width=2.5),
        ))

        # Mark current n
        fig_pred.add_vline(
            x=n_p, line_dash="dot", line_color="grey",
            annotation_text=f"Current n={n_p}",
        )

        # Mark target count if specified
        if target_count > 0:
            # Find the n where partial dup first exceeds target
            n_target = None
            for nv, ev in zip(n_values, e_partial_traj):
                if ev >= target_count:
                    n_target = nv
                    break
            if n_target:
                fig_pred.add_hline(
                    y=target_count, line_dash="dot", line_color="#2ca02c",
                    annotation_text=f"Target: {target_count:,}",
                )
                fig_pred.add_vline(
                    x=n_target, line_dash="dot", line_color="#2ca02c",
                    annotation_text=f"n≈{n_target} needed",
                )

        fig_pred.update_layout(
            title=f"Predicted Motif Count Growth — k={k_p} motif, π̂={pi_hat_p:.3f}",
            xaxis_title="n (total TFs in network)",
            yaxis_title="E[|M(n)|]",
            height=380, margin=dict(t=50, b=30),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Prediction summary at key horizons
        st.subheader("Forecast Summary")
        horizons = [n_p, n_p + extra_dups // 4, n_p + extra_dups // 2,
                    n_p + 3 * extra_dups // 4, n_p + extra_dups]
        rows = []
        for nv in horizons:
            ef = expected_full(k_p, m_p, nv)
            ep = expected_partial(pi_hat_p, m_p, nv)
            sf = variance_full(k_p, m_p, nv) ** 0.5
            rows.append({
                "n (total TFs)":              nv,
                "Δn (extra duplications)":    nv - n_p,
                "E[|M|] Full Dup":            round(ef, 2),
                "E[|M|] Partial Dup (π̂)":    round(ep, 4),
                "Fold change vs current":     round(ep / max(expected_partial(pi_hat_p, m_p, n_p), 1e-9), 2),
                "±2σ Full Dup":               f"{max(0, ef-2*sf):.1f} – {ef+2*sf:.1f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("n (total TFs)"), use_container_width=True)

        # Inverse query result
        if target_count > 0:
            if n_target:
                st.success(
                    f"To reach **{target_count:,}** expected motif instances under "
                    f"Partial Duplication (π̂ = {pi_hat_p:.4f}), the network would need "
                    f"approximately **n ≈ {n_target}** TFs "
                    f"(**{n_target - n_p} more duplications** beyond current n = {n_p}).",
                    icon="🎯",
                )
            else:
                st.warning(
                    f"The target count of {target_count:,} is not reached within "
                    f"the forecast horizon (n = {n_p + extra_dups}). "
                    f"Try increasing Δn or lowering the target.",
                    icon="⚠️",
                )


# ══════════════════════════════════════════════════════════════════════
# TAB 8: Glossary & References  (always last)
# ══════════════════════════════════════════════════════════════════════

with tab8:
    st.header("📖 Glossary & References")
    st.markdown(
        "Definitions of key terms used throughout the app, and the primary papers "
        "that underpin this model."
    )

    # ── Reference document ────────────────────────────────────────────
    st.subheader("📄 Method Reference Document")
    with st.container(border=True):
        st.markdown(
            "**Estimation Methods for Regulatory Inheritance Probability in Yeast Gene Duplication**  \n"
            "A self-contained reference covering all seven estimation methods (M1–M7) and the "
            "multi-signal ensemble: notation tables, data sources, strengths & limitations, "
            "accuracy rankings (per-family πᵢ and aggregate π̂), and practical recommendations "
            "for choosing an estimator."
        )
        _pdf_path = Path(__file__).parent / "static" / "Pi_estimation.pdf"
        if _pdf_path.exists():
            with open(_pdf_path, "rb") as _pdf_fh:
                _pdf_bytes = _pdf_fh.read()
            _pdf_b64 = base64.b64encode(_pdf_bytes).decode()
            _col_dl, _col_open = st.columns(2)
            with _col_dl:
                st.download_button(
                    label="⬇️ Download Pi_estimation.pdf",
                    data=_pdf_bytes,
                    file_name="Pi_estimation.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with _col_open:
                components.html(
                    f"""
                    <script>
                    function openPDF() {{
                        const b64 = "{_pdf_b64}";
                        const binary = atob(b64);
                        const bytes = new Uint8Array(binary.length);
                        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
                        const blob = new Blob([bytes], {{type: "application/pdf"}});
                        window.open(URL.createObjectURL(blob), "_blank");
                    }}
                    </script>
                    <button onclick="openPDF()" style="width:100%;padding:0.4rem 0.8rem;
                        background:#FF4B4B;color:white;border:none;border-radius:0.4rem;
                        cursor:pointer;font-size:0.9rem;font-family:sans-serif;">
                        ↗️ Open in new tab
                    </button>
                    """,
                    height=45,
                )
        else:
            st.warning("Pi_estimation.pdf not found in static/.", icon="⚠️")

    st.divider()

    # ── References ────────────────────────────────────────────────────
    st.subheader("📚 Primary References")

    with st.container(border=True):
        st.markdown("#### 1. Scruse, Arnold & Robinson (2026) — *The Model Paper*")
        st.markdown("""
**Counting Subnetworks Under Gene Duplication in Genetic Regulatory Networks**
Ashley Scruse, Jonathan Arnold, Robert Robinson
*Bulletin of Mathematical Biology 88, 31 (2026) · University of Georgia · doi:10.1007/s11538-025-01592-1*

This is the primary theoretical paper implemented in this app. It introduces the
gene duplication and inheritance model, defines **subnetwork motifs**, and derives
the exact moments (mean and variance) for their count under both Full and Partial
Duplication via combinatorial probability and generating functions.
        """)
        st.link_button("View paper (doi:10.1007/s11538-025-01592-1)", "https://doi.org/10.1007/s11538-025-01592-1")

    with st.container(border=True):
        st.markdown("#### 2. Harbison et al. (2004) — *Yeast Transcriptional Regulatory Code*")
        st.markdown("""
**Transcriptional regulatory code of a eukaryotic genome**
Christopher T. Harbison, D. Benjamin Gordon, Tong Ihn Lee, et al.
*Nature 431, 99–104 · 2 September 2004 · doi:10.1038/nature02800*

Constructed an initial genome-wide map of the yeast transcriptional regulatory code by
identifying DNA sequence elements bound by 203 transcription factors under multiple
conditions. Used genome-wide location analysis (ChIP-chip), phylogenetically conserved
sequences, and prior knowledge to produce a compendium of sequence motifs for
102 regulators. This paper is a key empirical source for the **transcription factor binding
site** data used to characterise regulatory edges (TF → target gene) in this app.
        """)
        st.link_button("View paper (doi:10.1038/nature02800)", "https://doi.org/10.1038/nature02800")

    with st.container(border=True):
        st.markdown("#### 3. Ren et al. (2000) — *Genome-Wide Location Analysis*")
        st.markdown("""
**Genome-Wide Location and Function of DNA Binding Proteins**
Bing Ren, François Robert, John J. Wyrick, et al.
*Science 290, 2306–2309 · 22 December 2000 · doi:10.1126/science.290.5500.2306*

Introduced the **genome-wide location analysis** (ChIP-chip) method — combining
chromatin immunoprecipitation with DNA microarray hybridisation — to monitor protein-DNA
interactions across the entire yeast genome. Applied the method to the transcriptional
activators **Gal4** and **Ste12**, revealing direct regulatory targets and coordinated
pathway control. This methodology underlies the experimental data from which TF binding
interactions in this app are ultimately derived.
        """)
        st.link_button("View paper (doi:10.1126/science.290.5500.2306)", "https://doi.org/10.1126/science.290.5500.2306")

    st.divider()

    # ── Glossary ──────────────────────────────────────────────────────
    st.subheader("🔤 Glossary of Terms")
    st.caption("Alphabetically ordered. Click any section to expand.")

    with st.expander("Binary Inheritance"):
        st.markdown("""
A specific **refinement of Partial Duplication** (Definition 1, Scruse et al.) in which,
when a gene *b* in family *i* is duplicated, all subnetwork motif instances that share *b*
either all inherit the new instance or none do — with probability πᵢ.

Binary Inheritance is chosen as the working model because it is tractable and, by
**Theorem 5**, gives the **maximum second moment** of |M(n)| over all refinements
of Partial Duplication. This makes the variance and significance bounds conservative.
        """)

    with st.expander("Binding Site / TFBS (Transcription Factor Binding Site)"):
        st.markdown("""
A short DNA sequence motif — typically 6–20 bp — located in the **promoter region**
of a gene (usually 100–500 bp upstream of the coding sequence). A transcription factor
(TF) recognises and binds its cognate TFBS sequence via a DNA-binding domain.

- Binding **activates** or **represses** transcription of the downstream gene.
- TFBS sequences are represented compactly as **Position Frequency Matrices (PFMs)**
  in databases such as **JASPAR**.
- In the Scruse et al. framework, the TF → target regulatory edge is the unit whose
  **inheritance probability πᵢ** the model estimates.
        """)

    with st.expander("Composition (of n into m parts)"):
        st.markdown("""
A way of writing a positive integer *n* as an ordered sum of *m* positive integers:
*n* = c₁ + c₂ + ⋯ + cₘ, where each cᵢ ≥ 1.

In this model, **cᵢ is the size of the i-th gene family** after duplication.
**Proposition 1** (Scruse et al.) proves that after *n − m* duplications starting from
*m* singletons, all C(n−1, m−1) compositions are **equally likely**.

This uniform distribution is the probabilistic foundation of the entire model.
        """)

    with st.expander("Evidence Code (SGD)"):
        st.markdown("""
A structured vocabulary tag assigned by the **Saccharomyces Genome Database (SGD)**
to each gene-function annotation, indicating how the annotation was determined.

Common codes used in this app:

| Code | Method | π weight |
|------|--------|---------|
| **IDA** | Inferred from Direct Assay | 0.90 |
| **IMP** | Inferred from Mutant Phenotype | 0.82 |
| **IPI** | Inferred from Physical Interaction | 0.68 |
| **HDA** | High-throughput Direct Assay | 0.55 |
| **IEA** | Inferred from Electronic Annotation | 0.30 |

The **evidence score** in this app is the mean weight across all codes for a TF,
used as a prior for the inheritance probability π.
        """)

    with st.expander("Expected Motif Count  E[|M(n)|]"):
        st.markdown(r"""
The **expected number of subnetwork motif instances** at stage *n* (i.e., after *d = n − m*
duplication events), derived analytically in Scruse et al.:

- **Full Duplication** (Theorem 1):
  `E[|M(n)|; k, m, n] = Γ(n+k)Γ(m) / [Γ(n)Γ(m+k)]`
  Grows as Θ(nᵏ).

- **Partial Duplication** (Theorem 4):
  `E[|M(n)|; m, n, π̂, k] = Γ(π̂+n)Γ(m) / [Γ(π̂+m)Γ(n)]`
  Depends on $\\vec{\\pi}$ only through π̂ = Σπᵢ; grows as Θ(n^π̂).

These expectations form the **null model** against which an observed motif count is tested.
        """)

    with st.expander("f(p, s) — Single-gene motif expectation"):
        st.markdown(r"""
Defined in **Lemma 4** (Scruse et al.):

`f(p, s) = Γ(p + s) / [Γ(s) Γ(p + 1)]`

- *p* = inheritance probability for that gene family
- *s* = current size of the gene family

`f(p, s)` is the expected number of instances of a **single-gene subnetwork motif**
conditioned on the family having size *s*. It satisfies the recurrence
`f(p, s+1) = f(p, s) × (p+s)/s`, which drives the induction in Theorem 3.
        """)

    with st.expander("Full Duplication"):
        st.markdown("""
The inheritance mode in which **every regulatory link is perfectly copied** at each
duplication step — i.e., the inheritance probability vector $\\vec{\\pi}$ = (1, 1, …, 1).

Under Full Duplication, the set of motif instances at stage *n* equals the full
Cartesian product of the relevant gene families:
M(n) = X¹ₘ,ₙ × … × Xᵏₘ,ₙ

Full Duplication is the **upper-bound null model**: if an observed motif count is
significantly *above* the Full Duplication expectation, it strongly suggests active
selection. If it is *below*, regulatory links are being lost faster than the neutral
duplication rate.

**Full Duplication is a special case of Partial Duplication** with $\\vec{\\pi}$ = $\\vec{1}$.
        """)

    with st.expander("Gene Duplication"):
        st.markdown("""
A fundamental evolutionary mechanism in which a segment of DNA containing a gene
is copied, producing two copies in the genome. The new copy (**paralog**) initially
shares the same sequence and regulatory connections as the original.

Over time the two copies may:
- **Subfunctionalise** — divide the original gene's functions between them.
- **Neofunctionalise** — one copy acquires a new function.
- **Pseudogenise** — one copy accumulates mutations and becomes non-functional.

This model focuses on how **regulatory connections** (not just sequence) are inherited
through duplication, which is less studied than sequence duplication.
        """)

    with st.expander("Gene Family"):
        st.markdown("""
In this app, a gene family is the set of transcription factors that share a common
**GO Biological Process** term. This operationalises the gene family concept from
Scruse et al.: all TFs in the same family are assumed to share a common ancestor
and are therefore related by duplication.

Model parameters:
- **m** = number of distinct gene families
- **cᵢ** = size of the i-th family (number of TFs in it)
- **n** = Σcᵢ = total number of TFs across all families

The family composition vector **$\\vec{c}$ = (c₁, …, cₘ)** is a composition of *n* into *m* parts.
        """)

    with st.expander("GRN — Genetic Regulatory Network"):
        st.markdown("""
A **Genetic Regulatory Network (GRN)** is a collection of genes and their molecular
products (proteins, RNAs) that interact to control the expression of one another,
governing a specific cellular function or state.

In the context of this app:
- **Nodes** = transcription factor genes and their target genes
- **Edges** = regulatory relationships (TF binds promoter → activates/represses target)
- The **yeast GRN** is inferred from JASPAR PFM data, YEASTRACT consensus sequences,
  and SGD GO annotations for *Saccharomyces cerevisiae*

GRNs are characterised by recurring structural patterns called **network motifs** (or,
in this framework, **subnetwork motifs**) that reflect the evolutionary history of the
network.
        """)

    with st.expander("Inheritance Probability  π (pi)"):
        st.markdown("""
The probability that a **regulatory link is retained** when the gene it connects to is
duplicated. Formally, πᵢ is the probability that a new instance of subnetwork motif M
is inherited in family *i* at any given duplication step.

- **π = 1**: Full Duplication — every link is always inherited.
- **π = 0**: No inheritance — regulatory links are never passed to duplicate copies.
- **0 < π < 1**: Partial Duplication — stochastic inheritance.

The vector **$\\vec{\\pi}$ = (π₁, …, πₖ)** characterises the inheritance behaviour of a motif M
across its *k* gene families. The model shows (Theorem 4) that the expected motif count
depends on $\\vec{\\pi}$ only through the scalar **π̂ = Σπᵢ**.

In this app, π is estimated from SGD evidence codes, Moment Estimation on observed counts, SNP
divergence proxies, or YEASTRACT consensus binding data.
        """)

    with st.expander("JASPAR"):
        st.markdown("""
**JASPAR** (jaspar.elixir.no) is the open-access database of **transcription factor
binding profiles** (Position Frequency Matrices). It stores experimentally validated
PFMs representing the sequence preferences of TF DNA-binding domains.

Key statistics used in this app (JASPAR 2024):
- 177 *S. cerevisiae* TFs with PFMs
- ChIP-based and PBM-based experimental methods
- Mean motif width ~10 bp; mean information content ~12 bits

JASPAR PFMs are used to adjust the inheritance probability prior: TFs with high-IC,
narrow motifs have more specific binding and are expected to be harder to disrupt
by duplication (higher πᵢ).
        """)

    with st.expander("k — Motif Size"):
        st.markdown("""
The number of **distinct gene families** involved in a subnetwork motif M.

- A **k = 1** motif involves one gene family (a single TF family whose members all
  regulate a shared target).
- A **k = 2** motif involves two families (two distinct TF families co-regulating
  a target).
- Higher *k* motifs are rarer but more informative about correlated regulatory evolution.

The expected count E[|M(n)|] grows as **Θ(nᵏ)** under Full Duplication and
**Θ(n^π̂)** under Partial Duplication — the motif size determines the polynomial
growth rate.
        """)

    with st.expander("m — Number of Gene Families"):
        st.markdown("""
The number of distinct **gene families** in the model. Denoted *m*, it is the
number of parts in the composition of *n*.

In the gene duplication process, the model starts with *m* singleton genes (one per
family) and grows to *n* total genes through *d = n − m* duplication events.

Proposition 1 shows that given *m* and *n*, all C(n−1, m−1) compositions of *n* into
*m* parts are equally likely — the key uniformity result underlying the exact formulas.
        """)

    with st.expander("Network Motif vs Subnetwork Motif"):
        st.markdown("""
| Concept | Network Motif | Subnetwork Motif |
|---------|---------------|-----------------|
| **Definition** | Patterns of interconnection that recur more often in a complex network than in a random network (Milo et al. 2002) | Gene-family-specific substructures — instances must involve specific labelled gene families |
| **Specificity** | Isomorphic subgraphs (family labels ignored) | Family-labelled subgraphs (family identity matters) |
| **Evolutionary info** | Does not incorporate evolutionary history | Incorporates inheritance probabilities π from duplication history |
| **Statistical test** | Compared to randomised networks | Compared to duplication-model null (this app) |

In Figure 1 of Scruse et al.: subgraphs A, B, C, D are the same **network motif**
(3-node, isomorphic) but four different **subnetwork motifs** because the coloured
gene families differ.
        """)

    with st.expander("n — Total Gene Count / Stage"):
        st.markdown("""
The total number of genes (or TFs) in the model at a given point in the duplication
process. Also called the **stage** of the duplication process.

- Starts at *n = m* (one gene per family, no duplications yet).
- Each duplication event increments *n* by 1.
- After *d* duplications: *n = m + d*.

The Scruse et al. formulas are expressed in terms of *n*, so the model output
(expected motif count) is a function of genome size.
        """)

    with st.expander("Partial Duplication"):
        st.markdown("""
The general inheritance model in which **regulatory links are inherited stochastically**
at each duplication step, controlled by the vector $\\vec{\\pi}$ = (π₁, …, πₖ).

At each step, if a gene in family *i* is duplicated (1 ≤ i ≤ k), each existing motif
instance that includes that gene produces a new instance with probability πᵢ.

Key result (**Theorem 4**):

`E[|M(n)|; m, n, π, k] = Γ(π̂+n)Γ(m) / [Γ(π̂+m)Γ(n)]`

where π̂ = π₁ + … + πₖ. The expected count depends on $\\vec{\\pi}$ only through the scalar π̂,
which determines the polynomial growth rate Θ(n^π̂).

Full Duplication ($\\vec{\\pi}$ = **1**, π̂ = k) is a special case.
        """)

    with st.expander("PFM — Position Frequency Matrix"):
        st.markdown("""
A matrix representation of a **transcription factor binding site** motif. Rows
correspond to the four nucleotides (A, C, G, T); columns to positions in the motif.
Each entry gives the count (or frequency) of that nucleotide at that position across
all experimentally observed binding instances.

From a PFM one can compute:
- **Consensus sequence** — most frequent nucleotide at each position
- **Information content (IC, bits)** — how conserved each position is (max = 2 bits)
- **Position Weight Matrix (PWM)** — log-likelihood scores for sequence scanning

Higher IC indicates a more specific, conserved binding sequence, which in this model
correlates with a higher inheritance probability prior (the binding sequence is harder
to disrupt by mutation after duplication).
        """)

    with st.expander("π̂ (pi-hat) — Total Inheritance Probability"):
        st.markdown(r"""
The scalar sum of the inheritance probability vector:

**π̂ = π₁ + π₂ + ⋯ + πₖ**

This single number fully determines the expected motif count under Partial Duplication
(Theorem 4). It ranges from 0 (no inheritance) to *k* (Full Duplication).

The growth rate of E[|M(n)|] with respect to *n* is exactly **n^π̂**, so:
- π̂ < 1: sub-linear growth (motif count grows slower than genome)
- π̂ = 1: linear growth
- π̂ = k: polynomial growth of degree k (Full Duplication)

The **π̂ Sensitivity Analysis** chart on the π Estimator tab shows E[|M(n)|] as a
function of π̂ for fixed *m*, *n*, and *k*.
        """)

    with st.expander("Pólya Urn Model"):
        st.markdown("""
A classical probability model for reinforcement processes. An urn contains balls of
different colours; at each step, one ball is drawn at random, and a new ball of the
same colour is added.

**Connection to gene duplication (Section 7.1 / Theorem 8):**

The gene duplication process is equivalent to a Pólya urn where:
- **Urn** = the genome
- **Balls** = genes
- **Colours** = gene families

**Theorem 8** gives the exact probability:

P[$\\vec{X}$ = $\\vec{t}$ | $\\vec{s}$] = C(n−m; t₁,…,tω) × (m−1)! / (n−1)! × Π(sⱼ+tⱼ−1)! / (sⱼ−1)!

Family proportions cᵢ/n converge almost surely to a **Dirichlet(1,…,1)** distribution
as n → ∞, giving the model its exact probabilistic grounding. This also connects it to
the Kriz multi-urn model for disease spread.
        """)

    with st.expander("SGD — Saccharomyces Genome Database"):
        st.markdown("""
The **Saccharomyces Genome Database** (yeastgenome.org) is the primary curated
knowledgebase for the biology of the budding yeast *Saccharomyces cerevisiae*.

Data used in this app from SGD:
- **GO annotations** (~120,000 records) for gene function, process, and component
- **TF annotations** with evidence codes (IDA, IMP, IEA, …)
- **Chromosome lengths** (16 chromosomes, ~12 Mb genome)
- **Gene IDs, names, and synonyms**
        """)

    with st.expander("Significance Test (Z-score / p-value)"):
        st.markdown("""
The **significance test** in this app asks: is the observed subnetwork motif count
|M(n)| consistent with the null model (Full or Partial Duplication), or is it
significantly over- or under-represented?

**Z-score:**
`Z = (observed − expected) / std_dev`

- Z > 0: motif is more common than the null predicts (**over-represented** — possible selection)
- Z < 0: motif is less common (**under-represented** — possible loss/avoidance)

**p-value:** computed via the normal approximation to the distribution of |M(n)|.

Significance stars: `***` p < 0.001 · `**` p < 0.01 · `*` p < 0.05 · `ns` p ≥ 0.05

The **variance** needed for the Z-score is computed from:
- **Corollary 2** for Full Duplication
- **Corollary 16** (Binary Inheritance) for Partial Duplication
        """)

    with st.expander("Subnetwork Motif M"):
        st.markdown("""
A **subnetwork motif M** is a gene-family-specific substructure in a GRN,
characterised by:
- **k gene families** (the families whose members participate in the motif)
- An **inheritance probability vector $\\vec{\\pi}$ = (π₁, …, πₖ)** (one πᵢ per family)

An **instance** of M is a specific k-tuple of genes (a₁, …, aₖ) where aᵢ belongs
to family *i*, all co-participating in the regulatory pattern.

Key sets:
- **M(m)** — contains only the original instance (the k founding genes)
- **M(n)** — all instances present after n − m duplication events
- **|M(n)|** — the count of motif instances, which this model analyses

Subnetwork motifs differ from network motifs (Milo et al.) in that they carry
**family labels** and encode **evolutionary information** about inheritance.
        """)

    with st.expander("Transcription Factor (TF)"):
        st.markdown("""
A protein that binds specific DNA sequences (**binding sites / TFBS**) to control
the rate of transcription of nearby genes. TFs are the primary regulatory elements
in a GRN.

In this app, TFs are sourced from:
- **JASPAR 2024**: 177 *S. cerevisiae* TFs with experimentally validated PFMs
- **YEASTRACT**: 127 curated TFs with consensus binding sequences
- **SGD**: GO-annotated TFs with DNA-binding evidence (GO:0003677, GO:0000981)

A TF can be an **activator** (increases gene expression), a **repressor** (decreases
expression), or **dual** (both, depending on context).
        """)

    with st.expander("YEASTRACT"):
        st.markdown("""
**YEASTRACT** (Yeast Search for Transcriptional Regulators And Consensus Tracking;
yeastract.com) is a curated database of transcriptional regulatory associations and
TF binding site sequences for *S. cerevisiae*.

Data used in this app:
- **127 TFs** with curated regulatory evidence
- **IUPAC consensus sequences** for 115 TFs that overlap with JASPAR
- Consensus sequences are used as the basis for the **Method 4 (Consensus-adjusted)**
  π estimation: more specific (less ambiguous IUPAC) sequences → higher πᵢ prior
        """)

    st.divider()
    st.caption(
        "Glossary compiled from Scruse, Arnold & Robinson (2026) Bull. Math. Biol. 88:31, "
        "Harbison et al. (2004) Nature 431:99–104, and Ren et al. (2000) Science 290:2306–2309."
    )


# ══════════════════════════════════════════════════════════════════════
# TAB 7: Y1000+ π Estimators
# ══════════════════════════════════════════════════════════════════════

with tab7:
    st.header("🌍 Y1000+ Cross-Species π Estimators")
    st.markdown(
        "Three new inheritance-probability estimators derived from the "
        "[Y1000+ dataset](https://y1000plus.wei.wisc.edu/) (Opulente et al. 2024, *Science*) "
        "and JASPAR 2024 CORE yeast binding profiles."
    )

    with st.expander("📖 Methodology — why cross-species data gives a better estimate of π", expanded=False):
        st.markdown("""
**The core idea: use 1,154 yeast genomes as an evolutionary experiment**

The SGD-based methods (π₁–π₄ on the π Estimator tab) estimate the inheritance probability from a single snapshot
of *S. cerevisiae* — one species, one time point. The fundamental problem is that we cannot directly observe
whether a regulatory link was inherited or lost in the past; we can only infer it from current data.

The Y1000+ dataset (Opulente et al. 2024, *Science*) solves this by providing 1,154 yeast genome assemblies
spanning ~1 billion years of evolution. Each genome is an independent evolutionary replicate: if a regulatory
link in *S. cerevisiae* is truly inherited and functionally important, it should be present in orthologous
genes across many of these 1,154 genomes. If it is lineage-specific or recently acquired, it will be absent
in most.

**Why 1,154 genomes and not just a few?**

Statistical power. A single other species gives a binary signal (present/absent). With 1,154 genomes spanning
all major Saccharomycotina clades, we can estimate a continuous retention fraction with much lower variance.
The 48-species representative subset used here was chosen to maximise phylogenetic diversity (not redundancy
from closely related strains) while keeping compute tractable.

**The three estimators and their biological rationale:**

| Estimator | What it measures | Formula | Why it estimates π |
|-----------|-----------------|---------|-------------------|
| **π₂ — Sequence Homology** | Protein sequence identity between TF paralogs | π₂ = pct_identity / 100 | Higher identity → more recently diverged → less time for binding specificity to diverge → higher probability that both paralogs still bind the same sites |
| **π₃ — TFBS Conservation** | Fraction of 1,154 genomes with a significant PWM hit in the 1,000 bp upstream of the orthologous gene | π₃ = n_genomes_with_hit / n_genomes_scanned | The retention fraction is a direct empirical measurement of inheritance: if 80% of yeast species retain a detectable binding site at the orthologous locus, π₃ = 0.80 |
| **π₄ — SNP at Binding Sites** | IC-weighted polymorphism rate at binding-site positions | π₄ = 1 − Σ(IC_weight × poly_rate) | Sites that are under purifying selection (being actively maintained) have low polymorphism at their high-information-content positions; sites being lost have high polymorphism at the most critical nucleotides |

**Why π₃ is the most direct estimator:**

The scanning pipeline extracts exactly **1,000 bp upstream of the translational start** for each gene in each
of the 48 representative genomes. For each TF with a JASPAR PWM, it scores this upstream window and
reports whether the score exceeds a significance threshold (p < 0.001 under the position weight matrix).
The fraction of genomes passing this threshold is the retention fraction — directly analogous to asking
"in what fraction of evolutionary replicates did this regulatory link survive?"

**Why π₄ refines the π₃ signal:**

π₃ is binary (hit/no-hit). π₄ asks a subtler question: among the genomes that do have a hit, how degenerate
are the binding-site positions? IC weighting means that a mutation at the most conserved position
(maximum information content = 2 bits) contributes more to the polymorphism score than a mutation at a
highly ambiguous position. A site that is detectable (π₃ counted) but accumulating mutations at its
most critical positions (high IC-weighted poly rate → low π₄) is a site in the process of being lost.

**Putting it all together:**

High π₃ + high π₄ → site is present and intact across species → strong evidence of conserved inheritance.
High π₃ + low π₄ → site is present but eroding → transitional state; may be in the process of regulatory rewiring.
Low π₃ + any π₄ → site is absent in most species → likely lineage-specific in *S. cerevisiae* or recently gained.
""")
        st.divider()
        st.markdown("**🧮 Method Estimation Test — validate these estimators**")
        st.markdown(
            "The Method Estimation Test tab lets you measure how accurately each estimator "
            "(including Y1000+ π₃) recovers a known true-π profile, and compare all five "
            "methods side by side via per-family MSE. Click below to navigate there directly:"
        )
        components.html("""
        <a href="#"
           onclick="
             (function() {
               var tabs = window.parent.document.querySelectorAll('button[data-baseweb=tab]');
               for (var i = 0; i < tabs.length; i++) {
                 if (tabs[i].innerText.indexOf('Method Estimation Test') !== -1) {
                   tabs[i].click();
                   window.parent.scrollTo(0, 0);
                   break;
                 }
               }
             })();
             return false;
           "
           style="display:inline-block; padding:7px 18px; background:#2563eb; color:white;
                  border-radius:6px; text-decoration:none; font-size:0.88em; font-weight:500;
                  font-family:-apple-system,BlinkMacSystemFont,sans-serif; cursor:pointer;"
           onmouseover="this.style.background='#1d4ed8'"
           onmouseout="this.style.background='#2563eb'">
          🧮 Open Method Estimation Test &rarr;
        </a>
        """, height=48)

    # ── Phylogenetic context ──────────────────────────────────────────
    _LABELED_TREE = Path(__file__).parent / "assets" / "y1000plus_species_labeled.png"
    _PHYLO_TREE   = Path(__file__).parent / "assets" / "y1000plus_phylogeny.png"

    _either_exists = _LABELED_TREE.exists() or _PHYLO_TREE.exists()

    if _either_exists:
        st.subheader("Species panel & phylogenetic context")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if _LABELED_TREE.exists():
                st.image(
                    str(_LABELED_TREE),
                    caption=(
                        "Species included in Y1000+ (Opulente et al. 2024, Science). "
                        "Colored regions indicate major clades. "
                        "WGD = whole-genome duplication; CTG = CTG-clade yeasts "
                        "that decode CUG as serine rather than leucine."
                    ),
                    use_container_width=True,
                )
        with col_t2:
            if _PHYLO_TREE.exists():
                st.image(
                    str(_PHYLO_TREE),
                    caption=(
                        "Maximum-likelihood phylogeny with branch lengths "
                        "(substitutions/site) and bootstrap support values. "
                        "Colored bars on the right indicate taxonomic families. "
                        "Numbers at nodes = posterior probability / bootstrap support."
                    ),
                    use_container_width=True,
                )
    else:
        st.caption(
            "📷 To show the species phylogeny here, save the two tree images from the "
            "Y1000+ paper into the `assets/` folder:  \n"
            "`assets/y1000plus_species_labeled.png` and `assets/y1000plus_phylogeny.png`"
        )

    st.divider()

    # ── Generation status banner ──────────────────────────────────────
    from model.y1000plus_generator import get_status, all_done, csvs_ready, start_generation_if_needed

    _gen_status = get_status()
    _ready = csvs_ready()
    _status_code = _gen_status.get("status", "idle")

    if _status_code == "done" or all_done():
        st.success(
            "All three cross-species π datasets are ready.",
            icon="✅",
        )
    elif _status_code == "error":
        st.error(
            f"Generation failed: {_gen_status.get('error', 'unknown error')}  \n"
            "Fix the error and restart the app to retry.",
            icon="❌",
        )
    else:
        # Running or idle — show live status
        _pct = _gen_status.get("pct", 0)
        _msg = _gen_status.get("message", "Waiting to start…")

        _banner = st.container()
        with _banner:
            if _status_code == "idle":
                st.info(
                    "Cross-species data generation will start automatically. "
                    "If it has not begun, click **Generate now**.",
                    icon="ℹ️",
                )
                if st.button("Generate now", key="gen_now_btn"):
                    start_generation_if_needed()
                    st.rerun()
            else:
                st.info(
                    f"⏳ **Generating in background** — {_msg}",
                    icon="⏳",
                )

            # Per-CSV status pills
            pill_cols = st.columns(3)
            for col, (key, label) in zip(
                pill_cols,
                [("pi2", "π₂ Sequence"), ("pi3", "π₃ TFBS"), ("pi4", "π₄ SNP")]
            ):
                if _ready[key]:
                    col.success(f"{label} ✓")
                elif _status_code.endswith(key):
                    col.warning(f"{label} ⏳")
                else:
                    col.info(f"{label} …")

            st.progress(_pct / 100)

        # Auto-refresh every 5 s while generation is in progress
        if _status_code not in ("done", "error", "idle"):
            time.sleep(5)
            st.rerun()

    st.divider()

    # ── Method selector ──────────────────────────────────────────────
    method_choice = st.radio(
        "Select estimator to display",
        ["π₃ — TFBS Conservation", "π₂ — Sequence Homology", "π₄ — SNP at Binding Sites",
         "All methods comparison"],
        horizontal=True,
    )

    family_def = st.selectbox(
        "Family definition",
        ["TFBS-based (π₃/π₄)", "Sequence identity clusters (π₂)", "GO Biological Process (π₁)"],
        help="Determines which genes are grouped into one 'family' for each estimator.",
    )

    st.divider()

    # ── Load data helpers ─────────────────────────────────────────────
    @st.cache_data(show_spinner="Loading Y1000+ manifest…")
    def _load_y1000_manifest():
        try:
            from model.y1000plus_loader import load_manifest
            return load_manifest()
        except Exception as e:
            return None, str(e)

    @st.cache_data(show_spinner="Loading π₃ TFBS conservation data…")
    def _load_pi3():
        try:
            from model.pi3_tfbs_conservation import load_pi3_results, build_pairwise_histogram
            df = load_pi3_results()
            hist = build_pairwise_histogram(save=False)
            return df, hist, None
        except FileNotFoundError:
            return None, None, "pi3_tfbs_conservation.csv not found"
        except Exception as e:
            return None, None, str(e)

    @st.cache_data(show_spinner="Loading π₂ sequence homology data…")
    def _load_pi2():
        try:
            from model.pi2_sequence_homology import load_pi2_results
            return load_pi2_results(), None
        except FileNotFoundError:
            return None, "pi2_sequence_homology.csv not found"
        except Exception as e:
            return None, str(e)

    @st.cache_data(show_spinner="Loading π₄ SNP binding data…")
    def _load_pi4():
        try:
            from model.pi4_snp_binding import load_pi4_results
            return load_pi4_results(), None
        except FileNotFoundError:
            return None, "pi4_snp_binding_sites.csv not found"
        except Exception as e:
            return None, str(e)

    def _not_ready_box(csv_key: str, label: str) -> None:
        if _status_code in ("running_pi2", "running_pi3", "running_pi4"):
            st.info(f"{label} is being generated in the background — check back shortly.")
        else:
            st.warning(
                f"{label} not yet available. "
                "Generation will start automatically on the next app load, "
                "or click **Generate now** above.",
                icon="⚠️",
            )

    # ── π₃ TFBS Conservation ─────────────────────────────────────────
    if method_choice in ("π₃ — TFBS Conservation", "All methods comparison"):
        st.subheader("π₃ — TFBS Conservation across Y1000+ species")
        st.markdown(
            "For each TF→gene edge identified in *S. cerevisiae* S288C, π₃ = "
            "the fraction of Y1000+ genomes that retain a significant PWM hit "
            "in the upstream promoter of the orthologous gene."
        )

        pi3_df, pi3_hist, pi3_err = _load_pi3()

        if pi3_err:
            _not_ready_box("pi3", "π₃ TFBS conservation")
        else:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("TF→gene edges", f"{len(pi3_df):,}")
            col_b.metric("TFs covered", pi3_df["tf_name"].nunique())
            col_c.metric("Mean π₃", f"{pi3_df['pi3_estimate'].mean():.3f}")

            # TF selector
            tf_opts = sorted(pi3_df["tf_name"].unique().tolist())
            sel_tf = st.selectbox("Select TF (for family histogram)", tf_opts, key="pi3_tf")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Retention fraction distribution (all TFs)**")
                fig_hist = px.histogram(
                    pi3_df, x="retention_fraction", nbins=40,
                    labels={"retention_fraction": "Retention fraction (π₃)"},
                    color_discrete_sequence=["#2563eb"],
                )
                fig_hist.update_layout(height=320, margin=dict(t=20, b=30))
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                st.markdown(f"**Pairwise shared binding-site distribution — {sel_tf}**")
                if pi3_hist is not None and not pi3_hist.empty:
                    tf_hist_row = pi3_hist[pi3_hist["tf_name"] == sel_tf]
                    if not tf_hist_row.empty:
                        r = tf_hist_row.iloc[0]
                        st.markdown(
                            f"Family size: **{int(r['n_target_genes'])}** genes · "
                            f"Mean retention: **{r['mean_retention']:.3f}** · "
                            f"σ = **{r['std_retention']:.3f}**"
                        )
                        st.markdown(
                            f"Pairwise sharing mean: **{r['pairwise_sharing_mean']:.3f}** "
                            f"(π̂₃ estimate for family)"
                        )

                tf_edges = pi3_df[pi3_df["tf_name"] == sel_tf]
                _n_genes = len(tf_edges)
                _bar_height = max(320, min(520, 260 + _n_genes * 6))
                fig_bar = px.bar(
                    tf_edges.sort_values("retention_fraction"),
                    x="target_gene_name", y="retention_fraction",
                    labels={"target_gene_name": "Target gene", "retention_fraction": "π₃"},
                    color="retention_fraction",
                    color_continuous_scale="Blues",
                )
                fig_bar.update_layout(
                    height=_bar_height,
                    margin=dict(t=20, b=90),
                    xaxis_tickangle=-50,
                    xaxis=dict(
                        tickfont=dict(size=max(7, min(10, 120 // max(_n_genes, 1)))),
                        rangeslider=dict(visible=_n_genes > 20),
                    ),
                    yaxis=dict(range=[0, 1.05], title="Retention fraction (π₃)"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Gene-centric upstream retention figure ────────────────
            st.markdown("---")
            st.markdown(
                "**Retention Fraction in Upstream Regions of Genes Involved**  \n"
                "_For each target gene, what fraction of Y1000+ species retain a significant "
                "PWM hit in the 1,000 bp upstream of its translational start — averaged across "
                "all TFs known to target that gene._"
            )
            _gene_ret = (
                pi3_df.groupby("target_gene_name")
                .agg(
                    mean_retention=("retention_fraction", "mean"),
                    n_tfs_targeting=("tf_name", "nunique"),
                    std_retention=("retention_fraction", "std"),
                )
                .reset_index()
                .sort_values("mean_retention", ascending=False)
            )
            _top_n_genes = st.slider(
                "Number of genes to display (sorted by mean retention)",
                min_value=10, max_value=min(150, len(_gene_ret)), value=min(50, len(_gene_ret)),
                step=10, key="gene_ret_topn",
            )
            _gene_ret_top = _gene_ret.head(_top_n_genes)
            fig_gene_ret = px.bar(
                _gene_ret_top,
                x="target_gene_name",
                y="mean_retention",
                error_y="std_retention",
                color="mean_retention",
                color_continuous_scale="Blues",
                hover_data=["n_tfs_targeting", "std_retention"],
                labels={
                    "target_gene_name": "Gene",
                    "mean_retention": "Mean retention fraction (π₃)",
                    "n_tfs_targeting": "# TFs targeting gene",
                    "std_retention": "Std Dev",
                },
            )
            _overall_mean = pi3_df["retention_fraction"].mean()
            fig_gene_ret.add_hline(
                y=_overall_mean,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Overall mean π₃ = {_overall_mean:.3f}",
                annotation_position="top right",
            )
            fig_gene_ret.update_layout(
                height=420,
                margin=dict(t=30, b=100),
                xaxis_tickangle=-50,
                xaxis=dict(tickfont=dict(size=max(7, min(10, 800 // max(_top_n_genes, 1))))),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_gene_ret, use_container_width=True)
            st.caption(
                "Each bar = one target gene in the S. cerevisiae regulatory network. Height = mean "
                "retention fraction across all TFs whose JASPAR PWMs were scanned in the 1,000 bp "
                "upstream of that gene's translational start, averaged over the Y1000+ species panel. "
                "Error bars = ±1 SD across TFs targeting that gene. Red dashed line = overall mean π₃ "
                "across all TF→gene edges. Genes with no error bar are targeted by only one TF in the dataset."
            )

            # Interpretation
            with st.expander("🔍 What do these π₃ results mean?"):
                mean_pi3 = pi3_df["pi3_estimate"].mean()
                if mean_pi3 >= 0.7:
                    pi3_level = "high"
                    pi3_bio = (
                        "TF binding sites in S. cerevisiae are largely conserved across the yeast tree. "
                        "This is consistent with strong purifying selection on these regulatory elements — "
                        "orthologous genes in distantly related yeasts are still regulated by the same TFs."
                    )
                elif mean_pi3 >= 0.4:
                    pi3_level = "moderate"
                    pi3_bio = (
                        "Around half of S. cerevisiae TF→gene edges have conserved binding sites in other yeasts. "
                        "This mixture reflects both conserved core regulatory circuits and lineage-specific rewiring."
                    )
                else:
                    pi3_level = "low"
                    pi3_bio = (
                        "Many S. cerevisiae binding sites are not detected in other yeast species. "
                        "This could reflect rapid TF binding-site turnover, low PWM specificity causing false-negative scans, "
                        "or genuine lineage-specific regulatory innovation in S. cerevisiae."
                    )
                st.markdown(f"""
**Mean π₃ = {mean_pi3:.3f}** across {len(pi3_df):,} TF→gene edges ({pi3_df['tf_name'].nunique()} TFs)

This is a **{pi3_level}** cross-species conservation signal: {pi3_bio}

**How π₃ differs from π₁–π₄ (SGD-based):** π₃ is a direct empirical measurement — for each TF→gene edge in S. cerevisiae, we counted how many of the 1,154 Y1000+ genomes have a PWM hit in the orthologous upstream region. A retention fraction of 0.8 means 80% of yeasts still have a detectable binding site at that location.

**Why might retention fractions be low for some TFs?** (1) The JASPAR PWM may not generalise to diverged species; (2) ortholog assignment across ~1 billion years of yeast evolution is imperfect; (3) some binding sites are genuinely lineage-specific innovations. Low π₃ for a TF does **not** necessarily mean the TF is non-functional — it may have acquired different targets in other species.
""")

            with st.expander("📖 How π₃ is computed — step by step"):
                st.markdown("""
**The π₃ pipeline: counting how many yeast genomes retain a TF binding site**

π₃ is the most direct of all seven estimators — it directly counts evolutionary replicates
where a regulatory link is retained, rather than inferring retention from proxy signals.

**Step 1 — Identify all TF→gene edges in *S. cerevisiae* S288C**

Starting from the SGD regulatory network, collect all experimentally supported TF→gene
regulatory relationships. Each edge is a (TF, target gene) pair with a known direction.

**Step 2 — Retrieve a JASPAR PWM for each TF**

For each TF, retrieve its position weight matrix (PWM) from JASPAR 2024 CORE (yeast). The
PWM encodes the probability of each base (A/T/G/C) at each position of the binding motif.
TFs without a JASPAR PWM are excluded from π₃ analysis.

**Step 3 — Extract upstream sequences across 48 representative Y1000+ genomes**

From each of 48 phylogenetically diverse Y1000+ genomes (selected to maximise clade
coverage across Saccharomycotina, ~1 billion years of evolution):
1. Identify the ortholog of each S. cerevisiae target gene
2. Extract the 1,000 bp immediately upstream of its translational start codon
3. This upstream window is the putative promoter region where TF binding sites reside

**Step 4 — Score each upstream region with the TF's PWM**

The JASPAR PWM is scanned across the 1,000 bp window. A hit is reported if any position
scores at **p < 0.001** relative to the background nucleotide distribution. This threshold
was chosen to balance sensitivity (not missing real sites) against specificity (not counting
random matches).

**Step 5 — Compute the retention fraction**

```
π₃(TF→gene) = n_genomes_with_hit / n_genomes_scanned
```

If 38 of 48 genomes have a significant PWM hit in the orthologous upstream region,
π₃ = 38/48 ≈ 0.79.

**Step 6 — Aggregate to the gene family level**

For a TF family (all paralogs of a given TF), the family-level π̂₃ is the mean pairwise
sharing across all pairs of target genes in the family — the "pairwise sharing mean" shown
in the histogram above.

---

**Why 48 genomes rather than all 1,154?**

The 48-genome panel was selected to maximise phylogenetic diversity. Using all 1,154 genomes
would dramatically oversample closely related strains (especially *S. cerevisiae* and its
nearest relatives), biasing the retention fraction upward for lineage-specific sites.
The 48 representative genomes span all major Saccharomycotina clades at roughly equal branch
lengths, giving each evolutionary lineage equal weight in the retention fraction.

**What a low π₃ means**

Low π₃ for a TF→gene edge does not necessarily mean the regulatory link is non-functional.
Possible explanations: (1) the JASPAR PWM doesn't generalise to diverged species; (2) ortholog
assignment across ~1 billion years of evolution is imperfect; (3) the link is genuinely
*S. cerevisiae*-specific (a recent regulatory innovation). Use π₃ alongside π₁ and π₄ to
get a fuller picture.
""")

            # Full table
            with st.expander("Full π₃ table"):
                st.dataframe(
                    pi3_df[["tf_name", "jaspar_matrix_id", "target_gene_id",
                             "target_gene_name", "n_genomes_scanned",
                             "n_genomes_with_hit", "retention_fraction", "pi3_estimate"]],
                    use_container_width=True,
                )

    st.divider()

    # ── π₂ Sequence Homology ─────────────────────────────────────────
    if method_choice in ("π₂ — Sequence Homology", "All methods comparison"):
        st.subheader("π₂ — Sequence Homology")
        st.markdown(
            "Pairwise protein sequence identity between TF paralogs. "
            "π₂ = identity / 100 (higher identity → more recently duplicated → "
            "higher probability of retained regulatory links)."
        )

        pi2_df, pi2_err = _load_pi2()

        if pi2_err:
            _not_ready_box("pi2", "π₂ sequence homology")
        else:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("TF pairs aligned", f"{len(pi2_df):,}")
            col_b.metric("Mean identity", f"{pi2_df['pct_identity'].mean():.1f}%")
            col_c.metric("Mean π₂", f"{pi2_df['pi2_estimate'].mean():.3f}")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Pairwise identity distribution**")
                fig = px.histogram(
                    pi2_df, x="pct_identity", nbins=40,
                    labels={"pct_identity": "Pairwise identity (%)"},
                    color_discrete_sequence=["#16a34a"],
                )
                fig.update_layout(height=320, margin=dict(t=20, b=30))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**π₂ vs pct_identity scatter**")
                fig2 = px.scatter(
                    pi2_df, x="pct_identity", y="pi2_estimate",
                    hover_data=["gene1_name", "gene2_name"],
                    labels={"pct_identity": "Identity (%)", "pi2_estimate": "π₂"},
                    color_discrete_sequence=["#16a34a"],
                    opacity=0.6,
                )
                fig2.update_layout(height=320, margin=dict(t=20, b=30))
                st.plotly_chart(fig2, use_container_width=True)

            thr = st.select_slider(
                "Identity threshold for family clustering",
                options=[30, 50, 80], value=50,
            )
            cluster_col = f"family_{thr}pct"
            if cluster_col in pi2_df.columns:
                n_clusters = pi2_df[cluster_col].nunique()
                st.info(f"At {thr}% identity: **{n_clusters}** families among TF proteins.")

            # Interpretation
            with st.expander("🔍 What do these π₂ results mean?"):
                mean_id = pi2_df["pct_identity"].mean()
                mean_pi2 = pi2_df["pi2_estimate"].mean()
                st.markdown(f"""
**Mean pairwise identity = {mean_id:.1f}%** · **Mean π₂ = {mean_pi2:.3f}** across {len(pi2_df):,} TF pairs

**What π₂ measures:** Protein sequence identity between TF paralogs is used as a proxy for how recently they diverged. Higher identity → more recent duplication → less time for regulatory divergence → higher probability that both copies retain the same binding sites.

**How to read the scatter plot:** Points above the diagonal would indicate that π₂ overestimates inheritance (identity is high but the binding sites may have already diverged). Since π₂ = identity/100, the scatter is linear by definition — the plot mainly reveals the range of identity values and any outliers.

**Why does identity vary so much?**
- Pairs near 100%: very recent duplications (e.g., tandem repeats, recent whole-genome duplication remnants). These TFs likely have nearly identical binding specificities.
- Pairs near 30–50%: ancient duplications from the S. cerevisiae whole-genome duplication (~100 Mya). These TF paralogs have had time to diverge substantially in binding domain sequence, and likely have partially different binding specificities and regulatory targets.
- π₂ is the most conservative (sequence-based) estimate. It does not directly measure whether binding sites are retained — use π₃ for a direct TFBS measurement.
""")

            with st.expander("📖 How π₂ is computed — step by step"):
                st.markdown("""
**The π₂ pipeline: paralog sequence identity as a proxy for regulatory divergence**

π₂ is the simplest of the Y1000+ estimators. It does not scan upstream sequences at all —
it uses protein sequence divergence as a proxy for how likely binding specificity has been
preserved after gene duplication.

**Biological rationale**

A regulatory link is inherited when, after gene duplication, one daughter gene still encodes
a TF that recognises the same binding sites as the ancestral gene. TF binding specificity is
determined largely by the DNA-binding domain. If two TF paralogs are nearly identical in
protein sequence, their binding domains are also nearly identical, and they very likely bind
the same sites — so the regulatory link is inherited. If the paralogs have diverged substantially
(e.g., 40% identity), their binding domains may have different specificities, and the regulatory
link may have been lost or rewired.

**Step 1 — Identify TF paralog pairs**

In *S. cerevisiae*, TF paralogs are identified by sequence clustering. Three identity thresholds
are available (30%, 50%, 80%), defining progressively stricter family definitions:
- 80%: only very recent duplicates in the same tight family
- 50%: the standard threshold (captures most post-WGD pairs)
- 30%: includes more ancient duplications

**Step 2 — Align paralog pairs**

Each pair of TF sequences is aligned using BLASTP (or equivalent). The alignment identity
`pct_identity` is the fraction of aligned positions with identical residues.

**Step 3 — Convert to π₂**

```
π₂ = pct_identity / 100
```

This is a linear mapping. A pair with 75% identity gets π₂ = 0.75.

**Step 4 — Aggregate to the family level**

For a TF family, the family-level π̂₂ is the mean pairwise identity across all TF pairs within
that family at the chosen threshold.

---

**Key limitation**

π₂ assumes a linear relationship between sequence identity and regulatory inheritance, but this
is a simplification. Binding specificity can be maintained even at 40% identity (if key contact
residues are conserved) or lost even at 90% identity (if one key residue changes). π₂ is best
treated as a rough prior; use π₃ for a direct measurement of TFBS conservation.
""")

            with st.expander("Full π₂ table"):
                st.dataframe(pi2_df, use_container_width=True)

    st.divider()

    # ── π₄ SNP at Binding Sites ──────────────────────────────────────
    if method_choice in ("π₄ — SNP at Binding Sites", "All methods comparison"):
        st.subheader("π₄ — SNP rate at binding site positions")
        st.markdown(
            "IC-weighted polymorphism rate at PWM binding site positions across Y1000+ "
            "species. π₄ = 1 − Σ(IC_weight[pos] × polymorphism_rate[pos]). "
            "Perfectly conserved sites → π₄ = 1; hypervariable sites → π₄ → 0."
        )

        pi4_df, pi4_err = _load_pi4()

        if pi4_err:
            _not_ready_box("pi4", "π₄ SNP at binding sites")
        else:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("TF→gene edges", f"{len(pi4_df):,}")
            col_b.metric("Mean π₄", f"{pi4_df['pi4_estimate'].mean():.3f}")
            col_c.metric("Mean poly rate", f"{pi4_df['weighted_polymorphism_rate'].mean():.3f}")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**π₄ distribution**")
                fig = px.histogram(
                    pi4_df, x="pi4_estimate", nbins=40,
                    labels={"pi4_estimate": "π₄ (SNP at binding sites)"},
                    color_discrete_sequence=["#d97706"],
                )
                fig.update_layout(height=320, margin=dict(t=20, b=30))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Weighted polymorphism rate vs π₄**")
                fig2 = px.scatter(
                    pi4_df, x="weighted_polymorphism_rate", y="pi4_estimate",
                    color="tf_name",
                    hover_data=["target_gene_name", "binding_site_seq"],
                    labels={"weighted_polymorphism_rate": "Weighted poly rate",
                            "pi4_estimate": "π₄"},
                    opacity=0.6,
                )
                fig2.update_layout(height=320, margin=dict(t=20, b=30),
                                   showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            # Interpretation
            with st.expander("🔍 What do these π₄ results mean?"):
                mean_pi4 = pi4_df["pi4_estimate"].mean()
                mean_poly = pi4_df["weighted_polymorphism_rate"].mean()
                st.markdown(f"""
**Mean π₄ = {mean_pi4:.3f}** · **Mean weighted polymorphism rate = {mean_poly:.3f}** across {len(pi4_df):,} TF→gene edges

**What π₄ measures:** IC-weighted polymorphism rate at binding-site positions across Y1000+ species. Each position in the PWM is weighted by its information content (IC, bits) — highly conserved positions count more. A high polymorphism rate at high-IC positions strongly indicates the binding site is diverging or has been lost.

- **π₄ close to 1:** The binding site is nearly invariant across the 1,154 yeast genomes. Every position, especially the most specific ones, is polymorphism-free. This binding site is under strong purifying selection and is very likely functional in all sampled yeasts.
- **π₄ close to 0:** The binding site is highly polymorphic at its most specific positions. The site may be non-functional in many species, recently acquired (not yet conserved), or under positive selection for divergence.

**Why does this differ from π₃?** π₃ counts binary presence/absence of a PWM hit. π₄ measures the *quality* of retained sites — a site may be detectable by PWM scan (π₃ counted) but have elevated polymorphism at key positions (π₄ lower), indicating the site is eroding. Together, high π₃ and high π₄ give the strongest evidence of conserved, functional inheritance.

**Why are some polymorphism rates high?** (1) Genuine positive selection driving binding site divergence; (2) the Y1000+ species are very broadly sampled (spanning ~1 billion years) — some variation is expected even at constrained sites; (3) ortholog-region misalignment can introduce apparent polymorphism.
""")

            with st.expander("📖 How π₄ is computed — step by step"):
                st.markdown("""
**The π₄ pipeline: IC-weighted polymorphism at binding-site positions**

π₄ refines the signal from π₃. While π₃ asks "is a binding site present in other species?",
π₄ asks "among sites that are present, how intact are the most critical positions?"

**Why information content (IC) weighting?**

A TF binding site has positions with very different diagnostic value. Some positions are nearly
invariant across all binding sites — mutations there always abolish binding (high IC, up to 2 bits).
Other positions tolerate any nucleotide — mutations there are essentially silent (IC ≈ 0). A naive
polymorphism rate would count all positions equally; IC weighting ensures that mutations at
biologically critical positions count more.

**Step 1 — Identify TF binding-site sequences**

For each TF→gene edge with a π₃ PWM hit, extract the actual binding-site sequence (the specific
window in the upstream region that scored highest under the JASPAR PWM scan).

**Step 2 — Compute per-position information content**

For each position j in the PWM, IC(j) = 2 − H(j), where H(j) is the Shannon entropy of the
base distribution at that position in the JASPAR matrix. IC(j) ranges from 0 (fully degenerate)
to 2 bits (completely invariant — one base only).

**Step 3 — Compute per-position polymorphism rate across Y1000+ species**

For each position j in the binding site of a TF→gene edge, scan the same position across the
orthologous upstream sequences in all Y1000+ genomes. The polymorphism rate at position j is:

```
poly_rate(j) = fraction of Y1000+ genomes where base at position j ≠ S. cerevisiae S288C base
```

**Step 4 — IC-weighted average polymorphism**

```
weighted_poly_rate = Σ_j [IC(j) × poly_rate(j)] / Σ_j IC(j)
```

This is the average polymorphism rate, weighted so that high-IC positions dominate.

**Step 5 — Convert to π₄**

```
π₄ = 1 − weighted_poly_rate
```

A binding site where the most critical positions are never polymorphic across 1,154 genomes
gets π₄ ≈ 1. A site where critical positions are frequently polymorphic gets π₄ → 0.

---

**π₃ vs π₄: what each adds**

| | π₃ | π₄ |
|---|---|---|
| **What it counts** | Binary: hit/no-hit per genome | Continuous: mutation rate at key positions |
| **What high values mean** | Site is present in most genomes | Site is intact at its most critical positions |
| **What low values mean** | Site absent in most genomes | Site is eroding at its most critical positions |
| **Best combined interpretation** | High π₃ + high π₄ = site present and intact | Low π₃ but high π₄ = site absent but structurally stable where it does appear |

Sites in a transitional state (being lost) often show moderate π₃ with declining π₄ as key
positions accumulate mutations before the site drops below the PWM detection threshold.
""")

            with st.expander("Full π₄ table"):
                st.dataframe(pi4_df, use_container_width=True)

    st.divider()

    # ── Cross-method comparison ───────────────────────────────────────
    if method_choice == "All methods comparison":
        st.subheader("Cross-estimator comparison: π₃ vs π₁ (evidence-based)")
        pi3_df2, _, pi3_err2 = _load_pi3()
        if pi3_err2 is None and pi3_df2 is not None:
            # Load evidence-based π for each TF
            from model.inheritance_estimator import estimate_pi_from_evidence
            tf_list_comp = sorted(pi3_df2["tf_name"].unique().tolist())

            ev_rows = []
            for tf in tf_list_comp:
                res = estimate_pi_from_evidence([tf])
                ev_rows.append({
                    "tf_name": tf,
                    "pi1_evidence": res["pi_vec"][0] if res["pi_vec"] else None,
                })
            ev_df = pd.DataFrame(ev_rows)

            pi3_by_tf = (
                pi3_df2.groupby("tf_name")["pi3_estimate"].mean().reset_index()
                .rename(columns={"pi3_estimate": "pi3_tfbs"})
            )
            comp_df = pi3_by_tf.merge(ev_df, on="tf_name")

            fig_cmp = px.scatter(
                comp_df, x="pi1_evidence", y="pi3_tfbs", text="tf_name",
                labels={"pi1_evidence": "π₁ (evidence-based)", "pi3_tfbs": "π₃ (TFBS conservation)"},
                color_discrete_sequence=["#7c3aed"],
            )
            fig_cmp.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                              line=dict(dash="dash", color="gray"))
            fig_cmp.update_traces(textposition="top center", textfont_size=9)
            fig_cmp.update_layout(height=450)
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.caption(
                "Points above the diagonal: π₃ > π₁ (conservation signal exceeds "
                "what evidence codes predict). Points below: experimental evidence "
                "overestimates conservation."
            )

    # ── Dataset status panel ─────────────────────────────────────────
    with st.expander("Y1000+ dataset status"):
        try:
            from model.y1000plus_loader import load_manifest, manifest_summary
            ms = manifest_summary()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total assemblies", ms["n_unique_assemblies"])
            c2.metric("Annotated (final)", ms["n_final"])
            c3.metric("SGD reference", ms["n_sgd"])
            c4.metric("With genome", ms["n_with_genome"])

            from model.y1000plus_loader import PROCESSED_DIR
            gff3_extracted = len(list((PROCESSED_DIR / "y1000p_gff3_files").glob("*.gff3")))
            st.info(
                f"GFF3 files extracted: **{gff3_extracted}** / {ms['n_unique_assemblies']}  \n"
                f"Processed directory: `{PROCESSED_DIR}`"
            )
        except Exception as e:
            st.error(f"Could not load Y1000+ manifest: {e}")


# ══════════════════════════════════════════════════════════════════════
# TAB 9: Method Estimation Test
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _val_load_pi3_for_test():
    """Load pi3 TFBS conservation data for use in the Method Estimation Test tab."""
    try:
        from model.pi3_tfbs_conservation import load_pi3_results
        return load_pi3_results(), None
    except FileNotFoundError:
        return None, "pi3_tfbs_conservation.csv not found — generate Y1000+ data first (Y1000+ π Estimators tab)"
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def _val_load_pi2_for_test():
    """Load pi2 sequence homology data for use in the Method Estimation Test tab."""
    try:
        from model.pi2_sequence_homology import load_pi2_results
        return load_pi2_results(), None
    except FileNotFoundError:
        return None, "pi2_sequence_homology.csv not found — generate Y1000+ data first (Y1000+ π Estimators tab)"
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def _val_load_pi4_for_test():
    """Load pi4 binding-site SNP data for use in the Method Estimation Test tab."""
    try:
        from model.pi4_snp_binding import load_pi4_results
        return load_pi4_results(), None
    except FileNotFoundError:
        return None, "pi4_snp_binding_sites.csv not found — generate Y1000+ data first (Y1000+ π Estimators tab)"
    except Exception as e:
        return None, str(e)


_VAL_PROFILES = {
    3: {
        "Linear":    [0.1, 0.5, 0.9],
        "Quadratic": [0.3, 0.5, 0.3],
    },
    4: {
        "Linear":    [0.2, 0.4, 0.6, 0.8],
        "Quadratic": [0.1, 0.5, 0.5, 0.1],
    },
}
_VAL_N_VALUES = [20, 50, 100]
_VAL_M = 7
_PROFILE_COLORS = {"Linear": "#2563eb", "Quadratic": "#7c3aed"}


def _val_per_family_chart(x_labels, true_pi_vec, est_pi_vec, est_label, title):
    fig = go.Figure()
    fig.add_bar(x=x_labels, y=true_pi_vec, name="True πᵢ", marker_color="#2563eb")
    fig.add_bar(x=x_labels, y=est_pi_vec, name=est_label, marker_color="#f97316")
    fig.update_layout(
        barmode="group",
        height=260,
        title=title,
        yaxis=dict(title="π", range=[0, 1.05]),
        legend=dict(orientation="h", y=-0.3),
        margin=dict(t=40, b=10),
    )
    return fig


with tab9:
    st.header("Method Estimation Test — Known π Recovery")

    # ── Method selector ───────────────────────────────────────────────
    val_method = st.radio(
        "Estimation method to test",
        [
            "Method 1 — Evidence-based",
            "Method 2 — Moment Estimation (Theorem 4)",
            "Method 3 — SNP Divergence",
            "Method 4 — Consensus-adjusted",
            "Method 5 — Y1000+ (π₃ TFBS Conservation)",
            "Method 6 — Y1000+ (π₂ Sequence Homology)",
            "Method 7 — Y1000+ (π₄ Binding-site SNPs)",
            "Multi-signal Ensemble — per-family πᵢ",
        ],
        horizontal=True,
    )
    use_m1  = val_method.startswith("Method 1")
    use_m2  = val_method.startswith("Method 2")
    use_m3  = val_method.startswith("Method 3")
    use_m4  = val_method.startswith("Method 4")
    use_m5  = val_method.startswith("Method 5")
    use_m6  = val_method.startswith("Method 6")
    use_m7  = val_method.startswith("Method 7")
    use_ens = val_method.startswith("Multi-signal")

    _METHOD_WORKFLOWS = {
        "Method 1 — Evidence-based": (
            "**Workflow:** select k TFs from the SGD dataset. Method 1 maps each TF's "
            "experimental evidence codes (IDA, IMP, IEA, …) to a π prior. The returned "
            "per-family πᵢ values are compared directly to the true profile."
        ),
        "Method 2 — Moment Estimation (Theorem 4)": (
            "**Workflow:** given a known true π vector, compute the expected motif count "
            "via Theorem 4 (forward), feed that count into Method 2's Brent root-find to "
            "recover π̂ (backward), then distribute π̂ uniformly across k families (π̂/k)."
        ),
        "Method 3 — SNP Divergence": (
            "**Workflow:** Method 3 is calibrated on 11 strains of gene YFL039C. Each "
            "strain's π = 1 − pct_alt/100; the cross-strain mean μ is applied to all k "
            "families. No n or m parameters are needed."
        ),
        "Method 4 — Consensus-adjusted": (
            "**Workflow:** select k TFs. Method 4 takes each TF's evidence-based π (Method 1) "
            "and multiplies by a binding-flexibility factor derived from YEASTRACT IUPAC "
            "consensus sequences. TFs with more ambiguous binding sequences get a higher "
            "factor (≥ 1.0), pushing π upward. The adjusted values are compared to the profile."
        ),
        "Method 5 — Y1000+ (π₃ TFBS Conservation)": (
            "**Workflow:** select k TFs from those available in the Y1000+ π₃ dataset. "
            "For each TF, Method 5 computes the mean retention fraction across all its "
            "target genes in the 48-species Y1000+ panel — the fraction of genomes retaining "
            "a significant PWM hit in the 1,000 bp upstream of the orthologous gene. "
            "This empirical conservation fraction is used as the per-family π estimate. "
            "Requires Y1000+ data to be generated first."
        ),
        "Method 6 — Y1000+ (π₂ Sequence Homology)": (
            "**Workflow:** select k TFs from the full TF list. For each TF, Method 6 "
            "computes its mean pairwise protein sequence identity with all other TFs in "
            "the Y1000+ dataset (π₂ = pct_identity / 100). A TF with high average identity "
            "to its paralogs is more likely to share regulatory targets after duplication. "
            "Requires Y1000+ data to be generated first."
        ),
        "Method 7 — Y1000+ (π₄ Binding-site SNPs)": (
            "**Workflow:** select k TFs from those available in the Y1000+ π₄ dataset. "
            "For each TF, Method 7 computes the mean IC-weighted polymorphism rate at the "
            "exact binding-site positions across Y1000+ genomes: π₄ = 1 − Σ IC_weight[pos] × "
            "polymorphism_rate[pos]. Positions with high information content (critical positions) "
            "contribute more to the score. Requires Y1000+ data to be generated first."
        ),
        "Multi-signal Ensemble — per-family πᵢ": (
            "**Workflow:** select k TFs. The ensemble runs Methods 1, 3, 4, 5, 6, and 7 "
            "independently for each family and averages the available signals into a single "
            "πᵢ per family. **Method 2 is deliberately excluded** — Theorem 4's expected-count "
            "formula depends only on π̂ = Σπᵢ, so inverting it recovers the total inheritance "
            "probability but cannot distinguish how it is distributed across individual families. "
            "Y1000+ signals (M5, M6, M7) are included when the data has been generated; SGD "
            "signals (M1, M3, M4) are always available."
        ),
    }
    st.markdown(_METHOD_WORKFLOWS[val_method])

    with st.expander("Parameter glossary"):
        st.markdown(f"""
        | Parameter | Applies to | Meaning |
        |-----------|-----------|---------|
        | **m = {_VAL_M}** | Method 2 only | Number of founding gene families before any duplication |
        | **n** | Method 2 only | Total gene count after duplication; controls the Theorem 4 expected count |
        | **k** | All | Number of gene families in the subnetwork motif; length of the π vector |
        | **π̂ = Σπᵢ** | Method 2 | Total inheritance probability; the only quantity Theorem 4 depends on |
        | **Evidence score** | Method 1 | Quality weight assigned to each SGD evidence code (IDA → 0.90, IEA → 0.30, …) |
        | **pct_alt / μ** | Method 3 | Per-strain % alternative alleles at YFL039C; μ is the cross-strain mean π |
        | **Consensus factor** | Method 4 | YEASTRACT binding-flexibility multiplier (1.02 – 1.12); higher = more ambiguous consensus = tolerates more drift |
        | **Retention fraction / π₃** | Method 5 | Fraction of Y1000+ genomes with a significant PWM hit upstream of the orthologous gene; averaged across all target genes per TF |
        | **MSE** | All | Mean squared error between true πᵢ and estimated πᵢ across k families |
        | **Signal** | Ensemble | One independent per-family πᵢ estimate contributed by a single method; the ensemble mean is the unweighted average of all available signals |
        | **n_signals** | Ensemble | How many of the 6 eligible methods contributed a non-missing estimate for this family; higher is more reliable |
        """)

    st.divider()

    # ── Controls ─────────────────────────────────────────────────────
    col_k, col_n = st.columns(2)
    with col_k:
        val_k = st.radio(
            "Motif size **k**",
            [3, 4],
            horizontal=True,
            help="k=3: 3-family regulatory pattern. k=4: 4-family pattern.",
        )
    with col_n:
        val_n_detail = st.radio(
            "Detail view for **n**",
            _VAL_N_VALUES,
            index=1,
            horizontal=True,
            help="Only used by Method 2. Selects which n to show in the per-family breakdown.",
            disabled=not use_m2,
        )

    profiles = _VAL_PROFILES[val_k]

    # ════════════════════════════════════════════════════════════════
    # METHOD 1 branch — SGD evidence-based π
    # ════════════════════════════════════════════════════════════════
    if use_m1:
        tfs_m1   = _load_tfs().sort_values("gene_name")
        tf_names = tfs_m1["gene_name"].tolist()

        # Defaults: k genes evenly spread across the evidence score range
        tfs_sorted_ev = tfs_m1.sort_values("pi_prior").reset_index(drop=True)
        indices       = [int(i * (len(tfs_sorted_ev) - 1) / (val_k - 1)) for i in range(val_k)]
        default_genes = [tfs_sorted_ev.loc[i, "gene_name"] for i in indices]

        selected_genes = st.multiselect(
            f"Select exactly **{val_k}** TF gene names",
            tf_names,
            default=default_genes,
            help=(
                "Method 1 uses each TF's SGD evidence codes to assign a π prior. "
                "Defaults are spread from the lowest to highest evidence score in the dataset."
            ),
        )

        if len(selected_genes) != val_k:
            st.warning(f"Please select exactly {val_k} genes (currently {len(selected_genes)}).")
            st.stop()

        m1_result = estimate_pi_from_evidence(selected_genes)
        est_pi_m1 = m1_result["pi_vec"]
        ev_scores = m1_result["evidence_scores"]

        with st.expander("Evidence scores for selected genes"):
            ev_range_lo = tfs_m1["pi_prior"].min()
            ev_range_hi = tfs_m1["pi_prior"].max()
            st.caption(
                f"Evidence scores in the SGD dataset range from **{ev_range_lo:.4f}** "
                f"(weak automated annotation) to **{ev_range_hi:.4f}** (strong direct assay). "
                "Values outside this range — especially below 0.31 or above 0.77 — cannot "
                "be reached by Method 1 regardless of which genes are chosen."
            )
            st.dataframe(
                pd.DataFrame({
                    "Gene":           selected_genes,
                    "Evidence score": [round(s, 4) for s in ev_scores],
                }).assign(
                    **{"Range min": ev_range_lo, "Range max": ev_range_hi}
                ),
                use_container_width=True, hide_index=True,
            )

        st.divider()

        m1_rows = []
        for profile_name, true_pi_vec in profiles.items():
            family_se = [(e - t) ** 2 for e, t in zip(est_pi_m1, true_pi_vec)]
            mse       = float(np.mean(family_se))
            m1_rows.append({
                "Profile":        profile_name,
                "True π̂":        round(sum(true_pi_vec), 4),
                "Estimated π̂":   round(sum(est_pi_m1), 4),
                "Per-family MSE": round(mse, 6),
            })

        st.subheader("Summary — all profiles")
        st.caption(
            f"Method 1 returns fixed evidence-score priors for the selected genes "
            f"({', '.join(selected_genes)}). These values are compared against each profile. "
            f"MSE is computed analytically over **2 test profiles** (Linear, Quadratic), each with "
            f"**k = {val_k}** synthetic true-πᵢ values. No partial duplication simulation is needed "
            f"— Method 1's estimate comes from SGD evidence codes, not from observed motif counts."
        )
        st.dataframe(pd.DataFrame(m1_rows), use_container_width=True, hide_index=True)

        st.divider()

        st.subheader(f"Per-family MSE by profile (k = {val_k})")
        fig_mse1 = px.bar(
            pd.DataFrame(m1_rows),
            x="Profile", y="Per-family MSE",
            color="Profile", color_discrete_map=_PROFILE_COLORS,
            text="Per-family MSE",
            title=f"Per-family MSE — Method 1, k={val_k}, genes: {', '.join(selected_genes)}",
        )
        fig_mse1.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_mse1.update_layout(height=350, showlegend=False, yaxis_title="MSE")
        st.plotly_chart(fig_mse1, use_container_width=True)

        st.divider()

        st.subheader(f"Per-family breakdown — k = {val_k}")
        st.caption("True πᵢ (blue) vs evidence-based estimate (orange). Each family maps to one selected gene.")

        for profile_name, true_pi_vec in profiles.items():
            family_se = [(e - t) ** 2 for e, t in zip(est_pi_m1, true_pi_vec)]
            mse_val   = float(np.mean(family_se))

            col_tbl, col_fig = st.columns([1, 2])
            with col_tbl:
                st.markdown(f"**{profile_name}** · Est. π̂ = {sum(est_pi_m1):.4f} · MSE = {mse_val:.6f}")
                st.dataframe(
                    pd.DataFrame({
                        "Family":    [f"F{i+1}" for i in range(val_k)],
                        "Gene":      selected_genes,
                        "True πᵢ":   true_pi_vec,
                        "Est. πᵢ":   [round(p, 4) for p in est_pi_m1],
                        "Sq. error": [round(se, 6) for se in family_se],
                    }),
                    use_container_width=True, hide_index=True,
                )
            with col_fig:
                st.plotly_chart(
                    _val_per_family_chart(
                        [f"F{i+1}\n({g})" for i, g in enumerate(selected_genes)],
                        true_pi_vec, est_pi_m1, "Est. πᵢ (evidence)", profile_name,
                    ),
                    use_container_width=True,
                )
            st.divider()

        st.markdown("""
        **What these results mean — Method 1 accuracy**

        Method 1 assigns π based on the quality of experimental evidence in SGD, not on any
        observed network property. Its estimates are bounded between **0.31 and 0.77** by the
        evidence-code weight table — meaning it structurally cannot represent very low (near 0)
        or very high (near 1) inheritance probabilities.

        - **High MSE on Linear:** the Linear profile has families at 0.1 and 0.9, both outside
          Method 1's reachable range. The method compresses all estimates toward the middle,
          underestimating strongly-inherited links and overestimating weakly-inherited ones.
        - **Lower MSE on Quadratic:** the Quadratic profile's values cluster in the 0.3–0.5
          range, which sits within Method 1's window. The error comes from the shape mismatch,
          not a range violation.
        - **Gene-dependence:** changing the selected genes shifts all estimates simultaneously.
          No combination of SGD genes can produce estimates below ~0.31 or above ~0.77, so the
          ceiling and floor errors on extreme profiles are irreducible.
        - **Practical interpretation:** Method 1 is best used as a conservative *prior* —
          a biologically-grounded starting point that reflects how well-studied each regulatory
          link is. It should not be used alone to predict links with extreme inheritance
          probabilities.
        """)

    # ════════════════════════════════════════════════════════════════
    # METHOD 2 branch
    # ════════════════════════════════════════════════════════════════
    elif use_m2:
        all_rows = []
        for profile_name, true_pi_vec in profiles.items():
            true_pi_hat = sum(true_pi_vec)
            for n_val in _VAL_N_VALUES:
                expected_count = expected_partial(true_pi_hat, _VAL_M, n_val)
                est_pi_hat = estimate_pi_hat(expected_count, _VAL_M, n_val)
                if est_pi_hat is None or np.isnan(est_pi_hat):
                    est_pi_hat = float("nan")
                    est_pi_vec = [float("nan")] * val_k
                else:
                    est_pi_vec = [est_pi_hat / val_k] * val_k

                family_se = [
                    (e - t) ** 2
                    for e, t in zip(est_pi_vec, true_pi_vec)
                    if not np.isnan(e)
                ]
                mse = float(np.mean(family_se)) if family_se else float("nan")
                pi_hat_se = (est_pi_hat - true_pi_hat) ** 2 if not np.isnan(est_pi_hat) else float("nan")

                all_rows.append({
                    "Profile":        profile_name,
                    "n":              n_val,
                    "True π̂":        round(true_pi_hat, 4),
                    "E[|M(n)|]":      round(expected_count, 6),
                    "Estimated π̂":   round(est_pi_hat, 8) if not np.isnan(est_pi_hat) else float("nan"),
                    "π̂ sq. error":   f"{pi_hat_se:.2e}" if not np.isnan(pi_hat_se) else "—",
                    "Per-family MSE": round(mse, 6) if not np.isnan(mse) else float("nan"),
                })

        results_df = pd.DataFrame(all_rows)

        st.subheader("Summary — all profiles × n combinations")
        st.caption(
            "π̂ squared error is ~0 for all rows: the Brent inversion recovers the total π̂ "
            "exactly. Per-family MSE is non-zero because Method 2 allocates π̂ equally across "
            f"k families — this only matches a uniform true distribution. "
            f"Analytical MSE is shown over **2 test profiles × {len(_VAL_N_VALUES)} n values** "
            f"({', '.join(str(n) for n in _VAL_N_VALUES)}) = "
            f"**{2 * len(_VAL_N_VALUES)} test combinations**, each with k = {val_k} families. "
            f"See the **1,000-replica Poisson simulation** section below for noise-aware MSE."
        )
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader(f"Per-family MSE by profile (k = {val_k})")
        st.caption("MSE is the same for all n values since π̂ is always recovered exactly.")
        mse_summary = (
            results_df.groupby("Profile")["Per-family MSE"].mean().reset_index()
        )
        mse_summary["Profile"] = pd.Categorical(
            mse_summary["Profile"], categories=["Linear", "Quadratic"], ordered=True
        )
        mse_summary = mse_summary.sort_values("Profile")

        fig_mse = px.bar(
            mse_summary,
            x="Profile", y="Per-family MSE",
            color="Profile", color_discrete_map=_PROFILE_COLORS,
            text="Per-family MSE",
            title=f"Per-family MSE — Method 2, k={val_k}, m={_VAL_M}, uniform allocation",
        )
        fig_mse.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_mse.update_layout(height=350, showlegend=False, yaxis_title="MSE")
        st.plotly_chart(fig_mse, use_container_width=True)

        st.divider()

        st.subheader(f"Per-family breakdown — n = {val_n_detail}, m = {_VAL_M}, k = {val_k}")
        st.caption("True πᵢ (blue) vs uniform allocation π̂/k (orange). The gap drives the MSE.")

        for profile_name, true_pi_vec in profiles.items():
            true_pi_hat = sum(true_pi_vec)
            expected_count = expected_partial(true_pi_hat, _VAL_M, val_n_detail)
            est_pi_hat = estimate_pi_hat(expected_count, _VAL_M, val_n_detail)
            if est_pi_hat is None or np.isnan(est_pi_hat):
                est_pi_hat = float("nan")
                est_pi_vec = [float("nan")] * val_k
            else:
                est_pi_vec = [est_pi_hat / val_k] * val_k

            family_se = [(e - t) ** 2 for e, t in zip(est_pi_vec, true_pi_vec)]
            mse_val = float(np.mean(family_se))

            col_tbl, col_fig = st.columns([1, 2])
            with col_tbl:
                st.markdown(f"**{profile_name}** · π̂ = {true_pi_hat:.2f} · MSE = {mse_val:.6f}")
                st.dataframe(
                    pd.DataFrame({
                        "Family":         [f"F{i+1}" for i in range(val_k)],
                        "True πᵢ":        true_pi_vec,
                        "Est. πᵢ (π̂/k)": [round(p, 4) for p in est_pi_vec],
                        "Sq. error":      [round(se, 6) for se in family_se],
                    }),
                    use_container_width=True, hide_index=True,
                )
            with col_fig:
                st.plotly_chart(
                    _val_per_family_chart(
                        [f"F{i+1}" for i in range(val_k)],
                        true_pi_vec, est_pi_vec, "Est. πᵢ (π̂/k)", profile_name,
                    ),
                    use_container_width=True,
                )
            st.divider()

        # ── Simulation-based MSE (N=1000 Poisson replicas) ──────────────────
        st.divider()
        st.subheader("Simulation-based MSE — 1,000 Poisson replicas of partial duplication")
        st.caption(
            "Unlike all other methods, Method 2 takes an *observed motif count* as input. "
            "In real experiments this count is Poisson-noisy. Below, 1,000 counts are sampled "
            f"from Poisson(λ = E[|M(n)|]) for each profile × n combination (λ from Theorem 4). "
            "Method 2 inverts each sampled count to get π̂, allocates uniformly, and per-family "
            "MSE is averaged over all 1,000 replicas."
        )

        _N_SIM = 1000
        rng = np.random.default_rng(42)
        sim_rows = []
        for profile_name, true_pi_vec in profiles.items():
            true_pi_hat_s = sum(true_pi_vec)
            for n_val in _VAL_N_VALUES:
                lam = expected_partial(true_pi_hat_s, _VAL_M, n_val)
                sim_counts = rng.poisson(lam, size=_N_SIM)
                mse_list = []
                for cnt in sim_counts:
                    ph = estimate_pi_hat(float(cnt), _VAL_M, n_val)
                    if ph is None or np.isnan(ph):
                        continue
                    pv = [ph / val_k] * val_k
                    mse_list.append(float(np.mean([(e - t) ** 2 for e, t in zip(pv, true_pi_vec)])))
                sim_rows.append({
                    "Profile":            profile_name,
                    "n":                  n_val,
                    "λ (E[|M(n)|])":      round(lam, 4),
                    "True π̂":            round(true_pi_hat_s, 4),
                    f"Mean MSE ({_N_SIM} sims)": round(float(np.mean(mse_list)), 6) if mse_list else float("nan"),
                    "Std MSE":            round(float(np.std(mse_list)), 6) if mse_list else float("nan"),
                    "n_converged":        len(mse_list),
                })

        sim_df = pd.DataFrame(sim_rows)
        st.dataframe(sim_df, use_container_width=True, hide_index=True)
        st.caption(
            f"**{_N_SIM} Poisson replicas** per (profile × n) combination "
            f"({_N_SIM} × {2 * len(_VAL_N_VALUES)} = {_N_SIM * 2 * len(_VAL_N_VALUES):,} total inversions). "
            "Mean MSE across replicas is higher than the analytical MSE above because Poisson "
            "noise adds estimation error on top of the allocation error. "
            "'n_converged' is the number of replicas where Brent root-find converged."
        )

        st.markdown("""
        **What these results mean — Method 2 accuracy**

        Method 2 inverts Theorem 4 to find the π̂ that makes the model's expected motif count
        match the observed count. The π̂ squared error is essentially zero for every row —
        the Brent root-find is a mathematical inversion, not an approximation.

        - **The real limitation is allocation, not recovery:** once π̂ is known, Method 2
          distributes it uniformly across k families (π̂/k each). This is the only source of
          per-family MSE. If the true profile is heterogeneous — like Linear, where one family
          inherits rarely and another almost always — uniform allocation misassigns π to every
          family simultaneously.
        - **MSE is profile-shape dependent, not n-dependent:** because π̂ is recovered exactly,
          the per-family error is fixed regardless of whether n = 20 or n = 100. Network size
          only matters when the observed count is noisy (a real experiment); here it is exact.
        - **Simulation MSE > analytical MSE:** the 1,000-replica simulation adds Poisson
          observation noise to the exact expected count. The gap between simulation and
          analytical MSE quantifies how much error comes from count stochasticity versus
          the structural uniform-allocation limitation.
        - **Linear MSE > Quadratic MSE:** the Linear profile has the highest within-profile
          variance. Any constant allocation to a spread-out distribution will produce larger
          average squared error than to a peaked one.
        - **Practical interpretation:** Method 2 is the right tool when you have a reliable
          observed motif count and need an aggregate π̂ for significance testing (the Motif
          Significance tab). It should not be used to infer which specific families have higher
          or lower inheritance without additional weighting information.
        """)

    # ════════════════════════════════════════════════════════════════
    # METHOD 3 branch — real YFL039C calibrated estimator
    # ════════════════════════════════════════════════════════════════
    elif use_m3:
        m3_result  = estimate_pi_from_snp(["YFL039C"] * val_k)
        est_pi_vec = m3_result["pi_vec"]
        mu         = est_pi_vec[0]

        with st.expander("YFL039C calibration data (source of the estimate)"):
            ivec_m3 = _load_inheritance()
            st.caption(
                f"Method 3 is calibrated on **{len(ivec_m3)} strains** of gene YFL039C. "
                f"Each strain's π = 1 − pct_alt/100. The cross-strain mean "
                f"(**μ = {mu:.4f}**) is applied uniformly to all k families."
            )
            fig_cal = px.bar(
                ivec_m3, x="strain", y="pi_snp", color="pct_alt",
                labels={"pi_snp": "πᵢ proxy", "pct_alt": "% alt alleles"},
                color_continuous_scale="RdYlGn_r",
                title="Per-strain π (YFL039C) — calibration source",
            )
            fig_cal.update_layout(height=280, xaxis_tickangle=-30, margin=dict(t=40, b=10))
            fig_cal.add_hline(
                y=mu, line_dash="dash", line_color="black",
                annotation_text=f"μ = {mu:.4f}", annotation_position="right",
            )
            st.plotly_chart(fig_cal, use_container_width=True)
            st.dataframe(
                ivec_m3[["strain", "pct_alt", "pi_snp"]].rename(columns={"pi_snp": "π proxy"}),
                use_container_width=True, hide_index=True, height=220,
            )

        st.divider()

        m3_rows = []
        for profile_name, true_pi_vec in profiles.items():
            family_se = [(mu - t) ** 2 for t in true_pi_vec]
            mse       = float(np.mean(family_se))
            m3_rows.append({
                "Profile":        profile_name,
                "True π̂":        round(sum(true_pi_vec), 4),
                "Estimated π̂":   round(sum(est_pi_vec), 4),
                "Per-family MSE": round(mse, 6),
            })

        st.subheader("Summary — all profiles")
        st.caption(
            f"Method 3 applies the same calibrated estimate (μ = {mu:.4f}) to every "
            f"family. MSE is higher for profiles whose true values spread far from μ. "
            f"MSE is computed analytically over **2 test profiles** (Linear, Quadratic) with "
            f"**k = {val_k}** families. No partial duplication simulation is needed — "
            f"Method 3's estimate comes from YFL039C SNP data, not from observed motif counts."
        )
        st.dataframe(pd.DataFrame(m3_rows), use_container_width=True, hide_index=True)

        st.divider()

        st.subheader(f"Per-family MSE by profile (k = {val_k})")
        fig_mse3 = px.bar(
            pd.DataFrame(m3_rows),
            x="Profile", y="Per-family MSE",
            color="Profile", color_discrete_map=_PROFILE_COLORS,
            text="Per-family MSE",
            title=f"Per-family MSE — Method 3, k={val_k}, μ = {mu:.4f}",
        )
        fig_mse3.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_mse3.update_layout(height=350, showlegend=False, yaxis_title="MSE")
        st.plotly_chart(fig_mse3, use_container_width=True)

        st.divider()

        st.subheader(f"Per-family breakdown — k = {val_k}")
        st.caption(
            f"True πᵢ (blue) vs Method 3 estimate μ = {mu:.4f} (orange, constant across families). "
            "The gap at each family drives the MSE."
        )

        for profile_name, true_pi_vec in profiles.items():
            family_se = [(mu - t) ** 2 for t in true_pi_vec]
            mse_val   = float(np.mean(family_se))

            col_tbl, col_fig = st.columns([1, 2])
            with col_tbl:
                st.markdown(f"**{profile_name}** · Est. π̂ = {mu * val_k:.4f} · MSE = {mse_val:.6f}")
                st.dataframe(
                    pd.DataFrame({
                        "Family":    [f"F{i+1}" for i in range(val_k)],
                        "True πᵢ":   true_pi_vec,
                        "Est. πᵢ":   [round(mu, 4)] * val_k,
                        "Sq. error": [round(se, 6) for se in family_se],
                    }),
                    use_container_width=True, hide_index=True,
                )
            with col_fig:
                st.plotly_chart(
                    _val_per_family_chart(
                        [f"F{i+1}" for i in range(val_k)],
                        true_pi_vec, [mu] * val_k, f"Est. πᵢ (μ={mu:.4f})", profile_name,
                    ),
                    use_container_width=True,
                )
            st.divider()

        st.markdown(f"""
        **What these results mean — Method 3 accuracy**

        Method 3 derives its estimate from SNP divergence at gene YFL039C — the only locus in
        the SGD dataset with dense per-strain variation data. The cross-strain mean
        (μ = {mu:.4f}) is a genuine biological measurement, but it is a **scalar**: the same
        value is applied to every family in the motif.

        - **No per-family differentiation:** Method 3 cannot distinguish between a family
          that almost always inherits its regulatory links and one that rarely does. All k
          families receive μ regardless of their true πᵢ. The per-family bars in the chart
          above are all identical on the orange side.
        - **MSE is driven by profile variance around μ:** the MSE for a given profile equals
          the variance of the true πᵢ values around μ. Quadratic profiles — whose values
          cluster near the center — produce lower MSE than Linear profiles, whose extremes
          (0.1 and 0.9 for k=3) are far from μ ≈ 0.59.
        - **The estimate is well-calibrated for moderate inheritance:** μ ≈ 0.59 sits in a
          biologically plausible range for yeast regulatory links. Strains with 0 alt alleles
          give π = 1.0 (fully conserved site); strains with 2 alt alleles give π = 0 (fully
          diverged). The mean reflects a mix of conserved and diverging links.
        - **Practical interpretation:** Method 3 is best used when you have no gene-specific
          evidence (no evidence codes, no observed motif count) and need a biologically grounded
          single estimate. Its MSE is irreducible for heterogeneous profiles without additional
          per-family sequence data at more loci.
        """)

    # ════════════════════════════════════════════════════════════════
    # METHOD 4 branch — consensus-adjusted (YEASTRACT)
    # ════════════════════════════════════════════════════════════════
    elif use_m4:
        tfs_m4    = _load_tfs().sort_values("gene_name")
        tf_names4 = tfs_m4["gene_name"].tolist()

        # Defaults: k genes evenly spread across the evidence score range
        tfs_sorted_ev4 = tfs_m4.sort_values("pi_prior").reset_index(drop=True)
        indices4       = [int(i * (len(tfs_sorted_ev4) - 1) / (val_k - 1)) for i in range(val_k)]
        default_genes4 = [tfs_sorted_ev4.loc[i, "gene_name"] for i in indices4]

        selected_genes4 = st.multiselect(
            f"Select exactly **{val_k}** TF gene names",
            tf_names4,
            default=default_genes4,
            key="m4_genes",
            help=(
                "Method 4 starts from Method 1's evidence-based π and multiplies by a "
                "YEASTRACT binding-flexibility factor. For genes not in the YEASTRACT "
                "consensus dataset the factor defaults to 1.0 (no adjustment)."
            ),
        )

        if len(selected_genes4) != val_k:
            st.warning(f"Please select exactly {val_k} genes (currently {len(selected_genes4)}).")
            st.stop()

        m4_result   = estimate_pi_consensus_adjusted(selected_genes4)
        est_pi_m4   = m4_result["pi_vec"]
        base_pi_m4  = m4_result["base_pi"]
        factors_m4  = m4_result["consensus_factors"]
        con_stats   = m4_result["consensus_stats"]

        with st.expander("Consensus factors for selected genes"):
            from model.consensus_loader import load_tf_consensus_stats
            all_stats    = load_tf_consensus_stats()
            factor_range = (all_stats["pi_consensus_factor"].min(),
                            all_stats["pi_consensus_factor"].max())
            st.caption(
                f"Consensus factors across all YEASTRACT TFs range from "
                f"**{factor_range[0]:.4f}** (near-zero ambiguity, tight binding) to "
                f"**{factor_range[1]:.4f}** (more degenerate consensus). "
                "Genes not in YEASTRACT receive factor = 1.0."
            )
            st.dataframe(
                pd.DataFrame({
                    "Gene":           selected_genes4,
                    "Base πᵢ (M1)":   [round(b, 4) for b in base_pi_m4],
                    "Consensus factor":[round(f, 4) for f in factors_m4],
                    "Adjusted πᵢ":    [round(p, 4) for p in est_pi_m4],
                }),
                use_container_width=True, hide_index=True,
            )

        st.divider()

        m4_rows = []
        for profile_name, true_pi_vec in profiles.items():
            family_se = [(e - t) ** 2 for e, t in zip(est_pi_m4, true_pi_vec)]
            mse       = float(np.mean(family_se))
            m4_rows.append({
                "Profile":        profile_name,
                "True π̂":        round(sum(true_pi_vec), 4),
                "Estimated π̂":   round(sum(est_pi_m4), 4),
                "Per-family MSE": round(mse, 6),
            })

        st.subheader("Summary — all profiles")
        st.caption(
            f"Method 4 estimates for selected genes: "
            f"{', '.join(f'{g} → {p:.4f}' for g, p in zip(selected_genes4, est_pi_m4))}. "
            f"MSE is computed analytically over **2 test profiles** (Linear, Quadratic) with "
            f"**k = {val_k}** families. No partial duplication simulation is needed — "
            f"Method 4's estimate comes from YEASTRACT consensus data, not from observed motif counts."
        )
        st.dataframe(pd.DataFrame(m4_rows), use_container_width=True, hide_index=True)

        st.divider()

        st.subheader(f"Per-family MSE by profile (k = {val_k})")
        fig_mse4 = px.bar(
            pd.DataFrame(m4_rows),
            x="Profile", y="Per-family MSE",
            color="Profile", color_discrete_map=_PROFILE_COLORS,
            text="Per-family MSE",
            title=(
                f"Per-family MSE — Method 4, k={val_k}, "
                f"genes: {', '.join(selected_genes4)}"
            ),
        )
        fig_mse4.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_mse4.update_layout(height=350, showlegend=False, yaxis_title="MSE")
        st.plotly_chart(fig_mse4, use_container_width=True)

        st.divider()

        st.subheader(f"Per-family breakdown — k = {val_k}")
        st.caption(
            "True πᵢ (blue) vs Method 4 consensus-adjusted estimate (orange). "
            "Orange is always ≥ the Method 1 base (green reference line) because "
            "the consensus factor is always ≥ 1.0."
        )

        for profile_name, true_pi_vec in profiles.items():
            family_se = [(e - t) ** 2 for e, t in zip(est_pi_m4, true_pi_vec)]
            mse_val   = float(np.mean(family_se))

            col_tbl, col_fig = st.columns([1, 2])
            with col_tbl:
                st.markdown(
                    f"**{profile_name}** · Est. π̂ = {sum(est_pi_m4):.4f} · "
                    f"MSE = {mse_val:.6f}"
                )
                st.dataframe(
                    pd.DataFrame({
                        "Family":      [f"F{i+1}" for i in range(val_k)],
                        "Gene":        selected_genes4,
                        "True πᵢ":     true_pi_vec,
                        "Base πᵢ (M1)":[round(b, 4) for b in base_pi_m4],
                        "Factor":      [round(f, 4) for f in factors_m4],
                        "Adj. πᵢ (M4)":[round(p, 4) for p in est_pi_m4],
                        "Sq. error":   [round(se, 6) for se in family_se],
                    }),
                    use_container_width=True, hide_index=True,
                )
            with col_fig:
                x_labels = [f"F{i+1}\n({g})" for i, g in enumerate(selected_genes4)]
                fig_fam4 = go.Figure()
                fig_fam4.add_bar(
                    x=x_labels, y=true_pi_vec,
                    name="True πᵢ", marker_color="#2563eb",
                )
                fig_fam4.add_bar(
                    x=x_labels, y=base_pi_m4,
                    name="Base πᵢ (M1)", marker_color="#94a3b8",
                )
                fig_fam4.add_bar(
                    x=x_labels, y=est_pi_m4,
                    name="Adj. πᵢ (M4)", marker_color="#f97316",
                )
                fig_fam4.update_layout(
                    barmode="group",
                    height=280,
                    title=profile_name,
                    yaxis=dict(title="π", range=[0, 1.05]),
                    legend=dict(orientation="h", y=-0.3),
                    margin=dict(t=40, b=10),
                )
                st.plotly_chart(fig_fam4, use_container_width=True)
            st.divider()

        m4_adj_range = (
            min(est_pi_m4), max(est_pi_m4),
            min(base_pi_m4), max(base_pi_m4),
        )
        st.markdown(f"""
        **What these results mean — Method 4 accuracy**

        Method 4 is built on top of Method 1: it takes the evidence-based π prior and
        applies an upward multiplier derived from each TF's YEASTRACT binding-consensus
        flexibility. A TF whose IUPAC consensus sequence contains many ambiguous positions
        (N, R, Y, …) is a **promiscuous binder** — its regulatory links can survive more
        sequence drift after duplication, so π is adjusted upward.

        - **Always an upward adjustment:** the consensus factor is always ≥ 1.0, so Method 4
          can never produce a lower estimate than Method 1 for the same gene. The selected
          genes have base π in [{m4_adj_range[2]:.4f}, {m4_adj_range[3]:.4f}] and adjusted π
          in [{m4_adj_range[0]:.4f}, {m4_adj_range[1]:.4f}].
        - **Narrow factor range (1.02 – 1.12):** because YEASTRACT consensus sequences are
          generally quite specific (low ambiguity), the adjustment is modest. Method 4 sits
          just above Method 1 in practice — it refines the estimate rather than transforming it.
        - **Same ceiling problem as Method 1:** the combined effect of the evidence-code floor
          (~0.31) and the modest factor means Method 4 still cannot reach extreme π values
          (below ~0.37 or above ~0.82). Linear profile extremes (0.1, 0.9) are out of range.
        - **Gene-dependent and factor-dependent:** changing the gene selection changes both
          the base and the factor simultaneously. TFs not in the YEASTRACT consensus dataset
          receive factor = 1.0, making Method 4 identical to Method 1 for those genes.
        - **Practical interpretation:** Method 4 is best used when you have YEASTRACT binding
          data for your TFs and want to account for the fact that promiscuous binders are more
          likely to maintain regulatory links after duplication. The adjustment is small but
          biologically motivated. Use it in combination with Method 1 to see whether binding
          specificity shifts the estimate meaningfully for your gene set.
        """)

    # ════════════════════════════════════════════════════════════════
    # METHOD 5 branch — Y1000+ TFBS Conservation (π₃)
    # ════════════════════════════════════════════════════════════════
    elif use_m5:
        pi3_df5, pi3_err5 = _val_load_pi3_for_test()

        if pi3_err5:
            st.warning(
                f"Y1000+ π₃ data not available: {pi3_err5}",
                icon="⚠️",
            )
            st.stop()

        tf_opts5 = sorted(pi3_df5["tf_name"].unique().tolist())

        # Defaults: k TFs spread across the mean-π₃ range
        tf_mean5 = (
            pi3_df5.groupby("tf_name")["pi3_estimate"]
            .mean()
            .sort_values()
            .reset_index()
        )
        idx5 = [int(i * (len(tf_mean5) - 1) / max(val_k - 1, 1)) for i in range(val_k)]
        default_genes5 = [tf_mean5.iloc[i]["tf_name"] for i in idx5]

        selected_genes5 = st.multiselect(
            f"Select exactly **{val_k}** TFs (from those in the Y1000+ π₃ dataset)",
            tf_opts5,
            default=default_genes5,
            key="m5_genes",
            help=(
                "Each selected TF's mean retention fraction across all its Y1000+ target-gene "
                "edges becomes its per-family π estimate. Defaults are spread from the "
                "lowest to the highest mean π₃ in the dataset."
            ),
        )

        if len(selected_genes5) != val_k:
            st.warning(f"Please select exactly {val_k} TFs (currently {len(selected_genes5)}).")
            st.stop()

        est_pi_m5 = []
        for tf in selected_genes5:
            tf_rows5 = pi3_df5[pi3_df5["tf_name"] == tf]
            est_pi_m5.append(round(float(tf_rows5["pi3_estimate"].mean()), 4))

        with st.expander("π₃ estimates for selected TFs (source data)"):
            tf_summary5 = []
            for tf, est in zip(selected_genes5, est_pi_m5):
                tf_rows5 = pi3_df5[pi3_df5["tf_name"] == tf]
                tf_summary5.append({
                    "TF":             tf,
                    "n_target_genes": len(tf_rows5),
                    "Mean π₃ (est.)": est,
                    "Min π₃":         round(float(tf_rows5["pi3_estimate"].min()), 4),
                    "Max π₃":         round(float(tf_rows5["pi3_estimate"].max()), 4),
                })
            st.caption(
                "For each selected TF, the mean π₃ is averaged across all its target-gene edges "
                "in the Y1000+ dataset. This scalar is the per-family estimate used below."
            )
            st.dataframe(pd.DataFrame(tf_summary5), use_container_width=True, hide_index=True)

        st.divider()

        m5_rows = []
        for profile_name, true_pi_vec in profiles.items():
            family_se = [(e - t) ** 2 for e, t in zip(est_pi_m5, true_pi_vec)]
            mse = float(np.mean(family_se))
            m5_rows.append({
                "Profile":        profile_name,
                "True π̂":        round(sum(true_pi_vec), 4),
                "Estimated π̂":   round(sum(est_pi_m5), 4),
                "Per-family MSE": round(mse, 6),
            })

        st.subheader("Summary — all profiles")
        st.caption(
            f"Method 5 estimates: "
            f"{', '.join(f'{g} → {p:.4f}' for g, p in zip(selected_genes5, est_pi_m5))}. "
            f"MSE is computed analytically over **2 test profiles** (Linear, Quadratic) with "
            f"**k = {val_k}** families. No partial duplication simulation is needed — "
            f"Method 5's estimate comes from Y1000+ TFBS retention data, not from observed motif counts."
        )
        st.dataframe(pd.DataFrame(m5_rows), use_container_width=True, hide_index=True)

        st.divider()

        st.subheader(f"Per-family MSE by profile (k = {val_k})")
        fig_mse5 = px.bar(
            pd.DataFrame(m5_rows),
            x="Profile", y="Per-family MSE",
            color="Profile", color_discrete_map=_PROFILE_COLORS,
            text="Per-family MSE",
            title=f"Per-family MSE — Method 5 (Y1000+ π₃), k={val_k}, TFs: {', '.join(selected_genes5)}",
        )
        fig_mse5.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_mse5.update_layout(height=350, showlegend=False, yaxis_title="MSE")
        st.plotly_chart(fig_mse5, use_container_width=True)

        st.divider()

        st.subheader(f"Per-family breakdown — k = {val_k}")
        st.caption(
            "True πᵢ (blue) vs Y1000+ π₃ mean retention fraction per TF (orange). "
            "Unlike Methods 2–3, each family receives a different estimate reflecting "
            "that TF's specific conservation history."
        )

        for profile_name, true_pi_vec in profiles.items():
            family_se = [(e - t) ** 2 for e, t in zip(est_pi_m5, true_pi_vec)]
            mse_val = float(np.mean(family_se))

            col_tbl, col_fig = st.columns([1, 2])
            with col_tbl:
                st.markdown(
                    f"**{profile_name}** · Est. π̂ = {sum(est_pi_m5):.4f} · MSE = {mse_val:.6f}"
                )
                st.dataframe(
                    pd.DataFrame({
                        "Family":       [f"F{i+1}" for i in range(val_k)],
                        "TF":           selected_genes5,
                        "True πᵢ":      true_pi_vec,
                        "Est. πᵢ (π₃)": [round(p, 4) for p in est_pi_m5],
                        "Sq. error":    [round(se, 6) for se in family_se],
                    }),
                    use_container_width=True, hide_index=True,
                )
            with col_fig:
                st.plotly_chart(
                    _val_per_family_chart(
                        [f"F{i+1}\n({g})" for i, g in enumerate(selected_genes5)],
                        true_pi_vec, est_pi_m5, "Est. πᵢ (π₃)", profile_name,
                    ),
                    use_container_width=True,
                )
            st.divider()

        overall_mean_pi3_5 = round(float(pi3_df5["pi3_estimate"].mean()), 3)
        st.markdown(f"""
        **What these results mean — Method 5 (Y1000+ π₃) accuracy**

        Method 5 derives each family's estimate from empirical TFBS conservation data: for each
        selected TF, the fraction of Y1000+ genomes (spanning ~1 billion years of yeast evolution)
        retaining a significant PWM hit upstream of the orthologous gene is averaged across all of
        that TF's target genes. This retention fraction is used directly as πᵢ.

        This is qualitatively different from all four SGD-based methods. Method 5 does not
        infer π from evidence codes, model moments, or single-gene SNP counts — it *measures*
        how often regulatory links are retained across independent evolutionary replicates.

        - **Per-family differentiation:** unlike Methods 2 and 3, each family in Method 5 gets
          its own estimate based on its TF's evolutionary conservation history. A TF whose binding
          sites are retained in most yeasts gets high π; a lineage-specific TF gets low π.
        - **Overall mean π₃ = {overall_mean_pi3_5}** across all TF→gene edges in the Y1000+ dataset.
          TFs with mean π₃ above this are better-conserved-than-average regulators.
        - **Range: [0, 1] with no structural floor/ceiling:** unlike Methods 1/4 (bounded by
          evidence-code range ~0.31–0.82), Method 5 can in principle recover any π value from
          0 to 1, limited only by the diversity of the Y1000+ species panel and PWM sensitivity.
        - **MSE reflects biological heterogeneity:** high MSE means the selected TFs' conservation
          patterns do not align with the synthetic profile's shape. This is biologically informative —
          it tells you how well the profile matches the evolutionary reality for those TFs.
        - **Practical interpretation:** Method 5 is the most data-rich estimate when Y1000+ data
          is available. Its MSE reflects real biological heterogeneity rather than structural model
          limitations, making it the most trustworthy indicator of actual inheritance for TFs
          with good JASPAR PWM coverage in the Y1000+ dataset.
        """)
    # ════════════════════════════════════════════════════════════════
    # METHOD 6 branch — Y1000+ Sequence Homology (π₂)
    # ════════════════════════════════════════════════════════════════
    elif use_m6:
        pi2_df6, pi2_err6 = _val_load_pi2_for_test()

        if pi2_err6:
            st.warning(f"Y1000+ π₂ data not available: {pi2_err6}", icon="⚠️")
            st.stop()

        all_tfs6 = sorted(set(pi2_df6["gene1"].tolist() + pi2_df6["gene2"].tolist()))

        def _mean_pi2(tf, df):
            rows = df[(df["gene1"] == tf) | (df["gene2"] == tf)]
            if rows.empty:
                return float("nan")
            return float(rows["pi2_estimate"].mean())

        tf_mean6 = pd.DataFrame({
            "tf_name":  all_tfs6,
            "mean_pi2": [_mean_pi2(tf, pi2_df6) for tf in all_tfs6],
        }).sort_values("mean_pi2").reset_index(drop=True)

        idx6 = [int(i * (len(tf_mean6) - 1) / max(val_k - 1, 1)) for i in range(val_k)]
        default_genes6 = [tf_mean6.iloc[i]["tf_name"] for i in idx6]

        selected_genes6 = st.multiselect(
            f"Select exactly **{val_k}** TFs (from those in the Y1000+ π₂ dataset)",
            all_tfs6,
            default=default_genes6,
            key="m6_genes",
            help=(
                "Each selected TF's mean pairwise protein sequence identity with all other "
                "TFs in the dataset becomes its per-family π estimate (π₂ = pct_identity / 100). "
                "Defaults are spread from the lowest to the highest mean π₂."
            ),
        )

        if len(selected_genes6) != val_k:
            st.warning(f"Please select exactly {val_k} TFs (currently {len(selected_genes6)}).")
            st.stop()

        est_pi_m6 = [round(_mean_pi2(tf, pi2_df6), 4) for tf in selected_genes6]

        with st.expander("π₂ estimates for selected TFs (source data)"):
            tf_summary6 = []
            for tf, est in zip(selected_genes6, est_pi_m6):
                rows6 = pi2_df6[(pi2_df6["gene1"] == tf) | (pi2_df6["gene2"] == tf)]
                tf_summary6.append({
                    "TF":             tf,
                    "n_pairs":        len(rows6),
                    "Mean π₂ (est.)": est,
                    "Min π₂":         round(float(rows6["pi2_estimate"].min()), 4),
                    "Max π₂":         round(float(rows6["pi2_estimate"].max()), 4),
                })
            st.caption(
                "For each selected TF, the mean π₂ is averaged across all pairwise alignments "
                "in which the TF appears. This scalar is the per-family estimate used below."
            )
            st.dataframe(pd.DataFrame(tf_summary6), use_container_width=True, hide_index=True)

        st.divider()

        m6_rows = []
        for profile_name, true_pi_vec in profiles.items():
            family_se = [(e - t) ** 2 for e, t in zip(est_pi_m6, true_pi_vec)]
            mse = float(np.mean(family_se))
            m6_rows.append({
                "Profile":        profile_name,
                "True π̂":        round(sum(true_pi_vec), 4),
                "Estimated π̂":   round(sum(est_pi_m6), 4),
                "Per-family MSE": round(mse, 6),
            })

        st.subheader("Summary — all profiles")
        st.caption(
            f"Method 6 estimates: "
            f"{', '.join(f'{g} → {p:.4f}' for g, p in zip(selected_genes6, est_pi_m6))}. "
            f"MSE is computed analytically over **2 test profiles** (Linear, Quadratic) with "
            f"**k = {val_k}** families. No partial duplication simulation is needed — "
            f"Method 6's estimate comes from Y1000+ protein sequence identity data, not from observed motif counts."
        )
        st.dataframe(pd.DataFrame(m6_rows), use_container_width=True, hide_index=True)

        st.divider()

        st.subheader(f"Per-family MSE by profile (k = {val_k})")
        fig_mse6 = px.bar(
            pd.DataFrame(m6_rows),
            x="Profile", y="Per-family MSE",
            color="Profile", color_discrete_map=_PROFILE_COLORS,
            text="Per-family MSE",
            title=f"Per-family MSE — Method 6 (Y1000+ π₂), k={val_k}, TFs: {', '.join(selected_genes6)}",
        )
        fig_mse6.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_mse6.update_layout(height=350, showlegend=False, yaxis_title="MSE")
        st.plotly_chart(fig_mse6, use_container_width=True)

        st.divider()

        st.subheader(f"Per-family breakdown — k = {val_k}")
        st.caption(
            "True πᵢ (blue) vs Y1000+ π₂ mean pairwise identity per TF (orange). "
            "Each family receives a different estimate reflecting its TF's average "
            "sequence similarity to all other TFs in the dataset."
        )

        for profile_name, true_pi_vec in profiles.items():
            family_se = [(e - t) ** 2 for e, t in zip(est_pi_m6, true_pi_vec)]
            mse_val = float(np.mean(family_se))
            col_tbl, col_fig = st.columns([1, 2])
            with col_tbl:
                st.markdown(
                    f"**{profile_name}** · Est. π̂ = {sum(est_pi_m6):.4f} · MSE = {mse_val:.6f}"
                )
                st.dataframe(
                    pd.DataFrame({
                        "Family":       [f"F{i+1}" for i in range(val_k)],
                        "TF":           selected_genes6,
                        "True πᵢ":      true_pi_vec,
                        "Est. πᵢ (π₂)": [round(p, 4) for p in est_pi_m6],
                        "Sq. error":    [round(se, 6) for se in family_se],
                    }),
                    use_container_width=True, hide_index=True,
                )
            with col_fig:
                st.plotly_chart(
                    _val_per_family_chart(
                        [f"F{i+1}\n({g})" for i, g in enumerate(selected_genes6)],
                        true_pi_vec, est_pi_m6, "Est. πᵢ (π₂)", profile_name,
                    ),
                    use_container_width=True,
                )
            st.divider()

        overall_mean_pi2_6 = round(float(pi2_df6["pi2_estimate"].mean()), 3)
        st.markdown(f"""
        **What these results mean — Method 6 (Y1000+ π₂) accuracy**

        Method 6 measures how similar each TF's protein sequence is to every other TF
        in the Y1000+ dataset on average. A TF with high mean pairwise identity (π₂ close to 1)
        is structurally very similar to its paralogs, suggesting regulatory links would be
        retained across duplication. A TF with low mean identity (π₂ near 0) has diverged
        substantially — regulatory links are less likely to survive.

        - **Protein proxy, not binding proxy:** π₂ does not directly measure binding site
          retention. A TF can have high sequence identity but bind different sites after
          duplication. Method 5 (TFBS conservation) is more direct for TFs with JASPAR coverage.
        - **Broadest coverage:** π₂ is available for every TF with a Y1000+ FASTA entry — no
          JASPAR PWM is required. For TFs without PWM data (not scorable by Methods 5/7), π₂
          is often the only cross-species estimate available.
        - **Overall mean π₂ = {overall_mean_pi2_6}** across all pairwise TF comparisons.
          TFs above this are more structurally conserved relative to their paralogs.
        - **Practical interpretation:** use Method 6 as a cross-species baseline when JASPAR
          data is unavailable, or as a structural complement to the binding-site-focused
          Methods 5 and 7 for TFs where all three are available.
        """)

    # ════════════════════════════════════════════════════════════════
    # METHOD 7 branch — Y1000+ Binding-site SNPs (π₄)
    # ════════════════════════════════════════════════════════════════
    elif use_m7:
        pi4_df7, pi4_err7 = _val_load_pi4_for_test()

        if pi4_err7:
            st.warning(f"Y1000+ π₄ data not available: {pi4_err7}", icon="⚠️")
            st.stop()

        tf_opts7 = sorted(pi4_df7["tf_name"].unique().tolist())

        tf_mean7 = (
            pi4_df7.groupby("tf_name")["pi4_estimate"]
            .mean()
            .sort_values()
            .reset_index()
        )
        idx7 = [int(i * (len(tf_mean7) - 1) / max(val_k - 1, 1)) for i in range(val_k)]
        default_genes7 = [tf_mean7.iloc[i]["tf_name"] for i in idx7]

        selected_genes7 = st.multiselect(
            f"Select exactly **{val_k}** TFs (from those in the Y1000+ π₄ dataset)",
            tf_opts7,
            default=default_genes7,
            key="m7_genes",
            help=(
                "Each selected TF's mean IC-weighted polymorphism rate across all its "
                "target genes becomes its per-family π estimate "
                "(π₄ = 1 − Σ IC_weight[pos] × polymorphism_rate[pos]). "
                "Defaults are spread from the lowest to the highest mean π₄."
            ),
        )

        if len(selected_genes7) != val_k:
            st.warning(f"Please select exactly {val_k} TFs (currently {len(selected_genes7)}).")
            st.stop()

        est_pi_m7 = []
        for tf in selected_genes7:
            tf_rows7 = pi4_df7[pi4_df7["tf_name"] == tf]
            est_pi_m7.append(round(float(tf_rows7["pi4_estimate"].mean()), 4))

        with st.expander("π₄ estimates for selected TFs (source data)"):
            tf_summary7 = []
            for tf, est in zip(selected_genes7, est_pi_m7):
                tf_rows7 = pi4_df7[pi4_df7["tf_name"] == tf]
                tf_summary7.append({
                    "TF":             tf,
                    "n_target_genes": len(tf_rows7),
                    "Mean π₄ (est.)": est,
                    "Min π₄":         round(float(tf_rows7["pi4_estimate"].min()), 4),
                    "Max π₄":         round(float(tf_rows7["pi4_estimate"].max()), 4),
                })
            st.caption(
                "For each selected TF, mean π₄ is averaged across all its target-gene edges. "
                "π₄ = 1 − Σ IC_weight[pos] × polymorphism_rate[pos]: positions with high "
                "information content are weighted more heavily."
            )
            st.dataframe(pd.DataFrame(tf_summary7), use_container_width=True, hide_index=True)

        st.divider()

        m7_rows = []
        for profile_name, true_pi_vec in profiles.items():
            family_se = [(e - t) ** 2 for e, t in zip(est_pi_m7, true_pi_vec)]
            mse = float(np.mean(family_se))
            m7_rows.append({
                "Profile":        profile_name,
                "True π̂":        round(sum(true_pi_vec), 4),
                "Estimated π̂":   round(sum(est_pi_m7), 4),
                "Per-family MSE": round(mse, 6),
            })

        st.subheader("Summary — all profiles")
        st.caption(
            f"Method 7 estimates: "
            f"{', '.join(f'{g} → {p:.4f}' for g, p in zip(selected_genes7, est_pi_m7))}. "
            f"MSE is computed analytically over **2 test profiles** (Linear, Quadratic) with "
            f"**k = {val_k}** families. No partial duplication simulation is needed — "
            f"Method 7's estimate comes from Y1000+ IC-weighted SNP data, not from observed motif counts."
        )
        st.dataframe(pd.DataFrame(m7_rows), use_container_width=True, hide_index=True)

        st.divider()

        st.subheader(f"Per-family MSE by profile (k = {val_k})")
        fig_mse7 = px.bar(
            pd.DataFrame(m7_rows),
            x="Profile", y="Per-family MSE",
            color="Profile", color_discrete_map=_PROFILE_COLORS,
            text="Per-family MSE",
            title=f"Per-family MSE — Method 7 (Y1000+ π₄), k={val_k}, TFs: {', '.join(selected_genes7)}",
        )
        fig_mse7.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_mse7.update_layout(height=350, showlegend=False, yaxis_title="MSE")
        st.plotly_chart(fig_mse7, use_container_width=True)

        st.divider()

        st.subheader(f"Per-family breakdown — k = {val_k}")
        st.caption(
            "True πᵢ (blue) vs Y1000+ π₄ IC-weighted polymorphism rate per TF (orange). "
            "Each family gets a separate estimate from its TF's binding-site mutation "
            "rate across Y1000+ genomes."
        )

        for profile_name, true_pi_vec in profiles.items():
            family_se = [(e - t) ** 2 for e, t in zip(est_pi_m7, true_pi_vec)]
            mse_val = float(np.mean(family_se))
            col_tbl, col_fig = st.columns([1, 2])
            with col_tbl:
                st.markdown(
                    f"**{profile_name}** · Est. π̂ = {sum(est_pi_m7):.4f} · MSE = {mse_val:.6f}"
                )
                st.dataframe(
                    pd.DataFrame({
                        "Family":       [f"F{i+1}" for i in range(val_k)],
                        "TF":           selected_genes7,
                        "True πᵢ":      true_pi_vec,
                        "Est. πᵢ (π₄)": [round(p, 4) for p in est_pi_m7],
                        "Sq. error":    [round(se, 6) for se in family_se],
                    }),
                    use_container_width=True, hide_index=True,
                )
            with col_fig:
                st.plotly_chart(
                    _val_per_family_chart(
                        [f"F{i+1}\n({g})" for i, g in enumerate(selected_genes7)],
                        true_pi_vec, est_pi_m7, "Est. πᵢ (π₄)", profile_name,
                    ),
                    use_container_width=True,
                )
            st.divider()

        overall_mean_pi4_7 = round(float(pi4_df7["pi4_estimate"].mean()), 3)
        st.markdown(f"""
        **What these results mean — Method 7 (Y1000+ π₄) accuracy**

        Method 7 estimates inheritance probability from the mutation rate at the exact
        binding-site positions of each TF's target genes across Y1000+ genomes. Positions
        with high information content (IC) — most critical to binding specificity — are
        up-weighted. A conserved binding site gets high π₄; a rapidly mutating site gets low π₄.

        - **Mechanistic precision:** unlike π₂ (whole-protein homology) or π₃ (site
          presence/absence), π₄ captures *which positions* within the binding site are under
          selection. This is the most mechanistically precise of the three Y1000+ methods.
        - **PWM required:** only available for TFs with a JASPAR 2024 PWM entry. TFs not in
          JASPAR cannot be scored with this method.
        - **Complements π₃:** Method 5 asks "is there any binding site here?" while Method 7
          asks "how conserved are the specific positions?" A site can be present (high π₃)
          but diverging at critical positions (low π₄).
        - **Overall mean π₄ = {overall_mean_pi4_7}** across all TF→gene edges in the dataset.
        - **Practical interpretation:** Method 7 is best when you need position-level resolution
          to detect subtle binding-site erosion not visible in site-presence/absence data.
          Use alongside Method 5 for a complete picture of binding-site evolution.
        """)

    # ════════════════════════════════════════════════════════════════
    # ENSEMBLE branch — per-family multi-signal ensemble
    # ════════════════════════════════════════════════════════════════
    elif use_ens:
        st.subheader("Why a per-family ensemble?")
        st.markdown("""
Every method in this tab makes a different biological assumption about what
drives regulatory inheritance:

| Signal | What it measures |
|--------|-----------------|
| **M1 — Evidence** | How well-confirmed the TF's regulatory role is in *S. cerevisiae* |
| **M3 — SNP Divergence** | How much sequence has drifted at gene YFL039C across strains |
| **M4 — Consensus** | Whether the TF's binding sequence is flexible enough to tolerate drift |
| **M5 — TFBS Conservation** | How often the binding site is retained in 1,154 yeast genomes |
| **M6 — Seq. Homology** | How similar the TF's protein sequence is to its paralogs |
| **M7 — Binding-site SNPs** | How many mutations have accumulated at the exact binding positions |

No single method is complete. Evidence codes say nothing about cross-species
conservation. Sequence homology doesn't capture whether the binding *site*
survived. TFBS conservation doesn't account for how easy it is to bind in the
first place. By averaging across all available signals, the ensemble reduces the
impact of any one method's blind spots.

**Why is Method 2 excluded?** Theorem 4 gives the expected number of motif
instances as a function of *π̂ = Σπᵢ only*. Inverting it recovers the total
inheritance probability perfectly — but tells you nothing about whether that
total is concentrated in one family or spread evenly across all of them.
Feeding a uniform π̂/k allocation into the ensemble would just anchor every
family toward the same value and dilute the per-family signal from the other
methods.
""")

        tfs_ens      = _load_tfs().sort_values("gene_name")
        tf_names_ens = tfs_ens["gene_name"].tolist()
        tfs_srt_ens  = tfs_ens.sort_values("pi_prior").reset_index(drop=True)
        idx_ens      = [int(i * (len(tfs_srt_ens) - 1) / max(val_k - 1, 1)) for i in range(val_k)]
        default_ens  = [tfs_srt_ens.loc[i, "gene_name"] for i in idx_ens]

        selected_ens = st.multiselect(
            f"Select exactly **{val_k}** TF gene names",
            tf_names_ens,
            default=default_ens,
            key="ens_genes",
            help=(
                "Each selected TF is treated as one gene family. The ensemble runs "
                "all six eligible methods for every family and averages the results."
            ),
        )

        if len(selected_ens) != val_k:
            st.warning(f"Please select exactly {val_k} genes (currently {len(selected_ens)}).")
            st.stop()

        with st.spinner("Running all signals…"):
            ens_result  = estimate_pi_per_family_ensemble(selected_ens)

        est_pi_ens  = ens_result["pi_vec"]
        details     = ens_result["per_family_details"]
        all_signals = ens_result["all_signals"]

        # ── Per-family signal breakdown ───────────────────────────────────
        st.subheader("Per-family signal breakdown")
        st.caption(
            "Each row is one gene family. Columns show the estimate from each method. "
            "The ensemble mean is the unweighted average of all non-missing values. "
            "±σ shows how much the methods disagree — a large spread means conflicting "
            "signals and the estimate should be treated with more caution."
        )
        signal_labels = list(all_signals.keys())
        detail_rows = []
        for d in details:
            row = {"Gene": d["gene"]}
            for lbl in signal_labels:
                row[lbl] = d.get(lbl, None)
            row["Ensemble πᵢ"] = d["pi_ensemble"]
            row["±σ"]          = d["pi_std"]
            row["# signals"]   = d["n_signals"]
            detail_rows.append(row)
        detail_df = pd.DataFrame(detail_rows)
        st.dataframe(
            detail_df.style.format(
                {c: "{:.4f}" for c in detail_df.columns if c not in ("Gene", "# signals")}
            ),
            use_container_width=True, hide_index=True,
        )

        with st.expander("Which signals are missing?"):
            st.markdown(
                "M5, M6, and M7 require Y1000+ data (generate it in the Y1000+ tab). "
                "M3 returns the dataset-wide YFL039C mean for genes not in that file, "
                "so it will look identical across all families if none of the selected "
                "TFs are in the YFL039C SNP dataset."
            )
            avail_rows = []
            for lbl, vec in all_signals.items():
                n_avail = sum(
                    1 for v in vec
                    if v is not None and not np.isnan(float(v))
                )
                avail_rows.append({"Signal": lbl, "Available for": f"{n_avail} / {val_k} families"})
            st.dataframe(pd.DataFrame(avail_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Per-signal bar chart with ensemble overlay ────────────────────
        st.subheader("Signal comparison — all families")
        x_labels_ens = [f"F{i+1}\n({g})" for i, g in enumerate(selected_ens)]
        fig_ens_sig = go.Figure()
        _sig_colors = {
            "M1 Evidence":          "#94a3b8",
            "M3 SNP Divergence":    "#64748b",
            "M4 Consensus":         "#475569",
            "M5 TFBS Conservation": "#0ea5e9",
            "M6 Seq. Homology":     "#7c3aed",
            "M7 Binding-site SNPs": "#10b981",
        }
        for lbl, color in _sig_colors.items():
            vec = all_signals.get(lbl, [])
            y_vals = [
                v if (v is not None and not np.isnan(float(v))) else None
                for v in vec
            ]
            if any(v is not None for v in y_vals):
                fig_ens_sig.add_bar(
                    x=x_labels_ens, y=y_vals,
                    name=lbl, marker_color=color, opacity=0.75,
                )
        fig_ens_sig.add_scatter(
            x=x_labels_ens, y=est_pi_ens,
            mode="markers+lines",
            name="Ensemble πᵢ",
            marker=dict(color="#d62728", size=10, symbol="diamond"),
            line=dict(color="#d62728", width=2, dash="dash"),
        )
        fig_ens_sig.update_layout(
            barmode="group", height=380,
            title="Per-family estimates from each signal + ensemble mean",
            yaxis=dict(title="π", range=[0, 1.05]),
            legend=dict(orientation="h", y=-0.4),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_ens_sig, use_container_width=True)

        st.divider()

        # ── MSE against synthetic profiles ────────────────────────────────
        st.subheader(f"Accuracy against synthetic profiles (k = {val_k})")
        st.caption(
            f"MSE is computed analytically over **2 test profiles** (Linear, Quadratic) with "
            f"**k = {val_k}** families. No partial duplication simulation is needed — "
            "the ensemble averages M1, M3, M4, M5, M6, and M7, none of which use observed motif counts. "
            "The number of contributing signals per family is shown in the breakdown table above (# signals)."
        )
        ens_rows = []
        for profile_name, true_pi_vec in profiles.items():
            family_se = [(e - t) ** 2 for e, t in zip(est_pi_ens, true_pi_vec)]
            mse = float(np.mean(family_se))
            ens_rows.append({
                "Profile":        profile_name,
                "True π̂":        round(sum(true_pi_vec), 4),
                "Estimated π̂":   round(sum(v for v in est_pi_ens if not np.isnan(v)), 4),
                "Per-family MSE": round(mse, 6),
            })
        st.dataframe(pd.DataFrame(ens_rows), use_container_width=True, hide_index=True)

        fig_mse_ens = px.bar(
            pd.DataFrame(ens_rows),
            x="Profile", y="Per-family MSE",
            color="Profile", color_discrete_map=_PROFILE_COLORS,
            text="Per-family MSE",
            title=f"Per-family MSE — Ensemble, k={val_k}, genes: {', '.join(selected_ens)}",
        )
        fig_mse_ens.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_mse_ens.update_layout(height=350, showlegend=False, yaxis_title="MSE")
        st.plotly_chart(fig_mse_ens, use_container_width=True)

        st.divider()
        st.subheader(f"Per-family breakdown — k = {val_k}")
        for profile_name, true_pi_vec in profiles.items():
            family_se = [(e - t) ** 2 for e, t in zip(est_pi_ens, true_pi_vec)]
            mse_val   = float(np.mean(family_se))
            col_tbl, col_fig = st.columns([1, 2])
            with col_tbl:
                st.markdown(
                    f"**{profile_name}** · Est. π̂ = "
                    f"{sum(v for v in est_pi_ens if not np.isnan(v)):.4f} · "
                    f"MSE = {mse_val:.6f}"
                )
                st.dataframe(
                    pd.DataFrame({
                        "Family":      [f"F{i+1}" for i in range(val_k)],
                        "Gene":        selected_ens,
                        "True πᵢ":     true_pi_vec,
                        "Ensemble πᵢ": [round(p, 4) for p in est_pi_ens],
                        "Sq. error":   [round(se, 6) for se in family_se],
                    }),
                    use_container_width=True, hide_index=True,
                )
            with col_fig:
                st.plotly_chart(
                    _val_per_family_chart(
                        x_labels_ens, true_pi_vec, est_pi_ens,
                        "Ensemble πᵢ", profile_name,
                    ),
                    use_container_width=True,
                )
            st.divider()

        n_sgd_only = sum(1 for d in details if d["n_signals"] <= 3)
        st.markdown(f"""
**What these results mean — Multi-signal Ensemble accuracy**

The ensemble is the most data-rich per-family estimate available. Because it
combines up to six independent biological signals, it is less likely than any
single method to be systematically wrong in one direction.

**How to read the signal breakdown table:**
- A family with **6 signals** has the most reliable estimate — all data sources contribute.
- A family with **3 signals** uses only the SGD-based methods (M1, M3, M4). This is still
  useful but misses the cross-species perspective from Y1000+.
- A family with a **high ±σ** means the methods disagree. That conflict is itself
  informative: a TF well-established in *S. cerevisiae* but poorly conserved across
  yeasts will show high M1/M4 and low M5/M7, signalling a recent or species-specific
  regulatory link.

**Why the ensemble is not always the best on MSE:**
- Against a **Linear profile** (maximally spread π values), the ensemble is only as
  differentiated as its best per-family signals (M5 and M7). If Y1000+ data is missing,
  M1/M3/M4 all tend toward a similar range and the ensemble may not outperform M1 alone.
- Against a **Quadratic profile** (values clustered near the centre), the SGD signals
  already do well, and adding cross-species signals may increase variance without
  reducing bias.

**Practical recommendation:** use the ensemble πᵢ as your default per-family estimate.
Use the per-signal breakdown table to understand *why* a family's estimate is high or
low — that breakdown is more informative than the final number alone.

Currently **{n_sgd_only} of {val_k}** selected {"TF" if n_sgd_only == 1 else "TFs"} have
only SGD signals (Y1000+ data missing or gene not covered). Generate the Y1000+ data in
the **Y1000+ π Estimators** tab to unlock M5, M6, and M7 for those families.
""")

    # ════════════════════════════════════════════════════════════════
    # COMPARE ALL METHODS — side-by-side MSE across all methods
    # ════════════════════════════════════════════════════════════════
    st.divider()

    with st.expander("📊 Compare All Methods — which estimator is most accurate?", expanded=False):
        st.markdown(
            "Runs all 7 estimation methods plus the Ensemble against the **Linear** and **Quadratic** "
            f"true-π profiles (k = {val_k}, m = {_VAL_M}) and computes per-family MSE for each. "
            "Methods 1, 4, 5, 6, 7, and Ensemble use the gene set selected below; "
            "Methods 2 and 3 are parameterless (M2 uses n = 50). "
            "Methods 5, 6, 7, and Ensemble Y1000+ signals require pre-generated Y1000+ data."
        )

        # Gene selector shared by M1, M4, M5, M6, M7, Ensemble
        _cmp_tfs = _load_tfs().sort_values("pi_prior").reset_index(drop=True)
        _cmp_tf_names = _load_tfs().sort_values("gene_name")["gene_name"].tolist()
        _cmp_idx = [int(i * (len(_cmp_tfs) - 1) / max(val_k - 1, 1)) for i in range(val_k)]
        _cmp_defaults = [_cmp_tfs.loc[i, "gene_name"] for i in _cmp_idx]

        cmp_genes = st.multiselect(
            f"Select **{val_k}** TFs for Methods 1, 4, 5, 6, 7, and Ensemble (same gene set):",
            _cmp_tf_names,
            default=_cmp_defaults,
            key="compare_genes",
        )

        if len(cmp_genes) != val_k:
            st.info(f"Select exactly {val_k} TFs above to run the comparison (currently {len(cmp_genes)}).")
        else:
            _M_COLORS = {
                "M1 Evidence":       "#2563eb",
                "M2 Moment":         "#16a34a",
                "M3 SNP":            "#d97706",
                "M4 Consensus":      "#7c3aed",
                "M5 π₃ TFBS":        "#dc2626",
                "M6 π₂ Seq.Hom.":   "#0891b2",
                "M7 π₄ SNP Sites":   "#059669",
                "Ensemble":          "#0f172a",
            }

            # Compute per-method estimate vectors
            _pi_m1 = estimate_pi_from_evidence(cmp_genes)["pi_vec"]
            _pi_m4 = estimate_pi_consensus_adjusted(cmp_genes)["pi_vec"]
            _mu_m3 = estimate_pi_from_snp(["YFL039C"] * val_k)["pi_vec"][0]
            _pi_m3 = [_mu_m3] * val_k

            # M5 — π₃ TFBS conservation
            _pi3_cmp, _pi3_cmp_err = _val_load_pi3_for_test()
            if _pi3_cmp is not None:
                _overall_m5 = float(_pi3_cmp["pi3_estimate"].mean())
                _pi_m5 = []
                for _tf in cmp_genes:
                    _rows = _pi3_cmp[_pi3_cmp["tf_name"] == _tf]
                    _pi_m5.append(
                        float(_rows["pi3_estimate"].mean()) if not _rows.empty else _overall_m5
                    )
            else:
                _pi_m5 = None

            # M6 — π₂ sequence homology
            _pi2_cmp, _pi2_cmp_err = _val_load_pi2_for_test()
            if _pi2_cmp is not None:
                _overall_m6 = float(_pi2_cmp["pi2_estimate"].mean())
                _pi_m6 = []
                for _tf in cmp_genes:
                    _rows_m6 = _pi2_cmp[
                        (_pi2_cmp["gene1"] == _tf) | (_pi2_cmp["gene2"] == _tf)
                    ]
                    _pi_m6.append(
                        float(_rows_m6["pi2_estimate"].mean()) if not _rows_m6.empty else _overall_m6
                    )
            else:
                _pi_m6 = None

            # M7 — π₄ IC-weighted SNP at binding sites
            _pi4_cmp, _pi4_cmp_err = _val_load_pi4_for_test()
            if _pi4_cmp is not None:
                _overall_m7 = float(_pi4_cmp["pi4_estimate"].mean())
                _pi_m7 = []
                for _tf in cmp_genes:
                    _rows_m7 = _pi4_cmp[_pi4_cmp["tf_name"] == _tf]
                    _pi_m7.append(
                        float(_rows_m7["pi4_estimate"].mean()) if not _rows_m7.empty else _overall_m7
                    )
            else:
                _pi_m7 = None

            with st.spinner("Running ensemble…"):
                _ens_cmp = estimate_pi_per_family_ensemble(cmp_genes)
            _pi_ens = _ens_cmp["pi_vec"]

            cmp_rows = []
            for profile_name, true_pi_vec in profiles.items():
                # M1
                _se_m1 = float(np.mean([(e - t) ** 2 for e, t in zip(_pi_m1, true_pi_vec)]))
                # M2 — Brent inversion with n=50, uniform allocation
                _ph = sum(true_pi_vec)
                _exp = expected_partial(_ph, _VAL_M, 50)
                _est_ph = estimate_pi_hat(_exp, _VAL_M, 50)
                if _est_ph and not np.isnan(_est_ph):
                    _pi_m2 = [_est_ph / val_k] * val_k
                    _se_m2 = float(np.mean([(e - t) ** 2 for e, t in zip(_pi_m2, true_pi_vec)]))
                else:
                    _se_m2 = float("nan")
                # M3
                _se_m3 = float(np.mean([(_mu_m3 - t) ** 2 for t in true_pi_vec]))
                # M4
                _se_m4 = float(np.mean([(e - t) ** 2 for e, t in zip(_pi_m4, true_pi_vec)]))
                # M5
                _se_m5 = (
                    float(np.mean([(e - t) ** 2 for e, t in zip(_pi_m5, true_pi_vec)]))
                    if _pi_m5 else float("nan")
                )
                # M6
                _se_m6 = (
                    float(np.mean([(e - t) ** 2 for e, t in zip(_pi_m6, true_pi_vec)]))
                    if _pi_m6 else float("nan")
                )
                # M7
                _se_m7 = (
                    float(np.mean([(e - t) ** 2 for e, t in zip(_pi_m7, true_pi_vec)]))
                    if _pi_m7 else float("nan")
                )
                # Ensemble
                _se_ens = float(np.mean([(e - t) ** 2 for e, t in zip(_pi_ens, true_pi_vec)]))

                cmp_rows.append({
                    "Profile":          profile_name,
                    "M1 Evidence":      round(_se_m1, 6),
                    "M2 Moment":        round(_se_m2, 6) if not np.isnan(_se_m2) else float("nan"),
                    "M3 SNP":           round(_se_m3, 6),
                    "M4 Consensus":     round(_se_m4, 6),
                    "M5 π₃ TFBS":       round(_se_m5, 6) if not np.isnan(_se_m5) else float("nan"),
                    "M6 π₂ Seq.Hom.":  round(_se_m6, 6) if not np.isnan(_se_m6) else float("nan"),
                    "M7 π₄ SNP Sites":  round(_se_m7, 6) if not np.isnan(_se_m7) else float("nan"),
                    "Ensemble":         round(_se_ens, 6),
                })

            cmp_df = pd.DataFrame(cmp_rows)
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            # Grouped bar chart
            _cmp_long = cmp_df.melt(id_vars="Profile", var_name="Method", value_name="Per-family MSE")
            _cmp_long = _cmp_long.dropna(subset=["Per-family MSE"])
            fig_cmp_all = px.bar(
                _cmp_long,
                x="Method", y="Per-family MSE",
                color="Method", color_discrete_map=_M_COLORS,
                facet_col="Profile",
                text="Per-family MSE",
                title=f"Per-family MSE — all 7 methods + Ensemble, k={val_k}, m={_VAL_M} (n=50 for M2)",
                height=450,
            )
            fig_cmp_all.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_cmp_all.update_layout(showlegend=True)
            st.plotly_chart(fig_cmp_all, use_container_width=True)

            # Winner summary
            _method_cols = [
                "M1 Evidence", "M2 Moment", "M3 SNP", "M4 Consensus",
                "M5 π₃ TFBS", "M6 π₂ Seq.Hom.", "M7 π₄ SNP Sites", "Ensemble",
            ]
            _avg_mse = {
                m: float(cmp_df[m].mean())
                for m in _method_cols
                if m in cmp_df.columns and not cmp_df[m].isna().all()
            }
            if _avg_mse:
                _avg_sorted = sorted(_avg_mse.items(), key=lambda x: x[1])
                _best = _avg_sorted[0]
                st.success(
                    f"**{_best[0]}** achieves the lowest average MSE "
                    f"({_best[1]:.6f}) across profiles for this gene set and k = {val_k}.",
                    icon="🏆",
                )
                _rank_df = pd.DataFrame(
                    [{"Method": m, "Avg MSE across profiles": round(v, 6)} for m, v in _avg_sorted]
                )
                st.dataframe(_rank_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Average MSE is computed analytically over Linear and Quadratic profiles. "
                    "Results for Methods 1, 4, 5, 6, 7, and Ensemble depend on which TFs are selected above — "
                    "try different gene combinations to see how rankings change. "
                    "Y1000+ methods (M5, M6, M7) are omitted from ranking if data is not yet generated. "
                    "M2 uses n = 50 for this comparison."
                )

                # ── Per-method accuracy explanation ───────────────────────────────────
                st.subheader("Why does each method have this level of accuracy?")

                # Build live per-profile strings for each method
                def _profile_mse_str(col):
                    parts = []
                    for r in cmp_rows:
                        if not np.isnan(r.get(col, float("nan"))):
                            parts.append(f"{r['Profile']} MSE = {r[col]:.4f}")
                    return "; ".join(parts) if parts else "data not available"

                _lin_mse  = {col: next((r[col] for r in cmp_rows if r["Profile"]=="Linear"), float("nan"))
                             for col in _method_cols if col in _avg_mse}
                _quad_mse = {col: next((r[col] for r in cmp_rows if r["Profile"]=="Quadratic"), float("nan"))
                             for col in _method_cols if col in _avg_mse}

                # M1 explanation
                if "M1 Evidence" in _avg_mse:
                    _m1_lin  = _lin_mse.get("M1 Evidence", float("nan"))
                    _m1_quad = _quad_mse.get("M1 Evidence", float("nan"))
                    _m1_rank = next(i+1 for i,(m,_) in enumerate(_avg_sorted) if m=="M1 Evidence")
                    st.markdown(f"""
**M1 — Evidence-based** (ranked #{_m1_rank}, avg MSE = {_avg_mse['M1 Evidence']:.6f})

Method 1 maps each TF's SGD experimental evidence codes to a π prior using a fixed quality-weight
table (IDA → 0.90, IEA → 0.30, …). This imposes a hard structural range of roughly **0.31 – 0.77**
on every estimate — no combination of TF selections can push a family's πᵢ outside that window.

- **Linear profile** (MSE = {_m1_lin:.4f}): high error because the Linear profile contains values
  at both extremes (e.g. 0.1 and 0.9 for k=3), which Method 1 physically cannot reach. All families
  are pulled toward the middle, simultaneously underestimating the high-π families and overestimating
  the low-π ones.
- **Quadratic profile** (MSE = {_m1_quad:.4f}): lower error because the Quadratic values cluster
  closer to the centre of Method 1's reachable range. The residual error comes from shape mismatch,
  not a range violation.
- **Bottom line:** the accuracy ceiling for Method 1 is set by the evidence-code weight table, not
  by the amount of data or the choice of gene. It is best treated as a conservative prior rather
  than a precise point estimate.
""")

                # M2 explanation
                if "M2 Moment" in _avg_mse:
                    _m2_lin  = _lin_mse.get("M2 Moment", float("nan"))
                    _m2_quad = _quad_mse.get("M2 Moment", float("nan"))
                    _m2_rank = next(i+1 for i,(m,_) in enumerate(_avg_sorted) if m=="M2 Moment")
                    st.markdown(f"""
**M2 — Moment Estimation / Theorem 4** (ranked #{_m2_rank}, avg MSE = {_avg_mse['M2 Moment']:.6f})

Method 2 inverts Theorem 4 (`E[|M(n)|] = Γ(π̂+n)Γ(m)/[Γ(π̂+m)Γ(n)]`) via Brent root-finding to
recover the total π̂ = Σπᵢ. The inversion is mathematically exact — the squared error on π̂ itself
is essentially zero. The **only** source of per-family error is the allocation step: once π̂ is
known, Method 2 distributes it uniformly (π̂/k to every family), because Theorem 4 gives no
information about how inheritance is distributed *within* the motif.

- **Linear profile** (MSE = {_m2_lin:.4f}): high per-family error because the Linear profile is
  maximally heterogeneous — one family has low π and another has high π. A flat allocation of π̂/k
  misses every family simultaneously; the total is right but the shape is wrong.
- **Quadratic profile** (MSE = {_m2_quad:.4f}): lower error because the Quadratic profile is
  symmetric and closer to uniform, so π̂/k is a better approximation of each πᵢ.
- **Bottom line:** Method 2's per-family MSE equals the variance of the true profile around its
  mean (π̂/k). It is the right tool for aggregate significance tests (the Motif Significance tab)
  where only π̂ matters, but cannot distinguish which specific families inherit more or less.
""")

                # M3 explanation
                if "M3 SNP" in _avg_mse:
                    _m3_lin  = _lin_mse.get("M3 SNP", float("nan"))
                    _m3_quad = _quad_mse.get("M3 SNP", float("nan"))
                    _m3_rank = next(i+1 for i,(m,_) in enumerate(_avg_sorted) if m=="M3 SNP")
                    st.markdown(f"""
**M3 — SNP Divergence** (ranked #{_m3_rank}, avg MSE = {_avg_mse['M3 SNP']:.6f})

Method 3 is calibrated on the 11 sequenced strains of gene YFL039C: for each strain,
π = 1 − pct_alt/100, and the cross-strain mean (μ ≈ {_mu_m3:.4f}) is applied to **every** family
in the motif. Like Method 2's uniform allocation, this produces a single constant across all k
families — the MSE for any profile equals the variance of the true πᵢ values around μ.

- **Linear profile** (MSE = {_m3_lin:.4f}): the Linear profile's extreme values (e.g. 0.1 and 0.9)
  are far from μ ≈ {_mu_m3:.2f}, so each family contributes large squared deviation.
- **Quadratic profile** (MSE = {_m3_quad:.4f}): the Quadratic values cluster nearer to μ, yielding
  lower squared deviations on average.
- **Bottom line:** Method 3's irreducible error comes from the fact that a single-gene SNP dataset
  can only produce a scalar estimate. Its accuracy improves as the true profile concentrates near
  μ ≈ {_mu_m3:.2f}, and worsens as the profile spreads away from that anchor. It is most useful
  when no gene-specific evidence is available and a biologically grounded central estimate is needed.
""")

                # M4 explanation
                if "M4 Consensus" in _avg_mse:
                    _m4_lin  = _lin_mse.get("M4 Consensus", float("nan"))
                    _m4_quad = _quad_mse.get("M4 Consensus", float("nan"))
                    _m4_rank = next(i+1 for i,(m,_) in enumerate(_avg_sorted) if m=="M4 Consensus")
                    st.markdown(f"""
**M4 — Consensus-adjusted** (ranked #{_m4_rank}, avg MSE = {_avg_mse['M4 Consensus']:.6f})

Method 4 builds on Method 1 by multiplying each TF's evidence-based π by a YEASTRACT
binding-flexibility factor (range 1.02 – 1.12). TFs whose IUPAC consensus sequences contain
many ambiguous positions are treated as more promiscuous binders whose regulatory links can
survive more sequence drift after duplication, so their π is nudged upward.

- **Why it closely tracks M1:** the factor range is narrow (1.02 – 1.12), so Method 4 sits
  just above Method 1 in practice. It inherits the same structural floor (~0.37 adjusted) and
  ceiling (~0.82 adjusted), meaning it still cannot represent very low or very high inheritance.
- **Linear profile** (MSE = {_m4_lin:.4f}): essentially the same problem as M1 — the adjusted
  range still doesn't reach 0.1 or 0.9, so extreme families are mis-estimated.
- **Quadratic profile** (MSE = {_m4_quad:.4f}): similar to M1 but with a slight upward shift.
  Whether this helps or hurts depends on where the Quadratic values sit relative to the adjusted range.
- **Bottom line:** Method 4 adds biological nuance (promiscuous binders retain links more easily)
  without fundamentally changing the range problem. It is most useful when YEASTRACT data is
  available and you want to account for binding-specificity differences between TFs in the same family.
""")

                # M5 explanation
                if "M5 π₃ TFBS" in _avg_mse:
                    _m5_lin  = _lin_mse.get("M5 π₃ TFBS", float("nan"))
                    _m5_quad = _quad_mse.get("M5 π₃ TFBS", float("nan"))
                    _m5_rank = next(i+1 for i,(m,_) in enumerate(_avg_sorted) if m=="M5 π₃ TFBS")
                    st.markdown(f"""
**M5 — Y1000+ π₃ TFBS Conservation** (ranked #{_m5_rank}, avg MSE = {_avg_mse['M5 π₃ TFBS']:.6f})

Method 5 gives each family a per-TF estimate from real cross-species data: the mean retention
fraction of a significant PWM hit upstream of the orthologous gene across the Y1000+ 48-species
panel. It is the most direct measurement of binding-site inheritance available.

- **No structural floor or ceiling:** unlike Methods 1 and 4, Method 5 can in principle reach
  any π value from 0 to 1, limited only by the conservation patterns in the Y1000+ dataset.
- **Per-family differentiation:** a TF whose binding sites are retained in most yeast species
  gets a high πᵢ; a lineage-specific TF gets a low one.
- **Linear profile** (MSE = {_m5_lin:.4f}): error depends on how well the selected TFs'
  real conservation spread matches the profile's spread.
- **Quadratic profile** (MSE = {_m5_quad:.4f}): similarly depends on the match between TF
  conservation patterns and the symmetric Quadratic shape.
- **Bottom line:** Method 5's accuracy is constrained by the *choice of TFs*, not by model
  structure. It is the most trustworthy method for TFs with broad JASPAR PWM coverage in Y1000+.
""")

                # M6 explanation
                if "M6 π₂ Seq.Hom." in _avg_mse:
                    _m6_lin  = _lin_mse.get("M6 π₂ Seq.Hom.", float("nan"))
                    _m6_quad = _quad_mse.get("M6 π₂ Seq.Hom.", float("nan"))
                    _m6_rank = next(i+1 for i,(m,_) in enumerate(_avg_sorted) if m=="M6 π₂ Seq.Hom.")
                    st.markdown(f"""
**M6 — Y1000+ π₂ Sequence Homology** (ranked #{_m6_rank}, avg MSE = {_avg_mse['M6 π₂ Seq.Hom.']:.6f})

Method 6 estimates each TF family's πᵢ from mean pairwise protein sequence identity with all
other TFs in the Y1000+ dataset (π₂ = pct_identity / 100). A TF highly similar to its paralogs
is expected to retain regulatory links; a diverged TF is not.

- **Protein proxy, not binding proxy:** π₂ does not directly measure binding-site retention. A
  TF can have high sequence identity but bind different sites after duplication. Method 5 is more
  direct for TFs with JASPAR coverage.
- **Broadest coverage:** available for every TF with a Y1000+ FASTA entry — no JASPAR PWM needed.
  For TFs without PWM data, π₂ is often the only cross-species estimate available.
- **Linear profile** (MSE = {_m6_lin:.4f}): depends on how spread the selected TFs' sequence
  identities are relative to the Linear profile's extreme values.
- **Quadratic profile** (MSE = {_m6_quad:.4f}): TFs with intermediate mean identity tend to
  land near the Quadratic profile's centre, so Method 6 often performs well here.
- **Bottom line:** use Method 6 as a cross-species baseline when JASPAR data is unavailable,
  or as a structural complement to Methods 5 and 7 for TFs where all three are available.
""")

                # M7 explanation
                if "M7 π₄ SNP Sites" in _avg_mse:
                    _m7_lin  = _lin_mse.get("M7 π₄ SNP Sites", float("nan"))
                    _m7_quad = _quad_mse.get("M7 π₄ SNP Sites", float("nan"))
                    _m7_rank = next(i+1 for i,(m,_) in enumerate(_avg_sorted) if m=="M7 π₄ SNP Sites")
                    st.markdown(f"""
**M7 — Y1000+ π₄ Binding-site SNPs** (ranked #{_m7_rank}, avg MSE = {_avg_mse['M7 π₄ SNP Sites']:.6f})

Method 7 estimates inheritance from the IC-weighted mutation rate at the exact binding-site
positions of each TF across Y1000+ genomes: π₄ = 1 − Σ IC_weight[pos] × polymorphism_rate[pos].
Positions with high information content (most critical to binding specificity) are up-weighted.

- **Mechanistic precision:** unlike π₂ (whole-protein homology) or π₃ (site presence/absence),
  π₄ captures *which positions within the binding site* are under selection.
- **PWM required:** only available for TFs with a JASPAR 2024 PWM entry. TFs without a PWM
  cannot be scored and fall back to the dataset-wide mean.
- **Complements M5:** Method 5 asks "is there any binding site here?" while Method 7 asks
  "how conserved are the specific positions?" A site can be present (high π₃) but diverging
  at critical positions (low π₄).
- **Linear profile** (MSE = {_m7_lin:.4f}): depends on how spread the selected TFs' binding-site
  conservation patterns are relative to the Linear profile's extreme values.
- **Quadratic profile** (MSE = {_m7_quad:.4f}): TFs with intermediate π₄ tend to land near
  the Quadratic profile's centre.
- **Bottom line:** Method 7 is best when you need position-level resolution to detect subtle
  binding-site erosion not visible in presence/absence data. Use alongside Method 5.
""")

                # Ensemble explanation
                if "Ensemble" in _avg_mse:
                    _ens_lin  = _lin_mse.get("Ensemble", float("nan"))
                    _ens_quad = _quad_mse.get("Ensemble", float("nan"))
                    _ens_rank = next(i+1 for i,(m,_) in enumerate(_avg_sorted) if m=="Ensemble")
                    st.markdown(f"""
**Ensemble — Multi-signal per-family** (ranked #{_ens_rank}, avg MSE = {_avg_mse['Ensemble']:.6f})

The ensemble averages Methods 1, 3, 4, 5, 6, and 7 for each family independently. Method 2 is
excluded because Theorem 4 only recovers π̂ = Σπᵢ and cannot distinguish how inheritance is
distributed across families — including it would dilute the per-family signal.

- **When all Y1000+ data is available:** the ensemble combines M5 (TFBS conservation), M6
  (sequence homology), and M7 (binding-site SNPs) with the three SGD signals. These cross-species
  signals provide per-family differentiation that M1–4 alone cannot reach.
- **When only SGD data is available:** the ensemble falls back to M1, M3, M4. M3 returns a
  near-constant across families, so it adds little differentiation.
- **Linear profile** (MSE = {_ens_lin:.4f}): accuracy depends heavily on Y1000+ availability.
  With cross-species signals the ensemble can reach extreme π values (near 0 or 1).
- **Quadratic profile** (MSE = {_ens_quad:.4f}): the Quadratic profile clusters near the centre
  of the reachable range, so SGD signals alone already contribute meaningfully.
- **Bottom line:** use the ensemble as your default per-family πᵢ estimate. Its per-signal
  breakdown (visible in the Ensemble tab) is more informative than the final number alone —
  conflicting signals flag biologically interesting TFs.
""")
