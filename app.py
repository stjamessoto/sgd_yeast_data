"""
app.py — Streamlit frontend for the Scruse et al. (2024) inheritance probability model.

Tabs:
  0. Introduction   — plain-language guide: what the model does and how to navigate the app
  1. Overview       — dataset summary & paper overview
  2. TF Explorer    — browse TFs, binding sites, consensus sequences, regulatory targets
  3. Gene Families  — family size distribution and Pólya urn parameters
  4. π Estimator    — estimate inheritance probability vector four ways
  5. Motif Significance — test whether a k-motif is over/under-represented

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
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
    estimate_pi_all_methods,
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
)

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
.theorem-box pre {
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
             padding:10px 14px;border-radius:4px;margin:8px 0;}
.warn-box   {background:#fff3e0;border-left:4px solid #ff7f0e;
             padding:10px 14px;border-radius:4px;margin:8px 0;}

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
def _load_tf_families(min_size):
    return build_tf_families(min_family_size=min_size)

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
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://www.yeastgenome.org/images/SGD_logo.gif", width=160)
    st.title("GRN Inheritance Model")
    st.caption("Scruse, Arnold & Robinson (2024)")
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
    min_family_size = st.slider("Min gene-family size", 1, 20, 2)

    st.divider()
    summary = _dataset_summary()
    st.metric("TFs loaded", summary["n_tfs"])
    st.metric("GO annotations", f"{summary['n_go_annotations']:,}")
    st.metric("Genome size", f"{summary['genome_size_mb']} Mb")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📘 Introduction",
    "📊 Overview",
    "🔬 TF Explorer",
    "👨‍👩‍👧 Gene Families",
    "🎲 π Estimator",
    "🧪 Motif Significance",
    "📖 Glossary & References",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 0: Introduction
# ══════════════════════════════════════════════════════════════════════

with tab0:
    st.header("Welcome — What This App Does")
    st.markdown(
        "_Scruse, Arnold & Robinson (2024) · arXiv:2405.03148v1 · University of Georgia_"
    )
    st.markdown("""
    This app implements a mathematical model for studying **how gene regulatory networks
    (GRNs) evolve after gene duplication** in brewer's yeast (*Saccharomyces cerevisiae*).
    It draws on **179 transcription factors** from three curated sources:
    [JASPAR 2024](https://jaspar.elixir.no/) (177 TFs with experimentally validated
    position frequency matrices), [YEASTRACT](https://www.yeastract.com/) (127 curated TFs
    with consensus binding sequences), and gene annotations from the
    [Saccharomyces Genome Database (SGD)](https://www.yeastgenome.org/).
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
    st.caption("Work through the tabs left to right for the full analysis pipeline.")

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        with st.container(border=True):
            st.markdown("**📊 Overview**")
            st.markdown(
                "Dataset summary and key theorem equations. Live metric counts "
                "for JASPAR TFs, GO annotations, and JASPAR binding site statistics."
            )
    with row1_c2:
        with st.container(border=True):
            st.markdown("**🔬 TF Explorer**")
            st.markdown(
                "Browse all JASPAR-validated transcription factors. Select any TF "
                "to see its JASPAR PFM (consensus, IC, motif width), YEASTRACT "
                "sequences, and regulatory targets."
            )

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        with st.container(border=True):
            st.markdown("**👨‍👩‍👧 Gene Families**")
            st.markdown(
                "Genes grouped into families by shared GO Biological Process terms. "
                "Shows family size distributions and Pólya urn parameters m and n."
            )
    with row2_c2:
        with st.container(border=True):
            st.markdown("**🎲 π Estimator**")
            st.markdown(
                "Select k families to form a motif and estimate the inheritance "
                "probability vector $\\vec{\\pi}$ using four methods: evidence-based, MLE, "
                "SNP divergence, and YEASTRACT consensus-adjusted."
            )

    with st.container(border=True):
        st.markdown("**🧪 Motif Significance**")
        st.markdown(
            "Run a full significance test: compare the observed motif count against "
            "the Full and Partial Duplication null models. Outputs Z-scores, p-values, "
            "and a plain-language interpretation of the result."
        )

    st.divider()
    st.subheader("📚 Data Sources")
    _js = jaspar_summary()
    src_c1, src_c2, src_c3 = st.columns(3)
    with src_c1:
        st.markdown(f"""
        **JASPAR 2024** *(jaspar.elixir.no)*
        - **{_js['n_jaspar_tfs']} TFs** with experimentally validated PFMs
        - {_js['n_chip_based']} ChIP-based · {_js['n_pbm_based']} PBM-based
        - Mean motif width: {_js['mean_motif_width']} bp · mean IC: {_js['mean_ic_bits']} bits
        - Primary source for TFBS binding sequences and π adjustment
        """)
    with src_c2:
        st.markdown("""
        **YEASTRACT** *(yeastract.com)*
        - 127 curated *S. cerevisiae* transcription factors
        - IUPAC consensus sequences for 115 TFs also in JASPAR
        - Supplementary binding sequence reference
        """)
    with src_c3:
        st.markdown("""
        **SGD** *(yeastgenome.org)*
        - GO annotations for 6,446 genes (~120,000 records)
        - TF evidence codes (IDA, IMP, IEA, …) for π priors
        - Chromosome lengths, gene IDs, synonyms
        """)

    st.info(
        "💡 **Start here**, then move through the tabs left to right — "
        "Overview → TF Explorer → Gene Families → π Estimator → Motif Significance.",
        icon="💡",
    )


# ══════════════════════════════════════════════════════════════════════
# TAB 1: Overview
# ══════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Counting Subnetworks Under Gene Duplication in GRNs")
    st.markdown(
        "_Scruse, Arnold & Robinson (2024) · arXiv:2405.03148v1 · University of Georgia_"
    )

    st.markdown("""
    This application implements the mathematical framework from the paper to:

    - **Identify** transcription factors (TFs) and their regulatory targets in the
      *S. cerevisiae* GRN from SGD data.
    - **Estimate** the per-family inheritance probability vector **$\\vec{\\pi}$ = (π₁, …, πₖ)**
      using three data-driven methods.
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
        st.markdown("""
<div class="theorem-box">
<b>Theorem 1 — Full Duplication ($\\vec{\\pi}$ = 1)</b><br>
<pre>E[|M(n)|; k, m, n] = Γ(n+k)Γ(m) / [Γ(n)Γ(m+k)]</pre>
Growth rate: Θ(nᵏ) — degree equals motif size k.
</div>

<div class="theorem-box">
<b>Theorem 4 — Partial Duplication</b><br>
<pre>E[|M(n)|; m,n,π⃗,k] = Γ(π̂+n)Γ(m) / [Γ(π̂+m)Γ(n)]</pre>
Depends on $\\vec{\\pi}$ only through π̂ = Σπᵢ.<br>
Growth rate: Θ(n^π̂) — exponent <i>equals</i> total inheritance probability.
</div>
""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
<div class="theorem-box">
<b>Lemma 4 — Single-gene motif</b><br>
<pre>f(p, s) = Γ(p+s) / [Γ(s) Γ(p+1)]</pre>
Expected instances when family size is s and inheritance probability is p.
</div>

<div class="theorem-box">
<b>Theorem 6 — Binary Inheritance (max 2nd moment)</b><br>
<pre>E[|M(n)|²] = Γ(m)/Γ(n) × Σ_{A⊆K} (-1)^|A| × 2^k ×
              Γ(n+π̂+Σ_{i∉A}πᵢ) / [2^|A| × Γ(m+π̂+Σ_{i∉A}πᵢ)]</pre>
Binary Inheritance maximises E[|M(n)|²] over all Partial Duplication refinements (Theorem 5).
</div>
""", unsafe_allow_html=True)

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

with tab2:
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

with tab3:
    st.header("👨‍👩‍👧 Gene Family Analysis")
    st.markdown("""
Each GO Biological Process term defines a **gene family** — the set of TFs that
share that process.  This operationalises the Scruse et al. framework:

- **m** = number of gene families
- **n** = total TFs across families (Σcᵢ)
- **d = n − m** = estimated duplication events (Proposition 1)
""")

    with st.spinner("Building TF families…"):
        fam_df = _load_tf_families(min_family_size)

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
            title="Gene Family Size Distribution (TF families by GO process)",
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

        st.subheader("Top Gene Families")
        st.dataframe(
            fam_df[["go_id", "family_size", "mean_evidence_score",
                    "n_activators", "n_repressors"]].head(30),
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

with tab4:
    st.header("🎲 Inheritance Probability Estimator")
    st.markdown("""
Estimate **$\\vec{\\pi}$ = (π₁, …, πₖ)** — the per-family probability that regulatory
links are inherited through gene duplication.
""")

    st.markdown("Three estimation methods are available (Scruse et al. Sections 4, 6, 9.2):")

    with st.spinner("Loading families…"):
        fam_df4 = _load_tf_families(min_family_size)

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
    st.subheader("Step 2 — Estimate $\\vec{\\pi}$")

    # Method selection
    method4 = st.radio(
        "Estimation method",
        [
            "Method 1: Evidence-based (Lemma 4 / evidence codes)",
            "Method 2: MLE Theorem 4 inversion",
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

    elif "Method 2" in method4:
        obs_input = st.number_input(
            "Observed motif count |M(n)|",
            min_value=1.0, max_value=float(obs_count4),
            value=float(min(obs_count4, max(1, obs_count4 // 2))),
            step=1.0,
        )
        result4 = estimate_pi_from_mle(obs_input, m4, n4, gene_names4)
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"**Method 2 — MLE (Theorem 4 inversion)**")
        st.markdown(result4.get("note", ""))
        st.markdown(f"pi_hat = {result4['pi_hat']:.4f}   |   {result4['description'][:100]}...")
        st.markdown('</div>', unsafe_allow_html=True)
        pi_vec4 = result4["pi_vec"]

    elif "Method 3" in method4:
        result4 = estimate_pi_from_snp(gene_names4)
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"**Method 3 — SNP Divergence Proxy**\n\n{result4['description']}")
        st.markdown(f"pi_hat = {result4['pi_hat']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        pi_vec4 = result4["pi_vec"]

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

        fig_comp = go.Figure()
        x = comp_df["gene_family"]
        for col, name, colour in [
            ("pi_evidence", "Evidence-based", "#1f77b4"),
            ("pi_mle", "MLE (Theorem 4)", "#ff7f0e"),
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
    if "All three" not in method4:
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
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=sens_df["pi_hat"], y=sens_df["expected"],
        name="E[|M(n)|]", line=dict(color="#1f77b4", width=2),
    ))
    fig_sens.add_trace(go.Scatter(
        x=pd.concat([sens_df["pi_hat"], sens_df["pi_hat"].iloc[::-1]]),
        y=pd.concat([sens_df["upper_bound"], sens_df["lower_bound"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(31,119,180,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Corollary 9 bounds",
    ))
    if "All three" not in method4:
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


# ══════════════════════════════════════════════════════════════════════
# TAB 5: Motif Significance
# ══════════════════════════════════════════════════════════════════════

with tab5:
    st.header("🧪 Subnetwork Motif Significance Testing")

    inf_tab, pred_tab = st.tabs(["📊 Inferential Test", "🔮 Predictive Forecast"])

    # ── Inferential Test ─────────────────────────────────────────────
    with inf_tab:
        st.markdown("""
Tests whether a subnetwork motif M of size k is **significantly over- or under-represented**
in the GRN relative to the gene duplication null model (Scruse et al. Sections 4–6).

**Z-score** = (observed − expected) / std dev  ·  **p-value** via normal approximation.
""")

        with st.spinner("Loading families…"):
            fam_df5 = _load_tf_families(min_family_size)

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
                ["Evidence-based", "MLE (Theorem 4)", "SNP proxy", "Manual"],
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
        elif pi_method5 == "MLE (Theorem 4)":
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

        with st.spinner("Loading families…"):
            fam_df_p = _load_tf_families(min_family_size)

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
                ["Evidence-based", "MLE (Theorem 4)", "Manual"],
                key="pm_pred",
            )

        fams_p = select_motif_families(fam_df_p, k_p, strategy=strat_p)
        gene_names_p = [f["go_id"] for f in fams_p]
        obs_full_p = count_observed_motif_instances(fams_p)

        if pi_method_p == "Evidence-based":
            pi_res_p = estimate_pi_from_evidence(gene_names_p)
            pi_vec_p = pi_res_p["pi_vec"]
        elif pi_method_p == "MLE (Theorem 4)":
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
# TAB 6: Glossary & References
# ══════════════════════════════════════════════════════════════════════

with tab6:
    st.header("📖 Glossary & References")
    st.markdown(
        "Definitions of key terms used throughout the app, and the primary papers "
        "that underpin this model."
    )

    # ── References ────────────────────────────────────────────────────
    st.subheader("📚 Primary References")

    with st.container(border=True):
        st.markdown("#### 1. Scruse, Arnold & Robinson (2024) — *The Model Paper*")
        st.markdown("""
**Counting Subnetworks Under Gene Duplication in Genetic Regulatory Networks**
Ashley Scruse, Jonathan Arnold, Robert Robinson
*arXiv:2405.03148v1 [q-bio.MN] · University of Georgia · 6 May 2024*

This is the primary theoretical paper implemented in this app. It introduces the
gene duplication and inheritance model, defines **subnetwork motifs**, and derives
the exact moments (mean and variance) for their count under both Full and Partial
Duplication via combinatorial probability and generating functions.
        """)
        st.link_button("View on arXiv (arXiv:2405.03148)", "https://arxiv.org/abs/2405.03148")

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
  `E[|M(n)|; m, n, π⃗, k] = Γ(π̂+n)Γ(m) / [Γ(π̂+m)Γ(n)]`
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

In this app, π is estimated from SGD evidence codes, MLE on observed counts, SNP
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
        st.markdown(r"""
The general inheritance model in which **regulatory links are inherited stochastically**
at each duplication step, controlled by the vector $\vec{\pi}$ = (π₁, …, πₖ).

At each step, if a gene in family *i* is duplicated (1 ≤ i ≤ k), each existing motif
instance that includes that gene produces a new instance with probability πᵢ.

Key result (**Theorem 4**):

`E[|M(n)|; m, n, π⃗, k] = Γ(π̂+n)Γ(m) / [Γ(π̂+m)Γ(n)]`

where π̂ = π₁ + … + πₖ. The expected count depends on $\vec{\pi}$ only through the scalar π̂,
which determines the polynomial growth rate Θ(n^π̂).

Full Duplication ($\vec{\pi}$ = $\vec{1}$, π̂ = k) is a special case.
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
        st.markdown(r"""
A classical probability model for reinforcement processes. An urn contains balls of
different colours; at each step, one ball is drawn at random, and a new ball of the
same colour is added.

**Connection to gene duplication (Section 7.1 / Theorem 8):**

The gene duplication process is equivalent to a Pólya urn where:
- **Urn** = the genome
- **Balls** = genes
- **Colours** = gene families

**Theorem 8** gives the exact probability:

`P[X⃗ = t⃗ | s⃗] = C(n−m; t₁,…,tω) × (m−1)! / (n−1)! × Π(sⱼ+tⱼ−1)! / (sⱼ−1)!`

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
        "Glossary compiled from Scruse, Arnold & Robinson (2024) arXiv:2405.03148, "
        "Harbison et al. (2004) Nature 431:99–104, and Ren et al. (2000) Science 290:2306–2309."
    )
