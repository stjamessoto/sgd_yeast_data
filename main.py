"""
main.py — CLI entry point for the Scruse et al. inheritance probability model.

Usage examples:
  python main.py summary
  python main.py tfs --min-evidence 0.5 --dna-binding
  python main.py families --min-size 3
  python main.py estimate --genes GO:0006355 GO:0045944 --method evidence
  python main.py significance --k 3 --method evidence --observed 150
  python main.py binding --tf ABF1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.data_loader import dataset_summary, EVIDENCE_QUALITY
from model.tf_network import (
    get_transcription_factors,
    build_tf_target_map,
    describe_binding_sites,
    network_statistics,
)
from model.gene_families import (
    build_tf_families,
    estimate_model_parameters,
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
)


def _section(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def cmd_summary(args):
    _section("Dataset Summary")
    s = dataset_summary()
    for k, v in s.items():
        print(f"  {k:<30} {v}")


def cmd_tfs(args):
    _section("Transcription Factors")
    tfs = get_transcription_factors(
        require_dna_binding=args.dna_binding,
        min_evidence_score=args.min_evidence,
    )
    print(f"  Found {len(tfs)} TFs\n")
    cols = ["gene_name", "has_dna_binding", "is_activator", "is_repressor",
            "evidence_score", "pi_prior"]
    print(tfs[cols].head(args.limit).to_string(index=False))


def cmd_families(args):
    _section("Gene Families (TF clusters by GO process)")
    fam = build_tf_families(min_family_size=args.min_size)
    if fam.empty:
        print("  No families found. Try --min-size 1")
        return
    params = estimate_model_parameters(fam, "family_size")
    print(f"  m (families) = {params['m']}")
    print(f"  n (total TFs) = {params['n']}")
    print(f"  d (duplication events) = {params['d']}")
    print(f"  Mean family size = {params['mean_family_size']}\n")
    print(fam[["go_id", "family_size", "mean_evidence_score"]].head(args.limit).to_string(index=False))


def cmd_binding(args):
    _section(f"Binding Site Info: {args.tf}")
    info = describe_binding_sites(args.tf)
    if "error" in info:
        print(f"  Error: {info['error']}")
        return
    for k, v in info.items():
        if k not in ("binding_go_terms",):
            print(f"  {k:<30} {v}")
    print("\n  Binding GO terms:")
    for goid, label in info.get("binding_go_terms", {}).items():
        print(f"    {goid} — {label}")


def cmd_estimate(args):
    _section("Inheritance Probability Estimation")
    gene_names = args.genes
    fam = build_tf_families(min_family_size=1)
    if fam.empty:
        print("  No families available.")
        return
    params = estimate_model_parameters(fam, "family_size")
    m, n = params["m"], params["n"]

    if args.method == "evidence":
        result = estimate_pi_from_evidence(gene_names)
    elif args.method == "mle":
        obs = args.observed or expected_full(len(gene_names), m, n)
        result = estimate_pi_from_mle(obs, m, n, gene_names)
    elif args.method == "snp":
        result = estimate_pi_from_snp(gene_names)
    else:
        print("  Unknown method. Choose: evidence | mle | snp")
        return

    print(f"  Method:  {result['method']}")
    print(f"  π̂ (sum): {result['pi_hat']}")
    print(f"  π⃗:       {result['pi_vec']}")
    print(f"\n  {result.get('description', '')}")


def cmd_significance(args):
    _section("Subnetwork Motif Significance Test")
    fam = build_tf_families(min_family_size=1)
    if fam.empty:
        print("  No families available.")
        return
    params = estimate_model_parameters(fam, "family_size")
    m, n = params["m"], params["n"]
    k = args.k

    selected = select_motif_families(fam, k, strategy=args.strategy)
    gene_names = [f["go_id"] for f in selected]

    print(f"  Motif size k = {k}")
    print(f"  m = {m}, n = {n}")
    print(f"  Selected families: {gene_names}")

    pi_res = estimate_pi_from_evidence(gene_names)
    pi_vec = pi_res["pi_vec"]
    pi_hat = pi_res["pi_hat"]

    obs = float(args.observed) if args.observed else count_observed_motif_instances(selected)

    result = full_significance_analysis(pi_vec, m, n, obs, k)

    print(f"\n  Observed count:              {result['observed_count']}")
    print(f"  Expected (Full Dup):         {result['expected_full']}")
    print(f"  Expected (Partial Dup, pi_hat={pi_hat}): {result['expected_partial']}")
    print(f"  Variance (Full Dup):         {result['variance_full']:.4f}")
    print(f"  Variance (Binary Inherit.):  {result['variance_binary']:.4f}")
    print(f"  Z-score  (Full Dup):         {result['z_full']}")
    print(f"  Z-score  (Partial Dup):      {result['z_partial']}")
    print(f"  p-value  (Full Dup):         {result['p_full']}")
    print(f"  p-value  (Partial Dup):      {result['p_partial']}")
    print(f"  Significance (Full Dup):     {result['sig_full']}")
    print(f"  Significance (Partial Dup):  {result['sig_partial']}")
    print(f"\n  Interpretation:\n  {result['interpretation']}")


def cmd_math_demo(args):
    _section("Mathematical Functions Demo (Scruse et al.)")
    print("\n  Lemma 4 -- f(p, s): single-gene motif expected count")
    print(f"  f(0.5, 5)  = {f_func(0.5, 5):.4f}")
    print(f"  f(0.9, 10) = {f_func(0.9, 10):.4f}")
    print(f"  f(1.0, 5)  = {f_func(1.0, 5):.4f}  (should equal 5 under Full Dup)")

    print("\n  Theorem 1 -- Full Duplication expected count E[|M(n)|; k,m,n]")
    for k, m, n in [(2, 50, 732), (3, 50, 732), (1, 50, 732)]:
        print(f"  E[|M(n)|; k={k}, m={m}, n={n}] = {expected_full(k, m, n):.4f}")

    print("\n  Theorem 4 -- Partial Duplication E[|M(n)|; m,n,pi_hat]")
    for pi_hat in [0.5, 1.0, 1.5, 2.0]:
        print(f"  E[|M(n)|; m=50, n=732, pi_hat={pi_hat}] = {expected_partial(pi_hat, 50, 732):.4f}")

    print("\n  Corollary 2 -- Full Duplication variance")
    print(f"  Var(|M(n)|; k=2, m=50, n=732) = {variance_full(2, 50, 732):.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Scruse et al. (2024) Inheritance Probability Model — CLI"
    )
    sub = parser.add_subparsers(dest="command")

    # summary
    sub.add_parser("summary", help="Print dataset summary statistics")

    # tfs
    p_tfs = sub.add_parser("tfs", help="List transcription factors")
    p_tfs.add_argument("--dna-binding", action="store_true")
    p_tfs.add_argument("--min-evidence", type=float, default=0.0)
    p_tfs.add_argument("--limit", type=int, default=20)

    # families
    p_fam = sub.add_parser("families", help="Show gene families")
    p_fam.add_argument("--min-size", type=int, default=2)
    p_fam.add_argument("--limit", type=int, default=20)

    # binding
    p_bind = sub.add_parser("binding", help="Binding site info for a TF")
    p_bind.add_argument("--tf", required=True, help="TF gene name, e.g. ABF1")

    # estimate
    p_est = sub.add_parser("estimate", help="Estimate π⃗ for gene families")
    p_est.add_argument("--genes", nargs="+", required=True,
                       help="Space-separated GO IDs or gene names for families")
    p_est.add_argument("--method", choices=["evidence", "mle", "snp"], default="evidence")
    p_est.add_argument("--observed", type=float, default=None)

    # significance
    p_sig = sub.add_parser("significance", help="Motif significance test")
    p_sig.add_argument("--k", type=int, default=2)
    p_sig.add_argument("--strategy", choices=["largest", "highest_ev", "balanced", "random"],
                       default="largest")
    p_sig.add_argument("--method", choices=["evidence", "mle", "snp"], default="evidence")
    p_sig.add_argument("--observed", type=float, default=None)

    # math-demo
    sub.add_parser("math-demo", help="Demo the core mathematical functions")

    args = parser.parse_args()

    dispatch = {
        "summary": cmd_summary,
        "tfs": cmd_tfs,
        "families": cmd_families,
        "binding": cmd_binding,
        "estimate": cmd_estimate,
        "significance": cmd_significance,
        "math-demo": cmd_math_demo,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
