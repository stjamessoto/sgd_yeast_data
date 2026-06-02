"""
scruse_math.py
Mathematical backbone from Scruse, Arnold & Robinson (2024).
arXiv:2405.03148v1 — Counting Subnetworks Under Gene Duplication in Genetic Regulatory Networks

All theorems, lemmas, and corollaries referenced by number match the paper.
Numerical stability is achieved throughout via log-gamma (gammaln).
"""

import numpy as np
from scipy.special import gammaln
from scipy.optimize import brentq
from scipy.stats import norm
from itertools import combinations
from math import comb, factorial
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Utility: Gamma function ratios (numerical stability via log-space)
# ---------------------------------------------------------------------------

def log_gamma_ratio(n: float, z: float) -> float:
    """log(Γ(n+z) / Γ(n))  — numerically stable via gammaln."""
    return gammaln(n + z) - gammaln(n)


def gamma_ratio(n: float, z: float) -> float:
    """Γ(n+z) / Γ(n)."""
    return np.exp(log_gamma_ratio(n, z))


# ---------------------------------------------------------------------------
# Lemma 4 — Single-gene motif expected count (Partial Duplication)
# f(p, s) = Γ(p+s) / [Γ(s) Γ(p+1)]
# ---------------------------------------------------------------------------

def f_func(p: float, s: int) -> float:
    """
    Lemma 4: Expected instances of a single-gene subnetwork motif.
    p = inheritance probability, s = family size.
    f(p, s) = Γ(p+s) / [Γ(s) Γ(p+1)]
    Boundary: f(p, 1) = 1 (only the original instance exists).
    """
    if s < 1 or p < 0:
        return float("nan")
    return np.exp(gammaln(p + s) - gammaln(s) - gammaln(p + 1))


# ---------------------------------------------------------------------------
# Corollary 12 — Single-gene second moment
# g(p, s) = 2Γ(s+2p)/[Γ(s)Γ(2p+1)] − Γ(p+s)/[Γ(s)Γ(p+1)]
# ---------------------------------------------------------------------------

def h_func(p: float, s: int) -> float:
    """h(p, s) = g(p, s) + f(p, s) = 2Γ(s+2p)/[Γ(s)Γ(2p+1)]."""
    if s < 1 or p < 0:
        return float("nan")
    return 2.0 * np.exp(gammaln(s + 2 * p) - gammaln(s) - gammaln(2 * p + 1))


def g_func(p: float, s: int) -> float:
    """
    Corollary 12: Expected ordered pairs of single-gene motif instances.
    g(p, s) = h(p,s) − f(p,s)
    """
    return h_func(p, s) - f_func(p, s)


# ---------------------------------------------------------------------------
# Theorem 1 — Full Duplication: Expected motif count
# E[|M(n)|; k, m, n] = Γ(n+k)Γ(m) / [Γ(n)Γ(m+k)]
# ---------------------------------------------------------------------------

def expected_full(k: int, m: int, n: int) -> float:
    """
    Theorem 1: Expected subnetwork motif count under Full Duplication (π⃗ = 1).
    k = motif size, m = initial gene count, n = current gene count (n ≥ m ≥ k ≥ 1).
    """
    if k < 1 or m < k or n < m:
        return float("nan")
    return np.exp(gammaln(n + k) + gammaln(m) - gammaln(n) - gammaln(m + k))


# ---------------------------------------------------------------------------
# Theorem 2 — Full Duplication: Second moment
# ---------------------------------------------------------------------------

def second_moment_full(k: int, m: int, n: int) -> float:
    """
    Theorem 2: Second moment E[|M(n)|²] under Full Duplication.
    = Γ(n+k)Γ(m)Γ(n−m+1)/Γ(n) × Σᵢ₌₀ᵏ C(k,i)·2ⁱ·[Γ(m+k+i)Γ(n−m−i+1)]⁻¹
    """
    if k < 1 or m < k or n < m:
        return float("nan")

    log_prefix = gammaln(n + k) + gammaln(m) + gammaln(n - m + 1) - gammaln(n)
    total = 0.0
    for i in range(k + 1):
        denom_arg = n - m - i + 1
        if denom_arg <= 0:
            continue
        log_term = (
            np.log(comb(k, i))
            + i * np.log(2)
            - gammaln(m + k + i)
            - gammaln(denom_arg)
        )
        total += np.exp(log_prefix + log_term)
    return total


# ---------------------------------------------------------------------------
# Corollary 2 — Full Duplication: Variance
# ---------------------------------------------------------------------------

def variance_full(k: int, m: int, n: int) -> float:
    """
    Corollary 2: Var(|M(n)|) = E[|M(n)|²] − (E[|M(n)|])²  under Full Duplication.
    Equals 0 when m = 1 or m = n.
    """
    e1 = expected_full(k, m, n)
    e2 = second_moment_full(k, m, n)
    return max(0.0, e2 - e1 ** 2)


# ---------------------------------------------------------------------------
# Theorem 4 — Partial Duplication: Expected count (unconditional)
# E[|M(n)|; m,n,π⃗,k] = Γ(π̂+n)Γ(m) / [Γ(π̂+m)Γ(n)]   where π̂ = Σπᵢ
# ---------------------------------------------------------------------------

def expected_partial(pi_hat: float, m: int, n: int) -> float:
    """
    Theorem 4: Expected motif count under Partial Duplication.
    Depends on π⃗ only through its sum π̂ = Σπᵢ (Theorem 4 key observation).
    Growth rate Θ(n^π̂) — exponent equals total inheritance probability.
    """
    if pi_hat < 0 or m < 1 or n < m:
        return float("nan")
    return np.exp(
        gammaln(pi_hat + n) + gammaln(m) - gammaln(pi_hat + m) - gammaln(n)
    )


# ---------------------------------------------------------------------------
# Corollary 9 — Partial Duplication: Bounds on expected count
# ---------------------------------------------------------------------------

def expected_bounds_partial(
    pi_hat: float, m: int, n: int, k: int
) -> Tuple[float, float]:
    """
    Corollary 9: Tight bounds on E[|M(n)|] for fixed m, n ≥ 2k².
    Lower: n^π̂·Γ(m)/Γ(π̂+m) · (1 − (k²+1/12)/n)
    Upper: n^π̂·Γ(m)/Γ(π̂+m) · (1 + 3k²/(2n))
    """
    main = np.exp(pi_hat * np.log(max(n, 1e-10)) + gammaln(m) - gammaln(pi_hat + m))
    lower = main * (1.0 - (k ** 2 + 1.0 / 12.0) / n)
    upper = main * (1.0 + 3.0 * k ** 2 / (2.0 * n))
    return lower, upper


# ---------------------------------------------------------------------------
# Theorem 3 — Partial Duplication: Conditional expected count given s⃗
# E[|M(s⃗)|] = Π f(πᵢ, sᵢ)
# ---------------------------------------------------------------------------

def expected_conditional(pi_vec: List[float], s_vec: List[int]) -> float:
    """
    Theorem 3: Conditional expected count given family size vector s⃗.
    Families contribute multiplicatively (independence conditioned on sizes).
    """
    result = 1.0
    for pi, s in zip(pi_vec, s_vec):
        result *= f_func(pi, s)
    return result


# ---------------------------------------------------------------------------
# Theorem 5 (Binary Inheritance maximises E[|M(n)|²])
# Theorem 6 — Binary Inheritance: Second moment
# E[|M(n)|²; m,n,π⃗,k] = Γ(m)/Γ(n) × Σ_{A⊆K} (−1)^|A| · 2^k · Γ(n+π̂+Σ_{i∉A}πᵢ)
#                                                             / [2^|A| · Γ(m+π̂+Σ_{i∉A}πᵢ)]
# ---------------------------------------------------------------------------

def second_moment_binary(pi_vec: List[float], m: int, n: int) -> float:
    """
    Theorem 6: Second moment under Binary Inheritance (maximum over all refinements
    of Partial Duplication, per Theorem 5).
    Inclusion-exclusion sum over all 2^k subsets A of K = {1,...,k}.
    """
    k = len(pi_vec)
    if k == 0 or m < 1 or n < m:
        return float("nan")
    pi_hat = sum(pi_vec)

    log_prefix = gammaln(m) - gammaln(n)
    total = 0.0
    K = list(range(k))

    for size in range(k + 1):
        for A in combinations(K, size):
            A_set = set(A)
            sum_not_A = sum(pi_vec[i] for i in K if i not in A_set)
            sign = (-1) ** len(A)
            log_term = (
                np.log(2) * k
                - np.log(2) * len(A)
                + gammaln(n + pi_hat + sum_not_A)
                - gammaln(m + pi_hat + sum_not_A)
            )
            total += sign * np.exp(log_prefix + log_term)

    return total


# ---------------------------------------------------------------------------
# Corollary 16 — Binary Inheritance: Variance
# ---------------------------------------------------------------------------

def variance_binary(pi_vec: List[float], m: int, n: int) -> float:
    """
    Corollary 16: Var(|M(n)|) under Binary Inheritance.
    = E[|M(n)|²] − (E[|M(n)|])²
    """
    pi_hat = sum(pi_vec)
    e1 = expected_partial(pi_hat, m, n)
    e2 = second_moment_binary(pi_vec, m, n)
    return max(0.0, e2 - e1 ** 2)


# ---------------------------------------------------------------------------
# Corollary 17 — Binary Inheritance: Bounds on second moment
# ---------------------------------------------------------------------------

def second_moment_bounds_binary(
    pi_vec: List[float], m: int, n: int
) -> Tuple[float, float]:
    """
    Corollary 17: Upper/lower bounds for E[|M(n)|²] under Binary Inheritance.
    Upper: Γ(m)·2^k·Γ(n+2π̂) / [Γ(n)·Γ(m+2π̂)]
    Lower: upper − Γ(m)/Γ(n) · Σⱼ 2^(k-1)·Γ(n+2π̂−πⱼ)/Γ(m+2π̂−πⱼ)
    """
    k = len(pi_vec)
    pi_hat = sum(pi_vec)
    log_mn = gammaln(m) - gammaln(n)

    upper = np.exp(
        log_mn + k * np.log(2) + gammaln(n + 2 * pi_hat) - gammaln(m + 2 * pi_hat)
    )
    correction = sum(
        np.exp(
            log_mn
            + (k - 1) * np.log(2)
            + gammaln(n + 2 * pi_hat - pj)
            - gammaln(m + 2 * pi_hat - pj)
        )
        for pj in pi_vec
    )
    lower = max(0.0, upper - correction)
    return lower, upper


# ---------------------------------------------------------------------------
# Theorem 7 — Full Duplication, multiple motifs: total expected count
# E[X | m,n] = Σᵢ Γ(n+kᵢ)Γ(m) / [Γ(n)Γ(m+kᵢ)]
# ---------------------------------------------------------------------------

def expected_total_full(k_list: List[int], m: int, n: int) -> float:
    """
    Theorem 7: Expected total motif instances across r motifs under Full Duplication.
    Follows from Theorem 1 and linearity of expectation.
    """
    return sum(expected_full(k, m, n) for k in k_list)


# ---------------------------------------------------------------------------
# Lemma 11 — Cross-motif expectation E[|Mᵢ(n)|·|Mⱼ(n)|]
# ---------------------------------------------------------------------------

def cross_motif_expected(
    ki: int, kj: int, overlap: int, m: int, n: int
) -> float:
    """
    Lemma 11: E[|Mᵢ(n)|·|Mⱼ(n)|] for two motifs with ℓᵢⱼ shared families.
    = Γ(n+kᵢ+kⱼ)Γ(m)Γ(n−m+1)/Γ(n)
      × Σᵤ₌₀^ℓᵢⱼ 2^u·C(ℓ,u)·[Γ(m+kᵢ+kⱼ+u)·Γ(n−m−u+1)]⁻¹
    """
    if ki < 1 or kj < 1 or m < max(ki, kj) or n < m:
        return float("nan")
    ell = overlap
    log_prefix = (
        gammaln(n + ki + kj) + gammaln(m) + gammaln(n - m + 1) - gammaln(n)
    )
    total = 0.0
    for u in range(ell + 1):
        denom_arg = n - m - u + 1
        if denom_arg <= 0:
            continue
        log_term = (
            u * np.log(2)
            + np.log(comb(ell, u))
            - gammaln(m + ki + kj + u)
            - gammaln(denom_arg)
        )
        total += np.exp(log_prefix + log_term)
    return total


# ---------------------------------------------------------------------------
# Theorem 8 — Polya urn probability
# ---------------------------------------------------------------------------

def polya_probability(
    s_vec: List[int], t_vec: List[int], m: int, n: int
) -> float:
    """
    Theorem 8: P[X⃗ = t⃗ | s⃗] under the multi-family Polya urn model.
    Generalised start (families may have initial sizes > 1).
    """
    assert len(s_vec) == len(t_vec), "s_vec and t_vec must have equal length"
    w = len(s_vec)
    n_minus_m = n - m

    # Multinomial coefficient C(n-m; t₁,...,tᵥ)
    log_multi = gammaln(n_minus_m + 1) - sum(gammaln(t + 1) for t in t_vec)
    # Numerator: (m-1)! × Π (sⱼ+tⱼ-1)!/(sⱼ-1)!
    log_num = gammaln(m) + sum(
        gammaln(s + t) - gammaln(s) for s, t in zip(s_vec, t_vec)
    )
    # Denominator: (n-1)!
    log_den = gammaln(n)
    return np.exp(log_multi + log_num - log_den)


# ---------------------------------------------------------------------------
# MLE: Invert Theorem 4 to estimate π̂ from observed count
# ---------------------------------------------------------------------------

def estimate_pi_hat(
    observed_count: float,
    m: int,
    n: int,
    pi_bounds: Tuple[float, float] = (1e-6, 50.0),
) -> Optional[float]:
    """
    MLE inversion of Theorem 4: find π̂ such that E[|M(n)|] = observed_count.
    Uses Brent's method on the monotone function expected_partial(·, m, n).
    Returns None if no solution exists in pi_bounds.
    """

    def equation(pi_hat: float) -> float:
        return expected_partial(pi_hat, m, n) - observed_count

    lo, hi = pi_bounds
    f_lo = equation(lo)
    f_hi = equation(hi)

    if np.isnan(f_lo) or np.isnan(f_hi):
        return None
    if f_lo * f_hi > 0:
        # observed_count outside range achievable by pi_bounds
        return lo if f_lo > 0 else hi
    try:
        return brentq(equation, lo, hi, xtol=1e-10, maxiter=200)
    except (ValueError, RuntimeError):
        return None


def allocate_pi(pi_hat: float, weights: List[float]) -> List[float]:
    """
    Distribute total inheritance probability π̂ across k families
    proportional to weights (e.g., evidence-quality scores).
    Each πᵢ is clamped to [0, 1].
    """
    total_w = sum(weights)
    if total_w == 0 or len(weights) == 0:
        k = len(weights)
        return [min(1.0, pi_hat / max(k, 1))] * k
    raw = [pi_hat * w / total_w for w in weights]
    return [min(1.0, max(0.0, p)) for p in raw]


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------

def z_score(observed: float, expected: float, variance: float) -> float:
    """Z-score = (observed − expected) / sqrt(variance)."""
    if variance <= 0:
        return float("nan")
    return (observed - expected) / np.sqrt(variance)


def p_value_from_z(z: float) -> float:
    """Two-tailed p-value under normal approximation."""
    if np.isnan(z):
        return float("nan")
    return float(2.0 * (1.0 - norm.cdf(abs(z))))


def significance_label(p: float) -> str:
    """Human-readable significance label."""
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Lemma 5 / 6 — Bounds on Gamma function ratios (used in Corollary 9)
# ---------------------------------------------------------------------------

def gamma_ratio_bounds(y: float, z: float) -> Tuple[float, float]:
    """
    Lemma 6: Bounds on Γ(y+z)/Γ(y) for z ≥ 0, y ≥ 1, y ≥ 2z².
    Lower: y^z · (1 − (z²+1/12)/y)
    Upper: y^z · (1 + 3z²/y)
    """
    main = y ** z
    lower = main * (1.0 - (z ** 2 + 1.0 / 12.0) / y)
    upper = main * (1.0 + 3.0 * z ** 2 / y)
    return lower, upper


# ---------------------------------------------------------------------------
# Corollary 5 — Full Duplication: growing-m bounds on expected count
# ---------------------------------------------------------------------------

def expected_full_growing_m_bounds(
    k: int, m: int, n: int
) -> Tuple[float, float]:
    """
    Corollary 5: Bounds when m is growing (n ≥ m ≥ k²).
    Lower: (n/m)^k · (1 − k²/(2m))
    Upper: (n/m)^k · (1 + 3k²/(4n))
    """
    ratio = (n / m) ** k
    lower = ratio * (1.0 - k ** 2 / (2.0 * m))
    upper = ratio * (1.0 + 3.0 * k ** 2 / (4.0 * n))
    return lower, upper
