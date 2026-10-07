"""Reliability, validity and effect-size estimators.

Each function takes a plain DataFrame of item responses and returns a plain
float or dict, so they can be unit-tested and reused independently of the
pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from scipy import stats

from .config import EXTRACTION, N_PARALLEL_ITER, RNG_SEED

__all__ = [
    "cronbach_alpha", "single_factor_loadings", "omega_cr_ave",
    "corrected_item_total", "htmt", "parallel_analysis",
    "hedges_g", "cohens_d_av", "cohens_d_z", "bootstrap_dz_ci", "fmt_p",
]


def cronbach_alpha(X: pd.DataFrame) -> float:
    """Cronbach's alpha, listwise-deleted."""
    X = X.dropna()
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))


def single_factor_loadings(X: pd.DataFrame, method: str = EXTRACTION) -> np.ndarray:
    """Standardised loadings from a one-factor model fitted to ``X``."""
    X = X.dropna()
    fa = FactorAnalyzer(n_factors=1, rotation=None, method=method)
    fa.fit(X)
    return fa.loadings_[:, 0]


def omega_cr_ave(X: pd.DataFrame, method: str = EXTRACTION) -> tuple[float, float, float]:
    """McDonald's omega-total, composite reliability and AVE.

    All three are derived from the standardised loadings of a one-factor model,
    so all three inherit the choice of ``method``. For a unidimensional
    congeneric model omega-total and composite reliability share an estimator
    and are numerically identical; they are returned separately because the two
    are reported under different names in the article.

    Returns
    -------
    (omega, cr, ave)
    """
    lam = single_factor_loadings(X, method)
    err = 1 - lam ** 2
    sl2 = lam.sum() ** 2
    omega = sl2 / (sl2 + err.sum())
    cr = omega
    ave = float(np.mean(lam ** 2))
    return float(omega), float(cr), ave


def corrected_item_total(X: pd.DataFrame, item: str) -> float:
    """Correlation of ``item`` with the sum of the REMAINING items."""
    X = X.dropna()
    rest = X.drop(columns=[item]).sum(axis=1)
    return float(stats.pearsonr(X[item], rest)[0])


def htmt(df: pd.DataFrame, dims: dict[str, list[str]]) -> dict[str, float]:
    """Heterotrait-monotrait ratio of correlations for every construct pair."""
    C = df[sum(dims.values(), [])].corr()
    out: dict[str, float] = {}
    keys = list(dims)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            hetero = C.loc[dims[a], dims[b]].values.flatten()
            mono_a = C.loc[dims[a], dims[a]].values[np.triu_indices(len(dims[a]), 1)]
            mono_b = C.loc[dims[b], dims[b]].values[np.triu_indices(len(dims[b]), 1)]
            out[f"{a}-{b}"] = float(
                np.mean(np.abs(hetero)) / np.sqrt(np.mean(np.abs(mono_a)) * np.mean(np.abs(mono_b)))
            )
    return out


def parallel_analysis(X: pd.DataFrame, n_iter: int = N_PARALLEL_ITER,
                      pct: int = 95, seed: int = RNG_SEED,
                      method: str = EXTRACTION) -> tuple[np.ndarray, np.ndarray, int]:
    """Horn's parallel analysis against random normal data of the same shape.

    Returns
    -------
    (observed_eigenvalues, percentile_threshold, n_factors_retained)
    """
    X = X.dropna()
    n, p = X.shape
    fa = FactorAnalyzer(n_factors=p, rotation=None, method=method)
    fa.fit(X)
    real = fa.get_eigenvalues()[0]

    rng = np.random.default_rng(seed)
    sim = np.empty((n_iter, p))
    for i in range(n_iter):
        R = pd.DataFrame(rng.normal(size=(n, p)))
        f = FactorAnalyzer(n_factors=p, rotation=None, method=method)
        f.fit(R)
        sim[i] = f.get_eigenvalues()[0]

    thresh = np.percentile(sim, pct, axis=0)
    return real, thresh, int((real > thresh).sum())


# ------------------------------------------------------------- effect sizes
def hedges_g(a, b) -> float:
    """Bias-corrected standardised mean difference for independent groups."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    d = (np.mean(a) - np.mean(b)) / sp
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def cohens_d_z(x, y) -> float:
    """Within-subject effect size: mean difference over the SD of the differences."""
    diff = np.asarray(x) - np.asarray(y)
    return float(diff.mean() / diff.std(ddof=1))


def cohens_d_av(x, y) -> float:
    """Mean difference standardised by the AVERAGE SD of the two measures.

    This is the metric reported in Table 4 of the article. It is systematically
    different from :func:`cohens_d_z` whenever the two measures differ in
    variance, which is why the article now labels both explicitly.
    """
    x, y = np.asarray(x), np.asarray(y)
    pooled = np.sqrt((x.std(ddof=1) ** 2 + y.std(ddof=1) ** 2) / 2)
    return float((x - y).mean() / pooled)


def bootstrap_dz_ci(diff, n_boot: int = 5000, seed: int = RNG_SEED,
                    alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for d_z."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff)
    stats_ = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(diff, len(diff), replace=True)
        stats_[i] = s.mean() / s.std(ddof=1)
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def fmt_p(p: float) -> str:
    """APA-style p-value: '<.001' or a leading-zero-stripped three-decimal value."""
    return "<.001" if p < .001 else f"{p:.3f}".lstrip("0")
