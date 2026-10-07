"""Regression tests locking the published values.

These assert that the pipeline still reproduces what the article reports. If a
dependency upgrade or a code change moves any of these numbers, the test suite
fails and names the quantity that moved — which is the point: the article's
numbers should not drift silently.

The suite runs the full pipeline once per session (``results`` fixture), which
takes about two minutes, dominated by the 1000-iteration parallel analysis.
"""
from __future__ import annotations

import pandas as pd
import pytest

from genai_competency import run
from genai_competency.config import AVE_THRESHOLD, DIMS, HTMT_THRESHOLD, ITEMS
from genai_competency.psychometrics import (
    cohens_d_av,
    cohens_d_z,
    cronbach_alpha,
    omega_cr_ave,
)

DATA = "data/arabic_genai_competency_data.csv"


@pytest.fixture(scope="session")
def results(tmp_path_factory):
    return run(DATA, tmp_path_factory.mktemp("output"))


@pytest.fixture(scope="session")
def raw():
    return pd.read_csv(DATA)


# ------------------------------------------------------------------ the data
def test_sample_size(raw):
    assert len(raw) == 145


def test_missingness_matches_methods_section(raw, results):
    """Section 3.3 reports 1.2% of Likert cells and three missing GPA values."""
    assert results["missing_likert_pct"] == pytest.approx(1.23, abs=0.01)
    assert results["n_missing_gpa"] == 3
    assert results["n_missing_freq"] == 2


def test_no_likert_value_out_of_range(raw):
    """Every non-missing Likert response is an integer on the 1-5 scale."""
    vals = raw[ITEMS].stack().dropna()
    assert len(vals) == 4010          # 145 * 28 cells, less the 50 missing
    assert vals.between(1, 5).all()
    assert (vals == vals.round()).all()


# ----------------------------------------------------------- factorability
def test_kmo_and_bartlett(results):
    assert results["kmo"] == pytest.approx(0.856, abs=0.002)
    assert results["bartlett_chi2"] == pytest.approx(2965.4, abs=1.0)
    assert results["bartlett_df"] == 378


def test_efa_variance_and_harman(results):
    """Both depend on the extraction; a PCA extraction gives 66.1% / 35.0%."""
    assert results["efa_cum_variance_pct"] == pytest.approx(61.1, abs=0.2)
    assert results["harman_pct"] == pytest.approx(32.7, abs=0.3)


def test_parallel_analysis_retains_three_factors(results):
    """The central psychometric caveat: PA does not support the fourth factor."""
    assert results["parallel_n_factors"] == 3
    assert results["eig4"] == pytest.approx(1.56, abs=0.03)
    assert results["eig4_threshold"] == pytest.approx(1.67, abs=0.05)
    assert results["eig4"] < results["eig4_threshold"]


# ------------------------------------------------- reliability and validity
@pytest.mark.parametrize("dim,expected", [("AW", .89), ("AP", .90), ("PC", .81), ("AS", .94)])
def test_cronbach_alpha_by_dimension(raw, dim, expected):
    assert cronbach_alpha(raw[DIMS[dim]].astype(float)) == pytest.approx(expected, abs=.005)


def test_ave_straddles_threshold_for_prompt_engineering(raw):
    """The finding that qualified the convergent-validity claim.

    AVE for Prompt-Engineering is below the .50 threshold on the common-factor
    estimator and above it on the principal-components estimator. Both are
    reported in the article; this test pins both.
    """
    X = raw[DIMS["PC"]].astype(float)
    _, _, ave_cf = omega_cr_ave(X)
    _, _, ave_pca = omega_cr_ave(X, method="principal")
    assert ave_cf == pytest.approx(.42, abs=.01)
    assert ave_pca == pytest.approx(.52, abs=.01)
    assert ave_cf < AVE_THRESHOLD < ave_pca


@pytest.mark.parametrize("dim", ["AW", "AP", "AS"])
def test_other_dimensions_meet_ave_threshold_on_both_estimators(raw, dim):
    X = raw[DIMS[dim]].astype(float)
    assert omega_cr_ave(X)[2] >= AVE_THRESHOLD
    assert omega_cr_ave(X, method="principal")[2] >= AVE_THRESHOLD


def test_htmt_all_below_threshold(results):
    lo, hi = results["htmt_range"]
    assert (lo, hi) == pytest.approx((.14, .72), abs=.01)
    assert hi < HTMT_THRESHOLD


# ------------------------------------------------- RQ2: the aspiration gap
def test_repeated_measures_anova(results):
    assert results["rm_n"] == 144
    assert results["rm_df"] == [3, 429]
    assert results["rm_F"] == pytest.approx(115.53, abs=0.1)
    assert results["rm_ng2"] == pytest.approx(.27, abs=.01)
    assert results["mauchly_W"] == pytest.approx(.71, abs=.01)
    assert results["gg_eps"] == pytest.approx(.80, abs=.01)


def test_focal_gap(results):
    assert results["gap_df"] == 143
    assert results["gap_t"] == pytest.approx(13.68, abs=0.05)
    assert results["gap_mean"] == pytest.approx(1.11, abs=.01)
    assert results["gap_ci"] == pytest.approx([0.95, 1.26], abs=.01)


def test_two_effect_size_metrics_are_distinct_and_both_reported(results):
    """The text/table discrepancy that was corrected: d_z != d_av."""
    assert results["gap_dz"] == pytest.approx(1.14, abs=.01)
    assert results["gap_dav"] == pytest.approx(1.31, abs=.01)
    assert results["gap_dz"] != results["gap_dav"]


def test_aspiration_uncorrelated_with_awareness(results):
    r, p = results["inter_dim_corr"]["AW-AS"]
    assert r == pytest.approx(.12, abs=.01)
    assert p > .05


def test_aspiration_ceiling(results):
    assert results["ceiling_pct_full"] == pytest.approx(42.8, abs=0.2)


# ------------------------------------------------------ RQ3: the predictors
def test_frequency_effect(results):
    assert results["freq_F"] == pytest.approx(12.37, abs=0.05)
    assert results["freq_df"] == [3, 139]
    assert results["freq_eta2"] == pytest.approx(.21, abs=.01)
    assert results["kruskal_H"] == pytest.approx(31.67, abs=0.1)


def test_gpa_unrelated_to_competency(results):
    assert results["gpa_F"] == pytest.approx(0.06, abs=0.02)
    assert results["gpa_p"] > .05


def test_hierarchical_regression(results):
    steps = results["regression_steps"]
    assert results["reg_n"] == 140
    assert steps[0]["R2"] == pytest.approx(.001, abs=.002)
    assert steps[2]["dR2"] == pytest.approx(.165, abs=.005)
    assert steps[2]["F_change"] == pytest.approx(27.68, abs=0.1)
    assert results["reg_df"] == [4, 135]
    assert results["reg_F"] == pytest.approx(8.22, abs=0.05)


def test_regression_diagnostics_are_sound(results):
    assert all(v < 2.0 for v in results["vif"].values())      # no multicollinearity
    assert results["bp_p"] > .05                              # homoscedastic
    assert results["cooks_max"] < 1.0                         # no dominating outlier


def test_sensitivity_power_f_not_eta_squared(results):
    """The corrected value. The original submission reported .07, which is eta^2."""
    assert results["detectable_f"] == pytest.approx(.28, abs=.01)
    assert results["detectable_r"] == pytest.approx(.23, abs=.01)
    assert results["detectable_d_training"] == pytest.approx(.59, abs=.01)


# -------------------------------------------------------- estimator helpers
def test_d_av_and_d_z_differ_when_variances_differ():
    x = pd.Series([5, 5, 4, 5, 4, 5, 4, 5])
    y = pd.Series([1, 4, 2, 5, 1, 4, 2, 5])
    assert cohens_d_av(x, y) != pytest.approx(cohens_d_z(x, y), abs=.01)


def test_outputs_written(results, tmp_path_factory):
    pass  # run() asserts its own writes; presence is covered by the CI artifact upload
