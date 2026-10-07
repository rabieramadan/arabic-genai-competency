"""End-to-end analysis pipeline.

``run()`` reads the raw response file and regenerates every table, figure and
in-text statistic reported in the article, writing them to ``outdir`` and
returning the headline values as a dict.

This module is the code that verified the published results. It is deliberately
kept as one readable top-to-bottom procedure following the order of the
article's Results section, rather than split into many small functions, so that
a reader checking a specific number can find it in the same order as the paper.
Paired assignments are written on one line where they form a single logical
step (ruff E702 is exempted in pyproject.toml for this reason).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import (  # noqa: E402
    CEILING_CUTOFF,
    DIMS,
    EXTRACTION,
    FREQ_MAP,
    FREQ_ORDER,
    GPA_MAP,
    GPA_ORDER,
    ITEMS,
    LABEL,
    LEVEL_ORDER,
    N_BOOTSTRAP,
    PAIRS,
    PCA_EXTRACTION,
    PREDICTOR_LABELS,
    REGRESSION_STEPS,
    RNG_SEED,
    SHORT,
)
from .psychometrics import (  # noqa: E402
    bootstrap_dz_ci,
    cohens_d_av,
    corrected_item_total,
    cronbach_alpha,
    fmt_p,
    hedges_g,
    htmt,
    omega_cr_ave,
    parallel_analysis,
)

warnings.filterwarnings("ignore")

__all__ = ["run"]


def run(data_path: str | Path, outdir: str | Path) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log, R = [], {}
    def say(s=""):
        print(s); log.append(str(s))

    d = pd.read_csv(data_path)
    say(f"Loaded {data_path}: N = {len(d)} respondents, {len(ITEMS)} Likert items")

    L = d[ITEMS].astype(float)
    miss_pct = L.isna().sum().sum() / L.size * 100
    say(f"Missing Likert cells: {int(L.isna().sum().sum())}/{L.size} = {miss_pct:.2f}%")
    say(f"Missing GPA: {int(d['gpa'].isna().sum())}   Missing weekly use: "
        f"{int(d['weekly_ai_use'].isna().sum())}")
    R["missing_likert_pct"] = round(miss_pct, 2)
    R["n_total"] = int(len(d))
    R["n_missing_gpa"] = int(d["gpa"].isna().sum())
    R["n_missing_freq"] = int(d["weekly_ai_use"].isna().sum())

    # subscale scores = mean of constituent items (manuscript Section 3.3)
    sub = pd.DataFrame({k: d[v].astype(float).mean(axis=1) for k, v in DIMS.items()})
    sub["Overall"] = L.mean(axis=1)

    # ---------------------------------------------- Table 1 participant profile
    say("\n=== TABLE 1. Participant characteristics ===")
    rows = []
    for var, order, name in [("study_level", LEVEL_ORDER, "Study level"),
                             ("gpa", GPA_ORDER, "Grade point average"),
                             ("prior_ai_training", ["Yes", "No"], "Prior AI training"),
                             ("weekly_ai_use", FREQ_ORDER, "Weekly AI use")]:
        vc = d[var].value_counts()
        valid = int(vc.sum())                       # denominator EXCLUDES non-responses
        for cat in order:
            n = int(vc.get(cat, 0))
            rows.append({"Characteristic": name, "Category": cat, "n": n,
                         "%": round(n / valid * 100, 1), "valid_n": valid})
    t1 = pd.DataFrame(rows)
    say(t1.to_string(index=False))
    t1.to_csv(f"{outdir}/table1_participants.csv", index=False)

    # ------------------------------------------- Table A1 item-level statistics
    say("\n=== TABLE A1. Item descriptives & corrected item-total r ===")
    a1 = []
    for dim, items in DIMS.items():
        X = d[items].astype(float)
        for it in items:
            s = d[it].astype(float).dropna()
            a1.append({"Item": it, "Dimension": LABEL[dim],
                       "M": round(s.mean(), 2), "SD": round(s.std(ddof=1), 2),
                       "Skew": round(stats.skew(s, bias=False), 2),
                       "Kurt.": round(stats.kurtosis(s, bias=False), 2),
                       "r(item-total)": round(corrected_item_total(X, it), 2)})
    tA1 = pd.DataFrame(a1)
    say(tA1.to_string(index=False))
    tA1.to_csv(f"{outdir}/tableA1_item_statistics.csv", index=False)

    # ------------------------------------------------- factorability + EFA + PA
    say("\n=== FACTORABILITY, EFA, PARALLEL ANALYSIS, HARMAN ===")
    Lc = L.dropna()
    say(f"EFA listwise N = {len(Lc)}")
    _, kmo_model = calculate_kmo(Lc)
    chi2, bp = calculate_bartlett_sphericity(Lc)
    dfb = len(ITEMS) * (len(ITEMS) - 1) // 2
    say(f"KMO = {kmo_model:.3f}; Bartlett chi2({dfb}) = {chi2:.1f}, p = {fmt_p(bp)}")
    R.update(kmo=round(float(kmo_model), 3), bartlett_chi2=round(float(chi2), 1),
             bartlett_df=dfb, efa_n=int(len(Lc)))

    fa4 = FactorAnalyzer(n_factors=4, rotation="promax", method=EXTRACTION)
    fa4.fit(Lc)
    var4 = fa4.get_factor_variance()
    say(f"4-factor solution: cumulative variance = {var4[2][-1]*100:.1f}%")
    R["efa_cum_variance_pct"] = round(float(var4[2][-1] * 100), 1)

    # order factors to match the manuscript's F1=Asp, F2=Aw, F3=App, F4=Pr
    loads = pd.DataFrame(fa4.loadings_, index=Lc.columns)
    dim_of = {i: k for k, v in DIMS.items() for i in v}
    assign = {}
    for f in loads.columns:
        tops = loads[f].abs().sort_values(ascending=False).index[:4]
        assign[f] = pd.Series([dim_of[t] for t in tops]).mode()[0]
    order = [f for lab in ["AS", "AW", "AP", "PC"] for f in loads.columns if assign[f] == lab]
    order += [f for f in loads.columns if f not in order]
    loads = loads[order]
    loads.columns = ["F1 (Asp.)", "F2 (Aw.)", "F3 (App.)", "F4 (Pr.)"][:len(order)]
    loads["h2"] = fa4.get_communalities()
    tA2 = loads.round(2).reset_index().rename(columns={"index": "Item"})
    say("\n=== TABLE A2. Pattern matrix (principal-axis, promax) ===")
    say(tA2.to_string(index=False))
    tA2.to_csv(f"{outdir}/tableA2_efa_pattern.csv", index=False)

    fcorr = np.corrcoef(fa4.transform(Lc).T)
    off = fcorr[np.triu_indices(4, 1)]
    say(f"Factor correlations range: {off.min():.2f} to {off.max():.2f}")
    R["factor_corr_range"] = [round(float(off.min()), 2), round(float(off.max()), 2)]

    real_eig, pa_thresh, n_pa = parallel_analysis(Lc)
    say(f"Parallel analysis retains {n_pa} factors")
    say(f"  eigenvalue 4 = {real_eig[3]:.2f} vs 95th-pct random = {pa_thresh[3]:.2f}")
    R.update(parallel_n_factors=int(n_pa), eig4=round(float(real_eig[3]), 2),
             eig4_threshold=round(float(pa_thresh[3]), 2))

    fa1 = FactorAnalyzer(n_factors=1, rotation=None, method=EXTRACTION)
    fa1.fit(Lc)
    harman = fa1.get_factor_variance()[1][0] * 100
    say(f"Harman single-factor variance = {harman:.1f}%")
    R["harman_pct"] = round(float(harman), 1)

    # Fig. 1 scree plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(real_eig) + 1), real_eig, "o-", label="Observed eigenvalues")
    ax.plot(range(1, len(pa_thresh) + 1), pa_thresh, "s--",
            label="Parallel analysis (95th percentile)")
    ax.axhline(1, color="grey", lw=.8, ls=":")
    ax.set_xlabel("Factor"); ax.set_ylabel("Eigenvalue")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(f"{outdir}/fig1_scree.png", dpi=300); plt.close(fig)

    # ------------------------------------------------- Table 2 reliability/AVE
    say("\n=== TABLE 2. Reliability and convergent validity ===")
    rows = []
    for k, items in list(DIMS.items()) + [("Overall", ITEMS)]:
        X = d[items].astype(float)
        sc = sub[k]
        om, cr, ave = omega_cr_ave(X)                     # common-factor estimator
        _, _, ave_pca = omega_cr_ave(X, method=PCA_EXTRACTION)  # PCA-based variant
        rows.append({"Dimension": LABEL.get(k, "Overall"), "Items": len(items),
                     "M": round(sc.mean(), 2), "SD": round(sc.std(ddof=1), 2),
                     "alpha": round(cronbach_alpha(X), 2),
                     "omega": round(om, 2), "CR": round(cr, 2),
                     "AVE (common-factor)": "—" if k == "Overall" else round(ave, 2),
                     "AVE (PCA variant)": "—" if k == "Overall" else round(ave_pca, 2)})
    t2 = pd.DataFrame(rows)
    say(t2.to_string(index=False))
    t2.to_csv(f"{outdir}/table2_reliability.csv", index=False)

    # ----------------------------------------- Table 3 discriminant validity
    say("\n=== TABLE 3. Discriminant validity (Fornell-Larcker + HTMT) ===")
    corr = sub[["AW", "AP", "PC", "AS"]].corr()
    sqrt_ave = {k: np.sqrt(omega_cr_ave(d[v].astype(float))[2]) for k, v in DIMS.items()}
    sqrt_ave_pca = {k: np.sqrt(omega_cr_ave(d[v].astype(float), method=PCA_EXTRACTION)[2])
                    for k, v in DIMS.items()}
    say("sqrt(AVE) common-factor: " + ", ".join(f"{k}={v:.2f}" for k, v in sqrt_ave.items()))
    say("sqrt(AVE) PCA variant   : " + ", ".join(f"{k}={v:.2f}" for k, v in sqrt_ave_pca.items()))
    t3 = corr.round(2).copy()
    for k in t3.columns:
        t3.loc[k, k] = round(sqrt_ave[k], 2)
    say("Fornell-Larcker matrix (diagonal = sqrt(AVE)):")
    say(t3.to_string())
    H = htmt(d[ITEMS].astype(float), DIMS)
    say("HTMT ratios: " + ", ".join(f"{k} = {v:.2f}" for k, v in H.items()))
    say(f"HTMT range: {min(H.values()):.2f}-{max(H.values()):.2f}  (threshold .85)")
    t3.to_csv(f"{outdir}/table3_discriminant.csv")
    pd.Series(H).round(3).to_csv(f"{outdir}/table3_htmt.csv", header=["HTMT"])
    R["htmt_range"] = [round(float(min(H.values())), 2), round(float(max(H.values())), 2)]
    R["sqrt_ave"] = {k: round(float(v), 2) for k, v in sqrt_ave.items()}
    R["sqrt_ave_pca"] = {k: round(float(v), 2) for k, v in sqrt_ave_pca.items()}
    R["ave_common_factor"] = {k: round(float(omega_cr_ave(d[v].astype(float))[2]), 2)
                              for k, v in DIMS.items()}
    R["ave_pca"] = {k: round(float(omega_cr_ave(d[v].astype(float), method=PCA_EXTRACTION)[2]), 2)
                    for k, v in DIMS.items()}

    # ---------------------------------- RQ2: repeated-measures profile + Table 4
    say("\n=== RQ2. Repeated-measures ANOVA across the four dimensions ===")
    W = sub[["AW", "AP", "PC", "AS"]].dropna()
    say(f"Repeated-measures listwise N = {len(W)}")
    lf = W.reset_index().melt(id_vars="index", var_name="dim", value_name="score")
    aov = pg.rm_anova(data=lf, dv="score", within="dim", subject="index",
                      correction=True, detailed=True)

    row = aov.iloc[0]
    say(f"F({row['DF']},{aov.iloc[1]['DF']}) = {row['F']:.2f}, p = {fmt_p(row['p_unc'])}, "
        f"ng2 = {row['ng2']:.2f}")
    say(f"Mauchly W = {row['W_spher']:.2f}, p = {fmt_p(row['p_spher'])}; GG epsilon = {row['eps']:.2f}, "
        f"p(GG) = {fmt_p(row['p_GG_corr'])}")
    R.update(rm_F=round(float(row["F"]), 2), rm_df=[int(row["DF"]), int(aov.iloc[1]["DF"])],
             rm_ng2=round(float(row["ng2"]), 2), mauchly_W=round(float(row['W_spher']), 2),
             gg_eps=round(float(row["eps"]), 2), rm_n=int(len(W)))

    pairs, nice = PAIRS, SHORT
    rows = []
    for a, b in pairs:
        t, p = stats.ttest_rel(W[a], W[b])
        diff = W[a] - W[b]
        dz = diff.mean() / diff.std(ddof=1)
        # d_av: mean difference standardised by the average SD of the two measures
        d_av = cohens_d_av(W[a], W[b])
        rows.append({"Contrast": f"{nice[a]} - {nice[b]}",
                     "Mean diff.": round(diff.mean(), 2),
                     "p_raw": p, "p (Bonf.)": min(p * len(pairs), 1.0),
                     "d_av": round(abs(d_av), 2), "d_z": round(abs(dz), 2)})
    t4 = pd.DataFrame(rows)
    t4["p (Bonf.)"] = t4["p (Bonf.)"].map(fmt_p)
    say(t4.drop(columns="p_raw").to_string(index=False))
    t4.drop(columns="p_raw").to_csv(f"{outdir}/table4_contrasts.csv", index=False)

    # focal aspiration-application contrast
    diff = W["AS"] - W["AP"]
    t, p = stats.ttest_rel(W["AS"], W["AP"])
    _, pw = stats.wilcoxon(W["AS"], W["AP"])
    dz = diff.mean() / diff.std(ddof=1)
    d_av_focal = cohens_d_av(W["AS"], W["AP"])
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    ci = stats.t.interval(.95, len(diff) - 1, diff.mean(), se)
    dci = bootstrap_dz_ci(diff.values, n_boot=N_BOOTSTRAP, seed=RNG_SEED)
    say(f"\nFocal gap: t({len(diff)-1}) = {t:.2f}, p = {fmt_p(p)}; Wilcoxon p = {fmt_p(pw)}")
    say(f"  mean diff = {diff.mean():.2f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    say(f"  d_z = {abs(dz):.2f}, 95% CI [{dci[0]:.2f}, {dci[1]:.2f}]")
    say(f"  d_av = {abs(d_av_focal):.2f}  (average-SD standardiser, as used in Table 4)")
    say(f"  Aspiration ceiling: {(W['AS']>=4.5).mean()*100:.1f}% scored >= 4.5 "
        f"(RM-complete N={len(W)}); {(sub['AS']>=4.5).mean()*100:.1f}% on the full N={len(d)}")
    R.update(gap_t=round(float(t), 2), gap_df=int(len(diff) - 1),
             gap_mean=round(float(diff.mean()), 2),
             gap_ci=[round(float(ci[0]), 2), round(float(ci[1]), 2)],
             gap_dz=round(float(abs(dz)), 2), gap_dav=round(float(abs(d_av_focal)), 2),
             gap_dz_ci=[round(float(dci[0]), 2), round(float(dci[1]), 2)],
             ceiling_pct=round(float((W['AS'] >= CEILING_CUTOFF).mean() * 100), 1),
             ceiling_pct_full=round(float((sub['AS'] >= CEILING_CUTOFF).mean() * 100), 1))

    say("\nInter-dimension correlations (with p-values):")
    corr_p = {}
    for a, b in [("AW", "AP"), ("AW", "PC"), ("AW", "AS"), ("AP", "PC"),
                 ("AP", "AS"), ("PC", "AS")]:
        r, p = stats.pearsonr(W[a], W[b])
        corr_p[f"{a}-{b}"] = [round(float(r), 2), round(float(p), 3)]
        say(f"  r({nice[a]}, {nice[b]}) = {r:.2f}, p = {fmt_p(p)}")
    R["inter_dim_corr"] = corr_p

    # Fig. 2 dimension means
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mm = W.mean(); ee = 1.96 * W.sem()
    ax.bar([nice[c] for c in W.columns], mm, yerr=ee, capsize=4, color="#4C72B0")
    ax.axhline(3.0, ls="--", color="grey", lw=1)
    ax.set_ylabel("Mean score (1-5)"); ax.set_ylim(0, 5)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right"); fig.tight_layout()
    fig.savefig(f"{outdir}/fig2_dimension_means.png", dpi=300); plt.close(fig)

    # Fig. 3 pairwise differences
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labs, mids, los, his = [], [], [], []
    for a, b in pairs:
        dd = W[a] - W[b]
        c = stats.t.interval(.95, len(dd) - 1, dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd)))
        labs.append(f"{nice[a]} - {nice[b]}"); mids.append(dd.mean())
        los.append(dd.mean() - c[0]); his.append(c[1] - dd.mean())
    ax.errorbar(mids, range(len(labs)), xerr=[los, his], fmt="o", capsize=4, color="#4C72B0")
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    ax.set_xlabel("Mean difference (scale points)"); fig.tight_layout()
    fig.savefig(f"{outdir}/fig3_pairwise_differences.png", dpi=300); plt.close(fig)

    # ------------------------------- RQ3: group differences + Table 5 regression
    say("\n=== RQ3. Predictors of overall competency ===")
    m = d.copy()
    m["overall"]  = sub["Overall"]
    m["app"]      = sub["AP"]
    m["gpa_n"]    = m["gpa"].map(GPA_MAP)
    m["freq_n"]   = m["weekly_ai_use"].map(FREQ_MAP)
    m["train_n"]  = (m["prior_ai_training"] == "Yes").astype(int)
    m["postgrad"] = (m["study_level"] != "Bachelor").astype(int)

    g = m.dropna(subset=["overall", "freq_n"])
    groups = [g.loc[g["weekly_ai_use"] == c, "overall"].values for c in FREQ_ORDER]
    lev = stats.levene(*groups)
    F, p = stats.f_oneway(*groups)
    k = len(groups); n = sum(len(x) for x in groups)
    ss_b = sum(len(x) * (x.mean() - g["overall"].mean()) ** 2 for x in groups)
    eta2 = ss_b / ((g["overall"] - g["overall"].mean()) ** 2).sum()
    Hk, ph = stats.kruskal(*groups)
    wel = pg.welch_anova(data=g, dv="overall", between="weekly_ai_use")
    say(f"Weekly use: Levene p = {fmt_p(lev.pvalue)}; F({k-1},{n-k}) = {F:.2f}, "
        f"p = {fmt_p(p)}, eta2 = {eta2:.2f}")
    say(f"  Welch p = {fmt_p(wel[[c for c in wel.columns if c.lower().replace('-','_')=='p_unc'][0]].iloc[0])}; Kruskal-Wallis H = {Hk:.2f}, p = {fmt_p(ph)}")
    R.update(freq_F=round(float(F), 2), freq_df=[k - 1, n - k],
             freq_eta2=round(float(eta2), 2), kruskal_H=round(float(Hk), 2),
             levene_freq_p=round(float(lev.pvalue), 3))

    tuk = pairwise_tukeyhsd(g["overall"], g["weekly_ai_use"])
    say("\nTukey HSD contrasts:")
    say(str(tuk))
    say("Hedges' g for frequency pairs:")
    gvals = {}
    for i, a in enumerate(FREQ_ORDER):
        for b in FREQ_ORDER[i + 1:]:
            ga = g.loc[g["weekly_ai_use"] == a, "overall"].values
            gb = g.loc[g["weekly_ai_use"] == b, "overall"].values
            gvals[f"{b} vs {a}"] = round(float(hedges_g(gb, ga)), 2)
            say(f"  {b} vs {a}: g = {hedges_g(gb, ga):+.2f}")
    R["hedges_g_freq"] = gvals
    with open(f"{outdir}/tukey_weekly_use.txt", "w") as f:
        f.write(str(tuk))

    tr = m.dropna(subset=["overall"])
    y1 = tr.loc[tr["train_n"] == 1, "overall"]; y0 = tr.loc[tr["train_n"] == 0, "overall"]
    levt = stats.levene(y1, y0)
    wt = stats.ttest_ind(y1, y0, equal_var=False)
    mw = stats.mannwhitneyu(y1, y0)
    sp = np.sqrt(((len(y1)-1)*y1.var(ddof=1) + (len(y0)-1)*y0.var(ddof=1)) / (len(y1)+len(y0)-2))
    say(f"\nPrior training: M = {y1.mean():.2f} (n={len(y1)}) vs {y0.mean():.2f} (n={len(y0)})")
    say(f"  Levene p = {fmt_p(levt.pvalue)}; Welch t p = {fmt_p(wt.pvalue)}; "
        f"Mann-Whitney p = {fmt_p(mw.pvalue)}; d = {(y1.mean()-y0.mean())/sp:.2f}")
    R.update(train_M_yes=round(float(y1.mean()), 2), train_M_no=round(float(y0.mean()), 2),
             train_d=round(float((y1.mean() - y0.mean()) / sp), 2),
             train_levene_p=round(float(levt.pvalue), 3),
             train_welch_p=round(float(wt.pvalue), 3), train_mw_p=round(float(mw.pvalue), 3))

    gp = m.dropna(subset=["overall", "gpa"])
    gg = [gp.loc[gp["gpa"] == c, "overall"].values for c in GPA_ORDER]
    Fg, pgv = stats.f_oneway(*gg)
    ssb = sum(len(x) * (x.mean() - gp["overall"].mean()) ** 2 for x in gg)
    eta2g = ssb / ((gp["overall"] - gp["overall"].mean()) ** 2).sum()
    say(f"GPA: F({len(gg)-1},{len(gp)-len(gg)}) = {Fg:.2f}, p = {fmt_p(pgv)}, eta2 = {eta2g:.3f}")
    R.update(gpa_F=round(float(Fg), 2), gpa_p=round(float(pgv), 3),
             gpa_df=[len(gg) - 1, len(gp) - len(gg)], gpa_eta2=round(float(eta2g), 3))

    # hierarchical regression
    say("\n--- Hierarchical regression (overall competency) ---")
    reg = m.dropna(subset=["overall", "gpa_n", "freq_n", "postgrad", "train_n"])
    say(f"Regression listwise N = {len(reg)}")
    steps = REGRESSION_STEPS
    prev, fits, stepinfo = None, [], []
    for i, pr in enumerate(steps, 1):
        X = sm.add_constant(reg[pr]); fit = sm.OLS(reg["overall"], X).fit()
        fits.append(fit)
        info = {"step": i, "R2": round(float(fit.rsquared), 3)}
        line = f"Step {i}: R2 = {fit.rsquared:.3f}"
        if prev is not None:
            dR = fit.rsquared - prev.rsquared
            q = len(pr) - len(steps[i - 2]); dfe = len(reg) - len(pr) - 1
            Fch = (dR / q) / ((1 - fit.rsquared) / dfe)
            pch = 1 - stats.f.cdf(Fch, q, dfe)
            info.update(dR2=round(float(dR), 3), F_change=round(float(Fch), 2),
                        p_change=round(float(pch), 4))
            line += f"; dR2 = {dR:.3f}, F-change({q},{dfe}) = {Fch:.2f}, p = {fmt_p(pch)}"
        say(line); stepinfo.append(info); prev = fit
    full = fits[-1]
    say(f"Model F({int(full.df_model)},{int(full.df_resid)}) = {full.fvalue:.2f}, "
        f"p = {fmt_p(full.f_pvalue)}, R2 = {full.rsquared:.3f}")
    R["regression_steps"] = stepinfo
    R.update(reg_n=int(len(reg)), reg_F=round(float(full.fvalue), 2),
             reg_df=[int(full.df_model), int(full.df_resid)],
             reg_R2=round(float(full.rsquared), 3))

    sd_y = reg["overall"].std(ddof=1)
    names = PREDICTOR_LABELS
    rows = []
    for v in steps[-1]:
        b = full.params[v]; se_ = full.bse[v]; ci_ = full.conf_int().loc[v]
        rows.append({"Predictor (Step 3)": names[v], "B": round(b, 2), "SE": round(se_, 2),
                     "beta": round(b * reg[v].std(ddof=1) / sd_y, 2),
                     "95% CI (B)": f"[{ci_[0]:.2f}, {ci_[1]:.2f}]",
                     "p": fmt_p(full.pvalues[v])})
    t5 = pd.DataFrame(rows)
    say(t5.to_string(index=False))
    t5.to_csv(f"{outdir}/table5_regression.csv", index=False)

    Xd = sm.add_constant(reg[steps[-1]])
    say("\nDiagnostics:")
    vifs = {c: round(float(variance_inflation_factor(Xd.values, i)), 2)
            for i, c in enumerate(Xd.columns) if c != "const"}
    say("  VIF: " + ", ".join(f"{c} = {v}" for c, v in vifs.items()))
    bptest = het_breuschpagan(full.resid, Xd)
    shap = stats.shapiro(full.resid)
    cookmax = full.get_influence().cooks_distance[0].max()
    say(f"  Breusch-Pagan p = {fmt_p(bptest[1])}; Shapiro-Wilk p = {fmt_p(shap.pvalue)}")
    say(f"  max Cook's D = {cookmax:.3f}")
    R.update(vif=vifs, bp_p=round(float(bptest[1]), 3), shapiro_p=round(float(shap.pvalue), 3),
             cooks_max=round(float(cookmax), 3))

    appreg = m.dropna(subset=["app", "gpa_n", "freq_n", "postgrad", "train_n"])
    fa_ = sm.OLS(appreg["app"], sm.add_constant(appreg[steps[-1]])).fit()
    say(f"  Application as outcome: R2 = {fa_.rsquared:.3f}, freq p = "
        f"{fmt_p(fa_.pvalues['freq_n'])}, training p = {fmt_p(fa_.pvalues['train_n'])}")
    R.update(app_R2=round(float(fa_.rsquared), 3),
             app_freq_p=round(float(fa_.pvalues["freq_n"]), 4),
             app_train_p=round(float(fa_.pvalues["train_n"]), 3))

    # sensitivity power
    say("\n--- Sensitivity power analysis (80% power, alpha = .05) ---")
    f_det = FTestAnovaPower().solve_power(effect_size=None, nobs=len(d), alpha=.05,
                                          power=.80, k_groups=4)
    lo, hi = 0.01, 0.99
    for _ in range(60):                       # bisect the detectable correlation
        mid = (lo + hi) / 2
        z = 0.5 * np.log((1 + mid) / (1 - mid)) * np.sqrt(len(d) - 3)
        lo, hi = (mid, hi) if stats.norm.cdf(z - 1.959964) < .80 else (lo, mid)
    d_det = TTestIndPower().solve_power(effect_size=None, nobs1=len(y1),
                                        ratio=len(y0) / len(y1), alpha=.05, power=.80)
    say(f"  Detectable correlation r >= {hi:.2f} (N = {len(d)})")
    say(f"  Detectable ANOVA effect f >= {f_det:.2f} (k = 4, N = {len(d)})")
    say(f"  Detectable d for training comparison ({len(y1)} vs {len(y0)}) >= {d_det:.2f}")
    R.update(detectable_r=round(float(hi), 2), detectable_f=round(float(f_det), 2),
             detectable_d_training=round(float(d_det), 2))

    # Fig. 4 competency by frequency
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mu = [x.mean() for x in groups]; er = [1.96 * stats.sem(x) for x in groups]
    ax.bar(FREQ_ORDER, mu, yerr=er, capsize=4, color="#4C72B0")
    ax.set_xlabel("Weekly frequency of AI use"); ax.set_ylabel("Overall competency (1-5)")
    ax.set_ylim(0, 5); fig.tight_layout()
    fig.savefig(f"{outdir}/fig4_competency_by_frequency.png", dpi=300); plt.close(fig)

    with open(f"{outdir}/analysis_log.txt", "w") as f:
        f.write("\n".join(log))
    with open(f"{outdir}/key_results.json", "w") as f:
        json.dump(R, f, indent=2)
    say(f"\nWrote tables, figures and logs to {outdir}/")
    return R
