# Arabic GenAI competency — data and reproduction package

[![reproduce](https://github.com/rabieramadan/arabic-genai-competency/actions/workflows/reproduce.yml/badge.svg)](https://github.com/rabieramadan/arabic-genai-competency/actions/workflows/reproduce.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Data, instrument and analysis code for:

> **Aspiration without application: Developing and initially validating a four-dimensional measure
> of generative-AI competency and the readiness gap among Arabic language and literature students**
> Torky, F. S. A., Al Kindi, K. S. M., & Hassanein, M. M.
> Department of Arabic Language, Sultan Qaboos University, Muscat, Sultanate of Oman

*Title note: this is the title of the **revised** manuscript. The original submission read
"…: Validating a four-dimensional measure of …". "Developing and initially validating" was adopted
in revision because Horn's parallel analysis retained three factors rather than four, so the
original wording asserted a settled structure the data do not establish (see
`docs/overstatement_register.csv`). The title is provisional until the article is accepted.*

A survey of 145 Arabic language and literature students measuring generative-AI competency across
four dimensions — awareness, functional application, prompt engineering and critical appraisal, and
aspiration. The headline finding is a gap of 1.11 scale points between what students aspire to do
with these tools and what they can currently do (*d*<sub>z</sub> = 1.14), with aspiration
essentially uncorrelated with awareness (*r* = .12, *p* = .16).

Everything reported in the article is regenerated from the raw response file by one command.

---

## Install and reproduce

```bash
git clone https://github.com/rabieramadan/arabic-genai-competency.git
cd arabic-genai-competency
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

genai-competency --data data/arabic_genai_competency_data.csv --outdir output
```

Or `make install && make reproduce`. Runtime is roughly two minutes, dominated by the
1,000-iteration parallel analysis and the 5,000-sample bootstrap. Both are seeded, so repeated runs
give identical output.

Opening the project in PyCharm: see **[PYCHARM_SETUP.md](PYCHARM_SETUP.md)**.

> **`scikit-learn` must stay below 1.6.** `factor_analyzer` calls
> `check_array(force_all_finite=...)`, renamed in scikit-learn 1.6; a newer version raises
> `TypeError` at the factor-analysis step. The constraint is pinned in `pyproject.toml`.

## Use it as a library

```python
from genai_competency import run

results = run("data/arabic_genai_competency_data.csv", "output")
print(results["gap_mean"], results["gap_dz"])       # 1.11 1.14
print(results["parallel_n_factors"])                 # 3
```

Individual estimators are importable and independently testable:

```python
from genai_competency.psychometrics import omega_cr_ave, htmt, parallel_analysis
from genai_competency.config import DIMS
```

## Verify the reproduction

```bash
pytest
```

27 regression tests assert that the pipeline still produces the published values — sample size,
missingness, KMO and Bartlett, the three-factor parallel-analysis result, per-dimension alpha, AVE
under both estimators, the repeated-measures ANOVA, the focal gap and its confidence interval, the
frequency effect, the hierarchical regression and its diagnostics, and the corrected sensitivity
power. If a dependency upgrade moves any of them, the suite fails and names the quantity that moved.

## Layout

```
├── data/
│   ├── arabic_genai_competency_data.csv        raw anonymised responses (N = 145)
│   ├── arabic_genai_competency_processed.csv   derived scores and model recodes
│   ├── codebook_extended.csv                  bilingual codebook, incl. derived-variable formulas
│   └── analytic_samples.csv                   the six analytic samples and their n
├── instrument/
│   └── S1_Survey_instrument_bilingual.docx    the questionnaire as administered, AR + EN
├── src/genai_competency/
│   ├── config.py                              dimensions, recodes, thresholds, seeds
│   ├── psychometrics.py                       alpha, omega/CR/AVE, HTMT, parallel analysis, effect sizes
│   ├── pipeline.py                            the analysis, in the order of the article's Results
│   └── cli.py                                 `genai-competency` entry point
├── scripts/reproduce_analysis.py              thin wrapper; the filename cited in the article
├── tests/test_reproduction.py                 the published values, asserted
└── docs/                                      verification table, compliance matrix, claim register
```

## The instrument

28 statements on a five-point Likert scale (5 = strongly agree), no reverse-scored items, grouped
into four a priori dimensions, plus four background items and seven open-ended items.

| Prefix | Dimension | Items | α | ω | AVE (CF / PCA) |
|---|---|---|---|---|---|
| `AW` | Awareness & Availability | 6 | .89 | .89 | .58 / .64 |
| `AP` | Functional Application | 8 | .90 | .90 | .52 / .58 |
| `PC` | Prompt Eng. & Critical Appraisal | 6 | .81 | .81 | **.42** / .52 |
| `AS` | Aspiration & Vision | 8 | .94 | .94 | .68 / .72 |

## Two estimators, reported side by side

McDonald's ω, composite reliability and AVE are all derived from the standardised loadings of a
one-factor model, so all three inherit the choice of extraction — and for one dimension the two
common choices straddle the conventional .50 AVE threshold:

- **Common-factor estimator** (minimum residual — consistent with the extraction used for the
  exploratory factor analysis): Prompt-Engineering AVE = **.42**, below threshold.
- **Principal-components estimator** (more often seen in applied work): AVE = **.52**, above it.

The pipeline computes both, the article reports both, and the fourth dimension is described as
provisional rather than established. Three further findings point the same way: parallel analysis
retained three factors (eigenvalue 1.56 against a 1.67 criterion), two of the six Prompt-Engineering
items load more strongly on Awareness than on their intended factor, and the exploratory solution
does not cleanly separate the two. A three-factor model merging them is a competing account these
data cannot rule out.

Set `EXTRACTION` in `config.py` to change which estimator is treated as primary.

## Missing data

50 of 4,060 Likert cells are missing (1.23%), plus three GPA and two weekly-use non-responses.
Nothing is imputed. Each analysis uses listwise deletion on the variables it needs, so *n* varies
from 136 (factor analysis) to 145 (descriptive statistics) — `data/analytic_samples.csv` records
which result rests on which sample. Table 1 percentages are computed on valid responses per item,
so the GPA denominator is 142 and the weekly-use denominator 143.

## Open-ended responses

`Q1`–`Q7` are free text in Arabic, analysed in a separate mixed-methods paper and used in **no**
analysis here. They were screened for inadvertently self-identifying content before release and are
otherwise unedited. `arabic_genai_competency_processed.csv` omits them.

## Citation

See [CITATION.cff](CITATION.cff), or use GitHub's "Cite this repository" button.

## Licence

Code (`src/`, `scripts/`, `tests/`) under the [MIT Licence](LICENSE). Data and instrument
(`data/`, `instrument/`) under [CC BY 4.0](LICENSE-DATA).

## Contact

Fayez Sobhy Abdelsalam Torky — f.torky@squ.edu.om
