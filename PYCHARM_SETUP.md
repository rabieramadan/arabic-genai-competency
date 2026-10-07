# Opening this project in PyCharm

The repository ships a working `.idea/` configuration: two run configurations, a
NumPy-docstring setting, source roots and a Git mapping. You only need to point
it at an interpreter.

## 1. Open the project

**File → Open**, select the repository folder (the one containing `pyproject.toml`),
and choose **Open as Project** — *not* "Attach". PyCharm will detect the existing
`.idea/` folder and keep it.

## 2. Create the interpreter

**Settings → Project → Python Interpreter → Add Interpreter → Add Local
Interpreter → Virtualenv Environment → New**

- Location: `<project>/.venv`
- Base interpreter: Python 3.10, 3.11 or 3.12
- Name it `Python 3.12 (arabic-genai-competency)` so it matches the SDK name the
  shipped run configurations expect. If you name it differently, reselect the
  interpreter in each run configuration once.

## 3. Install the package in editable mode

In the PyCharm **Terminal** tab:

```bash
pip install -e ".[dev]"
```

Editable mode matters: it puts `src/` on the path, so `import genai_competency`
resolves without any `sys.path` manipulation, and your edits take effect
immediately.

> `scikit-learn` is pinned below 1.6 in `pyproject.toml`. If PyCharm offers to
> upgrade it, decline — `factor_analyzer` calls an API that was renamed in 1.6
> and the factor analysis will raise `TypeError`.

## 4. Mark the source roots (usually automatic)

If `import genai_competency` shows as unresolved, right-click `src` →
**Mark Directory as → Sources Root**, and `tests` → **Test Sources Root**.

## 5. Run it

Two configurations appear in the run-configuration dropdown, top right:

| Configuration | What it does |
|---|---|
| **Reproduce all results** | Runs the full pipeline, writing tables, figures and logs to `output/` |
| **Regression tests** | Runs the pytest suite that locks the published values |

Both use the project root as their working directory, so the default relative
data path resolves correctly.

## 6. Where to look first

- `src/genai_competency/config.py` — every dimension, recode and threshold
- `src/genai_competency/pipeline.py` — the analysis, in the order of the article's Results section
- `tests/test_reproduction.py` — the published values, asserted
- `docs/verification_table.csv` — reported vs recomputed, with verdicts

## Debugging a specific statistic

Set a breakpoint in `pipeline.py` at the section you care about — the sections
are commented with the table or research question they produce — and start
**Reproduce all results** with the debug button. The `R` dict accumulates every
headline value as the run proceeds, so you can inspect it mid-run in the
Variables pane.
