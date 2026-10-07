"""Reproduction package for the Arabic GenAI competency study.

Regenerates every table, figure and statistic reported in:

    Torky, F. S. A., Al Kindi, K. S. M., & Hassanein, M. M.
    "Aspiration without application: Developing and initially validating a
    four-dimensional measure of generative-AI competency and the readiness gap
    among Arabic language and literature students."

Title note: the above is the title of the REVISED manuscript, under review. The
original submission read "...: Validating a four-dimensional measure of ...";
the wording was changed in revision because parallel analysis retained three
factors, not four. Provisional until acceptance.

Typical use::

    from genai_competency import run
    results = run("data/arabic_genai_competency_data.csv", "output")
    print(results["gap_mean"], results["gap_dz"])

or from a shell::

    genai-competency --data data/arabic_genai_competency_data.csv --outdir output
"""
from .pipeline import run

__version__ = "1.0.0"
__all__ = ["run", "__version__"]
