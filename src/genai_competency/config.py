"""Instrument definition, coding schemes and analysis constants.

Every magic value used anywhere in the pipeline lives here, so that changing a
dimension definition or a recode is a one-line edit in a single file.
"""
from __future__ import annotations

# --------------------------------------------------------------------- seeds
RNG_SEED = 20240915
"""Seed for Horn's parallel analysis and the bootstrap CI. Fixed so that two
runs of the pipeline produce identical output."""

N_PARALLEL_ITER = 1000
N_BOOTSTRAP = 5000

# ---------------------------------------------------------------- extraction
EXTRACTION = "minres"
"""Minimum-residual (principal-axis) common-factor extraction.

This is the extraction that reproduces the published EFA: 61.1% cumulative
variance, Harman single-factor 32.7%, communalities .46/.97 for AW1/AW2. A
principal-components extraction gives 66.1% and materially different
communalities, so the two are NOT interchangeable.
"""

PCA_EXTRACTION = "principal"
"""Second estimator, reported alongside the first for omega, CR and AVE.

These three indices are sensitive to the extraction used, and for the
Prompt-Engineering dimension the two estimators straddle the conventional .50
AVE threshold (.42 common-factor vs .52 principal-components). The pipeline
reports both rather than choosing between them.
"""

# --------------------------------------------------------------- dimensions
AW = [f"AW{i}" for i in range(1, 7)]
AP = [f"AP{i}" for i in range(1, 9)]
PC = [f"PC{i}" for i in range(1, 7)]
AS = [f"AS{i}" for i in range(1, 9)]

DIMS: dict[str, list[str]] = {"AW": AW, "AP": AP, "PC": PC, "AS": AS}

LABEL = {
    "AW": "Awareness & Availability",
    "AP": "Functional Application",
    "PC": "Prompt Eng. & Critical Appraisal",
    "AS": "Aspiration & Vision",
}

SHORT = {
    "AW": "Awareness",
    "AP": "Application",
    "PC": "Prompt/Appraisal",
    "AS": "Aspiration",
}

ITEMS: list[str] = AW + AP + PC + AS

OPEN_ENDED = [f"Q{i}" for i in range(1, 8)]

# ------------------------------------------------------------------ recodes
GPA_MAP = {"Good": 1, "Very good": 2, "Excellent": 3}
FREQ_MAP = {"Rarely": 1, "1-2": 2, "3-5": 3, ">5": 4}

LEVEL_ORDER = ["Bachelor", "Master", "Doctorate"]
FREQ_ORDER = ["Rarely", "1-2", "3-5", ">5"]
GPA_ORDER = ["Excellent", "Very good", "Good"]

# ----------------------------------------------------------------- contrasts
PAIRS = [("AS", "AW"), ("AS", "AP"), ("AS", "PC"),
         ("PC", "AW"), ("PC", "AP"), ("AP", "AW")]
"""Pairwise dimension contrasts, in the order they appear in Table 4."""

REGRESSION_STEPS = [
    ["gpa_n", "postgrad"],                            # Step 1: achievement
    ["gpa_n", "postgrad", "train_n"],                 # Step 2: + prior training
    ["gpa_n", "postgrad", "train_n", "freq_n"],       # Step 3: + frequency of use
]

PREDICTOR_LABELS = {
    "gpa_n": "GPA (ordinal)",
    "postgrad": "Postgraduate (vs. BA)",
    "train_n": "Prior AI training",
    "freq_n": "Frequency of use",
}

# ----------------------------------------------------------------- thresholds
AVE_THRESHOLD = 0.50
HTMT_THRESHOLD = 0.85
CEILING_CUTOFF = 4.5
