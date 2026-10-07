#!/usr/bin/env python3
"""Thin wrapper kept for the filename cited in the article's supplementary material.

Equivalent to `genai-competency --data ... --outdir ...`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai_competency.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
