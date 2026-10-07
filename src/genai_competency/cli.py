"""Command-line entry point: ``genai-competency``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="genai-competency",
        description="Reproduce every table, figure and statistic in the Arabic "
                    "GenAI competency study from the raw response file.",
    )
    p.add_argument("--data", type=Path,
                   default=Path("data/arabic_genai_competency_data.csv"),
                   help="path to the raw response CSV (default: %(default)s)")
    p.add_argument("--outdir", type=Path, default=Path("output"),
                   help="directory for tables, figures and logs (default: %(default)s)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data.exists():
        print(f"error: data file not found: {args.data}", file=sys.stderr)
        return 2
    run(args.data, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
