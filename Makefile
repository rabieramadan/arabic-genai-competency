.PHONY: install reproduce test lint clean

install:            ## create a venv and install the package with dev extras
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

reproduce:          ## regenerate every table and figure from the raw data
	genai-competency --data data/arabic_genai_competency_data.csv --outdir output

test:               ## run the regression tests that lock the published values
	pytest

lint:
	ruff check src tests

clean:
	rm -rf output .pytest_cache .ruff_cache **/__pycache__
