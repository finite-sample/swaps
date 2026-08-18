PYTHON ?= python3
SOURCE_DATE_EPOCH ?= 0
FORCE_SOURCE_DATE ?= 1
LATEX_ENV = SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=$(FORCE_SOURCE_DATE)

.PHONY: analysis analysis-fast paper test lint format check ci ci-docker

analysis:
	MPLCONFIGDIR=/tmp/validation-swaps-mpl $(PYTHON) replicate_final.py

analysis-fast:
	MPLCONFIGDIR=/tmp/validation-swaps-mpl $(PYTHON) replicate_final.py --fast --outdir /tmp/validation-swaps-fast

paper: analysis
	$(LATEX_ENV) pdflatex -interaction=nonstopmode -halt-on-error validation_swaps.tex
	$(LATEX_ENV) pdflatex -interaction=nonstopmode -halt-on-error validation_swaps.tex

test:
	MPLCONFIGDIR=/tmp/validation-swaps-mpl $(PYTHON) -m pytest

lint:
	$(PYTHON) -m black --check replicate_final.py tests
	$(PYTHON) -m isort --check-only replicate_final.py tests
	$(PYTHON) -m flake8 replicate_final.py tests
	$(PYTHON) -m ruff check replicate_final.py tests

format:
	$(PYTHON) -m isort replicate_final.py tests
	$(PYTHON) -m black replicate_final.py tests
	$(PYTHON) -m ruff check --fix replicate_final.py tests

check: lint test

ci: check analysis-fast

ci-docker:
	docker run --rm -v "$(CURDIR):/work" -w /work python:3.13 sh -c "python -m pip install --disable-pip-version-check -e '.[dev]' && make ci"
