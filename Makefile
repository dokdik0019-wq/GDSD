# GDSD edge detector -- local quadratic surface fit + zero-crossing
PYTHON ?= python3

.PHONY: setup demo test clean

setup:            ## Install the package in editable mode with dev tools
	$(PYTHON) -m pip install -e ".[dev]"

demo:             ## Run the demo on the generated sample image
	$(PYTHON) demo.py

test:             ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

clean:            ## Remove generated files
	rm -rf output/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
