.PHONY: install install-dev test format lint

install:
	pip install -e . 

install-dev:
	pip install -e ".[dev]"

test:
	pytest

format:
	black .

lint:
	ruff check .
