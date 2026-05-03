.PHONY: install install-dev test format lint serve docker-build docker-up

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

serve:
	uvicorn api.main:app --reload --port 8000

docker-build:
	docker build -t velum .

docker-up:
	docker-compose up
