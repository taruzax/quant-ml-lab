.PHONY: up down demo notebook dagster test ingest

up:
	docker compose up -d

down:
	docker compose down

demo:
	uv run quant-ml-demo

notebook:
	uv run jupyter lab notebooks/research_walkthrough.ipynb

dagster:
	uv run dagster dev -m lab.definitions

test:
	uv run pytest

ingest:
	uv run run-pipeline
