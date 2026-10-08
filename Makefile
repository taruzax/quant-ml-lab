.PHONY: up down demo notebook dagster test ingest snapshot-list prepare experiment campaign-plan campaign-run

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

STREAM ?= equities-daily
CAMPAIGN ?= config/campaign.example.yaml
CONFIG ?= config/pipeline.yaml

ingest:
	uv run quant-ml ingest $(STREAM)

snapshot-list:
	uv run quant-ml --json snapshot-list

prepare:
	uv run quant-ml --json prepare --config $(CONFIG)

experiment:
	uv run quant-ml --json experiment --config $(CONFIG)

campaign-plan:
	uv run quant-ml --json campaign-plan --campaign $(CAMPAIGN)

campaign-run:
	uv run quant-ml --json campaign-run --campaign $(CAMPAIGN)
