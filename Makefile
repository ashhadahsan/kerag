# KERAG Makefile
# Common tasks for development and testing

.PHONY: help install test test-cov lint format clean docs

help: ## Show this help message
	@echo "KERAG Development Commands"
	@echo "========================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests
	python -m pytest tests/test_chains.py tests/test_knowledge_bases.py tests/test_workflows.py -v

test-all: ## Run all tests
	python -m pytest -v

test-cov: ## Run tests with coverage
	python -m pytest --cov=. --cov-report=term-missing --cov-report=html:htmlcov tests/test_chains.py tests/test_knowledge_bases.py tests/test_workflows.py -v

test-cov-all: ## Run all tests with coverage
	python -m pytest --cov=. --cov-report=term-missing --cov-report=html:htmlcov -v

lint: ## Run linting
	flake8 .
	mypy .

format: ## Format code
	black .

clean: ## Clean up generated files
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs: ## Generate documentation
	@echo "Documentation generation not implemented yet"

coverage: ## Run coverage analysis
	python scripts/run_coverage.py

setup: ## Setup development environment
	pip install -r requirements.txt
	pre-commit install || echo "pre-commit not available"

ci: ## Run CI pipeline locally
	make lint
	make test-cov
	make format

dev: ## Run in development mode
	@echo "Starting KERAG in development mode..."
	@echo "Available commands:"
	@echo "  make test      - Run tests"
	@echo "  make test-cov  - Run tests with coverage"
	@echo "  make lint      - Run linting"
	@echo "  make format    - Format code"
	@echo "  make clean     - Clean up files"

