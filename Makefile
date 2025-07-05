# Wagehood Trading System Makefile

.PHONY: install install-dev test lint format clean run-api run-tests benchmark docs

# Installation targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e .

# Code quality targets
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-performance:
	pytest tests/ -v -m performance

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

check: format lint test

# Application targets
run-api:
	python -m src.api.app

run-tests:
	python run_tests.py --all --coverage

benchmark:
	python scripts/benchmark_performance.py

# Analysis targets
backtest:
	python scripts/run_analysis.py

generate-data:
	python scripts/generate_mock_data.py

# Documentation targets
docs:
	@echo "API Documentation: http://localhost:8000/docs (start API first)"
	@echo "Research Documentation: .local/"
	@echo "Technical Documentation: README.md"

# Maintenance targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/
	rm -rf dist/ build/

clean-data:
	rm -rf data/cache/
	rm -rf results/

# Development targets
setup-dev: install-dev
	pre-commit install

update-deps:
	pip list --outdated
	pip freeze > requirements.txt

# Quick development workflow
dev: format lint test-unit
	@echo "Development checks complete"

# Full CI pipeline
ci: format lint test
	@echo "All CI checks passed"

# Help target
help:
	@echo "Available targets:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test             Run all tests with coverage"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  lint             Run code linting"
	@echo "  format           Format code with black and isort"
	@echo "  check            Run format, lint, and test"
	@echo "  run-api          Start the API server"
	@echo "  run-tests        Run comprehensive test suite"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  backtest         Run example backtest analysis"
	@echo "  generate-data    Generate sample mock data"
	@echo "  docs             Show documentation links"
	@echo "  clean            Clean generated files"
	@echo "  dev              Quick development checks"
	@echo "  ci               Full CI pipeline"