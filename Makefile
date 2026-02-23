# Real Estate Price Prediction - Makefile
# Provides shortcuts for common development tasks

.PHONY: help install install-dev test test-cov lint format type-check clean docker-build docker-run docker-compose-up docker-compose-down run-api run-frontend run-all venv-check check-all ci-setup pre-commit-install update-deps docs

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
VENV_DIR := venv
ACTIVATE := $(VENV_DIR)/Scripts/activate
DOCKER_IMAGE := real-estate-prediction
DOCKER_TAG := latest

# Detect OS
ifeq ($(OS),Windows_NT)
    ACTIVATE := $(VENV_DIR)/Scripts/activate
    RM := rmdir /s /q
    PYTHON := python
else
    ACTIVATE := $(VENV_DIR)/bin/activate
    RM := rm -rf
    PYTHON := python3
endif

# =============================================================================
# HELP
# =============================================================================
help: ## Show this help message
	@echo "Real Estate Price Prediction - Available Commands:"
	@echo "================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# SETUP
# =============================================================================
venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV_DIR)

install: ## Install production dependencies
	$(PIP) install -e "."

install-dev: ## Install development dependencies with all extras
	$(PIP) install -e ".[dev]"
	$(MAKE) pre-commit-install

ci-setup: ## Setup for CI environment (no pre-commit)
	$(PIP) install -e ".[dev]"

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

update-deps: ## Update all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -e ".[dev]"

# =============================================================================
# LINTING & CODE QUALITY
# =============================================================================
lint: ## Run Ruff linter
	ruff check src/ api/ tests/

lint-fix: ## Run Ruff linter with auto-fix
	ruff check --fix src/ api/ tests/

format: ## Format code with Ruff
	ruff format src/ api/ tests/

check-format: ## Check code formatting without modifying files
	ruff format --check src/ api/ tests/

type-check: ## Run mypy type checker
	mypy src/ api/

security-check: ## Run security checks
	bandit -r src/ api/

# =============================================================================
# TESTING
# =============================================================================
test: ## Run all tests
	pytest

test-cov: ## Run tests with coverage report
	pytest --cov=src --cov=api --cov-report=term-missing --cov-report=html

test-fast: ## Run tests excluding slow ones
	pytest -m "not slow"

test-integration: ## Run only integration tests
	pytest -m integration

test-unit: ## Run only unit tests
	pytest -m unit

test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw

# =============================================================================
# CLEANING
# =============================================================================
clean: ## Remove build artifacts, cache files, etc.
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean ## Remove everything including virtual environment
	$(RM) $(VENV_DIR) 2>/dev/null || true

# =============================================================================
# RUNNING APPLICATIONS
# =============================================================================
run-api: ## Run the FastAPI server
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-api-prod: ## Run the FastAPI server in production mode
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

run-frontend: ## Run the React frontend development server
	cd frontend && npm run dev

run-frontend-build: ## Build the React frontend
	cd frontend && npm run build

run-all: ## Run the full ML pipeline
	$(PYTHON) run_all.py

run-notebook: ## Launch Jupyter Notebook
	jupyter notebook

# =============================================================================
# DOCKER
# =============================================================================
docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose-up: ## Start all services with Docker Compose
	docker-compose up -d

docker-compose-down: ## Stop all Docker Compose services
	docker-compose down

docker-compose-logs: ## View logs from Docker Compose services
	docker-compose logs -f

docker-compose-build: ## Build and start all services
	docker-compose up --build -d

docker-push: ## Push Docker image to registry (requires DOCKER_REGISTRY)
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

# =============================================================================
# ML & TRAINING
# =============================================================================
train: ## Train the model
	$(PYTHON) run_all.py --mode train

evaluate: ## Evaluate the model
	$(PYTHON) run_all.py --mode evaluate

predict: ## Make predictions
	$(PYTHON) run_all.py --mode predict

mlflow-ui: ## Start MLflow tracking UI
	mlflow ui --backend-store-uri ./mlruns

# =============================================================================
# DATABASE & CACHING
# =============================================================================
redis-start: ## Start Redis server (requires Redis installed)
	redis-server

redis-cli: ## Open Redis CLI
	redis-cli

# =============================================================================
# MONITORING
# =============================================================================
monitoring-up: ## Start monitoring stack (Prometheus, Grafana)
	docker-compose -f docker-compose.monitoring.yml up -d

monitoring-down: ## Stop monitoring stack
	docker-compose -f docker-compose.monitoring.yml down

# =============================================================================
# DOCUMENTATION
# =============================================================================
docs-serve: ## Serve documentation locally
	cd docs && $(PYTHON) -m http.server 8080

generate-diagram: ## Generate architecture diagram
	dot -Tpng docs/architecture.dot -o docs/architecture.png

# =============================================================================
# VERIFICATION
# =============================================================================
check-all: ## Run all checks (lint, type-check, test)
	$(MAKE) check-format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test

ci: ## Full CI pipeline simulation
	$(MAKE) clean
	$(MAKE) ci-setup
	$(MAKE) check-all
	$(MAKE) docker-build

# =============================================================================
# VERSION MANAGEMENT
# =============================================================================
version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version
	bump2version minor

version-major: ## Bump major version
	bump2version major
