.PHONY: test test-unit test-integration test-e2e test-watch test-coverage build-test clean-test

# Test commands
test-unit:
	@echo "Running unit tests..."
	./test.sh unit

test-integration:
	@echo "Running integration tests..."
	./test.sh integration

test-e2e:
	@echo "Running E2E tests..."
	./test.sh e2e

test:
	@echo "Running all tests..."
	./test.sh all

test-watch:
	@echo "Starting test watch mode..."
	./test.sh watch

test-coverage:
	@echo "Running tests with coverage..."
	docker compose -f docker-compose.test.yml run --rm backend-test python -m pytest tests/ --cov=app --cov-report=html --cov-report=term

# Docker commands
build-test:
	@echo "Building test environment..."
	docker compose -f docker-compose.test.yml build

clean-test:
	@echo "Cleaning test environment..."
	docker compose -f docker-compose.test.yml down --volumes --remove-orphans
	docker system prune -f

# Development commands
dev:
	@echo "Starting development environment..."
	docker compose up --build

dev-down:
	@echo "Stopping development environment..."
	docker compose down

# Linting and formatting
lint:
	@echo "Running linting..."
	docker compose -f docker-compose.test.yml run --rm backend-test python -m flake8 app tests

format:
	@echo "Formatting code..."
	docker compose -f docker-compose.test.yml run --rm backend-test python -m black app tests
	docker compose -f docker-compose.test.yml run --rm backend-test python -m isort app tests

# Help
help:
	@echo "Available commands:"
	@echo "  test-unit        - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-e2e         - Run E2E tests"
	@echo "  test             - Run all tests"
	@echo "  test-watch       - Run tests in watch mode"
	@echo "  test-coverage    - Run tests with coverage report"
	@echo "  build-test       - Build test environment"
	@echo "  clean-test       - Clean test environment"
	@echo "  dev              - Start development environment"
	@echo "  dev-down         - Stop development environment"
	@echo "  lint             - Run linting"
	@echo "  format           - Format code"
