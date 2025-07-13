#!/bin/bash

# Simple test runner script for development

set -e

echo "🐳 Starting test environment..."

# Function to clean up on exit
cleanup() {
    echo "🧹 Cleaning up test environment..."
    docker compose -f docker-compose.test.yml down --remove-orphans
}

# Set trap to ensure cleanup happens
trap cleanup EXIT

# Start test services
docker compose -f docker-compose.test.yml up --build -d db-test

# Wait for database to be ready
echo "⏳ Waiting for test database to be ready..."
docker compose -f docker-compose.test.yml exec -T db-test pg_isready -U postgres || sleep 5

# Run tests based on argument
case "${1:-unit}" in
    "unit")
        echo "🧪 Running unit tests..."
        docker compose -f docker-compose.test.yml run --rm backend-test python -m pytest tests/unit/ -v
        ;;
    "integration")
        echo "🔗 Running integration tests..."
        docker compose -f docker-compose.test.yml run --rm backend-test python -m pytest tests/integration/ -v
        ;;
    "e2e")
        echo "🎯 Running E2E tests..."
        docker compose -f docker-compose.test.yml run --rm backend-test python -m pytest tests/e2e/ -v
        ;;
    "all")
        echo "🚀 Running all tests..."
        docker compose -f docker-compose.test.yml run --rm backend-test python -m pytest tests/ -v
        ;;
    "watch")
        echo "👀 Starting test watch mode..."
        docker compose -f docker-compose.test.yml up backend-test-watch
        ;;
    *)
        echo "Usage: $0 {unit|integration|e2e|all|watch}"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests only"
        echo "  e2e         - Run E2E tests only"
        echo "  all         - Run all tests"
        echo "  watch       - Run tests in watch mode"
        exit 1
        ;;
esac

echo "✅ Tests completed!"
